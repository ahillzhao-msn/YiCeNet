// yicenet-hook — native hook client for every agent CLI (Claude Code, Kimi Code, ...).
//
//   yicenet-hook <platform> <event>   < payload.json   > hook stdout
//
// Deliberately dumb: reads the raw hook payload from stdin, POSTs it to the local
// YiCeNet daemon at /hook/<event>?platform=<platform>, and copies the reply body to
// stdout byte for byte.  All platform semantics live in yicenet.daemon.platforms.
// If the daemon is not running it is spawned (detached) and the request retried.
// Never fails the agent: every error path exits 0 with nothing on stdout.
//
// Port:   $YICENET_DAEMON_PORT, else <tempdir>/yicenet-daemon.port, else 7788.
// Python: $YICENET_DAEMON_PYTHON, else first line of ~/.yicenet/daemon-python
//         (written by the yicenet installers).

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <thread>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <mstcpip.h>
#  include <windows.h>
#  include <fcntl.h>
#  include <io.h>
using socket_t = SOCKET;
static const socket_t kBadSocket = INVALID_SOCKET;
static void close_socket(socket_t s) { closesocket(s); }
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <sys/time.h>
#  include <sys/wait.h>
#  include <unistd.h>
using socket_t = int;
static const socket_t kBadSocket = -1;
static void close_socket(socket_t s) { close(s); }
#endif

namespace {

const int kDefaultPort = 7788;
const int kReplyTimeoutMs = 30000;  // first request after a spawn loads the model
const int kSpawnWaitMs = 10000;
const int kSpawnPollMs = 50;
const char* kDaemonModule = "yicenet.daemon.hook_server";

std::string getenv_str(const char* name) {
#ifdef _WIN32
    wchar_t wname[64];
    MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 64);
    DWORD n = GetEnvironmentVariableW(wname, nullptr, 0);
    if (n == 0) return {};
    std::wstring w(n, L'\0');
    n = GetEnvironmentVariableW(wname, &w[0], n);
    w.resize(n);
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], len, nullptr, nullptr);
    return s;
#else
    const char* v = std::getenv(name);
    return v ? v : "";
#endif
}

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], len);
    return w;
}
const char kSep = '\\';
#else
const char kSep = '/';
#endif

FILE* open_file(const std::string& path, const char* mode) {
#ifdef _WIN32
    return _wfopen(widen(path).c_str(), widen(mode).c_str());
#else
    return std::fopen(path.c_str(), mode);
#endif
}

std::string trim(std::string s) {
    const char* ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    size_t end = s.find_last_not_of(ws);
    s.erase(end == std::string::npos ? 0 : end + 1);
    return s;
}

std::string read_first_line(const std::string& path) {
    FILE* f = open_file(path, "rb");
    if (!f) return {};
    char buf[4096];
    size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = '\0';
    std::string s(buf);
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF)
        s.erase(0, 3);  // UTF-8 BOM (PowerShell)
    size_t nl = s.find_first_of("\r\n");
    return trim(nl == std::string::npos ? s : s.substr(0, nl));
}

// Same directory as Python's tempfile.gettempdir(), so the daemon's port file is found.
std::string temp_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH + 1];
    DWORD n = GetTempPathW(MAX_PATH + 1, buf);
    std::wstring w(buf, n);
    while (!w.empty() && (w.back() == L'\\' || w.back() == L'/')) w.pop_back();
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], len, nullptr, nullptr);
    return s;
#else
    for (const char* name : {"TMPDIR", "TEMP", "TMP"}) {
        std::string v = getenv_str(name);
        if (!v.empty()) return v;
    }
    return "/tmp";
#endif
}

std::string home_dir() {
#ifdef _WIN32
    return getenv_str("USERPROFILE");
#else
    return getenv_str("HOME");
#endif
}

int resolve_port() {
    int port = std::atoi(getenv_str("YICENET_DAEMON_PORT").c_str());
    if (port > 0) return port;
    port = std::atoi(read_first_line(temp_dir() + kSep + "yicenet-daemon.port").c_str());
    return port > 0 ? port : kDefaultPort;
}

// ── HTTP over loopback ──────────────────────────────────────────────────────

socket_t connect_loopback(int port) {
    socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == kBadSocket) return kBadSocket;
#ifdef _WIN32
    // Windows retries a refused SYN for ~2 s even on loopback; a missing daemon must fail fast.
    TCP_INITIAL_RTO_PARAMETERS rto{};
    rto.Rtt = TCP_INITIAL_RTO_UNSPECIFIED_RTT;
    rto.MaxSynRetransmissions = TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS;
    DWORD ignored = 0;
    WSAIoctl(s, SIO_TCP_INITIAL_RTO, &rto, sizeof(rto), nullptr, 0, &ignored, nullptr, nullptr);
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(s, (sockaddr*)&addr, sizeof(addr)) != 0) {
        close_socket(s);
        return kBadSocket;
    }
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
#ifdef _WIN32
    DWORD tv = kReplyTimeoutMs;
#else
    timeval tv{kReplyTimeoutMs / 1000, (kReplyTimeoutMs % 1000) * 1000};
#endif
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
    return s;
}

bool send_all(socket_t s, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        int n = send(s, data.data() + off, (int)(data.size() - off), 0);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

std::string url_encode(const std::string& v) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : v) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += (char)c;
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

enum class Post { Ok, Unreachable, Failed };

// POST body on an already-connected socket; on success `reply` holds the response body.
Post post_on(socket_t s, const std::string& platform, const std::string& event,
             const std::string& body, std::string& reply) {
    std::string req = "POST /hook/" + url_encode(event) + "?platform=" + url_encode(platform) +
                      " HTTP/1.0\r\nHost: 127.0.0.1\r\n"
                      "Content-Type: application/json; charset=utf-8\r\n"
                      "Content-Length: " + std::to_string(body.size()) +
                      "\r\nConnection: close\r\n\r\n";
    if (!send_all(s, req) || !send_all(s, body)) return Post::Failed;

    std::string resp;
    char buf[16384];
    for (;;) {
        int n = recv(s, buf, sizeof(buf), 0);
        if (n < 0) return Post::Failed;  // timeout or reset
        if (n == 0) break;
        resp.append(buf, (size_t)n);
    }
    size_t sp = resp.find(' ');
    size_t hdr_end = resp.find("\r\n\r\n");
    if (sp == std::string::npos || hdr_end == std::string::npos) return Post::Failed;
    if (resp.compare(sp + 1, 3, "200") != 0) {
        std::fprintf(stderr, "[YiCeNet] daemon: %s\n", resp.substr(hdr_end + 4, 300).c_str());
        return Post::Failed;
    }
    reply = resp.substr(hdr_end + 4);
    return Post::Ok;
}

Post post(int port, const std::string& platform, const std::string& event,
          const std::string& body, std::string& reply) {
    socket_t s = connect_loopback(port);
    if (s == kBadSocket) return Post::Unreachable;
    Post r = post_on(s, platform, event, body, reply);
    close_socket(s);
    return r;
}

// ── Daemon spawn ────────────────────────────────────────────────────────────

std::string daemon_python() {
    std::string py = getenv_str("YICENET_DAEMON_PYTHON");
    if (!py.empty()) return py;
    return read_first_line(home_dir() + kSep + ".yicenet" + kSep + "daemon-python");
}

std::string log_path() {
    std::string dir = home_dir() + kSep + ".yicenet" + kSep + "logs";
#ifdef _WIN32
    CreateDirectoryW(widen(home_dir() + "\\.yicenet").c_str(), nullptr);
    CreateDirectoryW(widen(dir).c_str(), nullptr);
#else
    mkdir((home_dir() + "/.yicenet").c_str(), 0755);
    mkdir(dir.c_str(), 0755);
#endif
    return dir + kSep + "daemon.log";
}

#ifdef _WIN32
// The agent's stdout pipe must not leak into the long-lived daemon, or the agent never
// sees EOF on the hook's output: inherit exactly the NUL/log handles and nothing else.
bool spawn_daemon(const std::string& python) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                             OPEN_EXISTING, 0, nullptr);
    HANDLE log = CreateFileW(widen(log_path()).c_str(), FILE_APPEND_DATA,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_ALWAYS, 0, nullptr);
    if (log == INVALID_HANDLE_VALUE) log = nul;

    HANDLE inherit[2] = {nul, log};
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    auto* attrs = (LPPROC_THREAD_ATTRIBUTE_LIST)HeapAlloc(GetProcessHeap(), 0, size);
    InitializeProcThreadAttributeList(attrs, 1, 0, &size);
    UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                              (log == nul ? 1 : 2) * sizeof(HANDLE), nullptr, nullptr);

    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof(si);
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = nul;
    si.StartupInfo.hStdOutput = log;
    si.StartupInfo.hStdError = log;
    si.lpAttributeList = attrs;

    // CREATE_NO_WINDOW, not DETACHED_PROCESS: Windows ignores CREATE_NO_WINDOW when both are
    // set. uv's venv\Scripts\python(w).exe is a console-subsystem trampoline that forwards to
    // the base python.exe; with no console of its own to inherit, that child gets a new,
    // visible console window. A windowless console is inherited instead.
    std::wstring cmd = L"\"" + widen(python) + L"\" -m " + widen(kDaemonModule);
    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, TRUE,
                             EXTENDED_STARTUPINFO_PRESENT | CREATE_NEW_PROCESS_GROUP |
                                 CREATE_NO_WINDOW,
                             nullptr, nullptr, &si.StartupInfo, &pi);
    DeleteProcThreadAttributeList(attrs);
    HeapFree(GetProcessHeap(), 0, attrs);
    if (log != nul) CloseHandle(log);
    CloseHandle(nul);
    if (!ok) return false;
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}
#else
bool spawn_daemon(const std::string& python) {
    std::string log = log_path();
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        setsid();
        if (fork() != 0) _exit(0);  // double fork: the daemon is reparented to init
        int in = open("/dev/null", O_RDONLY);
        int out = open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (out < 0) out = open("/dev/null", O_WRONLY);
        dup2(in, 0);
        dup2(out, 1);
        dup2(out, 2);
        long max_fd = sysconf(_SC_OPEN_MAX);
        for (long fd = 3; fd < (max_fd > 0 ? max_fd : 1024); ++fd) close((int)fd);
        execl(python.c_str(), python.c_str(), "-m", kDaemonModule, (char*)nullptr);
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}
#endif

// Spawn the daemon, then wait for it to accept connections (port file may name a new port).
socket_t spawn_and_connect() {
    std::string python = daemon_python();
    if (python.empty()) {
        std::fprintf(stderr, "[YiCeNet] daemon not running and ~/.yicenet/daemon-python not set; "
                             "run the yicenet installer\n");
        return kBadSocket;
    }
    if (!spawn_daemon(python)) {
        std::fprintf(stderr, "[YiCeNet] failed to start daemon with %s\n", python.c_str());
        return kBadSocket;
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kSpawnWaitMs);
    while (std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kSpawnPollMs));
        socket_t s = connect_loopback(resolve_port());
        if (s != kBadSocket) return s;
    }
    std::fprintf(stderr, "[YiCeNet] daemon did not come up within %d ms\n", kSpawnWaitMs);
    return kBadSocket;
}

std::string read_stdin() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    std::string data;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) data.append(buf, n);
    return data;
}

void write_stdout(const std::string& data) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::fwrite(data.data(), 1, data.size(), stdout);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: yicenet-hook <platform> <event>   (payload on stdin)\n");
        return 0;
    }
    const std::string platform = argv[1];
    const std::string event = argv[2];
    const std::string body = read_stdin();

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 0;
#endif

    std::string reply;
    Post r = post(resolve_port(), platform, event, body, reply);
    if (r == Post::Unreachable) {
        socket_t s = spawn_and_connect();
        if (s != kBadSocket) {
            r = post_on(s, platform, event, body, reply);
            close_socket(s);
        }
    }
    if (r == Post::Ok) write_stdout(reply);

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
