# yicenet-hook — native hook client

One small C++ binary for every agent CLI that runs shell-command hooks
(Claude Code, Kimi Code, ...):

```
yicenet-hook <platform> <event>   < payload.json   > hook stdout
```

It is deliberately dumb. It reads the raw hook payload from stdin and POSTs it
to the local YiCeNet daemon at `/hook/<event>?platform=<platform>`. It then
copies the reply body to stdout byte for byte. Event names, payload fields and
output format all live in the daemon, in `src/yicenet/daemon/platforms.py`.
Supporting a new CLI means adding a handler there; the binary stays the same.

| | |
|---|---|
| Port | `$YICENET_DAEMON_PORT`, else `<tempdir>/yicenet-daemon.port`, else 7788 |
| Daemon spawn | if nothing listens: `<python> -m yicenet.daemon.hook_server`, detached. `<python>` is `$YICENET_DAEMON_PYTHON`, else the first line of `~/.yicenet/daemon-python` (written by the installers) |
| Failure | always exit 0, nothing on stdout, reason on stderr; a hook never blocks the agent |

Platforms: `claude-code` (events `pre`, `post_tool`, `stop`) and `kimi-code`
(the event names of `kimi_code_hook._COMMANDS`). Hermes runs YiCeNet in-process
and needs no client.

## Latency (Windows 11, warm daemon)

| event | end to end |
|---|---|
| `pre` (model prediction) | ~27 ms |
| `post_tool` / `stop` | ~7 ms |

The Python runner it replaces took ~250 ms per call, most of it interpreter
startup. On a cold start the client spawns the daemon, and the first `pre`
takes ~2–3 s while the model loads.

## Build / install

Prebuilt binaries for Windows x64, Linux x64/arm64 and macOS arm64 are attached
to every GitHub release. `yicenet-bootstrap`, `deploy-hermes.ps1` and
`install-yicenet.sh` put the binary in `~/.yicenet/bin`. They register it in
Claude Code's `settings.json` whenever it is present, and fall back to the
Python runner otherwise.

From source (needs CMake + a C++17 compiler):

```
scripts/build-hook.ps1     # Windows: finds Visual Studio's bundled CMake
scripts/build-hook.sh      # Linux / macOS: CMake, or c++ directly
```

Kimi Code: `kimi-plugin/kimi.plugin.json` calls the Python dispatcher, which
speaks the same protocol. For native speed, change a hook's command to
`~/.yicenet/bin/yicenet-hook` with args `["kimi-code", "<event>"]`.
