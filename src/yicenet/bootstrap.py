#!/usr/bin/env python3
"""YiCeNet Bootstrap — 一鍵安裝/卸載初始化。

安裝到現有 venv（共享），必要時創建獨立 venv。
按 --target 自動配置 Hermes 工具鏈接 / Claude Code MCP server，
註冊 flywheel cron。首次運行自動初始化 ~/.yicenet/ 目錄結構、
config.yaml 模板和 SOUL.md 身份文件。

用法：
  yicenet-bootstrap                            # auto: 檢測並配置所有已裝的 IDE
  yicenet-bootstrap --target hermes            # 只配置 Hermes
  yicenet-bootstrap --target claude-code       # 只配置 Claude Code
  yicenet-bootstrap --auto                     # 全自動（無交互確認）
  yicenet-bootstrap --soul ~/my-soul.md        # 自定義 SOUL 模板
  yicenet-bootstrap --venv /path/to/venv       # 指定 venv
  python3 scripts/bootstrap.py                 # 源碼樹直接運行

卸載：
  yicenet-uninstall                            # 移除目標環境註冊
  yicenet-uninstall --clean-data               # + 刪除 ~/.yicenet/ 所有數據
  pip uninstall yicenet                        # 移除包
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ── 路徑 ───────────────────────────────────────

def _project_root() -> Path:
    """自動檢測 YiCeNet 項目根目錄（僅從 __file__ 推導，不依賴已導入的包）。"""
    p = Path(__file__).resolve().parent.parent
    if (p / "pyproject.toml").exists():
        return p
    # 也可能是 scripts/ 的上兩層（__file__ 在包內時）
    if (p.parent / "pyproject.toml").exists():
        return p.parent
    return Path.cwd()


PROJECT = _project_root()


# ══════════════════════════════════════════════════
# 環境檢測
# ══════════════════════════════════════════════════


def detect_hermes() -> tuple[bool, str]:
    """檢測 Hermes 環境。"""
    hermes_home = os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))

    try:
        r = subprocess.run(["hermes", "--version"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            # 找 Hermes venv
            for v in [
                Path(hermes_home) / ".venv" / "bin" / "python3",
                Path(hermes_home) / ".venv" / "bin" / "python",
            ]:
                if v.exists():
                    return True, str(v)
            return True, ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 無 CLI 時按路徑猜
    for v in [
        Path(hermes_home) / ".venv" / "bin" / "python3",
        Path(hermes_home) / ".venv" / "bin" / "python",
    ]:
        if v.exists():
            return False, str(v)

    return False, ""


def detect_torch() -> tuple[bool, str]:
    """檢測 PyTorch（YiCeNet 核心依賴）。"""
    try:
        import torch
        cuda = torch.cuda.is_available()
        ver = torch.__version__
        gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
        return True, f"{ver} ({'GPU: ' + gpu if cuda else 'CPU'})"
    except ImportError:
        return False, "not installed"


# ══════════════════════════════════════════════════
# 安裝
# ══════════════════════════════════════════════════


def install_to_venv(venv_python: str, extras: str = "") -> bool:
    """pip install -e . 到指定 venv。extras 如 'mcp' 則安裝 .[mcp]。"""
    pkg = f"{PROJECT}[{extras}]" if extras else str(PROJECT)
    try:
        r = subprocess.run(
            [venv_python, "-m", "pip", "install", "-e", pkg],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            print(f"  ✓ YiCeNet installed into {venv_python}")
            verify = subprocess.run(
                [venv_python, "-c", "import yicenet; print(yicenet.__version__)"],
                capture_output=True, text=True, timeout=10,
            )
            if verify.returncode == 0:
                print(f"  ✓ Version: {verify.stdout.strip()}")
            return True
        else:
            print(f"  ⚠ pip install failed: {r.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ⚠ pip install error: {e}")
        return False


def ensure_deps(venv_python: str) -> bool:
    """確保 PyTorch 等核心依賴已安裝。"""
    try:
        r = subprocess.run(
            [venv_python, "-c", "import torch; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return True
    except Exception:
        pass

    print("  → Installing PyTorch (CPU)...")
    try:
        subprocess.run(
            [venv_python, "-m", "pip", "install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            capture_output=True, text=True, timeout=300,
        )
        return True
    except Exception:
        print("  ⚠ PyTorch install failed (soft — continue)")
        return False


# ══════════════════════════════════════════════════
# 檢查點
# ══════════════════════════════════════════════════


def ensure_checkpoints() -> bool:
    """確保至少有一個可用的檢查點。

    策略：
      1. 已有 registry.json + 存在的 .pt → OK
      2. 從同機器其它 YiCeNet 克隆複製
      3. 從 HuggingFace 下載發布版（未來）
      4. 生成隨機初始化模型（緊急 fallback）
    """
    checkpoints_dir = PROJECT / "checkpoints"
    registry_path = checkpoints_dir / "registry.json"

    # 1. 已有可用檢查點
    if registry_path.exists():
        try:
            reg = json.loads(registry_path.read_text())
            active = reg.get("active", {})
            active_path = active.get("path", "")
            if active_path and (checkpoints_dir / active_path).exists():
                print(f"  ✓ Checkpoint active: {active.get('version', '?')} ({active_path})")
                return True
        except Exception:
            pass

    # 2. 掃描 checkpoints 目錄找 .pt
    pt_files = list(checkpoints_dir.glob("*.pt")) if checkpoints_dir.exists() else []
    if pt_files:
        # 有文件但 registry 不正確——重建
        try:
            r = subprocess.run(
                [sys.executable, str(PROJECT / "scripts" / "checkpoint_manager.py"),
                 "fresh"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                print(f"  ✓ Registry rebuilt from {len(pt_files)} checkpoint(s)")
                return True
        except Exception:
            pass

    # 3. 從同機源碼複製（如果存在 ~/YiCeNet 且有檢查點）
    source_dir = Path.home() / "YiCeNet"
    if source_dir != PROJECT:
        src_pt = list((source_dir / "checkpoints").glob("*.pt")) if (source_dir / "checkpoints").exists() else []
        if src_pt:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            for f in src_pt:
                shutil.copy2(f, checkpoints_dir / f.name)
            print(f"  ✓ Copied {len(src_pt)} checkpoint(s) from {source_dir}")
            # 刷新 registry
            try:
                subprocess.run(
                    [sys.executable, str(PROJECT / "scripts" / "checkpoint_manager.py"),
                     "fresh"],
                    capture_output=True, text=True, timeout=30,
                )
            except Exception:
                pass
            return True

    # 4. 生成最小模型（緊急 fallback）
    print("  · No pre-trained checkpoints found. Generating minimal model...")
    try:
        code = """
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path('__ROOT__')))
from yicenet.model import YiCeNet
from yicenet.config import YiCeNetConfig
model = YiCeNet(YiCeNetConfig())
ckpt = Path('__ROOT__') / 'checkpoints' / 'init_minimal.pt'
import torch
torch.save(model.state_dict(), ckpt)
print(f'Generated: {ckpt} ({ckpt.stat().st_size / 1e6:.1f}MB)')
""".replace("__ROOT__", str(PROJECT))
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            # 重建 registry
            subprocess.run(
                [sys.executable, str(PROJECT / "scripts" / "checkpoint_manager.py"),
                 "fresh"],
                capture_output=True, text=True, timeout=30,
            )
            print(f"  ✓ Minimal model generated (no training, random init)")
            return True
        else:
            print(f"  ⚠ Model generation failed: {r.stderr[:100]}")
            return False
    except Exception as e:
        print(f"  ⚠ Model generation error: {e}")
        return False


# ══════════════════════════════════════════════════
# Hermes 集成
# ══════════════════════════════════════════════════


def setup_hermes_tool(hermes_available: bool):
    """設置 Hermes 工具鏈接（軟性——失敗不中斷）。"""
    if not hermes_available:
        return

    tool_src = PROJECT / "src" / "yicenet" / "hermes_tool.py"
    if not tool_src.exists():
        print("  · hermes_tool.py not found, skip Hermes integration")
        return

    hermes_home = os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))
    tool_dst = Path(hermes_home) / "hermes-agent" / "tools" / "yicenet_tool.py"
    tool_dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if tool_dst.exists() or tool_dst.is_symlink():
            tool_dst.unlink()
        tool_dst.symlink_to(str(tool_src))
        print(f"  ✓ Hermes tool linked: {tool_dst}")
    except Exception as e:
        print(f"  ⚠ Hermes tool link failed: {e}")


def _get_flywheel_command() -> str:
    """生成调用 flywheel_run() 的 Python 命令（平台无关）。"""
    # 使用 sys.executable 确保用当前 Python，不走 PATH 猜测
    return f'"{sys.executable}" -m yicenet.flywheel'


def _cron_schedule_expr(hours: int) -> str:
    """生成 crontab 表达式。"""
    return f"0 */{max(1, hours)} * * *"


def _register_crontab(schedule: str, command: str, name: str) -> bool:
    """在 Linux/macOS 上注册 crontab 条目。"""
    import subprocess as sp
    comment = f"# YiCeNet Flywheel — {name}"
    entry = f"{schedule} {command} >> $HOME/.yicenet/logs/flywheel.log 2>&1"

    try:
        # 读取现有 crontab
        r = sp.run(["crontab", "-l"], capture_output=True, text=True, timeout=10)
        existing = r.stdout if r.returncode == 0 else ""
        if name in existing:
            return False  # 已注册

        # 追加新条目
        new_cron = existing.rstrip() + "\n" + comment + "\n" + entry + "\n"
        p = sp.run(["crontab", "-"], input=new_cron, capture_output=True,
                    text=True, timeout=10)
        return p.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _remove_crontab(name: str) -> bool:
    """从 crontab 移除 YiCeNet 条目。"""
    try:
        r = sp.run(["crontab", "-l"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return True  # 没有 crontab = 已清理
        lines = [line for line in r.stdout.split("\n")
                 if name not in line and "# YiCeNet" not in line]
        new_cron = "\n".join(lines).strip() + "\n"
        p = sp.run(["crontab", "-"], input=new_cron, capture_output=True,
                    text=True, timeout=10)
        return p.returncode == 0
    except FileNotFoundError:
        return True
    except Exception:
        return False


def _register_schtasks(schedule_hours: int, command: str, name: str) -> bool:
    """在 Windows 上注册 Task Scheduler 任务。"""
    task_name = f"YiCeNet\\{name}"
    try:
        # 检查是否已存在
        r = subprocess.run(
            ["schtasks.exe", "/Query", "/TN", task_name, "/FO", "LIST"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return False  # 已注册

        # 创建任务（每 6 小时重复，无限期）
        subprocess.run(
            ["schtasks.exe", "/Create", "/SC", "HOURLY",
             "/MO", str(schedule_hours),
             "/TN", task_name,
             "/TR", command,
             "/F",  # 强制覆盖
             ],
            capture_output=True, text=True, timeout=15,
        )
        return True
    except FileNotFoundError:
        print("  ⚠ schtasks.exe not found — Windows Task Scheduler unavailable")
        return False
    except Exception:
        return False


def _remove_schtasks(name: str) -> bool:
    """从 Windows Task Scheduler 移除 YiCeNet 任务。"""
    task_name = f"YiCeNet\\{name}"
    try:
        r = subprocess.run(
            ["schtasks.exe", "/Delete", "/TN", task_name, "/F"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def register_flywheel_cron(schedule_hours: int = 6):
    """註冊 flywheel 定時任務（平台自適應）。

    Linux/macOS → crontab 條目
    Windows     → Task Scheduler (schtasks.exe)

    調度周期取自 ~/.yicenet/config.yaml flywheel.schedule_hours，默認 6h。
    """
    import platform
    system = platform.system().lower()

    cron_name = "yicenet-flywheel"
    command = _get_flywheel_command()

    if system in ("linux", "darwin"):
        schedule = _cron_schedule_expr(schedule_hours)
        ok = _register_crontab(schedule, command, cron_name)
        if ok:
            print(f"  ✓ Flywheel cron registered (crontab, every {schedule_hours}h)")
        elif ok is False:
            print(f"  · Flywheel cron already registered (crontab)")
        else:
            print(f"  ⚠ Flywheel cron failed — crontab not available")
            print(f"    手動添加到 crontab (crontab -e)：")
            print(f"    {schedule} {command} >> $HOME/.yicenet/logs/flywheel.log 2>&1")

    elif system == "windows":
        ok = _register_schtasks(schedule_hours, command, cron_name)
        if ok:
            print(f"  ✓ Flywheel task registered (Task Scheduler, every {schedule_hours}h)")
        elif ok is False:
            print(f"  · Flywheel task already registered (Task Scheduler)")
        else:
            print(f"  ⚠ Flywheel task registration failed")

    else:
        print(f"  · Flywheel: unsupported platform '{system}' — skip")
        print(f"    手動設置定時任務：")
        print(f"    {command}")


def unregister_flywheel_cron(silent: bool = False):
    """移除飛輪定時任務（平台自適應）。"""
    import platform
    system = platform.system().lower()
    cron_name = "yicenet-flywheel"
    log = lambda msg: None if silent else print(f"  {msg}")

    if system in ("linux", "darwin"):
        ok = _remove_crontab(cron_name)
        log(f"{'✓' if ok else '·'} Flywheel crontab: {'removed' if ok else 'not found'}")

    elif system == "windows":
        ok = _remove_schtasks(cron_name)
        log(f"{'✓' if ok else '·'} Flywheel Task Scheduler: {'removed' if ok else 'not found'}")

    else:
        log(f"· Flywheel: {system} — manual cleanup needed")


# ══════════════════════════════════════════════════
# Claude Code 集成
# ══════════════════════════════════════════════════


def detect_claude_code() -> tuple[bool, str]:
    """檢測 Claude Code 安裝。返回 (found, claude_path)。"""
    claude_bin = shutil.which("claude")
    if claude_bin:
        return True, claude_bin
    # 即使 CLI 不在 PATH，~/.claude/ 目錄存在也視爲已安裝
    if (Path.home() / ".claude").exists():
        return True, ""
    return False, ""


def setup_claude_code_mcp(python: str) -> bool:
    """寫入 mcpServers 條目到 ~/.claude/settings.json。

    只寫 MCP server 配置（最小化）。
    完整的 hooks（PostToolUse / Stop）由 install-claudecode-hooks.sh 負責。
    """
    # 先安裝 mcp extras
    print("  → Installing yicenet[mcp] extras...")
    r = subprocess.run(
        [python, "-m", "pip", "install", "-e", f"{PROJECT}[mcp]"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        print(f"  ⚠ mcp extras install failed: {r.stderr[:150]}")
        print("  · Continuing without mcp extras (install manually: pip install mcp>=1.0)")

    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # 讀取現有配置（不破壞用戶已有設置）
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ⚠ {settings_path} is malformed JSON — will overwrite mcpServers only")

    # 寫入 MCP server 條目
    settings.setdefault("mcpServers", {})
    settings["mcpServers"]["yicenet"] = {
        "command": "yicenet-serve",
        "args": [],
        "env": {"YICENET_HOME": str(PROJECT)},
    }

    try:
        settings_path.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  ✓ mcpServers.yicenet written to {settings_path}")
        print(f"  · Tools: yicenet_attend · yicenet_predict · yicenet_feedback · yicenet_switch")
        print(f"  · For full hooks (flywheel), run: scripts/install/install-claudecode-hooks.sh")
        return True
    except Exception as e:
        print(f"  ⚠ Failed to write {settings_path}: {e}")
        return False


# ══════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════


def init_data_root(soul_path: str = "") -> None:
    """创建 ~/.yicenet/ 目录结构 + config.yaml 模板 + SOUL 副本.

    - ~/.yicenet/config.yaml         (从 DEFAULT_CONFIG_YAML 写入，不覆盖已有文件)
    - ~/.yicenet/SOUL.md             (从模板拷贝，不覆盖已有文件)
    - ~/.yicenet/checkpoints/
    - ~/.yicenet/data/
    - ~/.yicenet/logs/
    """
    try:
        from yicenet.config import yicenet_data_dir, DEFAULT_CONFIG_YAML
    except ImportError:
        print("  ⚠ yicenet not installed yet — skipping data root init")
        return

    data_root = yicenet_data_dir()
    created = []

    # Directories
    for sub in ("checkpoints", "data", "logs"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)

    # config.yaml (from template)
    cfg_path = data_root / "config.yaml"
    if not cfg_path.exists():
        cfg_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
        created.append(f"config.yaml")

    # SOUL.md
    soul_dst = data_root / "SOUL.md"
    if not soul_dst.exists():
        if soul_path and os.path.isfile(soul_path):
            shutil.copy2(soul_path, soul_dst)
            created.append(f"SOUL.md (from {soul_path})")
        else:
            # Look for SOUL-template.md in project root
            project_soul = PROJECT / "SOUL-template.md"
            if project_soul.exists():
                shutil.copy2(str(project_soul), str(soul_dst))
                created.append("SOUL.md (from SOUL-template.md)")
            else:
                print("  · SOUL.md: not found (SKIP)")

    if created:
        print(f"  ✓ ~/.yicenet/ created: {', '.join(created)}")
    else:
        print("  · ~/.yicenet/ already initialized (no changes)")


# ══════════════════════════════════════════════════
# 卸载
# ══════════════════════════════════════════════════


def uninstall(clean_data: bool = False, silent: bool = False) -> None:
    """卸载 YiCeNet 并从目标环境移除注册.

    Steps:
    1. 从 Hermes plugin 目录移除 yicenet-hooks
    2. 从 ~/.claude/settings.json 移除 MCP 条目
    3. pip uninstall yicenet (optional)
    4. --clean-data: rm -rf ~/.yicenet/
    """
    log = lambda msg: None if silent else print(f"  {msg}")

    print()
    print("╔══════════════════════════════════════════╗")
    print("║  YiCeNet Uninstall                       ║")
    print("╚══════════════════════════════════════════╝")

    # 1. Hermes plugin
    hermes_plugins = Path.home() / ".hermes" / "plugins"
    yicenet_dir = hermes_plugins / "yicenet-hooks"
    if yicenet_dir.exists():
        try:
            shutil.rmtree(str(yicenet_dir))
            log("✓ Hermes plugin: yicenet-hooks removed")
        except Exception as e:
            log(f"✗ Hermes plugin removal failed: {e}")
    else:
        log("· Hermes plugin: not found")

    # 2. Claude Code MCP config
    claude_settings = Path.home() / ".claude" / "settings.json"
    if claude_settings.exists():
        try:
            settings = json.loads(claude_settings.read_text(encoding="utf-8"))
            mcps = settings.get("mcpServers", {})
            if "yicenet" in mcps:
                del mcps["yicenet"]
                settings["mcpServers"] = mcps
                claude_settings.write_text(
                    json.dumps(settings, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                log("✓ Claude Code MCP: yicenet entry removed")
            else:
                log("· Claude Code MCP: not found")
        except Exception as e:
            log(f"✗ Claude Code MCP cleanup failed: {e}")
    else:
        log("· Claude Code settings: not found")

    # 3. Flywheel scheduler
    log("· Flywheel scheduler: cleaning up...")
    unregister_flywheel_cron(silent=silent)

    # 4. Data cleanup
    if clean_data:
        data_root = Path.home() / ".yicenet"
        if data_root.exists():
            try:
                shutil.rmtree(str(data_root))
                log(f"✓ ~/.yicenet/ removed")
            except Exception as e:
                log(f"✗ ~/.yicenet/ removal failed: {e}")
        else:
            log("· ~/.yicenet/: not found")

    print()
    print("  卸载完成。如需彻底删除包，请运行：")
    print("    pip uninstall yicenet")


def uninstall_cli() -> None:
    parser = argparse.ArgumentParser(description="Uninstall YiCeNet")
    parser.add_argument("--clean-data", action="store_true",
                        help="删除 ~/.yicenet/ 所有数据 (checkpoints, config, 日志)")
    args = parser.parse_args()
    uninstall(clean_data=args.clean_data)


def setup_env():
    """創建 .env（如果不存在）。"""
    env_path = PROJECT / ".env"
    if env_path.exists():
        return

    try:
        shutil.copy2(PROJECT / ".env.example", env_path) if (PROJECT / ".env.example").exists() else None
        if env_path.exists():
            print(f"  ✓ .env created from template")
    except Exception:
        pass


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════


def bootstrap(auto: bool = False, venv: str = "",
              skip_cron: bool = False, skip_hermes: bool = False,
              target: str = "auto", soul: str = ""):
    """執行 YiCeNet 完整初始化。

    target: 'auto'        — 檢測所有已安裝 IDE，逐一配置
            'hermes'      — 只配置 Hermes
            'claude-code' — 只配置 Claude Code MCP server
    """
    print()
    print("╔══════════════════════════════════════════╗")
    print("║  YiCeNet Bootstrap — 一鍵初始化         ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  項目:   {PROJECT}")
    print(f"  Target: {target}")
    print()

    # ── Phase 1: 環境檢測 ──
    print("── Phase 1: 環境檢測 ──")
    hermes_ok, hermes_python = detect_hermes()
    claude_ok, claude_bin = detect_claude_code()
    torch_ok, torch_info = detect_torch()
    print(f"  Hermes:      {'✓' if hermes_ok else '✗ not found'}"
          f"{' (' + hermes_python + ')' if hermes_python else ''}")
    print(f"  Claude Code: {'✓' if claude_ok else '✗ not found'}"
          f"{' (' + claude_bin + ')' if claude_bin else ''}")
    print(f"  PyTorch:     {'✓ ' + torch_info if torch_ok else '✗ not installed'}")
    print()

    # ── Phase 2: 安裝到目標 venv ──
    print("── Phase 2: 安裝 YiCeNet ──")

    # target_python 優先順序：--venv > Hermes venv > 當前 Python
    if venv:
        target_python = venv
    elif hermes_python and target in ("auto", "hermes"):
        target_python = hermes_python
    else:
        # Claude Code target 或找不到 Hermes venv：裝到當前 Python
        # 這樣 yicenet-serve 才能出現在用戶的 PATH 裏
        standalone_venv = PROJECT / ".venv"
        if not hermes_python and not standalone_venv.exists():
            print("  → Creating standalone venv...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(standalone_venv)],
                capture_output=True, text=True, timeout=30,
            )
        target_python = (
            str(standalone_venv / "bin" / "python3")
            if standalone_venv.exists() and not hermes_python
            else sys.executable
        )

    print(f"  目標 Python: {target_python}")
    # claude-code target 額外安裝 mcp extras
    extras = "mcp" if target == "claude-code" else ""
    install_ok = install_to_venv(target_python, extras=extras)
    if not install_ok:
        print("  ⚠ Install failed — will continue with fallback (sys.path)")
    print()

    # ── Phase 3: 依賴檢查 ──
    print("── Phase 3: 依賴檢查 ──")
    ensure_deps(target_python)
    print()

    # ── Phase 4: 檢查點 ──
    print("── Phase 4: 檢查點 ──")
    ensure_checkpoints()
    print()

    # ── Phase 5: 資料根初始化 ──
    print("── Phase 5: 資料根初始化 ──")
    init_data_root(soul_path=soul)
    setup_env()
    print()

    # ── Phase 6: IDE 集成 ──
    print("── Phase 6: IDE 集成 ──")

    do_hermes = not skip_hermes and (target in ("auto", "hermes"))
    do_claude = target in ("auto", "claude-code")

    if do_hermes and hermes_ok:
        print("  [Hermes]")
        setup_hermes_tool(hermes_ok)
    elif do_hermes and not hermes_ok:
        print("  [Hermes] ✗ not detected — skip")

    if do_claude and claude_ok:
        print("  [Claude Code]")
        setup_claude_code_mcp(target_python)
    elif do_claude and not claude_ok:
        print("  [Claude Code] ✗ not detected — skip")
        print("  · To force: yicenet-bootstrap --target claude-code")

    if not do_hermes and not do_claude:
        print("  (skipped)")
    print()

    # ── Phase 7: 飛輪定時任務 ──
    if not skip_cron:
        print("── Phase 7: 飛輪定時任務 ──")
        try:
            from yicenet.config import load_user_config
            user_cfg = load_user_config()
            schedule_hours = (
                user_cfg.get("flywheel", {}).get("schedule_hours", 6)
                or 6
            )
        except Exception:
            schedule_hours = 6
        register_flywheel_cron(schedule_hours=schedule_hours)
        print()

    # ── 完成 ──
    print("╔══════════════════════════════════════════╗")
    print("║  YiCeNet Bootstrap 完成！                ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("  驗證:  python3 -c \"import yicenet; print(yicenet.__version__)\"")
    if do_hermes and hermes_ok:
        print("  Hermes: yicenet_predict 工具可用（需重啟 Hermes）")
    if do_claude and claude_ok:
        print("  Claude Code: 重啟 Claude Code 後 MCP server 自動啓動")
        print("               完整 hooks → scripts/install/install-claudecode-hooks.sh")
    print()


def main():
    parser = argparse.ArgumentParser(description="YiCeNet Bootstrap")
    parser.add_argument("--auto", action="store_true", help="全自動（無交互確認）")
    parser.add_argument("--venv", default="", help="目標 venv python 路徑")
    parser.add_argument("--skip-cron", action="store_true", help="跳過 cron 注冊")
    parser.add_argument("--skip-hermes", action="store_true", help="跳過 Hermes 集成")
    parser.add_argument(
        "--target",
        default="auto",
        choices=["auto", "hermes", "claude-code"],
        help="配置目標：auto=檢測所有, hermes=只配 Hermes, claude-code=只配 Claude Code",
    )
    parser.add_argument(
        "--soul", default="",
        help="SOUL 模板路徑（預設: YiCeNet/SOUL-template.md → ~/.yicenet/SOUL.md）",
    )
    args = parser.parse_args()
    bootstrap(
        auto=args.auto,
        venv=args.venv,
        skip_cron=args.skip_cron,
        skip_hermes=args.skip_hermes,
        target=args.target,
        soul=args.soul,
    )


if __name__ == "__main__":
    main()
