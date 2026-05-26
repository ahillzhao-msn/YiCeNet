#!/usr/bin/env python3
"""
YiCeNet Bootstrap — 一鍵安裝初始化。

安裝到現有 venv（共享），必要時創建獨立 venv。
註冊 flywheel cron、Hermes 工具鏈接、檢查點導入。

用法：
  yicenet-bootstrap                       # 交互式（默認）
  yicenet-bootstrap --auto                # 全自動
  yicenet-bootstrap --venv /path/to/venv  # 指定 venv
  python3 scripts/bootstrap.py            # 源碼樹直接運行
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


def install_to_venv(venv_python: str) -> bool:
    """pip install -e . 到指定 venv。"""
    try:
        r = subprocess.run(
            [venv_python, "-m", "pip", "install", "-e", str(PROJECT)],
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


def register_flywheel_cron(hermes_available: bool):
    """註冊 flywheel cron（每 6 小時——學習新對話模式）。"""
    if not hermes_available:
        print("  · Flywheel cron: Hermes not available, skip")
        return

    cron_name = "yicenet-flywheel"
    cron_script_path = PROJECT / "scripts" / "flywheel_cron.sh"

    # 創建 cron wrapper 腳本
    if not cron_script_path.exists():
        _create_flywheel_script(cron_script_path)

    try:
        r = subprocess.run(
            ["hermes", "cron", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if cron_name in r.stdout:
            print(f"  · Flywheel cron '{cron_name}' already registered")
            return

        subprocess.run(
            ["hermes", "cron", "create",
             "--name", cron_name,
             "--schedule", "0 */6 * * *",
             "--script", str(cron_script_path),
             ],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  ✓ Flywheel cron registered (every 6h)")
    except Exception as e:
        print(f"  ⚠ Flywheel cron error: {e}")


def _create_flywheel_script(path: Path):
    """創建 flywheel cron wrapper。"""
    content = f"""#!/usr/bin/env bash
# YiCeNet Flywheel — Hermes cron wrapper
# Runs model update training, logs to {PROJECT}/logs/

cd {PROJECT}
python3 -m yicenet.flywheel >> logs/flywheel.log 2>&1
echo "flywheel: done at $(date)"
"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)
    except Exception:
        pass


# ══════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════


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
              skip_cron: bool = False, skip_hermes: bool = False):
    """執行 YiCeNet 完整初始化。"""
    print()
    print("╔══════════════════════════════════════════╗")
    print("║  YiCeNet Bootstrap — 一鍵初始化         ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  項目: {PROJECT}")
    print()

    # ── Phase 1: 環境檢測 ──
    print("── Phase 1: 環境檢測 ──")
    hermes_ok, hermes_python = detect_hermes()
    torch_ok, torch_info = detect_torch()
    print(f"  Hermes:        {'✓' if hermes_ok else '✗ not found'} "
          f"{'(' + hermes_python + ')' if hermes_python else ''}")
    print(f"  PyTorch:       {'✓ ' + torch_info if torch_ok else '✗ not installed'}")
    print()

    # ── Phase 2: 安裝到目標 venv ──
    print("── Phase 2: 安裝 YiCeNet ──")

    target_python = ""
    if venv:
        # 用戶指定
        target_python = venv
    elif hermes_python:
        # Hermes venv
        target_python = hermes_python
    else:
        # 創建獨立 venv
        standalone_venv = PROJECT / ".venv"
        if not standalone_venv.exists():
            print("  → Creating standalone venv...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(standalone_venv)],
                capture_output=True, text=True, timeout=30,
            )
        target_python = str(standalone_venv / "bin" / "python3")

    print(f"  目標 Python: {target_python}")
    install_ok = install_to_venv(target_python)

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

    # ── Phase 5: 配置 ──
    print("── Phase 5: 配置 ──")
    setup_env()
    print()

    # ── Phase 6: Hermes 集成 ──
    if not skip_hermes:
        print("── Phase 6: Hermes 集成 ──")
        setup_hermes_tool(hermes_ok)
        print()

    # ── Phase 7: Cron ──
    if not skip_cron:
        print("── Phase 7: Cron ──")
        register_flywheel_cron(hermes_ok)
        print()

    # ── 完成 ──
    print("╔══════════════════════════════════════════╗")
    print("║  YiCeNet Bootstrap 完成！                ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("  驗證:  python3 -c \"import yicenet; print(yicenet.__version__)\"")
    print("  推理:  python3 -m yicenet.predict \"你的問題\"")
    print("  Hermes: yicenet_predict 工具可用（需重啟 Hermes）")
    print()


def main():
    parser = argparse.ArgumentParser(description="YiCeNet Bootstrap")
    parser.add_argument("--auto", action="store_true", help="全自動")
    parser.add_argument("--venv", default="", help="目標 venv python 路徑")
    parser.add_argument("--skip-cron", action="store_true", help="跳過 cron")
    parser.add_argument("--skip-hermes", action="store_true", help="跳過 Hermes 集成")
    args = parser.parse_args()
    bootstrap(auto=args.auto, venv=args.venv,
              skip_cron=args.skip_cron, skip_hermes=args.skip_hermes)


if __name__ == "__main__":
    main()
