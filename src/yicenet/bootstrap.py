#!/usr/bin/env python3
"""YiCeNet Bootstrap — 一鍵安裝/卸載初始化 (orchestrator).

Delegates IDE-specific work to install/ package (HermesInstaller,
ClaudeCodeInstaller). This file owns only platform-agnostic concerns:
checkpoint management, tokenizer cache, data-root init, flywheel cron.

Usage:
  yicenet-bootstrap                   # auto: configure all detected IDEs
  yicenet-bootstrap --target hermes   # Hermes only
  yicenet-bootstrap --target claude-code  # Claude Code hooks (Option C)
  yicenet-uninstall [--clean-data]    # remove registrations
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── project root ───────────────────────────────────────────────────────────────

def _project_root() -> Path:
    p = Path(__file__).resolve().parent.parent
    if (p / "pyproject.toml").exists():
        return p
    if (p.parent / "pyproject.toml").exists():
        return p.parent
    return Path.cwd()


PROJECT = _project_root()


# ── environment probes ─────────────────────────────────────────────────────────

def detect_torch() -> tuple[bool, str]:
    try:
        import torch
        cuda = torch.cuda.is_available()
        ver = torch.__version__
        gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
        return True, f"{ver} ({'GPU: ' + gpu if cuda else 'CPU'})"
    except ImportError:
        return False, "not installed"


def ensure_deps(python: str) -> None:
    try:
        r = subprocess.run([python, "-c", "import torch; print('ok')"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return
    except Exception:
        pass
    print("  → Installing PyTorch (CPU)...")
    try:
        subprocess.run(
            [python, "-m", "pip", "install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            capture_output=True, text=True, timeout=300,
        )
    except Exception:
        print("  ⚠ PyTorch install failed (soft — continue)")


# ── checkpoints ────────────────────────────────────────────────────────────────

_GITHUB_RELEASE_BASE = (
    "https://github.com/ahillzhao-msn/YiCeNet/releases/latest/download"
)
_RELEASE_ASSETS = [
    ("yicenet_base.pt", "yicenet_base.pt"),
    ("world_model.pt",  "world_model_best.pt"),
    ("registry.json",   "registry.json"),
]


def _download_release_checkpoints(dest_dir: Path) -> bool:
    import urllib.request
    import urllib.error

    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []

    def _progress(n: int, bs: int, total: int) -> None:
        if total > 0:
            pct = min(100, n * bs * 100 // total)
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r    [{bar}] {pct:3d}%", end="", flush=True)

    for remote_name, local_name in _RELEASE_ASSETS:
        dst = dest_dir / local_name
        try:
            print(f"  → {remote_name}")
            urllib.request.urlretrieve(
                f"{_GITHUB_RELEASE_BASE}/{remote_name}", str(dst), reporthook=_progress
            )
            print(f"\n    saved → {dst.name}  ({dst.stat().st_size / 1_048_576:.1f} MB)")
            downloaded.append(local_name)
        except urllib.error.HTTPError as e:
            print(f"\n  ⚠ HTTP {e.code}: {remote_name}")
            if dst.exists():
                dst.unlink()
        except urllib.error.URLError as e:
            print(f"\n  ⚠ Network: {e.reason}")
            if dst.exists():
                dst.unlink()
            break
        except Exception as e:
            print(f"\n  ⚠ {e}")
            if dst.exists():
                dst.unlink()
    return "yicenet_base.pt" in downloaded


def ensure_checkpoints() -> bool:
    checkpoints_dir = PROJECT / "checkpoints"
    registry_path = checkpoints_dir / "registry.json"

    # 1. registry points to an existing .pt
    if registry_path.exists():
        try:
            reg = json.loads(registry_path.read_text())
            ap = reg.get("active", {}).get("path", "")
            if ap and (checkpoints_dir / ap).exists():
                print(f"  ✓ Checkpoint: {reg['active'].get('version', '?')} ({ap})")
                return True
        except Exception:
            pass

    # 2. .pt files present but registry stale — rebuild
    pt_files = list(checkpoints_dir.glob("*.pt")) if checkpoints_dir.exists() else []
    if pt_files:
        try:
            r = subprocess.run(
                [sys.executable,
                 str(PROJECT / "scripts" / "checkpoint_manager.py"), "fresh"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                print(f"  ✓ Registry rebuilt from {len(pt_files)} checkpoint(s)")
                return True
        except Exception:
            pass

    # 3. copy from sibling clone
    source_dir = Path.home() / "YiCeNet"
    if source_dir != PROJECT and (source_dir / "checkpoints").exists():
        src_pt = list((source_dir / "checkpoints").glob("*.pt"))
        if src_pt:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            for f in src_pt:
                shutil.copy2(f, checkpoints_dir / f.name)
            print(f"  ✓ Copied {len(src_pt)} checkpoint(s) from {source_dir}")
            return True

    # 4. download from GitHub Releases
    print("  · Trying GitHub Releases...")
    if _download_release_checkpoints(checkpoints_dir):
        try:
            subprocess.run(
                [sys.executable,
                 str(PROJECT / "scripts" / "checkpoint_manager.py"), "fresh"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            pass
        if any(checkpoints_dir.glob("yicenet_base.pt")):
            print("  ✓ yicenet_base.pt downloaded")
            return True

    # 5. generate random-init model
    print("  · Generating minimal model (random init)...")
    try:
        code = (
            "import sys, torch\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(PROJECT)!r})\n"
            "from yicenet.model import YiCeNet\nfrom yicenet.config import YiCeNetConfig\n"
            "m = YiCeNet(YiCeNetConfig())\n"
            f"ck = Path({str(PROJECT / 'checkpoints')!r}) / 'init_minimal.pt'\n"
            "ck.parent.mkdir(parents=True, exist_ok=True)\n"
            "torch.save(m.state_dict(), ck)\nprint('ok')"
        )
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            print("  ✓ Minimal model generated (random init)")
            return True
        print(f"  ⚠ Model generation failed: {r.stderr[:100]}")
    except Exception as e:
        print(f"  ⚠ {e}")
    return False


# ── data root ──────────────────────────────────────────────────────────────────

def init_data_root(soul_path: str = "") -> None:
    try:
        from yicenet.config import yicenet_data_dir, DEFAULT_CONFIG_YAML
    except ImportError:
        print("  ⚠ yicenet not installed yet — skipping data root init")
        return

    data_root = yicenet_data_dir()
    for sub in ("checkpoints", "data", "logs"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    cfg_path = data_root / "config.yaml"
    if not cfg_path.exists():
        cfg_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
        created.append("config.yaml")

    soul_dst = data_root / "SOUL.md"
    if not soul_dst.exists():
        src = Path(soul_path) if soul_path and os.path.isfile(soul_path) else PROJECT / "SOUL-template.md"
        if src.exists():
            shutil.copy2(str(src), str(soul_dst))
            created.append("SOUL.md")

    if created:
        print(f"  ✓ ~/.yicenet/: {', '.join(created)}")
    else:
        print("  · ~/.yicenet/: already initialized")


# ── flywheel cron ──────────────────────────────────────────────────────────────

def _flywheel_cmd() -> str:
    return f'"{sys.executable}" -m yicenet.flywheel'


def register_flywheel_cron(hours: int = 6) -> None:
    import platform
    system = platform.system().lower()
    name = "yicenet-flywheel"
    cmd = _flywheel_cmd()

    if system in ("linux", "darwin"):
        schedule = f"0 */{max(1, hours)} * * *"
        try:
            r = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=10)
            existing = r.stdout if r.returncode == 0 else ""
            if name in existing:
                print(f"  · Flywheel cron: already registered")
                return
            new_cron = (
                existing.rstrip()
                + f"\n# YiCeNet Flywheel — {name}\n"
                + f"{schedule} {cmd} >> $HOME/.yicenet/logs/flywheel.log 2>&1\n"
            )
            subprocess.run(["crontab", "-"], input=new_cron, capture_output=True,
                           text=True, timeout=10)
            print(f"  ✓ Flywheel cron registered (every {hours}h)")
        except FileNotFoundError:
            print(f"  · Flywheel: crontab not available — add manually:\n    {schedule} {cmd}")

    elif system == "windows":
        task = f"YiCeNet\\{name}"
        try:
            r = subprocess.run(
                ["schtasks.exe", "/Query", "/TN", task, "/FO", "LIST"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"  · Flywheel task: already registered")
                return
            subprocess.run(
                ["schtasks.exe", "/Create", "/SC", "HOURLY", "/MO", str(hours),
                 "/TN", task, "/TR", cmd, "/F"],
                capture_output=True, text=True, timeout=15,
            )
            print(f"  ✓ Flywheel task registered (every {hours}h)")
        except FileNotFoundError:
            print(f"  · Flywheel: schtasks.exe not found")
    else:
        print(f"  · Flywheel: {system} — add manually: {cmd}")


def unregister_flywheel_cron(silent: bool = False) -> None:
    import platform
    system = platform.system().lower()
    log = (lambda m: None) if silent else (lambda m: print(f"  {m}"))
    name = "yicenet-flywheel"

    if system in ("linux", "darwin"):
        try:
            r = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return
            lines = [l for l in r.stdout.split("\n")
                     if name not in l and "YiCeNet Flywheel" not in l]
            subprocess.run(["crontab", "-"], input="\n".join(lines).strip() + "\n",
                           capture_output=True, text=True, timeout=10)
            log("✓ Flywheel crontab: removed")
        except FileNotFoundError:
            pass

    elif system == "windows":
        try:
            r = subprocess.run(
                ["schtasks.exe", "/Delete", "/TN", f"YiCeNet\\{name}", "/F"],
                capture_output=True, text=True, timeout=10,
            )
            log(f"{'✓' if r.returncode == 0 else '·'} Flywheel task: "
                f"{'removed' if r.returncode == 0 else 'not found'}")
        except Exception:
            pass


# ── uninstall ──────────────────────────────────────────────────────────────────

def uninstall(clean_data: bool = False, silent: bool = False) -> None:
    log = (lambda m: None) if silent else (lambda m: print(f"  {m}"))
    print("\n╔══════════════════════════════════════╗")
    print("║  YiCeNet Uninstall                   ║")
    print("╚══════════════════════════════════════╝")

    from yicenet.install import HermesInstaller, ClaudeCodeInstaller

    if HermesInstaller().detect():
        try:
            HermesInstaller().unregister()
            log("✓ Hermes plugin: removed")
        except Exception as e:
            log(f"✗ Hermes removal failed: {e}")
    else:
        log("· Hermes plugin: not found")

    try:
        ClaudeCodeInstaller().unregister()
        log("✓ Claude Code hooks: removed")
    except Exception as e:
        log(f"· Claude Code: {e}")

    log("· Flywheel: cleaning up...")
    unregister_flywheel_cron(silent=silent)

    if clean_data:
        data_root = Path.home() / ".yicenet"
        if data_root.exists():
            try:
                shutil.rmtree(str(data_root))
                log("✓ ~/.yicenet/ removed")
            except Exception as e:
                log(f"✗ ~/.yicenet/ removal failed: {e}")
        else:
            log("· ~/.yicenet/: not found")

    print("\n  卸载完成。如需彻底删除包：pip uninstall yicenet\n")


def uninstall_cli() -> None:
    parser = argparse.ArgumentParser(description="Uninstall YiCeNet")
    parser.add_argument("--clean-data", action="store_true",
                        help="删除 ~/.yicenet/ 所有数据")
    args = parser.parse_args()
    uninstall(clean_data=args.clean_data)


# ── bootstrap orchestrator ─────────────────────────────────────────────────────

def bootstrap(
    auto: bool = False,
    venv: str = "",
    skip_cron: bool = False,
    skip_hermes: bool = False,
    target: str = "auto",
    soul: str = "",
) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║  YiCeNet Bootstrap — 一鍵初始化     ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Project: {PROJECT}\n  Target:  {target}\n")

    from yicenet.install import HermesInstaller, ClaudeCodeInstaller

    hermes = HermesInstaller()
    claude = ClaudeCodeInstaller()
    hermes_ok = hermes.detect()
    claude_ok = claude.detect()
    torch_ok, torch_info = detect_torch()

    print("── Phase 1: Environment ──")
    print(f"  Hermes:      {'✓' if hermes_ok else '✗ not found'}")
    print(f"  Claude Code: {'✓' if claude_ok else '✗ not found'}")
    print(f"  PyTorch:     {'✓ ' + torch_info if torch_ok else '✗ not installed'}\n")

    print("── Phase 2: Install Package ──")
    installer = hermes if (hermes_ok and target in ("auto", "hermes")) else claude
    pkg_path = Path(venv) if venv else PROJECT
    if installer.install_package(editable_path=pkg_path):
        print(f"  ✓ YiCeNet installed")
    else:
        print("  ⚠ Install failed — continuing with current sys.path")
    print()

    print("── Phase 3: Dependencies ──")
    ensure_deps(sys.executable)
    print()

    print("── Phase 4: Checkpoints ──")
    ensure_checkpoints()
    print()

    print("── Phase 4b: Tokenizer ──")
    try:
        from yicenet.tokenizer import download_tokenizer, tokenizer_available
        if tokenizer_available():
            print("  ✓ Already cached")
        else:
            download_tokenizer()
    except Exception as e:
        print(f"  ⚠ {e}\n  (Tokenizer will load from HF Hub at first use)")
    print()

    print("── Phase 5: Data Root ──")
    init_data_root(soul_path=soul)
    env_path = PROJECT / ".env"
    if not env_path.exists() and (PROJECT / ".env.example").exists():
        shutil.copy2(PROJECT / ".env.example", env_path)
        print("  ✓ .env created from template")
    print()

    print("── Phase 6: IDE Integration ──")
    do_hermes = not skip_hermes and target in ("auto", "hermes")
    do_claude = target in ("auto", "claude-code")

    if do_hermes:
        if hermes_ok:
            print("  [Hermes]")
            try:
                hermes.register_hooks()
                print("  ✓ Hermes plugin registered")
            except Exception as e:
                print(f"  ⚠ Hermes registration failed: {e}")
        else:
            print("  [Hermes] ✗ not detected — skip")

    if do_claude:
        if claude_ok:
            print("  [Claude Code]")
            try:
                claude.register_hooks()
                print("  ✓ Claude Code hooks registered (UserPromptSubmit/PostToolUse/Stop)")
            except Exception as e:
                print(f"  ⚠ Claude Code registration failed: {e}")
        else:
            print("  [Claude Code] ✗ not detected — skip")

    if not do_hermes and not do_claude:
        print("  (skipped)")
    print()

    if not skip_cron:
        print("── Phase 7: Flywheel Cron ──")
        try:
            from yicenet.config import load_user_config
            hours = load_user_config().get("flywheel", {}).get("schedule_hours", 6) or 6
        except Exception:
            hours = 6
        register_flywheel_cron(hours)
        print()

    print("╔══════════════════════════════════════╗")
    print("║  YiCeNet Bootstrap 完成！            ║")
    print("╚══════════════════════════════════════╝\n")
    print('  验证: python3 -c "import yicenet; print(yicenet.__version__)"')
    if do_claude and claude_ok:
        print("  Claude Code: restart Claude Code to activate hooks")
    if do_hermes and hermes_ok:
        print("  Hermes: restart Hermes to activate yicenet-hooks plugin")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="YiCeNet Bootstrap")
    parser.add_argument("--auto", action="store_true", help="全自動（無交互確認）")
    parser.add_argument("--venv", default="", help="目標 venv python 路徑")
    parser.add_argument("--skip-cron", action="store_true", help="跳過 cron 注冊")
    parser.add_argument("--skip-hermes", action="store_true", help="跳過 Hermes 集成")
    parser.add_argument(
        "--target", default="auto",
        choices=["auto", "hermes", "claude-code"],
        help="配置目標：auto=所有, hermes=只配 Hermes, claude-code=只配 Claude Code",
    )
    parser.add_argument("--soul", default="", help="SOUL 模板路徑")
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
