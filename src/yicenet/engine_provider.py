"""
EngineProvider — unified engine factory for all integration points.

Replaces the duplicated _get_engine() in hermes_tool.py and mcp_server.py.
Adding a new platform (VSCode extension, Neovim, etc.) requires zero new
engine-init logic — just call EngineProvider.get_engine().
"""
from __future__ import annotations

import json
from pathlib import Path


class EngineProvider:
    """Class-level singleton engine with registry-aware hot-switch support."""

    _engine = None
    _active_version: str = ""

    @classmethod
    def get_engine(cls):
        """Return the active engine, lazy-initializing on first call."""
        if cls._engine is None:
            cls._apply_runtime_env()
            cls._engine = cls._build_engine()
        return cls._engine

    @classmethod
    def _apply_runtime_env(cls) -> None:
        """Set TQDM_DISABLE / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE from config.

        Must run before the model is loaded (sentence-transformers reads these
        at instantiation time). Defaults match config.yaml runtime defaults.
        """
        import os
        try:
            from .config import load_user_config
            rt = load_user_config().get("runtime", {})
        except Exception:
            rt = {}
        if rt.get("transformers_offline", True):
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if rt.get("hf_hub_offline", True):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if rt.get("tqdm_disable", True):
            os.environ.setdefault("TQDM_DISABLE", "1")

    @classmethod
    def check_switch(cls) -> bool:
        """Check registry.json for a version change and hot-switch if needed.

        Returns True if a switch occurred. Call periodically (e.g. per-request
        in Hermes or from a cron job via engine.check_for_switch()).
        """
        from .config import yicenet_checkpoint_dir
        reg_path = yicenet_checkpoint_dir() / "registry.json"
        if not reg_path.exists():
            return False
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            active = reg.get("active", {})
            new_version = active.get("version", "")
            new_path = active.get("path", "")
            if new_path and not Path(new_path).is_absolute():
                from .config import yicenet_checkpoint_dir as _ckpt_dir
                new_path = str(_ckpt_dir() / new_path)
            if new_version == cls._active_version or not new_path:
                return False
            if not Path(new_path).exists():
                return False
            if cls._engine is not None:
                cls._engine.switch_model(new_path)
            else:
                cls._engine = cls._build_engine(checkpoint=new_path)
            cls._active_version = new_version
            return True
        except Exception:
            return False

    @classmethod
    def switch(cls, checkpoint: str) -> bool:
        """Explicit hot-switch to a named checkpoint path."""
        return cls.get_engine().switch_model(checkpoint)

    @classmethod
    def _build_engine(cls, checkpoint: str = ""):
        from .yicenet_engine import YiCeNetEngine
        from .config import yicenet_home
        ckpt_path = checkpoint or cls._resolve_checkpoint()
        cls._active_version = cls._read_active_version()
        return YiCeNetEngine(
            checkpoint=ckpt_path,
            project_root=str(yicenet_home()),
        )

    @classmethod
    def _resolve_checkpoint(cls) -> str:
        from .config import yicenet_checkpoint_dir
        checkpoint_dir = yicenet_checkpoint_dir()
        reg_path = checkpoint_dir / "registry.json"
        if reg_path.exists():
            try:
                reg = json.loads(reg_path.read_text(encoding="utf-8"))
                active = reg.get("active", {})
                ckpt = active.get("path", "")
                if ckpt and not Path(ckpt).is_absolute():
                    ckpt = str(checkpoint_dir / ckpt)
                if ckpt and Path(ckpt).exists():
                    return ckpt
            except Exception:
                pass
        # Auto-discover latest versioned checkpoint
        pt_files = sorted(checkpoint_dir.glob("yicenet_v*.pt"))
        if pt_files:
            return str(pt_files[-1])
        # Final fallback: minimal checkpoint distributed with releases
        fallback = checkpoint_dir / "minimal.pt"
        if fallback.exists():
            return str(fallback)
        raise RuntimeError(
            f"No YiCeNet checkpoint found in {checkpoint_dir}. "
            "Run 'yicenet-bootstrap' to download checkpoints."
        )

    @classmethod
    def _read_active_version(cls) -> str:
        from .config import yicenet_checkpoint_dir
        reg_path = yicenet_checkpoint_dir() / "registry.json"
        if not reg_path.exists():
            return ""
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            return reg.get("active", {}).get("version", "")
        except Exception:
            return ""
