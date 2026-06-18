"""
YiCeNet configuration — all hyperparameters in one place.

Path Resolution (three-tier priority):
  1. YICENET_HOME env var           — always wins (CI / container / forced override)
  2. Editable install detected      — uses source tree (developer mode)
  3. Wheel install (default)        — uses ~/.yicenet/ (end-user mode)

User config overlay:
  ~/.yicenet/config.yaml overrides YiCeNetConfig defaults for runtime-tunable
  params (flywheel schedule, inference temperature, eval API). Model architecture
  params are frozen per checkpoint and not exposed here.
"""

import json as _json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Install-mode detection ─────────────────────────────────────────────────

_INSTALL_MODE_CACHE: Optional[str] = None


def _detect_install_mode() -> str:
    """Return 'editable' or 'wheel' based on importlib.metadata."""
    global _INSTALL_MODE_CACHE
    if _INSTALL_MODE_CACHE is not None:
        return _INSTALL_MODE_CACHE
    try:
        import importlib.metadata
        raw = importlib.metadata.distribution("yicenet").read_text("direct_url.json")
        if raw and _json.loads(raw).get("dir_info", {}).get("editable", False):
            _INSTALL_MODE_CACHE = "editable"
            return _INSTALL_MODE_CACHE
    except Exception:
        pass
    _INSTALL_MODE_CACHE = "wheel"
    return _INSTALL_MODE_CACHE


# ── Home directory ─────────────────────────────────────────────────────────

_YICENET_HOME_CACHE: Optional[Path] = None


def yicenet_home() -> Path:
    """Unified root directory for YiCeNet runtime data (checkpoints, data, logs).

    Priority:
      1. YICENET_HOME env var  → explicit override (CI, containers, dev shortcuts)
      2. Editable install      → source tree root (developer working on YiCeNet itself)
      3. Wheel install         → ~/.yicenet/  (end users, multi-IDE shared store)
    """
    global _YICENET_HOME_CACHE
    if _YICENET_HOME_CACHE is not None:
        return _YICENET_HOME_CACHE

    env = os.environ.get("YICENET_HOME")
    if env:
        _YICENET_HOME_CACHE = Path(env).expanduser().resolve()
    elif _detect_install_mode() == "editable":
        # config.py lives at src/yicenet/config.py → go up three levels to project root
        _YICENET_HOME_CACHE = Path(__file__).resolve().parent.parent.parent
    else:
        _YICENET_HOME_CACHE = Path.home() / ".yicenet"

    return _YICENET_HOME_CACHE


# ── Sub-directories ───────────────────────────────────────────────────────

def yicenet_data_dir() -> Path:
    return yicenet_home() / "data"


def yicenet_checkpoint_dir() -> Path:
    return yicenet_home() / "checkpoints"


def yicenet_log_dir() -> Path:
    return yicenet_home() / "logs"


# ── User config overlay ────────────────────────────────────────────────────

# Template written to ~/.yicenet/config.yaml on first bootstrap
DEFAULT_CONFIG_YAML = """\
# YiCeNet user configuration — ~/.yicenet/config.yaml
# Model architecture params (hidden_dim, num_heads, …) are NOT here;
# they are frozen per checkpoint. Only runtime-tunable settings live here.
# Environment variables always take highest priority over these values.

eval:
  api_url: ""      # Teacher model endpoint  (overrides EVAL_API_URL)
  model:   ""      # Teacher model name       (overrides EVAL_MODEL)
  api_key: ""      # Prefer env var EVAL_API_KEY — avoid storing keys in files

flywheel:
  schedule_hours: 6      # Cron interval for autonomous training
  min_buffer_size: 20    # Minimum samples required before a training run
  slow_tau_days: 30.0    # World Model head-A power-law decay constant (hexagram)
  fast_tau_days: 3.0     # World Model head-B power-law decay constant (external)

inference:
  gumbel_tau_init: 1.0   # Initial Gumbel-Softmax temperature
  gumbel_tau_min:  0.1   # Minimum temperature after annealing
  default_temperature: 0.1

runtime:
  transformers_offline: true   # Disable HF Hub network calls (use local cache)
  hf_hub_offline: true         # Same for huggingface_hub
  tqdm_disable: true           # Suppress progress bars (MCP server / Hermes plugin)

# ── Display ──
# mode:             compact  — [䷟ 恒] * 亨无咎利贞  (default)
#                   detailed — multi-line with Q-values, candidates, hint
#                   json     — structured JSON for machine consumers
#                   silent   — no output (testing)
# hexagram_chain:   prepend turn-history prefix when session_id is provided
#                   requires hooks to supply chain argument; default false
# unicode_symbols:  true | false; absent = auto-detect from terminal encoding
display:
  mode: compact
  hexagram_chain: false
  # unicode_symbols: true   # uncomment to force; default = auto-detect

# ── SOUL template integration ──
soul:
  enabled: true               # Load SOUL.md for hexagram prior + injection
  template_path: ""           # Empty = ~/.yicenet/SOUL.md
  priority_weights: [0.3, 0.2, 0.2, 0.3]  # 心/骨/皮/用 prior contribution
  inject_targets: ["hermes", "claude-code"]
  injection_level: "summary"  # "none" | "summary" | "full"

# ── Per-platform overrides ────────────────────────────────────────────────────
# Settings under platforms.<id> are deep-merged on top of the globals above.
# Use platform_id values: "hermes", "claude-code", "claude-code-mcp".
# Any key not listed here inherits the global value.
platforms:
  hermes:
    display:
      hexagram_chain: true   # 长会话显示卦链演进
      mode: compact
    memory:
      persist_daemon_sessions: true
      store_vectors: false
      session_ttl_hours: 48.0

  claude-code:
    display:
      hexagram_chain: false  # 短上下文窗口，不显卦链
      mode: compact
    daemon:
      port: 7788             # HTTP side-channel port for hybrid mode IPC
    memory:
      persist_daemon_sessions: false
"""

_USER_CONFIG_CACHE: Optional[dict] = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_platform_config(platform_id: str) -> dict:
    """Return config merged for a specific platform.

    Merges global settings with the platforms.<platform_id> override section.
    Unknown platform_id → returns global config unchanged (safe default).

    Example config.yaml layout::

        display:
          mode: compact          # global default

        platforms:
          hermes:
            display:
              hexagram_chain: true   # overrides global for hermes only
          claude-code:
            daemon:
              port: 7788
    """
    user = load_user_config()
    global_cfg = {k: v for k, v in user.items() if k != "platforms"}
    platform_overrides = user.get("platforms", {}).get(platform_id, {})
    if not platform_overrides:
        return global_cfg
    return _deep_merge(global_cfg, platform_overrides)


def load_user_config(force_reload: bool = False) -> dict:
    """Load ~/.yicenet/config.yaml and return as a dict.

    Returns {} silently if the file is absent or pyyaml is not installed.
    Call with force_reload=True to invalidate the cache (e.g. after bootstrap).
    """
    global _USER_CONFIG_CACHE
    if _USER_CONFIG_CACHE is not None and not force_reload:
        return _USER_CONFIG_CACHE

    config_path = Path.home() / ".yicenet" / "config.yaml"
    if not config_path.exists():
        _USER_CONFIG_CACHE = {}
        return _USER_CONFIG_CACHE

    try:
        import yaml  # pyyaml — listed in core deps
        _USER_CONFIG_CACHE = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        ) or {}
    except ImportError:
        _USER_CONFIG_CACHE = {}  # graceful degradation without pyyaml
    except Exception:
        _USER_CONFIG_CACHE = {}

    return _USER_CONFIG_CACHE


def get_config() -> "YiCeNetConfig":
    """Return YiCeNetConfig with ~/.yicenet/config.yaml overrides applied.

    Load order: dataclass defaults → config.yaml → env vars (highest priority).
    Eval API settings from config.yaml are injected into os.environ so that
    existing callers reading env vars pick them up without code changes.
    """
    cfg = YiCeNetConfig()
    user = load_user_config()

    def _setf(attr: str, val) -> None:
        if val is not None:
            try:
                setattr(cfg, attr, float(val))
            except (TypeError, ValueError):
                pass

    # Inference overrides
    inf = user.get("inference", {})
    _setf("gumbel_tau_init", inf.get("gumbel_tau_init"))
    _setf("gumbel_tau_min",  inf.get("gumbel_tau_min"))

    # Flywheel / World Model overrides
    fw = user.get("flywheel", {})
    _setf("wm_slow_tau_days", fw.get("slow_tau_days"))
    _setf("wm_fast_tau_days", fw.get("fast_tau_days"))

    # Eval API: inject into env vars (only if env var not already set)
    ev = user.get("eval", {})
    for env_key, yaml_key in (
        ("EVAL_API_URL", "api_url"),
        ("EVAL_MODEL",   "model"),
        ("EVAL_API_KEY", "api_key"),
    ):
        val = ev.get(yaml_key, "")
        if val and not os.environ.get(env_key):
            os.environ[env_key] = str(val)

    return cfg


def get_display_config() -> dict:
    """Return display settings from ~/.yicenet/config.yaml as DisplayConfig.

    Returns a dict compatible with DisplayConfig TypedDict.
    unicode_symbols is absent when not set in config (→ auto-detect at render time).
    """
    user = load_user_config()
    disp = user.get("display", {})
    cfg: dict = {
        "mode": disp.get("mode", "compact"),
        "hexagram_chain": bool(disp.get("hexagram_chain", False)),
    }
    # Only include unicode_symbols if explicitly set; absent → auto-detect
    if "unicode_symbols" in disp:
        cfg["unicode_symbols"] = bool(disp["unicode_symbols"])
    return cfg


@dataclass
class YiCeNetConfig:
    # ── Encoder (Tiny Transformer) ──
    vocab_size: int = 8000
    hidden_dim: int = 256
    intermediate_dim: int = 1024
    num_heads: int = 4
    num_encoder_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

    # ── Hexagram / Trigram ──
    num_trigrams: int = 8
    num_hexagrams: int = 64
    num_lines: int = 6  # each hexagram = 6 lines (爻)

    # ── Action space ──
    num_actions: int = 50  # orchestration primitives

    # ── Gumbel-Softmax ──
    gumbel_tau_init: float = 1.0
    gumbel_tau_min: float = 0.1
    gumbel_tau_decay: float = 0.995

    # ── Value Network ──
    value_hidden: int = 128

    # ── Probe System ──
    probe_dim: int = 9  # 六探針合計維度
    """六探針合計維度：h密度(2) + logits形狀(1) + 卦象家族(3) + Q值差距(1) + 跳躍度(1) + 動作置信(1)"""

    # ── World Model v2 (Dual-Head + Power Law) ──
    wm_shared_dim: int = 128
    """世界模型共享層維度"""
    wm_hexagram_head_dim: int = 64
    """頭A：結果卦象分布（等於卦數）"""
    num_external_metrics: int = 3
    """頭B：外部向量維度（token消耗 + 續航長度 + 滿意度）"""
    wm_beta: float = 0.3
    """外部向量 loss 權重係數 β"""
    wm_slow_tau_days: float = 30.0
    """頭A 冪律衰減時間常數（天）"""
    wm_fast_tau_days: float = 3.0
    """頭B 冪律衰減時間常數（天）"""
    wm_alpha: float = 1.5
    """冪律衰減指數 α"""

    # ── External Metrics ──
    ext_continuation_weight: float = 0.4
    """續航針投影權重"""
    ext_correction_weight: float = 0.4
    """校正針投影權重"""
    ext_completion_weight: float = 0.4
    """完成針投影權重"""
    ext_projection_temperature: float = 0.5
    """投影 softmax 溫度（低=尖銳，高=平滑）"""

    # ── Training ──
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_epochs: int = 100
    clip_grad_norm: float = 1.0

    # ── RL ──
    ppo_clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ── Paths ──
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Hexagram line patterns (King Wen order, 64 hexagrams) ──
    # Each is a 6-bit integer; bit 5 = top line (上九/上六),
    # bit 0 = bottom line (初九/初六)
    hexagram_patterns: tuple = field(default_factory=lambda: tuple(
        _king_wen_hexagrams()
    ))

    def __post_init__(self):
        assert len(self.hexagram_patterns) == 64


def _king_wen_hexagrams() -> list[int]:
    """
    Return the 64 hexagram patterns in King Wen (文王) order.

    Each hexagram is encoded as a 6-bit integer:
      bit 5 (MSB) = top line (上爻)
      bit 0 (LSB) = bottom line (初爻)
    where 1 = solid/yang (阳), 0 = broken/yin (阴)

    Source: traditional I Ching sequence.
    """
    # fmt: off
    return [
        0b111111, 0b000000, 0b010001, 0b100010,  # 1-4: 乾 坤 屯 蒙
        0b010111, 0b111010, 0b000010, 0b010000,  # 5-8: 需 讼 师 比
        0b110111, 0b111011, 0b111000, 0b000111,  # 9-12: 小畜 履 泰 否
        0b101111, 0b111101, 0b001000, 0b000100,  # 13-16: 同人 大有 谦 豫
        0b011001, 0b100110, 0b100101, 0b101001,  # 17-20: 随 蛊 临 观
        0b100001, 0b011110, 0b100111, 0b111001,  # 21-24: 噬嗑 贲 剥 复
        0b111100, 0b001111, 0b000110, 0b011000,  # 25-28: 无妄 大畜 颐 大过
        0b010010, 0b101101, 0b001110, 0b011100,  # 29-32: 坎 离 咸 恒
        0b001100, 0b110000, 0b000011, 0b110011,  # 33-36: 遯 大壮 晋 明夷
        0b101011, 0b110101, 0b101110, 0b011101,  # 37-40: 家人 睽 蹇 解
        0b110010, 0b010011, 0b100011, 0b110001,  # 41-44: 损 益 夬 姤
        0b011110, 0b011011, 0b110110, 0b001001,  # 45-48: 萃 升 困 井
        0b011101, 0b101110, 0b101100, 0b001101,  # 49-52: 革 鼎 震 艮
        0b110100, 0b001011, 0b101010, 0b010101,  # 53-56: 渐 归妹 丰 旅
        0b011001, 0b100110, 0b110001, 0b100011,  # 57-60: 巽 兑 涣 节
        0b110010, 0b010011, 0b011111, 0b111110,  # 61-64: 中孚 小过 既济 未济
    ]
    # fmt: on
