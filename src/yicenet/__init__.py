"""
YiCeNet (易策网络) — I Ching inspired lightweight orchestration engine for Hermes.
~5.6M parameters (5,671,859), ~22 MB FP32, <3 ms inference.
"""

__version__ = "15.0.0"

# Public API
from .yicenet_engine import YiCeNetEngine, get_engine, predict
from .model import YiCeNet, count_parameters
from .config import YiCeNetConfig, yicenet_home, yicenet_data_dir, yicenet_checkpoint_dir
