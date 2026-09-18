"""电力看经济智能体 —— 工具层与智能体内核。

分层：
- contract    契约（输入包 / 工具 I/O / 五段式输出）
- ingest      通用数据摄入（换地区不改代码）
- series_utils / analysis   公共时序与分解内核
- tools       工具层（LLM 可调度的接口面）
- agent       智能体循环（plan → act → observe → replan）
"""

from .contract import (
    AnomalyEvent,
    CausalStatus,
    Conclusion,
    Confidence,
    DecompositionResult,
    Evidence,
    Frequency,
    RegionInventory,
    SeriesSummary,
)
from .ingest import RegionPackError, build_inventory, load_region
from .runtime import make_context

__version__ = "0.1.0"

__all__ = [
    "AnomalyEvent",
    "CausalStatus",
    "Conclusion",
    "Confidence",
    "DecompositionResult",
    "Evidence",
    "Frequency",
    "RegionInventory",
    "SeriesSummary",
    "RegionPackError",
    "build_inventory",
    "load_region",
    "make_context",
    "__version__",
]
