from __future__ import annotations

import os
from functools import lru_cache

from ..agents import PowerEconomyOrchestrator


@lru_cache(maxsize=4)
def get_orchestrator(config_path: str | None = None) -> PowerEconomyOrchestrator:
    path = config_path or os.getenv("POWER_ECON_CONFIG", "configs/demo.yaml")
    return PowerEconomyOrchestrator.from_config(path)


def clear_runtime_cache() -> None:
    get_orchestrator.cache_clear()
