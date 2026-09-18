from __future__ import annotations

import numpy as np


def seasonal_baseline_matrix(
    load: np.ndarray,
    indices: np.ndarray,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    """Return a causal seasonal-naive baseline for each target index and horizon.

    The configured seasonal period is preferred. If history is too short, the
    function falls back to a 24-step lag and then to the latest available value.
    It also supports a target index equal to ``len(load)`` for future forecasts.
    """
    values = np.asarray(load, dtype=float).reshape(-1)
    indices = np.asarray(indices, dtype=int)
    result = np.empty((len(indices), horizon), dtype=float)
    for i, target in enumerate(indices):
        for h in range(horizon):
            source = target + h - seasonal_period
            if source < 0 or source >= len(values):
                source = target + h - 24
            if source < 0 or source >= len(values):
                source = min(max(target - 1, 0), len(values) - 1)
            result[i, h] = values[source]
    return result
