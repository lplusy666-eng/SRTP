from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_raw_data(df: pd.DataFrame, expected_frequency: str = "1h") -> dict[str, Any]:
    required = ["timestamp", "load_mw"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"原始数据缺少必需列: {missing_required}")
    ts = pd.to_datetime(df["timestamp"])
    dup_count = int(ts.duplicated().sum())
    negative_load = int((pd.to_numeric(df["load_mw"], errors="coerce") < 0).sum())
    null_rates = {c: float(df[c].isna().mean()) for c in df.columns}
    diffs = ts.sort_values().diff().dropna()
    expected = pd.Timedelta(expected_frequency)
    irregular = int((diffs != expected).sum()) if len(diffs) else 0
    numeric = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0
    status = "pass"
    warnings: list[str] = []
    if dup_count:
        status = "warn"
        warnings.append(f"有 {dup_count} 个重复时间戳")
    if irregular:
        status = "warn"
        warnings.append(f"有 {irregular} 个非 {expected_frequency} 间隔")
    if negative_load:
        status = "fail"
        warnings.append(f"有 {negative_load} 个负负荷值")
    if inf_count:
        status = "fail"
        warnings.append(f"有 {inf_count} 个无穷值")
    return {
        "status": status,
        "rows": int(len(df)),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "duplicates": dup_count,
        "irregular_intervals": irregular,
        "negative_load": negative_load,
        "infinite_values": inf_count,
        "null_rates": null_rates,
        "warnings": warnings,
    }
