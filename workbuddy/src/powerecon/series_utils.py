"""时间序列的公共小工具：期次网格、期次偏移、稳健统计。

单独抽出来是因为分解、异常检测、质量检查三处都要用同一套期次语义，
各写一份必然出现"月频用 30 天近似、季频用 3 个月近似"这类漂移。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contract import Frequency


def parse_period(value: Any) -> pd.Timestamp | None:
    """把各种写法的期次解析成 Timestamp。

    支持：2021-01 / 2021Q1 / 2021-01-01 / 2021 / 2021-01-01 08:00。
    无法解析返回 None（而不是抛异常），因为源数据里混着空值和非标准写法是常态。
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "nat", "null"):
        return None

    m = re.fullmatch(r"(\d{4})\s*[-_/]?\s*[Qq]\s*([1-4])", text)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=(int(m.group(2)) - 1) * 3 + 1, day=1)

    m = re.fullmatch(r"(\d{4})\s*[-_/]\s*([01]?\d)", text)
    if m and 1 <= int(m.group(2)) <= 12:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)

    m = re.fullmatch(r"(\d{4})(0[1-9]|1[0-2])", text)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)

    m = re.fullmatch(r"(\d{4})", text)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=1, day=1)

    try:
        return pd.Timestamp(text)
    except Exception:
        return None


parse_label = parse_period

_PANDAS_FREQ = {
    Frequency.HOURLY: "h",
    Frequency.DAILY: "D",
    Frequency.WEEKLY: "W-MON",
    Frequency.MONTHLY: "MS",
    Frequency.QUARTERLY: "QS",
    Frequency.ANNUAL: "YS",
}

_OFFSET = {
    Frequency.HOURLY: ("hours", 1),
    Frequency.DAILY: ("days", 1),
    Frequency.WEEKLY: ("weeks", 1),
    Frequency.MONTHLY: ("months", 1),
    Frequency.QUARTERLY: ("months", 3),
    Frequency.ANNUAL: ("years", 1),
}


def pandas_freq(freq: Frequency) -> str | None:
    return _PANDAS_FREQ.get(freq)


def shift_period(ts: pd.Timestamp, freq: Frequency, n: int) -> pd.Timestamp:
    """按期次语义偏移。n 为负表示往前。"""
    unit = _OFFSET.get(freq)
    if unit is None:
        return ts + pd.Timedelta(days=30 * n)
    key, base = unit
    return ts + pd.DateOffset(**{key: base * n})


def expected_grid(start: pd.Timestamp, end: pd.Timestamp, freq: Frequency) -> list[pd.Timestamp]:
    """期望的完整期次网格，用来发现缺期。"""
    rule = pandas_freq(freq)
    if rule is None:
        return []
    return list(pd.date_range(start=start, end=end, freq=rule))


def period_label(ts: pd.Timestamp, freq: Frequency) -> str:
    if freq is Frequency.QUARTERLY:
        return f"{ts.year}Q{ts.quarter}"
    if freq is Frequency.ANNUAL:
        return f"{ts.year}"
    if freq is Frequency.MONTHLY:
        return f"{ts.year:04d}-{ts.month:02d}"
    if freq is Frequency.HOURLY:
        return ts.strftime("%Y-%m-%d %H:00")
    return ts.strftime("%Y-%m-%d")


def mad(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 0.0
    median = np.median(arr)
    return float(np.median(np.abs(arr - median)))


def robust_z(values: Iterable[float]) -> np.ndarray:
    """基于中位数绝对偏差的稳健 z 分数。对极端值不敏感，适合异常检测。"""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    median = np.median(arr)
    scale = mad(arr)
    if scale <= 1e-12:
        scale = float(np.std(arr)) or 1e-12
    return (arr - median) / (1.4826 * scale)


def safe_pct_change(new: float, old: float) -> float | None:
    if old is None or new is None:
        return None
    if abs(old) < 1e-12:
        return None
    return (new - old) / abs(old)


def trend_strength(residual: np.ndarray, trend: np.ndarray) -> float:
    """STL 趋势强度：max(0, 1 - Var(resid) / Var(resid + trend))。"""
    denom = float(np.var(residual + trend))
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - float(np.var(residual)) / denom))


def seasonal_strength(residual: np.ndarray, seasonal: np.ndarray) -> float:
    denom = float(np.var(residual + seasonal))
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - float(np.var(residual)) / denom))
