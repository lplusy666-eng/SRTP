"""分析内核：STL 分解与稳健异常打分。

分解和异常检测必须共用同一份实现 —— 否则会出现"分解工具说残差 3%，
检测工具说残差 8%"这种自相矛盾的结果，而模型会同时看到两者。

所以这里放唯一的 stl_decompose，上层两个工具都调它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .contract import Frequency
from .series_utils import mad, robust_z

DEFAULT_PERIOD: dict[Frequency, int] = {
    Frequency.HOURLY: 24,
    Frequency.DAILY: 7,
    Frequency.WEEKLY: 52,
    Frequency.MONTHLY: 12,
    Frequency.QUARTERLY: 4,
}


@dataclass
class StlOutput:
    periods: list[pd.Timestamp]
    freq: Frequency
    values: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    expected: np.ndarray
    residual_pct: np.ndarray
    z_scores: np.ndarray
    use_log: bool
    period: int
    trend_strength: float
    seasonal_strength: float
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def endpoint_warning(self) -> bool:
        return self.n < 3 * self.period


def resolve_period(freq: Frequency, period: int | None) -> int:
    if period is not None:
        return int(period)
    resolved = DEFAULT_PERIOD.get(freq)
    if resolved is None:
        raise ValueError(
            f"频率 {freq.value} 无法自动确定季节周期，请在调用时显式传入 period。"
        )
    return resolved


def stl_decompose(
    df: pd.DataFrame,
    freq: Frequency,
    *,
    period: int | None = None,
    robust: bool = True,
) -> StlOutput:
    """对标准化序列做 STL 分解。df 需要包含 period / value 两列。"""
    resolved = resolve_period(freq, period)
    ordered = df.sort_values("period").reset_index(drop=True)
    periods = list(ordered["period"])
    values = ordered["value"].to_numpy(dtype=float)
    n = len(values)

    notes: list[str] = []
    if n < 2 * resolved:
        raise ValueError(
            f"数据只有 {n} 期，不足 2 个完整季节周期（period={resolved}，至少需要 {2 * resolved} 期）。"
            f"样本量不足时 STL 不可靠，请改用同比等更朴素的方法。"
        )

    use_log = bool(np.all(values > 0))
    target = np.log(values) if use_log else values
    if use_log:
        notes.append("已取对数后分解，残差百分比可近似理解为相对偏离度。")
    else:
        notes.append("序列含非正值，未取对数，残差为绝对偏离。")

    seasonal_window = 13 if resolved >= 12 else max(7, resolved + 1)
    if seasonal_window % 2 == 0:
        seasonal_window += 1

    fit = STL(target, period=resolved, seasonal=seasonal_window, robust=robust).fit()
    trend = _fill(np.asarray(fit.trend, dtype=float))
    seasonal = _fill(np.asarray(fit.seasonal, dtype=float))
    residual = _fill(np.asarray(fit.resid, dtype=float))

    if use_log:
        trend_display = np.exp(trend)
        expected = np.exp(trend + seasonal)
    else:
        trend_display = trend
        expected = trend + seasonal

    with np.errstate(divide="ignore", invalid="ignore"):
        residual_pct = np.where(
            np.abs(expected) > 1e-12, (values - expected) / np.abs(expected), 0.0
        )
    residual_pct = np.nan_to_num(residual_pct, nan=0.0, posinf=0.0, neginf=0.0)

    ts = _strength(residual, trend)
    ss = _strength(residual, seasonal)

    return StlOutput(
        periods=periods,
        freq=freq,
        values=values,
        trend=trend_display,
        seasonal=seasonal,
        residual=residual,
        expected=expected,
        residual_pct=residual_pct,
        z_scores=robust_z(residual_pct),
        use_log=use_log,
        period=resolved,
        trend_strength=ts,
        seasonal_strength=ss,
        notes=notes,
    )


def _fill(arr: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(arr)
    if bad.any() and (~bad).any():
        idx = np.arange(len(arr))
        arr = arr.copy()
        arr[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _strength(residual: np.ndarray, component: np.ndarray) -> float:
    denom = float(np.var(residual + component))
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - float(np.var(residual)) / denom))


def yoy_at(out: StlOutput, idx: int) -> float | None:
    """第 idx 期的同比（与去年同期比）。

    这个值和残差偏离是两回事，必须同时给出，否则很容易混淆：
    - 同比大（如 -12%）可能完全由季节摆动和春节错位造成；
    - 残差偏离小（如 +0.03%）说明趋势和季节已经解释了这一期，没有额外的异常。
    只看同比会误判，只看残差会答不上用户"同比为什么降"的问题。
    """
    lag = {
        Frequency.MONTHLY: 12,
        Frequency.QUARTERLY: 4,
        Frequency.ANNUAL: 1,
        Frequency.WEEKLY: 52,
        Frequency.DAILY: 365,
        Frequency.HOURLY: 24 * 365,
    }.get(out.freq)
    if lag is None:
        return None
    j = idx - lag
    if j < 0:
        return None
    base = float(out.values[j])
    if abs(base) < 1e-12:
        return None
    return float((out.values[idx] - base) / abs(base))


def data_quality_risks(freq: Frequency, periods: list[pd.Timestamp], period: int) -> dict[int, list[str]]:
    """逐点标注数据构造风险。返回 {索引: [风险标签]}。

    这些标签必须在异常事件上原样出现 —— 否则模型会把"Q4 累计差分"
    造成的台阶当成经济冲击。
    """
    risks: dict[int, list[str]] = {}
    n = len(periods)

    for i in range(n):
        tags: list[str] = []
        p = periods[i]
        if i < 1 or i >= n - 1:
            tags.append("首尾期次，STL 端点估计不可靠")
        if freq is Frequency.QUARTERLY and getattr(p, "quarter", None) == 4:
            tags.append("Q4 由全年累计值差分得到，可能吸收年度核算修订")
        if freq is Frequency.MONTHLY and p.month == 12:
            tags.append("12 月承接全年差额，累计口径差分可能失真")
        if i < period:
            tags.append(f"不足一个完整季节周期（前 {period} 期）")
        if tags:
            risks[i] = tags
    return risks


def jump_scale(values: np.ndarray) -> float:
    """相邻差分的中位数绝对偏差，用来判断"这一跳是否异常"。"""
    if len(values) < 3:
        return 0.0
    return mad(np.diff(values))


def top_k_indices(arr: np.ndarray, k: int) -> list[int]:
    if arr.size == 0:
        return []
    k = min(k, arr.size)
    return list(np.argsort(-np.abs(arr))[:k])


def describe_distribution(arr: np.ndarray) -> dict[str, Any]:
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "mad": float(mad(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }
