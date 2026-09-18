"""分解类工具：STL 把序列拆成趋势 / 季节 / 残差。

这是"剥离非经济扰动"的第一刀。残差是趋势和季节都解释不了的部分，
也就是后续异常检测与归因的输入。

实现全部委托给 analysis.stl_decompose，保证与 detect_anomaly 用的是同一份算法。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..analysis import stl_decompose
from ..contract import DecompositionPoint, DecompositionResult
from ..series_utils import period_label, safe_pct_change
from .base import ToolContext, tool


class DecomposeArgs(BaseModel):
    dataset_id: str = Field(description="数据集 id。")
    column: str | None = Field(default=None, description="覆盖默认数值列。")
    period: int | None = Field(
        default=None,
        description="季节周期（多少个观测点构成一个周期）。留空则按月频12 / 季频4 / 日频7 / 小时频24 自动选择。",
    )
    robust: bool = Field(default=True, description="启用稳健迭代，降低异常点对趋势和季节项的拉扯。")
    include_points: bool = Field(default=False, description="是否返回逐期明细。默认只给强度与趋势摘要，以节省上下文。")


@tool(
    name="decompose_series",
    description=(
        "用 STL 把一条序列分解为 趋势(trend) / 季节(seasonal) / 残差(residual)。"
        "趋势反映长期景气走向；季节反映固定周期规律（夏天空调负荷、春节停工）；"
        "残差是这两者都解释不了的部分，才是有可能承载经济含义或突发事件的信号。\n"
        "返回的 trend_strength 和 seasonal_strength 说明序列被谁主导："
        "季节强度高（>0.7）意味着同比读数会大幅受季节摆动影响，判断前必须先做季节调整。\n"
        "重要边界：残差大不等于“出问题了”，只等于“趋势和季节解释不了”。"
        "要查原因必须接着调 explain_anomaly，不要自己编原因。"
    ),
    args_model=DecomposeArgs,
    returns="DecompositionResult：趋势/季节强度、趋势起止与变化率、逐期残差",
    tags=("analysis", "decompose"),
)
def decompose_series(ctx: ToolContext, args: DecomposeArgs) -> BaseModel:
    info = ctx.region.datasets.get(args.dataset_id)
    if info is None:
        raise KeyError(f"没有数据集 {args.dataset_id}。可用：{sorted(ctx.region.datasets)}")

    df = ctx.region.series(args.dataset_id, column=args.column)
    out = stl_decompose(df, info.frequency, period=args.period, robust=args.robust)

    points = [
        DecompositionPoint(
            period=period_label(p, out.freq),
            observed=float(v),
            trend=float(t),
            seasonal=float(s),
            residual=float(r),
            residual_pct=float(rp),
        )
        for p, v, t, s, r, rp in zip(
            out.periods, out.values, out.trend, out.seasonal, out.residual, out.residual_pct
        )
    ]
    if not args.include_points:
        keep = min(13, len(points))
        points = points[-keep:]

    notes = list(out.notes)
    notes.append(f"趋势强度 {out.trend_strength:.3f}，季节强度 {out.seasonal_strength:.3f}。")
    if out.seasonal_strength > 0.7:
        notes.append("季节主导明显，单月同比会严重受季节摆动影响，应优先看同比或先做季节调整。")
    if out.trend_strength > 0.7:
        notes.append("趋势主导明显，长期走向比短期波动更能说明问题。")
    if out.endpoint_warning:
        notes.append("样本量不足 3 个完整季节周期，首尾期次的趋势估计不可靠（endpoint_warning=true）。")

    return DecompositionResult(
        dataset_id=args.dataset_id,
        metric=info.metric if args.column is None else args.column,
        unit=info.unit,
        method="STL(log)“ if out.use_log else ”STL",
        period=out.period,
        robust=args.robust,
        trend_strength=float(out.trend_strength),
        seasonal_strength=float(out.seasonal_strength),
        residual_std=float(out.residual.std()),
        residual_mad=float(_mad(out.residual)),
        trend_start=float(out.trend[0]),
        trend_end=float(out.trend[-1]),
        trend_change_pct=safe_pct_change(float(out.trend[-1]), float(out.trend[0])) or 0.0,
        endpoint_warning=bool(out.endpoint_warning),
        n_points=int(out.n),
        points=points,
        notes=notes,
    )


def _mad(arr) -> float:
    import numpy as np

    return float(np.median(np.abs(arr - np.median(arr))))
