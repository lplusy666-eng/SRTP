"""异常检测工具。

检测器说明（这点必须对使用者诚实）：
当前用的是 **稳健统计检测器** —— 在 STL 残差上做相对偏离度阈值 + MAD 稳健 z 分数。
它不依赖 GPU、可解释、可复现，在月度/季度这种样本量下比深度模型更稳。

原项目里的 VAE 检测器是留给小时级高频数据的（那边样本量够）。
接口设计成可替换：只要新检测器产出 (residual_pct, z_score) 两个数组，
这个工具不用改。所以这里不做"假装有 VAE"的事。
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ..analysis import data_quality_risks, stl_decompose, yoy_at
from ..contract import AnomalyEvent, AnomalyScanResult, Confidence
from ..series_utils import period_label
from .base import ToolContext, tool


class DetectArgs(BaseModel):
    dataset_id: str = Field(description="数据集 id。")
    column: str | None = Field(default=None, description="覆盖默认数值列。")
    threshold_high: float = Field(
        default=0.08,
        description="高等级异常阈值：相对偏离度绝对值超过它判为 high。月频数据默认 8%。",
    )
    threshold_medium: float = Field(
        default=0.05,
        description="中等级异常阈值。默认 5%。低于它的点不会返回。",
    )
    period: int | None = Field(default=None, description="季节周期，留空自动判定。")
    robust: bool = Field(default=True, description="STL 是否启用稳健迭代。")
    max_events: int = Field(default=15, description="最多返回多少个事件，按偏离度绝对值从大到小排。")


@tool(
    name="detect_anomaly",
    description=(
        "在 STL 残差上检测异常期次，返回按偏离度排序的异常事件列表。\n"
        "判定规则：偏离度 = (实际值 - 趋势与季节解释的期望值) / |期望值|，"
        "绝对值超过 threshold_high 判 high，超过 threshold_medium 判 medium。\n"
        "每个事件都带 data_quality_risks —— 例如“Q4 由累计值差分”“首尾期次端点不可靠”。"
        "这些风险标签必须原样进入最终结论，不能当成经济现象来解释。\n"
        "这个工具只回答“哪几期不正常”，不回答“为什么”。原因必须调 explain_anomaly 去查，"
        "不要凭常识直接下结论。"
    ),
    args_model=DetectArgs,
    returns="AnomalyScanResult：异常事件列表（含方向、等级、稳健 z、数据构造风险）",
    tags=("analysis", "anomaly"),
)
def detect_anomaly(ctx: ToolContext, args: DetectArgs) -> BaseModel:
    info = ctx.region.datasets.get(args.dataset_id)
    if info is None:
        raise KeyError(f"没有数据集 {args.dataset_id}。可用：{sorted(ctx.region.datasets)}")

    df = ctx.region.series(args.dataset_id, column=args.column)
    out = stl_decompose(df, info.frequency, period=args.period, robust=args.robust)
    risks = data_quality_risks(out.freq, out.periods, out.period)

    abs_dev = np.abs(out.residual_pct)
    idx = [i for i in np.argsort(-abs_dev) if abs_dev[i] >= args.threshold_medium][: args.max_events]

    events: list[AnomalyEvent] = []
    for i in idx:
        dev = float(out.residual_pct[i])
        level = Confidence.HIGH if abs(dev) >= args.threshold_high else Confidence.MEDIUM
        label = period_label(out.periods[i], out.freq)
        events.append(
            AnomalyEvent(
                event_id=f"{args.dataset_id}:{label}",
                dataset_id=args.dataset_id,
                metric=info.metric if args.column is None else args.column,
                unit=info.unit,
                period=label,
                observed=float(out.values[i]),
                expected=float(out.expected[i]),
                residual_pct=dev,
                yoy_pct=yoy_at(out, i),
                direction="above_expected“ if dev > 0 else ”below_expected",
                level=level,
                robust_z=float(out.z_scores[i]),
                endpoint_warning=bool(i in (0, out.n - 1)),
                data_quality_risks=risks.get(i, []),
            )
        )

    n_high = sum(1 for e in events if e.level is Confidence.HIGH)
    notes = [
        f"共扫描 {out.n} 期，检出 {len(events)} 个异常（其中 high {n_high} 个）。",
        f"检测器为稳健统计检测器（STL 残差 + 相对偏离度阈值），非深度模型；"
        f"样本量为 {out.n} 期时它比神经网络更可靠。",
    ]
    if not events:
        notes.append("在给定阈值下没有检出异常。可以适当下调 threshold_medium 再看，但要接受误报上升。")

    return AnomalyScanResult(
        dataset_id=args.dataset_id,
        metric=info.metric if args.column is None else args.column,
        detector="robust-stl-residual",
        threshold_high_pct=args.threshold_high,
        threshold_medium_pct=args.threshold_medium,
        n_events=len(events),
        events=events,
        notes=notes,
    )
