"""输出类工具：出图 + 提交五段式结论。

submit_conclusion 是整个设计里最关键的一个工具。
它不是"让模型自由写报告"，而是**强制模型通过一次工具调用产出结构化结论**。
好处有三个：
1. schema 校验 —— 少写一段（比如没写反证）就调用不成功，模型必须补全；
2. 可追溯 —— 结论的每个字段都进了留痕，事后能审计模型是怎么想的；
3. 可评分 —— 评测集直接比对结构化字段，不需要解析自由文本。

这也是把"相关性写成因果"堵死的地方：causal_status 是枚举，模型填不了模糊表述。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..analysis import stl_decompose  # noqa: E402
from ..contract import Conclusion, CausalStatus, Evidence  # noqa: E402
from ..series_utils import parse_period, period_label  # noqa: E402
from .base import ToolContext, tool  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ChartArgs(BaseModel):
    dataset_id: str = Field(description="数据集 id。")
    column: str | None = Field(default=None, description="覆盖默认数值列。")
    start: str | None = Field(default=None, description="起始期次。")
    end: str | None = Field(default=None, description="结束期次。")
    annotate_anomalies: bool = Field(default=True, description="是否在图上标出偏离超过 8% 的异常点。")


class ChartResult(BaseModel):
    chart_path: str
    dataset_id: str
    metric: str
    n_points: int
    note: str


@tool(
    name="make_chart",
    description=(
        "为一条序列出图，包含 实际值 / 趋势 / 季节调整后期望值 / 残差偏离 四个面板，"
        "并在图上标出异常点。返回 PNG 路径。\n"
        "当结论需要给人类看趋势形态时调用它 —— 文字描述走势不如一张图直接。"
        "纯数值结论可以不调，以节省时间。"
    ),
    args_model=ChartArgs,
    returns="ChartResult：图片路径与说明",
    tags=("output", "chart"),
)
def make_chart(ctx: ToolContext, args: ChartArgs) -> BaseModel:
    info = ctx.region.datasets.get(args.dataset_id)
    if info is None:
        raise KeyError(f"没有数据集 {args.dataset_id}。可用：{sorted(ctx.region.datasets)}")

    df = ctx.region.series(args.dataset_id, column=args.column)
    if args.start:
        lo = parse_period(args.start)
        if lo is not None:
            df = df[df["period"] >= lo]
    if args.end:
        hi = parse_period(args.end)
        if hi is not None:
            df = df[df["period"] <= hi]
    if df.empty:
        raise ValueError("指定区间内没有数据，无法出图。")

    out = stl_decompose(df, info.frequency)
    labels = [period_label(p, out.freq) for p in out.periods]
    x = np.arange(len(labels))
    metric = info.metric if args.column is None else args.column

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(x, out.values, marker="o", ms=3, lw=1.4, label="实际值", color="#1f4e79")
    axes[0].plot(x, out.trend, lw=1.8, label="趋势", color="#c00000")
    axes[0].plot(x, out.expected, lw=1.2, ls="--", label="趋势+季节（期望值）", color="#7f7f7f")
    if args.annotate_anomalies:
        mask = np.abs(out.residual_pct) >= 0.08
        if mask.any():
            axes[0].scatter(x[mask], out.values[mask], s=70, facecolors="none",
                            edgecolors="#c00000", lw=1.6, zorder=5, label="异常点(|偏离|≥8%)")
    axes[0].set_ylabel(f"{metric}" + (f"（{info.unit}）" if info.unit else ""))
    axes[0].legend(fontsize=9, loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, out.residual_pct * 100, lw=1.2, color="#7030a0")
    axes[1].axhline(0, color="#7f7f7f", lw=0.8)
    for th in (8, -8):
        axes[1].axhline(th, color="#c00000", lw=0.8, ls=":")
    axes[1].set_ylabel("残差偏离 (%)")
    axes[1].grid(alpha=0.25)

    axes[2].bar(x, out.seasonal, color="#2e75b6", alpha=0.85)
    axes[2].set_ylabel("季节项")
    axes[2].grid(alpha=0.25)

    step = max(1, len(labels) // 18)
    axes[2].set_xticks(x[::step])
    axes[2].set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=45, ha="right", fontsize=8)
    fig.suptitle(
        f"{ctx.region.info.name} · {metric} · STL 分解"
        f"（趋势强度 {out.trend_strength:.2f} / 季节强度 {out.seasonal_strength:.2f}）",
        fontsize=12,
    )
    fig.tight_layout()

    path = ctx.chart_path(f"{args.dataset_id}_{datetime.now().strftime('%H%M%S')}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return ChartResult(
        chart_path=str(path),
        dataset_id=args.dataset_id,
        metric=metric,
        n_points=len(labels),
        note="图中红点为残差偏离绝对值超过 8% 的期次；季节项面板展示固定周期规律的幅度。",
    )


class EvidenceArg(BaseModel):
    claim: str = Field(description="一条可核查的陈述，写清楚数值和期次。")
    tool: str = Field(description="这条证据来自哪个工具，例如 decompose_series / get_calendar_context。")
    source: str | None = Field(default=None, description="数据来源或文档出处。")
    value: str | None = Field(default=None, description="关键数值，例如 '-9.13%' 或 '春节 7 天'。")
    period: str | None = Field(default=None, description="对应的期次。")


class SubmitConclusionArgs(BaseModel):
    conclusion: str = Field(
        description="结论正文。必须直接回答用户的问题，并明确说明这个偏离“能否”归因到经济。"
        "禁止使用“证明了”“说明经济走弱”这类确定性表述。"
    )
    causal_status: CausalStatus = Field(
        description=(
            "整条结论的因果强度。只能填：observed（只是观测到现象）、"
            "correlation（相关）、candidate_cause（候选原因，有机制先验待验证）、"
            "data_quality_risk（异常来自数据构造而非真实波动）、"
            "established（已证因果，本项目当前不应使用）。"
        )
    )
    confidence: float = Field(description="置信度 0—1。数据有缺口、上下文缺失、样本量不足时都必须调低。")
    evidence: list[EvidenceArg] = Field(description="支持结论的证据清单，每条都要能追溯到工具和数据。")
    counter_evidence: list[str] = Field(
        description="反证或削弱结论的事实。**必须至少写一条**，例如“缺少分行业用电数据，无法排除产业结构变化”。"
    )
    uncertainty: list[str] = Field(
        description="不确定性与限制。**必须至少写一条**，例如“天气为单点代理变量”。"
    )


@tool(
    name="submit_conclusion",
    description=(
        "提交最终结论。**这是回答用户的唯一方式** —— 分析做完后必须调用它，"
        "不能直接把分析过程当结论返回。\n"
        "五段式要求：结论 / 证据 / 反证 / 不确定性，缺任何一段都会被 schema 拒绝。\n"
        "写作纪律：\n"
        "1. causal_status 只能填枚举值，不要把相关性写成因果；\n"
        "2. 置信度要诚实 —— 数据有缺口、天气是代理变量、样本不足都要相应下调；\n"
        "3. counter_evidence 至少一条，写清楚什么事实会推翻你的结论；\n"
        "4. uncertainty 至少一条，写清楚还缺什么数据；\n"
        "5. 证据里的每个数字都必须来自你实际调用过的工具，禁止凭常识补数。\n"
        "调用这个工具后流程即结束。"
    ),
    args_model=SubmitConclusionArgs,
    returns="Conclusion：通过校验的五段式结论",
    tags=("output", "conclusion"),
)
def submit_conclusion(ctx: ToolContext, args: SubmitConclusionArgs) -> BaseModel:
    calls = ctx.cache_get("tool_calls") or []
    return Conclusion(
        region=ctx.region.info.name,
        question=ctx.cache_get("question") or "",
        conclusion=args.conclusion,
        causal_status=args.causal_status,
        evidence=[
            Evidence(
                claim=e.claim,
                tool=e.tool,
                source=e.source,
                value=e.value,
                period=e.period,
            )
            for e in args.evidence
        ],
        counter_evidence=args.counter_evidence,
        confidence=args.confidence,
        uncertainty=args.uncertainty,
        caveats=list(ctx.region.info.caveats),
        tool_calls=list(calls),
    )
