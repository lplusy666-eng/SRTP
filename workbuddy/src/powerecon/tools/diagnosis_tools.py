"""诊断工具：给一个异常期次找候选原因。

这个工具是"排除法"的落点，顺序是刻意的：

1. 先看**数据构造风险** —— 如果异常本身就是累计差分或端点造成的，后面都不用查了。
2. 再看**日历** —— 春节错位是月度电力同比最大的伪信号来源，必须先排。
3. 再看**天气** —— 夏冬两季的负荷波动基本由气温解释。
4. 最后才剩下**经济候选** —— 而且只给低置信度 + 需要什么证据，绝不下定论。

四类原因的 causal_status 是分开的：日历和天气是 candidate_cause（机制明确、可验证），
数据构造是 data_quality_risk，经济是 candidate_cause 但置信度最低。
这样模型在写结论时无法把"可能是经济"写成"就是经济"。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ..analysis import data_quality_risks, stl_decompose, yoy_at
from ..contract import (
    AnomalyEvent,
    CalendarContext,
    CandidateCause,
    CausalStatus,
    Confidence,
    KnowledgeTriple,
    RetrievedDoc,
    WeatherContext,
)
from ..series_utils import parse_period, period_label, shift_period
from .base import ToolContext, tool
from .context_tools import CalendarArgs, WeatherArgs, get_calendar_context, get_weather_context
from .knowledge_tools import KgQueryArgs, SearchArgs, query_knowledge_graph, search_knowledge


class ExplainArgs(BaseModel):
    dataset_id: str = Field(description="数据集 id，应与 detect_anomaly 用的一致。")
    period: str = Field(description="要解释的异常期次，例如 '2025-01' 或 '2024Q4'。")
    column: str | None = Field(default=None, description="覆盖默认数值列。")


class AnomalyExplanation(BaseModel):
    event: AnomalyEvent
    calendar: CalendarContext | None = None
    calendar_last_year: CalendarContext | None = None
    weather: WeatherContext | None = None
    weather_last_year: WeatherContext | None = None
    graph_triples: list[KnowledgeTriple] = Field(default_factory=list)
    knowledge_hits: list[RetrievedDoc] = Field(default_factory=list)
    rules_fired: list[str] = Field(default_factory=list)
    exclusion_summary: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _shift_year_label(label: str, freq) -> str:
    ts = parse_period(label)
    if ts is None:
        return label
    return period_label(shift_period(ts, freq, -12 if freq.value == "monthly" else -4), freq)


@tool(
    name="explain_anomaly",
    description=(
        "给一个已经检出的异常期次找候选原因。会自动做四步排除："
        "数据构造风险 → 日历（春节错位）→ 天气（气温负荷）→ 剩余的经济候选。\n"
        "同时会去知识图谱和知识文档里找机制先验作为佐证。\n"
        "**返回值里每个候选原因都带 causal_status 和 required_evidence。** "
        "写结论时必须原样保留这两项：causal_status 说明这只是候选，"
        "required_evidence 说明还需要什么数据才能确认。\n"
        "如果某个上下文返回 available=false（例如没有天气数据），"
        "说明这一类因素**没有被排除**，必须在不确定性里写明，不能默认它不存在。"
    ),
    args_model=ExplainArgs,
    returns="AnomalyExplanation：异常事件 + 候选原因 + 日历/天气对比 + 知识佐证 + 已触发的规则",
    tags=("analysis", "diagnosis"),
)
def explain_anomaly(ctx: ToolContext, args: ExplainArgs) -> BaseModel:
    info = ctx.region.datasets.get(args.dataset_id)
    if info is None:
        raise KeyError(f"没有数据集 {args.dataset_id}。可用：{sorted(ctx.region.datasets)}")

    df = ctx.region.series(args.dataset_id, column=args.column)
    out = stl_decompose(df, info.frequency, robust=True)
    freq = out.freq

    target = parse_period(args.period)
    if target is None:
        raise ValueError(f"期次 '{args.period}' 无法解析，请使用 '2025-01' 或 '2024Q4' 这类格式。")

    idx = next((i for i, p in enumerate(out.periods) if p == target), None)
    if idx is None:
        raise ValueError(
            f"{args.dataset_id} 里没有期次 {args.period}。"
            f"可用范围 {period_label(out.periods[0], freq)} 至 {period_label(out.periods[-1], freq)}。"
        )

    label = period_label(out.periods[idx], freq)
    last_year_label = _shift_year_label(label, freq)
    risks = data_quality_risks(freq, out.periods, out.period).get(idx, [])

    dev = float(out.residual_pct[idx])
    event = AnomalyEvent(
        event_id=f"{args.dataset_id}:{label}",
        dataset_id=args.dataset_id,
        metric=info.metric if args.column is None else args.column,
        unit=info.unit,
        period=label,
        observed=float(out.values[idx]),
        expected=float(out.expected[idx]),
        residual_pct=dev,
        yoy_pct=yoy_at(out, idx),
        direction="above_expected" if dev > 0 else "below_expected",
        level=Confidence.HIGH if abs(dev) >= 0.08 else Confidence.MEDIUM,
        robust_z=float(out.z_scores[idx]),
        endpoint_warning=bool(idx in (0, out.n - 1)),
        data_quality_risks=risks,
    )

    month = out.periods[idx].month
    quarter = out.periods[idx].quarter
    year = out.periods[idx].year

    # ---- 1. 日历（含去年同期对比，专门抓春节错位）
    this_cal = get_calendar_context(ctx, CalendarArgs(start=label, end=label))
    prev_cal = None
    if last_year_label != label:
        try:
            prev_cal = get_calendar_context(ctx, CalendarArgs(start=last_year_label, end=last_year_label))
        except Exception:
            prev_cal = None

    # ---- 2. 天气（同样带去年同期）
    this_wea = get_weather_context(ctx, WeatherArgs(start=label, end=label))
    prev_wea = None
    if last_year_label != label:
        try:
            prev_wea = get_weather_context(ctx, WeatherArgs(start=last_year_label, end=last_year_label))
        except Exception:
            prev_wea = None

    # ---- 3. 知识
    graph = query_knowledge_graph(
        ctx,
        KgQueryArgs(subject_keyword="温度", relation="AFFECTS", limit=10),
    )
    graph2 = query_knowledge_graph(ctx, KgQueryArgs(relation="PROXIES_FOR", limit=10))
    triples = list(graph.triples) + [t for t in graph2.triples if t not in graph.triples]

    direction_word = "下降" if dev < 0 else "上升"
    hits: list[RetrievedDoc] = []
    for q in (
        f"{year}年{month}月 用电量 同比 {direction_word} 原因",
        "春节错位 用电同比 影响",
        "累计值差分 季度数据 口径",
    ):
        try:
            res = search_knowledge(ctx, SearchArgs(query=q, top_k=2))
            for h in res.hits:
                if h.doc_id not in {x.doc_id for x in hits}:
                    hits.append(h)
        except Exception:
            continue

    # ---- 4. 规则：按排除顺序生成候选原因
    causes: list[CandidateCause] = []
    rules: list[str] = []
    exclusions: list[str] = []

    if risks:
        causes.append(
            CandidateCause(
                cause_id="data_construction",
                cause_name="数据构造风险（非经济因素）",
                relation="DATA_QUALITY_RISK",
                causal_status=CausalStatus.DATA_QUALITY_RISK,
                confidence=Confidence.HIGH,
                evidence="；".join(risks),
                required_evidence=["回到原始统计口径核对，确认该期数值未经累计差分或年度修订"],
                source="data_quality_risks",
            )
        )
        rules.append("data_construction_risk")
        exclusions.append(f"该期存在数据构造风险（{'；'.join(risks)}），此异常可能不是真实波动。")

    spring_this = this_cal.spring_festival_days or 0
    spring_prev = (prev_cal.spring_festival_days or 0) if prev_cal else 0
    if spring_this or spring_prev:
        shifted = spring_this != spring_prev
        causes.append(
            CandidateCause(
                cause_id="spring_festival_shift",
                cause_name="春节错位（日历效应）",
                relation="CANDIDATE_CAUSE",
                causal_status=CausalStatus.CANDIDATE_CAUSE,
                confidence=Confidence.HIGH if shifted else Confidence.MEDIUM,
                evidence=(
                    f"{label} 春节天数 {spring_this} 天，去年同期（{last_year_label}）{spring_prev} 天。"
                    + ("两期春节落月不同，同比读数被日历错位严重污染。" if shifted else "")
                ),
                counter_evidence=(
                    None
                    if shifted
                    else "今年与去年同期春节落月相同，单靠春节错位解释不了本次偏离。"
                ),
                required_evidence=["合并 1—2 月重新计算同比；若合并后偏离消失，则确认是日历效应"],
                source="get_calendar_context",
            )
        )
        rules.append("spring_festival_shift")
        if shifted:
            exclusions.append(
                f"春节在 {label} 与 {last_year_label} 的落月不同，同比不可比，"
                f"这是本次偏离的首要候选解释。"
            )

    if (this_cal.holiday_break_days or 0) > 0 and not spring_this:
        causes.append(
            CandidateCause(
                cause_id="holiday_effect",
                cause_name="法定节假日停工（日历效应）",
                relation="CANDIDATE_CAUSE",
                causal_status=CausalStatus.CANDIDATE_CAUSE,
                confidence=Confidence.MEDIUM,
                evidence=f"本期法定放假 {this_cal.holiday_break_days} 天，调休上班 {this_cal.makeup_workdays} 天。",
                required_evidence=["用工作日数归一化后重算日均用电，看偏离是否消失"],
                source="get_calendar_context",
            )
        )
        rules.append("holiday_effect")

    if this_wea.available:
        row = this_wea.monthly[0] if this_wea.monthly else {}
        temp = row.get("temp_mean_c")
        hdd, cdd = row.get("heating_degree_days"), row.get("cooling_degree_days")
        extreme = (hdd or 0) > 200 or (cdd or 0) > 200
        if month in (6, 7, 8, 12, 1, 2):
            causes.append(
                CandidateCause(
                    cause_id="weather_load",
                    cause_name="气温负荷（采暖/制冷）",
                    relation="CANDIDATE_CAUSE",
                    causal_status=CausalStatus.CANDIDATE_CAUSE,
                    confidence=Confidence.HIGH if extreme else Confidence.MEDIUM,
                    evidence=(
                        f"本期月均气温 {temp}℃，采暖度日 {hdd}，制冷度日 {cdd}。"
                        + ("度日值处于高位，气温对负荷的贡献很大。" if extreme else "")
                    ),
                    counter_evidence=None if extreme else "度日值不算极端，气温可能只解释了部分偏离。",
                    required_evidence=[
                        "用历史气温—负荷回归系数估算气温应贡献的负荷量，从总偏离中扣除"
                    ],
                    source="get_weather_context",
                )
            )
            rules.append("weather_load")
        exclusions.append(
            f"天气为单点代理变量（{this_wea.proxy_note}），气温贡献只能估算不能精确扣除。"
        )
    else:
        exclusions.append("该地区无天气数据，气温因素**未被排除**。")

    negligible = abs(dev) < 0.05
    if negligible:
        rules.append("residual_negligible")
        exclusions.append(
            f"{label} 在剥离趋势与季节后相对偏离仅 {dev * 100:+.2f}%，"
            f"说明趋势和季节已完整解释该期用电量，**不存在需要额外解释的异常**。"
            f"该期的同比变化应由季节摆动与日历因素解释，不承载经济信息。"
        )

    causes.append(
        CandidateCause(
            cause_id="economic_activity",
            cause_name="经济活动变化（真实经济信号）",
            relation="CANDIDATE_CAUSE",
            causal_status=CausalStatus.CANDIDATE_CAUSE,
            confidence=Confidence.LOW,
            evidence=(
                f"在剥离趋势与季节后，{label} 实际值 {out.values[idx]:.4g}，"
                f"期望值 {out.expected[idx]:.4g}，相对偏离 {dev * 100:+.2f}%。"
                + (
                    "残差绝对值接近零，这一期没有留给经济因素去解释的偏离。"
                    if negligible
                    else "这部分偏离尚未被日历和天气解释。"
                )
            ),
            counter_evidence=(
                "目前没有任何直接经济指标（工业增加值、PMI、分行业用电）佐证，"
                "仅凭电力偏离不能判定经济走弱。"
            ),
            required_evidence=[
                "分行业用电量同比（制造业/服务业分化情况）",
                "月度工业增加值或 PMI",
                "剔除同期新增大用户与检修计划的影响",
                "与相邻省份同期对比，排除区域性共同因素",
            ],
            source="decompose_series",
        )
    )
    rules.append("economic_activity_unverified")

    causes.append(
        CandidateCause(
            cause_id="structural_change",
            cause_name="产业结构或能效变化（长期因素）",
            relation="CANDIDATE_CAUSE",
            causal_status=CausalStatus.CANDIDATE_CAUSE,
            confidence=Confidence.LOW,
            evidence="趋势项长期变化可能来自产业结构调整或能效提升，与短期波动性质不同。",
            required_evidence=["对比分行业用电结构变化；检查单位 GDP 电耗的长期趋势"],
            source="decompose_series",
        )
    )

    event.candidate_causes = causes

    notes = [
        f"共生成 {len(causes)} 个候选原因，触发规则：{rules}。",
        "候选原因按排除顺序排列：数据构造 → 日历 → 天气 → 经济。"
        "排在前面的如果成立，后面的解释力就应下调。",
        "经济类候选的置信度被刻意压到 low —— 电力偏离只是线索，不是经济结论。",
    ]
    if not hits:
        notes.append("知识库未检索到相关片段，候选原因主要来自规则与上下文，缺少文档佐证。")

    return AnomalyExplanation(
        event=event,
        calendar=this_cal,
        calendar_last_year=prev_cal,
        weather=this_wea,
        weather_last_year=prev_wea,
        graph_triples=triples[:10],
        knowledge_hits=hits[:6],
        rules_fired=rules,
        exclusion_summary=exclusions,
        notes=notes,
    )
