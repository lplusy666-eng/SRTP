"""规划器：决定"下一步调哪个工具"。

这里放三个实现，对应三种使用场景：

- **RulePlanner** —— 不依赖任何大模型，按一条写死的但合理的工具链走完。
  它的作用是：① 在没接 LLM 之前把整条链路验证通；② LLM 不可用时兜底；
  ③ 作为评测的基线，用来判断 LLM 到底比"照章办事"强在哪。
  它的结论由模板合成，明确标注为规则版，不会假装自己是智能分析。

- **LLMPlanner** —— 真正的智能体规划器。**这里不实现任何大模型 API 调用**，
  只定义接缝：你传入一个 call_llm 回调，它负责拼消息、发工具清单、归一化返回。
  接入任何厂商（OpenAI / DeepSeek / 通义 / Anthropic）都只需要写那个回调。

- **ScriptedLLMPlanner** —— 回放预设的工具调用序列，用于测试循环本身。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from ..contract import CausalStatus, Conclusion, Confidence
from ..series_utils import parse_period
from ..tools.base import ToolResult, openai_tools
from .prompts import DEFAULT_SYSTEM_PROMPT

_CN_NUMERAL = {"一": 1, "二": 2, "三": 3, "四": 4}

_QUARTER_PATTERNS = (
    re.compile(r"(\d{4})\s*[Qq]\s*([1-4])"),
    re.compile(r"(\d{4})\s*年\s*第?\s*([1-4一二三四])\s*季度"),
)
_MONTH_PATTERNS = (
    re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月"),
    re.compile(r"(\d{4})-(\d{1,2})(?!\d)"),
)


def extract_period_from_question(question: str) -> str | None:
    """从自然语言问题里抽出用户关心的期次，返回 '2025-01' 或 '2024Q4' 形式。

    这是给规则规划器用的启发式。接了 LLM 之后它依然有用 ——
    作为"用户到底问的是哪一期"的兜底，避免模型跑完一整套分析却答非所问。
    """
    for pat in _QUARTER_PATTERNS:
        m = pat.search(question)
        if m:
            raw = m.group(2)
            q = _CN_NUMERAL.get(raw, raw if raw.isdigit() else None)
            if q is not None:
                return f"{m.group(1)}Q{q}"
    for pat in _MONTH_PATTERNS:
        m = pat.search(question)
        if m:
            month = int(m.group(2))
            if 1 <= month <= 12:
                return f"{m.group(1)}-{month:02d}"
    return None


# 预测类问法。系统做的是历史数据的异常检测与归因，不做经济预测 ——
# 遇到这类问题应该明确说"超出能力范围"，而不是拿历史规律硬凑一个预测。
_PREDICTION_HINTS = (
    "预测", "预计", "明年", "后年", "未来", "将会", "会不会", "会好转",
    "会变好", "会下降", "会上升", "走势预测", "前景如何", "接下来会",
)


def detect_out_of_scope(question: str, period_range: tuple[str, str] | None = None) -> str | None:
    """检查问题是否超出系统能力范围。返回拒答理由，或 None 表示可以分析。

    两类越界：
    1. 要求预测未来 —— 系统只做历史归因，不做预测；
    2. 问的期次超出数据覆盖范围 —— 没有数据就没有结论，不能编。

    这两条都必须显式拒答。默认"顺着问题往下答"是这类系统最危险的失败模式。
    """
    if any(h in question for h in _PREDICTION_HINTS):
        return (
            "问题要求对未来的经济走势做出判断，这超出了本系统的能力范围。"
            "本系统只基于已有历史数据做异常检测与原因归因，不做经济预测。"
        )
    period = extract_period_from_question(question)
    if period and period_range:
        start, end = period_range
        t = parse_period(period)
        lo, hi = parse_period(start), parse_period(end)
        if t is not None and lo is not None and hi is not None and not (lo <= t <= hi):
            return (
                f"问题涉及的期次 {period} 超出了数据覆盖范围（{start} 至 {end}），"
                f"没有数据支撑任何结论。"
            )
    return None


@dataclass
class PlanDecision:
    """规划器的输出：下一步做什么。tool=None 表示结束。"""

    tool: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    thought: str = ""

    @property
    def finish(self) -> bool:
        return self.tool is None


@dataclass
class AgentState:
    question: str
    region_name: str
    max_steps: int = 12
    observations: list[ToolResult] = field(default_factory=list)
    conclusion: Conclusion | None = None

    def payloads(self, tool_name: str) -> list[BaseModel]:
        return [
            o.payload
            for o in self.observations
            if o.tool == tool_name and o.ok and o.payload is not None
        ]

    def last_payload(self, tool_name: str) -> BaseModel | None:
        found = self.payloads(tool_name)
        return found[-1] if found else None

    def attempted(self, tool_name: str) -> int:
        return sum(1 for o in self.observations if o.tool == tool_name)

    def failed(self, tool_name: str) -> bool:
        return any(o.tool == tool_name and not o.ok for o in self.observations)

    @property
    def tool_calls(self) -> list[str]:
        return [o.tool for o in self.observations]

    @property
    def steps(self) -> int:
        return len(self.observations)


class Planner(Protocol):
    def plan(self, state: AgentState) -> PlanDecision: ...


# ------------------------------------------------------------------ 规则规划器


class RulePlanner:
    """确定性规划器。不调大模型，按排除法的顺序走完标准分析流程。

    它内部维护一个 _skipped 集合：某个工具连续失败到上限后就永久跳过，
    保证循环不会卡在同一个失败步骤上（这是自动分析系统最常见的死法）。
    """

    name = "rule-based"

    def __init__(self, dataset_id: str | None = None, max_attempts: int = 2) -> None:
        self.dataset_id = dataset_id
        self.max_attempts = max_attempts
        self._skipped: set[str] = set()

    def plan(self, state: AgentState) -> PlanDecision:
        for _ in range(30):
            decision = self._decide(state)
            if decision is not None:
                return decision
        return PlanDecision(tool=None, thought="规划器无法继续推进，直接收束。")

    def _decide(self, state: AgentState) -> PlanDecision | None:
        """返回 None 表示"这一步应跳过"，调用方会重新评估下一步。"""
        if state.steps >= state.max_steps:
            return PlanDecision(tool=None, thought="达到最大步数上限，收束并提交已有结论。")

        if self._done(state, "submit_conclusion"):
            return PlanDecision(tool=None, thought="结论已提交。")

        # 能力边界：预测类问题在拿到数据前就能判定越界，直接拒答。
        out_of_scope = detect_out_of_scope(state.question)
        if out_of_scope:
            return self._give_up(state, out_of_scope, kind="capability")

        inventory = state.last_payload("list_region_datasets")
        if inventory is None and not self._done(state, "list_region_datasets"):
            if self._exhausted(state, "list_region_datasets"):
                return self._give_up(state, "无法读取地区数据清单，分析无法开始。")
            return PlanDecision("list_region_datasets", {}, "先看清这个地区有哪些数据、有什么已知缺陷。")

        picked = self._pick_electricity(inventory) if inventory is not None else None
        ds = self.dataset_id or picked
        if ds is None:
            return self._give_up(state, "该地区没有可识别的用电量数据集。")

        # 期次范围检查：需要 inventory 才知道数据覆盖到哪一期。
        # 问一个不存在的期次时，正确行为是拒答，而不是拿最近的期次冒充。
        period_range = self._period_range(inventory, ds)
        if period_range:
            out_of_scope = detect_out_of_scope(state.question, period_range)
            if out_of_scope:
                return self._give_up(state, out_of_scope, kind="capability")

        if not self._done(state, "load_series"):
            if self._exhausted(state, "load_series"):
                return self._give_up(state, "无法读取用电量序列。")
            return PlanDecision("load_series", {"dataset_id": ds}, "取出用电量序列的统计摘要。")

        if not self._done(state, "data_quality_report"):
            if self._exhausted(state, "data_quality_report"):
                self._skipped.add("data_quality_report")
                return None
            return PlanDecision(
                "data_quality_report", {"dataset_id": ds}, "先确认数据能不能用：缺期、突变、单位口径。"
            )

        if not self._done(state, "decompose_series"):
            if self._exhausted(state, "decompose_series"):
                return self._give_up(state, "序列无法分解，样本量可能不足。")
            return PlanDecision(
                "decompose_series", {"dataset_id": ds}, "分解出趋势与季节，看残差里有没有解释不了的部分。"
            )

        if not self._done(state, "detect_anomaly"):
            if self._exhausted(state, "detect_anomaly"):
                self._skipped.add("detect_anomaly")
                return None
            return PlanDecision(
                "detect_anomaly", {"dataset_id": ds, "max_events": 5}, "检出偏离最大的几个期次。"
            )

        scan = state.last_payload("detect_anomaly")
        has_events = bool(scan is not None and getattr(scan, "events", None))
        if not has_events and scan is not None:
            # 没检出异常就没必要归因，直接标记跳过
            self._skipped.add("explain_anomaly")
        if has_events and not self._done(state, "explain_anomaly"):
            if self._exhausted(state, "explain_anomaly"):
                self._skipped.add("explain_anomaly")
                return None
            target, why = self._pick_target_period(state, scan)
            return PlanDecision(
                "explain_anomaly",
                {"dataset_id": ds, "period": target},
                f"对 {target} 做排除式诊断（{why}）：数据构造 → 日历 → 天气 → 经济。",
            )

        if not self._done(state, "nowcast_economy"):
            if self._exhausted(state, "nowcast_economy"):
                self._skipped.add("nowcast_economy")
                return None
            return PlanDecision(
                "nowcast_economy",
                {"period": self._target_period(state)},
                "把用电同比与 GDP 同比对照，看这次偏离有没有对应的经济变化。",
            )

        return PlanDecision("submit_conclusion", synthesize_conclusion(state), "汇总已有观察，提交五段式结论。")

    # -- 辅助

    def _done(self, state: AgentState, tool_name: str) -> bool:
        return tool_name in self._skipped or state.last_payload(tool_name) is not None

    def _exhausted(self, state: AgentState, tool_name: str) -> bool:
        return state.attempted(tool_name) >= self.max_attempts

    @staticmethod
    def _period_range(inventory: BaseModel | None, dataset_id: str) -> tuple[str, str] | None:
        if inventory is None:
            return None
        data = inventory.model_dump(mode="json")
        info = next((d for d in data.get("datasets", []) if d["id"] == dataset_id), None)
        if not info:
            return None
        start, end = info.get("period_start"), info.get("period_end")
        if start and end:
            return str(start), str(end)
        return None

    @staticmethod
    def _pick_electricity(inventory: BaseModel | None) -> str | None:
        if inventory is None:
            return None
        data = inventory.model_dump(mode="json")
        candidates = [
            d for d in data.get("datasets", [])
            if "electricity" in d["id"].lower() or "用电" in (d.get("metric") or "")
        ]
        if not candidates:
            return None
        monthly = [d for d in candidates if d.get("frequency") == "monthly"]
        return (monthly or candidates)[0]["id"]

    @staticmethod
    def _pick_target_period(state: AgentState, scan: BaseModel) -> tuple[str, str]:
        """决定该解释哪一期。

        优先级：用户问题里点明的期次 > 不在序列开头"样本不足"区间的最大偏离期 > 最大偏离期。
        第二条很重要 —— 序列前 12 期的残差被端点效应污染，拿它当头条结论会误导使用者。
        """
        asked = extract_period_from_question(state.question)
        already_failed = any(
            o.tool == "explain_anomaly" and not o.ok for o in state.observations
        )
        if asked and not already_failed:
            return asked, "用户问题中点明的期次"

        events = list(getattr(scan, "events", []) or [])
        for ev in events:
            risks = getattr(ev, "data_quality_risks", None) or []
            if not any("不足一个完整季节周期" in r for r in risks):
                return ev.period, "偏离最大且不在样本不足区间"
        if events:
            return events[0].period, "偏离最大的期次（该期同时存在数据构造风险，需谨慎）"
        return "", "无可解释期次"

    @staticmethod
    def _target_period(state: AgentState) -> str:
        asked = extract_period_from_question(state.question)
        if asked:
            return asked
        scan = state.last_payload("detect_anomaly")
        if scan is not None and getattr(scan, "events", None):
            return scan.events[0].period
        series = state.last_payload("load_series")
        if series is not None:
            return series.latest_period
        return ""

    def _give_up(self, state: AgentState, reason: str, kind: str = "data") -> PlanDecision:
        """无法继续分析时，仍然走 submit_conclusion 产出诚实结论（说明为什么没做成），
        而不是抛异常或者返回一段自由文本。

        kind="capability" 用于能力边界拒答（预测类、超范围期次），
        这类情况不能提示"补充数据"，因为补数据也解决不了。
        """
        return PlanDecision(
            "submit_conclusion",
            synthesize_conclusion(state, reason, abort_kind=kind),
            f"分析中断：{reason}",
        )


# ------------------------------------------------------------------ 结论合成

def synthesize_conclusion(
    state: AgentState,
    abort_reason: str | None = None,
    abort_kind: str = "data",
) -> dict[str, Any]:
    """把工具观察汇总成五段式结论的参数。

    RulePlanner 用它生成"规则版结论"，评测时也用它来对比 LLM 的产出。
    注意置信度是算出来的，不是拍的 —— 每缺一类上下文就扣分。
    """
    quality = state.last_payload("data_quality_report")
    series = state.last_payload("load_series")
    decomp = state.last_payload("decompose_series")
    scan = state.last_payload("detect_anomaly")
    explanation = state.last_payload("explain_anomaly")
    nowcast = state.last_payload("nowcast_economy")

    evidence: list[dict[str, Any]] = []
    uncertainty: list[str] = []
    counter: list[str] = []
    confidence = 0.62

    if series is not None:
        s = series.model_dump(mode="json")
        evidence.append(
            {
                "claim": f"{s['metric']} 最新值 {s['latest_value']:.4g}"
                + (f"，同比 {s['yoy_pct'] * 100:+.2f}%" if s.get("yoy_pct") is not None else "（无同比）"),
                "tool": "load_series",
                "period": s.get("latest_period"),
                "value": f"{s['latest_value']:.4g}",
            }
        )
    else:
        confidence -= 0.2
        uncertainty.append("未能读取到用电量序列，结论缺乏数据基础。")

    if decomp is not None:
        d = decomp.model_dump(mode="json")
        evidence.append(
            {
                "claim": (
                    f"STL 分解：趋势强度 {d['trend_strength']:.2f}、季节强度 {d['seasonal_strength']:.2f}，"
                    f"趋势区间变化 {d['trend_change_pct'] * 100:+.2f}%"
                ),
                "tool": "decompose_series",
                "value": f"趋势{d['trend_strength']:.2f}/季节{d['seasonal_strength']:.2f}",
            }
        )
        if d["seasonal_strength"] > 0.7:
            uncertainty.append("季节强度高，单月同比严重受季节摆动影响，未做季节调整的读数不可直接比较。")
        if d.get("endpoint_warning"):
            confidence -= 0.08
            uncertainty.append("样本不足 3 个完整季节周期，首尾期次趋势估计不可靠。")

    if quality is not None:
        q = quality.model_dump(mode="json")
        if q.get("flags"):
            confidence -= 0.06 * min(len(q["flags"]), 2)
            uncertainty.append(f"数据质量标记：{'、'.join(q['flags'])}。")
        if q.get("missing_periods"):
            uncertainty.append(f"存在 {len(q['missing_periods'])} 个缺期。")

    if scan is not None:
        sc = scan.model_dump(mode="json")
        evidence.append(
            {
                "claim": f"共检出 {sc['n_events']} 个残差异常期次",
                "tool": "detect_anomaly",
                "value": str(sc["n_events"]),
            }
        )
        if sc["n_events"] == 0:
            uncertainty.append("在当前阈值下未检出异常，这不等于没有异常，只等于偏离未超过阈值。")

    if explanation is not None:
        ex = explanation.model_dump(mode="json")
        ev = ex["event"]
        yoy_bit = (
            f"；该期同比 {ev['yoy_pct'] * 100:+.2f}%（同比与残差偏离是两个不同的量，"
            f"同比大不等于存在异常）"
            if ev.get("yoy_pct") is not None
            else ""
        )
        evidence.append(
            {
                "claim": (
                    f"{ev['period']} 实际值 {ev['observed']:.4g}，趋势与季节解释的期望值 {ev['expected']:.4g}，"
                    f"相对偏离 {ev['residual_pct'] * 100:+.2f}%"
                    + yoy_bit
                ),
                "tool": "explain_anomaly",
                "period": ev["period"],
                "value": f"{ev['residual_pct'] * 100:+.2f}%",
            }
        )
        for cause in ev.get("candidate_causes", [])[:4]:
            evidence.append(
                {
                    "claim": f"候选原因「{cause['cause_name']}」（{cause['causal_status']}，置信 {cause['confidence']}）：{cause['evidence']}",
                    "tool": "explain_anomaly",
                    "period": ev["period"],
                }
            )
        if ev.get("data_quality_risks"):
            confidence -= 0.1
            uncertainty.append(f"该期存在数据构造风险：{'；'.join(ev['data_quality_risks'])}。")
        if not ex.get("knowledge_hits"):
            confidence -= 0.05
            uncertainty.append("知识库未检索到相关领域知识，候选原因缺少文档佐证。")
        if ex.get("calendar") and not ex["calendar"].get("notes"):
            pass
        else:
            uncertainty.append("日历或天气上下文不完整，非经济因素可能未被完全排除。")
        counter.append(
            "候选原因中未包含任何直接经济指标（分行业用电、工业增加值、PMI），"
            "因此本次偏离尚不能归因于经济变化。"
        )
    else:
        confidence -= 0.12
        uncertainty.append("未做异常归因，非经济因素（春节错位、气温负荷）尚未排除。")
        counter.append("未执行排除式诊断，无法排除日历与天气解释。")

    if nowcast is not None:
        n = nowcast.model_dump(mode="json")
        if n.get("divergence_note"):
            evidence.append(
                {"claim": n["divergence_note"], "tool": "nowcast_economy", "period": n.get("period")}
            )
        counter.append(
            "用电与经济指标的口径并不严格对齐（月度当月值 vs 季度累计值），"
            "偏离度只能作为线索，不能作为证据。"
        )
    else:
        confidence -= 0.06
        uncertainty.append("未做经济指标交叉核对。")

    if not counter:
        counter.append("缺少反证检验，结论未经证伪尝试。")

    confidence = max(0.15, min(confidence, 0.75))

    if abort_reason:
        observed = "；".join(e["claim"] for e in evidence[:3])
        if abort_kind == "capability":
            # 能力边界不是数据缺口，说"建议补充数据"会误导使用者。
            conclusion_text = f"无法给出结论。{abort_reason}"
            if observed:
                conclusion_text += f"\n\n已有观察：{observed}"
        else:
            conclusion_text = (
                f"本次分析未能完成。原因：{abort_reason} "
                f"已有观察显示：{observed or '无'}。建议补充数据后重试。"
            )
        causal = CausalStatus.DATA_QUALITY_RISK if quality else CausalStatus.OBSERVED
        confidence = min(confidence, 0.3)

    elif explanation is not None:
        ex = explanation.model_dump(mode="json")
        ev = ex["event"]
        dev = ev["residual_pct"]
        yoy = ev.get("yoy_pct")
        yoy_txt = f"{yoy * 100:+.2f}%" if yoy is not None else "无法计算"
        causes = ev.get("candidate_causes", []) or []
        top_causes = [c["cause_name"] for c in causes[:3]]
        required = (causes[-1].get("required_evidence") if causes else []) or []
        n_events = len(scan.model_dump(mode="json")["events"]) if scan else 0

        if abs(dev) < 0.05:
            # 关键分支：这一期在剥离趋势和季节后几乎没有偏离。
            # 此时正确回答是"不是经济原因"，而不是硬凑一个异常叙事 ——
            # 同比数字再大，只要残差接近 0，就说明它被季节和日历解释干净了。
            conclusion_text = (
                f"**不能归因于经济原因。**\n\n"
                f"对你询问的 {ev['period']}：该期用电同比 {yoy_txt}，"
                f"但在剥离长期趋势和季节规律后，实际值 {ev['observed']:.4g} 与期望值 {ev['expected']:.4g} "
                f"的相对偏离只有 {dev * 100:+.2f}%。也就是说，**趋势和季节规律已经完整解释了这一期的用电量，"
                f"不存在需要额外解释的异常**。\n\n"
                f"该期同比读数之所以发生变化，主要由季节摆动和日历因素造成"
                f"（诊断出的候选因素：{'、'.join(top_causes) or '无'}），"
                f"属于可预期的规律性波动，不承载经济信息。\n\n"
                f"因此本次分析不支持「用电下降说明经济走弱」这类判断。"
                + (f"若要真正评估该地区经济状态，仍需补充：{'；'.join(required)}。" if required else "")
            )
            causal = CausalStatus.OBSERVED
            confidence = min(confidence + 0.12, 0.8)
            counter.append(
                "本结论是「不构成经济信号」，其反证条件为：若后续出现连续多期同向偏离，"
                "或分行业用电出现结构性分化（如制造业用电持续低于服务业），则该判断需要推翻。"
            )
            uncertainty.append(
                f"该期残差接近零只说明它被趋势和季节解释，不代表该地区经济没有问题 —— "
                f"经济状态需要独立的经济指标判断，不能由单期用电读数推断。"
            )

        else:
            conclusion_text = (
                f"在剥离趋势与季节规律后，{state.region_name}的 {ev['period']} 存在可检出的偏离："
                f"实际值 {ev['observed']:.4g}，期望值 {ev['expected']:.4g}，相对偏离 {dev * 100:+.2f}%"
                + (f"（该期用电同比 {yoy_txt}）" if yoy is not None else "")
                + f"。对这次偏离做排除式诊断后，候选解释按优先级为：{'、'.join(top_causes) or '无'}。\n\n"
                f"其中经济类原因的证据强度最弱 —— 目前只有电力偏离这一条线索，"
                f"没有分行业用电、工业增加值或 PMI 等直接经济指标佐证，"
                f"因此**不能据此判断该地区经济走弱或走强**。"
                + (f"要确认经济含义，需要补充：{'；'.join(required)}。" if required else "")
                + (f"\n\n（全序列共检出 {n_events} 个残差异常期次。）" if n_events else "")
            )
            causal = CausalStatus.CANDIDATE_CAUSE

    else:
        conclusion_text = (
            f"已完成对{state.region_name}用电数据的基础分析，"
            f"但由于未能完成异常归因，尚不能给出关于经济状态的可核查判断。"
        )
        causal = CausalStatus.OBSERVED

    return {
        "conclusion": conclusion_text,
        "causal_status": causal,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "counter_evidence": counter,
        "uncertainty": uncertainty,
    }


# ------------------------------------------------------------------ LLM 规划器


class LLMPlanner:
    """把工具清单和观察历史交给大模型，由模型决定下一步。

    **本类不包含任何具体厂商的 API 调用。** 接入方式：

        def call_llm(messages, tools):
            # messages: OpenAI 兼容的消息列表
            # tools:    OpenAI 兼容的 function-calling 清单（来自 tools.openai_tools()）
            # 返回值：OpenAI 的 response.choices[0].message 或 Anthropic 的 response.content
            return client.chat.completions.create(
                model="...", messages=messages, tools=tools
            ).choices[0].message

        loop = AgentLoop(ctx, LLMPlanner(call_llm))
        loop.run("2025年1月浙江用电同比下降是经济原因吗？")

    回调返回 None 或空的 tool_calls 表示模型认为可以结束了。
    """

    name = "llm"

    def __init__(
        self,
        call_llm: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any],
        system_prompt: str | None = None,
        max_observation_chars: int = 6000,
    ) -> None:
        self.call_llm = call_llm
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_observation_chars = max_observation_chars
        self._pending: list[dict[str, Any]] = []

    def build_messages(self, state: AgentState) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"分析地区：{state.region_name}\n"
                    f"用户问题：{state.question}\n\n"
                    f"请自主决定调用哪些工具来完成分析，最后用 submit_conclusion 提交结论。"
                ),
            },
        ]
        for i, obs in enumerate(state.observations):
            call_id = f"call_{i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": obs.tool,
                                "arguments": json.dumps(obs.arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            content = obs.text or ("（工具执行失败）" if not obs.ok else "（无返回）")
            if len(content) > self.max_observation_chars:
                content = content[: self.max_observation_chars] + "\n…（观察结果过长已截断）"
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
        return messages

    def plan(self, state: AgentState) -> PlanDecision:
        if state.steps >= state.max_steps:
            return PlanDecision(tool=None, thought="达到最大步数，强制收束。")

        messages = self.build_messages(state)
        tools = openai_tools()
        response = self.call_llm(messages, tools)
        tool_name, arguments, thought = self._normalize(response)
        if tool_name is None:
            # 模型没给工具调用就等同于想结束了，但结论必须走 submit_conclusion，
            # 所以这里兜底成规则合成的结论，避免产出一个没校验的自由文本。
            return PlanDecision(
                "submit_conclusion",
                synthesize_conclusion(state, "模型未通过 submit_conclusion 提交结论，已由兜底逻辑合成。"),
                thought or "模型未返回工具调用。",
            )
        return PlanDecision(tool_name, arguments, thought)

    @staticmethod
    def _normalize(response: Any) -> tuple[str | None, dict[str, Any], str]:
        """把 OpenAI / Anthropic / 原生 dict 三种返回形态归一成 (tool, args, thought)。"""
        if response is None:
            return None, {}, ""

        if isinstance(response, dict):
            data = response
        elif hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = {
                k: getattr(response, k)
                for k in ("content", "tool_calls", "reasoning_content")
                if hasattr(response, k)
            }

        thought = str(data.get("reasoning_content") or data.get("thought") or "")
        content = data.get("content")
        if isinstance(content, str) and not thought:
            thought = content[:400]

        calls = data.get("tool_calls")
        if calls:
            first = calls[0]
            fn = first.get("function") if isinstance(first, dict) else getattr(first, "function", None)
            if isinstance(fn, dict):
                name, raw_args = fn.get("name"), fn.get("arguments")
            else:
                name, raw_args = getattr(fn, "name", None), getattr(fn, "arguments", None)
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    raw_args = {}
            return name, dict(raw_args or {}), thought

        if isinstance(content, list):
            for block in content:
                b = block if isinstance(block, dict) else getattr(block, "__dict__", {})
                if b.get("type") == "tool_use":
                    return b.get("name"), dict(b.get("input") or {}), thought

        return None, {}, thought


class ScriptedPlanner:
    """按预设脚本回放工具调用。用于测试循环本身，不参与实际分析。"""

    name = "scripted"

    def __init__(self, script: list[tuple[str, dict[str, Any]]]) -> None:
        self.script = list(script)

    def plan(self, state: AgentState) -> PlanDecision:
        if state.steps >= len(self.script):
            return PlanDecision(
                "submit_conclusion", synthesize_conclusion(state), "脚本执行完毕，提交结论。"
            )
        tool_name, arguments = self.script[state.steps]
        return PlanDecision(tool_name, arguments, f"脚本第 {state.steps + 1} 步。")
