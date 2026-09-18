"""契约层：输入数据包、工具 I/O、五段式输出。

这个模块是全系统唯一的"接口真相"。三个人各自实现的部分（工具层 / 智能体内核 /
领域判断）都只通过这里的模型对话，不允许各自定义私有格式。

设计上刻意的三个约束：
1. 因果强度必须显式分级（CausalStatus），从类型层面堵住"相关性写成因果"。
2. 数据构造风险（端点、累计差分、单位推断）是一等公民，不是备注。
3. 结论固定五段式，缺任何一段都不算产出。
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

DISCLAIMER = (
    "本结论由自动化流程基于公开数据生成，给出的是带有证据与不确定性的原因候选，"
    "不构成已证明的因果关系，也不构成任何投资或政策建议。"
)


class Frequency(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    UNKNOWN = "unknown"


class DatasetKind(str, Enum):
    TIMESERIES = "timeseries"
    CALENDAR = "calendar"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TABLE = "table"


class CausalStatus(str, Enum):
    """因果强度分级。

    绝不允许把 CORRELATION 或 CANDIDATE_CAUSE 直接输出成 ESTABLISHED。
    ESTABLISHED 在本项目里是保留值，当前不应被任何工具产出。
    """

    OBSERVED = "observed"
    CORRELATION = "correlation"
    CANDIDATE_CAUSE = "candidate_cause"
    DATA_QUALITY_RISK = "data_quality_risk"
    ESTABLISHED = "established"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------- 地区与数据集


class RegionInfo(BaseModel):
    code: str
    name: str
    level: str = "unknown"
    timezone: str = "Asia/Shanghai"
    representative_point: dict[str, Any] | None = None
    caveats: list[str] = Field(default_factory=list)
    pack_dir: str | None = None


class DatasetInfo(BaseModel):
    id: str
    kind: DatasetKind
    path: str
    metric: str
    unit: str | None = None
    frequency: Frequency = Frequency.UNKNOWN
    industry: str | None = None
    n_rows: int = 0
    period_start: str | None = None
    period_end: str | None = None
    columns: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    discovered: bool = False


class RegionInventory(BaseModel):
    """list_region_datasets 的返回。LLM 靠它知道这个地区有什么可用。"""

    region: RegionInfo
    datasets: list[DatasetInfo]
    knowledge_graph_available: bool = False
    knowledge_doc_count: int = 0
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- 数据质量


class DataQualityReport(BaseModel):
    dataset_id: str
    metric: str
    n_rows: int
    n_missing: int = 0
    missing_periods: list[str] = Field(default_factory=list)
    duplicate_periods: list[str] = Field(default_factory=list)
    irregular_gaps: list[dict[str, Any]] = Field(default_factory=list)
    abrupt_jumps: list[dict[str, Any]] = Field(default_factory=list)
    endpoint_warning: bool = False
    flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SeriesSummary(BaseModel):
    dataset_id: str
    metric: str
    unit: str | None = None
    frequency: Frequency = Frequency.UNKNOWN
    n_points: int
    period_start: str
    period_end: str
    latest_period: str
    latest_value: float
    yoy_pct: float | None = None
    mom_pct: float | None = None
    mean: float
    std: float
    series: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------- 分解与异常


class DecompositionPoint(BaseModel):
    period: str
    observed: float
    trend: float
    seasonal: float
    residual: float
    residual_pct: float


class DecompositionResult(BaseModel):
    dataset_id: str
    metric: str
    unit: str | None = None
    method: str = "STL"
    period: int
    robust: bool = True
    trend_strength: float
    seasonal_strength: float
    residual_std: float
    residual_mad: float
    trend_start: float
    trend_end: float
    trend_change_pct: float
    endpoint_warning: bool = False
    n_points: int = 0
    points: list[DecompositionPoint] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CandidateCause(BaseModel):
    """一个候选原因。relation 用 MAY_AFFECT / CANDIDATE_CAUSE / DATA_QUALITY_RISK，
    禁止出现 causedBy。"""

    cause_id: str
    cause_name: str
    relation: str
    causal_status: CausalStatus = CausalStatus.CANDIDATE_CAUSE
    confidence: Confidence = Confidence.LOW
    evidence: str = ""
    counter_evidence: str | None = None
    required_evidence: list[str] = Field(default_factory=list)
    source: str | None = None


class AnomalyEvent(BaseModel):
    event_id: str
    dataset_id: str
    metric: str
    unit: str | None = None
    period: str
    observed: float
    expected: float
    residual_pct: float
    yoy_pct: float | None = None
    direction: str
    level: Confidence
    robust_z: float
    endpoint_warning: bool = False
    data_quality_risks: list[str] = Field(default_factory=list)
    candidate_causes: list[CandidateCause] = Field(default_factory=list)


class AnomalyScanResult(BaseModel):
    dataset_id: str
    metric: str
    detector: str
    threshold_high_pct: float
    threshold_medium_pct: float
    n_events: int
    events: list[AnomalyEvent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- 知识与上下文


class KnowledgeTriple(BaseModel):
    subject_id: str
    subject_name: str
    relation: str
    object_id: str
    object_name: str
    description: str | None = None
    direction: str | None = None
    confidence: str | None = None


class KnowledgeQueryResult(BaseModel):
    query: dict[str, Any]
    n_matches: int
    triples: list[KnowledgeTriple] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RetrievedDoc(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float
    source: str


class KnowledgeSearchResult(BaseModel):
    query: str
    backend: str
    n_hits: int
    hits: list[RetrievedDoc] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CalendarContext(BaseModel):
    region: str
    start: str
    end: str
    n_days: int = 0
    n_workdays: int = 0
    n_nonworkdays: int = 0
    holiday_break_days: int = 0
    makeup_workdays: int = 0
    holidays: list[dict[str, Any]] = Field(default_factory=list)
    spring_festival_days: int | None = None
    spring_festival_note: str | None = None
    notes: list[str] = Field(default_factory=list)


class WeatherContext(BaseModel):
    region: str
    start: str
    end: str
    available: bool
    source: str | None = None
    is_proxy: bool = True
    proxy_note: str | None = None
    monthly: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EconomicNowcast(BaseModel):
    region: str
    period: str
    indicator: str
    estimated_yoy_pct: float | None = None
    electricity_yoy_pct: float | None = None
    method: str = ""
    n_observations: int = 0
    divergence_note: str | None = None
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- 五段式输出


class Evidence(BaseModel):
    claim: str
    tool: str
    source: str | None = None
    value: Any = None
    period: str | None = None
    checkable: bool = True


class Conclusion(BaseModel):
    """五段式输出契约 —— 全系统唯一对外的结论格式。

    counter_evidence 和 uncertainty 被强制要求非空，理由很简单：
    这两个字段是模型最容易跳过、也最能暴露"没真正检验过自己结论"的地方。
    把纪律写进类型，比写进提示词可靠。

    evidence 允许为空 —— 在"什么都没跑成"的极端情况下确实没有证据可列，
    但那种情况必须靠 uncertainty 说明清楚。
    """

    region: str
    question: str
    conclusion: str
    causal_status: CausalStatus = CausalStatus.OBSERVED
    evidence: list[Evidence] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _enforce_five_parts(self) -> "Conclusion":
        if not self.counter_evidence:
            raise ValueError("五段式契约要求 counter_evidence 至少一条：什么事实会推翻这个结论？")
        if not self.uncertainty:
            raise ValueError("五段式契约要求 uncertainty 至少一条：还缺什么数据？")
        return self

    def to_markdown(self) -> str:
        lines = [
            f"# {self.region} · {self.question}",
            "",
            "## 结论",
            self.conclusion,
            "",
            f"因果强度：`{self.causal_status.value}`　置信度：`{self.confidence:.2f}`",
            "",
            "## 证据",
        ]
        if self.evidence:
            for i, e in enumerate(self.evidence, 1):
                bits = [f"{i}. {e.claim}"]
                if e.value is not None:
                    bits.append(f"　值：`{e.value}`")
                if e.period:
                    bits.append(f"　期：`{e.period}`")
                bits.append(f"　来源：`{e.tool}`")
                lines.append("".join(bits))
        else:
            lines.append("（无）")

        lines += ["", "## 反证"]
        lines += [f"- {c}" for c in self.counter_evidence] or ["- （无）"]

        lines += ["", "## 不确定性"]
        lines += [f"- {u}" for u in self.uncertainty] or ["- （无）"]

        if self.caveats:
            lines += ["", "## 数据边界"]
            lines += [f"- {c}" for c in self.caveats]

        if self.tool_calls:
            lines += ["", "## 调用轨迹"]
            lines += [f"- `{t}`" for t in self.tool_calls]

        lines += ["", "---", "", f"_{self.disclaimer}_", f"_生成时间：{self.generated_at}_"]
        return "\n".join(lines)


def dump_json(model: BaseModel, *, indent: int = 2) -> str:
    """统一的中文安全 JSON 序列化。工具返回值都走这里，保证 LLM 看到的是可解析文本。"""
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=indent)
