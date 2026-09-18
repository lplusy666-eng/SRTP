from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ..knowledge import DomainKnowledgeGraph, LocalKnowledgeBase
from ..llm.base import LLMProvider
from ..models.inference import InferenceEngine
from ..schemas import AnomalyEvent, CauseEvidence, DiagnosisResult
from ..storage import EventStore


class DiagnosisAgent:
    """Diagnosis agent: ablation explanation + executable KG rules + local RAG + LLM."""

    def __init__(
        self,
        engine: InferenceEngine,
        frame: pd.DataFrame,
        graph: DomainKnowledgeGraph,
        knowledge_base: LocalKnowledgeBase,
        llm: LLMProvider,
        store: EventStore,
    ):
        self.engine = engine
        self.frame = frame.sort_values("timestamp").reset_index(drop=True)
        self.graph = graph
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.store = store
        self.timestamps = pd.DatetimeIndex(pd.to_datetime(self.frame["timestamp"]))

    def _find_index(self, timestamp: datetime | str) -> int:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None and self.timestamps.tz is not None:
            ts = ts.tz_localize(self.timestamps.tz)
        elif self.timestamps.tz is not None:
            ts = ts.tz_convert(self.timestamps.tz)
        pos = int(self.timestamps.searchsorted(ts))
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(self.timestamps)]
        if not candidates:
            raise IndexError(f"事件时间不在特征数据范围内: {timestamp}")
        return min(candidates, key=lambda i: abs(self.timestamps[i] - ts))

    def _context(self, event: AnomalyEvent, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        context = row.to_dict()
        context.update(event.model_dump(mode="python"))
        context["residual_mw"] = event.residual_mw
        context["abs_residual_mw"] = abs(event.residual_mw)
        context["abs_load_diff_1"] = abs(float(row.get("load_diff_1", 0.0)))
        temperature = float(row.get("temperature_2m", 18.0))
        temp_anomaly = float(row.get("temp_anomaly", 0.0))
        context["weather_normal"] = int(5 <= temperature <= 31 and abs(temp_anomaly) < 4.0)
        return context

    def diagnose(self, event: AnomalyEvent | dict[str, Any]) -> DiagnosisResult:
        if not isinstance(event, AnomalyEvent):
            event = AnomalyEvent.model_validate(event)
        cached = self.store.get_diagnosis(event.event_id)
        if cached:
            return DiagnosisResult.model_validate(cached)

        index = self._find_index(event.timestamp)
        contributions = self.engine.explain_index(index, self.frame)
        context = self._context(event, index)
        causes = self.graph.rank_causes(context, contributions, top_k=5)
        if not causes:
            causes = [
                CauseEvidence(
                    cause_id="unresolved",
                    label="复合因素或知识库未覆盖原因",
                    category="other",
                    score=0.1,
                    confidence=0.2,
                    evidence=["异常分数超过阈值，但已有规则证据不足。"],
                    related_nodes=["人工复核"],
                )
            ]
        query = " ".join(
            [
                event.region,
                f"负荷残差 {event.residual_mw:+.1f} MW",
                *(c.label for c in causes[:3]),
                *(c.category for c in causes[:3]),
            ]
        )
        snippets = self.knowledge_base.search(query, top_k=4)
        payload = {
            "event": event.model_dump(mode="json"),
            "causes": [c.model_dump(mode="json") for c in causes],
            "feature_contributions": [c.model_dump(mode="json") for c in contributions],
            "knowledge_snippets": snippets,
            "local_context": {
                key: context.get(key)
                for key in [
                    "temperature_2m",
                    "temp_anomaly",
                    "is_holiday",
                    "is_business_hour",
                    "policy_event",
                    "sensor_quality",
                    "pmi_published",
                    "industrial_yoy_published",
                    "load_residual_mean_24",
                ]
            },
        }
        generated = self.llm.diagnosis(payload)
        result = DiagnosisResult(
            event=event,
            top_causes=causes,
            feature_contributions=contributions,
            knowledge_snippets=[
                f"[{x['source']} / {x['title']}] {x['text']}" for x in snippets
            ],
            economic_signal=generated.economic_signal,
            narrative=generated.narrative,
            uncertainty=generated.uncertainty,
            needs_human_review=generated.needs_human_review,
            generated_at=datetime.now(UTC),
        )
        self.store.upsert_diagnosis(event.event_id, result.model_dump(mode="json"))
        return result

    def diagnose_by_id(self, event_id: str) -> DiagnosisResult:
        cached = self.store.get_diagnosis(event_id)
        if cached:
            return DiagnosisResult.model_validate(cached)
        event = self.store.get_anomaly(event_id)
        if event is None:
            raise KeyError(f"未找到异常事件: {event_id}。请先执行 scan/monitor。")
        return self.diagnose(event)
