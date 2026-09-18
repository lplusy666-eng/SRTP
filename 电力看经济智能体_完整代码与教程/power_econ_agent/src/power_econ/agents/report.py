from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..llm.base import LLMProvider
from ..schemas import DiagnosisResult, PowerEconomyReport
from ..storage import EventStore
from ..utils import read_json, write_json
from .diagnosis import DiagnosisAgent
from .perception import PerceptionAgent


class ReportAgent:
    """Generation agent: closes the loop from period data to a traceable report."""

    def __init__(
        self,
        frame: pd.DataFrame,
        perception: PerceptionAgent,
        diagnosis: DiagnosisAgent,
        llm: LLMProvider,
        store: EventStore,
        report_dir: Path,
        validation_path: Path,
        monthly_signal_path: Path,
        region: str,
        model_version: str,
    ):
        self.frame = frame.sort_values("timestamp").reset_index(drop=True)
        self.perception = perception
        self.diagnosis = diagnosis
        self.llm = llm
        self.store = store
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.validation_path = validation_path
        self.monthly_signal_path = monthly_signal_path
        self.region = region
        self.model_version = model_version
        self.timestamps = pd.DatetimeIndex(pd.to_datetime(self.frame["timestamp"]))

    def _period(self, start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if self.timestamps.tz is not None:
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(self.timestamps.tz)
            else:
                start_ts = start_ts.tz_convert(self.timestamps.tz)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize(self.timestamps.tz)
            else:
                end_ts = end_ts.tz_convert(self.timestamps.tz)
        period = self.frame.loc[(self.timestamps >= start_ts) & (self.timestamps <= end_ts)].copy()
        if period.empty:
            raise ValueError("所选报告期没有数据")
        return start_ts, end_ts, period

    def _stats(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp, period: pd.DataFrame, anomaly_count: int) -> dict[str, float]:
        duration = end_ts - start_ts
        previous_end = start_ts - pd.Timedelta(nanoseconds=1)
        previous_start = previous_end - duration
        previous = self.frame.loc[
            (self.timestamps >= previous_start) & (self.timestamps <= previous_end)
        ]
        current_mean = float(period["load_mw"].mean())
        previous_mean = float(previous["load_mw"].mean()) if not previous.empty else current_mean
        return {
            "average_load_mw": current_mean,
            "peak_load_mw": float(period["load_mw"].max()),
            "valley_load_mw": float(period["load_mw"].min()),
            "energy_mwh": float(period["load_mw"].sum()),
            "comparison_change_pct": float((current_mean - previous_mean) / max(abs(previous_mean), 1e-6) * 100),
            "anomaly_count": float(anomaly_count),
            "sensor_quality_mean": float(period["sensor_quality"].mean()),
        }

    def _nowcast_context(self, end_ts: pd.Timestamp) -> dict[str, Any]:
        if not self.monthly_signal_path.exists():
            return {}
        monthly = pd.read_csv(self.monthly_signal_path, parse_dates=["month"])
        eligible = monthly.loc[monthly["month"] <= end_ts.tz_localize(None).to_period("M").to_timestamp()]
        if eligible.empty:
            return {}
        row = eligible.iloc[-1]
        return {k: row[k] for k in row.index if pd.notna(row[k])}

    def generate(
        self,
        start: str,
        end: str,
        max_diagnoses: int = 8,
    ) -> PowerEconomyReport:
        start_ts, end_ts, period = self._period(start, end)
        events = self.perception.monitor(start, end)
        events_by_score = sorted(events, key=lambda x: x.anomaly_score, reverse=True)
        diagnoses: list[DiagnosisResult] = [
            self.diagnosis.diagnose(event) for event in events_by_score[:max_diagnoses]
        ]
        stats = self._stats(start_ts, end_ts, period, len(events))
        validation = read_json(self.validation_path, default={}) or {}
        data_quality = (
            f"质量状态={validation.get('status', 'unknown')}；"
            f"平均传感器质量={stats['sensor_quality_mean']:.3f}；"
            f"提示={'; '.join(validation.get('warnings', [])) or '无'}"
        )
        key_evidence: list[str] = []
        for d in diagnoses:
            if d.top_causes:
                key_evidence.append(
                    f"{d.event.event_id}: {d.top_causes[0].label}；"
                    + "；".join(d.top_causes[0].evidence[:2])
                )
        payload = {
            "region": self.region,
            "period_start": start_ts.isoformat(),
            "period_end": end_ts.isoformat(),
            "stats": stats,
            "diagnoses": [d.model_dump(mode="json") for d in diagnoses],
            "key_evidence": key_evidence,
            "data_quality": data_quality,
            "economic_nowcast": self._nowcast_context(end_ts),
        }
        generated = self.llm.report(payload)
        report_id = hashlib.sha1(
            f"{self.region}|{start_ts.isoformat()}|{end_ts.isoformat()}".encode()
        ).hexdigest()[:16]
        report = PowerEconomyReport(
            report_id=report_id,
            title=f"{self.region}电力经济态势分析报告（{start_ts.date()}—{end_ts.date()}）",
            region=self.region,
            period_start=start_ts.to_pydatetime(),
            period_end=end_ts.to_pydatetime(),
            executive_summary=generated.executive_summary,
            load_observations=generated.load_observations,
            anomaly_assessment=generated.anomaly_assessment,
            economic_interpretation=generated.economic_interpretation,
            key_evidence=generated.key_evidence,
            risks_and_limitations=generated.risks_and_limitations,
            recommendations=generated.recommendations,
            data_quality=generated.data_quality,
            generated_at=datetime.now(UTC),
            model_version=self.model_version,
        )
        json_path = self.report_dir / f"{report_id}.json"
        md_path = self.report_dir / f"{report_id}.md"
        write_json(json_path, report.model_dump(mode="json"))
        md_path.write_text(self.to_markdown(report), encoding="utf-8")
        self.store.upsert_report(
            report_id,
            start_ts.isoformat(),
            end_ts.isoformat(),
            report.model_dump(mode="json"),
        )
        return report

    @staticmethod
    def to_markdown(report: PowerEconomyReport) -> str:
        def section(title: str, items: list[str]) -> str:
            body = "\n".join(f"- {item}" for item in items) if items else "- 无"
            return f"## {title}\n\n{body}\n"

        parts = [
            f"# {report.title}",
            "",
            f"- 报告编号：`{report.report_id}`",
            f"- 模型版本：`{report.model_version}`",
            f"- 生成时间：{report.generated_at.isoformat()}",
            "",
            "## 执行摘要",
            "",
            report.executive_summary,
            "",
            section("负荷观察", report.load_observations),
            section("异常研判", report.anomaly_assessment),
            section("经济含义", report.economic_interpretation),
            section("关键证据", report.key_evidence),
            section("风险与局限", report.risks_and_limitations),
            section("建议", report.recommendations),
            "## 数据质量",
            "",
            report.data_quality,
            "",
        ]
        return "\n".join(parts)
