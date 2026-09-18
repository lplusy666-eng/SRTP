from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import Settings, load_settings
from ..features.builder import FeatureBuilder
from ..knowledge import DomainKnowledgeGraph, LocalKnowledgeBase
from ..llm import build_llm_provider
from ..models import InferenceEngine
from ..schemas import AnomalyEvent, DiagnosisResult, PowerEconomyReport, QAResponse
from ..storage import EventStore
from .diagnosis import DiagnosisAgent
from .perception import PerceptionAgent
from .qa import QAAgent
from .report import ReportAgent


@dataclass
class PowerEconomyOrchestrator:
    settings: Settings
    frame: pd.DataFrame
    engine: InferenceEngine
    store: EventStore
    perception: PerceptionAgent
    diagnosis: DiagnosisAgent
    report: ReportAgent
    qa: QAAgent

    @classmethod
    def from_settings(cls, settings: Settings) -> PowerEconomyOrchestrator:
        feature_path = settings.resolve(settings.paths.processed_dir / "features.csv")
        spec_path = settings.resolve(settings.paths.processed_dir / "feature_spec.json")
        model_meta_path = settings.resolve(settings.paths.model_dir / "model_metadata.json")
        assert feature_path is not None and spec_path is not None and model_meta_path is not None
        if not feature_path.exists() or not spec_path.exists():
            raise FileNotFoundError("缺少特征文件，请先执行 power-econ features")
        if not model_meta_path.exists():
            raise FileNotFoundError("缺少模型文件，请先执行 power-econ train")
        frame = pd.read_csv(feature_path)
        spec = FeatureBuilder.load_feature_spec(spec_path)
        engine = InferenceEngine(settings, frame, spec)
        db_path = settings.resolve(settings.paths.db_path)
        knowledge_dir = settings.resolve(settings.paths.knowledge_dir)
        report_dir = settings.resolve(settings.paths.report_dir)
        validation_path = settings.resolve(settings.paths.raw_dir / "data_validation.json")
        monthly_path = settings.resolve(settings.paths.output_dir / "monthly_economic_signals.csv")
        assert all(x is not None for x in [db_path, knowledge_dir, report_dir, validation_path, monthly_path])
        store = EventStore(db_path)  # type: ignore[arg-type]
        graph = DomainKnowledgeGraph(Path(knowledge_dir) / "domain_knowledge.yaml")  # type: ignore[arg-type]
        kb = LocalKnowledgeBase(Path(knowledge_dir) / "docs")  # type: ignore[arg-type]
        llm = build_llm_provider(settings)
        perception = PerceptionAgent(engine, frame, store)
        diagnosis = DiagnosisAgent(engine, frame, graph, kb, llm, store)
        report = ReportAgent(
            frame=frame,
            perception=perception,
            diagnosis=diagnosis,
            llm=llm,
            store=store,
            report_dir=Path(report_dir),  # type: ignore[arg-type]
            validation_path=Path(validation_path),  # type: ignore[arg-type]
            monthly_signal_path=Path(monthly_path),  # type: ignore[arg-type]
            region=settings.project.region,
            model_version=str(engine.metadata.get("version", "unknown")),
        )
        qa = QAAgent(kb, llm, store)
        return cls(settings, frame, engine, store, perception, diagnosis, report, qa)

    @classmethod
    def from_config(cls, config_path: str = "configs/demo.yaml") -> PowerEconomyOrchestrator:
        return cls.from_settings(load_settings(config_path))

    def monitor(self, start: str | None = None, end: str | None = None) -> list[AnomalyEvent]:
        return self.perception.monitor(start, end)

    def diagnose_event(self, event_id: str) -> DiagnosisResult:
        return self.diagnosis.diagnose_by_id(event_id)

    def generate_report(self, start: str, end: str, max_diagnoses: int = 8) -> PowerEconomyReport:
        return self.report.generate(start, end, max_diagnoses=max_diagnoses)

    def ask(self, question: str, session_id: str = "default") -> QAResponse:
        return self.qa.ask(question, session_id)
