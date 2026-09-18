from __future__ import annotations

from pathlib import Path

import pandas as pd

from power_econ.agents import PowerEconomyOrchestrator
from power_econ.config import load_settings
from power_econ.data import collect_raw_data
from power_econ.features.builder import FeatureBuilder
from power_econ.models import TrainingPipeline


def test_end_to_end_smoke(tmp_path: Path) -> None:
    settings = load_settings("configs/quick_test.yaml")
    settings.root_dir = tmp_path
    settings.paths.knowledge_dir = Path.cwd() / "knowledge"
    settings.data.start = "2024-01-01"
    settings.data.end = "2024-02-25"
    settings.data.refresh = True
    settings.ensure_directories()

    raw = collect_raw_data(settings)
    frame, spec = FeatureBuilder(settings).run(raw)
    summary = TrainingPipeline(settings).run(frame, spec)
    assert summary.forecaster_metrics["mae"] > 0
    assert Path(summary.artifacts["forecaster"]).exists()

    orch = PowerEconomyOrchestrator.from_settings(settings)
    scored = orch.perception.scores()
    assert not scored.empty
    highest = scored.sort_values("anomaly_score", ascending=False).iloc[0]
    event = orch.engine._row_to_event(highest)  # smoke-test the complete diagnosis stack
    diagnosis = orch.diagnosis.diagnose(event)
    assert diagnosis.top_causes
    assert diagnosis.narrative

    end = pd.Timestamp(frame["timestamp"].iloc[-1])
    start = end - pd.Timedelta(days=7)
    report = orch.generate_report(str(start), str(end), max_diagnoses=2)
    assert report.executive_summary
    assert (settings.resolve(settings.paths.report_dir) / f"{report.report_id}.md").exists()

    answer = orch.ask("这个系统如何避免把天气波动误判成经济变化？", "pytest")
    assert answer.answer
    assert 0 <= answer.confidence <= 1
