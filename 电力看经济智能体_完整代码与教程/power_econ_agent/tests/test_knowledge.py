from __future__ import annotations

from pathlib import Path

from power_econ.knowledge.graph import DomainKnowledgeGraph
from power_econ.schemas import FeatureContribution


def test_sensor_fault_rule_ranks_first() -> None:
    graph = DomainKnowledgeGraph(Path("knowledge/domain_knowledge.yaml"))
    context = {
        "sensor_quality": 0.0,
        "abs_load_diff_1": 400.0,
        "vae_score": 4.0,
        "missing_fraction": 0.0,
        "residual_mw": 380.0,
        "abs_residual_mw": 380.0,
        "temperature_2m": 20.0,
        "temp_anomaly": 0.0,
        "weather_normal": 1,
    }
    contributions = [
        FeatureContribution(group="event", contribution_mw=100, normalized_importance=0.7),
        FeatureContribution(group="load", contribution_mw=30, normalized_importance=0.3),
    ]
    causes = graph.rank_causes(context, contributions)
    assert causes
    assert causes[0].cause_id == "sensor_fault"
    assert causes[0].confidence > 0.5
