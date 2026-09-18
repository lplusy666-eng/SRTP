from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    group: str
    contribution_mw: float
    normalized_importance: float


class AnomalyEvent(BaseModel):
    event_id: str
    timestamp: datetime
    region: str
    observed_load_mw: float
    expected_load_mw: float
    lower_mw: float
    upper_mw: float
    residual_mw: float
    forecast_score: float
    vae_score: float
    anomaly_score: float
    threshold: float
    severity: Literal["low", "medium", "high", "critical"]
    is_anomaly: bool
    gate_weights: dict[str, float] = Field(default_factory=dict)
    true_label: int | None = None
    true_cause: str | None = None


class CauseEvidence(BaseModel):
    cause_id: str
    label: str
    category: str
    score: float
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    related_nodes: list[str] = Field(default_factory=list)


class DiagnosisResult(BaseModel):
    event: AnomalyEvent
    top_causes: list[CauseEvidence]
    feature_contributions: list[FeatureContribution]
    knowledge_snippets: list[str]
    economic_signal: str
    narrative: str
    uncertainty: str
    needs_human_review: bool
    generated_at: datetime


class PowerEconomyReport(BaseModel):
    report_id: str
    title: str
    region: str
    period_start: datetime
    period_end: datetime
    executive_summary: str
    load_observations: list[str]
    anomaly_assessment: list[str]
    economic_interpretation: list[str]
    key_evidence: list[str]
    risks_and_limitations: list[str]
    recommendations: list[str]
    data_quality: str
    generated_at: datetime
    model_version: str


class QAResponse(BaseModel):
    answer: str
    evidence: list[str] = Field(default_factory=list)
    referenced_event_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    generated_at: datetime


class TrainSummary(BaseModel):
    forecaster_metrics: dict[str, float]
    anomaly_metrics: dict[str, float]
    nowcast_metrics: dict[str, float]
    artifacts: dict[str, str]
    metadata: dict[str, Any] = Field(default_factory=dict)
