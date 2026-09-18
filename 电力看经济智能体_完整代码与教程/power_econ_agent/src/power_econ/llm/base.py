from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class DiagnosisNarrative(BaseModel):
    economic_signal: str
    narrative: str
    uncertainty: str
    needs_human_review: bool = True


class ReportNarrative(BaseModel):
    executive_summary: str
    load_observations: list[str] = Field(default_factory=list)
    anomaly_assessment: list[str] = Field(default_factory=list)
    economic_interpretation: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    risks_and_limitations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_quality: str


class AnswerNarrative(BaseModel):
    answer: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class LLMProvider(ABC):
    @abstractmethod
    def diagnosis(self, payload: dict[str, Any]) -> DiagnosisNarrative:
        raise NotImplementedError

    @abstractmethod
    def report(self, payload: dict[str, Any]) -> ReportNarrative:
        raise NotImplementedError

    @abstractmethod
    def answer(self, payload: dict[str, Any]) -> AnswerNarrative:
        raise NotImplementedError
