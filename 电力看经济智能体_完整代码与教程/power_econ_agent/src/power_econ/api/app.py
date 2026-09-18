from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import load_settings
from ..utils import read_json
from .runtime import get_orchestrator


class ReportRequest(BaseModel):
    start: str
    end: str
    max_diagnoses: int = Field(default=8, ge=0, le=30)


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = "api"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Power Economy Agent API",
        version="0.1.0",
        description="特征解耦、异常融合、知识增强诊断、报告与问答 API",
    )

    def orchestrator():
        try:
            return get_orchestrator(os.getenv("POWER_ECON_CONFIG", "configs/demo.yaml"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        config_path = os.getenv("POWER_ECON_CONFIG", "configs/demo.yaml")
        try:
            settings = load_settings(config_path)
            metadata_path = settings.resolve(settings.paths.model_dir / "model_metadata.json")
            feature_path = settings.resolve(settings.paths.processed_dir / "features.csv")
            return {
                "status": "ok" if metadata_path and metadata_path.exists() else "not_trained",
                "config": str(settings.config_path),
                "features_ready": bool(feature_path and feature_path.exists()),
                "model_ready": bool(metadata_path and metadata_path.exists()),
            }
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    @app.get("/v1/forecast/latest")
    def latest_forecast() -> list[dict[str, Any]]:
        frame = orchestrator().perception.latest_forecast()
        return frame.assign(timestamp=frame["timestamp"].astype(str)).to_dict(orient="records")

    @app.get("/v1/anomalies")
    def anomalies(
        start: str | None = None,
        end: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        events = orchestrator().monitor(start, end)
        events = sorted(events, key=lambda x: x.anomaly_score, reverse=True)[:limit]
        return [x.model_dump(mode="json") for x in events]

    @app.get("/v1/diagnoses/{event_id}")
    def diagnose(event_id: str) -> dict[str, Any]:
        try:
            return orchestrator().diagnose_event(event_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/reports")
    def report(request: ReportRequest) -> dict[str, Any]:
        try:
            result = orchestrator().generate_report(
                request.start, request.end, request.max_diagnoses
            )
            return result.model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/qa")
    def qa(request: QARequest) -> dict[str, Any]:
        try:
            return orchestrator().ask(request.question, request.session_id).model_dump(mode="json")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/model/summary")
    def model_summary() -> dict[str, Any]:
        orch = orchestrator()
        path = orch.settings.resolve(orch.settings.paths.output_dir / "training_summary.json")
        assert path is not None
        return read_json(path, default={}) or {}

    return app


app = create_app()
