from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from power_econ.agents import PowerEconomyOrchestrator
from power_econ.api.app import create_app
from power_econ.api.runtime import clear_runtime_cache
from power_econ.utils import json_default, write_json


def timed(results: dict[str, Any], name: str, fn):
    started = time.perf_counter()
    value = fn()
    results[name] = {
        "status": "pass",
        "duration_seconds": round(time.perf_counter() - started, 4),
    }
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="验收电力看经济智能体全部主要链路")
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()

    config = str(Path(args.config).expanduser().resolve())
    os.environ["POWER_ECON_CONFIG"] = config
    results: dict[str, Any] = {"config": config, "checks": {}}
    checks = results["checks"]

    orch = timed(checks, "load_orchestrator", lambda: PowerEconomyOrchestrator.from_config(config))
    frame = orch.frame
    assert len(frame) > orch.engine.lookback

    split = orch.engine.metadata.get("split", {})
    test_start_index = int(split.get("val_end", max(orch.engine.lookback, len(frame) - 24 * 30)))
    test_start_index = min(max(test_start_index, orch.engine.lookback), len(frame) - 1)
    test_start = str(pd.Timestamp(frame["timestamp"].iloc[test_start_index]))
    test_end = str(pd.Timestamp(frame["timestamp"].iloc[-1]))

    forecast = timed(checks, "latest_forecast", orch.perception.latest_forecast)
    assert len(forecast) == orch.engine.horizon
    checks["latest_forecast"]["rows"] = len(forecast)

    events = timed(checks, "anomaly_monitor", lambda: orch.monitor(test_start, test_end))
    checks["anomaly_monitor"]["events"] = len(events)

    if events:
        event = max(events, key=lambda x: x.anomaly_score)
    else:
        scored = orch.perception.scores(test_start, test_end)
        assert not scored.empty
        row = scored.sort_values("anomaly_score", ascending=False).iloc[0]
        event = orch.engine._row_to_event(row)
        orch.store.upsert_anomaly(event.event_id, str(event.timestamp), event.model_dump(mode="json"))
        checks["anomaly_monitor"]["warning"] = (
            "该数据段无超过阈值事件；为验证诊断链路，使用最高分时点构造验收事件。"
        )

    diagnosis = timed(checks, "diagnosis", lambda: orch.diagnose_event(event.event_id))
    assert diagnosis.top_causes and diagnosis.narrative
    checks["diagnosis"]["event_id"] = event.event_id
    checks["diagnosis"]["top_cause"] = diagnosis.top_causes[0].label

    report_end = pd.Timestamp(frame["timestamp"].iloc[-1])
    report_start = report_end - pd.Timedelta(days=30)
    report = timed(
        checks,
        "report",
        lambda: orch.generate_report(str(report_start), str(report_end), max_diagnoses=5),
    )
    assert report.executive_summary
    report_path = orch.settings.resolve(orch.settings.paths.report_dir / f"{report.report_id}.md")
    assert report_path is not None and report_path.exists()
    checks["report"]["report_id"] = report.report_id
    checks["report"]["path"] = str(report_path)

    answer = timed(
        checks,
        "qa",
        lambda: orch.ask("最近异常的主要原因是什么，能否直接说明经济走弱？", "verify-all"),
    )
    assert answer.answer and 0 <= answer.confidence <= 1
    checks["qa"]["confidence"] = answer.confidence

    def api_checks() -> dict[str, Any]:
        from fastapi.testclient import TestClient

        clear_runtime_cache()
        with TestClient(create_app()) as client:
            health = client.get("/health")
            assert health.status_code == 200 and health.json().get("model_ready") is True
            latest = client.get("/v1/forecast/latest")
            assert latest.status_code == 200 and len(latest.json()) == orch.engine.horizon
            model = client.get("/v1/model/summary")
            assert model.status_code == 200 and "forecaster_metrics" in model.json()
            anomaly = client.get(
                "/v1/anomalies",
                params={"start": test_start, "end": test_end, "limit": 3},
            )
            assert anomaly.status_code == 200
            diagnose = client.get(f"/v1/diagnoses/{event.event_id}")
            assert diagnose.status_code == 200
            qa = client.post(
                "/v1/qa",
                json={"question": "系统如何排除天气扰动？", "session_id": "api-verify"},
            )
            assert qa.status_code == 200 and qa.json().get("answer")
            return {
                "health": health.json(),
                "forecast_rows": len(latest.json()),
                "anomaly_rows": len(anomaly.json()),
            }

    api_result = timed(checks, "fastapi", api_checks)
    checks["fastapi"].update(api_result)

    results["status"] = "pass"
    results["summary"] = {
        "rows": len(frame),
        "feature_count": len(orch.engine.feature_columns),
        "group_count": len(orch.engine.group_names),
        "forecast_horizon": orch.engine.horizon,
        "test_period": [test_start, test_end],
        "representative_events": len(events),
        "diagnosed_event_id": event.event_id,
        "report_id": report.report_id,
    }
    output = orch.settings.resolve(orch.settings.paths.output_dir / "verification_summary.json")
    assert output is not None
    write_json(output, results)
    print(json.dumps(results, ensure_ascii=False, indent=2, default=json_default))
    print(f"\n全部主要链路通过。验收结果：{output}")


if __name__ == "__main__":
    main()
