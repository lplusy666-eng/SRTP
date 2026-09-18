from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from .agents import PowerEconomyOrchestrator


@lru_cache(maxsize=1)
def _orchestrator() -> PowerEconomyOrchestrator:
    return PowerEconomyOrchestrator.from_config(
        os.getenv("POWER_ECON_CONFIG", "configs/demo.yaml")
    )


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("未安装 MCP SDK，请执行 pip install '.[mcp]'") from exc

    mcp = FastMCP("power-economy-agent")

    @mcp.tool()
    def list_anomalies(
        start: str | None = None,
        end: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """监测给定时间段的电力负荷异常并返回代表性事件。"""
        events = _orchestrator().monitor(start, end)
        events = sorted(events, key=lambda x: x.anomaly_score, reverse=True)[:limit]
        return [e.model_dump(mode="json") for e in events]

    @mcp.tool()
    def diagnose_event(event_id: str) -> dict[str, Any]:
        """对指定异常事件执行特征消融、知识图谱归因和知识增强诊断。"""
        return _orchestrator().diagnose_event(event_id).model_dump(mode="json")

    @mcp.tool()
    def generate_power_economy_report(
        start: str,
        end: str,
        max_diagnoses: int = 8,
    ) -> dict[str, Any]:
        """生成给定报告期的电力经济态势报告。"""
        return _orchestrator().generate_report(start, end, max_diagnoses).model_dump(mode="json")

    @mcp.tool()
    def ask_power_economy(question: str, session_id: str = "mcp") -> dict[str, Any]:
        """基于事件数据库和本地知识库回答电力经济问题。"""
        return _orchestrator().ask(question, session_id).model_dump(mode="json")

    @mcp.tool()
    def latest_load_forecast() -> list[dict[str, Any]]:
        """返回最新的多步负荷分位数预测。"""
        frame = _orchestrator().perception.latest_forecast()
        frame["timestamp"] = frame["timestamp"].astype(str)
        return frame.to_dict(orient="records")

    return mcp


def main() -> None:
    mcp = build_server()
    transport = os.getenv("POWER_ECON_MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
