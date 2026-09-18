"""决策留痕。

留痕是"可信输出"的依据：结论里的每一句话都应该能回放到是哪一步工具调用支撑的。
所以这里记录的不只是"调了什么工具"，还包括模型的思考、工具返回的摘要、
以及失败重试。

落盘用 JSONL，一行一步，追加写。这样即使中途崩了，也能看到崩在哪一步。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TraceStep:
    step: int
    phase: str  # plan / act / observe / error / finish
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    thought: str | None = None
    ok: bool | None = None
    duration_ms: int | None = None
    observation: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "phase": self.phase,
            "tool": self.tool,
            "arguments": self.arguments,
            "thought": self.thought,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "observation": self.observation,
            "timestamp": self.timestamp,
        }


class Trace:
    """一次运行的完整决策记录。"""

    def __init__(self, path: Path | None = None, meta: dict[str, Any] | None = None) -> None:
        self.path = Path(path) if path else None
        self.meta = meta or {}
        self.steps: list[TraceStep] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._append({"type": "run_start", **self.meta, "timestamp": datetime.now().isoformat(timespec="seconds")})

    def add(self, **kwargs: Any) -> TraceStep:
        step = TraceStep(**kwargs)
        self.steps.append(step)
        self._append({"type": "step", **step.to_dict()})
        return step

    def finish(self, **payload: Any) -> None:
        self._append({"type": "run_end", **payload, "timestamp": datetime.now().isoformat(timespec="seconds")})

    def _append(self, record: dict[str, Any]) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def replay(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.steps]

    @property
    def tool_call_count(self) -> int:
        return sum(1 for s in self.steps if s.phase == "act")
