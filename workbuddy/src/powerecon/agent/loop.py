"""智能体主循环：plan → tool call → observe → 再规划。

这是整个系统与"固定流水线"的分界线。
循环本身不认识任何具体工具，也不知道分析该分几步 —— 它只做四件事：
问规划器下一步、执行、把结果塞回状态、重复，直到规划器说结束或触发终止条件。

终止条件有三个，缺一不可：
1. 规划器主动结束（tool=None）；
2. submit_conclusion 成功返回（拿到通过校验的结论）；
3. 步数达到上限（防止规划器死循环）。

如果循环结束时还没拿到结论，会合成一个"诚实的失败结论"，
说明为什么没做成、已经观察到什么。绝不返回半成品当结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..contract import Conclusion
from ..tools.base import ToolContext, dispatch
from .planner import AgentState, PlanDecision, Planner, synthesize_conclusion
from .trace import Trace


@dataclass
class RunResult:
    ok: bool
    conclusion: Conclusion | None
    question: str
    region: str
    n_steps: int
    tool_sequence: list[str] = field(default_factory=list)
    trace_path: str | None = None
    error: str | None = None
    planner: str = ""

    def summary(self) -> str:
        head = "成功" if self.ok else "未完成"
        return (
            f"[{head}] 地区={self.region} 步数={self.n_steps} "
            f"规划器={self.planner} 工具链={' → '.join(self.tool_sequence) or '（无）'}"
        )


class AgentLoop:
    def __init__(
        self,
        ctx: ToolContext,
        planner: Planner,
        *,
        max_steps: int = 12,
        trace_dir: Path | None = None,
    ) -> None:
        self.ctx = ctx
        self.planner = planner
        self.max_steps = max_steps
        self.trace_dir = Path(trace_dir) if trace_dir else None

    def run(self, question: str) -> RunResult:
        state = AgentState(
            question=question,
            region_name=self.ctx.region.info.name,
            max_steps=self.max_steps,
        )
        self.ctx.cache_set("question", question)

        trace_path = None
        if self.trace_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_path = self.trace_dir / f"run_{stamp}.jsonl"
        trace = Trace(
            trace_path,
            meta={
                "question": question,
                "region": self.ctx.region.info.name,
                "region_code": self.ctx.region.info.code,
                "planner": getattr(self.planner, "name", type(self.planner).__name__),
                "max_steps": self.max_steps,
            },
        )

        error: str | None = None
        try:
            for step in range(self.max_steps):
                decision: PlanDecision = self.planner.plan(state)
                trace.add(
                    step=step,
                    phase="plan",
                    tool=decision.tool,
                    arguments=decision.arguments,
                    thought=decision.thought,
                )
                if decision.finish:
                    break

                # submit_conclusion 需要知道调用轨迹，所以在 dispatch 前更新
                self.ctx.cache_set("tool_calls", [o.tool for o in state.observations])

                result = dispatch(decision.tool or "", decision.arguments, self.ctx)
                state.observations.append(result)
                trace.add(
                    step=step,
                    phase="act",
                    tool=result.tool,
                    arguments=result.arguments,
                    ok=result.ok,
                    duration_ms=result.duration_ms,
                    observation=result.text[:1500] if result.text else None,
                )

                if result.ok and isinstance(result.payload, Conclusion):
                    state.conclusion = result.payload
                    trace.add(step=step, phase="finish", thought="已获得通过校验的五段式结论。")
                    break
        except Exception as exc:  # 循环本身出错也要留下痕迹，而不是静默失败
            error = f"{type(exc).__name__}: {exc}"
            trace.add(step=state.steps, phase="error", thought=error)

        if state.conclusion is None:
            state.conclusion = self._fallback_conclusion(state, error)

        trace.finish(
            ok=state.conclusion is not None,
            n_steps=state.steps,
            tool_sequence=state.tool_calls,
            confidence=state.conclusion.confidence if state.conclusion else None,
            causal_status=state.conclusion.causal_status.value if state.conclusion else None,
        )

        return RunResult(
            ok=state.conclusion is not None and error is None,
            conclusion=state.conclusion,
            question=question,
            region=self.ctx.region.info.name,
            n_steps=state.steps,
            tool_sequence=state.tool_calls,
            trace_path=str(trace_path) if trace_path else None,
            error=error,
            planner=getattr(self.planner, "name", type(self.planner).__name__),
        )

    def _fallback_conclusion(self, state: AgentState, error: str | None) -> Conclusion:
        """没拿到结论时的兜底：产出一个明确说明"没做成"的五段式结论。

        这里刻意不编造分析结果 —— 一个诚实的"未完成"比一个看起来完整的假结论有用得多。
        """
        args = synthesize_conclusion(
            state,
            error or "循环在达到步数上限前未提交结论。",
        )
        from ..contract import CausalStatus, Evidence

        return Conclusion(
            region=self.ctx.region.info.name,
            question=state.question,
            conclusion=args["conclusion"],
            causal_status=args["causal_status"],
            evidence=[
                Evidence(
                    claim=e["claim"],
                    tool=e["tool"],
                    source=e.get("source"),
                    value=e.get("value"),
                    period=e.get("period"),
                )
                for e in args["evidence"]
            ],
            counter_evidence=args["counter_evidence"],
            confidence=args["confidence"],
            uncertainty=args["uncertainty"] or ["流程未完成，结论不完整。"],
            caveats=list(self.ctx.region.info.caveats),
            tool_calls=list(state.tool_calls),
        )
