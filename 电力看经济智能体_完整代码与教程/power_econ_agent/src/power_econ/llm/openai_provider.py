from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import LLMConfig
from ..utils import json_default
from .base import AnswerNarrative, DiagnosisNarrative, LLMProvider, ReportNarrative

T = TypeVar("T", bound=BaseModel)


class OpenAILLMProvider(LLMProvider):
    """OpenAI Responses API provider with schema-constrained structured outputs."""

    def __init__(self, config: LLMConfig):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("llm.provider=openai 但未设置 OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("未安装 OpenAI SDK，请执行 pip install '.[llm]'") from exc
        self.client = OpenAI()
        self.config = config

    def _parse(self, schema: type[T], system: str, payload: dict[str, Any]) -> T:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=json_default),
                },
            ],
            "text_format": schema,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        response = self.client.responses.parse(**kwargs)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI 返回中没有可解析的结构化输出")
        return parsed

    def diagnosis(self, payload: dict[str, Any]) -> DiagnosisNarrative:
        return self._parse(
            DiagnosisNarrative,
            """你是电力经济分析专家。只依据输入证据写中文诊断。必须区分相关性与因果，
先排除数据质量、天气、日历、政策等非经济扰动；不得编造输入中不存在的事实。
输出需明确经济信号、诊断叙述、不确定性和是否需要人工复核。""",
            payload,
        )

    def report(self, payload: dict[str, Any]) -> ReportNarrative:
        return self._parse(
            ReportNarrative,
            """你是面向政府、电网和研究人员的电力经济报告撰写专家。
只使用输入统计、诊断和知识片段，采用审慎、可审计的中文表述；
不得把模型相关性直接写成因果结论。输出完整结构化报告。""",
            payload,
        )

    def answer(self, payload: dict[str, Any]) -> AnswerNarrative:
        return self._parse(
            AnswerNarrative,
            """你是电力看经济问答智能体。回答必须以给定事件、诊断和知识片段为依据，
没有证据时直接说明不足；不得编造实时数据或已确认的因果关系。""",
            payload,
        )
