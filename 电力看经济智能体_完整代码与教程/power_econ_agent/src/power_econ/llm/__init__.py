from __future__ import annotations

from ..config import Settings
from .base import LLMProvider
from .template import TemplateLLMProvider


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm.provider == "openai":
        from .openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(settings.llm)
    return TemplateLLMProvider()


__all__ = ["build_llm_provider", "LLMProvider"]
