from __future__ import annotations

from datetime import UTC, datetime

from ..knowledge import LocalKnowledgeBase
from ..llm.base import LLMProvider
from ..schemas import QAResponse
from ..storage import EventStore


class QAAgent:
    """Question-answering agent over local knowledge and persisted analysis records."""

    def __init__(self, knowledge_base: LocalKnowledgeBase, llm: LLMProvider, store: EventStore):
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.store = store

    def ask(self, question: str, session_id: str = "default") -> QAResponse:
        if not question.strip():
            raise ValueError("问题不能为空")
        self.store.add_message(session_id, "user", question)
        snippets = self.knowledge_base.search(question, top_k=5)
        events = self.store.list_anomalies(limit=10)
        diagnoses = self.store.list_diagnoses(limit=10)
        history = self.store.get_messages(session_id, limit=8)
        generated = self.llm.answer(
            {
                "question": question,
                "snippets": snippets,
                "events": events,
                "diagnoses": diagnoses,
                "conversation_history": history,
            }
        )
        response = QAResponse(
            answer=generated.answer,
            evidence=generated.evidence,
            referenced_event_ids=[str(e.get("event_id")) for e in events[:5] if e.get("event_id")],
            confidence=generated.confidence,
            generated_at=datetime.now(UTC),
        )
        self.store.add_message(session_id, "assistant", response.answer)
        return response
