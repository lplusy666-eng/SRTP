from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    title: str
    text: str


class LocalKnowledgeBase:
    """Offline Chinese RAG using character n-gram TF-IDF.

    It intentionally has no external vector database dependency, which makes the
    demo deterministic and easy to audit. It can later be replaced with a dense
    embedding store without changing the agent interface.
    """

    def __init__(self, docs_dir: Path):
        self.docs_dir = Path(docs_dir)
        self.chunks = self._load_chunks()
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=1,
                max_features=30000,
                sublinear_tf=True,
            )
            self.matrix = self.vectorizer.fit_transform(
                [f"{c.title}\n{c.text}" for c in self.chunks]
            )

    def _load_chunks(self) -> list[KnowledgeChunk]:
        if not self.docs_dir.exists():
            return []
        chunks: list[KnowledgeChunk] = []
        for path in sorted(self.docs_dir.glob("**/*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            title = path.stem
            sections = re.split(r"\n(?=#{1,3}\s)|\n\s*\n", text)
            for section in sections:
                section = section.strip()
                if len(section) < 20:
                    continue
                lines = section.splitlines()
                local_title = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else title
                body = "\n".join(lines[1:] if lines[0].startswith("#") else lines).strip()
                if body:
                    chunks.append(
                        KnowledgeChunk(
                            source=path.name,
                            title=local_title or title,
                            text=body[:1800],
                        )
                    )
        return chunks

    def search(self, query: str, top_k: int = 4, min_score: float = 0.02) -> list[dict[str, object]]:
        if not query.strip() or not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).ravel()
        order = np.argsort(scores)[::-1]
        results: list[dict[str, object]] = []
        for idx in order[: max(top_k * 2, top_k)]:
            if float(scores[idx]) < min_score:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "source": chunk.source,
                    "title": chunk.title,
                    "text": chunk.text,
                    "score": float(scores[idx]),
                }
            )
            if len(results) >= top_k:
                break
        return results


__all__ = ["LocalKnowledgeBase", "KnowledgeChunk"]
