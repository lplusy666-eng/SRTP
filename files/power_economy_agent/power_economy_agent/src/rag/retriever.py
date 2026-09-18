"""
RAG 检索增强模块
================
- 文档切分 -> 向量化 -> 相似度检索
- 两种嵌入后端：
    hash   : 离线、零依赖（字符 n-gram 哈希 + TF-IDF 加权），保证无网络可跑
    openai : 调用 text-embedding 接口（需在 llm 段配置可用Key）
供诊断/生成智能体检索领域知识与历史案例。
"""
import os
import re
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir, load_config

log = get_logger("rag")

DIM = 512


def _tokenize(text: str):
    text = re.sub(r"\s+", "", text)
    # 中文按字 + bigram，英文按词
    toks = list(text)
    toks += [text[i:i + 2] for i in range(len(text) - 1)]
    return toks


def _hash_embed(text: str, dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokenize(text):
        h = hash(tok) % dim
        vec[h] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _split_docs(docs_dir: str, chunk_size: int = 220):
    chunks = []
    for p in sorted(Path(docs_dir).glob("*")):
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        text = p.read_text(encoding="utf-8")
        # 按段落切
        paras = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
        buf = ""
        for para in paras:
            if len(buf) + len(para) > chunk_size and buf:
                chunks.append({"source": p.name, "text": buf})
                buf = para
            else:
                buf = (buf + "\n" + para) if buf else para
        if buf:
            chunks.append({"source": p.name, "text": buf})
    return chunks


class RAGIndex:
    def __init__(self, backend="hash", cfg=None):
        self.backend = backend
        self.cfg = cfg
        self.chunks = []
        self.matrix = None

    def _embed(self, text):
        if self.backend == "openai":
            return self._openai_embed(text)
        return _hash_embed(text)

    def _openai_embed(self, text):
        from openai import OpenAI
        key = os.environ.get(self.cfg["llm"]["api_key_env"], "")
        client = OpenAI(base_url=self.cfg["llm"]["base_url"], api_key=key)
        r = client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(r.data[0].embedding, dtype=np.float32)

    def build(self, docs_dir):
        self.chunks = _split_docs(docs_dir)
        vecs = [self._embed(c["text"]) for c in self.chunks]
        self.matrix = np.vstack(vecs) if vecs else np.zeros((0, DIM))
        log.info("RAG 索引已建立: %d 块 (backend=%s)", len(self.chunks), self.backend)
        return self

    def save(self, path):
        ensure_dir(path)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "matrix": self.matrix,
                         "backend": self.backend}, f)

    @classmethod
    def load(cls, path, cfg=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = cls(backend=d["backend"], cfg=cfg)
        idx.chunks = d["chunks"]
        idx.matrix = d["matrix"]
        return idx

    def search(self, query, top_k=4):
        if self.matrix is None or len(self.chunks) == 0:
            return []
        q = self._embed(query)
        sims = self.matrix @ q / (np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(q) + 1e-8)
        order = np.argsort(-sims)[:top_k]
        return [{"score": float(sims[i]), **self.chunks[i]} for i in order]


def build_rag(cfg: dict) -> RAGIndex:
    idx = RAGIndex(backend=cfg["rag"]["embed_backend"], cfg=cfg)
    idx.build(abspath(cfg["rag"]["docs_dir"]))
    idx.save(abspath(cfg["rag"]["index_path"]))
    return idx


if __name__ == "__main__":
    cfg = load_config()
    idx = build_rag(cfg)
    for r in idx.search("春节用电为什么下降", top_k=3):
        print(f"[{r['score']:.3f}] {r['source']}: {r['text'][:40]}...")
    print("---")
    for r in idx.search("持续负偏离说明什么经济问题", top_k=3):
        print(f"[{r['score']:.3f}] {r['source']}: {r['text'][:40]}...")
