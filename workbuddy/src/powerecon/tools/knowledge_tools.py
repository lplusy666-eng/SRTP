"""知识类工具：知识图谱查询 + 知识文档检索。

这两个工具给诊断提供"领域先验"和"机制候选"，而不是让模型凭空想原因。

关于知识图谱的现状要说清楚：现有的两省图谱主要是**数据目录型**图谱
（MEASURES / OBSERVED_IN / AT_TIME 占绝大多数），
真正的机制先验边（AFFECTS / PROXIES_FOR / DISTURBS）只有个位数。
所以这个工具当前的主要价值是"告诉你这个地区有哪些指标、彼此什么结构关系"，
而机制候选主要来自知识文档和内置先验表。
补齐机制边是领域侧的活（对应 judgment_rules），不是工具层能解决的。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ..contract import (
    KnowledgeQueryResult,
    KnowledgeSearchResult,
    KnowledgeTriple,
    RetrievedDoc,
)
from .base import ToolContext, tool

CAUSAL_RELATIONS = {
    "AFFECTS“, ”MAY_AFFECT“, ”CANDIDATE_CAUSE“, ”DISTURBS“, ”PROXIES_FOR",
    "DERIVES_FEATURE“, ”INFLUENCES“, ”DRIVES",
}


class KgQueryArgs(BaseModel):
    subject_keyword: str | None = Field(
        default=None,
        description="主语节点名称的关键词，模糊匹配。例如 '用电'、'温度'、'GDP'。",
    )
    relation: str | None = Field(
        default=None,
        description="关系类型，精确匹配。常见值：MEASURES / OBSERVED_IN / AT_TIME / "
        "BELONGS_TO / AFFECTS / PROXIES_FOR / DERIVES_FEATURE。留空则不过滤。",
    )
    object_keyword: str | None = Field(default=None, description="宾语节点名称的关键词。")
    limit: int = Field(default=25, description="最多返回多少条三元组。")


@tool(
    name="query_knowledge_graph",
    description=(
        "在当前地区的知识图谱里查三元组（主语—关系—宾语）。\n"
        "用途一：搞清楚这个地区有哪些指标、它们之间是什么结构关系"
        "（例如“温度 AFFECTS 用电量”这种机制先验）。\n"
        "用途二：给异常找机制候选。查 relation='AFFECTS' 或 'PROXIES_FOR' 能看到"
        "哪些因素被标注为可能影响用电。\n"
        "**重要：图谱里的关系是机制先验或候选解释，不是已证明的因果。** "
        "查到 AFFECTS 只能说明“这个因素被列为候选”，不能说明“它就是原因”。\n"
        "如果返回 0 条，不要编造关系；改用 search_knowledge 去查知识文档。"
    ),
    args_model=KgQueryArgs,
    returns="KnowledgeQueryResult：匹配到的三元组 + 图谱可用关系类型清单",
    tags=("knowledge", "graph"),
)
def query_knowledge_graph(ctx: ToolContext, args: KgQueryArgs) -> BaseModel:
    rels = ctx.region.kg_relationships
    nodes = ctx.region.kg_nodes
    if rels is None or nodes is None:
        return KnowledgeQueryResult(
            query=args.model_dump(),
            n_matches=0,
            notes=["该地区没有知识图谱数据，无法提供机制先验。原因候选需依赖知识文档与常识推理。"],
        )

    names: dict[str, str] = {}
    for _, row in nodes.iterrows():
        nid = str(row.get("id") or "")
        if nid:
            names[nid] = str(row.get("name") or nid)

    def _kw_match(node_id: str, keyword: str | None) -> bool:
        if not keyword:
            return True
        return keyword.lower() in names.get(node_id, node_id).lower()

    matched: list[KnowledgeTriple] = []
    for _, row in rels.iterrows():
        sid, tid = str(row.get("source_id") or ""), str(row.get("target_id") or "")
        rtype = str(row.get("type") or "")
        if args.relation and rtype.upper() != args.relation.upper():
            continue
        if not _kw_match(sid, args.subject_keyword):
            continue
        if not _kw_match(tid, args.object_keyword):
            continue
        matched.append(
            KnowledgeTriple(
                subject_id=sid,
                subject_name=names.get(sid, sid),
                relation=rtype,
                object_id=tid,
                object_name=names.get(tid, tid),
                description=_clean(row.get("description")),
                direction=_clean(row.get("direction")),
                confidence=_clean(row.get("confidence")),
            )
        )

    available = sorted({str(t) for t in rels["type"].dropna().unique()})
    causal = [t for t in available if t.upper() in CAUSAL_RELATIONS]

    notes: list[str] = [
        f"图谱共有 {len(rels)} 条关系、{len(nodes)} 个节点，可用关系类型：{available}。",
    ]
    if causal:
        notes.append(
            f"其中机制类关系（可作为原因候选）：{causal}。"
            f"注意这些是机制先验，关系名本身不是因果断言。"
        )
    else:
        notes.append(
            "当前图谱**没有**机制类关系（AFFECTS / CANDIDATE_CAUSE 等），"
            "它只是数据目录。原因候选请走 search_knowledge。"
        )
    if not matched:
        notes.append("本次查询没有命中。可以放宽关键词，或先不带 subject/object 只看 relation。")

    return KnowledgeQueryResult(
        query=args.model_dump(),
        n_matches=len(matched),
        triples=matched[: args.limit],
        notes=notes,
    )


class SearchArgs(BaseModel):
    query: str = Field(description="检索问题或关键词，用自然语言写，例如 '春节错位对用电同比的影响'。")
    top_k: int = Field(default=4, description="返回前几条。")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens += ["".join(cjk[i : i + 2]) for i in range(max(0, len(cjk) - 1))]
    return tokens


def _bm25(query: str, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not docs:
        return []
    n = len(docs)
    lengths = [len(d) for d in docs]
    avgdl = sum(lengths) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))

    scores = [0.0] * n
    q_terms = _tokenize(query)
    for term in q_terms:
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(term)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * lengths[i] / avgdl)
            scores[i] += idf * tf * (k1 + 1) / denom
    return scores


@tool(
    name="search_knowledge",
    description=(
        "在本地知识文档里做关键词检索，返回最相关的片段。\n"
        "这是给异常找原因的主要知识来源 —— 当知识图谱里没有机制关系时，"
        "领域知识（春节错位、累计值口径、天气负荷、因果边界规则）都在这里。\n"
        "典型用法：先 detect_anomaly 找到异常期次，再用形如"
        "'2025年1月用电同比下降的可能原因' 的查询词调这个工具。\n"
        "返回的是文档原文片段，可以作为证据引用；但引用时必须注明来自知识文档而非数据。"
    ),
    args_model=SearchArgs,
    returns="KnowledgeSearchResult：相关片段 + 相似度分数 + 出处",
    tags=("knowledge", "rag"),
)
def search_knowledge(ctx: ToolContext, args: SearchArgs) -> BaseModel:
    docs = ctx.region.knowledge_docs
    if not docs:
        return KnowledgeSearchResult(
            query=args.query,
            backend="keyword-bm25",
            n_hits=0,
            notes=[
                "该地区的知识文档目录为空，无法检索到领域知识。"
                "此时不要凭常识编原因，应在结论里声明“缺少领域知识支持”。"
            ],
        )

    tokenized = [_tokenize(d["text"] + " " + d["title"]) for d in docs]
    scores = _bm25(args.query, tokenized)
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    hits: list[RetrievedDoc] = []
    for i in order[: args.top_k]:
        if scores[i] <= 0:
            continue
        hits.append(
            RetrievedDoc(
                doc_id=docs[i]["doc_id"],
                title=docs[i]["title"],
                snippet=_condense(docs[i]["text"]),
                score=round(float(scores[i]), 4),
                source=docs[i]["source"],
            )
        )

    notes = [f"知识库共 {len(docs)} 个片段，检索后端为 BM25 关键词匹配（零外部依赖）。"]
    if not hits:
        notes.append("没有命中任何片段，说明知识库里缺少相关领域知识，不要在结论里假装有依据。")

    return KnowledgeSearchResult(
        query=args.query,
        backend="keyword-bm25",
        n_hits=len(hits),
        hits=hits,
        notes=notes,
    )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _condense(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "…"
