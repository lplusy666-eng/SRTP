from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

from ..schemas import CauseEvidence, FeatureContribution

LOGGER = logging.getLogger(__name__)


class DomainKnowledgeGraph:
    """A compact graph plus executable weighted rules for diagnosis ranking."""

    def __init__(self, yaml_path: Path):
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"知识图谱规则文件不存在: {self.yaml_path}")
        raw = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}
        self.raw = raw
        self.graph = nx.MultiDiGraph()
        for node in raw.get("nodes", []):
            node = dict(node)
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        for edge in raw.get("edges", []):
            self.graph.add_edge(edge["source"], edge["target"], relation=edge.get("relation", "related"))
        self.rules = list(raw.get("rules", []))

    @staticmethod
    def _matches(actual: Any, op: str, expected: Any) -> bool:
        if actual is None:
            return False
        try:
            a = float(actual)
            e = float(expected)
        except (TypeError, ValueError):
            a, e = actual, expected
        if op == "ge":
            return a >= e
        if op == "gt":
            return a > e
        if op == "le":
            return a <= e
        if op == "lt":
            return a < e
        if op == "eq":
            return a == e
        if op == "ne":
            return a != e
        if op == "abs_ge":
            return abs(float(a)) >= float(e)
        if op == "between":
            low, high = expected
            return float(low) <= float(a) <= float(high)
        raise ValueError(f"未知规则运算符: {op}")

    def related_nodes(self, cause_id: str) -> list[str]:
        if cause_id not in self.graph:
            return []
        related: list[str] = []
        for target in self.graph.successors(cause_id):
            label = self.graph.nodes[target].get("label", target)
            relation_values = [d.get("relation", "related") for d in self.graph.get_edge_data(cause_id, target).values()]
            related.append(f"{label}（{'/'.join(relation_values)}）")
        return related

    def rank_causes(
        self,
        context: dict[str, Any],
        contributions: list[FeatureContribution],
        top_k: int = 5,
    ) -> list[CauseEvidence]:
        importance = {c.group: c.normalized_importance for c in contributions}
        enriched = dict(context)
        for group, value in importance.items():
            enriched[f"importance_{group}"] = value

        ranked: list[CauseEvidence] = []
        for rule in self.rules:
            score = float(rule.get("base_score", 0.0))
            possible = score + sum(float(c.get("weight", 0.0)) for c in rule.get("conditions", []))
            evidence: list[str] = []
            counter: list[str] = []
            matched_weight = 0.0
            for condition in rule.get("conditions", []):
                feature = condition["feature"]
                actual = enriched.get(feature)
                matched = self._matches(actual, condition.get("op", "eq"), condition.get("value"))
                if matched:
                    weight = float(condition.get("weight", 0.0))
                    score += weight
                    matched_weight += weight
                    template = condition.get("evidence", feature)
                    try:
                        evidence.append(template.format(actual=float(actual)))
                    except (TypeError, ValueError):
                        evidence.append(template.format(actual=actual))
                elif actual is not None and condition.get("counter_evidence"):
                    counter.append(str(condition["counter_evidence"]).format(actual=actual))
            min_score = float(rule.get("min_score", 0.0))
            if score < min_score:
                continue
            confidence = float(np.clip(score / max(possible, 1e-6), 0.0, 1.0))
            # A rule with only its base score is never surfaced.
            if matched_weight <= 0:
                continue
            related = list(rule.get("related_nodes", []))
            related_labels = self.related_nodes(rule["id"])
            related.extend(x for x in related_labels if x not in related)
            ranked.append(
                CauseEvidence(
                    cause_id=rule["id"],
                    label=rule["label"],
                    category=rule.get("category", "other"),
                    score=float(score),
                    confidence=confidence,
                    evidence=evidence,
                    counter_evidence=counter,
                    related_nodes=related,
                )
            )
        ranked.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        return ranked[:top_k]


__all__ = ["DomainKnowledgeGraph"]
