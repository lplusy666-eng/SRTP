"""
诊断智能体 (Diagnosis Agent) —— 对应研究目标2
==============================================
职责：基于知识增强(知识图谱 + RAG + 大模型)对每个异常事件做成因分析。
流程：异常特征 -> 图谱匹配候选成因 -> RAG检索佐证知识 -> 交叉核对宏观指标 -> LLM 综合归因。
"""
import networkx as nx
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger
from agents.base_agent import BaseAgent, Tool
from knowledge_graph.build_kg import query_causes, match_signature
from rag.retriever import RAGIndex

log = get_logger("agent.diagnosis")


class DiagnosisAgent(BaseAgent):
    def __init__(self, ctx, cfg, kg: nx.MultiDiGraph, rag: RAGIndex, llm):
        super().__init__("诊断智能体", "基于知识增强的成因分析", ctx)
        self.cfg = cfg
        self.kg = kg
        self.rag = rag
        self.llm = llm
        self.tools = {
            "查图谱": Tool("查图谱", self._kg_causes, "根据异常特征查候选成因"),
            "检索知识": Tool("检索知识", self._rag_search, "检索领域知识佐证"),
        }

    def _kg_causes(self, signature):
        return query_causes(self.kg, signature)

    def _rag_search(self, query):
        return self.rag.search(query, top_k=self.cfg["rag"]["top_k"])

    def _offline_diagnosis(self, system, user):
        # 无大模型时的结构化兜底归因
        return user  # user 本身已是拼装好的结构化诊断文本

    def diagnose_event(self, event):
        # 1) 图谱匹配特征
        signature = match_signature({"mean_deviation": event["mean_deviation"]})
        if event.get("near_spring_festival"):
            signature = "春节后复工偏慢" if event["mean_deviation"] < 0 else signature
        self.ctx.log_step(self.name, "图谱匹配", f"{event['start']}~{event['end']} -> {signature}")

        # 2) 图谱候选成因
        causes = self.use("查图谱", signature)

        # 3) RAG 检索佐证
        query = f"{event['direction']}偏离 {signature} 经济成因"
        evidence = self.use("检索知识", query)

        # 4) 交叉核对宏观指标
        macro_note = self._macro_crosscheck(event)

        # 5) LLM 综合归因
        cause_txt = "; ".join([f"{c['cause']}(佐证:{'/'.join(c['corroborated_by']) or '—'})"
                               for c in causes]) or "无匹配成因"
        evid_txt = "\n".join([f"- [{e['source']}] {e['text'][:80]}" for e in evidence])

        system = ("你是电力经济分析专家。基于给定的异常事实、知识图谱候选成因、检索到的领域知识和宏观指标，"
                  "给出审慎的成因判断，区分事实与推断，并给出置信度(高/中/低)。语言简洁专业。")
        user = (
            f"【异常事件】{event['start']} ~ {event['end']}，方向：{event['direction']}，"
            f"幅度约 {event['magnitude_pct']}%，持续 {event['duration_days']} 天，"
            f"是否临近春节：{event['near_spring_festival']}。\n"
            f"【特征标签】{signature}\n"
            f"【图谱候选成因】{cause_txt}\n"
            f"【宏观交叉核对】{macro_note}\n"
            f"【检索知识】\n{evid_txt}\n"
            f"请输出：①最可能成因 ②佐证 ③置信度 ④建议进一步核实的指标。"
        )
        diagnosis = self.llm.chat(system, user, offline_fn=lambda s, u: self._format_offline(
            event, signature, causes, evidence, macro_note))
        return {
            "event": event,
            "signature": signature,
            "kg_causes": causes,
            "evidence": [{"source": e["source"], "score": e["score"]} for e in evidence],
            "macro_note": macro_note,
            "diagnosis": diagnosis,
        }

    def _macro_crosscheck(self, event):
        pmi = event.get("avg_pmi")
        notes = []
        if pmi is not None:
            if pmi < 50 and event["mean_deviation"] < 0:
                notes.append(f"PMI={pmi}<50 且用电负偏离，二者互相印证需求走弱")
            elif pmi >= 50 and event["mean_deviation"] > 0:
                notes.append(f"PMI={pmi}≥50 且用电正偏离，印证景气扩张")
            else:
                notes.append(f"PMI={pmi} 与用电方向不完全一致，需谨慎")
        return "；".join(notes) or "无可用宏观指标"

    def _format_offline(self, event, signature, causes, evidence, macro_note):
        top = causes[0]["cause"] if causes else "待定"
        conf = "中"
        if event.get("near_spring_festival"):
            conf = "低（受春节效应干扰，建议人工复核）"
        elif abs(event["magnitude_pct"]) > 8:
            conf = "高"
        lines = [
            f"① 最可能成因：{top}",
            f"② 佐证：图谱关联 {len(causes)} 项候选；" + macro_note,
            f"③ 置信度：{conf}",
            f"④ 建议核实：" + ("、".join(causes[0]["corroborated_by"]) if causes and causes[0]["corroborated_by"] else "PMI、固定资产投资、社零、进出口"),
        ]
        return "\n".join(lines)

    def run(self):
        events = self.ctx.get("anomaly_events", [])
        self.ctx.log_step(self.name, "开始诊断", f"共 {len(events)} 个事件")
        diagnoses = [self.diagnose_event(e) for e in events]
        self.ctx.set("diagnoses", diagnoses)
        self.ctx.log_step(self.name, "诊断完成")
        return diagnoses
