"""
生成智能体 (Generation Agent) —— 对应研究目标3
==============================================
职责：自动报告生成 + 交互问答。
报告：汇总观测期概况、异常清单、逐事件诊断、总体研判、风险提示。
问答：结合共享上下文(异常/诊断) + RAG 检索回答用户问题。
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir
from agents.base_agent import BaseAgent

log = get_logger("agent.generation")


class GenerationAgent(BaseAgent):
    def __init__(self, ctx, cfg, rag, llm):
        super().__init__("生成智能体", "报告生成与问答交互", ctx)
        self.cfg = cfg
        self.rag = rag
        self.llm = llm

    # ---------- 报告生成 ----------
    def generate_report(self):
        events = self.ctx.get("anomaly_events", [])
        diagnoses = self.ctx.get("diagnoses", [])
        region = self.cfg["project"]["region"]
        start = self.cfg["project"]["start_date"]
        end = self.cfg["project"]["end_date"]

        lines = [f"# {region}“电力看经济”分析报告", ""]
        lines += [f"**观测期**：{start} ~ {end}　|　**检测到异常事件**：{len(events)} 起", ""]
        lines += ["> 本报告由“电力看经济”智能体自动生成：多源数据 → 特征解耦 → VAE异常检测 → 知识增强诊断。", ""]

        lines += ["## 一、异常事件清单", ""]
        lines += ["| 时间区间 | 方向 | 幅度 | 持续 | 临近春节 | 平均PMI |", "|---|---|---|---|---|---|"]
        for e in events:
            lines.append(f"| {e['start']}~{e['end']} | {e['direction']} | {e['magnitude_pct']}% | "
                         f"{e['duration_days']}天 | {'是' if e['near_spring_festival'] else '否'} | {e.get('avg_pmi','—')} |")
        lines.append("")

        lines += ["## 二、逐事件成因诊断", ""]
        for i, d in enumerate(diagnoses, 1):
            e = d["event"]
            lines += [f"### 事件 {i}：{e['start']} ~ {e['end']}（{e['direction']} {e['magnitude_pct']}%）", ""]
            lines += [f"- **特征标签**：{d['signature']}"]
            lines += [f"- **诊断结论**："]
            for ln in d["diagnosis"].split("\n"):
                lines.append(f"  {ln}")
            srcs = "、".join(sorted({ev["source"] for ev in d["evidence"]})) or "—"
            lines += [f"- **知识来源**：{srcs}", ""]

        # 总体研判（可由LLM生成）
        overview = self._overview(events, diagnoses)
        lines += ["## 三、总体经济态势研判", "", overview, ""]
        lines += ["## 四、风险提示", "",
                  "- 电力单一数据源存在局限，重要结论需结合PMI、投资、消费、进出口等交叉验证。",
                  "- 临近春节的异常受节假日效应干扰较大，置信度偏低，建议人工复核。", ""]

        report = "\n".join(lines)
        out = abspath(self.cfg["report"]["output_md"])
        ensure_dir(out)
        Path(out).write_text(report, encoding="utf-8")
        self.ctx.log_step(self.name, "报告已生成", out)
        return out, report

    def _overview(self, events, diagnoses):
        n_up = sum(1 for e in events if e["direction"] == "上升")
        n_down = len(events) - n_up
        system = ("你是宏观经济分析师，请基于异常事件统计给出一段审慎的总体经济态势研判，150字以内。")
        user = (f"观测期内检出 {len(events)} 起用电异常，其中正偏离(景气偏强){n_up}起，"
                f"负偏离(景气偏弱){n_down}起。请概述总体态势与倾向，并提示不确定性。")
        return self.llm.chat(system, user, offline_fn=lambda s, u: (
            f"观测期内共检出 {len(events)} 起显著用电异常，正偏离 {n_up} 起、负偏离 {n_down} 起。"
            f"整体经济态势{'偏积极' if n_up >= n_down else '存在下行压力'}，"
            f"但样本有限且部分事件受节假日干扰，结论需结合宏观指标进一步确认。"))

    # ---------- 交互问答 ----------
    def answer(self, question: str):
        events = self.ctx.get("anomaly_events", [])
        diagnoses = self.ctx.get("diagnoses", [])
        evid = self.rag.search(question, top_k=self.cfg["rag"]["top_k"])
        context = []
        for d in diagnoses[:6]:
            e = d["event"]
            context.append(f"{e['start']}~{e['end']} {e['direction']}{e['magnitude_pct']}% 判断:{d['diagnosis'].splitlines()[0] if d['diagnosis'] else ''}")
        ctx_txt = "\n".join(context)
        evid_txt = "\n".join([f"- [{x['source']}] {x['text'][:80]}" for x in evid])
        system = ("你是电力经济分析问答助手。仅依据提供的异常事件、诊断结论与检索知识作答，"
                  "不臆造数据；无法回答时说明原因。")
        user = f"【问题】{question}\n【已知异常与诊断】\n{ctx_txt}\n【检索知识】\n{evid_txt}"
        return self.llm.chat(system, user, offline_fn=lambda s, u: self._offline_answer(question, events, diagnoses, evid))

    def _offline_answer(self, question, events, diagnoses, evid):
        if not events:
            return "当前未检测到显著异常事件。"
        # 简单规则问答兜底
        if "几" in question or "多少" in question:
            return f"观测期内共检测到 {len(events)} 起显著用电异常事件。"
        if "最" in question and ("严重" in question or "大" in question):
            worst = max(events, key=lambda e: abs(e["magnitude_pct"]))
            return f"幅度最大的异常为 {worst['start']}~{worst['end']}，{worst['direction']} {abs(worst['magnitude_pct'])}%。"
        head = evid[0]["text"][:120] if evid else "（无检索到相关知识）"
        return f"根据领域知识：{head}\n（离线模板作答；启用大模型可获得更完整的分析。）"

    def run(self):
        return self.generate_report()
