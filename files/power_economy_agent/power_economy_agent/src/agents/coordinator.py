"""
多智能体协同器 (Coordinator) —— 对应研究目标4
==============================================
用 MCP 风格的共享上下文 + 顺序编排，把三个智能体组成闭环：
    感知(数据→解耦→检测) → 诊断(知识增强归因) → 生成(报告+问答)
并保留完整协同留痕(trace)，作为"可信结果输出"的依据。
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import get_logger, load_config, abspath
from agents.base_agent import SharedContext
from agents.llm_client import LLMClient
from agents.perception_agent import PerceptionAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.generation_agent import GenerationAgent
from knowledge_graph.build_kg import build_kg
from rag.retriever import build_rag

log = get_logger("coordinator")


class PowerEconomyAgentSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ctx = SharedContext()
        self.llm = LLMClient(cfg)
        # 构建知识底座
        self.kg = build_kg(cfg)
        self.rag = build_rag(cfg)
        # 装配智能体
        self.perception = PerceptionAgent(self.ctx, cfg)
        self.diagnosis = DiagnosisAgent(self.ctx, cfg, self.kg, self.rag, self.llm)
        self.generation = GenerationAgent(self.ctx, cfg, self.rag, self.llm)

    def run_pipeline(self, rebuild=True):
        log.info("========== 启动“电力看经济”智能体闭环 ==========")
        self.perception.run(rebuild=rebuild)
        self.diagnosis.run()
        out, report = self.generation.run()
        log.info("========== 闭环完成，报告：%s ==========", out)
        return out, report

    def ask(self, question):
        return self.generation.answer(question)

    def save_trace(self, path=None):
        import json
        path = path or abspath("outputs/agent_trace.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ctx.trace, f, ensure_ascii=False, indent=2)
        return path


if __name__ == "__main__":
    cfg = load_config()
    system = PowerEconomyAgentSystem(cfg)
    out, report = system.run_pipeline(rebuild=True)
    system.save_trace()
    print("\n" + "=" * 60)
    print(report[:1500])
    print("...\n[问答测试]")
    for q in ["观测期内共检测到多少起异常？", "幅度最大的异常是哪次？"]:
        print(f"Q: {q}\nA: {system.ask(q)}\n")
