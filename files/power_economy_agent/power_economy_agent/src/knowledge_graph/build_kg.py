"""
电力-经济知识图谱构建
======================
编码领域知识：经济部门、扰动因素、异常特征 → 成因。
诊断智能体据此把"检测到的异常"关联到"可能的经济/非经济原因"。

图谱可从 configs 指定的 seed_csv 扩展；默认内置一套基础知识。
节点类型：Sector(经济部门) / Disturbance(扰动) / Signature(异常特征) / Cause(成因)
边类型：affects / explained_by / indicates
"""
import networkx as nx
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir, load_config

log = get_logger("kg")

# 基础领域知识（可按地区/行业扩展）
SEED_TRIPLES = [
    # (头实体, 关系, 尾实体, 类型标注)
    ("制造业", "affects", "工业用电", "Sector-Load"),
    ("房地产", "affects", "建材冶金用电", "Sector-Load"),
    ("居民消费", "affects", "商业与居民用电", "Sector-Load"),
    ("出口", "affects", "外向型制造用电", "Sector-Load"),
    ("新能源发电", "affects", "净负荷波动", "Sector-Load"),

    # 扰动因素 → 用电特征
    ("高温天气", "indicates", "制冷负荷骤升", "Disturbance-Sig"),
    ("寒潮", "indicates", "采暖负荷骤升", "Disturbance-Sig"),
    ("春节假期", "indicates", "工业用电断崖式下降", "Disturbance-Sig"),
    ("国庆假期", "indicates", "用电小幅回落", "Disturbance-Sig"),
    ("电价调整", "indicates", "用电时段结构变化", "Disturbance-Sig"),
    ("限电限产政策", "indicates", "工业用电阶段性下降", "Disturbance-Sig"),

    # 异常特征 → 可能成因（诊断核心）
    ("持续负偏离_非节假日", "explained_by", "制造业订单下滑/需求走弱", "Sig-Cause"),
    ("持续负偏离_非节假日", "explained_by", "行业限产或环保督察", "Sig-Cause"),
    ("持续正偏离_非节假日", "explained_by", "重大项目投产/产能扩张", "Sig-Cause"),
    ("持续正偏离_非节假日", "explained_by", "促消费政策/经济回暖", "Sig-Cause"),
    ("春节后复工偏慢", "explained_by", "外需疲弱/企业信心不足", "Sig-Cause"),
    ("高频剧烈波动", "explained_by", "新能源大发或极端天气", "Sig-Cause"),

    # 成因 → 佐证指标（供交叉验证）
    ("制造业订单下滑/需求走弱", "corroborated_by", "PMI下行", "Cause-Indicator"),
    ("重大项目投产/产能扩张", "corroborated_by", "固定资产投资上升", "Cause-Indicator"),
    ("促消费政策/经济回暖", "corroborated_by", "社会消费品零售总额回升", "Cause-Indicator"),
    ("行业限产或环保督察", "corroborated_by", "行业开工率下降", "Cause-Indicator"),
]


def build_kg(cfg: dict) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    triples = list(SEED_TRIPLES)

    # 合并用户自定义种子
    seed_csv = abspath(cfg["knowledge_graph"].get("seed_csv", ""))
    if seed_csv and Path(seed_csv).exists():
        extra = pd.read_csv(seed_csv)
        for _, r in extra.iterrows():
            triples.append((r["head"], r["relation"], r["tail"], r.get("type", "custom")))
        log.info("已合并自定义知识 %d 条", len(extra))

    for h, rel, t, typ in triples:
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, relation=rel, type=typ)

    ensure_dir(abspath(cfg["knowledge_graph"]["graph_path"]))
    nx.write_graphml(G, abspath(cfg["knowledge_graph"]["graph_path"]))
    log.info("知识图谱已构建: %d 节点, %d 边 -> %s",
             G.number_of_nodes(), G.number_of_edges(),
             abspath(cfg["knowledge_graph"]["graph_path"]))
    return G


def query_causes(G: nx.MultiDiGraph, signature: str, top_k: int = 4):
    """给定异常特征，返回可能成因及其佐证指标。"""
    causes = []
    for _, cause, data in G.out_edges(signature, data=True):
        if data.get("relation") == "explained_by":
            corrob = [c for _, c, d in G.out_edges(cause, data=True)
                      if d.get("relation") == "corroborated_by"]
            causes.append({"cause": cause, "corroborated_by": corrob})
    return causes[:top_k]


def match_signature(anomaly: dict) -> str:
    """把检测到的异常映射到图谱中的特征标签（简单规则；可换成分类器）。"""
    # anomaly: {mean_deviation, in_holiday, high_freq_energy, ...}
    dev = anomaly.get("mean_deviation", 0)
    if anomaly.get("high_freq_energy", 0) > anomaly.get("hf_threshold", 1e9):
        return "高频剧烈波动"
    if dev < 0:
        return "持续负偏离_非节假日"
    else:
        return "持续正偏离_非节假日"


if __name__ == "__main__":
    cfg = load_config()
    G = build_kg(cfg)
    print("负偏离可能成因:", query_causes(G, "持续负偏离_非节假日"))
    print("正偏离可能成因:", query_causes(G, "持续正偏离_非节假日"))
