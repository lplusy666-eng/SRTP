# 与开题答辩方案的对应关系

| 开题方案内容 | 本仓库实现 |
|---|---|
| 多源数据输入：电量、气象、节假日、经济 | `src/power_econ/data/`；支持合成、负荷 CSV、Open-Meteo、AKShare、本地气象/宏观 CSV、事件日历 |
| 特征解耦：剔除常规波动、识别跨尺度特征 | `features/decomposition.py` 的过去信息限定趋势/日/周/残差；`FeatureBuilder` 五组特征；预测模型独立分支、动态门控和正交正则 |
| 知识图谱 + LLM + VAE | `knowledge/domain_knowledge.yaml`、`DomainKnowledgeGraph`、`SequenceVAE`、模板/OpenAI 结构化生成 |
| RAG 检索增强 | `LocalKnowledgeBase` 对 `knowledge/docs` 做离线字符 n-gram TF-IDF 检索 |
| MCP 多智能体协同 | 四智能体 + `PowerEconomyOrchestrator`；`mcp_server.py` 暴露五个工具 |
| 异常监测 | 预测区间偏差 + VAE 重构误差 + 验证集稳健校准 + 时间去重 |
| 智能诊断 | 特征组消融、规则原因排序、图谱路径、RAG 证据、不确定性和人工复核标记 |
| 报告生成 | 报告期统计、异常诊断、经济含义、nowcast、风险、建议，输出 Markdown/JSON |
| 交互问答 | 最近事件、诊断数据库与知识检索联合回答，返回证据和事件 ID |
| 从数据到结论闭环 | CLI `power-econ demo`、FastAPI、Streamlit、MCP、SQLite |
| 软件原型 | `power-econ dashboard` 和 `power-econ serve` |
| 技术报告/模型评估 | `MODEL_CARD.md`、`training_summary.json`、`DEMO_VALIDATION.md` |

## 四个关键问题的工程回答

### 1. 复杂关系如何量化？

采用显式特征组、多分支编码、动态门控、组消融和月度 nowcast，把负荷、天气、日历、宏观、事件/质量的关系转换为可计算权重、贡献和指标。

### 2. 多源扰动如何精准解耦？

先做过去信息限定的趋势/日/周分解，再对五类来源独立编码，并通过规则优先排除数据质量、天气、日历和政策事件。该结果是可解释的结构化解耦，而非未经验证的严格因果识别。

### 3. 图谱、大模型与异常检测如何融合？

VAE 和预测残差负责发现异常；组消融负责模型侧解释；图谱规则负责业务候选原因；RAG 提供知识证据；LLM 只在这些结构化事实上生成诊断、报告和回答。

### 4. 多智能体如何协同输出可信结果？

感知智能体产生并登记事件，诊断智能体产生可审计原因，报告智能体汇总时间段结论，问答智能体读取事件/诊断/知识回答。编排器和 SQLite 保证事件 ID、证据和报告可追溯；API/MCP 负责外部调用。
