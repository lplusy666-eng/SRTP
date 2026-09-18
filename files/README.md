# 基于特征解耦与知识增强的“电力看经济”智能体

> 从数据获取 → 特征解耦 → VAE异常检测 → 知识图谱/RAG → 多智能体协同 → 报告与问答，
> 一套**开箱即用**的参考实现。默认用合成数据 + 免费天气/节假日API跑通全链路，
> 接入真实电量数据和大模型只需改配置。

---

## 一、这个项目在做什么

电力数据高频、连续、客观，是观测经济的“先行指标”。但电量变化同时受**天气、节假日、电价、新能源、政策**等非经济因素干扰。本系统的核心任务：**把经济信号从非经济扰动中剥离出来，并解释它、报告它、回答关于它的问题。**

对应开题的五步主线与四个研究目标：

| 主线阶段 | 模块 | 研究目标 |
|---|---|---|
| 多源数据输入 | `data_acquisition/` | — |
| 特征解耦分析 | `feature_decoupling/` | 目标1 感知（关键问题2 精准解耦） |
| 知识增强诊断 | `anomaly_detection/`(VAE) + `knowledge_graph/` + `rag/` | 目标2 诊断（关键问题3 融合） |
| 多智能体协同 | `agents/`(MCP风格协同) | 目标4 协同（关键问题4 可信输出） |
| 智能输出 | `agents/generation_agent.py` + `report/` | 目标3 生成 |

---

## 二、目录结构

```
power_economy_agent/
├── README.md                     # 本教程
├── requirements.txt
├── configs/config.yaml           # 全局配置（改这里即可切换真实数据/大模型）
├── data/
│   ├── raw/                       # 原始数据（电量/天气/经济/知识文档）
│   └── processed/                 # 整合与解耦后的数据
├── outputs/                       # 模型、异常结果、报告、可视化、协同留痕
├── scripts/
│   ├── run_pipeline.py            # 一键端到端
│   └── visualize.py               # 解耦与检测结果可视化
└── src/
    ├── utils.py                   # 配置/路径/日志
    ├── data_acquisition/          # ① 数据获取与整合
    │   ├── electricity_loader.py  #   电量（真实CSV或合成，含真值异常）
    │   ├── weather_fetcher.py     #   天气（Open-Meteo免费API）
    │   ├── calendar_features.py   #   节假日/距春节天数
    │   ├── economic_loader.py     #   宏观经济指标
    │   └── build_dataset.py       #   合并 + 派生HDD/CDD
    ├── feature_decoupling/        # ② 特征解耦（核心创新1）
    │   └── decouple.py            #   混杂因子回归 + STL + 小波跨尺度
    ├── anomaly_detection/         # ③ VAE异常检测（核心创新2）
    │   ├── vae_model.py
    │   └── detect.py              #   训练 + VAE重构×统计偏离融合打分
    ├── knowledge_graph/build_kg.py# ④ 知识图谱（成因推理）
    ├── rag/retriever.py           # ⑤ RAG检索（离线哈希或OpenAI嵌入）
    └── agents/                    # ⑥ 多智能体（核心创新3）
        ├── llm_client.py          #   兼容OpenAI接口 + 离线模板兜底
        ├── base_agent.py          #   MCP风格共享上下文/工具
        ├── perception_agent.py    #   感知智能体
        ├── diagnosis_agent.py     #   诊断智能体
        ├── generation_agent.py    #   生成智能体（报告+问答）
        └── coordinator.py         #   协同器（闭环调度 + 可信留痕）
```

---

## 三、安装

```bash
cd power_economy_agent
python -m venv venv && source venv/bin/activate    # 可选
pip install -r requirements.txt
```

> Python 3.10+。仅演示无需大模型Key；GPU可选（VAE很小，CPU几秒即可）。

---

## 四、快速开始（一条命令跑通全流程）

```bash
python scripts/run_pipeline.py
```

它会依次完成：生成/读取数据 → 解耦 → 训练VAE并检测异常 → 构建知识图谱与RAG → 三智能体协同 → 输出报告。产物在 `outputs/`：

- `economic_report.md`　自动分析报告
- `anomalies.csv`　　　 逐日异常分数与检出区间
- `agent_trace.json`　 多智能体协同留痕（可信输出依据）
- `kg.graphml` / `rag_index.pkl` / `vae_model.pt`

可视化：

```bash
python scripts/visualize.py        # -> outputs/visualization.png
```

提问：

```bash
python scripts/run_pipeline.py --no-rebuild --ask "幅度最大的异常是哪次？"
```

---

## 五、逐模块教程

### 步骤1：数据获取（`src/data_acquisition/`）
- **电量**：把真实数据放到 `data/raw/electricity.csv`（两列 `date, electricity`）即可自动使用；缺失时生成含真值异常的合成数据用于演示。
- **天气**：`weather_fetcher.py` 调用 Open-Meteo 历史API（免费无Key），取日均温，缓存到 `data/raw/weather.csv`。无网时回退合成温度。
- **节假日**：`calendar_features.py` 用 `chinesecalendar` 生成法定节假日、调休、`days_to_cny`（距春节天数，用于剥离春节前后复工爬坡）。
- **经济**：把真实宏观指标放到 `data/raw/economic.csv`（如 `date, gdp_yoy, pmi`），低频自动对齐到日频。

```bash
python -m src.data_acquisition.build_dataset   # 或在 src/ 下 python data_acquisition/build_dataset.py
```

### 步骤2：特征解耦（`feature_decoupling/decouple.py`）——创新1
三步剥离非经济扰动：
1. **混杂因子回归**：`log电量 ~ CDD + HDD + 周末 + 节假日 + 距春节逐日哑变量`，得到天气效应、日历效应，相减得到**经济信号**。
2. **年内气候态去除 + STL**：去掉残余年度季节性，分离**景气基线(trend)**与**冲击残差(resid)**。
3. **小波跨尺度分解**：对去趋势波动做DWT，得到各尺度能量占比（跨尺度特征）。
   
输出 `data/processed/decoupled_signal.csv`，关键列：`economic_signal / economic_trend / economic_residual / economic_deviation`。

### 步骤3：VAE异常检测（`anomaly_detection/detect.py`）——创新2
- 以 `[economic_deviation, economic_residual]` 构滑窗，训练时序VAE学习“正常经济波动”分布。
- 最终异常分数 = **VAE重构误差**（抓形态突变）与**统计偏离度稳健z分数**（抓持续偏移）**融合**。
- 连续异常日聚合成事件区间，写入 `outputs/anomalies.csv`。合成数据下会额外打印 P/R/F1/AUC 评估。

### 步骤4：知识图谱（`knowledge_graph/build_kg.py`）
编码“异常特征 → 候选成因 → 佐证指标”的领域知识（`SEED_TRIPLES`）。可在 `data/raw/kg_seed.csv`（列 `head,relation,tail,type`）追加你自己的行业/地区知识。

### 步骤5：RAG（`rag/retriever.py`）
把 `data/raw/knowledge_docs/` 下的政策/行业/宏观知识文档切块、向量化、检索。默认 `hash` 离线嵌入零依赖；配置 `rag.embed_backend: openai` 可用真实嵌入。

### 步骤6：多智能体协同（`agents/`）——创新3
- **感知智能体**：跑通数据→解耦→检测，产出异常事件。
- **诊断智能体**：对每个事件做 图谱匹配 → RAG检索 → PMI等宏观交叉核对 → 大模型综合归因（给出成因/佐证/置信度）。
- **生成智能体**：汇总为分析报告，并支持交互问答。
- **协同器**：用MCP风格共享上下文串成闭环，全过程留痕。

---

## 六、接入真实数据 / 真实大模型

**① 真实电量**：准备 `data/raw/electricity.csv`（`date,electricity`），无需改代码。同理 `economic.csv`。

**② 启用大模型**（DeepSeek / 通义千问兼容 / 智谱 / OpenAI 等，均为OpenAI兼容接口）：

```yaml
# configs/config.yaml
llm:
  enabled: true
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  api_key_env: "LLM_API_KEY"
```

```bash
export LLM_API_KEY="你的key"
python scripts/run_pipeline.py
```

未启用或无Key时，系统自动走**离线模板模式**，报告与问答仍可产出（只是归因文字较简略）。

---

## 七、结果与评估（合成数据）

合成数据中埋入了5个“经济事件”作为真值。解耦后，真值冲击段的经济偏离量约为正常段的 **2.8倍**，春节残留被压到 **2.1–2.3倍**（逐日剥离前高达3.2倍）。融合检测在纯无监督设定下 **AUC≈0.85**，可稳定命中制造业下滑、项目投产等主要事件。

> 说明：这是**无监督**检测的诚实结果。幅度最小（±8%）的冲击与个别季节性残留仍可能漏检/误报；接入更多辅助指标、按地区细化知识图谱、或引入少量标注做半监督，可进一步提升。

---

## 八、常见问题

- **天气API返回403/超时**：某些网络环境受限，系统会自动回退合成温度，不影响流程；本地正常联网即可拉到真实数据。
- **`chinesecalendar` 年份覆盖**：库通常覆盖到近几年，超出范围会退化为周末判断，建议观测期落在库覆盖区间内。
- **想换分析频率（周/月）**：改 `project.freq` 与 `decoupling.stl_period`（月频用12）。

---

## 九、如何扩展（面向毕设/竞赛）

- 解耦：把线性回归换成 GAM/梯度提升，或引入你论文里的“轻量化时序解耦”网络。
- 检测：把VAE换成 VAE-LSTM / Transformer，或加入知识图谱约束的异常评分。
- 知识：把 `SEED_TRIPLES` 扩成正式行业知识图谱，RAG接入真实政策库。
- 协同：把顺序编排升级为真正的 MCP Server/Client，或加入“路由/重写/多轮”RAG（对应开题PPT中的 RAG-I/II/III 架构）。
