# 电力看经济智能体 —— 工具层与智能体内核

> 做电力经济领域的 "Claude Code"：用户输入一个地区的电力数据 + 一句自然语言问题，
> 由大模型自主决定调用哪些工具、调几次、何时停止，最后产出一份**可核查**的经济状态判断。

**当前状态：工具层与智能体循环已完整可运行，评测集已建立，大模型 API 未接入（留了标准接缝）。**

---

## 30 秒上手

```bash
cd workbuddy
python -m venv .venv && ./.venv/Scripts/activate     # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt

# 看看有哪些地区、每个地区有什么数据
python -m powerecon regions
python -m powerecon inspect --region zhejiang

# 问一个问题，跑完整智能体循环
python -m powerecon ask "2025年1月浙江用电同比下降是经济原因吗？"

# 看工具清单（模型能调用的接口面）
python -m powerecon tools --format md
```

不需要任何 API Key、不需要 GPU、不需要联网。默认用**真实的公开省级数据**
（浙江、江苏两省），不是合成数据。

---

## 它现在能回答什么

**能回答：**

- 某个地区某期用电量出现了异常吗？偏离多少？
- 这个异常是经济原因，还是春节错位 / 气温负荷 / 数据口径造成的？
- 用电同比与经济指标（GDP）出现了多大背离？
- 这个地区的数据有什么已知缺陷、不能怎么用？

**明确不能回答：**

- 预测未来的经济走势（系统只做历史归因）
- 判断"经济好不好"这种需要多指标综合的问题（单靠电力做不到）
- 给出已证明的因果关系（只能给候选原因 + 需要什么证据）

后两类会被系统**显式拒答**，而不是硬编一个答案。这是设计的一部分。

---

## 它和"固定流水线"的区别在哪

```
用户问题 + 地区数据包
        │
        ▼
┌──────────────────────────────────────────────┐
│  智能体循环 agent/                            │
│  plan → tool call → observe → 再规划          │
│  · 不认识任何具体工具                          │
│  · 由规划器决定下一步做什么                    │
└───────────────┬──────────────────────────────┘
                │ 工具名 + JSON 参数
                ▼
┌──────────────────────────────────────────────┐
│  工具层 tools/（14 个）                       │
│  数据 / 分解 / 检测 / 诊断 / 上下文 / 知识 / 输出 │
│  · Pydantic 定义 I/O，自动导出 JSON Schema     │
└───────────────┬──────────────────────────────┘
                ▼
┌──────────────────────────────────────────────┐
│  摄入层 ingest/ + 领域内核 analysis/           │
│  任意地区任意格式 → 统一结构；STL / 稳健检测     │
└──────────────────────────────────────────────┘
```

关键差异：**分析顺序不是写死的**。循环只做四件事 —— 问规划器下一步、执行、把结果塞回状态、重复。
换成别的问题、别的地区，循环代码一行都不用改。

---

## 目录结构

```
workbuddy/
├── INTERFACE.md              # ★ 三人协作的唯一接口契约，先读这个
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── knowledge/                # 跨地区共享的领域知识（春节错位 / 因果边界 / 数据口径）
├── region_packs/             # ★ 输入数据包：一个地区 = 一个目录
│   ├── zhejiang/             #   显式声明 datasets 的参考样例
│   │   ├── region.yaml
│   │   └── data/
│   └── jiangsu/              #   走 datasets: auto 自动发现路径
│       ├── region.yaml
│       └── data/
│
├── src/powerecon/
│   ├── contract.py           # ★ 契约：输入包 / 工具 I/O / 五段式输出
│   ├── ingest.py             # 通用摄入：自动发现 + 格式嗅探 + 列名别名
│   ├── analysis.py           # STL 分解与稳健异常打分（唯一实现）
│   ├── series_utils.py       # 期次网格 / 偏移 / 稳健统计
│   ├── runtime.py            # 装配：pack + 产物目录 → 工具上下文
│   ├── cli.py                # 命令行入口
│   ├── tools/                # ★ 工具层（14 个工具）
│   │   ├── base.py           #   注册器 + schema 导出 + 调度 + 紧凑渲染
│   │   ├── data_tools.py     #   列出数据集 / 取序列 / 质量检查
│   │   ├── decompose_tools.py#   STL 分解
│   │   ├── detect_tools.py   #   异常检测
│   │   ├── diagnosis_tools.py#   排除式归因
│   │   ├── context_tools.py  #   日历 / 天气
│   │   ├── knowledge_tools.py#   图谱查询 / 知识检索
│   │   ├── economy_tools.py  #   用电 vs GDP 偏离度
│   │   └── report_tools.py   #   出图 / 提交结论
│   └── agent/                # ★ 智能体内核
│       ├── loop.py           #   plan → act → observe 循环
│       ├── planner.py        #   RulePlanner / LLMPlanner / ScriptedPlanner
│       ├── prompts.py        #   系统提示词（领域侧维护）
│       └── trace.py          #   决策留痕
│
├── evals/                    # ★ 评测集：判断"准不准"的唯一客观标准
│   ├── questions.yaml        #   20 道题 + 期望行为
│   └── run_eval.py           #   自动打分
├── tests/                    # 23 项回归测试
└── artifacts/                # 产物：留痕 / 图表 / 评测报告
```

---

## 接入大模型（唯一的接缝）

`LLMPlanner` 里**没有任何厂商 SDK 调用**。你只需要写一个回调：

```python
# my_llm.py
from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.deepseek.com/v1")

def call_llm(messages, tools):
    """messages: OpenAI 兼容消息列表
       tools:    来自 tools.openai_tools() 的 function-calling 清单"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
    )
    return resp.choices[0].message
```

```bash
python -m powerecon ask "2025年1月浙江用电同比下降是经济原因吗？" \
    --planner llm --llm-callable my_llm:call_llm
```

归一化已覆盖 OpenAI / Anthropic / 原生 dict 三种返回形态，换厂商只需换那个回调。

**没有 Key 也能验证整条链路** —— 默认的 `RulePlanner` 是不依赖大模型的确定性规划器，
按一条合理的固定工具链走完，结论由模板合成。它的作用是：
① 把链路验证通；② LLM 不可用时兜底；③ 作为评测基线。

---

## 新增一个地区（不改代码）

```bash
mkdir -p region_packs/mycity/data
cp 你的数据.csv region_packs/mycity/data/
```

```yaml
# region_packs/mycity/region.yaml
region:
  code: CN-XX
  name: 某某市
  level: city

datasets: auto          # 让摄入层自己嗅探

caveats:
  - 天气为单点代理变量，不代表全域平均。
  - 数据覆盖 2020-01 至 2025-06。
```

```bash
python -m powerecon inspect --region mycity     # 看它嗅探出了什么
python -m powerecon ask "这个地区最近的用电情况如何？" --region mycity
```

**列名不需要对齐。** 浙江用 `assumed_value_yi_kwh`、江苏用 `consumption_month_yi_kwh`、
知识图谱一边用 `id/type` 一边用 `node_id/relationship_type` —— 摄入层全部自动适配。

如果嗅探选错了列，用 `region.yaml` 的 `overrides` 或工具的 `column` 参数覆盖。

---

## 新增一个工具（不改调度代码）

见 `INTERFACE.md` 第 2.2 节。核心是两步：写一个 `@tool` 装饰的函数，
在 `tools/__init__.py` 里 import 一次。JSON Schema 和 function-calling 格式都是自动导出的。

**工具描述是模型唯一的说明书**，请写清楚"什么时候该用、什么时候不该用、返回值的边界"。

---

## 验证

```bash
python -m pytest tests/ -q            # 23 项回归测试
python evals/run_eval.py              # 20 题评测（规则规划器基线）
python evals/run_eval.py --id zj-01 --verbose    # 单题调试，打印完整结论
```

评测报告落在 `artifacts/evals/<时间戳>/report.md`。

**当前基线：20/20 通过（规则规划器）。** 注意这是自建评测集上的自评分数，
不代表真实业务精度 —— 它衡量的是"有没有违反判断纪律"，不是"判断得有多准"。
接上 LLM 之后这个分数应该更高，并且应该能通过那些规则规划器只能靠模板兜住的题。

---

## 数据来源与已知限制

**浙江**（2021-01 至 2025-03）
- 用电：浙江省月度全社会用电量，单位按数量级推断为亿千瓦时（`unit-inferred`）
- 经济：分产业季度增加值（累计口径）
- 天气：杭州单点 ERA5 再分析 —— **代理变量，不代表全省加权平均**
- 日历：国务院办公厅节假日安排
- 图谱：241 节点 / 4775 关系，以数据目录型关系为主

**江苏**（2024-01 至 2025-12）
- 用电：国家能源局江苏监管办公室公开月报
- 经济：江苏省统计局季度累计 GDP
- 天气：南京单点 ERA5 再分析 —— **代理变量**
- 最高负荷口径跨月不一致（`peak-load-scope-varies`）
- 图谱：123 节点 / 857 关系

**共同的限制：**

1. **天气是单点代理。** 浙江南北跨度、江苏苏南苏北差异都很大，单点气温估算全省负荷会有偏差。
2. **季度 GDP 是年内累计值。** 差分出单季值时，Q4 会吸收全年核算修订，产生假异常。
3. **知识图谱缺机制边。** 两省图谱主要是 `MEASURES` / `OBSERVED_IN` / `AT_TIME` 这类结构关系，
   真正的机制先验边（`AFFECTS` / `PROXIES_FOR`）只有个位数。补齐机制边是领域侧的活。
4. **不做因果推断。** 输出的是"带证据和不确定性的原因候选"，不是已证明的因果。

---

## 团队分工

| 谁 | 管什么 | 目录 | 判据 |
|---|---|---|---|
| 李佳昱 | 判断准不准（领域内核 + 评测） | `knowledge/` `evals/` `analysis.py` `prompts.py` | 评测集得分 |
| 薛天宏 | 手能不能动（工具层 + 摄入层） | `tools/` `ingest.py` | 换地区不改代码 |
| 朴盛珉 | 脑子怎么转（智能体内核） | `agent/` | 换问题不改链路 |

**开工前先读 `INTERFACE.md`。** 三个人各自实现的部分只通过 `contract.py` 里的模型对话，
不允许各自定义私有格式。

---

## 下一步

1. **接入大模型**（`LLMPlanner` 的接缝已就位）—— 写一个 `call_llm` 回调即可。
2. **用 LLM 规划器跑一遍评测集**，和规则规划器基线对比，看哪些题它做对了、哪些做错了。
3. **补齐知识图谱的机制边**（`MAY_AFFECT` / `CANDIDATE_CAUSE`）—— 这是领域侧的活，
   目前机制候选主要靠规则和知识文档，图谱贡献有限。
4. **接入真实小时级数据** —— 当前工具层对月度/季度验证充分，小时级需要新的季节周期参数。
5. **把工具层暴露成标准 MCP Server** —— 让 Claude Code / Cursor 等客户端也能直接调用。
