# INTERFACE.md —— 三人协作的唯一接口契约

> **这份文件是冻结物。** 三个人（领域内核 / 工具层 / 智能体内核）各自开工之前，
> 先在这里对齐；对齐之后，任何一方改动都要同步更新这份文件。
> 所有实现都必须以本文档为准，不允许各自定义私有格式。

版本：`0.1`　冻结日期：2026-09-14

---

## 0. 一句话架构

```
用户问题 + 地区数据包
        │
        ▼
┌───────────────────┐
│  智能体内核        │  plan → tool call → observe → 再规划
│  agent/            │  不认识任何具体工具，只调 dispatch(name, args)
└─────────┬─────────┘
          │ 工具名 + JSON 参数
          ▼
┌───────────────────┐
│  工具层            │  14 个工具，Pydantic 定义 I/O，自动导出 JSON Schema
│  tools/            │  每个工具 = 一段可独立验证的能力
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  摄入层 + 领域内核  │  任意地区任意格式 → 统一内部结构；STL/检测/诊断规则
│  ingest/ analysis/ │
└───────────────────┘
```

三条正交的职责线，对应三个人：

| 谁 | 管什么 | 目录 | 判据 |
|---|---|---|---|
| 李佳昱 | 判断准不准（领域内核 + 评测） | `knowledge/` `evals/` `analysis.py` | 评测集得分 |
| 薛天宏 | 手能不能动（工具层 / 摄入层） | `tools/` `ingest.py` | 换地区不改代码 |
| 朴盛珉 | 脑子怎么转（智能体内核） | `agent/` | 换问题不改链路 |

---

## 1. 输入数据包格式（region pack）

**这是"接入任意地区"的落点。** 一个地区 = 一个目录。

```
region_packs/<name>/
├── region.yaml          # 必需
└── data/                # 必需
    ├── *.csv            # 任意表，摄入层会自动发现
    └── knowledge/*.md   # 可选，地区特有知识文档
```

### region.yaml 最小要求

```yaml
region:
  code: CN-ZJ          # 必需，ISO 3166-2 风格
  name: 浙江省          # 必需
  level: province      # 可选：province / city / national
  timezone: Asia/Shanghai
  representative_point:      # 可选，天气数据的坐标
    name: 杭州
    latitude: 30.2741
    longitude: 120.1551
```

**只有 `code` 和 `name` 是必需的。** 其他全部可选 —— 这是刻意的：
新增一个地区的最低成本应该只是"把 CSV 丢进 data/ 并写两行 yaml"。

### 数据集声明的两种方式

**方式 A：显式声明（推荐用于主用地区）**

```yaml
datasets:
  - id: electricity_monthly
    kind: timeseries        # timeseries / calendar / knowledge_graph / table
    path: data/electricity_monthly.csv
    date_column: month
    value_column: assumed_value_yi_kwh
    metric: 全社会用电量
    unit: 亿千瓦时
    frequency: monthly      # 留空则自动推断
    group_column: indicator_name   # 长表（一行一个产业）时用
```

**方式 B：自动发现（推荐用于新地区）**

```yaml
datasets: auto
```

摄入层会扫描 `data/*.csv`，自动判断：
- 哪一列是期次（先看列名 `date/month/period/...`，再看内容能否解析成日期）
- 哪一列是主数值列（语义加权 + 单位后缀 + 当月值优先于累计值 + 变异度）
- 频率（相邻期次间隔的中位数 → 日/周/月/季/年）

**已知的嗅探陷阱（必须防）：**

| 陷阱 | 例子 | 后果 |
|---|---|---|
| 选中累计值而非当月值 | `consumption_ytd_yi_kwh` vs `consumption_month_yi_kwh` | 同比和季节分析全部失效 |
| 选中比率列 | `*_yoy_pct` | 数值量级完全错误 |
| 选中标记列 | `quality_flag` | 不可分析 |

嗅探逻辑已对这三类做了排除/惩罚，并有回归测试
（`test_jiangsu_auto_discovery_picks_monthly_not_ytd`）。

### 展示信息覆盖

只想改指标名/单位、不想关掉自动发现时：

```yaml
datasets: auto
overrides:
  electricity_monthly:
    metric: 全社会用电量
    unit: 亿千瓦时
```

### 数据边界声明（caveats）—— 强烈建议写

```yaml
caveats:
  - 天气为杭州单点 ERA5 再分析，是省级代表点代理变量，不代表全省加权平均。
  - 季度 GDP 为年内累计值，不可直接当作单季度当季值。
  - 用电量原始表未标注单位，按数量级推断为亿千瓦时。
```

**这些字符串会被原样注入模型上下文。** 凡是"这个数据不能怎么用"的话都写在这里，
让模型在判断时自己知道边界，而不是靠人记。

### 知识图谱（可选）

```yaml
kg:
  kind: knowledge_graph
  nodes: data/kg_nodes.csv
  relationships: data/kg_relationships.csv
  observations: data/kg_observations.csv
```

列名允许不一致，摄入层会做别名映射：

| 统一名 | 浙江包 | 江苏包 |
|---|---|---|
| `id` | `id` | `node_id` / `relationship_id` |
| `type` | `type` | `relationship_type` |

**新增地区时不需要为了统一列名去改原始数据** —— 这是硬性设计目标。

---

## 2. 工具层契约

### 2.1 工具清单

> 完整清单与 JSON Schema 由代码生成，不要手工维护：
> ```bash
> python -m powerecon tools --format md          # 人读表格
> python -m powerecon tools --format openai      # 塞进 OpenAI/DeepSeek/通义 的 tools=[]
> python -m powerecon tools --format anthropic   # 塞进 Anthropic 的 tools=[]
> ```

当前 14 个工具，按用途分五组：

| 组 | 工具 | 职责 |
|---|---|---|
| 数据 | `list_available_regions` `list_region_datasets` `load_series` `data_quality_report` | 看清有什么、取出序列、检查质量 |
| 分解 | `decompose_series` | STL 拆出趋势/季节/残差 |
| 检测 | `detect_anomaly` | 残差上找异常期次 |
| 诊断 | `explain_anomaly` | 排除式归因：数据构造 → 日历 → 天气 → 经济 |
| 上下文 | `get_calendar_context` `get_weather_context` | 非经济因素的排除依据 |
| 知识 | `query_knowledge_graph` `search_knowledge` | 机制先验与领域知识 |
| 经济 | `nowcast_economy` | 用电同比 vs GDP 同比的偏离度 |
| 输出 | `make_chart` `submit_conclusion` | 出图、提交五段式结论 |

### 2.2 新增一个工具（薛天宏）

只需要两步，**不需要改任何调度代码**：

```python
# src/powerecon/tools/my_tools.py
from pydantic import BaseModel, Field
from .base import ToolContext, tool

class MyArgs(BaseModel):
    dataset_id: str = Field(description="数据集 id，来自 list_region_datasets 的返回。")

@tool(
    name="my_tool",
    description=(
        "一句话说清这个工具做什么。\n"
        "再说清楚**什么时候该用它、什么时候不该用** —— 模型看不到你的代码，只看得到这段文字。\n"
        "最后说明返回值的边界，例如'这是代理变量''这不是因果'。"
    ),
    args_model=MyArgs,
    returns="MyResult：返回什么",
    tags=("analysis",),
)
def my_tool(ctx: ToolContext, args: MyArgs) -> BaseModel:
    return MyResult(...)
```

```python
# src/powerecon/tools/__init__.py 里加一行 import
from . import my_tools  # noqa
```

**工具描述的质量决定成败。** 描述太短会被测试拦下
（`test_every_tool_description_is_substantive` 要求 ≥40 字）。

### 2.3 工具返回值的三条硬规则

1. **返回 Pydantic 模型，不返回裸 dict / 裸 DataFrame。** 这样才能自动导出 schema、
   自动做紧凑渲染。
2. **长列表必须可控。** 默认只返回摘要，逐期明细用 `include_points=True` 显式索取。
   统一走 `render_for_llm()` 做截断（默认最多 8 项 + 总长 4000 字符）。
3. **失败不抛异常，返回可读错误。** `dispatch()` 会把参数校验失败、工具内部异常
   都转成文本回给模型，让模型自己改参数重试。整条链路不能因为一个工具出错而崩掉。

### 2.4 缺失数据必须显式声明

这是防幻觉的关键约定：

```python
# 正确：明确告诉模型"这一类因素没有被排除"
return WeatherContext(..., available=False,
                      notes=["该地区没有天气数据集，无法排除气温对负荷的影响。"])

# 错误：返回空列表，模型会误以为"没有异常"
return WeatherContext(..., monthly=[])
```

**规则：`available=false` 比空列表安全得多。** 模型对空列表的反应通常是
"没有异常"，对 `available=false` 才会正确地在结论里写上"缺少天气数据"。

---

## 3. 输出五段式契约

**全系统唯一的对外结论格式。** 定义在 `contract.py::Conclusion`。

| 段 | 字段 | 要求 |
|---|---|---|
| 结论 | `conclusion` | 直接回答用户问题。禁止"证明了""说明经济走弱"这类确定性表述 |
| 证据 | `evidence[]` | 每条 = `claim + tool + value + period`，必须能追溯到具体工具调用 |
| 反证 | `counter_evidence[]` | **强制非空。** 什么事实会推翻这个结论？ |
| 置信度 | `confidence` | 0—1。数据缺口、代理变量、样本不足都要相应下调 |
| 不确定性 | `uncertainty[]` | **强制非空。** 还缺什么数据才能确认？ |

**强制非空是写在类型里的**，不是写在提示词里的：

```python
@model_validator(mode="after")
def _enforce_five_parts(self):
    if not self.counter_evidence:
        raise ValueError("五段式契约要求 counter_evidence 至少一条")
    if not self.uncertainty:
        raise ValueError("五段式契约要求 uncertainty 至少一条")
    return self
```

**提交方式：模型必须调用 `submit_conclusion` 工具，不能直接返回自由文本。**
好处有三个：
1. schema 校验 —— 少写一段就调用不成功，模型必须补全；
2. 可追溯 —— 结论的每个字段都进留痕，事后能审计模型怎么想的；
3. 可评分 —— 评测集直接比对结构化字段，不需要解析自由文本。

---

## 4. 因果边界规则（李佳昱维护，全系统强制）

### 4.1 因果强度必须分级

```python
class CausalStatus(str, Enum):
    OBSERVED = "observed"                    # 只是观测到现象
    CORRELATION = "correlation"              # 相关
    CANDIDATE_CAUSE = "candidate_cause"      # 候选原因（有机制先验，待证据）
    DATA_QUALITY_RISK = "data_quality_risk"  # 异常来自数据构造而非真实波动
    ESTABLISHED = "established"              # 已证因果 —— 本项目当前不应产出
```

`ESTABLISHED` 是保留值。评测集把它列为全局禁止项。

### 4.2 知识图谱的关系命名

允许：`MAY_AFFECT` / `CANDIDATE_CAUSE` / `PROXIES_FOR` / `DISTURBS` / `DERIVES_FEATURE`
禁止：`causedBy` / `导致` / `引起` / `造成`

### 4.3 排除顺序不可颠倒

`explain_anomaly` 固定按这个顺序生成候选原因：

1. **数据构造风险** —— 该期是否来自累计值差分、是否处于序列端点、单位是否推断的
2. **日历** —— 春节在 1 月还是 2 月会让单月同比摆动几十个百分点
3. **天气** —— 夏冬两季负荷主要由气温驱动
4. **经济** —— 只给低置信度 + "需要什么证据才能确认"

排在前面的如果成立，后面的解释力必须相应下调。

### 4.4 残差与同比是两个量，不能混

这是最容易出错的地方：

| 量 | 含义 | 大 = ? |
|---|---|---|
| 同比 `yoy_pct` | 与去年同期比 | 可能完全由季节摆动和春节错位造成 |
| 残差偏离 `residual_pct` | 实际值 vs 趋势+季节的期望值 | 才说明"有东西没被解释" |

**残差接近零时，正确结论是"不构成经济信号"，而不是顺着用户预设编一个异常叙事。**
（见 `test_residual_negligible_is_not_reported_as_economic_signal`）

### 4.5 能力边界必须显式拒答

两类问题直接拒答，不做分析：

- **预测类**（"明年会好转吗""未来走势如何"）—— 系统只做历史归因，不做预测
- **超范围期次**（"2030年5月的用电量"）—— 没有数据就没有结论

---

## 5. 智能体内核契约（朴盛珉）

### 5.1 循环只做四件事

```python
for step in range(max_steps):
    decision = planner.plan(state)      # 问下一步
    if decision.finish: break
    result = dispatch(decision.tool, decision.arguments, ctx)   # 执行
    state.observations.append(result)   # 塞回状态
    if isinstance(result.payload, Conclusion): break            # 拿到结论即结束
```

**循环本身不认识任何具体工具。** 这是与"固定流水线"的分界线。

### 5.2 三个终止条件，缺一不可

1. 规划器主动结束（`tool=None`）
2. `submit_conclusion` 成功返回
3. 步数达到 `max_steps`

循环结束时若仍无结论，必须合成一个**诚实的失败结论**（说明为什么没做成、
已经观察到什么），绝不返回半成品当结论。

### 5.3 接入大模型（LLMPlanner）

**`LLMPlanner` 里没有任何厂商 SDK 调用。** 只定义接缝：

```python
def call_llm(messages, tools):
    """messages: OpenAI 兼容消息列表
       tools:    来自 tools.openai_tools() 的 function-calling 清单
       返回：OpenAI 的 response.choices[0].message，或 Anthropic 的 response.content，
             或任何能归一化的 dict。返回 None / 空 tool_calls 表示模型认为可以结束。"""
    ...

loop = AgentLoop(ctx, LLMPlanner(call_llm))
loop.run("2025年1月浙江用电同比下降是经济原因吗？")
```

命令行接法：

```bash
python -m powerecon ask "..." --planner llm --llm-callable my_llm:call_llm
```

归一化已覆盖 OpenAI / Anthropic / 原生 dict 三种返回形态。

### 5.4 系统提示词是领域侧的交付物

`agent/prompts.py` 里的 `DEFAULT_SYSTEM_PROMPT` 属于**李佳昱的 judgment_rules 职责范围**。
工具层和循环层不碰它。里面的判断纪律需要领域侧持续打磨。

### 5.5 决策留痕

每一步都写 JSONL（`artifacts/traces/run_*.jsonl`），包含：
`phase`（plan/act/finish/error）、`tool`、`arguments`、`thought`、`ok`、`duration_ms`、`observation`。

**留痕是"可信输出"的依据** —— 结论里每句话都应该能回放到是哪一步工具调用支撑的。

---

## 6. 评测契约（李佳昱维护）

`evals/questions.yaml` 是唯一的客观评分标准。每题包含：

```yaml
- id: zj-01
  region: zhejiang
  question: "2025年1月浙江用电同比下降，是经济原因吗？"
  period: "2025-01"
  expect:
    must_call: [...]            # 软检查：仅 LLM 规划器强制
    must_mention_any: [...]     # 硬检查：必须提到其中至少一个
    min_confidence: 0.3
    max_confidence: 0.85
  note: 这道题在考什么，以及为什么这个期望是对的
```

**硬检查**（任何规划器都要满足）：因果强度不越界、禁用词不作主张使用、
反证与不确定性非空、置信度在区间内、必须提到关键概念、必须涉及用户点明的期次。

**软检查**（仅 LLM 规划器）：必须调用过哪些工具。

全局上限：**任何结论的置信度不得超过 0.85**。输入是省级公开月度数据 + 单点气象代理 +
相关性级别的归因，任何"非常确信"的表述都说明系统高估了自己。

跑法：

```bash
python evals/run_eval.py                                    # 规则规划器基线
python evals/run_eval.py --planner llm --llm-callable ...   # 接入 LLM 后对比
python evals/run_eval.py --id zj-01 --verbose               # 单题调试
```

**规则规划器是下限。** LLM 规划器至少要明显好于它才有意义。

---

## 7. 变更流程

任何一方要改下列内容，必须先改这份文件、并知会另外两人：

- `region.yaml` 的字段（影响薛天宏的摄入层）
- 工具的 name / 入参 / 返回类型（影响朴盛珉的循环与提示词）
- `Conclusion` 的字段（影响所有人的输出）
- `CausalStatus` 的枚举值（影响评测与领域判断）
- 评测集的判据（影响"准不准"的定义）

改完跑一遍回归：

```bash
python -m pytest tests/ -q          # 23 项
python evals/run_eval.py            # 20 题
```

两个都绿了才算改完。
