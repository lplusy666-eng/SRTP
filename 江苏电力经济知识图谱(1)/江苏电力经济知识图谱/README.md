# 江苏电力经济知识图谱（2024—2025）

这是一个给初学者使用、可以追溯数据来源的“电力看经济”知识图谱。它把江苏省月度电力、季度GDP、南京代表点天气和国家法定节假日放在同一套结构中。

## 最容易使用的三个文件

1. `江苏电力经济知识图谱_交互浏览版.html`：双击打开，浏览图谱并选择月份做异常诊断。首次加载图谱需要联网。
2. `江苏电力经济知识图谱_实际数据版.xlsx`：适合在 Excel 中查数据、筛选和汇报。
3. `diagnose_anomaly.py`：在命令行输入月份，输出基于规则和证据的诊断。

## 数据范围与准确性

- 电力：2024-01—2025-12，逐月来自国家能源局江苏监管办公室公开月报。
- 经济：2024Q1—2025Q4，江苏省统计局发布的季度累计GDP；2025Q1使用人民日报转引的江苏统计数据并单独标注。
- 日历：国务院办公厅2024、2025年节假日安排。
- 天气：Open-Meteo ERA5再分析，使用南京坐标作为代表点，只是代理变量，不代表江苏全省平均。

## 重要理解

- “用电增长”可以支持经济活跃的判断，但不能单独证明经济快速增长。
- 春节在1月或2月之间移动，会让单月同比出现很大的正负波动，因此应合并1—2月观察。
- 夏季高温会推高空调负荷，必须把天气因素从经济因素中区分出来。
- 知识图谱里的 `MAY_AFFECT` 和 `CANDIDATE_CAUSE` 是候选机制，不是已经证明的因果。

## 文件说明

- `electricity_monthly.csv`：月度电力原始表
- `industry_quarterly.csv`：季度累计GDP表
- `weather_monthly.csv`：南京代表点月值天气
- `calendar_daily.csv` / `calendar_monthly.csv`：节假日日历
- `monthly_fusion.csv`：按月融合后的分析表
- `anomaly_diagnosis.csv`：每月自动诊断
- `nodes.csv` / `observations.csv` / `relationships.csv`：知识图谱三类核心表
- `neo4j_import.cypher`：Neo4j导入示例
- `example_queries.cypher`：查询示例
- `schema.ttl`：简化RDF结构
- `validation.json`：完整性检查
- `数据来源与口径.md`：数据源和限制的详细说明
