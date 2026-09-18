# 浙江电力经济 STL 分解成果包

## 1. 一键运行

```bash
python -m pip install -r requirements.txt
python stl_power_economy_pipeline.py --input-dir data/raw --output-dir outputs
```

## 2. 输入

- 月度全社会用电量：浙江全社会用电量信息（月度）.xlsx
- 第一、第二、第三产业季度累计 GDP：3 个 XLSX
- 国务院办公厅近五年节假日安排：国务院办公厅近五年节假日安排.docx

## 3. 关键设定

- 月度用电量：log-STL, period=12, seasonal=13, robust=True
- 季度 GDP：先将累计值差分为单季度值，再做 log-STL, period=4, seasonal=7, robust=True
- 异常阈值：|残差比例| >= 8% 为 high，5%-8% 为 medium
- 首尾各 1 个季节周期标记 endpoint_warning
- GDP Q4 标记 revision_risk，防止把年度核算修订误认作经济冲击

## 4. 本次运行摘要

- 电力月度记录：51 条
- 电力高等级异常：7 条
- 电力趋势强度：0.565
- 电力季节强度：0.784
- GDP 单季度记录：312 条（3 个产业合计）
- GDP 高等级异常：22 条

## 5. 主要输出

- `outputs/tables/electricity_monthly_stl.csv`
- `outputs/tables/gdp_quarterly_stl.csv`
- `outputs/tables/anomaly_events.csv`
- `outputs/model_input/model_input_records.jsonl`
- `outputs/kg/kg_nodes.csv`, `outputs/kg/kg_edges.csv`
- `outputs/figures/*.png`

## 6. 因果诊断边界

STL 只能分离趋势、季节与不规则项，不能单独证明异常原因。脚本把节假日、天气、工业活动、政策事件等组织为“候选原因+所需证据”，供后续知识图谱、RAG 和规则引擎核验，避免把时间重合直接写成因果关系。
