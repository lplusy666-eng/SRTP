# 演示验证报告

## 1. 验证对象

- 配置：`configs/demo.yaml`
- 数据：两年小时级合成数据
- 行数：17,544
- 时间范围：2023-01-01 00:00 至 2024-12-31 23:00，Asia/Shanghai
- 模型：特征解耦分位数预测 + GRU-VAE + Ridge nowcast
- 智能体：感知、诊断、报告、问答
- 生成模式：离线确定性模板

合成数据用于证明完整工程、训练、持久化和服务链路可复现，不代表任何实际地区电网测量结果。

## 2. 已执行验证

```bash
pytest
python -m compileall -q src tests
power-econ doctor -c configs/demo.yaml
power-econ forecast -c configs/demo.yaml
power-econ scan -c configs/demo.yaml --limit 3
power-econ diagnose 6786696aebfd6c7c -c configs/demo.yaml
power-econ report 2024-12-01 "2024-12-31 23:00:00" -c configs/demo.yaml
power-econ chat "最近一个月负荷异常的主要原因是什么？能否说明经济走弱？" -c configs/demo.yaml
python scripts/verify_all.py --config configs/demo.yaml
```

FastAPI 使用 `TestClient` 验证：

- `/health`
- `/v1/model/summary`
- `/v1/forecast/latest`
- `/v1/anomalies`
- `/v1/diagnoses/{event_id}`
- `/v1/reports`
- `/v1/qa`

## 3. 预测指标

| 指标 | 数值 |
|---|---:|
| MAE | 41.0456 MW |
| RMSE | 59.3582 MW |
| MAPE | 3.1378% |
| 一步 MAE | 39.8223 MW |
| 一步 RMSE | 57.4677 MW |
| P10–P90 覆盖率 | 0.7696 |
| P10–P90 平均宽度 | 110.9043 MW |
| 周季节朴素基线 MAE | 44.2975 MW |
| 相对基线 MAE 改善 | 7.3411% |

## 4. 异常指标

| 指标 | 数值 |
|---|---:|
| Precision | 0.3049 |
| Recall | 0.9434 |
| F1 | 0.4608 |
| ROC-AUC | 0.9848 |
| Average Precision | 0.7211 |
| 测试真实异常率 | 0.0201 |
| 测试预测异常率 | 0.0623 |

演示配置偏向高召回，适合展示漏检控制。真实部署可提高阈值或加入事件级评估以降低误报。

## 5. 经济 nowcast 指标

| 指标 | 数值 |
|---|---:|
| MAE | 0.2196 |
| RMSE | 0.2431 |
| MAPE | 3.7727% |
| R² | 0.6134 |
| 相关系数 | 0.8253 |
| 方向准确率 | 0.6000 |
| 月份数 | 24 |

## 6. 诊断样例

事件 `6786696aebfd6c7c`：

- 时间：2024-05-04 10:00 +08:00；
- 实测负荷：1317.6 MW；
- 期望负荷：1439.9 MW；
- 残差：-122.2 MW；
- 首要原因：需求响应、错峰或政策事件；
- 次要原因：节假日与调休效应；
- 经济结论：主要由非经济扰动解释，原始负荷不宜直接作为经济变化结论。

完整 JSON：`artifacts/outputs/demo_diagnosis.json`。

## 7. 持久化与服务验证

- 模型能够从磁盘重新加载；
- 异常校准文件可读取；
- 事件、诊断和报告可写入 SQLite；
- Markdown 和 JSON 报告可生成；
- QA 返回证据和事件 ID；
- FastAPI 主要接口均返回 200；
- 一键验收结果写入 `artifacts/outputs/verification_summary.json`。

## 8. 尚需真实项目完成的工作

1. 替换为当地真实负荷和分项负荷；
2. 对接准确的调度、检修、需求响应日志；
3. 将 PPI 代理替换为实际电价/交易价格；
4. 建立人工确认的异常原因数据集；
5. 重新选择阈值并做事件级 Precision/Recall；
6. 增加行业、区域、电压等级分项解释；
7. 进行安全、权限、审计和生产数据库改造；
8. 由电网和经济专家审核知识规则与报告措辞。
