from __future__ import annotations
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load_csv(name):
    with (BASE / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = {r["month"]: r for r in load_csv("monthly_fusion.csv")}
    diagnoses = {r["month"]: r for r in load_csv("anomaly_diagnosis.csv")}
    print("江苏电力经济知识图谱——月份异常诊断")
    print("可用月份：2024-01 到 2025-12")
    month = input("请输入月份（例如 2025-08）：").strip()
    if month not in rows:
        print("月份不存在，请按 YYYY-MM 格式输入范围内月份。")
        return
    r, d = rows[month], diagnoses[month]
    print("\n【基本数据】")
    print(f"全社会用电量：{r['consumption_month_yi_kwh']} 亿千瓦时，同比 {float(r['consumption_month_yoy_pct']):+.2f}%")
    print(f"发电量：{r['generation_month_yi_kwh']} 亿千瓦时，同比 {float(r['generation_month_yoy_pct']):+.2f}%")
    print(f"南京代表点月均温：{r['mean_temp_c']}℃；≥35℃天数：{r['hot_days_ge_35c']}天")
    print(f"本季度累计GDP同比：{r['gdp_yoy_pct']}%")
    print("\n【异常标签】", d["anomaly_labels"])
    print("【证据】", d["evidence"])
    print("【候选原因】", d["candidate_causes"])
    print("【结论边界】", d["conclusion"])
    print("【置信度】", d["confidence"])


if __name__ == "__main__":
    main()
