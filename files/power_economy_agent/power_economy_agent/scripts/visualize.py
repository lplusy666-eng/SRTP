"""
可视化：解耦分解 + 异常检测结果
================================
生成 outputs/visualization.png：
  子图1 原始电量 vs 天气/日历效应
  子图2 解耦后的经济信号与景气基线
  子图3 经济偏离量 + 检测到的异常区间(与真值对比)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import load_config, abspath  # noqa

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    cfg = load_config()
    dec = pd.read_csv(abspath(cfg["decoupling"]["output_csv"]), parse_dates=["date"])
    anom = pd.read_csv(abspath(cfg["vae"]["anomaly_csv"]), parse_dates=["date"])
    df = dec.merge(anom[["date", "anomaly_score", "is_anomaly"]], on="date", how="left")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1) 原始 vs 效应
    axes[0].plot(df["date"], df["log_elec"], lw=0.7, label="log electricity (raw)")
    axes[0].plot(df["date"], df["log_elec"] - df["weather_effect"] - df["calendar_effect"],
                 lw=0.7, label="after removing weather+calendar")
    axes[0].set_title("1) Raw vs. Confounder-removed (log electricity)")
    axes[0].legend(loc="upper left", fontsize=8)

    # 2) 经济信号与基线
    axes[1].plot(df["date"], df["economic_signal"], lw=0.7, label="economic signal")
    axes[1].plot(df["date"], df["economic_trend"], lw=1.5, label="economic baseline (trend)")
    axes[1].set_title("2) Decoupled economic signal & baseline")
    axes[1].legend(loc="upper left", fontsize=8)

    # 3) 偏离 + 异常
    axes[2].plot(df["date"], df["economic_deviation"], lw=0.7, color="gray", label="economic deviation")
    # 检测异常
    det = df[df["is_anomaly"] == 1]
    axes[2].scatter(det["date"], det["economic_deviation"], s=6, color="red", label="detected anomaly")
    # 真值
    if "_true_anomaly" in df.columns:
        tru = df[df["_true_anomaly"] == 1]
        axes[2].scatter(tru["date"], np.full(len(tru), df["economic_deviation"].min() * 1.1),
                        s=4, color="green", marker="|", label="ground-truth anomaly")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("3) Economic deviation with detected vs. ground-truth anomalies")
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out = abspath("outputs/visualization.png")
    plt.savefig(out, dpi=120)
    print("已保存可视化:", out)


if __name__ == "__main__":
    main()
