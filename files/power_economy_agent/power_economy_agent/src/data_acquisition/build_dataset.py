"""
数据整合
--------
将 电量 + 天气 + 日历 + 经济 合并为统一日频数据集，
并派生特征：HDD/CDD（度日）、对数电量等。
输出: data/processed/merged_dataset.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir, load_config
from data_acquisition.electricity_loader import load_electricity
from data_acquisition.weather_fetcher import fetch_weather
from data_acquisition.calendar_features import build_calendar_features
from data_acquisition.economic_loader import load_economic

log = get_logger("data.build")


def build_dataset(cfg: dict) -> pd.DataFrame:
    # 1) 电量
    elec = load_electricity(cfg)
    dates = pd.DatetimeIndex(elec["date"])

    # 2) 天气（失败回退合成温度）
    weather = fetch_weather(cfg, fallback_df=elec)

    # 3) 日历
    cal = build_calendar_features(dates)

    # 4) 经济
    eco = load_economic(cfg, dates)

    # 合并
    df = elec.merge(weather, on="date", how="left")
    df = df.merge(cal, on="date", how="left")
    df = df.merge(eco, on="date", how="left")
    df["temperature"] = df["temperature"].interpolate().bfill().ffill()

    # 派生度日特征
    base = cfg["decoupling"]["temperature_base"]
    df["CDD"] = np.clip(df["temperature"] - (base + 6), 0, None)  # 制冷度日
    df["HDD"] = np.clip((base - 8) - df["temperature"], 0, None)  # 采暖度日
    df["log_elec"] = np.log(df["electricity"])

    ensure_dir(abspath(cfg["data"]["merged_csv"]))
    out = abspath(cfg["data"]["merged_csv"])
    df.to_csv(out, index=False)
    log.info(f"整合数据集已保存: {out}  形状={df.shape}")
    log.info(f"字段: {list(df.columns)}")
    return df


if __name__ == "__main__":
    cfg = load_config()
    d = build_dataset(cfg)
    print(d.head())
