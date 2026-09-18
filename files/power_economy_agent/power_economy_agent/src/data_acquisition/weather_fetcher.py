"""
天气数据获取
------------
使用 Open-Meteo 历史天气 API（免费、无需API Key）。
获取日均温度，用于后续计算采暖度日(HDD)/制冷度日(CDD)。

若无网络，则回退到电量模块生成的合成温度，保证流程可跑通。
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir

log = get_logger("data.weather")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(cfg: dict, fallback_df: pd.DataFrame = None) -> pd.DataFrame:
    """返回列: date, temperature (日均温)。"""
    wcfg = cfg["data"]["weather"]
    cache = abspath(wcfg["cache_csv"])

    if Path(cache).exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        log.info(f"已读取天气缓存: {cache} ({len(df)} 行)")
        return df

    if wcfg.get("enabled", True):
        try:
            params = {
                "latitude": wcfg["latitude"],
                "longitude": wcfg["longitude"],
                "start_date": cfg["project"]["start_date"],
                "end_date": cfg["project"]["end_date"],
                "daily": "temperature_2m_mean",
                "timezone": "Asia/Shanghai",
            }
            log.info("正在请求 Open-Meteo 历史天气 ...")
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            r.raise_for_status()
            daily = r.json()["daily"]
            df = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "temperature": daily["temperature_2m_mean"],
            })
            df["temperature"] = df["temperature"].interpolate().bfill().ffill()
            ensure_dir(cache)
            df.to_csv(cache, index=False)
            log.info(f"天气数据已获取并缓存: {cache} ({len(df)} 行)")
            return df
        except Exception as e:
            log.warning(f"天气API请求失败({e})，回退到合成温度")

    # 回退：使用电量合成时的温度
    if fallback_df is not None and "_synthetic_temp" in fallback_df.columns:
        df = fallback_df[["date", "_synthetic_temp"]].rename(
            columns={"_synthetic_temp": "temperature"})
        log.info("使用合成温度作为天气数据")
        return df

    raise RuntimeError("无法获取天气数据，且无回退温度可用")


if __name__ == "__main__":
    from utils import load_config
    from electricity_loader import load_electricity
    cfg = load_config()
    elec = load_electricity(cfg)
    w = fetch_weather(cfg, fallback_df=elec)
    print(w.head())
