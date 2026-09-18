"""
宏观经济指标加载
----------------
真实数据放 data/raw/economic.csv (列: date, gdp_yoy, pmi, ...，月/季度频)。
本模块会将低频经济指标按日期前向填充对齐到日频，作为"经济景气"参照。
缺失时生成合成经济指标（与电量趋势弱相关，供演示）。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir

log = get_logger("data.economic")


def _synthetic_economic(dates: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    """生成月频合成宏观指标：GDP同比、PMI。"""
    rng = np.random.default_rng(seed)
    months = pd.date_range(dates.min(), dates.max(), freq="MS")
    t = np.arange(len(months))
    gdp_yoy = 5.5 + 1.2 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 0.3, len(months))
    pmi = 50 + 1.5 * np.sin(2 * np.pi * t / 30 + 0.5) + rng.normal(0, 0.6, len(months))
    return pd.DataFrame({"date": months,
                         "gdp_yoy": gdp_yoy.round(2),
                         "pmi": pmi.round(1)})


def load_economic(cfg: dict, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """加载并对齐到日频。"""
    csv_path = abspath(cfg["data"]["economic_csv"])
    if Path(csv_path).exists():
        eco = pd.read_csv(csv_path, parse_dates=["date"])
        log.info(f"已读取真实经济数据: {csv_path}")
    elif cfg["data"].get("use_synthetic_economic_if_missing", True):
        eco = _synthetic_economic(target_dates)
        ensure_dir(csv_path)
        eco.to_csv(csv_path, index=False)
        log.info(f"合成经济指标已保存: {csv_path}")
    else:
        raise FileNotFoundError(f"未找到经济数据: {csv_path}")

    # 对齐到日频（前向填充）
    daily = pd.DataFrame({"date": target_dates})
    eco = eco.sort_values("date")
    merged = pd.merge_asof(daily, eco, on="date", direction="backward")
    merged = merged.ffill().bfill()
    return merged


if __name__ == "__main__":
    from utils import load_config
    cfg = load_config()
    dts = pd.date_range(cfg["project"]["start_date"], cfg["project"]["end_date"], freq="D")
    print(load_economic(cfg, dts).head())
