"""
电量数据加载器
--------------
优先读取真实数据 data/raw/electricity.csv (列: date, electricity)。
若不存在且配置允许，则生成"物理可解释"的合成电量序列，内部结构：

    电量 = 经济趋势 × 季节性 × 温度效应 × 节假日效应 × 噪声 + 经济冲击

其中"经济冲击"是我们埋入的真值异常，供后续 VAE 检测验证。
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir

log = get_logger("data.electricity")


def _synthetic_temperature(dates: pd.DatetimeIndex) -> np.ndarray:
    """生成合成日均温度（用于内部构造，真实流程用 Open-Meteo）。"""
    doy = dates.dayofyear.values
    # 年周期温度：夏热冬冷，杭州气候近似
    temp = 17 + 12 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    temp += np.random.normal(0, 2.5, len(dates))
    return temp


def generate_synthetic_electricity(dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """构造带真值异常的合成电量数据。返回含真值列，便于评估。"""
    rng = np.random.default_rng(seed)
    n = len(dates)
    t = np.arange(n)

    # 1) 经济趋势：年均增长 ~5%，叠加缓慢景气波动
    annual_growth = 0.05
    trend = 100 * (1 + annual_growth) ** (t / 365.25)
    business_cycle = 1 + 0.04 * np.sin(2 * np.pi * t / (365.25 * 2.5))  # ~2.5年景气周期
    trend = trend * business_cycle

    # 2) 温度效应：制冷(CDD)+采暖(HDD)拉高用电
    temp = _synthetic_temperature(dates)
    cdd = np.clip(temp - 24, 0, None)   # 制冷度
    hdd = np.clip(10 - temp, 0, None)   # 采暖度
    temp_effect = 1 + 0.020 * cdd + 0.012 * hdd

    # 3) 周季节性：工作日高、周末低
    dow = dates.dayofweek.values
    weekly = np.where(dow < 5, 1.0, 0.82)

    # 4) 节假日效应：用真实节假日日期注入（与 calendar 模块严格对齐，保证可解耦）
    holiday_effect = np.ones(n)
    try:
        import chinese_calendar as cc
        for i, d in enumerate(dates):
            dt = d.date()
            on_holiday, name = cc.get_holiday_detail(dt)
            if on_holiday and name:
                if "Spring Festival" in name:
                    holiday_effect[i] = 0.68      # 春节假期用电骤降
                elif "National Day" in name:
                    holiday_effect[i] = 0.80
                else:
                    holiday_effect[i] = 0.88
        # 春节前后爬坡（复工/停工过渡，±14天平滑过渡）
        sf_days = np.array([1 if (cc.get_holiday_detail(d.date())[1] or "").find("Spring Festival") >= 0
                            else 0 for d in dates])
        for center in np.where(np.diff(np.concatenate([[0], sf_days])) == 1)[0]:
            for k in range(-14, 15):
                j = center + k
                if 0 <= j < n and holiday_effect[j] == 1.0:
                    holiday_effect[j] = 1 - 0.20 * np.exp(-(k / 8.0) ** 2)
    except Exception:
        for yr in np.unique(dates.year):
            cny = pd.Timestamp(f"{yr}-02-05")
            mask = (dates >= cny - pd.Timedelta(days=1)) & (dates <= cny + pd.Timedelta(days=6))
            holiday_effect[mask] = 0.68

    # 5) 随机噪声
    noise = rng.normal(1.0, 0.015, n)

    base = trend * temp_effect * weekly * holiday_effect * noise

    # 6) 埋入"经济冲击"真值异常（非温度、非节假日可解释）
    anomaly_flag = np.zeros(n, dtype=int)
    shock = np.zeros(n)
    # 定义若干经济事件：起始索引、持续天数、幅度(相对)
    events = [
        (int(n * 0.18), 25, -0.12, "制造业订单下滑"),
        (int(n * 0.33), 40, -0.18, "疫情式外部冲击"),
        (int(n * 0.55), 20, +0.10, "重大项目投产提振"),
        (int(n * 0.72), 30, -0.09, "行业限产"),
        (int(n * 0.88), 18, +0.08, "促消费政策拉动"),
    ]
    event_log = []
    for start, dur, amp, name in events:
        end = min(start + dur, n)
        ramp = np.hanning(end - start)  # 平滑的冲击包络
        shock[start:end] += amp * ramp
        anomaly_flag[start:end] = 1
        event_log.append((str(dates[start].date()), str(dates[end - 1].date()), amp, name))

    electricity = base * (1 + shock)

    df = pd.DataFrame({
        "date": dates,
        "electricity": electricity,
        "_true_anomaly": anomaly_flag,   # 真值标签（真实数据没有，仅评估用）
        "_synthetic_temp": temp,          # 供无天气API时兜底
    })
    log.info(f"合成电量已生成: {n} 天, 埋入 {len(events)} 个经济事件")
    for e in event_log:
        log.info(f"  事件 {e[0]}~{e[1]} 幅度{e[2]:+.0%} 原因:{e[3]}")
    return df


def load_electricity(cfg: dict) -> pd.DataFrame:
    """主入口：真实优先，缺失则合成。"""
    csv_path = abspath(cfg["data"]["electricity_csv"])
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
        log.info(f"已读取真实电量数据: {csv_path} ({len(df)} 行)")
        if "_true_anomaly" not in df.columns:
            df["_true_anomaly"] = np.nan
        return df

    if not cfg["data"].get("use_synthetic_if_missing", True):
        raise FileNotFoundError(f"未找到电量数据且未启用合成: {csv_path}")

    dates = pd.date_range(cfg["project"]["start_date"],
                          cfg["project"]["end_date"],
                          freq=cfg["project"]["freq"])
    df = generate_synthetic_electricity(dates)
    ensure_dir(csv_path)
    df.to_csv(csv_path, index=False)
    log.info(f"合成电量已保存: {csv_path}")
    return df


if __name__ == "__main__":
    from utils import load_config
    cfg = load_config()
    d = load_electricity(cfg)
    print(d.head())
    print(d.describe())
