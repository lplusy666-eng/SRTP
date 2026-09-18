"""
日历/节假日特征
--------------
使用 chinesecalendar 库生成中国法定节假日、调休、周末标记。
这些是"非经济扰动"的重要来源，需在特征解耦阶段剔除。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import get_logger

log = get_logger("data.calendar")

try:
    import chinese_calendar as cc
    _HAS_CC = True
except Exception:
    _HAS_CC = False
    log.warning("未安装 chinesecalendar，节假日将退化为仅周末判断")


def build_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """返回日历特征表。"""
    df = pd.DataFrame({"date": dates})
    df["dayofweek"] = dates.dayofweek
    df["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    df["month"] = dates.month
    df["dayofyear"] = dates.dayofyear

    is_holiday = np.zeros(len(dates), dtype=int)
    is_workday = np.ones(len(dates), dtype=int)
    holiday_name = [""] * len(dates)

    if _HAS_CC:
        for i, d in enumerate(dates):
            dt = d.date()
            try:
                on_holiday, name = cc.get_holiday_detail(dt)
                is_holiday[i] = int(on_holiday)
                is_workday[i] = int(cc.is_workday(dt))
                holiday_name[i] = name or ""
            except Exception:
                # 超出库覆盖年份时退化
                is_holiday[i] = int(d.dayofweek >= 5)
                is_workday[i] = int(d.dayofweek < 5)
    else:
        is_holiday = (dates.dayofweek >= 5).astype(int).values
        is_workday = (dates.dayofweek < 5).astype(int).values

    df["is_holiday"] = is_holiday
    df["is_workday"] = is_workday
    df["holiday_name"] = holiday_name

    # 春节窗口标记（前后各若干天，用电结构显著变化）
    df["is_spring_festival"] = df["holiday_name"].str.contains("Spring Festival|春节", na=False).astype(int)

    # 距离春节天数（用于剥离春节前后2-3周的复工/停工爬坡效应）
    df["days_to_cny"] = _days_to_spring_festival(df)

    log.info(f"日历特征已生成: {len(df)} 天, 其中节假日 {int(is_holiday.sum())} 天")
    return df


def _days_to_spring_festival(df: pd.DataFrame) -> np.ndarray:
    """每天到最近一次春节首日的天数（负=春节前，正=春节后）。"""
    dates = pd.DatetimeIndex(df["date"])
    # 每年春节首日 = 该年 Spring Festival 假期块的最早日期
    cny_starts = {}
    sf = df[df["is_spring_festival"] == 1]
    for yr, grp in sf.groupby(pd.DatetimeIndex(sf["date"]).year):
        cny_starts[yr] = pd.Timestamp(grp["date"].min())
    # 若某年缺失（库未覆盖），用近似农历日期兜底
    result = np.zeros(len(dates))
    all_cny = sorted(cny_starts.values())
    if not all_cny:
        return result
    for i, d in enumerate(dates):
        nearest = min(all_cny, key=lambda c: abs((d - c).days))
        result[i] = (d - nearest).days
    return result


if __name__ == "__main__":
    dts = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    print(build_calendar_features(dts).head(10))
