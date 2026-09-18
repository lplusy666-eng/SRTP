"""
温度修正 & 节假日平滑模块

功能:
  - 温度-用电分段线性回归模型（取暖区 / 舒适区 / 制冷区）
  - HDD/CDD 度日法温度修正
  - 温度效应分解与中性化
  - 中国节假日日历生成
  - 节假日效应平滑
  - 星期效应修正

方法参考:
  - Lan et al. (2025): 温度调整、春节效应处理
  - Stundziene et al. (2023): 温度修正用于去除非经济波动
"""

import logging
from typing import Optional, Dict, List, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor

try:
    from .config import (
        TEMPERATURE_BASELINE,
        HEATING_THRESHOLD,
        COOLING_THRESHOLD,
        HOLIDAY_SMOOTH_BEFORE,
        HOLIDAY_SMOOTH_AFTER,
        SPRING_FESTIVAL_WINDOW,
    )
except ImportError:
    from config import (
        TEMPERATURE_BASELINE,
        HEATING_THRESHOLD,
        COOLING_THRESHOLD,
        HOLIDAY_SMOOTH_BEFORE,
        HOLIDAY_SMOOTH_AFTER,
        SPRING_FESTIVAL_WINDOW,
    )

logger = logging.getLogger(__name__)


# ============================================================================
# 温度修正
# ============================================================================

def compute_degree_days(
    temp_series: pd.Series,
    base_temp: float = TEMPERATURE_BASELINE,
) -> pd.DataFrame:
    """
    计算 HDD (Heating Degree Days) 和 CDD (Cooling Degree Days)。

    HDD = max(0, base_temp - T)  → 取暖需求
    CDD = max(0, T - base_temp)  → 制冷需求

    度日是衡量建筑供暖/制冷能源需求的标准化指标。

    参数:
        temp_series: 温度时间序列（日度或月度均值）
        base_temp: 基准舒适温度（中国标准 18°C）

    返回:
        DataFrame with columns: [hdd, cdd, temp_deviation]
    """
    hdd = np.maximum(0, base_temp - temp_series)
    cdd = np.maximum(0, temp_series - base_temp)
    deviation = temp_series - base_temp

    return pd.DataFrame({
        "hdd": hdd,
        "cdd": cdd,
        "temp_deviation": deviation,
    }, index=temp_series.index)


def fit_temperature_model(
    consumption: pd.Series,
    temperature: pd.Series,
    method: Literal["piecewise", "linear", "ridge", "huber"] = "piecewise",
    add_month_dummies: bool = True,
    heating_threshold: float = HEATING_THRESHOLD,
    cooling_threshold: float = COOLING_THRESHOLD,
) -> dict:
    """
    拟合温度-用电关系模型。

    方法:
        "piecewise": 分段线性回归（默认）
            E = β₀ + β₁·heating_effect + β₂·cooling_effect + Σγ_m·month_m + ε
            其中:
              heating_effect = max(0, heating_threshold - T)  # 取暖需求
              cooling_effect = max(0, T - cooling_threshold)  # 制冷需求

        "linear": 单段线性回归
            E = β₀ + β₁·T + Σγ_m·month_m + ε

        "ridge": 岭回归（处理共线性）
        "huber": Huber 回归（对异常值鲁棒）

    参数:
        consumption: 用电量时间序列
        temperature: 温度时间序列
        method: 回归方法
        add_month_dummies: 是否添加月份哑变量（控制季节性基线）
        heating_threshold: 取暖阈值温度
        cooling_threshold: 制冷阈值温度

    返回:
        模型参数字典:
            - heating_coef: 取暖系数
            - cooling_coef: 制冷系数
            - intercept: 截距
            - r_squared: 拟合优度
            - month_coefs: 月份哑变量系数（如有）
            - method: 使用方法
            - thresholds: (heating_threshold, cooling_threshold)
    """
    # 准备数据：对齐时间索引
    df = pd.DataFrame({
        "consumption": consumption,
        "temperature": temperature,
    }).dropna()

    if len(df) < 12:
        logger.warning(f"有效数据点 ({len(df)}) 过少，可能无法可靠拟合")
        return {
            "heating_coef": 0.0,
            "cooling_coef": 0.0,
            "intercept": df["consumption"].mean(),
            "r_squared": 0.0,
            "month_coefs": {},
            "method": "fallback_mean",
            "thresholds": (heating_threshold, cooling_threshold),
        }

    X_data = df.copy()

    # 构建特征矩阵
    feature_cols = []

    if method == "piecewise":
        # 取暖效应 = 低于取暖阈值的度数
        X_data["heating_effect"] = np.maximum(0, heating_threshold - X_data["temperature"])
        # 制冷效应 = 高于制冷阈值的度数
        X_data["cooling_effect"] = np.maximum(0, X_data["temperature"] - cooling_threshold)
        feature_cols.extend(["heating_effect", "cooling_effect"])
    else:
        # 单段线性
        X_data["temperature_raw"] = X_data["temperature"]
        feature_cols.append("temperature_raw")

    # 月份哑变量
    if add_month_dummies:
        X_data["month"] = X_data.index.month
        month_dummies = pd.get_dummies(X_data["month"], prefix="month", drop_first=True)
        month_dummies.index = X_data.index
        X_data = pd.concat([X_data, month_dummies], axis=1)
        feature_cols.extend(month_dummies.columns.tolist())

    # 拟合模型
    X = X_data[feature_cols]
    y = X_data["consumption"]

    if method == "ridge":
        model = Ridge(alpha=1.0)
    elif method == "huber":
        model = HuberRegressor(max_iter=1000)
    else:
        model = LinearRegression()

    model.fit(X, y)
    y_pred = model.predict(X)

    # R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 提取系数
    coef_dict = dict(zip(feature_cols, model.coef_))

    result = {
        "intercept": model.intercept_,
        "r_squared": r_squared,
        "method": method,
        "thresholds": (heating_threshold, cooling_threshold),
        "n_samples": len(df),
    }

    if method == "piecewise":
        result["heating_coef"] = coef_dict.get("heating_effect", 0.0)
        result["cooling_coef"] = coef_dict.get("cooling_effect", 0.0)
    else:
        result["temp_coef"] = coef_dict.get("temperature_raw", 0.0)

    if add_month_dummies:
        result["month_coefs"] = {
            k: v for k, v in coef_dict.items() if k.startswith("month_")
        }

    logger.info(f"温度模型拟合完成 (method={method}): "
                f"R²={r_squared:.4f}, "
                f"取暖系数={result.get('heating_coef', 'N/A')}, "
                f"制冷系数={result.get('cooling_coef', 'N/A')}")

    return result


def temperature_correct(
    consumption: pd.Series,
    temperature: pd.Series,
    model: dict,
) -> pd.Series:
    """
    温度修正：去除空调/取暖对用电的非经济影响。

    方法:
        corrected = consumption - temperature_effect

    其中 temperature_effect 根据模型类型计算:
        - piecewise: heating_coef * max(0, thr_heat - T) + cooling_coef * max(0, T - thr_cool)
        - linear: temp_coef * T
        - fallback_mean: 0（无修正）

    修正后的用电量代表了"在舒适温度下应有的用电水平"，
    排除了极端温度导致的额外用电。

    参数:
        consumption: 原始用电量
        temperature: 温度序列
        model: fit_temperature_model 返回的模型字典

    返回:
        温度中性化用电量（weather-normalized consumption）
    """
    # 对齐索引
    common_idx = consumption.dropna().index.intersection(temperature.dropna().index)
    if len(common_idx) == 0:
        logger.warning("consumption 和 temperature 无重叠时间点，跳过温度修正")
        return consumption

    consumption = consumption.reindex(common_idx)
    temperature = temperature.reindex(common_idx)

    # 计算温度效应
    method = model.get("method", "piecewise")

    if method == "piecewise" or method == "ridge" or method == "huber":
        h_thr, c_thr = model["thresholds"]
        temp_effect = (
            model.get("heating_coef", 0) * np.maximum(0, h_thr - temperature)
            + model.get("cooling_coef", 0) * np.maximum(0, temperature - c_thr)
        )
    elif method == "linear":
        temp_effect = model.get("temp_coef", 0) * temperature
    else:  # fallback_mean
        temp_effect = pd.Series(0.0, index=consumption.index)

    # 修正
    corrected = consumption - temp_effect

    logger.info(f"温度修正完成: "
                f"原始均值={consumption.mean():.2f}, "
                f"修正后均值={corrected.mean():.2f}, "
                f"平均温度效应={temp_effect.mean():.2f}")

    return corrected


def fit_and_correct(
    consumption: pd.Series,
    temperature: pd.Series,
    method: str = "piecewise",
    **fit_kwargs,
) -> Tuple[pd.Series, dict]:
    """
    一步完成温度模型拟合 + 修正。

    参数:
        consumption: 原始用电量
        temperature: 温度序列
        method: 回归方法

    返回:
        (修正后用电量, 模型参数字典)
    """
    model = fit_temperature_model(consumption, temperature, method=method, **fit_kwargs)
    corrected = temperature_correct(consumption, temperature, model)
    return corrected, model


# ============================================================================
# 节假日处理
# ============================================================================

def build_holiday_calendar(
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    生成中国节假日日历。

    使用 chinese-calendar 库处理农历节日（春节、端午、中秋），
    同时包含公历固定节假日。

    参数:
        start_year: 起始年份
        end_year: 结束年份

    返回:
        DataFrame (DatetimeIndex) with columns:
            - is_holiday: 是否法定节假日
            - is_workday: 是否调休工作日
            - holiday_name: 节日名称
            - days_to_spring_festival: 距离春节天数
    """
    try:
        from chinese_calendar import is_holiday as cn_is_holiday
        from chinese_calendar import is_workday as cn_is_workday
        from chinese_calendar import get_holiday_detail
    except ImportError:
        logger.error(
            "chinese-calendar 库未安装。请运行: pip install chinese-calendar"
        )
        # 降级：仅标记周末
        date_range = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        return pd.DataFrame({
            "is_holiday": date_range.dayofweek >= 5,
            "is_workday": date_range.dayofweek < 5,
            "holiday_name": "N/A",
            "days_to_spring_festival": 999,
        }, index=date_range)

    date_range = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")

    holiday_info = []
    for d in date_range:
        d_date = d.to_pydatetime().date()
        is_hol = cn_is_holiday(d_date)
        is_work = cn_is_workday(d_date)
        try:
            _, name = get_holiday_detail(d_date)
        except Exception:
            name = ""
        holiday_info.append({
            "is_holiday": is_hol,
            "is_workday": is_work,
            "holiday_name": name if name else "",
        })

    calendar = pd.DataFrame(holiday_info, index=date_range)

    # 计算距离春节的天数
    spring_festival_dates = []
    import datetime as dt
    for year in range(start_year, end_year + 1):
        try:
            from chinese_calendar import get_holidays, get_holiday_detail
            holidays_in_year = get_holidays(dt.date(year, 1, 1), dt.date(year, 12, 31))
            for date_obj in holidays_in_year:
                _, name = get_holiday_detail(date_obj)
                if name and ("春节" in str(name) or "Spring" in str(name)):
                    spring_festival_dates.append(pd.Timestamp(date_obj))
                    break
        except Exception:
            pass

    calendar["days_to_spring_festival"] = 999
    for sf_date in spring_festival_dates:
        year_mask = calendar.index.year == sf_date.year
        if year_mask.any():
            calendar.loc[year_mask, "days_to_spring_festival"] = (
                (calendar.loc[year_mask].index - sf_date).days
            )

    # 春节影响期标记
    calendar["is_spring_festival_period"] = (
        calendar["days_to_spring_festival"].abs() <= SPRING_FESTIVAL_WINDOW
    )

    logger.info(f"节假日日历生成: {start_year}-{end_year}, "
                f"{calendar['is_holiday'].sum()} 个节假日")

    return calendar


def holiday_smooth(
    series: pd.Series,
    holiday_col: str = "is_holiday",
    window_before: int = HOLIDAY_SMOOTH_BEFORE,
    window_after: int = HOLIDAY_SMOOTH_AFTER,
    calendar: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    节假日平滑处理。

    方法: neighbor_avg
        对节假日期间的每个值，用节前N天+节后N天的非假期日均值替换。
        这保留了"假期期间的典型用电水平"，但去除了假日本身的短期扰动。

    例如：春节7天假期 → 用节前3天非假日的均值填补，使假期数据平滑。

    参数:
        series: 日度时间序列
        holiday_col: 节假日标记列名（仅当 calendar 为 None 时有意义）
        window_before: 节前窗口（天）
        window_after: 节后窗口（天）
        calendar: build_holiday_calendar 输出的节假日日历

    返回:
        节假日平滑后的序列
    """
    if calendar is None:
        logger.warning("未提供节假日日历，跳过节假日平滑")
        return series

    # 确保索引对齐
    common_idx = series.index.intersection(calendar.index)
    if len(common_idx) == 0:
        return series

    result = series.copy()
    holiday_mask = calendar.loc[common_idx, "is_holiday"]

    if not holiday_mask.any():
        logger.info("数据期间无节假日，跳过平滑")
        return result

    # 找出连续的假期段
    holiday_periods = _find_consecutive_periods(holiday_mask)

    n_smoothed = 0
    for start, end in holiday_periods:
        # 确定平滑参考窗口
        ref_start = start - pd.Timedelta(days=window_before + 1)
        ref_end = end + pd.Timedelta(days=window_after + 1)

        # 在参考窗口中取非假期日的值
        ref_mask = (
            (series.index >= ref_start)
            & (series.index <= ref_end)
            & ~calendar["is_holiday"]
        )
        ref_values = series[ref_mask].dropna()

        if len(ref_values) == 0:
            # 扩大搜索窗口
            ref_mask_wide = (
                (series.index >= start - pd.Timedelta(days=30))
                & (series.index <= end + pd.Timedelta(days=30))
                & ~calendar["is_holiday"]
            )
            ref_values = series[ref_mask_wide].dropna()

        if len(ref_values) > 0:
            replacement = ref_values.mean()
            holiday_idx = series.index[(series.index >= start) & (series.index <= end)]
            result.loc[holiday_idx] = replacement
            n_smoothed += len(holiday_idx)

    logger.info(f"节假日平滑: {n_smoothed} 个数据点被平滑 "
                f"({n_smoothed / max(1, len(series)) * 100:.1f}%)")

    return result


def _find_consecutive_periods(
    mask: pd.Series,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """找出连续的 True 段。"""
    periods = []
    in_period = False
    start = None

    for idx in mask.index:
        if mask.loc[idx] and not in_period:
            start = idx
            in_period = True
        elif not mask.loc[idx] and in_period:
            periods.append((start, idx - pd.Timedelta(days=1)))
            in_period = False

    if in_period:
        periods.append((start, mask.index[-1]))

    return periods


# ============================================================================
# 星期效应修正
# ============================================================================

def weekday_adjust(daily_series: pd.Series) -> pd.Series:
    """
    星期效应修正。

    工作日/周末用电模式显著不同。此函数用"该日用电 / 当周均值"的比率
    来量化每日的星期效应，然后移除该效应。

    方法:
        1. 计算每周均值
        2. 计算每日相对周均值的比率
        3. 用该比率反推"无星期效应"的值

    参数:
        daily_series: 日度时间序列

    返回:
        星期效应修正后的序列
    """
    series = daily_series.dropna()
    if len(series) < 14:  # 少于2周数据，无法可靠修正
        return daily_series

    df = pd.DataFrame({"value": series})
    df["dow"] = df.index.dayofweek  # 0=Mon, 6=Sun
    df["week"] = df.index.isocalendar().week
    df["year"] = df.index.year

    # 每周均值
    df["week_mean"] = df.groupby(["year", "week"])["value"].transform("mean")

    # 星期指数 = 该日值 / 周均值
    df["dow_index"] = df["value"] / df["week_mean"].replace(0, np.nan)

    # 各星期的平均指数
    avg_dow_index = df.groupby("dow")["dow_index"].mean()

    # 修正 = 原始值 / 星期指数 * 全周平均指数
    all_mean = avg_dow_index.mean()
    dow_index_mapped = df["dow"].map(avg_dow_index)
    df["adjusted"] = df["value"] / dow_index_mapped * all_mean

    result = df["adjusted"].reindex(daily_series.index)

    logger.info(f"星期效应修正完成: 工作日/周末指数范围 "
                f"[{avg_dow_index.min():.3f}, {avg_dow_index.max():.3f}]")

    return result


# ============================================================================
# 一站式修正入口
# ============================================================================

def apply_all_corrections(
    df: pd.DataFrame,
    consumption_col: str = "consumption",
    temperature_col: Optional[str] = "temperature",
    use_holiday_smooth: bool = True,
    use_weekday_adjust: bool = True,
    temp_method: str = "piecewise",
) -> pd.DataFrame:
    """
    一站式修正入口：依次应用温度修正、节假日平滑、星期效应修正。

    参数:
        df: 清洗后的 DataFrame（DatetimeIndex）
        consumption_col: 用电量列名
        temperature_col: 温度列名（None 则跳过温度修正）
        use_holiday_smooth: 是否启用节假日平滑
        use_weekday_adjust: 是否启用星期效应修正
        temp_method: 温度修正方法

    返回:
        修正后的 DataFrame，新增列:
            - {consumption}_temp_corrected: 温度修正后用电量
            - {consumption}_corrected: 全流程修正后用电量
    """
    df = df.copy()

    if consumption_col not in df.columns:
        raise ValueError(f"用电量列 '{consumption_col}' 不存在")

    consumption = df[consumption_col].copy()
    corrected = consumption.copy()

    # ---- 1. 温度修正 ----
    temp_model = None
    if temperature_col and temperature_col in df.columns:
        temperature = df[temperature_col]
        corrected, temp_model = fit_and_correct(
            corrected, temperature, method=temp_method
        )
        df[f"{consumption_col}_temp_corrected"] = corrected
        logger.info(f"温度修正已应用 (method={temp_method})")
    else:
        logger.info("无温度数据，跳过温度修正")

    # ---- 2. 节假日平滑 ----
    if use_holiday_smooth:
        try:
            start_year = df.index.min().year
            end_year = df.index.max().year
            calendar = build_holiday_calendar(start_year, end_year)

            # 重新索引日历以匹配数据
            calendar = calendar.reindex(df.index)

            corrected = holiday_smooth(corrected, calendar=calendar)
            df[f"{consumption_col}_holiday_smoothed"] = corrected
            logger.info("节假日平滑已应用")
        except Exception as e:
            logger.warning(f"节假日平滑失败: {e}")

    # ---- 3. 星期效应修正 ----
    if use_weekday_adjust:
        try:
            corrected = weekday_adjust(corrected)
            df[f"{consumption_col}_weekday_adjusted"] = corrected
            logger.info("星期效应修正已应用")
        except Exception as e:
            logger.warning(f"星期效应修正失败: {e}")

    # ---- 最终修正结果 ----
    df[f"{consumption_col}_corrected"] = corrected

    # 报告修正效果
    orig_std = consumption.std()
    corr_std = corrected.std()
    logger.info(f"全流程修正完成: "
                f"原始 std={orig_std:.2f}, "
                f"修正后 std={corr_std:.2f} "
                f"(减少 {(1 - corr_std / max(1e-10, orig_std)) * 100:.1f}%)")

    return df
