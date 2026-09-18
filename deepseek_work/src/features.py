"""
特征工程模块（核心）

功能:
  - 同比增速 (YoY) / 环比增速 (MoM) / 累计增速
  - 电力容量净增量变化率（Lan et al. 2025 核心特征）
  - 移动平均 / 滚动窗口统计
  - MIDAS 高频→低频数据聚合（Stundziene et al. 2023）
  - 多维行业用电分解
  - 统一特征矩阵生成

参考论文:
  - Lan et al. (2025): Composite GDP nowcasting using macroeconomic variables
    and electricity data. PLOS ONE.
  - Stundziene et al. (2023): Nowcasting Economic Activity Using Electricity
    Market Data: The Case of Lithuania. Economies.
"""

import logging
from typing import Optional, Dict, List, Union, Literal, Tuple

import numpy as np
import pandas as pd

try:
    from .config import (
        YOY_PERIODS,
        CAPACITY_WINDOW_MONTHS,
        MOVING_AVG_WINDOWS,
        MIDAS_N_LAGS,
        MIDAS_WEIGHT_TYPE,
        SECTOR_LABELS,
        QUARTER_MONTH_MAP,
    )
except ImportError:
    from config import (
        YOY_PERIODS,
        CAPACITY_WINDOW_MONTHS,
        MOVING_AVG_WINDOWS,
        MIDAS_N_LAGS,
        MIDAS_WEIGHT_TYPE,
        SECTOR_LABELS,
        QUARTER_MONTH_MAP,
    )

logger = logging.getLogger(__name__)


# ============================================================================
# 基础增长率
# ============================================================================

def compute_yoy_growth(
    series: pd.Series,
    periods: int = YOY_PERIODS,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    计算同比增速 (Year-over-Year growth rate)。

    公式:
        YoY_t = (E_t - E_{t-periods}) / E_{t-periods} * 100

    参数:
        series: 时间序列数据（通常为月度用电量）
        periods: 同比周期（月度数据为12）
        min_periods: 最小非NaN期数

    返回:
        同比增速 (%)，前 periods 期为 NaN
    """
    shifted = series.shift(periods)
    yoy = (series - shifted) / shifted.abs() * 100
    # 避免分母为零
    yoy = yoy.replace([np.inf, -np.inf], np.nan)
    return yoy


def compute_mom_growth(
    series: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """
    计算环比增速 (Month-over-Month growth rate)。

    公式:
        MoM_t = (E_t - E_{t-1}) / E_{t-1} * 100

    参数:
        series: 时间序列数据
        periods: 环比周期（月环比=1，季环比=3）

    返回:
        环比增速 (%)，前 periods 期为 NaN
    """
    shifted = series.shift(periods)
    mom = (series - shifted) / shifted.abs() * 100
    mom = mom.replace([np.inf, -np.inf], np.nan)
    return mom


def compute_cumulative_growth(series: pd.Series) -> pd.Series:
    """
    计算累计同比增速。

    公式:
        CumYoY_t = (sum_{i in year(t)} E_i / sum_{i in year(t-1)} E_i - 1) * 100

    即：年初至今累计用电量 / 去年同期累计用电量 - 1

    参数:
        series: 月度时间序列

    返回:
        累计同比增速 (%)
    """
    # 年内累计和
    cumsum = series.groupby(series.index.year).cumsum()

    # 去年同期的累计（将 cumsum 向前平移12个月）
    cumsum_prev_year = cumsum.shift(12)

    cumulative_growth = (cumsum / cumsum_prev_year - 1) * 100
    cumulative_growth = cumulative_growth.replace([np.inf, -np.inf], np.nan)

    return cumulative_growth


def compute_moving_average(
    series: pd.Series,
    window: int = 3,
    center: bool = True,
) -> pd.Series:
    """
    计算移动平均。

    参数:
        series: 时间序列
        window: 窗口大小
        center: 是否居中（True = 过去+未来各半，False = 仅过去）

    返回:
        移动平均序列
    """
    return series.rolling(window=window, center=center, min_periods=1).mean()


def compute_rolling_stats(
    series: pd.Series,
    windows: List[int] = MOVING_AVG_WINDOWS,
    stats: List[str] = ["mean", "std"],
) -> pd.DataFrame:
    """
    多窗口滚动统计量。

    参数:
        series: 时间序列
        windows: 窗口大小列表
        stats: 统计量列表 ("mean", "std", "min", "max", "skew", "kurt")

    返回:
        DataFrame，列名如 "roll_3_mean", "roll_3_std"
    """
    result = pd.DataFrame(index=series.index)
    for w in windows:
        roll = series.rolling(window=w, center=True, min_periods=min(3, w))
        for stat in stats:
            col_name = f"roll_{w}_{stat}"
            if stat == "mean":
                result[col_name] = roll.mean()
            elif stat == "std":
                result[col_name] = roll.std()
            elif stat == "min":
                result[col_name] = roll.min()
            elif stat == "max":
                result[col_name] = roll.max()
            elif stat == "skew":
                result[col_name] = roll.skew()
            elif stat == "kurt":
                result[col_name] = roll.kurt()
    return result


# ============================================================================
# 电力容量净增量变化率（Lan et al. 2025 核心特征）
# ============================================================================

def compute_capacity_net_increment(
    monthly_consumption: pd.Series,
    capacity_window: int = CAPACITY_WINDOW_MONTHS,
    lag_periods: int = YOY_PERIODS,
) -> pd.DataFrame:
    """
    计算电力容量净增量变化率。

    参考: Lan et al. (2025), PLOS ONE

    公式:
        ΔC_t  = sum(E_{t-1}, E_{t-2}, ..., E_{t-k})  # 过去k个月新增用电容量
        ΔC_rate_t = (ΔC_t - ΔC_{t-lag}) / E_{t-lag} * 100

    其中:
        - k = capacity_window (默认3个月): 容量累计窗口
        - lag = lag_periods (默认12个月): 同比周期
        - E_{t-lag}: 去年同期用电量，用于归一化

    经济含义:
        反映未来经济活动强度的前瞻性指标。新装容量/新开工项目会增加用电需求，
        但当前月份尚未完全释放。变化率上升预示未来经济活跃度增强。

    参数:
        monthly_consumption: 月度用电量序列
        capacity_window: 新增容量累计窗口（月数）
        lag_periods: 同比周期

    返回:
        DataFrame with columns:
            - capacity_delta: 过去k个月的累计用电量（代理容量）
            - capacity_delta_prev_year: 去年同期累计用电量
            - capacity_change_rate: 容量净增量变化率 (%)
    """
    # ΔC_t: 过去 capacity_window 个月的累计用电量
    capacity_delta = monthly_consumption.rolling(
        window=capacity_window, min_periods=1
    ).sum()

    # ΔC_{t-lag}: 去年同期过去 capacity_window 个月的累计用电量
    capacity_delta_prev_year = capacity_delta.shift(lag_periods)

    # E_{t-lag}: 去年同期的用电量（归一化分母）
    electricity_prev_year = monthly_consumption.shift(lag_periods)

    # ΔC_rate = (ΔC_t - ΔC_{t-lag}) / E_{t-lag} * 100
    capacity_change_rate = (
        (capacity_delta - capacity_delta_prev_year)
        / electricity_prev_year.abs()
        * 100
    )
    capacity_change_rate = capacity_change_rate.replace([np.inf, -np.inf], np.nan)

    result = pd.DataFrame({
        "capacity_delta": capacity_delta,
        "capacity_delta_prev_year": capacity_delta_prev_year,
        "capacity_change_rate": capacity_change_rate,
    }, index=monthly_consumption.index)

    logger.info(f"容量净增量变化率计算完成: "
                f"均值={capacity_change_rate.mean():.2f}%, "
                f"标准差={capacity_change_rate.std():.2f}%")

    return result


# ============================================================================
# MIDAS 高频→低频聚合（Stundziene et al. 2023）
# ============================================================================

def midas_almon_weights(
    n_lags: int,
    theta_1: float = -0.1,
    theta_2: float = -0.01,
) -> np.ndarray:
    """
    Almon 指数分布式滞后权重。

    公式:
        w(k) = exp(θ₁·k + θ₂·k²)
        w_normalized(k) = w(k) / Σ w(j)

    其中 k = 0, 1, ..., n_lags-1
    当 θ₁ < 0 且 θ₂ < 0 时，近期权重更大，符合"近期数据更相关"的直觉。

    参数:
        n_lags: 滞后阶数
        theta_1: 一阶 Almon 参数
        theta_2: 二阶 Almon 参数

    返回:
        归一化权重数组 (sum=1)
    """
    k = np.arange(n_lags)
    log_weights = theta_1 * k + theta_2 * k ** 2
    # 数值稳定处理
    log_weights = log_weights - log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum()


def midas_beta_weights(
    n_lags: int,
    alpha: float = 1.0,
    beta: float = 5.0,
) -> np.ndarray:
    """
    Beta 分布滞后权重（更灵活，可产生递增/递减/驼峰型权重）。

    公式:
        w(k) = (k/(n_lags-1))^(α-1) * (1 - k/(n_lags-1))^(β-1)
        w_normalized(k) = w(k) / Σ w(j)

    参数:
        n_lags: 滞后阶数
        alpha: Beta 分布 α 参数
        beta: Beta 分布 β 参数

    返回:
        归一化权重数组 (sum=1)
    """
    x = np.linspace(0.001, 0.999, n_lags)  # 避免 0 和 1
    weights = x ** (alpha - 1) * (1 - x) ** (beta - 1)
    return weights / weights.sum()


def midas_exponential_weights(n_lags: int, decay: float = 0.9) -> np.ndarray:
    """
    指数衰减权重。

    公式:
        w(k) = decay^k
        w_normalized(k) = w(k) / Σ w(j)

    参数:
        n_lags: 滞后阶数
        decay: 衰减因子 (0 < decay <= 1)

    返回:
        归一化权重数组 (sum=1)
    """
    k = np.arange(n_lags)
    weights = decay ** k
    return weights / weights.sum()


def midas_aggregate(
    daily_series: pd.Series,
    target_freq: str = "M",
    n_lags: int = MIDAS_N_LAGS,
    weight_type: str = MIDAS_WEIGHT_TYPE,
    **weight_params,
) -> pd.Series:
    """
    MIDAS 高频→低频数据聚合。

    将日度（或更高频率）数据通过参数化权重函数聚合为低频指标。

    参考: Stundziene et al. (2023)

    参数:
        daily_series: 高频时间序列（如日度用电数据）
        target_freq: 目标频率 "M" (月) / "W" (周) / "Q" (季)
        n_lags: 聚合时使用的滞后观测数
        weight_type: 权重类型
            - "almon": 指数 Almon 权重
            - "beta": Beta 分布权重
            - "exponential": 指数衰减权重
            - "equal": 等权重（简单平均）
        **weight_params: 权重函数的额外参数

    返回:
        与目标频率对齐的聚合序列
    """
    if len(daily_series) < n_lags:
        raise ValueError(
            f"数据长度 ({len(daily_series)}) 小于滞后阶数 ({n_lags})"
        )

    # 生成权重
    if weight_type == "almon":
        theta_1 = weight_params.get("theta_1", -0.1)
        theta_2 = weight_params.get("theta_2", -0.01)
        weights = midas_almon_weights(n_lags, theta_1, theta_2)
    elif weight_type == "beta":
        alpha = weight_params.get("alpha", 1.0)
        beta = weight_params.get("beta", 5.0)
        weights = midas_beta_weights(n_lags, alpha, beta)
    elif weight_type == "exponential":
        decay = weight_params.get("decay", 0.9)
        weights = midas_exponential_weights(n_lags, decay)
    elif weight_type == "equal":
        weights = np.ones(n_lags) / n_lags
    else:
        raise ValueError(f"未知权重类型: {weight_type}")

    # 构建低频索引
    freq_map = {"M": "ME", "W": "W", "Q": "QE"}
    pd_freq = freq_map.get(target_freq, target_freq)
    low_freq_index = daily_series.resample(pd_freq).mean().index

    # 对每个低频时间点计算加权平均
    result_values = []
    result_index = []

    series_values = daily_series.values

    for i, target_date in enumerate(low_freq_index):
        # 找到目标日期在原始数据中的位置
        try:
            pos = daily_series.index.get_indexer([target_date], method="pad")[0]
        except (KeyError, IndexError):
            continue

        if pos < n_lags - 1:
            continue  # 数据不足以计算

        # 取前 n_lags 个观测
        window_values = series_values[pos - n_lags + 1 : pos + 1]

        if len(window_values) < n_lags:
            continue

        # 加权平均
        aggregated = np.dot(window_values, weights)
        result_values.append(aggregated)
        result_index.append(target_date)

    result = pd.Series(result_values, index=pd.DatetimeIndex(result_index),
                       name=f"midas_{weight_type}_{n_lags}")

    logger.info(f"MIDAS 聚合: {len(daily_series)} 日观测 → "
                f"{len(result)} {target_freq} 观测 (权重={weight_type})")

    return result


# ============================================================================
# 多维行业用电分解（Stundziene et al. 2023）
# ============================================================================

def decompose_by_sector(
    df: pd.DataFrame,
    consumption_col: str = "consumption",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """
    按行业分解用电量。

    不同行业的用电变化对应不同经济含义:
        - 工业用电: 反映制造业/工业活动强度（实体经济的核心指标）
        - 商业用电: 反映服务业和消费活动
        - 居民用电: 反映生活消费水平、温度和节假日影响最大
        - 农业用电: 反映农业活动
        - 交通运输用电: 反映物流和出行活跃度

    参考: Stundziene et al. (2023) - 多维度经济判断

    参数:
        df: 含行业标签和用电量的 DataFrame
        consumption_col: 用电量列名
        sector_col: 行业分类列名

    返回:
        DataFrame，每列为一个行业的用电量时间序列
    """
    if sector_col not in df.columns:
        logger.warning(f"行业列 '{sector_col}' 不存在，无法分解")
        return df[[consumption_col]]

    sectors = df[sector_col].unique()
    logger.info(f"检测到 {len(sectors)} 个行业: {list(sectors)}")

    result = pd.DataFrame(index=df.index)

    for sector in sectors:
        mask = df[sector_col] == sector
        sector_series = df.loc[mask, consumption_col].copy()

        # 处理可能的重复时间戳
        sector_series = sector_series.groupby(sector_series.index).mean()

        # 重索引以匹配原始时间轴
        sector_series = sector_series.reindex(df.index.unique())

        col_name = f"consumption_{str(sector).lower().replace(' ', '_')}"
        result[col_name] = sector_series

    # 添加总量列
    result["consumption_total"] = df.groupby(df.index)[consumption_col].mean()

    # 计算各行业占比
    for col in result.columns:
        if col.startswith("consumption_") and col != "consumption_total":
            share_col = col.replace("consumption_", "share_")
            result[share_col] = result[col] / result["consumption_total"] * 100

    return result


def compute_electricity_price_index(
    price: pd.Series,
    base_period: Optional[str] = None,
) -> pd.Series:
    """
    计算电力价格指数（以基准期为100）。

    参数:
        price: 电价时间序列
        base_period: 基准期（如 "2020-01"），None 则使用整个序列的均值

    返回:
        价格指数序列
    """
    if base_period:
        base_value = price.loc[base_period]
    else:
        base_value = price.mean()

    return price / base_value * 100


# ============================================================================
# 统一特征矩阵生成
# ============================================================================

def generate_feature_matrix(
    df: pd.DataFrame,
    consumption_col: str = "consumption",
    temperature_col: Optional[str] = "temperature",
    capacity_col: Optional[str] = "capacity",
    sector_col: Optional[str] = None,
    add_rolling: bool = True,
    add_seasonal_labels: bool = True,
) -> pd.DataFrame:
    """
    一站式特征矩阵生成。

    输入清洗后的 DataFrame，输出可用于 VAE 训练和回归预测的完整特征矩阵。

    生成特征列表:
        - consumption: 原始用电量
        - consumption_yoy: 同比增速 (%)
        - consumption_mom: 环比增速 (%)
        - consumption_cumulative_yoy: 累计同比增速 (%)
        - consumption_ma_3: 3期移动平均
        - consumption_ma_6: 6期移动平均
        - consumption_ma_12: 12期移动平均
        - capacity_change_rate: 容量净增量变化率 (%)
        - capacity_delta: 容量累计量
        - temperature: 原始温度（如有）
        - temperature_deviation: 温度偏离基线（如有）
        - season: 季节标签 (1/2/3/4)
        - quarter_month: 季度内月份 (1/2/3)
        - is_heating_season: 取暖季标记
        - is_cooling_season: 制冷季标记

    参数:
        df: 清洗后 DataFrame（需含 DatetimeIndex）
        consumption_col: 用电量列名
        temperature_col: 温度列名（可选）
        capacity_col: 容量列名（可选）
        sector_col: 行业分类列名（可选）
        add_rolling: 是否添加滚动统计特征
        add_seasonal_labels: 是否添加季节标签

    返回:
        完整特征矩阵 DataFrame
    """
    features = pd.DataFrame(index=df.index)

    # ---- 1. 原始用电量 ----
    if consumption_col in df.columns:
        features["consumption"] = df[consumption_col]

        # 同比增速
        features["consumption_yoy"] = compute_yoy_growth(df[consumption_col])

        # 环比增速
        features["consumption_mom"] = compute_mom_growth(df[consumption_col])

        # 累计同比增速
        features["consumption_cumulative_yoy"] = compute_cumulative_growth(
            df[consumption_col]
        )

        # 移动平均
        for w in MOVING_AVG_WINDOWS:
            features[f"consumption_ma_{w}"] = compute_moving_average(
                df[consumption_col], window=w
            )

        # 滚动统计
        if add_rolling:
            roll_stats = compute_rolling_stats(df[consumption_col])
            features = pd.concat([features, roll_stats], axis=1)

        # 电力容量净增量变化率
        capacity_change = compute_capacity_net_increment(df[consumption_col])
        features = pd.concat([features, capacity_change], axis=1)

    # ---- 2. 容量特征（如有独立容量数据） ----
    if capacity_col and capacity_col in df.columns:
        features["capacity_raw"] = df[capacity_col]
        features["capacity_yoy"] = compute_yoy_growth(df[capacity_col])
        features["capacity_mom"] = compute_mom_growth(df[capacity_col])

    # ---- 3. 温度特征 ----
    if temperature_col and temperature_col in df.columns:
        features["temperature"] = df[temperature_col]

        # 温度偏离舒适基线
        try:
            from .config import TEMPERATURE_BASELINE, HEATING_THRESHOLD, COOLING_THRESHOLD
        except ImportError:
            from config import TEMPERATURE_BASELINE, HEATING_THRESHOLD, COOLING_THRESHOLD
        features["temperature_deviation"] = (
            df[temperature_col] - TEMPERATURE_BASELINE
        )

        # HDD / CDD (度日)
        features["hdd"] = np.maximum(0, HEATING_THRESHOLD - df[temperature_col])
        features["cdd"] = np.maximum(0, df[temperature_col] - COOLING_THRESHOLD)

        features["is_heating_season"] = (
            df[temperature_col] < HEATING_THRESHOLD
        ).astype(int)
        features["is_cooling_season"] = (
            df[temperature_col] > COOLING_THRESHOLD
        ).astype(int)

    # ---- 4. 季节标签 ----
    if add_seasonal_labels:
        features["month"] = df.index.month
        features["quarter"] = df.index.quarter
        features["season"] = df.index.quarter  # 1=Q1(春), 2=Q2(夏), 3=Q3(秋), 4=Q4(冬)
        features["quarter_month"] = features["month"].map(QUARTER_MONTH_MAP)
        features["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        features["day_of_year"] = df.index.dayofyear

    # ---- 5. 行业分解（如有行业列） ----
    if sector_col and sector_col in df.columns:
        sector_features = decompose_by_sector(df, consumption_col, sector_col)
        # 只保留占比列
        share_cols = [c for c in sector_features.columns if c.startswith("share_")]
        for c in share_cols:
            features[c] = sector_features[c]

    # ---- 6. 一阶差分特征（用于 VAE 关注波动） ----
    for col in ["consumption"]:
        if col in features.columns:
            features[f"{col}_diff"] = features[col].diff()

    # 清理
    features = features.replace([np.inf, -np.inf], np.nan)

    # 报告
    n_cols = len(features.columns)
    n_complete = features.dropna().shape[0]
    logger.info(f"特征矩阵生成: {n_cols} 个特征, "
                f"{n_complete}/{len(features)} 行完整 "
                f"({100 * n_complete / max(1, len(features)):.1f}%)")

    return features


def get_feature_matrix_stats(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    返回特征矩阵的统计摘要。

    参数:
        feature_matrix: generate_feature_matrix 的输出

    返回:
        每特征的描述性统计 DataFrame
    """
    stats = feature_matrix.describe().T
    stats["missing_rate"] = feature_matrix.isna().mean().values * 100
    stats["missing_count"] = feature_matrix.isna().sum().values
    return stats
