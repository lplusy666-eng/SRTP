"""
数据清洗模块

功能:
  - 原始 CSV 数据加载与时间戳解析
  - 缺失值检测与多策略填补（插值/前向填充/季节分解）
  - 离群值检测（IQR / 3-sigma / 滚动z-score）与处理
  - 列名标准化映射
  - 数据质量报告

参考论文:
  - Lan et al. (2025): 数据预处理含春节效应、温度调整
  - Stundziene et al. (2023): 频率对齐、缺失值处理
"""

import logging
import warnings
from typing import Optional, Dict, Union, Literal

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .config import (
        MISSING_THRESHOLD_DROP,
        MISSING_THRESHOLD_INTERP,
        ANOMALY_SIGMA,
        IQR_FACTOR,
    )
except ImportError:
    from config import (
        MISSING_THRESHOLD_DROP,
        MISSING_THRESHOLD_INTERP,
        ANOMALY_SIGMA,
        IQR_FACTOR,
    )

logger = logging.getLogger(__name__)

# ============================================================================
# 标准列名映射
# ============================================================================
STANDARD_COLUMN_MAP = {
    # 时间列变体
    "timestamp": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "日期": "timestamp",
    "时间": "timestamp",
    # 用电量列变体
    "consumption": "consumption",
    "electricity": "consumption",
    "electricity_kwh": "consumption",
    "power": "consumption",
    "用电量": "consumption",
    "电量": "consumption",
    # 温度列变体
    "temperature": "temperature",
    "temp": "temperature",
    "温度": "temperature",
    "气温": "temperature",
    # 容量列变体
    "capacity": "capacity",
    "capacity_kw": "capacity",
    "容量": "capacity",
    # GDP 列变体
    "gdp": "gdp",
    "gdp_growth": "gdp",
    "GDP": "gdp",
    # 行业列变体
    "sector": "sector",
    "industry": "sector",
    "行业": "sector",
    "类别": "sector",
}


def load_raw_data(
    filepath: str,
    column_mapping: Optional[Dict[str, str]] = None,
    date_column: Optional[str] = None,
    date_format: Optional[str] = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    加载原始 CSV 数据文件。

    参数:
        filepath: CSV 文件路径
        column_mapping: 自定义列名映射 {"原始列名": "标准列名"}
        date_column: 时间列名（自动检测如果为None）
        date_format: 日期格式 (e.g. "%Y-%m-%d")
        delimiter: CSV 分隔符
        encoding: 文件编码

    返回:
        以 DatetimeIndex 为索引的 DataFrame，列名已标准化
    """
    # 读取 CSV
    try:
        df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, delimiter=delimiter, encoding="gbk")

    if df.empty:
        raise ValueError(f"文件为空或无法读取: {filepath}")

    # 列名标准化
    df = standardize_columns(df, column_mapping)

    # 时间列解析
    if date_column is None:
        # 自动检测时间列
        for col in df.columns:
            if col == "timestamp":
                date_column = col
                break
        else:
            # 查找名称中包含时间关键字的列
            time_keywords = ["timestamp", "time", "date", "日期", "时间"]
            for col in df.columns:
                if any(kw in col.lower() for kw in time_keywords):
                    date_column = col
                    break

    if date_column is None:
        raise ValueError(
            f"无法自动检测时间列，请手动指定 date_column。"
            f"现有列: {list(df.columns)}"
        )

    if date_column not in df.columns:
        raise ValueError(f"时间列 '{date_column}' 不存在。现有列: {list(df.columns)}")

    # 解析时间并设为索引
    if date_format:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])

    df = df.set_index(date_column)
    df.index.name = "timestamp"

    # 按时间排序
    df = df.sort_index()

    # 检测并移除重复时间戳
    duplicates = df.index.duplicated()
    if duplicates.any():
        n_dups = duplicates.sum()
        logger.warning(f"发现 {n_dups} 个重复时间戳，保留第一个值")
        df = df[~df.index.duplicated(keep="first")]

    logger.info(f"成功加载 {filepath}: {len(df)} 行, {len(df.columns)} 列")
    logger.info(f"时间范围: {df.index.min()} ~ {df.index.max()}")

    return df


def standardize_columns(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    列名标准化映射。

    参数:
        df: 输入 DataFrame
        column_mapping: 自定义映射（优先于标准映射）

    返回:
        列名标准化后的 DataFrame
    """
    # 合并标准映射和自定义映射
    mapping = STANDARD_COLUMN_MAP.copy()
    if column_mapping:
        mapping.update(column_mapping)

    # 应用映射
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col in mapping:
            rename_dict[col] = mapping[col]
        elif col_lower in {k.lower(): v for k, v in mapping.items()}:
            # 大小写不敏感匹配
            for k, v in mapping.items():
                if k.lower() == col_lower:
                    rename_dict[col] = v
                    break

    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"列名标准化: {rename_dict}")

    return df


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成缺失值统计报告。

    参数:
        df: 输入 DataFrame

    返回:
        DataFrame 包含每列的 [缺失数, 缺失率, 数据类型]
    """
    n_total = len(df)
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_rate": (df.isna().sum() / n_total * 100).values,
        "dtype": df.dtypes.values.astype(str),
    })
    report = report.sort_values("missing_rate", ascending=False).reset_index(drop=True)

    # 数据质量评分
    complete_cols = (report["missing_rate"] == 0).sum()
    high_missing_cols = (report["missing_rate"] > MISSING_THRESHOLD_DROP * 100).sum()
    logger.info(
        f"缺失值报告: {len(df.columns)} 列, "
        f"{complete_cols} 列完整, "
        f"{high_missing_cols} 列缺失率 > {MISSING_THRESHOLD_DROP*100}%"
    )

    return report


def fill_missing(
    df: pd.DataFrame,
    strategy: Optional[Dict[str, str]] = None,
    groupby_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    多策略缺失值填补。

    支持的策略:
        - "forward_fill": 向前填充（时间序列首选），然后用向后填充处理边缘
        - "backward_fill": 向后填充
        - "linear_interp": 线性插值
        - "time_interp": 基于时间索引的插值
        - "seasonal_interp": 季节分解后插值残差（适合强季节性数据）
        - "mean": 用列均值填充
        - "median": 用列中位数填充
        - "zero": 用0填充（适合容量新增等指标）
        - "drop": 丢弃该列

    参数:
        df: 输入 DataFrame
        strategy: 列级策略字典 {"column_name": "forward_fill"}
        groupby_col: 分组填补的列名（如按行业分组填补）

    返回:
        缺失值填补后的 DataFrame
    """
    df = df.copy()
    missing_report = check_missing(df)

    if missing_report["missing_count"].sum() == 0:
        logger.info("数据完整，无需填补")
        return df

    # 自动生成策略
    user_provided_strategy = strategy is not None
    if strategy is None:
        strategy = _auto_strategy(df, missing_report)

    for col, method in list(strategy.items()):
        if col not in df.columns:
            continue

        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        missing_rate = n_missing / len(df)

        # 缺失率过高 → 丢弃列（仅在自动生成策略时）
        if not user_provided_strategy and missing_rate > MISSING_THRESHOLD_DROP:
            logger.warning(f"列 '{col}' 缺失率 {missing_rate:.1%} > "
                           f"{MISSING_THRESHOLD_DROP:.0%}，丢弃该列")
            df = df.drop(columns=[col])
            continue

        try:
            if groupby_col and groupby_col in df.columns:
                df[col] = df.groupby(groupby_col)[col].transform(
                    lambda s: _apply_fill_method(s, method)
                )
            else:
                df[col] = _apply_fill_method(df[col], method)

            remaining = df[col].isna().sum()
            logger.info(f"列 '{col}': {method} → 填补 {n_missing} 个缺失值"
                        f" ({remaining} 个剩余)")
        except Exception as e:
            logger.error(f"列 '{col}' 填补失败 ({method}): {e}")

    return df


def _auto_strategy(df: pd.DataFrame, missing_report: pd.DataFrame) -> Dict[str, str]:
    """根据列的数据类型和缺失模式自动选择填补策略。"""
    strategy = {}
    for _, row in missing_report.iterrows():
        col = row["column"]
        rate = row["missing_rate"] / 100.0
        dtype = row["dtype"]

        if rate == 0:
            continue

        # 数值列
        if "float" in dtype or "int" in dtype:
            if rate <= MISSING_THRESHOLD_INTERP:
                strategy[col] = "linear_interp"
            else:
                strategy[col] = "time_interp"
        # 非数值列
        else:
            strategy[col] = "forward_fill"

    return strategy


def _apply_fill_method(series: pd.Series, method: str) -> pd.Series:
    """对单个 Series 应用填补方法。"""
    if method == "forward_fill":
        series = series.ffill().bfill()
    elif method == "backward_fill":
        series = series.bfill().ffill()
    elif method == "linear_interp":
        series = series.interpolate(method="linear", limit_direction="both")
    elif method == "time_interp":
        series = series.interpolate(method="time", limit_direction="both")
    elif method == "seasonal_interp":
        series = _interpolate_seasonal(series)
    elif method == "mean":
        series = series.fillna(series.mean())
    elif method == "median":
        series = series.fillna(series.median())
    elif method == "zero":
        series = series.fillna(0)
    elif method == "drop":
        return series  # 由调用者处理列删除
    else:
        logger.warning(f"未知填补方法 '{method}'，使用 forward_fill")
        series = series.ffill().bfill()
    return series


def _interpolate_seasonal(series: pd.Series, period: Optional[int] = None) -> pd.Series:
    """
    季节分解后插值残差。

    方法:
        1. 用滑动平均估算趋势成分
        2. 去趋势后的残差做线性插值
        3. 重组趋势 + 插值后残差
    """
    if series.isna().all():
        return series

    # 自动检测周期
    if period is None:
        if len(series) >= 24:
            period = 12  # 月度数据默认12个月周期
        elif len(series) >= 14:
            period = 7   # 日度数据默认7天周期
        else:
            period = max(2, len(series) // 3)

    # 趋势成分（滑动平均）
    trend = series.rolling(window=period, center=True, min_periods=1).mean()

    # 残差 = 原始 - 趋势
    residual = series - trend

    # 对残差插值
    residual = residual.interpolate(method="linear", limit_direction="both")

    # 重组
    filled = trend + residual

    # 边界：用前向/后向填充处理滑动平均无法覆盖的边界
    filled = filled.ffill().bfill()

    return filled


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    factor: float = IQR_FACTOR,
) -> pd.Series:
    """
    IQR 法离群值检测。

    参数:
        df: 输入 DataFrame
        column: 目标列名
        factor: IQR 乘数 (默认 1.5)

    返回:
        Boolean Series，True 表示离群值
    """
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    mask = (df[column] < lower) | (df[column] > upper)
    n_outliers = mask.sum()

    if n_outliers > 0:
        logger.info(f"IQR 检测 '{column}': {n_outliers} 个离群值 "
                    f"({n_outliers / len(df) * 100:.2f}%), "
                    f"范围 [{lower:.2f}, {upper:.2f}]")

    return mask


def detect_outliers_sigma(
    df: pd.DataFrame,
    column: str,
    sigma: float = ANOMALY_SIGMA,
) -> pd.Series:
    """
    3-sigma 法离群值检测。

    参数:
        df: 输入 DataFrame
        column: 目标列名
        sigma: sigma 倍数 (默认 3)

    返回:
        Boolean Series，True 表示离群值
    """
    series = df[column].dropna()
    mean = series.mean()
    std = series.std()
    lower = mean - sigma * std
    upper = mean + sigma * std

    mask = (df[column] < lower) | (df[column] > upper)
    n_outliers = mask.sum()

    if n_outliers > 0:
        logger.info(f"3-sigma 检测 '{column}': {n_outliers} 个离群值 "
                    f"({n_outliers / len(df) * 100:.2f}%), "
                    f"范围 [{lower:.2f}, {upper:.2f}]")

    return mask


def detect_outliers_rolling_zscore(
    df: pd.DataFrame,
    column: str,
    window: int = 30,
    threshold: float = 3.0,
) -> pd.Series:
    """
    滚动 z-score 离群值检测（适合时间序列，捕捉局部异常）。

    参数:
        df: 输入 DataFrame
        column: 目标列名
        window: 滚动窗口大小
        threshold: z-score 阈值

    返回:
        Boolean Series，True 表示离群值
    """
    series = df[column]
    rolling_mean = series.rolling(window=window, center=True, min_periods=3).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=3).std()

    # 避免除以接近零的值
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (series - rolling_mean) / rolling_std
    mask = zscore.abs() > threshold

    n_outliers = mask.sum()
    if n_outliers > 0:
        logger.info(f"滚动 z-score 检测 '{column}' (window={window}): "
                    f"{n_outliers} 个离群值")

    return mask


def remove_or_flag_outliers(
    df: pd.DataFrame,
    mask: pd.Series,
    strategy: Literal["flag", "interpolate", "cap", "remove"] = "flag",
    column: Optional[str] = None,
) -> pd.DataFrame:
    """
    离群值处理。

    参数:
        df: 输入 DataFrame
        mask: 离群值布尔掩码
        strategy: 处理策略
            - "flag": 保留数据，添加 is_outlier 标记列
            - "interpolate": 将离群值替换为 NaN 后线性插值
            - "cap": 用分位数边界替换（缩尾处理）
            - "remove": 删除含离群值的行
        column: 目标列（flag 策略必需）

    返回:
        处理后的 DataFrame
    """
    df = df.copy()

    if not mask.any():
        return df

    n_outliers = mask.sum()

    if strategy == "flag":
        if column is None:
            raise ValueError("flag 策略需要指定 column 参数")
        flag_col = f"is_outlier_{column}"
        if flag_col in df.columns:
            df[flag_col] = df[flag_col] | mask
        else:
            df[flag_col] = mask
        logger.info(f"标记 {n_outliers} 个离群值到列 '{flag_col}'")

    elif strategy == "interpolate":
        if column is None:
            raise ValueError("interpolate 策略需要指定 column 参数")
        original = df[column].copy()
        df.loc[mask, column] = np.nan
        df[column] = df[column].interpolate(method="linear", limit_direction="both")
        logger.info(f"插值替换 {n_outliers} 个离群值，最大变化: "
                    f"{(df[column] - original).abs().max():.2f}")

    elif strategy == "cap":
        if column is None:
            raise ValueError("cap 策略需要指定 column 参数")
        series = df[column].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
        df[column] = df[column].clip(lower, upper)
        logger.info(f"缩尾处理 '{column}': 限制到 [{lower:.2f}, {upper:.2f}]")

    elif strategy == "remove":
        df = df[~mask]
        logger.info(f"删除 {n_outliers} 行含离群值的数据")

    return df


def prepare_cleaned_data(
    filepath: str,
    column_mapping: Optional[Dict[str, str]] = None,
    date_column: Optional[str] = None,
    fill_strategy: Optional[Dict[str, str]] = None,
    outlier_method: str = "iqr",
    outlier_column: str = "consumption",
    outlier_strategy: str = "flag",
) -> pd.DataFrame:
    """
    一站式数据清洗入口。

    执行流程:
        加载 → 列名标准化 → 缺失值填补 → 离群值检测 → 离群值标记

    参数:
        filepath: CSV 文件路径
        column_mapping: 列名映射
        date_column: 时间列名
        fill_strategy: 缺失值填补策略
        outlier_method: 离群值检测方法 ("iqr", "sigma", "rolling_zscore")
        outlier_column: 需检测离群值的列
        outlier_strategy: 离群值处理策略

    返回:
        清洗后的 DataFrame
    """
    # 1. 加载
    df = load_raw_data(filepath, column_mapping=column_mapping,
                       date_column=date_column)

    # 2. 缺失值检测报告
    report = check_missing(df)
    logger.info(f"\n{report.to_string()}")

    # 3. 缺失值填补
    df = fill_missing(df, strategy=fill_strategy)

    # 4. 离群值检测
    if outlier_column in df.columns:
        if outlier_method == "iqr":
            mask = detect_outliers_iqr(df, outlier_column)
        elif outlier_method == "sigma":
            mask = detect_outliers_sigma(df, outlier_column)
        elif outlier_method == "rolling_zscore":
            mask = detect_outliers_rolling_zscore(df, outlier_column)
        else:
            logger.warning(f"未知离群值检测方法 '{outlier_method}'，跳过")
            mask = pd.Series(False, index=df.index)

        # 5. 离群值处理
        if mask.any():
            df = remove_or_flag_outliers(df, mask, strategy=outlier_strategy,
                                         column=outlier_column)

    logger.info(f"数据清洗完成: {len(df)} 行, {len(df.columns)} 列")
    return df
