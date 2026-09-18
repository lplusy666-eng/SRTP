"""
全局配置常量

参考论文:
  - Lan et al. (2025): Composite GDP nowcasting using electricity data (PLOS ONE)
  - Stundziene et al. (2023): Nowcasting Economic Activity Using Electricity Market Data (Economies)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, List

# ============================================================================
# 温度修正配置
# ============================================================================
TEMPERATURE_BASELINE = 18.0  # 舒适温度基线（摄氏度），中国标准

# 温度分段阈值
HEATING_THRESHOLD = 10.0  # 低于此温度进入取暖区
COOLING_THRESHOLD = 25.0  # 高于此温度进入制冷区

# ============================================================================
# 滚动窗口配置
# ============================================================================
ROLLING_WINDOW_QUARTERS = 9   # Lan et al. 2025: 9季度滚动窗口
CAPACITY_WINDOW_MONTHS = 3    # 电力容量净增量：过去3个月

# ============================================================================
# 异常检测配置
# ============================================================================
ANOMALY_SIGMA = 3.0           # 3-sigma 规则
IQR_FACTOR = 1.5              # IQR 因子

# ============================================================================
# 缺失值处理策略
# ============================================================================
MISSING_THRESHOLD_DROP = 0.20  # 缺失率 > 20% → 丢弃列
MISSING_THRESHOLD_INTERP = 0.05  # 缺失率 < 5% → 简单插值

# ============================================================================
# 特征工程配置
# ============================================================================
YOY_PERIODS = 12              # 同比周期（月度数据为12个月）
MOVING_AVG_WINDOWS = [3, 6, 12]  # 移动平均窗口

# ============================================================================
# 中国法定节假日 (公历固定部分，农历部分由 chinese-calendar 库处理)
# ============================================================================
FIXED_HOLIDAYS = {
    "元旦":     [(1, 1)],
    "清明节":   [(4, 4), (4, 5), (4, 6)],
    "劳动节":   [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5)],
    "国庆节":   [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7)],
}

# 农历节日 (由 chinese-calendar 动态计算)
LUNAR_HOLIDAYS = [
    "春节",      # 农历正月初一
    "端午节",    # 农历五月初五
    "中秋节",    # 农历八月十五
]

# 节假日平滑窗口
HOLIDAY_SMOOTH_BEFORE = 3  # 节前N天
HOLIDAY_SMOOTH_AFTER = 3   # 节后N天
SPRING_FESTIVAL_WINDOW = 15  # 春节影响窗口（天）

# ============================================================================
# MIDAS 配置 (Stundziene et al. 2023)
# ============================================================================
MIDAS_N_LAGS = 30           # 日数据滞后天数
MIDAS_WEIGHT_TYPE = "almon"  # 权重类型: "almon", "beta", "exponential", "equal"

# ============================================================================
# 行业分类 (用于多维分解)
# ============================================================================
SECTOR_LABELS = {
    "industry":    "工业用电",
    "commercial":  "商业用电",
    "residential": "居民用电",
    "agriculture": "农业用电",
    "transport":   "交通运输用电",
}

# ============================================================================
# 季度内月份标记
# ============================================================================
QUARTER_MONTH_MAP = {
    1: 1, 2: 2, 3: 3,   # Q1
    4: 1, 5: 2, 6: 3,   # Q2
    7: 1, 8: 2, 9: 3,   # Q3
    10: 1, 11: 2, 12: 3, # Q4
}


@dataclass
class PipelineConfig:
    """流水线可配置参数"""
    # 数据路径
    data_paths: Dict[str, str] = field(default_factory=lambda: {
        "electricity": "data/sample/electricity.csv",
        "weather": "data/sample/weather.csv",
        "capacity": "data/sample/capacity.csv",
        "gdp": "data/sample/gdp.csv",
    })
    # 输出目录
    output_dir: str = "outputs/"
    # 目标频率
    target_freq: Literal["D", "W", "M", "Q"] = "M"
    # 温度修正方法
    temp_correction_method: Literal["linear", "piecewise", "ridge"] = "piecewise"
    # 节假日平滑方法
    holiday_smooth_method: Literal["neighbor_avg", "dummy_reg"] = "neighbor_avg"
    # 滚动窗口季度数
    rolling_window_quarters: int = 9
    # 是否启用 MIDAS
    use_midas: bool = False
    # 训练/测试时间范围
    train_start: Optional[str] = None
    train_end: Optional[str] = None
