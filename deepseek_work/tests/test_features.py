"""
特征工程模块单元测试
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import (
    compute_yoy_growth,
    compute_mom_growth,
    compute_cumulative_growth,
    compute_moving_average,
    compute_rolling_stats,
    compute_capacity_net_increment,
    midas_almon_weights,
    midas_beta_weights,
    midas_exponential_weights,
    midas_aggregate,
    decompose_by_sector,
    generate_feature_matrix,
    get_feature_matrix_stats,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def monthly_series():
    """创建模拟的月度用电量时间序列。"""
    dates = pd.date_range("2019-01-01", periods=48, freq="ME")
    # 基础增长 + 季节性 + 噪声
    trend = np.arange(48) * 2
    seasonal = 20 * np.sin(np.arange(48) * 2 * np.pi / 12)
    noise = np.random.RandomState(42).normal(0, 5, 48)
    return pd.Series(100 + trend + seasonal + noise, index=dates, name="consumption")


@pytest.fixture
def daily_series():
    """创建模拟的日度用电量时间序列。"""
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    trend = np.arange(365) * 0.05
    seasonal = 30 * np.sin(np.arange(365) * 2 * np.pi / 365)
    weekly = 10 * np.sin(np.arange(365) * 2 * np.pi / 7)
    noise = np.random.RandomState(42).normal(0, 3, 365)
    return pd.Series(100 + trend + seasonal + weekly + noise,
                     index=dates, name="consumption")


@pytest.fixture
def multi_sector_df():
    """创建多行业用电 DataFrame。"""
    dates = pd.date_range("2020-01-01", periods=60, freq="ME")
    sectors = ["工业", "商业", "居民"]
    data = []
    for sector in sectors:
        for d in dates:
            data.append({
                "timestamp": d,
                "sector": sector,
                "consumption": np.random.RandomState(hash(sector) % 10000).lognormal(4, 0.5),
            })
    return pd.DataFrame(data).set_index("timestamp")


# ============================================================================
# 测试: 同比增速
# ============================================================================

class TestYoYGrowth:
    def test_basic_yoy(self):
        """手工计算验证 YoY 公式。"""
        idx = pd.date_range("2020-01-01", periods=24, freq="ME")
        series = pd.Series(np.arange(100, 124, dtype=float), index=idx)
        yoy = compute_yoy_growth(series, periods=12)

        # 第13个月 (index=12): (112-100)/100 = 12%
        assert yoy.iloc[12] == pytest.approx(12.0, abs=0.01)
        # 第24个月 (index=23): (123-111)/111 ≈ 10.81%
        assert yoy.iloc[23] == pytest.approx(10.81, abs=0.1)

    def test_yoy_first_year_nan(self, monthly_series):
        yoy = compute_yoy_growth(monthly_series)
        # 前12个月应为 NaN（没有去年同期）
        assert yoy.iloc[:12].isna().all()
        # 之后应有值
        assert yoy.iloc[12:].notna().any()

    def test_yoy_zero_division(self):
        idx = pd.date_range("2020-01-01", periods=24, freq="ME")
        series = pd.Series([0.0] * 12 + [10.0] * 12, index=idx)
        yoy = compute_yoy_growth(series)
        # 分母为0 → NaN
        assert pd.isna(yoy.iloc[12])

    def test_yoy_with_seasonal_pattern(self, monthly_series):
        yoy = compute_yoy_growth(monthly_series)
        # 季节性数据下，YoY 应部分消除季节性
        yoy_std = yoy.dropna().std()
        raw_std = monthly_series.std()
        # YoY 的波动应小于原始数据（因为消除了部分季节性）
        assert yoy_std < raw_std


# ============================================================================
# 测试: 环比增速
# ============================================================================

class TestMoMGrowth:
    def test_basic_mom(self):
        idx = pd.date_range("2020-01-01", periods=6, freq="ME")
        series = pd.Series([100.0, 110.0, 121.0, 108.9, 120.0, 132.0], index=idx)
        mom = compute_mom_growth(series)

        # 第2个月: (110-100)/100 = 10%
        assert mom.iloc[1] == pytest.approx(10.0, abs=0.01)
        # 第3个月: (121-110)/110 = 10%
        assert mom.iloc[2] == pytest.approx(10.0, abs=0.01)
        # 第4个月: (108.9-121)/121 ≈ -10%
        assert mom.iloc[3] == pytest.approx(-10.0, abs=0.1)

    def test_mom_first_nan(self):
        series = pd.Series([1.0, 2.0, 3.0])
        mom = compute_mom_growth(series)
        assert pd.isna(mom.iloc[0])
        assert mom.iloc[1] == pytest.approx(100.0, abs=0.01)


# ============================================================================
# 测试: 累计增速
# ============================================================================

class TestCumulativeGrowth:
    def test_cumulative_growth(self):
        idx = pd.date_range("2020-01-01", periods=36, freq="ME")
        series = pd.Series(np.arange(1, 37, dtype=float), index=idx)
        cum = compute_cumulative_growth(series)

        # cumsum at month 24 (2021-12, index 23): sum of 13-24 = 222
        # cumsum_prev_year: sum of 1-12 = 78
        # cumulative_growth = (222/78 - 1) * 100 ≈ 184.6%
        expected = (222.0 / 78.0 - 1) * 100
        assert cum.iloc[23] == pytest.approx(expected, abs=0.1)
        # First 12 months should be NaN
        assert cum.iloc[:12].isna().all()


# ============================================================================
# 测试: 移动平均
# ============================================================================

class TestMovingAverage:
    def test_simple_ma(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = compute_moving_average(series, window=3, center=True)
        # 中心化 MA: index=1 → (1+2+3)/3 = 2
        assert ma.iloc[1] == pytest.approx(2.0, abs=0.01)
        assert ma.iloc[2] == pytest.approx(3.0, abs=0.01)


# ============================================================================
# 测试: 滚动统计
# ============================================================================

class TestRollingStats:
    def test_rolling_stats_output(self, monthly_series):
        stats = compute_rolling_stats(monthly_series, windows=[3, 6], stats=["mean", "std"])
        assert "roll_3_mean" in stats.columns
        assert "roll_3_std" in stats.columns
        assert "roll_6_mean" in stats.columns
        assert "roll_6_std" in stats.columns


# ============================================================================
# 测试: 电力容量净增量变化率（核心）
# ============================================================================

class TestCapacityNetIncrement:
    def test_capacity_change_rate(self, monthly_series):
        result = compute_capacity_net_increment(monthly_series, capacity_window=3, lag_periods=12)

        assert "capacity_delta" in result.columns
        assert "capacity_delta_prev_year" in result.columns
        assert "capacity_change_rate" in result.columns
        assert len(result) == len(monthly_series)

        # 前12+3个月内应有 NaN（数据不足以计算）
        assert result["capacity_change_rate"].iloc[14:].notna().any()

    def test_capacity_delta_rolling_sum(self):
        """验证 ΔC_t 是过去3个月的累计和。"""
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        series = pd.Series(np.arange(1, 25, dtype=float), index=dates)
        result = compute_capacity_net_increment(series, capacity_window=3, lag_periods=12)

        # 第3个月 (2): capacity_delta = 1+2+3 = 6
        assert result["capacity_delta"].iloc[2] == pytest.approx(6.0, abs=0.01)
        # 第4个月 (3): capacity_delta = 2+3+4 = 9
        assert result["capacity_delta"].iloc[3] == pytest.approx(9.0, abs=0.01)

    def test_capacity_change_rate_interpretation(self):
        """容量变化率为正 → 新增容量在增长 → 未来经济活动可能增强。"""
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        # 用电量快速增长的序列
        series = pd.Series(
            np.concatenate([
                np.linspace(100, 120, 12),   # 第一年缓慢增长
                np.linspace(120, 180, 12),    # 第二年快速增长
                np.linspace(180, 200, 12),    # 第三年继续增长
            ]),
            index=dates,
        )
        result = compute_capacity_net_increment(series)

        # 在快速增长期（第二年开始），容量变化率应为正
        mid_period = result.iloc[13:18]["capacity_change_rate"]
        assert mid_period.mean() > 0, f"快速增长期应有正的容量变化率，实际均值={mid_period.mean():.2f}"


# ============================================================================
# 测试: MIDAS 权重
# ============================================================================

class TestMidasWeights:
    def test_almon_weights_sum_to_one(self):
        weights = midas_almon_weights(30, theta_1=-0.1, theta_2=-0.01)
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)
        assert len(weights) == 30
        # 权重应递减（近期高，远期低）
        assert weights[0] > weights[-1]

    def test_beta_weights_sum_to_one(self):
        weights = midas_beta_weights(30, alpha=1.0, beta=5.0)
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_exponential_weights_sum_to_one(self):
        weights = midas_exponential_weights(30, decay=0.9)
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)
        assert weights[0] > weights[-1]

    def test_equal_almon_default_comparison(self):
        """Almon 权重应对近期观测分配更高权重。"""
        almon = midas_almon_weights(30)
        equal = np.ones(30) / 30
        # Almon 的第一个权重应大于等权重的平均权重
        assert almon[0] > 1 / 30


# ============================================================================
# 测试: MIDAS 聚合
# ============================================================================

class TestMidasAggregate:
    def test_basic_aggregation(self, daily_series):
        result = midas_aggregate(daily_series, target_freq="M", n_lags=30,
                                 weight_type="equal")
        assert len(result) > 0
        # 月度聚合应有约12个月
        assert 10 <= len(result) <= 14

    def test_different_weight_types(self, daily_series):
        eq = midas_aggregate(daily_series, n_lags=30, weight_type="equal")
        almon = midas_aggregate(daily_series, n_lags=30, weight_type="almon")
        # 不同权重应有不同的聚合结果
        assert not np.allclose(eq.values, almon.values, rtol=0.01)

    def test_insufficient_data_raises(self, daily_series):
        short = daily_series.iloc[:10]
        with pytest.raises(ValueError):
            midas_aggregate(short, n_lags=30)


# ============================================================================
# 测试: 行业分解
# ============================================================================

class TestDecomposeBySector:
    def test_sector_decomposition(self, multi_sector_df):
        result = decompose_by_sector(multi_sector_df)
        # 应生成各行业列
        assert any(c.startswith("consumption_") for c in result.columns)
        assert any(c.startswith("share_") for c in result.columns)
        # 总量应约等于各行业之和
        assert "consumption_total" in result.columns


# ============================================================================
# 测试: 特征矩阵生成
# ============================================================================

class TestFeatureMatrix:
    def test_basic_generation(self, monthly_series):
        """测试基础特征矩阵生成。"""
        df = pd.DataFrame({
            "consumption": monthly_series.values,
            "temperature": 15 + 10 * np.sin(np.arange(len(monthly_series)) * np.pi / 6),
        }, index=monthly_series.index)

        features = generate_feature_matrix(df)

        # 核心特征应存在
        expected_cols = ["consumption", "consumption_yoy", "consumption_mom",
                         "consumption_ma_3", "capacity_change_rate"]
        for col in expected_cols:
            assert col in features.columns, f"缺少列: {col}"

        # 温度相关特征
        assert "temperature" in features.columns
        assert "hdd" in features.columns
        assert "cdd" in features.columns

        # 季节标签
        assert "season" in features.columns
        assert "quarter_month" in features.columns

    def test_generation_without_temperature(self, monthly_series):
        df = pd.DataFrame({"consumption": monthly_series.values},
                          index=monthly_series.index)
        features = generate_feature_matrix(df)
        # 温度列为空但不应报错
        assert "consumption" in features.columns

    def test_feature_stats(self, monthly_series):
        df = pd.DataFrame({
            "consumption": monthly_series.values,
            "temperature": 15 + 10 * np.sin(np.arange(len(monthly_series)) * np.pi / 6),
        }, index=monthly_series.index)
        features = generate_feature_matrix(df)
        stats = get_feature_matrix_stats(features)
        assert "missing_rate" in stats.columns
        assert "mean" in stats.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
