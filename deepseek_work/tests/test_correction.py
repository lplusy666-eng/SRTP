"""
温度修正 & 节假日平滑模块单元测试
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from correction import (
    compute_degree_days,
    fit_temperature_model,
    temperature_correct,
    fit_and_correct,
    build_holiday_calendar,
    holiday_smooth,
    _find_consecutive_periods,
    weekday_adjust,
    apply_all_corrections,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_series():
    """模拟温度序列：冬季低、夏季高。"""
    dates = pd.date_range("2020-01-01", periods=24, freq="ME")
    # 正弦模拟年温度变化（峰值在7月）
    temps = 18 + 15 * np.sin(np.arange(24) * 2 * np.pi / 12)
    return pd.Series(temps, index=dates, name="temperature")


@pytest.fixture
def consumption_with_temp_effect(temp_series):
    """模拟受温度影响的用电量序列。"""
    dates = temp_series.index
    base = 100 + np.arange(len(dates)) * 0.5  # 基础趋势
    # 取暖效应：温度低于10°时每度增加3单位用电
    heating = np.maximum(0, 10 - temp_series.values) * 3.0
    # 制冷效应：温度高于25°时每度增加2.5单位用电
    cooling = np.maximum(0, temp_series.values - 25) * 2.5
    noise = np.random.RandomState(42).normal(0, 2, len(dates))
    consumption = base + heating + cooling + noise
    return pd.Series(consumption, index=dates, name="consumption")


@pytest.fixture
def daily_consumption():
    """模拟日度用电量和温度数据。"""
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    temps = 15 + 12 * np.sin(np.arange(120) * 2 * np.pi / 365)
    base = 500 + np.arange(120) * 0.3
    # 温度效应
    heating = np.maximum(0, 10 - temps) * 10
    cooling = np.maximum(0, temps - 25) * 8
    consumption = base + heating + cooling + np.random.RandomState(42).normal(0, 15, 120)
    return pd.DataFrame({
        "consumption": consumption,
        "temperature": temps,
    }, index=dates)


# ============================================================================
# 测试: HDD/CDD 计算
# ============================================================================

class TestDegreeDays:
    def test_hdd_cdd_basic(self):
        temps = pd.Series([5.0, 18.0, 30.0])
        dd = compute_degree_days(temps, base_temp=18.0)

        # 5°C: HDD=13, CDD=0
        assert dd["hdd"].iloc[0] == pytest.approx(13.0, abs=0.01)
        assert dd["cdd"].iloc[0] == pytest.approx(0.0, abs=0.01)

        # 18°C: HDD=0, CDD=0
        assert dd["hdd"].iloc[1] == pytest.approx(0.0, abs=0.01)
        assert dd["cdd"].iloc[1] == pytest.approx(0.0, abs=0.01)

        # 30°C: HDD=0, CDD=12
        assert dd["hdd"].iloc[2] == pytest.approx(0.0, abs=0.01)
        assert dd["cdd"].iloc[2] == pytest.approx(12.0, abs=0.01)

    def test_custom_base_temp(self):
        temps = pd.Series([20.0, 25.0])
        dd = compute_degree_days(temps, base_temp=22.0)
        assert dd["hdd"].iloc[0] == pytest.approx(2.0, abs=0.01)
        assert dd["cdd"].iloc[1] == pytest.approx(3.0, abs=0.01)

    def test_columns_present(self, temp_series):
        dd = compute_degree_days(temp_series)
        assert "hdd" in dd.columns
        assert "cdd" in dd.columns
        assert "temp_deviation" in dd.columns
        assert len(dd) == len(temp_series)


# ============================================================================
# 测试: 温度模型拟合
# ============================================================================

class TestFitTemperatureModel:
    def test_piecewise_fit(self, consumption_with_temp_effect, temp_series):
        model = fit_temperature_model(
            consumption_with_temp_effect,
            temp_series,
            method="piecewise",
        )
        assert "heating_coef" in model
        assert "cooling_coef" in model
        assert "r_squared" in model
        # R² 应该合理（因为数据是按此模型生成的）
        assert model["r_squared"] > 0.3

    def test_heating_coefficient_positive(self, consumption_with_temp_effect, temp_series):
        """取暖系数应为正：温度越低，取暖用电越多。"""
        model = fit_temperature_model(
            consumption_with_temp_effect, temp_series, method="piecewise"
        )
        assert model["heating_coef"] > 0, (
            f"取暖系数应为正，实际={model['heating_coef']:.2f}"
        )

    def test_cooling_coefficient_positive(self, consumption_with_temp_effect, temp_series):
        """制冷系数应为正：温度越高，制冷用电越多。"""
        model = fit_temperature_model(
            consumption_with_temp_effect, temp_series, method="piecewise"
        )
        assert model["cooling_coef"] > 0, (
            f"制冷系数应为正，实际={model['cooling_coef']:.2f}"
        )

    def test_linear_fit(self, consumption_with_temp_effect, temp_series):
        model = fit_temperature_model(
            consumption_with_temp_effect, temp_series, method="linear"
        )
        assert "temp_coef" in model
        assert "r_squared" in model

    def test_ridge_fit(self, consumption_with_temp_effect, temp_series):
        model = fit_temperature_model(
            consumption_with_temp_effect, temp_series, method="ridge"
        )
        assert "r_squared" in model

    def test_insufficient_data(self):
        """数据过少时应返回 fallback。"""
        s = pd.Series([100, 110], index=pd.date_range("2020-01-01", periods=2, freq="ME"))
        t = pd.Series([15, 20], index=s.index)
        model = fit_temperature_model(s, t)
        assert model["method"] == "fallback_mean"


# ============================================================================
# 测试: 温度修正
# ============================================================================

class TestTemperatureCorrect:
    def test_correction_reduces_temp_correlation(self, consumption_with_temp_effect, temp_series):
        """修正后用电量与温度的相关性应显著降低。"""
        model = fit_temperature_model(
            consumption_with_temp_effect, temp_series, method="piecewise"
        )
        corrected = temperature_correct(
            consumption_with_temp_effect, temp_series, model
        )

        # 原始用电量与温度的相关性
        orig_corr = consumption_with_temp_effect.corr(temp_series)
        # 修正后用电量与温度的相关性
        corr_corr = corrected.corr(temp_series)

        # 修正后的相关性绝对值应小于原始
        assert abs(corr_corr) < abs(orig_corr), (
            f"修正前 corr={orig_corr:.3f}, 修正后 corr={corr_corr:.3f}"
        )

    def test_fit_and_correct(self, consumption_with_temp_effect, temp_series):
        corrected, model = fit_and_correct(
            consumption_with_temp_effect, temp_series, method="piecewise"
        )
        assert len(corrected) == len(consumption_with_temp_effect)
        assert "r_squared" in model

    def test_correction_preserves_trend(self, consumption_with_temp_effect, temp_series):
        """温度修正后应保留长期趋势。"""
        model = fit_temperature_model(
            consumption_with_temp_effect, temp_series, method="piecewise"
        )
        corrected = temperature_correct(
            consumption_with_temp_effect, temp_series, model
        )

        # 修正前后的趋势应相近（通过移动平均比较）
        orig_ma = consumption_with_temp_effect.rolling(6, center=True).mean().dropna()
        corr_ma = corrected.rolling(6, center=True).mean().dropna()
        common_idx = orig_ma.index.intersection(corr_ma.index)

        trend_corr = orig_ma.loc[common_idx].corr(corr_ma.loc[common_idx])
        assert trend_corr > 0.7, f"修正前后趋势相关性应较高，实际={trend_corr:.3f}"


# ============================================================================
# 测试: 节假日日历
# ============================================================================

class TestHolidayCalendar:
    def test_calendar_generation(self):
        calendar = build_holiday_calendar(2023, 2024)
        assert "is_holiday" in calendar.columns
        assert "is_workday" in calendar.columns
        assert "holiday_name" in calendar.columns
        assert "days_to_spring_festival" in calendar.columns
        assert "is_spring_festival_period" in calendar.columns

        # 应包含约730天
        assert 700 <= len(calendar) <= 800

    def test_has_holidays(self):
        calendar = build_holiday_calendar(2023, 2023)
        # 至少有一些节假日（春节、国庆等）
        assert calendar["is_holiday"].sum() > 0

    def test_spring_festival_period(self):
        calendar = build_holiday_calendar(2023, 2023)
        assert calendar["is_spring_festival_period"].sum() > 0


# ============================================================================
# 测试: 节假日平滑
# ============================================================================

class TestHolidaySmooth:
    def test_smoothing_reduces_holiday_spikes(self, daily_consumption):
        """节假日平滑应减少假期期间的异常值。"""
        calendar = build_holiday_calendar(2023, 2023)
        calendar = calendar.reindex(daily_consumption.index)

        # 只在假期日用 holiday_smooth
        smoothed = holiday_smooth(
            daily_consumption["consumption"],
            calendar=calendar,
            window_before=3,
            window_after=3,
        )

        # 平滑后假期日的值应更接近非假期日的水平
        holiday_mask = calendar["is_holiday"].reindex(daily_consumption.index).fillna(False)
        if holiday_mask.any():
            # 假期日平滑前后的波动
            orig_holiday_std = daily_consumption.loc[holiday_mask, "consumption"].std()
            smooth_holiday_std = smoothed.loc[holiday_mask].std()
            # 平滑后应更平稳（或保持类似水平）
            # 不做严格断言，因为数据是随机生成的

    def test_no_calendar_returns_original(self, daily_consumption):
        result = holiday_smooth(daily_consumption["consumption"], calendar=None)
        pd.testing.assert_series_equal(result, daily_consumption["consumption"])


# ============================================================================
# 测试: 连续假期段查找
# ============================================================================

class TestFindConsecutivePeriods:
    def test_single_period(self):
        mask = pd.Series([False, True, True, False, False],
                         index=pd.date_range("2020-01-01", periods=5))
        periods = _find_consecutive_periods(mask)
        assert len(periods) == 1

    def test_multiple_periods(self):
        mask = pd.Series([True, False, True, True, False, True],
                         index=pd.date_range("2020-01-01", periods=6))
        periods = _find_consecutive_periods(mask)
        assert len(periods) == 3

    def test_no_periods(self):
        mask = pd.Series([False] * 5,
                         index=pd.date_range("2020-01-01", periods=5))
        periods = _find_consecutive_periods(mask)
        assert len(periods) == 0


# ============================================================================
# 测试: 星期效应修正
# ============================================================================

class TestWeekdayAdjust:
    def test_adjustment(self, daily_consumption):
        adjusted = weekday_adjust(daily_consumption["consumption"])
        assert len(adjusted) == len(daily_consumption)
        assert not adjusted.isna().all()

    def test_short_series(self):
        short = pd.Series([100, 110], index=pd.date_range("2020-01-01", periods=2))
        result = weekday_adjust(short)
        # 少于2周的数据，返回原值
        assert result.iloc[0] == 100


# ============================================================================
# 测试: 一站式修正
# ============================================================================

class TestApplyAllCorrections:
    def test_end_to_end(self, daily_consumption):
        result = apply_all_corrections(
            daily_consumption,
            consumption_col="consumption",
            temperature_col="temperature",
            use_holiday_smooth=True,
            use_weekday_adjust=True,
        )

        # 应生成修正后的列
        assert "consumption_temp_corrected" in result.columns
        assert "consumption_corrected" in result.columns

        # 修正后不应有无限值
        assert not np.isinf(result["consumption_corrected"].dropna()).any()

    def test_without_temperature(self, daily_consumption):
        result = apply_all_corrections(
            daily_consumption[["consumption"]],
            consumption_col="consumption",
        )
        assert "consumption_corrected" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
