"""
数据清洗模块单元测试
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# 确保 src 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import (
    load_raw_data,
    standardize_columns,
    check_missing,
    fill_missing,
    detect_outliers_iqr,
    detect_outliers_sigma,
    detect_outliers_rolling_zscore,
    remove_or_flag_outliers,
    prepare_cleaned_data,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_csv_with_missing():
    """创建含缺失值的示例 CSV 文件。"""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=36, freq="ME"),
        "power": [100 + i * 5 + (np.sin(i) * 20) for i in range(36)],
        "temp": [15 + 10 * np.sin(i * np.pi / 6) for i in range(36)],
    })
    # 人为插入缺失值
    df.loc[5, "power"] = np.nan
    df.loc[10:12, "power"] = np.nan
    df.loc[20, "temp"] = np.nan

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        return f.name


@pytest.fixture
def clean_df():
    """创建干净的测试 DataFrame。"""
    dates = pd.date_range("2020-01-01", periods=24, freq="ME")
    return pd.DataFrame({
        "timestamp": dates,
        "consumption": np.arange(100, 124, dtype=float),
        "temperature": 15 + 10 * np.sin(np.arange(24) * np.pi / 6),
    }).set_index("timestamp")


@pytest.fixture
def df_with_outliers():
    """创建含离群值的 DataFrame。"""
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "consumption": np.random.normal(100, 10, 60),
    }).set_index("timestamp")
    # 插入极端值
    df.loc[dates[10], "consumption"] = 500  # 明显离群
    df.loc[dates[30], "consumption"] = -50  # 负值
    return df


# ============================================================================
# 测试: 列名标准化
# ============================================================================

class TestStandardizeColumns:
    def test_rename_standard_columns(self):
        df = pd.DataFrame({"用电量": [1, 2, 3], "温度": [20, 21, 22]})
        result = standardize_columns(df)
        assert "consumption" in result.columns
        assert "temperature" in result.columns

    def test_custom_mapping(self):
        df = pd.DataFrame({"elec": [1, 2], "tmp": [20, 21]})
        result = standardize_columns(df, {"elec": "consumption", "tmp": "temperature"})
        assert "consumption" in result.columns
        assert "temperature" in result.columns


# ============================================================================
# 测试: 缺失值检测
# ============================================================================

class TestCheckMissing:
    def test_no_missing(self, clean_df):
        report = check_missing(clean_df)
        assert (report["missing_count"] == 0).all()

    def test_with_missing(self):
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [np.nan, np.nan, np.nan],
        })
        report = check_missing(df)
        assert report.loc[report["column"] == "a", "missing_rate"].values[0] == pytest.approx(33.33, abs=0.1)
        assert report.loc[report["column"] == "b", "missing_rate"].values[0] == pytest.approx(100.0, abs=0.1)


# ============================================================================
# 测试: 缺失值填补
# ============================================================================

class TestFillMissing:
    def test_forward_fill(self):
        series = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
        df = pd.DataFrame({"value": series})
        result = fill_missing(df, {"value": "forward_fill"})
        assert result["value"].iloc[0] == 1.0
        assert result["value"].iloc[1] == 1.0
        assert result["value"].iloc[2] == 1.0
        assert result["value"].isna().sum() == 0

    def test_linear_interp(self):
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        df = pd.DataFrame({"value": series})
        result = fill_missing(df, {"value": "linear_interp"})
        assert result["value"].iloc[1] == pytest.approx(2.0, abs=0.01)
        assert result["value"].iloc[3] == pytest.approx(4.0, abs=0.01)

    def test_seasonal_interp(self):
        # 创建有季节性的数据
        n = 36
        seasonal = 10 * np.sin(np.arange(n) * 2 * np.pi / 12)
        trend = np.arange(n) * 0.5
        data = pd.Series(seasonal + trend,
                         index=pd.date_range("2020-01-01", periods=n, freq="ME"))
        data.iloc[6] = np.nan  # 人为缺失
        df = pd.DataFrame({"val": data})
        result = fill_missing(df, {"val": "seasonal_interp"})
        assert result["val"].isna().sum() == 0
        # 填补值应接近季节性趋势
        assert abs(result["val"].iloc[6] - (trend[6] + seasonal[6])) < 3.0

    def test_high_missing_drops_column(self):
        df = pd.DataFrame({
            "keep": [1, 2, 3],
            "drop": [np.nan, np.nan, 2],
        })  # 66% 缺失
        result = fill_missing(df)
        assert "keep" in result.columns
        # > 20% 缺失的列被丢弃
        assert "drop" not in result.columns

    def test_zero_fill(self):
        series = pd.Series([1.0, np.nan, np.nan, 4.0])
        df = pd.DataFrame({"value": series})
        result = fill_missing(df, {"value": "zero"})
        assert result["value"].iloc[1] == 0.0
        assert result["value"].iloc[2] == 0.0


# ============================================================================
# 测试: 离群值检测
# ============================================================================

class TestOutlierDetection:
    def test_iqr_normal_data(self, clean_df):
        mask = detect_outliers_iqr(clean_df, "consumption")
        # 干净数据不应有离群值
        assert mask.sum() == 0

    def test_iqr_with_outliers(self, df_with_outliers):
        mask = detect_outliers_iqr(df_with_outliers, "consumption")
        assert mask.sum() >= 2  # 至少检测到两个离群值

    def test_sigma_with_outliers(self, df_with_outliers):
        mask = detect_outliers_sigma(df_with_outliers, "consumption")
        assert mask.sum() >= 1

    def test_rolling_zscore(self, df_with_outliers):
        mask = detect_outliers_rolling_zscore(df_with_outliers, "consumption",
                                              window=10, threshold=2.0)
        assert mask.sum() >= 1


# ============================================================================
# 测试: 离群值处理
# ============================================================================

class TestOutlierHandling:
    def test_flag_strategy(self, df_with_outliers):
        mask = detect_outliers_iqr(df_with_outliers, "consumption")
        result = remove_or_flag_outliers(df_with_outliers, mask,
                                         strategy="flag", column="consumption")
        assert "is_outlier_consumption" in result.columns
        assert result["is_outlier_consumption"].sum() == mask.sum()

    def test_interpolate_strategy(self, df_with_outliers):
        mask = detect_outliers_iqr(df_with_outliers, "consumption")
        result = remove_or_flag_outliers(df_with_outliers, mask,
                                         strategy="interpolate", column="consumption")
        # 极端离群值（500, -50）应在插值后被替换
        # 插值后的结果应更接近均值
        orig_range = df_with_outliers["consumption"].max() - df_with_outliers["consumption"].min()
        result_range = result["consumption"].max() - result["consumption"].min()
        assert result_range < orig_range, (
            f"插值后范围 ({result_range:.1f}) 应小于原始范围 ({orig_range:.1f})"
        )

    def test_cap_strategy(self, df_with_outliers):
        mask = detect_outliers_iqr(df_with_outliers, "consumption")
        result = remove_or_flag_outliers(df_with_outliers, mask,
                                         strategy="cap", column="consumption")
        # 缩尾后，值应在分位数范围内
        q1, q3 = df_with_outliers["consumption"].quantile([0.25, 0.75])
        iqr = q3 - q1
        assert result["consumption"].min() >= q1 - 1.5 * iqr - 0.01
        assert result["consumption"].max() <= q3 + 1.5 * iqr + 0.01

    def test_remove_strategy(self, df_with_outliers):
        mask = detect_outliers_iqr(df_with_outliers, "consumption")
        result = remove_or_flag_outliers(df_with_outliers, mask, strategy="remove")
        assert len(result) < len(df_with_outliers)


# ============================================================================
# 测试: 数据加载
# ============================================================================

class TestLoadRawData:
    def test_load_csv(self, sample_csv_with_missing):
        df = load_raw_data(sample_csv_with_missing,
                          column_mapping={"power": "consumption"})
        assert "consumption" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            load_raw_data("nonexistent_file.csv")


# ============================================================================
# 测试: 一站式清洗
# ============================================================================

class TestPrepareCleanedData:
    def test_end_to_end(self, sample_csv_with_missing):
        df = prepare_cleaned_data(
            sample_csv_with_missing,
            column_mapping={"power": "consumption", "temp": "temperature"},
            fill_strategy={"consumption": "linear_interp", "temperature": "linear_interp"},
            outlier_column="consumption",
            outlier_method="iqr",
            outlier_strategy="flag",
        )
        assert "consumption" in df.columns
        assert "temperature" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
