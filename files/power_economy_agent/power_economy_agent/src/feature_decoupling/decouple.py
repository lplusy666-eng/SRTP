"""
特征解耦（Feature Decoupling）—— 项目核心创新点之一
=====================================================
目标：从电量序列中剥离"非经济扰动"（天气/节假日/周季节性），
      分离出可解释的"经济信号"，并做跨尺度特征分析。

三步：
  1) 混杂因子回归剥离：log_elec ~ CDD + HDD + 周末 + 节假日 + 春节
     → 得到 weather_effect / calendar_effect / economic_signal
  2) STL 分解经济信号 → economic_trend(景气基线) + economic_residual(冲击候选)
  3) 小波多尺度分解 → 各尺度能量特征（跨尺度特征）

轻量化：全部基于 OLS + STL + DWT，无需大规模训练，可秒级完成。
"""
import numpy as np
import pandas as pd
import pywt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir, load_config

log = get_logger("decouple")


# ---------- 步骤1：混杂因子回归剥离 ----------
def _cny_proximity_dummies(df: pd.DataFrame, span: int = 21):
    """距春节 ±span 天逐日哑变量，让回归精确学到完整春节用电廓线并剔除。"""
    d2c = df["days_to_cny"].values if "days_to_cny" in df.columns else np.zeros(len(df))
    out = {}
    for k in range(-span, span + 1):
        out[f"cny_d{k:+d}"] = (d2c == k).astype(float)
    return pd.DataFrame(out, index=df.index)


def remove_confounders(df: pd.DataFrame):
    """用 OLS 剥离天气与日历效应，返回经济信号及各效应分量。"""
    y = df["log_elec"].values
    base_cols = ["CDD", "HDD", "is_weekend", "is_holiday"]
    X = df[base_cols].astype(float).copy()
    cny = _cny_proximity_dummies(df)          # 春节邻近分段
    X = pd.concat([X, cny], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    params = model.params

    weather_effect = (params["CDD"] * df["CDD"] + params["HDD"] * df["HDD"]).values
    calendar_effect = (params["is_weekend"] * df["is_weekend"]
                       + params["is_holiday"] * df["is_holiday"]).values
    # 春节效应（分段哑变量之和）
    cny_effect = np.zeros(len(df))
    for c in cny.columns:
        cny_effect += params[c] * cny[c].values
    calendar_effect = calendar_effect + cny_effect

    economic_signal = y - weather_effect - calendar_effect

    log.info("混杂因子回归 R²=%.3f | 天气&日历系数: %s", model.rsquared,
             {k: round(params[k], 4) for k in base_cols})
    return economic_signal, weather_effect, calendar_effect, model


# ---------- 步骤2：STL 分离趋势与冲击 ----------
def stl_decompose(signal: np.ndarray, period: int):
    """STL 分解经济信号：趋势(景气基线) + 季节 + 残差(冲击候选)。"""
    s = pd.Series(signal)
    # period 至少为7；若数据长可用年周期辅助去除残余季节性
    stl = STL(s, period=max(period, 7), robust=True)
    res = stl.fit()
    return res.trend.values, res.seasonal.values, res.resid.values


# ---------- 步骤3：小波跨尺度分解 ----------
def wavelet_multiscale(signal: np.ndarray, wavelet="db4", level=4):
    """离散小波分解，返回各尺度重构分量与能量占比（跨尺度特征）。"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 逐尺度重构
    recons = []
    for i in range(len(coeffs)):
        c = [np.zeros_like(x) for x in coeffs]
        c[i] = coeffs[i]
        rec = pywt.waverec(c, wavelet)[: len(signal)]
        recons.append(rec)
    # 能量占比
    energies = np.array([np.sum(r ** 2) for r in recons])
    energy_ratio = energies / (energies.sum() + 1e-12)
    scale_names = ["A%d(趋势)" % level] + ["D%d" % (level - i) for i in range(level)]
    log.info("小波尺度能量占比: %s",
             {n: round(float(e), 3) for n, e in zip(scale_names, energy_ratio)})
    return recons, scale_names, energy_ratio


# ---------- 主流程 ----------
def run_decoupling(cfg: dict) -> pd.DataFrame:
    merged = abspath(cfg["data"]["merged_csv"])
    df = pd.read_csv(merged, parse_dates=["date"])

    economic_signal, weather_effect, calendar_effect, ols = remove_confounders(df)

    # 去除残余年内气候态（day-of-year 平滑季节廓线），消除非线性温度等造成的年度季节性
    doy = pd.DatetimeIndex(df["date"]).dayofyear
    tmp = pd.DataFrame({"doy": doy, "sig": economic_signal})
    clim = tmp.groupby("doy")["sig"].mean()
    clim = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
    # 循环平滑
    ext = np.concatenate([clim.values[-15:], clim.values, clim.values[:15]])
    smooth = pd.Series(ext).rolling(15, center=True, min_periods=1).mean().values[15:-15]
    clim_smooth = pd.Series(smooth, index=range(1, 367)) - np.mean(smooth)
    seasonal_annual = clim_smooth.reindex(doy).values
    economic_signal = economic_signal - seasonal_annual

    period = cfg["decoupling"]["stl_period"]
    trend, seasonal, resid = stl_decompose(economic_signal, period)

    # 小波作用于"去趋势波动"(经济信号 - 景气基线)，聚焦跨尺度波动结构
    fluctuation = economic_signal - trend
    recons, scale_names, energy_ratio = wavelet_multiscale(
        fluctuation, cfg["decoupling"]["wavelet"], cfg["decoupling"]["wavelet_level"])

    out = pd.DataFrame({
        "date": df["date"],
        "log_elec": df["log_elec"],
        "weather_effect": weather_effect,
        "calendar_effect": calendar_effect,
        "economic_signal": economic_signal,       # 剥离扰动后的经济信号
        "economic_trend": trend,                  # 景气基线（趋势+周期）
        "economic_residual": resid,               # 冲击候选（异常检测输入）
        "seasonal_residual": seasonal,
    })
    # 附加：最高频细节分量能量（用于后续解释"波动剧烈程度"）
    out["wavelet_highfreq"] = recons[-1]

    # 相对长期基线的偏离：捕捉 STL 残差易漏掉的"持续性"经济冲击
    baseline = out["economic_signal"].rolling(91, center=True, min_periods=15).median()
    out["economic_deviation"] = (out["economic_signal"] - baseline).fillna(0.0)
    if "_true_anomaly" in df.columns:
        out["_true_anomaly"] = df["_true_anomaly"]

    ensure_dir(abspath(cfg["decoupling"]["output_csv"]))
    outpath = abspath(cfg["decoupling"]["output_csv"])
    out.to_csv(outpath, index=False)
    log.info("解耦结果已保存: %s  形状=%s", outpath, out.shape)
    return out


if __name__ == "__main__":
    cfg = load_config()
    d = run_decoupling(cfg)
    print(d.head())
    print(d[["economic_signal", "economic_trend", "economic_residual"]].describe())
