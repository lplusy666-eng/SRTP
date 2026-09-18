"""
三阶段滚动回归模型

参考: Lan et al. (2025), PLOS ONE
  "Composite GDP nowcasting using macroeconomic variables and electricity data"

核心思想:
  在季度内的第1、2、3月末，分别能获取不同数量的月度用电数据。
  因此针对每个阶段设计不同的回归方程，使用9季度滚动窗口动态估计参数。

三阶段回归方程:
  Stage 1 (第1月末): GDP ~ β₀ + β₁·prev_GDP + β₂·M1_YoY + β₃·M1_capacity_rate
  Stage 2 (第2月末): GDP ~ β₀ + β₁·prev_GDP + β₂·M1_YoY + β₃·M2_YoY + β₄·M2_capacity_rate
  Stage 3 (第3月末): GDP ~ β₀ + β₁·prev_GDP + β₂·M1_YoY + β₃·M2_YoY + β₄·M3_YoY

滚动窗口:
  对每个预测季度，使用过去9个季度的数据估计模型参数，
  以捕捉电力-经济关系随时间的变化。
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from .config import ROLLING_WINDOW_QUARTERS
except ImportError:
    from config import ROLLING_WINDOW_QUARTERS

logger = logging.getLogger(__name__)


class ThreeStageRegression:
    """
    Lan et al. (2025) 三阶段季度 GDP 预测模型。

    使用方法:
        >>> model = ThreeStageRegression(window_quarters=9)
        >>> model.fit(df)  # df 需包含月度用电 + 季度 GDP 数据
        >>> predictions = model.predict(df)
        >>> coefs = model.get_coefficients()
    """

    # 三阶段特征配置
    STAGE_FEATURES = {
        1: ["prev_gdp", "month1_yoy", "month1_cap_rate"],
        2: ["prev_gdp", "month1_yoy", "month2_yoy", "month2_cap_rate"],
        3: ["prev_gdp", "month1_yoy", "month2_yoy", "month3_yoy"],
    }

    def __init__(self, window_quarters: int = ROLLING_WINDOW_QUARTERS):
        """
        参数:
            window_quarters: 滚动窗口大小（季度数），论文默认 9
        """
        self.window_quarters = window_quarters
        self.models: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
        self.predictions: Optional[pd.DataFrame] = None
        self.coefficient_history: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        gdp_col: str = "gdp",
        yoy_col: str = "consumption_yoy",
        cap_rate_col: str = "capacity_change_rate",
    ) -> "ThreeStageRegression":
        """
        用历史数据拟合三阶段滚动回归模型。

        数据准备:
            输入 DataFrame 需按时间排序的月度数据。
            该方法自动将月度数据组织为季度级的三阶段格式。

        参数:
            df: 月度 DataFrame (DatetimeIndex)
            gdp_col: GDP 列名（季度值，其他月份为 NaN）
            yoy_col: 用电量同比增速列名
            cap_rate_col: 容量净增量变化率列名

        返回:
            self (链式调用)
        """
        df = df.copy()

        # 验证必要列
        required = [yoy_col]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"必要列 '{col}' 不存在于 DataFrame")

        # 构建季度级特征矩阵
        quarterly_df = self._build_quarterly_features(
            df, gdp_col=gdp_col, yoy_col=yoy_col, cap_rate_col=cap_rate_col
        )

        if len(quarterly_df) < self.window_quarters + 1:
            logger.warning(
                f"数据量 ({len(quarterly_df)} 个季度) 不足，"
                f"需要至少 {self.window_quarters + 1} 个季度"
            )
            return self

        # 对每个阶段分别拟合滚动窗口
        for stage in [1, 2, 3]:
            features = self.STAGE_FEATURES[stage]
            self._fit_stage_rolling(quarterly_df, stage, features)

        self._is_fitted = True
        logger.info(f"三阶段回归拟合完成: {len(quarterly_df)} 个季度, "
                    f"窗口={self.window_quarters}")

        return self

    def _build_quarterly_features(
        self,
        df: pd.DataFrame,
        gdp_col: str = "gdp",
        yoy_col: str = "consumption_yoy",
        cap_rate_col: str = "capacity_change_rate",
    ) -> pd.DataFrame:
        """将月度 DataFrame 转换为季度三阶段格式。"""
        # 添加季度和月份标记
        df = df.copy()
        df["quarter"] = df.index.to_period("Q")
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["quarter_month"] = df["month"].map({1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3,
                                                7: 1, 8: 2, 9: 3, 10: 1, 11: 2, 12: 3})

        # 每个季度的三个月份数据
        quarterly_records = []

        for q, group in df.groupby("quarter"):
            record = {"quarter": q}

            # 前季 GDP（如有）
            prev_q = q - 1
            prev_gdp_data = df[df.index.to_period("Q") == prev_q]
            if gdp_col in df.columns:
                prev_gdp = prev_gdp_data[gdp_col].dropna()
                record["prev_gdp"] = prev_gdp.iloc[-1] if len(prev_gdp) > 0 else np.nan

            # 本月1/2/3的用电量 YoY
            for qm in [1, 2, 3]:
                qm_data = group[group["quarter_month"] == qm]
                if len(qm_data) > 0:
                    record[f"month{qm}_yoy"] = qm_data[yoy_col].iloc[-1] if yoy_col in df.columns else np.nan
                    if cap_rate_col in df.columns:
                        record[f"month{qm}_cap_rate"] = qm_data[cap_rate_col].iloc[-1]
                else:
                    record[f"month{qm}_yoy"] = np.nan
                    record[f"month{qm}_cap_rate"] = np.nan

            # 当前季度 GDP（目标变量）
            if gdp_col in df.columns:
                target_gdp = group[gdp_col].dropna()
                record["gdp"] = target_gdp.iloc[-1] if len(target_gdp) > 0 else np.nan

            quarterly_records.append(record)

        quarterly_df = pd.DataFrame(quarterly_records).set_index("quarter")
        return quarterly_df

    def _fit_stage_rolling(
        self,
        quarterly_df: pd.DataFrame,
        stage: int,
        features: List[str],
    ):
        """对指定阶段执行滚动窗口 OLS 回归。"""
        available_features = [f for f in features if f in quarterly_df.columns]

        if len(available_features) < 2:
            logger.warning(f"Stage {stage}: 可用特征不足 ({available_features})")
            return

        n_quarters = len(quarterly_df)

        for i in range(self.window_quarters, n_quarters):
            # 滚动窗口: [i-window, i)
            train_start = i - self.window_quarters
            train_end = i

            train_data = quarterly_df.iloc[train_start:train_end]
            test_data = quarterly_df.iloc[i:i + 1]

            # 提取训练特征和目标
            X_train = train_data[available_features].dropna()
            y_train = train_data.loc[X_train.index, "gdp"].dropna()

            # 对齐
            common_idx = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common_idx]
            y_train = y_train.loc[common_idx]

            if len(X_train) < len(available_features) + 1:
                continue  # 样本不足以拟合

            # 添加常数项
            X_train_sm = sm.add_constant(X_train)

            try:
                model = sm.OLS(y_train, X_train_sm).fit()
            except Exception as e:
                logger.debug(f"Stage {stage}, quarter {test_data.index[0]}: OLS 失败: {e}")
                continue

            # 预测
            X_test = test_data[available_features].dropna()
            if len(X_test) == 0:
                continue

            X_test_sm = sm.add_constant(X_test)
            X_test_sm = X_test_sm.reindex(columns=X_train_sm.columns, fill_value=0)

            try:
                pred = model.predict(X_test_sm).iloc[0]
                actual = test_data["gdp"].iloc[0] if "gdp" in test_data.columns else np.nan
            except Exception:
                continue

            # 存储结果
            self.models[stage].append({
                "quarter": test_data.index[0],
                "prediction": pred,
                "actual": actual,
                "coefficients": model.params.to_dict(),
                "r_squared": model.rsquared,
                "rmse": np.sqrt(model.mse_resid),
                "p_values": model.pvalues.to_dict(),
                "train_quarters": (train_data.index[0], train_data.index[-1]),
            })

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用拟合的模型进行预测。

        参数:
            df: 输入 DataFrame（与 fit 相同的列结构）

        返回:
            DataFrame with columns:
                - quarter: 预测季度
                - stage: 阶段 (1/2/3)
                - prediction: 预测 GDP 增速
                - actual: 实际 GDP 增速（如有）
                - error: 预测误差
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

        all_predictions = []

        for stage in [1, 2, 3]:
            for result in self.models[stage]:
                all_predictions.append({
                    "quarter": result["quarter"],
                    "stage": stage,
                    "prediction": result["prediction"],
                    "actual": result["actual"],
                    "error": result["prediction"] - result["actual"]
                    if not np.isnan(result["actual"]) else np.nan,
                    "r_squared": result["r_squared"],
                })

        self.predictions = pd.DataFrame(all_predictions)

        # 按季度合并三阶段预测
        if len(self.predictions) > 0:
            self.predictions = self.predictions.sort_values(
                ["quarter", "stage"]
            ).reset_index(drop=True)

        return self.predictions

    def get_coefficients(self) -> pd.DataFrame:
        """
        返回各阶段回归系数的历史变化。

        用于分析各因素（用电增长、容量变化等）对 GDP 预测的贡献度
        随时间如何演变。

        返回:
            DataFrame，每行为一次滚动窗口回归的系数
        """
        all_coefs = []

        for stage in [1, 2, 3]:
            for result in self.models[stage]:
                coef_row = {
                    "stage": stage,
                    "quarter": result["quarter"],
                    "r_squared": result["r_squared"],
                }
                coef_row.update(result["coefficients"])
                all_coefs.append(coef_row)

        self.coefficient_history = pd.DataFrame(all_coefs)
        return self.coefficient_history

    def evaluate(self) -> Dict[str, Any]:
        """
        评估模型性能。

        返回:
            包含各阶段评估指标的字典:
                - stage_1/2/3: {"rmse", "mae", "r2", "n_predictions"}
                - overall: 综合评估
        """
        if self.predictions is None:
            # Build predictions from stored model results
            all_predictions = []
            for stage in [1, 2, 3]:
                for result in self.models[stage]:
                    all_predictions.append({
                        "quarter": result["quarter"],
                        "stage": stage,
                        "prediction": result["prediction"],
                        "actual": result["actual"],
                        "error": result["prediction"] - result["actual"]
                        if not np.isnan(result["actual"]) else np.nan,
                        "r_squared": result["r_squared"],
                    })
            if not all_predictions:
                return {}
            self.predictions = pd.DataFrame(all_predictions)

        metrics = {}
        df = self.predictions.copy()
        df_valid = df[df["actual"].notna() & df["prediction"].notna()]

        for stage in [1, 2, 3]:
            stage_data = df_valid[df_valid["stage"] == stage]
            if len(stage_data) > 0:
                metrics[f"stage_{stage}"] = {
                    "rmse": np.sqrt(mean_squared_error(stage_data["actual"], stage_data["prediction"])),
                    "mae": mean_absolute_error(stage_data["actual"], stage_data["prediction"]),
                    "r2": r2_score(stage_data["actual"], stage_data["prediction"]),
                    "n_predictions": len(stage_data),
                }

        # 综合评估
        if len(df_valid) > 0:
            metrics["overall"] = {
                "rmse": np.sqrt(mean_squared_error(df_valid["actual"], df_valid["prediction"])),
                "mae": mean_absolute_error(df_valid["actual"], df_valid["prediction"]),
                "r2": r2_score(df_valid["actual"], df_valid["prediction"]),
                "n_predictions": len(df_valid),
            }

        # 打印评估结果
        for key, vals in metrics.items():
            logger.info(f"{key}: RMSE={vals['rmse']:.4f}, "
                        f"MAE={vals['mae']:.4f}, "
                        f"R²={vals['r2']:.4f}, "
                        f"N={vals['n_predictions']}")

        return metrics

    def get_stage_improvement(self) -> Dict[str, float]:
        """
        计算三阶段逐步改进效果。

        随着季度内可用月份增加（第1月→第3月），
        预测精度应该逐步提升。此方法量化这种改进。

        返回:
            {"stage1_to_stage2_rmse_reduction": ..., "stage2_to_stage3_rmse_reduction": ...}
        """
        if self.predictions is None:
            self.predict()

        metrics = self.evaluate()

        improvement = {}
        rmse_1 = metrics.get("stage_1", {}).get("rmse")
        rmse_2 = metrics.get("stage_2", {}).get("rmse")
        rmse_3 = metrics.get("stage_3", {}).get("rmse")

        if rmse_1 and rmse_2:
            improvement["stage1_to_stage2_pct"] = (rmse_1 - rmse_2) / rmse_1 * 100
        if rmse_2 and rmse_3:
            improvement["stage2_to_stage3_pct"] = (rmse_2 - rmse_3) / rmse_2 * 100
        if rmse_1 and rmse_3:
            improvement["stage1_to_stage3_pct"] = (rmse_1 - rmse_3) / rmse_1 * 100

        return improvement

    def summary(self) -> str:
        """返回模型摘要字符串。"""
        if not self._is_fitted:
            return "ThreeStageRegression (未拟合)"

        n_stage_1 = len(self.models[1])
        n_stage_2 = len(self.models[2])
        n_stage_3 = len(self.models[3])

        lines = [
            "=" * 60,
            "ThreeStageRegression 模型摘要",
            "=" * 60,
            f"滚动窗口: {self.window_quarters} 个季度",
            f"拟合次数: Stage1={n_stage_1}, Stage2={n_stage_2}, Stage3={n_stage_3}",
        ]

        try:
            metrics = self.evaluate()
            for key, vals in metrics.items():
                lines.append(f"{key}: RMSE={vals['rmse']:.4f}, "
                             f"MAE={vals['mae']:.4f}, R²={vals['r2']:.4f}")
        except Exception as e:
            lines.append(f"评估失败: {e}")

        lines.append("=" * 60)
        return "\n".join(lines)
