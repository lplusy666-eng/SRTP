from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def parse_month(value: object) -> pd.Timestamp:
    text = str(value).strip()
    match = re.search(r"(\d{4})\D+(\d{1,2})", text)
    if match:
        return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=1)
    try:
        ts = pd.to_datetime(text)
        return pd.Timestamp(year=ts.year, month=ts.month, day=1)
    except Exception as exc:
        raise ValueError(f"无法解析月份: {value!r}") from exc


class AKShareMacroClient:
    """Fetch national monthly macro indicators through AKShare.

    Upstream web pages can change. Every call is isolated so one failed source does
    not destroy the whole dataset; the missing column can later be imputed.
    """

    def __init__(self):
        try:
            import akshare as ak
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "未安装 AKShare。请执行 pip install '.[real-data]'，或把 macro.provider 改为 csv/synthetic。"
            ) from exc
        self.ak = ak

    def _safe(self, name: str, fn: Callable[[], pd.DataFrame], converter: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        try:
            raw = fn()
            if raw is None or raw.empty:
                raise ValueError("空数据")
            result = converter(raw.copy())
            LOGGER.info("AKShare %s: %s 行", name, len(result))
            return result
        except Exception as exc:  # pragma: no cover - network/upstream dependent
            LOGGER.warning("AKShare %s 获取失败，将留空: %s", name, exc)
            return pd.DataFrame(columns=["month"])

    @staticmethod
    def _simple(raw: pd.DataFrame, date_col: str, value_map: dict[str, str]) -> pd.DataFrame:
        cols = [date_col, *value_map.keys()]
        missing = [c for c in cols if c not in raw.columns]
        if missing:
            raise KeyError(f"缺少列 {missing}; 实际列={list(raw.columns)}")
        out = raw[cols].rename(columns=value_map)
        out["month"] = raw[date_col].map(parse_month)
        out = out.drop(columns=[date_col], errors="ignore")
        for col in value_map.values():
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.sort_values("month").drop_duplicates("month", keep="last")

    def fetch(self) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        parts.append(
            self._safe(
                "industrial",
                self.ak.macro_china_gyzjz,
                lambda df: self._simple(df, "月份", {"同比增长": "industrial_yoy"}),
            )
        )
        parts.append(
            self._safe(
                "pmi",
                self.ak.macro_china_pmi,
                lambda df: self._simple(df, "月份", {"制造业-指数": "pmi"}),
            )
        )
        parts.append(
            self._safe(
                "retail",
                self.ak.macro_china_consumer_goods_retail,
                lambda df: self._simple(df, "月份", {"同比增长": "retail_yoy"}),
            )
        )
        parts.append(
            self._safe(
                "society_electricity",
                self.ak.macro_china_society_electricity,
                lambda df: self._simple(
                    df,
                    "统计时间",
                    {"全社会用电量同比": "power_consumption_yoy"},
                ),
            )
        )
        # PPI is used as an energy/input-price proxy, not as a literal tariff series.
        parts.append(
            self._safe(
                "ppi_proxy",
                self.ak.macro_china_ppi,
                lambda df: self._simple(df, "月份", {"当月同比增长": "electricity_price_index"}),
            )
        )

        merged: pd.DataFrame | None = None
        for part in parts:
            if part.empty or len(part.columns) <= 1:
                continue
            merged = part if merged is None else merged.merge(part, on="month", how="outer")
        if merged is None:
            raise RuntimeError("所有 AKShare 宏观接口均失败")
        merged = merged.sort_values("month")
        merged["renewable_share"] = np.nan
        return merged
