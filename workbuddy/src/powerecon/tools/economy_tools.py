"""经济感知工具：用电同比与 GDP 同比的偏离度。

这里刻意不做"用电量 → GDP 增速"的回归预测。原因是：
本项目的样本是省级月度/季度数据，几十个点，用这么少的样本做回归，
出来的系数不稳定、外推更不可信，还会给使用者一种"系统能预测 GDP"的错觉。

所以这个工具做的是**偏离度对照**：把用电同比和同期 GDP 同比摆在一起，
看两者差了多少个百分点。偏离本身就是一个有价值的信号 ——
它说明"用电的变化没有对应的经济变化来解释"，但绝不等于经济出了问题。
"""

from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel, Field

from ..contract import EconomicNowcast, Frequency
from ..series_utils import parse_period
from .base import ToolContext, tool


def _to_pct(value) -> float | None:
    """源表 yoy 有的存成 8.5，有的存成 0.085，按量级归一到百分数。"""
    if value is None or pd.isna(value):
        return None
    v = float(value)
    return v if abs(v) > 1.5 else v * 100


def _lookup_gdp_yoy(ctx: ToolContext, targets: set[pd.Timestamp]) -> tuple[float | None, str | None]:
    """在季度经济表里找与目标期次对齐的同比，优先第二产业（与工业用电关联最直接）。

    兼容两种表结构：
    - 长表（浙江）：一行一个产业，有 indicator_name 列；
    - 宽表（江苏）：一行多个产业列，如 secondary_yoy_pct。
    """
    gdp_id = next(
        (k for k in ctx.region.datasets if "industry" in k.lower() or "gdp" in k.lower()),
        None,
    )
    if gdp_id is None:
        return None, None
    try:
        gdp = ctx.region.frame(gdp_id)
    except Exception:
        return None, None

    pcol = next((c for c in ("period", "quarter", "month") if c in gdp.columns), None)
    if pcol is None:
        return None, None

    sub = gdp[gdp[pcol].map(lambda v: parse_period(v) in targets)]
    if sub.empty:
        return None, None

    name_col = next((c for c in sub.columns if "indicator_name" in str(c).lower()), None)
    if name_col is not None:
        pref = sub[sub[name_col].astype(str).str.contains("第二产业|工业", na=False)]
        row = (pref if not pref.empty else sub).iloc[0]
        label = str(row[name_col])
        yoy_col = next((c for c in sub.columns if "yoy" in str(c).lower()), None)
        if yoy_col is None:
            return None, label
        return _to_pct(pd.to_numeric(pd.Series([row[yoy_col]]), errors="coerce").iloc[0]), label

    row = sub.iloc[0]
    cols = list(sub.columns)
    pref = [
        c for c in cols
        if re.search(r"secondary|第二产业|industry", str(c), re.I) and "yoy" in str(c).lower()
    ]
    yoy_col = pref[0] if pref else next((c for c in cols if "yoy" in str(c).lower()), None)
    if yoy_col is None:
        return None, None
    label = "第二产业增加值" if pref else "GDP 总量"
    return _to_pct(pd.to_numeric(pd.Series([row[yoy_col]]), errors="coerce").iloc[0]), label


class NowcastArgs(BaseModel):
    period: str = Field(description="要对齐的期次，例如 '2025-01'（月频）或 '2024Q4'（季频）。")


@tool(
    name="nowcast_economy",
    description=(
        "把用电量同比与同期 GDP 同比摆在一起，输出两者的偏离度。\n"
        "**它不做 GDP 预测。** 省级月度样本只有几十个点，回归外推不可信，"
        "所以这个工具只做对照：偏离度大，说明用电变化没有对应的经济变化来解释。\n"
        "月度期次会自动映射到所在季度再匹配 GDP，因为经济指标是季度的。\n"
        "典型用法：在 explain_anomaly 之后，用这个工具交叉核对"
        "「这次用电偏离是否伴随同向的经济变化」。\n"
        "如果偏离度很大，正确结论是「用电与经济指标出现背离，需要进一步排查」，"
        "而不是「经济走弱」或「经济向好」。"
    ),
    args_model=NowcastArgs,
    returns="EconomicNowcast：用电同比、GDP同比、偏离度与解释边界",
    tags=("analysis", "economy"),
)
def nowcast_economy(ctx: ToolContext, args: NowcastArgs) -> BaseModel:
    target = parse_period(args.period)
    if target is None:
        raise ValueError(f"期次 '{args.period}' 无法解析。")

    elec_id = next((k for k in ctx.region.datasets if "electricity" in k.lower()), None)
    if elec_id is None:
        elec_id = next((k for k, v in ctx.region.datasets.items() if "用电" in v.metric), None)
    if elec_id is None:
        raise ValueError("该地区没有可识别的用电量数据集。")

    elec = ctx.region.series(elec_id).sort_values("period").reset_index(drop=True)
    freq = ctx.region.datasets[elec_id].frequency
    lag = 12 if freq is Frequency.MONTHLY else 4

    elec_yoy = None
    match = elec.index[elec["period"] == target]
    if len(match):
        i = int(match[0])
        if i - lag >= 0 and abs(elec["value"].iloc[i - lag]) > 1e-12:
            elec_yoy = float(
                (elec["value"].iloc[i] - elec["value"].iloc[i - lag])
                / abs(elec["value"].iloc[i - lag])
            )

    # 月度/日度期次映射到所在季度的季初，因为经济指标是季度频率的
    targets = {target}
    if freq in (Frequency.MONTHLY, Frequency.DAILY):
        targets.add(pd.Timestamp(year=target.year, month=(target.quarter - 1) * 3 + 1, day=1))
    gdp_yoy, gdp_indicator = _lookup_gdp_yoy(ctx, targets)

    caveats: list[str] = []
    if elec_yoy is not None and gdp_yoy is not None:
        divergence = (elec_yoy * 100) - gdp_yoy
        divergence_note = (
            f"用电同比 {elec_yoy * 100:+.2f}%，{gdp_indicator or 'GDP'} 同比 {gdp_yoy:+.2f}%，"
            f"相差 {divergence:+.2f} 个百分点。"
        )
        if abs(divergence) > 10:
            divergence_note += (
                "偏离超过 10 个百分点，属于显著背离。"
                "常见非经济解释：春节错位、气温异常、统计口径差异（累计值 vs 当月值）、"
                "上年同期基数异常。这些必须先排除，才能谈经济含义。"
            )
            caveats.append("用电与经济指标显著背离，但背离原因未定，不能直接解读为经济变化。")
        else:
            divergence_note += "偏离在常见波动范围内，用电与经济走势基本一致。"
    else:
        missing = []
        if elec_yoy is None:
            missing.append("目标期次的上年同期用电数据")
        if gdp_yoy is None:
            missing.append("对齐季度或同期的 GDP 同比")
        divergence_note = f"缺少{'与'.join(missing)}，无法做偏离度对照。"
        caveats.append("缺少经济指标对照，用电变化无法与经济变化交叉验证。")

    caveats.append(
        "本工具不做 GDP 预测：省级月度样本量不足以支撑可靠的回归外推。输出的是对照关系，不是预测值。"
    )
    if gdp_indicator:
        caveats.append(f"对照使用的经济指标为「{gdp_indicator}」，为年内累计口径同比。")
    if freq is Frequency.MONTHLY:
        caveats.append("用电为月度当月值，GDP 为季度累计值，两者时间口径并不严格对齐。")

    return EconomicNowcast(
        region=ctx.region.info.name,
        period=args.period,
        indicator=gdp_indicator or "（未找到可对照的经济指标）",
        estimated_yoy_pct=None,
        electricity_yoy_pct=None if elec_yoy is None else elec_yoy * 100,
        method="electricity-vs-gdp-divergence",
        n_observations=int(len(elec)),
        divergence_note=divergence_note,
        caveats=caveats,
    )
