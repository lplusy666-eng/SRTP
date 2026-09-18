"""上下文类工具：节假日日历与天气。

这两个工具的作用是给"残差异常"提供**排除性证据**。
春节错位、高温负荷这类非经济因素必须先被排除，剩下的才可能是经济信号。

设计上的关键一点：当某个地区没有天气或日历数据时，工具**明确返回 available=false**，
而不是返回空列表。因为模型对空列表的反应往往是"没有异常"，对 available=false 才会
正确地在结论里写上"缺少天气数据，无法排除高温因素"。这是防止幻觉的关键。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ..contract import CalendarContext, WeatherContext
from ..series_utils import parse_period
from .base import ToolContext, tool

DATE_HINTS = ("date", "日期")
HOLIDAY_NAME_HINTS = ("holiday_name", "holiday_names", "节假日", "节日")
WORKDAY_HINTS = ("is_workday", "workdays")
NONWORK_HINTS = ("is_nonworkday", "nonworkdays")
BREAK_HINTS = ("holiday_break_days",)
MAKEUP_HINTS = ("makeup_workdays",)
DAYTYPE_HINTS = ("day_type",)

TEMP_HINTS = ("temperature_mean_c", "mean_temp_c", "temp_mean", "月均气温")
HDD_HINTS = ("heating_degree_days",)
CDD_HINTS = ("cooling_degree_days",)
HOT_HINTS = ("hot_days_ge_35c",)
COLD_HINTS = ("cold_days_le_0c",)


def _pick(df: pd.DataFrame, hints: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for h in hints:
        if h in lowered:
            return lowered[h]
    for low, original in lowered.items():
        if any(h in low for h in hints):
            return original
    return None


def _find_calendar_frame(ctx: ToolContext) -> tuple[pd.DataFrame, str] | None:
    """优先用日粒度日历，没有就退到月粒度。两种列名结构都能吃。"""
    for ds_id in ("calendar_daily", "calendar_monthly"):
        if ds_id in ctx.region.datasets:
            try:
                return ctx.region.frame(ds_id), ds_id
            except Exception:
                continue
    for ds_id, info in ctx.region.datasets.items():
        if info.kind.value == "calendar":
            try:
                return ctx.region.frame(ds_id), ds_id
            except Exception:
                continue
    return None


class CalendarArgs(BaseModel):
    start: str = Field(description="起始期次，如 '2025-01'。")
    end: str = Field(description="结束期次，如 '2025-03'。")


@tool(
    name="get_calendar_context",
    description=(
        "查询指定区间的日历结构：工作日数、非工作日数、法定放假天数、调休上班天数、"
        "具体节假日名称，以及春节落在区间内的天数。\n"
        "**这是排除春节错位的关键工具。** 春节在 1 月还是 2 月，会让单月同比出现几十个百分点的"
        "正负摆动，纯属日历效应。凡是 1—2 月的同比异常，必须先看这里再下结论。\n"
        "如果返回 n_days=0，说明该地区没有日历数据，此时不能排除节假日因素，"
        "必须在结论的不确定性里写明。"
    ),
    args_model=CalendarArgs,
    returns="CalendarContext：工作日结构 + 节假日清单 + 春节天数",
    tags=("context", "calendar"),
)
def get_calendar_context(ctx: ToolContext, args: CalendarArgs) -> BaseModel:
    found = _find_calendar_frame(ctx)
    if found is None:
        return CalendarContext(
            region=ctx.region.info.name,
            start=args.start,
            end=args.end,
            notes=["该地区没有日历数据集，无法排除节假日因素，结论中必须声明这一缺失。"],
        )

    df, ds_id = found
    date_col = _pick(df, DATE_HINTS) or next(
        (c for c in df.columns if str(c).strip().lower() in ("month", "period")), None
    )
    if date_col is None:
        return CalendarContext(
            region=ctx.region.info.name,
            start=args.start,
            end=args.end,
            notes=[f"日历数据集 {ds_id} 缺少可识别的期次列，实际列：{list(df.columns)}"],
        )

    lo, hi = parse_period(args.start), parse_period(args.end)
    if lo is None or hi is None:
        raise ValueError("start / end 无法解析，请使用 '2025-01' 或 '2025-01-01' 这类格式。")

    parsed = df[date_col].map(parse_period)
    mask = parsed.notna() & (parsed >= lo) & (parsed <= hi + pd.DateOffset(days=31))
    sub = df[mask].copy()
    sub["_period"] = parsed[mask]

    holiday_col = _pick(sub, HOLIDAY_NAME_HINTS)
    work_col = _pick(sub, WORKDAY_HINTS)
    nonwork_col = _pick(sub, NONWORK_HINTS)
    break_col = _pick(sub, BREAK_HINTS)
    makeup_col = _pick(sub, MAKEUP_HINTS)

    def _num(col: str | None, default: float = 0.0) -> float:
        if col is None:
            return default
        s = pd.to_numeric(sub[col], errors="coerce")
        return float(s.fillna(0).sum())

    is_daily = len(sub) > 0 and (sub["_period"].dt.day > 1).any()
    n_days = int(len(sub)) if is_daily else int(_num(_pick(sub, ("days",)), 0.0))
    n_workdays = int(_num(work_col))
    n_nonwork = int(_num(nonwork_col))
    break_days = int(_num(break_col))
    makeup = int(_num(makeup_col))

    holidays: list[dict[str, Any]] = []
    if holiday_col is not None:
        for name, grp in sub.groupby(holiday_col):
            label = str(name).strip()
            if not label or label.lower() in ("nan", "none", ""):
                continue
            holidays.append(
                {
                    "name": label,
                    "n_days": int(len(grp)),
                    "first_day": str(grp["_period"].min().date()),
                    "last_day": str(grp["_period"].max().date()),
                }
            )

    spring_days: int | None = None
    spring_note: str | None = None
    if holiday_col is not None:
        spring_mask = sub[holiday_col].astype(str).str.contains("春节", na=False)
        if spring_mask.any():
            spring_days = int(spring_mask.sum())
            first = sub.loc[spring_mask, "_period"].min()
            spring_note = (
                f"春节落在本期 {spring_days} 天，起始 {first.date()}。"
                f"春节期间工业停工、商业活动减少，用电会显著低于常态，"
                f"这是日历效应而非经济走弱。"
            )

    notes: list[str] = []
    if not is_daily:
        notes.append("使用的是月粒度日历，只有汇总工作日数，没有逐日节假日明细。")
    if spring_days is None and (lo.month <= 2 or hi.month <= 2):
        notes.append("区间覆盖 1—2 月但未识别到春节，请确认日历数据是否覆盖该年份。")

    return CalendarContext(
        region=ctx.region.info.name,
        start=str(lo.date()),
        end=str(hi.date()),
        n_days=n_days,
        n_workdays=n_workdays,
        n_nonworkdays=n_nonwork,
        holiday_break_days=break_days,
        makeup_workdays=makeup,
        holidays=holidays[:12],
        spring_festival_days=spring_days,
        spring_festival_note=spring_note,
        notes=notes,
    )


class WeatherArgs(BaseModel):
    start: str = Field(description="起始期次，如 '2025-01'。")
    end: str = Field(description="结束期次，如 '2025-03'。")


@tool(
    name="get_weather_context",
    description=(
        "查询指定区间的天气：月均气温、采暖度日(HDD)、制冷度日(CDD)、高温/低温天数。\n"
        "**这是排除天气负荷的关键工具。** 夏季高温推高空调负荷、冬季低温推高采暖负荷，"
        "都会让用电量上升而与经济无关。凡是用电异常发生在夏冬两季，必须先看这里。\n"
        "返回里 is_proxy=true 表示这是单点气象站代理变量，不是全省加权平均 —— "
        "这个限制必须写进结论。\n"
        "如果 available=false，说明该地区没接天气数据，此时无法排除天气因素。"
    ),
    args_model=WeatherArgs,
    returns="WeatherContext：月度气温/度日/极端天数 + 代理变量声明",
    tags=("context", "weather"),
)
def get_weather_context(ctx: ToolContext, args: WeatherArgs) -> BaseModel:
    ds_id = next(
        (k for k in ctx.region.datasets if "weather" in k.lower()),
        None,
    )
    if ds_id is None:
        return WeatherContext(
            region=ctx.region.info.name,
            start=args.start,
            end=args.end,
            available=False,
            notes=["该地区没有天气数据集，无法排除气温对负荷的影响，结论中必须声明这一缺失。"],
        )

    df = ctx.region.frame(ds_id)
    date_col = _pick(df, DATE_HINTS) or next(
        (c for c in df.columns if str(c).strip().lower() in ("month", "period")), None
    )
    if date_col is None:
        return WeatherContext(
            region=ctx.region.info.name,
            start=args.start,
            end=args.end,
            available=False,
            notes=[f"天气数据集 {ds_id} 缺少可识别的期次列，实际列：{list(df.columns)}"],
        )

    lo, hi = parse_period(args.start), parse_period(args.end)
    if lo is None or hi is None:
        raise ValueError("start / end 无法解析，请使用 '2025-01' 这类格式。")
    parsed = df[date_col].map(parse_period)
    sub = df[parsed.notna() & (parsed >= lo) & (parsed <= hi + pd.DateOffset(days=31))].copy()
    sub["_period"] = parsed[parsed.notna() & (parsed >= lo) & (parsed <= hi + pd.DateOffset(days=31))]

    temp_col = _pick(sub, TEMP_HINTS)
    hdd_col = _pick(sub, HDD_HINTS)
    cdd_col = _pick(sub, CDD_HINTS)
    hot_col = _pick(sub, HOT_HINTS)
    cold_col = _pick(sub, COLD_HINTS)

    def _val(row: pd.Series, col: str | None) -> float | None:
        if col is None:
            return None
        v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        return None if pd.isna(v) else float(v)

    monthly = []
    for _, row in sub.sort_values("_period").iterrows():
        monthly.append(
            {
                "period": str(row["_period"].date())[:7],
                "temp_mean_c": _val(row, temp_col),
                "heating_degree_days": _val(row, hdd_col),
                "cooling_degree_days": _val(row, cdd_col),
                "hot_days_ge_35c": _val(row, hot_col),
                "cold_days_le_0c": _val(row, cold_col),
            }
        )

    proxy_note = None
    for caveat in ctx.region.info.caveats:
        if "代理" in caveat or "再分析" in caveat:
            proxy_note = caveat
            break
    if proxy_note is None:
        proxy_note = "天气为单点气象站代理变量，不代表该地区全域平均。"

    notes = []
    if temp_col is None:
        notes.append("未识别到气温列，返回可能不完整。")
    if not monthly:
        notes.append("指定区间内没有天气记录，可能是数据覆盖范围之外。")

    return WeatherContext(
        region=ctx.region.info.name,
        start=str(lo.date()),
        end=str(hi.date()),
        available=bool(monthly),
        source=ds_id,
        is_proxy=True,
        proxy_note=proxy_note,
        monthly=monthly,
        notes=notes,
    )
