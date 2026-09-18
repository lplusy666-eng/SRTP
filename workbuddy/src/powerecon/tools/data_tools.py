"""数据类工具：这个地区有什么、取一条序列、检查数据质量。

这三个是几乎所有问题的起点。LLM 的第一跳通常是 list_region_datasets，
第二跳是 data_quality_report（先确认数据能不能用），第三跳才进入分析。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..contract import (
    DataQualityReport,
    Frequency,
    RegionInventory,
    SeriesSummary,
)
from ..ingest import available_packs, build_inventory, load_region
from ..series_utils import expected_grid, mad, parse_label, period_label, shift_period
from .base import ToolContext, tool


class ListRegionsArgs(BaseModel):
    pass


class ListDatasetsArgs(BaseModel):
    pass


class LoadSeriesArgs(BaseModel):
    dataset_id: str = Field(description="数据集 id，来自 list_region_datasets 的返回。")
    column: str | None = Field(
        default=None,
        description="覆盖默认数值列。当自动嗅探选错了列时用它指定，例如 'consumption_month_yi_kwh'。",
    )
    start: str | None = Field(default=None, description="起始期次，如 '2024-01' 或 '2024Q1'。")
    end: str | None = Field(default=None, description="结束期次。")
    include_points: bool = Field(
        default=False,
        description="是否返回逐期明细。默认只给统计摘要；需要看具体数值走势时再设为 true。",
    )


class QualityReportArgs(BaseModel):
    dataset_id: str = Field(description="数据集 id。")
    column: str | None = Field(default=None, description="覆盖默认数值列。")


@tool(
    name="list_available_regions",
    description=(
        "列出本机已装载的所有地区数据包（region pack）。"
        "当用户提到的地区不清楚、或需要确认能分析哪些地区时先调这个。"
        "注意：一次运行只能分析一个地区，切换地区需要重新启动并指定 --region。"
    ),
    args_model=ListRegionsArgs,
    returns="可用地区的名称与代码列表",
    needs_region=False,
    tags=("data",),
)
def list_available_regions(ctx: ToolContext, args: ListRegionsArgs) -> BaseModel:
    packs = available_packs(ctx.packs_root)
    rows = []
    for name in packs:
        try:
            r = load_region(ctx.packs_root / name)
            rows.append({"pack": name, "code": r.info.code, "name": r.info.name, "level": r.info.level})
        except Exception as exc:
            rows.append({"pack": name, "error": str(exc)})

    class _Out(BaseModel):
        current_region: str
        current_pack: str
        n_regions: int
        regions: list[dict[str, Any]]

    return _Out(
        current_region=ctx.region.info.name,
        current_pack=ctx.region.pack_dir.name,
        n_regions=len(rows),
        regions=rows,
    )


@tool(
    name="list_region_datasets",
    description=(
        "列出当前地区所有可用的数据集、指标名、单位、频率、覆盖期次，以及数据边界声明（caveats）。"
        "这是分析的起点：先看清有什么数据、数据有什么已知缺陷，再决定怎么分析。"
        "返回里的 caveats 必须被遵守，例如'天气是单点代理变量''季度GDP是累计值'。"
    ),
    args_model=ListDatasetsArgs,
    returns="RegionInventory：地区信息 + 数据集清单 + 数据边界声明",
    tags=("data",),
)
def list_region_datasets(ctx: ToolContext, args: ListDatasetsArgs) -> BaseModel:
    cached = ctx.cache_get("inventory")
    if cached is not None:
        return cached
    return ctx.cache_set("inventory", build_inventory(ctx.region))


@tool(
    name="load_series",
    description=(
        "取出一条时间序列并返回统计摘要（最新值、同比、环比、均值、标准差）。"
        "适合回答'某指标现在什么水平''同比变化多少'这类问题。"
        "默认只返回摘要以节省上下文；需要逐期明细时把 include_points 设为 true。"
        "如果返回的 metric 看起来不对（嗅探选错了列），用 column 参数显式指定。"
    ),
    args_model=LoadSeriesArgs,
    returns="SeriesSummary：序列摘要 + 可选逐期明细",
    tags=("data",),
)
def load_series(ctx: ToolContext, args: LoadSeriesArgs) -> BaseModel:
    info = ctx.region.datasets.get(args.dataset_id)
    if info is None:
        raise KeyError(f"没有数据集 {args.dataset_id}。可用：{sorted(ctx.region.datasets)}")

    df = ctx.region.series(args.dataset_id, column=args.column)
    if args.start:
        lo = parse_label(args.start)
        if lo is not None:
            df = df[df["period"] >= lo]
    if args.end:
        hi = parse_label(args.end)
        if hi is not None:
            df = df[df["period"] <= hi]
    if df.empty:
        raise ValueError(f"{args.dataset_id} 在指定区间内没有数据。")

    df = df.sort_values("period").reset_index(drop=True)
    values = df["value"].to_numpy(dtype=float)
    periods = list(df["period"])
    freq = info.frequency
    latest_period, latest_value = periods[-1], float(values[-1])

    yoy = mom = None
    if len(periods) >= 2:
        prev = values[-2]
        if abs(prev) > 1e-12:
            mom = float((values[-1] - prev) / abs(prev))
    target = shift_period(latest_period, freq, -1)
    match = next((i for i, p in enumerate(periods) if p == target), None)
    if match is None:
        # 期次不严格对齐时（缺期、OCR 数据）退化为找最近的一期
        deltas = [abs((p - target).days) for p in periods]
        if deltas:
            idx = int(np.argmin(deltas))
            tolerance = 20 if freq is Frequency.MONTHLY else 400
            if deltas[idx] <= tolerance and idx != len(periods) - 1:
                match = idx
    if match is not None and abs(values[match]) > 1e-12:
        yoy = float((values[-1] - values[match]) / abs(values[match]))

    points: list[dict[str, Any]] = []
    if args.include_points:
        points = [
            {"period": period_label(p, freq), "value": float(v)}
            for p, v in zip(periods, values)
        ]
    else:
        tail = min(13, len(periods))
        points = [
            {"period": period_label(p, freq), "value": float(v)}
            for p, v in zip(periods[-tail:], values[-tail:])
        ]

    return SeriesSummary(
        dataset_id=args.dataset_id,
        metric=info.metric if args.column is None else args.column,
        unit=info.unit,
        frequency=freq,
        n_points=int(len(periods)),
        period_start=period_label(periods[0], freq),
        period_end=period_label(periods[-1], freq),
        latest_period=period_label(latest_period, freq),
        latest_value=latest_value,
        yoy_pct=yoy,
        mom_pct=mom,
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        series=points,
    )


@tool(
    name="data_quality_report",
    description=(
        "检查一条序列的数据质量：缺期、重复期、间隔异常、突变跳点，以及端点/样本量风险。"
        "在给出任何结论之前都应该先跑一次 —— 如果数据本身有缺口或单位是推断的，"
        "结论的置信度必须相应下调。返回的 flags 要原样带进最终结论的不确定性说明。"
    ),
    args_model=QualityReportArgs,
    returns="DataQualityReport：缺失/重复/间隔/突变 + 风险标记",
    tags=("data", "quality"),
)
def data_quality_report(ctx: ToolContext, args: QualityReportArgs) -> BaseModel:
    info = ctx.region.datasets.get(args.dataset_id)
    if info is None:
        raise KeyError(f"没有数据集 {args.dataset_id}。可用：{sorted(ctx.region.datasets)}")

    df = ctx.region.series(args.dataset_id, column=args.column).sort_values("period").reset_index(drop=True)
    freq = info.frequency
    periods = list(df["period"])
    values = df["value"].to_numpy(dtype=float)

    notes: list[str] = []
    flags: list[str] = list(info.quality_flags)

    dup_mask = df["period"].duplicated(keep=False)
    duplicates = [period_label(p, freq) for p in df.loc[dup_mask, "period"].unique()]

    missing: list[str] = []
    grid = expected_grid(periods[0], periods[-1], freq) if periods else []
    if grid:
        have = set(periods)
        missing = [period_label(p, freq) for p in grid if p not in have]
        if missing:
            notes.append(f"期望 {len(grid)} 期，实际 {len(periods)} 期，缺 {len(missing)} 期。")
    else:
        notes.append("频率未知，无法判定是否缺期。")

    gaps: list[dict[str, Any]] = []
    if len(periods) >= 3:
        deltas = np.array([(periods[i + 1] - periods[i]).days for i in range(len(periods) - 1)])
        typical = float(np.median(deltas))
        for i, d in enumerate(deltas):
            if typical > 0 and (d > typical * 2.5 or d < typical * 0.4):
                gaps.append(
                    {
                        "after": period_label(periods[i], freq),
                        "before": period_label(periods[i + 1], freq),
                        "gap_days": int(d),
                        "typical_gap_days": int(typical),
                    }
                )

    jumps: list[dict[str, Any]] = []
    if len(values) >= 5:
        diffs = np.diff(values)
        scale = mad(diffs)
        if scale > 1e-12:
            threshold = 8.0 * 1.4826 * scale
            for i, d in enumerate(diffs):
                if abs(d) > threshold:
                    jumps.append(
                        {
                            "period": period_label(periods[i + 1], freq),
                            "from": float(values[i]),
                            "to": float(values[i + 1]),
                            "change_pct": float(d / abs(values[i])) if abs(values[i]) > 1e-12 else None,
                        }
                    )

    n_missing = int(df["value"].isna().sum())
    cycles = len(periods) / 12 if freq is Frequency.MONTHLY else (
        len(periods) / 4 if freq is Frequency.QUARTERLY else 99
    )
    endpoint_warning = cycles < 2
    if endpoint_warning:
        flags.append("short-series")
        notes.append(f"仅约 {cycles:.1f} 个完整季节周期，STL 首尾估计不可靠，端点结论须标注 endpoint_warning。")
    if missing:
        flags.append("missing-periods")
    if duplicates:
        flags.append("duplicate-periods")
    if jumps:
        flags.append("abrupt-jumps")
    if info.discovered:
        notes.append("该数据集为自动发现，列名与单位系推断所得，关键结论前建议人工复核原始表。")

    return DataQualityReport(
        dataset_id=args.dataset_id,
        metric=info.metric if args.column is None else args.column,
        n_rows=int(len(df)),
        n_missing=n_missing,
        missing_periods=missing[:24],
        duplicate_periods=duplicates,
        irregular_gaps=gaps,
        abrupt_jumps=jumps,
        endpoint_warning=endpoint_warning,
        flags=sorted(set(flags)),
        notes=notes,
    )
