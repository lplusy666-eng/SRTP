#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""浙江电力经济数据 STL 分解与异常候选生成。

输入：包含以下文件的 ZIP（文件名可带额外前缀/目录）：
  - 浙江全社会用电量信息（月度）.xlsx
  - 浙江省第一产业季度.xlsx
  - 浙江省第二产业季度.xlsx
  - 浙江省第三产业季度.xlsx
  - 国务院办公厅近五年节假日安排.docx

输出：清洗数据、STL 分解结果、异常候选、模型可读 JSONL、图表和运行日志。

设计原则：
1. 不依赖 pandas/openpyxl，使用 Python 标准库解析 OOXML；
2. STL 只负责“拆分”和“异常候选”，不把残差直接等同于因果；
3. 对移动春节、疫情低基数、GDP 年度核算调整等建立显式规则。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import statsmodels
from statsmodels.tsa.seasonal import STL


NS_XLSX = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_WORD = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# 政府公布的春节放假区间；脚本还会从随附 DOCX 抽取原文并做存在性校验。
SPRING_FESTIVAL_RANGES: dict[int, tuple[date, date]] = {
    2021: (date(2021, 2, 11), date(2021, 2, 17)),
    2022: (date(2022, 1, 31), date(2022, 2, 6)),
    2023: (date(2023, 1, 21), date(2023, 1, 27)),
    2024: (date(2024, 2, 10), date(2024, 2, 17)),
    2025: (date(2025, 1, 28), date(2025, 2, 4)),
}

ELECTRICITY_STL = {
    "period": 12,
    "seasonal": 13,
    "trend": 21,
    "low_pass": 13,
    "robust": True,
}
GDP_GROWTH_STL = {
    "period": 4,
    "seasonal": 7,
    "trend": 9,
    "low_pass": 5,
    "robust": True,
}
GDP_FLOW_STL = GDP_GROWTH_STL.copy()


@dataclass(frozen=True)
class ElectricityPoint:
    month: date
    value_10k_kwh: float

    @property
    def value_100m_kwh(self) -> float:
        # 1 亿千瓦时 = 10,000 万千瓦时
        return self.value_10k_kwh / 10_000.0


@dataclass(frozen=True)
class GDPPoint:
    sector: str
    year: int
    quarter: int
    cumulative_value_100m_yuan: float
    cumulative_yoy_pct: float
    single_quarter_value_100m_yuan: float

    @property
    def period_label(self) -> str:
        return f"{self.year}Q{self.quarter}"

    @property
    def period_date(self) -> date:
        month = self.quarter * 3
        return date(self.year, month, 1)


def col_letters_to_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        raise ValueError(f"无法解析单元格地址：{cell_ref}")
    value = 0
    for ch in letters.group(0):
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def read_xlsx_first_sheet(path: Path) -> list[list[Any]]:
    """读取简单 XLSX 第一张表，支持 shared string、inline string 和数值。"""
    with ZipFile(path) as zf:
        names = set(zf.namelist())
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("x:si", NS_XLSX):
                text = "".join((t.text or "") for t in si.findall(".//x:t", NS_XLSX))
                shared_strings.append(text)

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in names:
            candidates = sorted(n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
            if not candidates:
                raise ValueError(f"{path.name} 中未找到工作表 XML")
            sheet_name = candidates[0]

        root = ET.fromstring(zf.read(sheet_name))
        rows: list[list[Any]] = []
        for row in root.findall(".//x:sheetData/x:row", NS_XLSX):
            values: dict[int, Any] = {}
            max_col = -1
            for cell in row.findall("x:c", NS_XLSX):
                ref = cell.attrib.get("r", "A1")
                col_idx = col_letters_to_index(ref)
                max_col = max(max_col, col_idx)
                cell_type = cell.attrib.get("t")
                raw = cell.find("x:v", NS_XLSX)
                if cell_type == "inlineStr":
                    value: Any = "".join((t.text or "") for t in cell.findall(".//x:t", NS_XLSX))
                elif raw is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(raw.text or 0)]
                elif cell_type == "b":
                    value = (raw.text == "1")
                else:
                    txt = raw.text or ""
                    try:
                        number = float(txt)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = txt
                values[col_idx] = value
            rows.append([values.get(i, "") for i in range(max_col + 1)] if max_col >= 0 else [])
    return rows


def extract_docx_text(path: Path) -> str:
    with ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    paragraphs: list[str] = []
    for p in root.findall(".//w:p", NS_WORD):
        text = "".join((t.text or "") for t in p.findall(".//w:t", NS_WORD)).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def find_required_file(root: Path, keyword: str, suffix: str) -> Path:
    matches = [p for p in root.rglob(f"*{suffix}") if keyword in p.name]
    if not matches:
        raise FileNotFoundError(f"未找到包含“{keyword}”的 {suffix} 文件")
    if len(matches) > 1:
        matches.sort(key=lambda p: len(str(p)))
    return matches[0]


def parse_month(value: Any) -> date:
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
        try:
            dt = datetime.strptime(text, fmt)
            return date(dt.year, dt.month, 1)
        except ValueError:
            continue
    raise ValueError(f"无法解析月份：{value!r}")


def parse_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"--", "—", "-"}:
        return math.nan
    return float(text)


def read_electricity(path: Path) -> list[ElectricityPoint]:
    rows = read_xlsx_first_sheet(path)
    if not rows:
        raise ValueError("月度用电量文件为空")
    header = [str(x).strip() for x in rows[0]]
    try:
        date_i = header.index("统计年月")
        name_i = header.index("指标名称")
        value_i = header.index("用电量")
    except ValueError as exc:
        raise ValueError(f"月度用电量列名不符合预期：{header}") from exc

    points: list[ElectricityPoint] = []
    for row in rows[1:]:
        if len(row) <= max(date_i, name_i, value_i):
            continue
        indicator = str(row[name_i]).strip()
        if indicator != "全社会用电量":
            continue
        month = parse_month(row[date_i])
        value = parse_float(row[value_i])
        if math.isfinite(value):
            points.append(ElectricityPoint(month, value))
    points.sort(key=lambda p: p.month)
    if len({p.month for p in points}) != len(points):
        raise ValueError("月度用电量存在重复月份")
    if len(points) < 24:
        raise ValueError(f"月度 STL 至少建议有 24 个月，当前仅 {len(points)} 个月")
    return points


def parse_quarter_header(text: str) -> tuple[int, int] | None:
    m = re.search(r"(\d{4})年([1-4])季度", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def read_gdp_sector(path: Path, sector: str) -> list[GDPPoint]:
    rows = read_xlsx_first_sheet(path)
    if len(rows) < 3:
        raise ValueError(f"{path.name} 行数不足")
    headers = [str(x).strip() for x in rows[0]]
    value_row = rows[1]
    growth_row = rows[2]
    # 原表是累计值，需按同一年相邻季度做差得到单季值。
    cumulative: list[tuple[int, int, float, float]] = []
    for i in range(1, len(headers)):
        parsed = parse_quarter_header(headers[i])
        if parsed is None:
            continue
        year, quarter = parsed
        if i >= len(value_row) or i >= len(growth_row):
            continue
        value = parse_float(value_row[i])
        growth = parse_float(growth_row[i])
        if math.isfinite(value) and math.isfinite(growth):
            cumulative.append((year, quarter, value, growth))
    cumulative.sort(key=lambda x: (x[0], x[1]))

    by_year: dict[int, dict[int, float]] = defaultdict(dict)
    growth_by_period: dict[tuple[int, int], float] = {}
    for year, quarter, value, growth in cumulative:
        by_year[year][quarter] = value
        growth_by_period[(year, quarter)] = growth

    points: list[GDPPoint] = []
    for year in sorted(by_year):
        previous_cumulative = 0.0
        for quarter in range(1, 5):
            if quarter not in by_year[year]:
                continue
            cumulative_value = by_year[year][quarter]
            single = cumulative_value - previous_cumulative
            previous_cumulative = cumulative_value
            points.append(
                GDPPoint(
                    sector=sector,
                    year=year,
                    quarter=quarter,
                    cumulative_value_100m_yuan=cumulative_value,
                    cumulative_yoy_pct=growth_by_period[(year, quarter)],
                    single_quarter_value_100m_yuan=single,
                )
            )
    return points


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_holiday_month_features(start_month: date, end_month: date) -> list[dict[str, Any]]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for start, end in SPRING_FESTIVAL_RANGES.values():
        for day in daterange(start, end):
            counts[(day.year, day.month)] += 1
    rows: list[dict[str, Any]] = []
    current = start_month
    while current <= end_month:
        rows.append(
            {
                "month": current.isoformat(),
                "spring_festival_holiday_days": counts.get((current.year, current.month), 0),
                "spring_festival_window": int(current.month in (1, 2)),
            }
        )
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return rows


def validate_holiday_text(text: str) -> list[str]:
    checks = {
        2021: ["2月11日至17日", "共7天"],
        2022: ["1月31日至2月6日", "共7天"],
        2023: ["1月21日至27日", "共7天"],
        2024: ["2月10日至17日", "共8天"],
        2025: ["1月28日", "2月4日", "共8天"],
    }
    messages: list[str] = []
    for year, tokens in checks.items():
        ok = all(token in text for token in tokens)
        messages.append(f"{year} 春节区间校验：{'通过' if ok else '未通过'}")
    return messages


def stl_strength(trend: np.ndarray, seasonal: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    def safe_strength(component: np.ndarray) -> float:
        denom = float(np.var(component + resid, ddof=1))
        if denom <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(np.var(resid, ddof=1)) / denom))
    return safe_strength(trend), safe_strength(seasonal)


def month_diff(a: date, b: date) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)


def electricity_severity(residual_pct: float) -> str:
    x = abs(residual_pct)
    if x >= 10.0:
        return "strong"
    if x >= 6.0:
        return "watch"
    return "normal"


def gdp_growth_severity(residual_pp: float) -> str:
    x = abs(residual_pp)
    if x >= 5.0:
        return "strong"
    if x >= 3.0:
        return "watch"
    return "normal"


def flow_severity(residual_pct: float) -> str:
    x = abs(residual_pct)
    if x >= 10.0:
        return "strong"
    if x >= 5.0:
        return "watch"
    return "normal"


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_chinese_font() -> None:
    # Matplotlib 在部分容器中不会自动索引 TTC 字体，因此显式注册。
    regular = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    bold = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    for font_path in (regular, bold):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "AR PL UMing CN",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_line_plot(
    path: Path,
    dates: Sequence[date],
    series: Sequence[tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    zero_line: bool = False,
    note: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.2, 5.4), dpi=160)
    ax = fig.add_subplot(111)
    for label, values in series:
        ax.plot(dates, values, linewidth=1.8, label=label)
    if zero_line:
        ax.axhline(0.0, linewidth=0.9)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if len(series) > 1:
        ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    if note:
        fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.4, 5.2), dpi=160)
    ax = fig.add_subplot(111)
    bars = ax.bar(labels, values)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.8 if value >= 0 else -0.8
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset, f"{value:.1f}", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def analyze_electricity(points: list[ElectricityPoint], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dates = [p.month for p in points]
    values = np.array([p.value_100m_kwh for p in points], dtype=float)
    model = STL(values, **ELECTRICITY_STL)
    result = model.fit()
    trend = np.asarray(result.trend, dtype=float)
    seasonal = np.asarray(result.seasonal, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    expected = trend + seasonal
    residual_pct = np.divide(resid, trend, out=np.zeros_like(resid), where=np.abs(trend) > 1e-12) * 100.0
    trend_strength, seasonal_strength = stl_strength(trend, seasonal, resid)

    holiday_features = {
        row["month"]: row for row in build_holiday_month_features(dates[0], dates[-1])
    }
    rows: list[dict[str, Any]] = []
    for i, dt in enumerate(dates):
        severity = electricity_severity(float(residual_pct[i]))
        edge = int(i < 6 or i >= len(dates) - 6)
        holiday = holiday_features[dt.isoformat()]
        rows.append(
            {
                "region": "浙江省",
                "date": dt.isoformat(),
                "indicator": "全社会用电量",
                "unit": "亿千瓦时",
                "observed": round(float(values[i]), 6),
                "trend": round(float(trend[i]), 6),
                "seasonal": round(float(seasonal[i]), 6),
                "expected": round(float(expected[i]), 6),
                "residual": round(float(resid[i]), 6),
                "residual_pct_of_trend": round(float(residual_pct[i]), 6),
                "severity": severity,
                "edge_flag": edge,
                "spring_festival_holiday_days": holiday["spring_festival_holiday_days"],
                "spring_festival_window": holiday["spring_festival_window"],
            }
        )

    monthly_seasonal: dict[int, list[float]] = defaultdict(list)
    for dt, value in zip(dates, seasonal):
        monthly_seasonal[dt.month].append(float(value))
    seasonal_profile = {str(m): mean(monthly_seasonal[m]) for m in range(1, 13)}

    annual_totals: dict[int, float] = defaultdict(float)
    annual_counts: dict[int, int] = defaultdict(int)
    for p in points:
        annual_totals[p.month.year] += p.value_100m_kwh
        annual_counts[p.month.year] += 1

    full_years = [year for year, count in annual_counts.items() if count == 12]
    summary = {
        "n_observations": len(points),
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "stl_parameters": ELECTRICITY_STL,
        "trend_start_100m_kwh": float(trend[0]),
        "trend_end_100m_kwh": float(trend[-1]),
        "trend_growth_pct": float((trend[-1] / trend[0] - 1.0) * 100.0),
        "trend_annualized_growth_pct": float(((trend[-1] / trend[0]) ** (12.0 / month_diff(dates[-1], dates[0])) - 1.0) * 100.0),
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "seasonal_profile_100m_kwh": seasonal_profile,
        "annual_totals_100m_kwh": {str(y): annual_totals[y] for y in sorted(annual_totals)},
        "full_years": full_years,
        "strong_anomaly_count": sum(r["severity"] == "strong" for r in rows),
        "watch_anomaly_count": sum(r["severity"] == "watch" for r in rows),
    }

    figures = out_dir / "figures"
    save_line_plot(
        figures / "01_浙江月度用电量_观测值与趋势.png",
        dates,
        [("观测值", values), ("STL 趋势", trend)],
        "浙江省月度全社会用电量：观测值与 STL 趋势",
        "亿千瓦时",
        note="注：边界区间的趋势和残差估计稳定性相对较低。",
    )
    save_line_plot(
        figures / "02_浙江月度用电量_季节项.png",
        dates,
        [("季节项", seasonal)],
        "浙江省月度全社会用电量：STL 季节项",
        "亿千瓦时",
        zero_line=True,
    )
    save_line_plot(
        figures / "03_浙江月度用电量_残差率.png",
        dates,
        [("残差/趋势", residual_pct)],
        "浙江省月度全社会用电量：异常候选残差率",
        "%",
        zero_line=True,
        note="阈值：|残差率|>=10% 为 strong，6%-10% 为 watch；残差仅是候选，不是因果证明。",
    )
    years = sorted(full_years)
    save_bar_plot(
        figures / "04_浙江年度用电量汇总.png",
        [str(y) for y in years],
        [annual_totals[y] for y in years],
        "由月度数据汇总的浙江省年度全社会用电量",
        "亿千瓦时",
    )
    save_bar_plot(
        figures / "05_浙江月度平均季节项.png",
        [f"{m}月" for m in range(1, 13)],
        [seasonal_profile[str(m)] for m in range(1, 13)],
        "浙江省月度全社会用电量：平均季节项",
        "亿千瓦时",
    )
    return rows, summary


def analyze_gdp(all_points: list[GDPPoint], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    growth_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "growth_model": GDP_GROWTH_STL,
        "flow_model": GDP_FLOW_STL,
        "analysis_start": "2010Q1",
        "sectors": {},
    }
    figures = out_dir / "figures"

    for sector in ("第一产业", "第二产业", "第三产业"):
        sector_points = [p for p in all_points if p.sector == sector and p.year >= 2010]
        sector_points.sort(key=lambda p: (p.year, p.quarter))
        if len(sector_points) < 16:
            raise ValueError(f"{sector} 有效季度数不足：{len(sector_points)}")
        dates = [p.period_date for p in sector_points]

        growth = np.array([p.cumulative_yoy_pct for p in sector_points], dtype=float)
        growth_result = STL(growth, **GDP_GROWTH_STL).fit()
        g_trend = np.asarray(growth_result.trend, dtype=float)
        g_seasonal = np.asarray(growth_result.seasonal, dtype=float)
        g_resid = np.asarray(growth_result.resid, dtype=float)
        g_expected = g_trend + g_seasonal
        g_trend_strength, g_seasonal_strength = stl_strength(g_trend, g_seasonal, g_resid)

        for p, tr, se, re, ex in zip(sector_points, g_trend, g_seasonal, g_resid, g_expected):
            growth_rows.append(
                {
                    "region": "浙江省",
                    "sector": sector,
                    "period": p.period_label,
                    "date": p.period_date.isoformat(),
                    "indicator": "累计增加值同比增速",
                    "unit": "百分点",
                    "observed": round(p.cumulative_yoy_pct, 6),
                    "trend": round(float(tr), 6),
                    "seasonal": round(float(se), 6),
                    "expected": round(float(ex), 6),
                    "residual_pp": round(float(re), 6),
                    "severity": gdp_growth_severity(float(re)),
                    "pandemic_period": int(p.year == 2020 and p.quarter == 1),
                    "low_base_recovery_period": int(p.year == 2021 and p.quarter in (1, 2, 3)),
                }
            )

        flow = np.array([p.single_quarter_value_100m_yuan for p in sector_points], dtype=float)
        if np.any(flow <= 0):
            bad = [p.period_label for p, v in zip(sector_points, flow) if v <= 0]
            raise ValueError(f"{sector} 2010 年以后出现非正单季增加值：{bad}")
        log_flow = np.log(flow)
        flow_result = STL(log_flow, **GDP_FLOW_STL).fit()
        f_trend = np.asarray(flow_result.trend, dtype=float)
        f_seasonal = np.asarray(flow_result.seasonal, dtype=float)
        f_resid = np.asarray(flow_result.resid, dtype=float)
        f_expected = np.exp(f_trend + f_seasonal)
        f_resid_pct = (np.exp(f_resid) - 1.0) * 100.0
        f_trend_strength, f_seasonal_strength = stl_strength(f_trend, f_seasonal, f_resid)

        for p, tr, se, re, ex, rp in zip(sector_points, f_trend, f_seasonal, f_resid, f_expected, f_resid_pct):
            flow_rows.append(
                {
                    "region": "浙江省",
                    "sector": sector,
                    "period": p.period_label,
                    "date": p.period_date.isoformat(),
                    "indicator": "单季增加值",
                    "unit": "亿元",
                    "observed": round(p.single_quarter_value_100m_yuan, 6),
                    "log_trend": round(float(tr), 6),
                    "log_seasonal": round(float(se), 6),
                    "expected": round(float(ex), 6),
                    "log_residual": round(float(re), 6),
                    "residual_pct": round(float(rp), 6),
                    "severity": flow_severity(float(rp)),
                    "annual_reconciliation_risk": int(p.quarter == 4),
                }
            )

        summary["sectors"][sector] = {
            "n_quarters": len(sector_points),
            "growth_trend_strength": g_trend_strength,
            "growth_seasonal_strength": g_seasonal_strength,
            "flow_trend_strength_log_scale": f_trend_strength,
            "flow_seasonal_strength_log_scale": f_seasonal_strength,
            "growth_strong_anomaly_count": sum(
                r["sector"] == sector and r["severity"] == "strong" for r in growth_rows
            ),
            "growth_watch_anomaly_count": sum(
                r["sector"] == sector and r["severity"] == "watch" for r in growth_rows
            ),
        }

        save_line_plot(
            figures / f"GDP_{sector}_同比增速_观测值与趋势.png",
            dates,
            [("观测同比增速", growth), ("STL 趋势", g_trend)],
            f"浙江省{sector}累计增加值同比增速：观测值与 STL 趋势",
            "%",
            zero_line=True,
        )
        save_line_plot(
            figures / f"GDP_{sector}_同比增速_残差.png",
            dates,
            [("残差", g_resid)],
            f"浙江省{sector}累计同比增速：STL 残差",
            "百分点",
            zero_line=True,
            note="阈值：|残差|>=5 个百分点为 strong，3-5 个百分点为 watch。",
        )
        save_line_plot(
            figures / f"GDP_{sector}_单季增加值_残差率.png",
            dates,
            [("对数 STL 残差率", f_resid_pct)],
            f"浙江省{sector}单季增加值：对数 STL 残差率",
            "%",
            zero_line=True,
            note="Q4 易混入年度核算、修订和累计值差分误差，须标记 annual_reconciliation_risk。",
        )

    return growth_rows, flow_rows, summary


def electricity_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["severity"] != "normal"]


def gdp_growth_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["severity"] != "normal"]


def gdp_flow_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["severity"] != "normal"]


def build_candidate_causes(record_type: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    causes: list[dict[str, Any]] = []
    if record_type == "electricity":
        month = datetime.strptime(row["date"], "%Y-%m-%d").date()
        if row["spring_festival_window"]:
            causes.append(
                {
                    "entity": "春节及调休",
                    "relation": "calendar_effect",
                    "confidence": "high" if row["spring_festival_holiday_days"] > 0 else "medium",
                    "evidence": f"当月春节假期天数={row['spring_festival_holiday_days']}；春节在1-2月跨月移动",
                }
            )
        if month.month in (7, 8, 9):
            causes.append(
                {
                    "entity": "气温与制冷负荷",
                    "relation": "weather_effect",
                    "confidence": "unknown",
                    "evidence_required": ["月均温", "高温日数", "制冷度日CDD"],
                }
            )
        causes.append(
            {
                "entity": "工业生产与行业结构",
                "relation": "economic_activity_effect",
                "confidence": "unknown",
                "evidence_required": ["规上工业增加值", "制造业分行业用电", "重点企业开工率"],
            }
        )
    elif record_type == "gdp_growth":
        period = row["period"]
        if period == "2020Q1":
            causes.append(
                {
                    "entity": "新冠疫情冲击",
                    "relation": "external_shock",
                    "confidence": "high",
                    "evidence": "2020Q1 与疫情初期停工停产时段重合",
                }
            )
        if period in {"2021Q1", "2021Q2", "2021Q3"}:
            causes.append(
                {
                    "entity": "低基数与复苏",
                    "relation": "base_effect_and_recovery",
                    "confidence": "high",
                    "evidence": "同比口径受到2020年同期低基数显著影响",
                }
            )
        causes.append(
            {
                "entity": "产业景气与政策",
                "relation": "sector_cycle_or_policy",
                "confidence": "unknown",
                "evidence_required": ["行业增加值", "投资", "出口", "政策事件"],
            }
        )
    elif record_type == "gdp_flow":
        if row["annual_reconciliation_risk"]:
            causes.append(
                {
                    "entity": "年度核算与数据修订",
                    "relation": "statistical_reconciliation",
                    "confidence": "high",
                    "evidence": "单季值由累计值做差得到，Q4承接全年核算差额",
                }
            )
        causes.append(
            {
                "entity": "真实经济波动",
                "relation": "economic_activity_effect",
                "confidence": "unknown",
                "evidence_required": ["季度核算说明", "行业月度数据", "政策与外部冲击"],
            }
        )
    return causes


def build_machine_records(
    electricity_rows: list[dict[str, Any]],
    growth_rows: list[dict[str, Any]],
    flow_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record_type, rows in (
        ("electricity", electricity_candidates(electricity_rows)),
        ("gdp_growth", gdp_growth_candidates(growth_rows)),
        ("gdp_flow", gdp_flow_candidates(flow_rows)),
    ):
        for row in rows:
            key_time = row.get("period") or row.get("date")
            residual_value = row.get("residual_pct_of_trend", row.get("residual_pp", row.get("residual_pct")))
            residual_unit = "% of trend" if record_type == "electricity" else ("percentage point" if record_type == "gdp_growth" else "%")
            records.append(
                {
                    "node_type": "AnomalyCandidate",
                    "record_type": record_type,
                    "region": row["region"],
                    "time": key_time,
                    "indicator": row["indicator"],
                    "sector": row.get("sector"),
                    "observed": row["observed"],
                    "expected": row["expected"],
                    "residual": residual_value,
                    "residual_unit": residual_unit,
                    "severity": row["severity"],
                    "quality_flags": {
                        "edge_flag": row.get("edge_flag", 0),
                        "annual_reconciliation_risk": row.get("annual_reconciliation_risk", 0),
                    },
                    "candidate_causes": build_candidate_causes(record_type, row),
                    "guardrails": [
                        "STL residual is an anomaly candidate, not causal proof",
                        "Require external evidence before producing a causal conclusion",
                        "Down-weight edge estimates and statistical-reconciliation periods",
                    ],
                }
            )
    records.sort(key=lambda x: (x["record_type"], str(x["time"]), str(x.get("sector"))))
    return records


def build_combined_anomaly_table(machine_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in machine_records:
        causes = "; ".join(
            f"{c['entity']}[{c['confidence']}]" for c in record["candidate_causes"]
        )
        rows.append(
            {
                "record_type": record["record_type"],
                "region": record["region"],
                "time": record["time"],
                "indicator": record["indicator"],
                "sector": record.get("sector") or "",
                "observed": record["observed"],
                "expected": record["expected"],
                "residual": record["residual"],
                "residual_unit": record["residual_unit"],
                "severity": record["severity"],
                "edge_flag": record["quality_flags"]["edge_flag"],
                "annual_reconciliation_risk": record["quality_flags"]["annual_reconciliation_risk"],
                "candidate_causes": causes,
            }
        )
    return rows


def clean_gdp_rows(points: list[GDPPoint]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(points, key=lambda x: (x.sector, x.year, x.quarter)):
        rows.append(
            {
                "region": "浙江省",
                "sector": p.sector,
                "period": p.period_label,
                "date": p.period_date.isoformat(),
                "cumulative_value_100m_yuan": p.cumulative_value_100m_yuan,
                "cumulative_yoy_pct": p.cumulative_yoy_pct,
                "single_quarter_value_100m_yuan": p.single_quarter_value_100m_yuan,
                "analysis_eligible_from_2010": int(p.year >= 2010 and p.single_quarter_value_100m_yuan > 0),
            }
        )
    return rows


def run(zip_path: Path, out_dir: Path) -> dict[str, Any]:
    set_chinese_font()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="zhejiang_stl_") as temp_dir:
        temp_root = Path(temp_dir)
        with ZipFile(zip_path) as zf:
            zf.extractall(temp_root)

        electricity_path = find_required_file(temp_root, "全社会用电量", ".xlsx")
        gdp_paths = {
            "第一产业": find_required_file(temp_root, "第一产业季度", ".xlsx"),
            "第二产业": find_required_file(temp_root, "第二产业季度", ".xlsx"),
            "第三产业": find_required_file(temp_root, "第三产业季度", ".xlsx"),
        }
        holiday_path = find_required_file(temp_root, "节假日安排", ".docx")

        electricity_points = read_electricity(electricity_path)
        gdp_points: list[GDPPoint] = []
        for sector, path in gdp_paths.items():
            gdp_points.extend(read_gdp_sector(path, sector))
        holiday_text = extract_docx_text(holiday_path)
        holiday_checks = validate_holiday_text(holiday_text)

    # 清洗数据
    clean_electricity = [
        {
            "region": "浙江省",
            "date": p.month.isoformat(),
            "indicator": "全社会用电量",
            "raw_unit": "万千瓦时",
            "raw_value": p.value_10k_kwh,
            "analysis_unit": "亿千瓦时",
            "analysis_value": p.value_100m_kwh,
        }
        for p in electricity_points
    ]
    write_csv(out_dir / "data" / "clean_electricity_monthly.csv", clean_electricity)
    write_csv(out_dir / "data" / "clean_gdp_quarterly.csv", clean_gdp_rows(gdp_points))
    holiday_features = build_holiday_month_features(electricity_points[0].month, electricity_points[-1].month)
    write_csv(out_dir / "data" / "holiday_features.csv", holiday_features)
    (out_dir / "data" / "holiday_source_text.txt").write_text(holiday_text, encoding="utf-8")

    electricity_rows, electricity_summary = analyze_electricity(electricity_points, out_dir)
    growth_rows, flow_rows, gdp_summary = analyze_gdp(gdp_points, out_dir)
    write_csv(out_dir / "results" / "electricity_stl.csv", electricity_rows)
    write_csv(out_dir / "results" / "gdp_growth_stl.csv", growth_rows)
    write_csv(out_dir / "results" / "gdp_flow_log_stl.csv", flow_rows)

    machine_records = build_machine_records(electricity_rows, growth_rows, flow_rows)
    combined_rows = build_combined_anomaly_table(machine_records)
    write_csv(out_dir / "results" / "anomalies.csv", combined_rows)
    with (out_dir / "results" / "anomaly_records.jsonl").open("w", encoding="utf-8") as f:
        for record in machine_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    quality_issues: list[str] = []
    for sector in ("第一产业", "第二产业", "第三产业"):
        bad = [p.period_label for p in gdp_points if p.sector == sector and p.year < 2010 and p.single_quarter_value_100m_yuan <= 0]
        if bad:
            quality_issues.append(f"{sector} 2010年前累计值差分出现非正单季值：{', '.join(bad)}；主分析从2010Q1开始。")

    summary = {
        "input_zip": str(zip_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "holiday_document_validation": holiday_checks,
        "electricity": electricity_summary,
        "gdp": gdp_summary,
        "data_quality_issues": quality_issues,
        "anomaly_candidate_count": len(machine_records),
        "guardrail": "STL残差仅作为异常候选；最终原因必须由节假日、天气、行业、政策等外部证据复核。",
    }
    (out_dir / "results" / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 运行日志采用人类可读格式，便于答辩复核。
    e = electricity_summary
    log_lines = [
        "浙江电力经济数据 STL 分解运行日志",
        "=" * 44,
        f"输入：{zip_path}",
        f"输出：{out_dir}",
        f"Python {platform.python_version()} | numpy {np.__version__} | statsmodels {statsmodels.__version__} | matplotlib {matplotlib.__version__}",
        "",
        "[节假日文档校验]",
        *holiday_checks,
        "",
        "[月度用电量]",
        f"样本：{e['n_observations']} 个月，{e['start']} 至 {e['end']}",
        f"STL 参数：{json.dumps(ELECTRICITY_STL, ensure_ascii=False)}",
        f"趋势起点：{e['trend_start_100m_kwh']:.3f} 亿千瓦时",
        f"趋势终点：{e['trend_end_100m_kwh']:.3f} 亿千瓦时",
        f"趋势累计增长：{e['trend_growth_pct']:.2f}%",
        f"折算年化趋势增速：{e['trend_annualized_growth_pct']:.2f}%",
        f"趋势强度：{e['trend_strength']:.3f}",
        f"季节强度：{e['seasonal_strength']:.3f}",
        f"strong 候选：{e['strong_anomaly_count']} 个；watch 候选：{e['watch_anomaly_count']} 个",
        "",
        "月度平均季节项（亿千瓦时）：",
    ]
    for m in range(1, 13):
        log_lines.append(f"  {m:02d}月：{e['seasonal_profile_100m_kwh'][str(m)]:+.3f}")
    log_lines.extend(["", "年度汇总（亿千瓦时）："])
    for year, total in e["annual_totals_100m_kwh"].items():
        suffix = "（完整年度）" if int(year) in e["full_years"] else "（不完整年度）"
        log_lines.append(f"  {year}: {total:.3f} {suffix}")

    log_lines.extend(["", "[用电量异常候选]"])
    for row in electricity_candidates(electricity_rows):
        log_lines.append(
            f"  {row['date'][:7]} {row['severity']}: observed={row['observed']:.2f}, residual/trend={row['residual_pct_of_trend']:+.2f}%, "
            f"春节假期天数={row['spring_festival_holiday_days']}, edge={row['edge_flag']}"
        )

    log_lines.extend(["", "[GDP累计同比增速异常候选]"])
    for row in sorted(gdp_growth_candidates(growth_rows), key=lambda r: (r["sector"], -abs(r["residual_pp"]))):
        log_lines.append(
            f"  {row['sector']} {row['period']} {row['severity']}: observed={row['observed']:.2f}%, residual={row['residual_pp']:+.2f}个百分点"
        )

    log_lines.extend(["", "[数据质量与解释约束]"])
    log_lines.extend(f"  - {x}" for x in quality_issues)
    log_lines.extend(
        [
            "  - 1-2月移动春节不能由固定12月季节项自动吸收，需显式加入节假日特征。",
            "  - 2020Q1/2021Q1-Q3的同比异常需优先解释为疫情冲击与低基数复苏。",
            "  - 累计GDP做差后的Q4异常需优先检查年度核算与数据修订。",
            "  - STL残差不是因果结论。",
        ]
    )
    (out_dir / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="浙江电力经济数据 STL 分解")
    parser.add_argument("--zip", dest="zip_path", required=True, type=Path, help="原始数据 ZIP 路径")
    parser.add_argument("--out", dest="out_dir", required=True, type=Path, help="输出目录")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.zip_path.exists():
        print(f"错误：输入文件不存在：{args.zip_path}", file=sys.stderr)
        return 2
    try:
        summary = run(args.zip_path.resolve(), args.out_dir.resolve())
    except Exception as exc:  # noqa: BLE001 - CLI需要明确失败信息
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
