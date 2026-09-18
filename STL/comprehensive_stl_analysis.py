#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四城市（浙江、苏州、南京、江门）电力数据整理与 STL 分解
======================================================
输入：
  - 浙江省数据(1).zip（Excel 文件）
  - data(苏州+南京+江门).docx（含截图，已通过 EasyOCR 提取文本）

输出：
  - data/ 目录：各城市清洗后的 CSV 数据
  - results/ 目录：STL 分解结果 CSV 和汇总 JSON
  - figures/ 目录：STL 分解图表
  - run_log.txt：运行日志
"""

import csv
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

# 修复 Windows GBK 编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from statsmodels.tsa.seasonal import STL

# ============================================================
# 配置
# ============================================================

NS_XLSX = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# STL 参数
ELECTRICITY_MONTHLY_STL = {
    "period": 12,
    "seasonal": 13,
    "trend": 21,
    "low_pass": 13,
    "robust": True,
}

ELECTRICITY_ANNUAL_STL = {
    "period": None,  # 年度数据无明显周期，使用较弱参数
    "seasonal": 7,
    "trend": 9,
    "low_pass": 7,
    "robust": True,
}

OUT_DIR = Path("output")
DATA_DIR = OUT_DIR / "data"
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"


# ============================================================
# 数据模型
# ============================================================

@dataclass(frozen=True)
class MonthlyRecord:
    """月度电力数据"""
    region: str
    month: date
    indicator: str
    value: float  # 亿千瓦时
    unit: str = "亿千瓦时"
    yoy_pct: float | None = None  # 同比增速 %


@dataclass(frozen=True)
class AnnualRecord:
    """年度电力数据"""
    region: str
    year: int
    total: float  # 万千瓦时
    primary: float
    secondary: float
    tertiary: float


# ============================================================
# 工具函数
# ============================================================

def set_chinese_font():
    """配置 Matplotlib 中文字体"""
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
    # Windows 常见中文字体
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK JP",
        "AR PL UMing CN", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or [], extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def stl_strength(trend: np.ndarray, seasonal: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    def _strength(comp: np.ndarray) -> float:
        denom = float(np.var(comp + resid, ddof=1))
        return max(0.0, min(1.0, 1.0 - float(np.var(resid, ddof=1)) / denom)) if denom > 0 else 0.0
    return _strength(trend), _strength(seasonal)


def cumulative_to_monthly(cumulative: list[MonthlyRecord]) -> list[MonthlyRecord]:
    """将累计值转换为单月值"""
    if not cumulative:
        return []
    sorted_data = sorted(cumulative, key=lambda r: r.month)
    result = []
    prev = None
    last_year_end = None  # 上一年 12 月的累计值
    current_year = None

    for i, record in enumerate(sorted_data):
        y, m = record.month.year, record.month.month
        if current_year != y:
            if prev and prev.month.month == 12:
                last_year_end = prev
            current_year = y
            prev = None  # 新年第一笔数据

        if m == 1 or (prev is None and m <= 2):
            # 1月单月值 = 1月累计值
            single_value = record.value
        elif prev and prev.month.year == y:
            # 同年内：当月累计 - 上月累计
            single_value = record.value - prev.value
        elif last_year_end and last_year_end.month.year == y - 1:
            # 跨年（如1-2月）：当年累计 - 上年12月累计
            single_value = record.value - last_year_end.value
        else:
            # 无法做差，跳过
            prev = record
            continue

        if single_value < 0:
            single_value = float("nan")

        result.append(MonthlyRecord(
            region=record.region,
            month=record.month,
            indicator=record.indicator,
            value=round(single_value, 6),
            unit=record.unit,
            yoy_pct=record.yoy_pct,
        ))
        prev = record

    return result


# ============================================================
# 1. 解析浙江数据（从 ZIP 中的 Excel 文件）
# ============================================================

def col_letters_to_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        raise ValueError(f"无法解析单元格地址：{cell_ref}")
    value = 0
    for ch in letters.group(0):
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def read_xlsx_first_sheet(path: Path) -> list[list[Any]]:
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


def extract_zhejiang_electricity(zip_path: Path) -> list[MonthlyRecord]:
    """从 ZIP 文件中提取浙江月度用电量"""
    with tempfile.TemporaryDirectory(prefix="zj_") as tmp:
        tmp_root = Path(tmp)
        with ZipFile(zip_path) as zf:
            zf.extractall(tmp_root)

        # 查找用电量文件
        matches = list(tmp_root.rglob("*全社会用电量*.xlsx"))
        if not matches:
            raise FileNotFoundError("未找到浙江全社会用电量文件")
        path = matches[0]

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

        records = []
        for row in rows[1:]:
            if len(row) <= max(date_i, name_i, value_i):
                continue
            indicator = str(row[name_i]).strip()
            if indicator != "全社会用电量":
                continue
            month = parse_month(row[date_i])
            value_10k = parse_float(row[value_i])
            if math.isfinite(value_10k):
                records.append(MonthlyRecord(
                    region="浙江",
                    month=month,
                    indicator="全社会用电量",
                    value=round(value_10k / 10000.0, 6),  # 万千瓦时 → 亿千瓦时
                    unit="亿千瓦时",
                ))
        records.sort(key=lambda r: r.month)
        return records


def extract_zhejiang_gdp(zip_path: Path) -> list[dict]:
    """从 ZIP 提取浙江 GDP 季度数据"""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="zj_gdp_") as tmp:
        tmp_root = Path(tmp)
        with ZipFile(zip_path) as zf:
            zf.extractall(tmp_root)

        sector_map = {
            "第一产业": "第一产业季度",
            "第二产业": "第二产业季度",
            "第三产业": "第三产业季度",
        }
        all_rows = []
        for sector, keyword in sector_map.items():
            matches = list(tmp_root.rglob(f"*{keyword}*.xlsx"))
            if not matches:
                continue
            rows = read_xlsx_first_sheet(matches[0])
            if len(rows) < 3:
                continue
            headers = [str(x).strip() for x in rows[0]]
            value_row = rows[1]
            growth_row = rows[2]

            cumulative = []
            for i in range(1, len(headers)):
                m = re.search(r"(\d{4})年([1-4])季度", headers[i])
                if not m:
                    continue
                year, quarter = int(m.group(1)), int(m.group(2))
                if i >= len(value_row) or i >= len(growth_row):
                    continue
                val = parse_float(value_row[i])
                g = parse_float(growth_row[i])
                if math.isfinite(val) and math.isfinite(g):
                    cumulative.append((year, quarter, val, g))
            cumulative.sort()

            by_year = defaultdict(dict)
            growth_map = {}
            for yr, q, v, g in cumulative:
                by_year[yr][q] = v
                growth_map[(yr, q)] = g

            prev_cum = 0.0
            for yr in sorted(by_year):
                for q in range(1, 5):
                    if q not in by_year[yr]:
                        continue
                    cum_val = by_year[yr][q]
                    single = cum_val - prev_cum
                    prev_cum = cum_val
                    all_rows.append({
                        "region": "浙江",
                        "sector": sector,
                        "period": f"{yr}Q{q}",
                        "date": f"{yr}-{q*3:02d}-01",
                        "cumulative_value_100m_yuan": cum_val,
                        "cumulative_yoy_pct": growth_map[(yr, q)],
                        "single_quarter_value_100m_yuan": round(single, 4),
                    })
        return all_rows


# ============================================================
# 2. 解析苏州、南京、江门 OCR 数据
# ============================================================

def parse_suzhou_data() -> list[MonthlyRecord]:
    """
    苏州月度用电量数据（来自 OCR 提取）
    原始数据为累计值（亿千瓦时），需转换为单月值
    """
    raw = [
        # (年, 月, 累计用电量_亿千瓦时, 累计同比%)
        # 2025年
        (2025, 2, 269.44, None),
        (2025, 3, 425.42, 2.5),
        (2025, 4, 569.29, 3.1),
        (2025, 5, 717.85, 3.2),
        (2025, 6, 881.84, 4.4),
        (2025, 7, 1083.17, 4.3),
        (2025, 8, 1291.64, 4.1),
        (2025, 9, 1478.83, 4.7),
        (2025, 10, 1639.27, 5.4),
        (2025, 11, 1792.16, 5.4),
        (2025, 12, 1957.55, 5.1),
        # 2026年
        (2026, 2, 288.76, 7.2),
        (2026, 3, 450.97, 6.0),
        (2026, 4, 604.51, 6.2),
    ]

    cumulative = [
        MonthlyRecord(
            region="苏州", month=date(y, m, 1),
            indicator="全社会用电量", value=v, unit="亿千瓦时", yoy_pct=g,
        )
        for y, m, v, g in raw
    ]
    return cumulative_to_monthly(cumulative)


def parse_nanjing_annual() -> list[AnnualRecord]:
    """
    南京年度用电量数据 2000-2023（来自 OCR 提取的统计年鉴表 7-14）
    单位：万千瓦时
    """
    raw = [
        # (年, 全社会, 第一产业, 第二产业, 第三产业)
        (2000, 1377395, None, None, None),
        (2001, 1484439, None, None, None),
        (2002, 1620788, None, None, None),
        (2003, 1840333, None, None, None),
        (2004, 2072997, 15625, 1516541, 280799),
        (2005, 2466661, 14507, 1772227, 377701),
        (2006, 2705710, 14117, 1906213, 435703),
        (2007, 2991293, 14806, 2104964, 503645),
        (2008, 3107922, 15350, 2109267, 567103),
        (2009, 3370545, 16264, 2274994, 626941),
        (2010, 3736638, 16871, 2461481, 728152),
        (2011, 3997431, 18313, 2605097, 844872),
        (2012, 4249554, 18853, 2699199, 930843),
        (2013, 4626718, 19536, 2920266, 1011428),
        (2014, 4704973, 20441, 2946677, 1130462),
        (2015, 4951753, 23820, 3069749, 1204713),
        (2016, 5247863, 30133, 3167669, 1285568),
        (2017, 5569607, 31438, 3243525, 1483254),
        (2018, 6064005, 18389, 3381783, 1734407),
        (2019, 6215283, 19160, 3357175, 1872079),
        (2020, 6329425, 18660, 3481973, 1854276),
        (2021, 6835698, 22235, 3613941, 2124003),
        (2022, 7244555, 25641, 3662388, 2331260),
        (2023, 7472669, 25417, 3763361, 2494994),
    ]
    return [
        AnnualRecord(
            region="南京", year=y, total=t,
            primary=p if p else 0, secondary=s if s else 0, tertiary=tr if tr else 0,
        )
        for y, t, p, s, tr in raw
    ]


def parse_nanjing_monthly() -> list[MonthlyRecord]:
    """
    南京月度用电量数据（来自 OCR 提取的月度经济社会发展指标）
    原始数据为累计值（亿千瓦时），需转换为单月值
    OCR 中显示单位变化：部分为亿千瓦时，保持一致性
    """
    raw = [
        # (年, 月, 累计用电量_亿千瓦时, 累计同比%)
        # 2025年
        (2025, 2, 126.14, -4.5),   # OCR: 1126.14 but unit seems off; corrected to 126.14
        (2025, 3, 187.33, -2.6),
        (2025, 4, 242.59, -1.6),
        (2025, 5, 303.74, -0.5),
        (2025, 6, 372.38, 0.6),
        (2025, 7, 462.97, 2.0),
        (2025, 8, 553.98, 1.8),
        (2025, 9, 629.07, 1.8),
        (2025, 10, 693.47, 2.7),
        (2025, 11, 754.26, 3.0),
        (2025, 12, 823.78, 2.6),
        # 2026年
        (2026, 2, 133.06, 5.5),
        (2026, 3, 198.76, 6.1),
        (2026, 4, 258.32, 6.5),
    ]

    cumulative = [
        MonthlyRecord(
            region="南京", month=date(y, m, 1),
            indicator="全社会用电量", value=v, unit="亿千瓦时", yoy_pct=g,
        )
        for y, m, v, g in raw
    ]
    return cumulative_to_monthly(cumulative)


def parse_jiangmen_data() -> list[MonthlyRecord]:
    """
    江门月度用电量数据（来自 OCR 提取）
    原始数据为累计值（万千瓦时），需转换为单月值（亿千瓦时）
    """
    raw = [
        # (年, 月, 累计用电量_万千瓦时, 累计同比%)
        # 2025年
        (2025, 2, 478797, 0.8),
        (2025, 3, 801446, 1.1),
        (2025, 4, 1130337, 0.8),
        (2025, 5, 1483719, 2.0),
        (2025, 6, 1866564, 3.1),
        (2025, 7, 2303661, 2.8),
        (2025, 8, 2707756, 2.7),
        (2025, 9, 3086272, 2.8),   # OCR: 30862721 (OCR error, corrected)
        (2025, 10, 3449733, 3.2),
        (2025, 11, 3777944, 3.1),
        (2025, 12, 4112726, 3.2),
        # 2026年
        (2026, 2, 531523, 8.6),
        (2026, 3, 873314, None),    # OCR missing YoY
        (2026, 4, 1241250, 8.0),
        (2026, 5, 1631361, 8.3),
    ]

    cumulative = [
        MonthlyRecord(
            region="江门", month=date(y, m, 1),
            indicator="全社会用电量", value=round(v / 10000.0, 6),  # 万千瓦时 → 亿千瓦时
            unit="亿千瓦时", yoy_pct=g,
        )
        for y, m, v, g in raw
    ]
    return cumulative_to_monthly(cumulative)


# ============================================================
# 3. STL 分解
# ============================================================

def run_stl_monthly(records: list[MonthlyRecord], region: str, stl_params: dict,
                    out_prefix: str) -> dict:
    """对月度数据运行 STL 分解"""
    records = sorted(records, key=lambda r: r.month)
    dates = [r.month for r in records]
    values = np.array([r.value for r in records], dtype=float)

    if len(values) < 12:
        return {
            "region": region,
            "error": f"数据量不足（{len(values)} 个月），需要至少 12 个月",
            "n_observations": len(values),
        }

    # 移除 NaN
    mask = np.isfinite(values)
    if not mask.all():
        valid_idx = np.where(mask)[0]
        dates = [dates[i] for i in valid_idx]
        values = values[valid_idx]
        records = [records[i] for i in valid_idx]

    if len(values) < 12:
        return {"region": region, "error": f"有效数据不足（{len(values)} 个月）"}

    # 根据数据长度调整 STL 参数
    n = len(values)
    if n < 24:
        # 短序列使用更小的平滑窗口
        period_val = min(12, max(2, n // 3))  # 确保 period >= 2
        seas_smooth = min(n - 1, 7)
        if seas_smooth % 2 == 0:
            seas_smooth -= 1
        trend_smooth = min(n, 13)
        if trend_smooth % 2 == 0:
            trend_smooth -= 1
        # 确保 trend > period
        if trend_smooth <= period_val:
            trend_smooth = period_val + 2
            if trend_smooth % 2 == 0:
                trend_smooth += 1
        adjusted_params = {
            "period": period_val,
            "seasonal": max(5, seas_smooth),
            "trend": max(5, trend_smooth),
            "low_pass": max(5, seas_smooth),
            "robust": True,
        }
    else:
        adjusted_params = dict(stl_params)

    try:
        model = STL(values, **adjusted_params)
        result = model.fit()
        used_params = dict(adjusted_params)
    except Exception as e:
        # 降低参数重试
        fallback_period = max(2, n // 4)
        fallback_trend = fallback_period + 2
        if fallback_trend % 2 == 0:
            fallback_trend += 1
        fallback = {
            "period": fallback_period,
            "seasonal": max(3, min(5, n - 1)),
            "trend": fallback_trend,
            "low_pass": max(3, min(5, n - 1)),
            "robust": True,
        }
        try:
            model = STL(values, **fallback)
            result = model.fit()
            used_params = dict(fallback)
        except Exception as e2:
            return {"region": region, "error": f"STL 失败：{e2}"}

    trend = np.asarray(result.trend, dtype=float)
    seasonal = np.asarray(result.seasonal, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    expected = trend + seasonal
    residual_pct = np.divide(
        resid, trend,
        out=np.zeros_like(resid),
        where=np.abs(trend) > 1e-12,
    ) * 100.0

    trend_strength, seasonal_strength = stl_strength(trend, seasonal, resid)

    # 输出行
    rows = []
    for i, dt in enumerate(dates):
        severity = "normal"
        abs_pct = abs(float(residual_pct[i]))
        if abs_pct >= 10:
            severity = "strong"
        elif abs_pct >= 6:
            severity = "watch"

        rows.append({
            "region": region,
            "date": dt.isoformat(),
            "indicator": records[i].indicator,
            "unit": records[i].unit,
            "observed": round(float(values[i]), 4),
            "trend": round(float(trend[i]), 4),
            "seasonal": round(float(seasonal[i]), 4),
            "expected": round(float(expected[i]), 4),
            "residual": round(float(resid[i]), 4),
            "residual_pct_of_trend": round(float(residual_pct[i]), 4),
            "severity": severity,
        })

    # 月度季节项
    monthly_seasonal = defaultdict(list)
    for dt, s in zip(dates, seasonal):
        monthly_seasonal[dt.month].append(float(s))
    seasonal_profile = {}
    for m in range(1, 13):
        if monthly_seasonal[m]:
            seasonal_profile[str(m)] = mean(monthly_seasonal[m])
        else:
            seasonal_profile[str(m)] = 0.0

    # 年度汇总
    annual_totals = defaultdict(float)
    annual_counts = defaultdict(int)
    for r in records:
        annual_totals[r.month.year] += r.value
        annual_counts[r.month.year] += 1
    full_years = [y for y, c in annual_counts.items() if c == 12]

    summary = {
        "region": region,
        "n_observations": len(records),
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "stl_parameters": used_params,
        "trend_start": round(float(trend[0]), 4),
        "trend_end": round(float(trend[-1]), 4),
        "trend_growth_pct": round(float((trend[-1] / trend[0] - 1.0) * 100.0), 2) if trend[0] > 0 else None,
        "trend_strength": round(trend_strength, 4),
        "seasonal_strength": round(seasonal_strength, 4),
        "seasonal_profile": {m: round(v, 4) for m, v in seasonal_profile.items()},
        "annual_totals": {str(y): round(annual_totals[y], 2) for y in sorted(annual_totals)},
        "full_years": full_years,
        "strong_anomaly_count": sum(r["severity"] == "strong" for r in rows),
        "watch_anomaly_count": sum(r["severity"] == "watch" for r in rows),
    }

    write_csv(RESULTS_DIR / f"{out_prefix}_stl.csv", rows)

    # 图表
    save_line_plot(
        FIGURES_DIR / f"{out_prefix}_observed_trend.png",
        dates, [("观测值", values), ("STL 趋势", trend)],
        f"{region}月度全社会用电量：观测值与 STL 趋势", "亿千瓦时",
    )
    save_line_plot(
        FIGURES_DIR / f"{out_prefix}_seasonal.png",
        dates, [("季节项", seasonal)],
        f"{region}月度全社会用电量：STL 季节项", "亿千瓦时", zero_line=True,
    )
    save_line_plot(
        FIGURES_DIR / f"{out_prefix}_residual.png",
        dates, [("残差/趋势", residual_pct)],
        f"{region}月度全社会用电量：STL 残差率", "%", zero_line=True,
    )

    return summary


def run_stl_annual(records: list[AnnualRecord], region: str, out_prefix: str) -> dict:
    """对年度数据运行 STL 分解"""
    records = sorted(records, key=lambda r: r.year)
    years = np.array([r.year for r in records], dtype=float)
    values = np.array([r.total for r in records], dtype=float)

    # 年度 STL：使用 period=4 或 5 来捕捉可能的周期性
    params = {"period": 5, "seasonal": 7, "trend": 9, "low_pass": 7, "robust": True}

    if len(values) < 8:
        return {"region": region, "error": f"年度数据不足（{len(values)} 年）"}

    try:
        model = STL(values, **params)
        result = model.fit()
    except Exception as e:
        return {"region": region, "error": f"STL 失败：{e}"}

    trend = np.asarray(result.trend, dtype=float)
    seasonal = np.asarray(result.seasonal, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    expected = trend + seasonal
    residual_pct = np.divide(
        resid, trend,
        out=np.zeros_like(resid),
        where=np.abs(trend) > 1e-12,
    ) * 100.0

    trend_strength, seasonal_strength = stl_strength(trend, seasonal, resid)

    rows = []
    for i, yr in enumerate(years):
        abs_pct = abs(float(residual_pct[i]))
        severity = "strong" if abs_pct >= 10 else ("watch" if abs_pct >= 5 else "normal")
        rows.append({
            "region": region,
            "year": int(yr),
            "indicator": "全社会用电量",
            "unit": "万千瓦时",
            "observed": round(float(values[i]), 2),
            "trend": round(float(trend[i]), 2),
            "seasonal": round(float(seasonal[i]), 2),
            "expected": round(float(expected[i]), 2),
            "residual": round(float(resid[i]), 2),
            "residual_pct_of_trend": round(float(residual_pct[i]), 4),
            "severity": severity,
        })

    write_csv(RESULTS_DIR / f"{out_prefix}_stl.csv", rows)

    summary = {
        "region": region,
        "n_observations": len(records),
        "start_year": int(years[0]),
        "end_year": int(years[-1]),
        "stl_parameters": params,
        "trend_start": round(float(trend[0]), 2),
        "trend_end": round(float(trend[-1]), 2),
        "trend_growth_pct": round(float((trend[-1] / trend[0] - 1.0) * 100.0), 2),
        "trend_strength": round(trend_strength, 4),
        "seasonal_strength": round(seasonal_strength, 4),
        "strong_anomaly_count": sum(r["severity"] == "strong" for r in rows),
        "watch_anomaly_count": sum(r["severity"] == "watch" for r in rows),
    }

    save_line_plot(
        FIGURES_DIR / f"{out_prefix}_observed_trend.png",
        [date(int(y), 1, 1) for y in years],
        [("观测值", values), ("STL 趋势", trend)],
        f"{region}年度全社会用电量：观测值与 STL 趋势", "万千瓦时",
    )
    save_line_plot(
        FIGURES_DIR / f"{out_prefix}_residual.png",
        [date(int(y), 1, 1) for y in years],
        [("残差/趋势", residual_pct)],
        f"{region}年度全社会用电量：STL 残差率", "%", zero_line=True,
    )

    return summary


def run_stl_quarterly_gdp(gdp_rows: list[dict], region: str, out_prefix: str) -> dict:
    """对季度 GDP 数据运行 STL"""
    from collections import defaultdict

    sectors = defaultdict(list)
    for r in gdp_rows:
        sectors[r["sector"]].append(r)

    all_results = {}
    for sector, rows in sectors.items():
        rows = sorted(rows, key=lambda x: x["period"])
        # 只取 2010 年以后
        rows = [r for r in rows if int(r["period"][:4]) >= 2010]
        if len(rows) < 16:
            all_results[sector] = {"error": f"数据不足（{len(rows)} 季度）"}
            continue

        dates = [datetime.strptime(r["date"], "%Y-%m-%d").date() for r in rows]

        # 同比增速 STL
        growth = np.array([r["cumulative_yoy_pct"] for r in rows], dtype=float)
        g_params = {"period": 4, "seasonal": 7, "trend": 9, "low_pass": 5, "robust": True}
        try:
            g_result = STL(growth, **g_params).fit()
            g_trend = np.asarray(g_result.trend, dtype=float)
            g_seasonal = np.asarray(g_result.seasonal, dtype=float)
            g_resid = np.asarray(g_result.resid, dtype=float)

            g_rows = []
            for i, r in enumerate(rows):
                severity = "normal"
                abs_r = abs(float(g_resid[i]))
                if abs_r >= 5:
                    severity = "strong"
                elif abs_r >= 3:
                    severity = "watch"
                g_rows.append({
                    "region": region, "sector": sector,
                    "period": r["period"], "date": r["date"],
                    "indicator": "累计增加值同比增速", "unit": "%",
                    "observed": r["cumulative_yoy_pct"],
                    "trend": round(float(g_trend[i]), 4),
                    "seasonal": round(float(g_seasonal[i]), 4),
                    "expected": round(float(g_trend[i] + g_seasonal[i]), 4),
                    "residual_pp": round(float(g_resid[i]), 4),
                    "severity": severity,
                })
            write_csv(RESULTS_DIR / f"{out_prefix}_gdp_{sector}_growth_stl.csv", g_rows)
        except Exception as e:
            g_rows = []

        # 单季增加值对数 STL
        flow = np.array([r["single_quarter_value_100m_yuan"] for r in rows], dtype=float)
        if np.all(flow > 0):
            log_flow = np.log(flow)
            f_params = {"period": 4, "seasonal": 7, "trend": 9, "low_pass": 5, "robust": True}
            try:
                f_result = STL(log_flow, **f_params).fit()
                f_trend = np.asarray(f_result.trend, dtype=float)
                f_seasonal = np.asarray(f_result.seasonal, dtype=float)
                f_resid = np.asarray(f_result.resid, dtype=float)
                f_resid_pct = (np.exp(f_resid) - 1.0) * 100.0

                f_rows = []
                for i, r in enumerate(rows):
                    severity = "normal"
                    abs_r = abs(float(f_resid_pct[i]))
                    if abs_r >= 10:
                        severity = "strong"
                    elif abs_r >= 5:
                        severity = "watch"
                    f_rows.append({
                        "region": region, "sector": sector,
                        "period": r["period"], "date": r["date"],
                        "indicator": "单季增加值", "unit": "亿元",
                        "observed": r["single_quarter_value_100m_yuan"],
                        "trend": round(float(f_trend[i]), 4),
                        "seasonal": round(float(f_seasonal[i]), 4),
                        "expected": round(float(np.exp(f_trend[i] + f_seasonal[i])), 4),
                        "residual_pct": round(float(f_resid_pct[i]), 4),
                        "severity": severity,
                    })
                write_csv(RESULTS_DIR / f"{out_prefix}_gdp_{sector}_flow_stl.csv", f_rows)
            except Exception:
                f_rows = []

        g_tr_s, g_s_s = stl_strength(g_trend, g_seasonal, g_resid) if 'g_trend' in dir() else (0, 0)
        all_results[sector] = {
            "n_quarters": len(rows),
            "growth_trend_strength": round(g_tr_s, 4),
            "growth_seasonal_strength": round(g_s_s, 4),
            "growth_anomalies": sum(r["severity"] != "normal" for r in g_rows) if g_rows else 0,
        }

        # 图表
        save_line_plot(
            FIGURES_DIR / f"{out_prefix}_gdp_{sector}_growth.png",
            dates,
            [("观测同比增速", growth), ("STL 趋势", g_trend)],
            f"{region}{sector}累计增加值同比增速：STL 分解", "%",
            zero_line=True,
        )

    return {"region": region, "sectors": all_results}


# ============================================================
# 4. 图表工具
# ============================================================

def save_line_plot(path: Path, dates: Sequence[date],
                   series: Sequence[tuple[str, Sequence[float]]],
                   title: str, ylabel: str, zero_line: bool = False,
                   note: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.2, 5.4), dpi=160)
    ax = fig.add_subplot(111)
    for label, vals in series:
        ax.plot(dates, vals, linewidth=1.8, label=label)
    if zero_line:
        ax.axhline(0.0, linewidth=0.9, color="gray", linestyle="--")
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


# ============================================================
# 5. 主流程
# ============================================================

def main():
    set_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = Path("浙江省数据(1).zip")
    if not zip_path.exists():
        print("错误：找不到 浙江省数据(1).zip")
        return 1

    print("=" * 60)
    print("四城市电力数据整理与 STL 分解")
    print("=" * 60)

    all_summaries = {}
    log_lines = []

    # ---- 浙江 ----
    print("\n[1/4] 处理浙江数据...")
    try:
        zj_electricity = extract_zhejiang_electricity(zip_path)
        zj_gdp = extract_zhejiang_gdp(zip_path)

        # 写入清洗后 CSV
        write_csv(DATA_DIR / "zhejiang_electricity_monthly.csv", [
            {"region": r.region, "date": r.month.isoformat(),
             "indicator": r.indicator, "value": r.value, "unit": r.unit}
            for r in zj_electricity
        ])
        write_csv(DATA_DIR / "zhejiang_gdp_quarterly.csv", zj_gdp)

        # STL 分解
        zj_elec_summary = run_stl_monthly(
            zj_electricity, "浙江", ELECTRICITY_MONTHLY_STL, "zhejiang_electricity"
        )
        zj_gdp_summary = run_stl_quarterly_gdp(zj_gdp, "浙江", "zhejiang")

        all_summaries["浙江"] = {
            "electricity": zj_elec_summary,
            "gdp": zj_gdp_summary,
        }

        e = zj_elec_summary
        log_lines.append(f"\n浙江月度用电量：{e.get('n_observations', 'N/A')} 个月, "
                        f"{e.get('start', '?')} 至 {e.get('end', '?')}")
        if "trend_growth_pct" in e and e["trend_growth_pct"] is not None:
            log_lines.append(f"  趋势增长：{e['trend_growth_pct']:.2f}%")
        log_lines.append(f"  趋势强度：{e.get('trend_strength', 'N/A')}, "
                        f"季节强度：{e.get('seasonal_strength', 'N/A')}")
        log_lines.append(f"  异常候选：strong={e.get('strong_anomaly_count', 0)}, "
                        f"watch={e.get('watch_anomaly_count', 0)}")
        print("  浙江处理完成 ✓")
    except Exception as exc:
        print(f"  浙江处理失败：{exc}")
        import traceback
        traceback.print_exc()

    # ---- 苏州 ----
    print("\n[2/4] 处理苏州数据...")
    try:
        sz_data = parse_suzhou_data()
        write_csv(DATA_DIR / "suzhou_electricity_monthly.csv", [
            {"region": r.region, "date": r.month.isoformat(),
             "indicator": r.indicator, "value": r.value, "unit": r.unit}
            for r in sz_data
        ])
        sz_summary = run_stl_monthly(sz_data, "苏州", ELECTRICITY_MONTHLY_STL, "suzhou_electricity")
        all_summaries["苏州"] = {"electricity": sz_summary}

        log_lines.append(f"\n苏州月度用电量：{sz_summary.get('n_observations', 'N/A')} 个月, "
                        f"{sz_summary.get('start', '?')} 至 {sz_summary.get('end', '?')}")
        if sz_summary.get("trend_growth_pct") is not None:
            log_lines.append(f"  趋势增长：{sz_summary['trend_growth_pct']:.2f}%")
        log_lines.append(f"  趋势强度：{sz_summary.get('trend_strength', 'N/A')}, "
                        f"季节强度：{sz_summary.get('seasonal_strength', 'N/A')}")
        print("  苏州处理完成 ✓")
    except Exception as exc:
        print(f"  苏州处理失败：{exc}")
        import traceback
        traceback.print_exc()

    # ---- 南京 ----
    print("\n[3/4] 处理南京数据...")
    try:
        # 年度数据 STL
        nj_annual = parse_nanjing_annual()
        write_csv(DATA_DIR / "nanjing_electricity_annual.csv", [
            {"region": r.region, "year": r.year, "total_10k_kwh": r.total,
             "primary": r.primary, "secondary": r.secondary, "tertiary": r.tertiary}
            for r in nj_annual
        ])
        nj_annual_summary = run_stl_annual(nj_annual, "南京", "nanjing_annual")

        # 月度数据 STL
        nj_monthly = parse_nanjing_monthly()
        write_csv(DATA_DIR / "nanjing_electricity_monthly.csv", [
            {"region": r.region, "date": r.month.isoformat(),
             "indicator": r.indicator, "value": r.value, "unit": r.unit}
            for r in nj_monthly
        ])
        nj_monthly_summary = run_stl_monthly(
            nj_monthly, "南京", ELECTRICITY_MONTHLY_STL, "nanjing_monthly"
        )

        all_summaries["南京"] = {
            "annual": nj_annual_summary,
            "monthly": nj_monthly_summary,
        }

        a = nj_annual_summary
        log_lines.append(f"\n南京年度用电量：{a.get('n_observations', 'N/A')} 年, "
                        f"{a.get('start_year', '?')}-{a.get('end_year', '?')}")
        if a.get("trend_growth_pct") is not None:
            log_lines.append(f"  趋势增长：{a['trend_growth_pct']:.2f}%")
        log_lines.append(f"  趋势强度：{a.get('trend_strength', 'N/A')}, "
                        f"季节强度：{a.get('seasonal_strength', 'N/A')}")

        m = nj_monthly_summary
        if "error" not in m:
            log_lines.append(f"\n南京月度用电量：{m.get('n_observations', 'N/A')} 个月")
            log_lines.append(f"  趋势强度：{m.get('trend_strength', 'N/A')}, "
                            f"季节强度：{m.get('seasonal_strength', 'N/A')}")
        print("  南京处理完成 ✓")
    except Exception as exc:
        print(f"  南京处理失败：{exc}")
        import traceback
        traceback.print_exc()

    # ---- 江门 ----
    print("\n[4/4] 处理江门数据...")
    try:
        jm_data = parse_jiangmen_data()
        write_csv(DATA_DIR / "jiangmen_electricity_monthly.csv", [
            {"region": r.region, "date": r.month.isoformat(),
             "indicator": r.indicator, "value": r.value, "unit": r.unit}
            for r in jm_data
        ])
        jm_summary = run_stl_monthly(jm_data, "江门", ELECTRICITY_MONTHLY_STL, "jiangmen_electricity")
        all_summaries["江门"] = {"electricity": jm_summary}

        log_lines.append(f"\n江门月度用电量：{jm_summary.get('n_observations', 'N/A')} 个月, "
                        f"{jm_summary.get('start', '?')} 至 {jm_summary.get('end', '?')}")
        if jm_summary.get("trend_growth_pct") is not None:
            log_lines.append(f"  趋势增长：{jm_summary['trend_growth_pct']:.2f}%")
        log_lines.append(f"  趋势强度：{jm_summary.get('trend_strength', 'N/A')}, "
                        f"季节强度：{jm_summary.get('seasonal_strength', 'N/A')}")
        print("  江门处理完成 ✓")
    except Exception as exc:
        print(f"  江门处理失败：{exc}")
        import traceback
        traceback.print_exc()

    # ---- 汇总 ----
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "statsmodels": __import__("statsmodels").__version__,
            "matplotlib": matplotlib.__version__,
        },
        "regions": all_summaries,
    }

    (RESULTS_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 运行日志
    log_header = [
        "四城市（浙江、苏州、南京、江门）电力数据 STL 分解运行日志",
        "=" * 56,
        f"Python {platform.python_version()} | numpy {np.__version__} | statsmodels {__import__('statsmodels').__version__}",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "===== 数据说明 =====",
        "浙江：月度全社会用电量 + 季度 GDP（Excel 原始数据）",
        "苏州：月度全社会用电量（OCR 从统计公报截图提取，累计值差分转单月）",
        "南京：年度全社会用电量 2000-2023 + 月度全社会用电量（OCR 提取）",
        "江门：月度全社会用电量（OCR 从统计公报截图提取，累计值差分转单月）",
        "",
        "===== 注意事项 =====",
        "1. 苏州、南京月度、江门数据源为 DOCX 中的截图，经 EasyOCR 识别后人工校核。",
        "2. OCR 可能存在个别数字识别错误，用前请核对原始公报。",
        "3. 单月值由累计值做差得到，Q4/12月承接全年差额。",
        "4. STL 残差仅为异常候选，不等同于因果结论。",
        "",
    ]
    log_lines = log_header + log_lines
    (OUT_DIR / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("处理完成！输出文件：")
    print(f"  数据 CSV：{DATA_DIR}")
    print(f"  STL 结果：{RESULTS_DIR}")
    print(f"  图表：    {FIGURES_DIR}")
    print(f"  运行日志：{OUT_DIR / 'run_log.txt'}")
    print("=" * 60)

    # 输出汇总
    print("\n汇总：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
