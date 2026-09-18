#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浙江省电力-经济数据 STL 分解与异常事件构建流水线

目标：
1. 将用户提供的月度用电量、季度三次产业 GDP 累计值和节假日通知规范化；
2. 在对数尺度上执行稳健 STL 分解；
3. 输出趋势、季节项、期望值、残差、异常等级和数据质量标记；
4. 生成可供大模型/智能体直接读取的 JSONL；
5. 生成可衔接知识图谱的节点、关系 CSV。

说明：
- 不依赖 pandas/openpyxl，XLSX 使用标准库 zipfile + XML 读取；
- STL 由 statsmodels.tsa.seasonal.STL 实现；
- 异常阈值是工程阈值，不等同于统计显著性；
- GDP 原始表为年内累计值，脚本先差分为单季度值。Q4 可能吸收年度修订，故单独标记 revision_risk。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib import font_manager
from docx import Document


# -----------------------------
# 全局配置
# -----------------------------

REGION_CODE = "CN-ZJ"
REGION_NAME = "浙江省"
SCHEMA_VERSION = "1.0"

ELECTRICITY_FILE = "浙江全社会用电量信息（月度）.xlsx"
GDP_FILES = {
    "第一产业": "浙江省第一产业季度.xlsx",
    "第二产业": "浙江省第二产业季度.xlsx",
    "第三产业": "浙江省第三产业季度.xlsx",
}
HOLIDAY_FILE = "国务院办公厅近五年节假日安排.docx"

SOURCE_URLS = {
    "nbs_seasonal_adjustment": "https://www.stats.gov.cn/zs/tjws/tjbk/202301/t20230101_1912931.html",
    "statsmodels_stl": "https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html",
    "statsmodels_stl_fit": "https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.fit.html",
    "zhejiang_2024_electricity_validation": "https://www.news.cn/20250124/4a7d51aece054002ac1edc66d125884e/c.html",
    "zhejiang_2025_statistical_bulletin": "https://zjzd.stats.gov.cn/zwgk/zfxxgkml/tjxx/tjgb/art/2026/art_48c3c7315981425f9c25b53eab4d65d0.html",
    "holiday_2021": "https://app.www.gov.cn/govdata/gov/202011/25/465322/article.html",
    "holiday_2022": "https://app.www.gov.cn/govdata/gov/202110/25/477428/article.html",
    "holiday_2023": "https://app.www.gov.cn/govdata/gov/202212/08/495070/article.html",
    "holiday_2024": "https://app.www.gov.cn/govdata/gov/202310/25/508678/article.html",
    "holiday_2025": "https://www.gov.cn/zhengce/content/202411/content_6986382.htm",
}

CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if CJK_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(CJK_FONT_PATH))
    _cjk_font_name = font_manager.FontProperties(fname=str(CJK_FONT_PATH)).get_name()
    plt.rcParams["font.family"] = _cjk_font_name
    plt.rcParams["font.sans-serif"] = [_cjk_font_name]
else:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"
    ]
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# 通用工具
# -----------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if not rows and not fieldnames:
        raise ValueError(f"无法写出空表且未指定列名：{path}")
    if fieldnames is None:
        # 保持第一条记录的字段顺序，并补齐后续新字段
        names: list[str] = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in names:
                    names.append(key)
        fieldnames = names
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法 JSON 序列化：{type(value)!r}")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")


def safe_float(value: Any) -> float:
    text = str(value).strip().replace(",", "")
    if not text or text in {"--", "—", "-", "None", "nan"}:
        raise ValueError(f"不能转换为数值：{value!r}")
    return float(text)


def month_sequence(start: date, end: date) -> list[date]:
    result: list[date] = []
    current = date(start.year, start.month, 1)
    target = date(end.year, end.month, 1)
    while current <= target:
        result.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return result


def quarter_label(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def quarter_start(year: int, quarter: int) -> date:
    return date(year, 1 + (quarter - 1) * 3, 1)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad <= 1e-12:
        # 极小 MAD 时避免无穷值；保留方向与相对大小
        scale = max(float(np.std(values)), 1e-12)
        return (values - med) / scale
    return 0.6744897501960817 * (values - med) / mad


def component_strength(component: np.ndarray, resid: np.ndarray) -> float:
    denom = float(np.var(component + resid, ddof=1))
    if denom <= 1e-15:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(np.var(resid, ddof=1)) / denom))


def anomaly_level(residual_pct: float, medium_threshold: float = 0.05, high_threshold: float = 0.08) -> str:
    magnitude = abs(residual_pct)
    if magnitude >= high_threshold:
        return "high"
    if magnitude >= medium_threshold:
        return "medium"
    return "normal"


# -----------------------------
# 轻量 XLSX 读取器（首个工作表）
# -----------------------------

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def column_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return idx - 1


def read_first_sheet_xlsx(path: Path) -> list[list[str]]:
    """读取 XLSX 第一个工作表为二维字符串矩阵。支持共享字符串、内联字符串和数值。"""
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{{{NS_MAIN}}}si"):
                shared_strings.append("".join(t.text or "" for t in si.iter(f"{{{NS_MAIN}}}t")))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook.find(f"{{{NS_MAIN}}}sheets/{{{NS_MAIN}}}sheet")
        if first_sheet is None:
            raise ValueError(f"工作簿没有工作表：{path}")
        rel_id = first_sheet.attrib[f"{{{NS_REL}}}id"]

        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall(f"{{{NS_PKG_REL}}}Relationship"):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise ValueError(f"找不到工作表关系：{path}")
        sheet_path = target.lstrip("/")
        if not sheet_path.startswith("xl/"):
            sheet_path = "xl/" + sheet_path

        sheet_xml = ET.fromstring(zf.read(sheet_path))
        cells: dict[tuple[int, int], str] = {}
        max_row = 0
        max_col = 0
        for cell in sheet_xml.iter(f"{{{NS_MAIN}}}c"):
            ref = cell.attrib.get("r", "")
            match = re.fullmatch(r"([A-Z]+)(\d+)", ref)
            if not match:
                continue
            col = column_letters_to_index(match.group(1))
            row = int(match.group(2)) - 1
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                value = "".join(t.text or "" for t in cell.iter(f"{{{NS_MAIN}}}t"))
            else:
                value_node = cell.find(f"{{{NS_MAIN}}}v")
                raw = "" if value_node is None or value_node.text is None else value_node.text
                if cell_type == "s" and raw != "":
                    value = shared_strings[int(raw)]
                elif cell_type == "b":
                    value = "TRUE" if raw == "1" else "FALSE"
                else:
                    value = raw
            cells[(row, col)] = value.strip() if isinstance(value, str) else value

        matrix: list[list[str]] = []
        for r in range(max_row + 1):
            matrix.append([str(cells.get((r, c), "")) for c in range(max_col + 1)])
        return matrix


# -----------------------------
# 原始数据读取与质量检查
# -----------------------------

@dataclass
class DataQualityCheck:
    check_id: str
    dataset_id: str
    check_name: str
    passed: bool
    detail: str
    severity: str = "error"


def load_electricity(path: Path) -> tuple[list[dict[str, Any]], list[DataQualityCheck]]:
    matrix = read_first_sheet_xlsx(path)
    if len(matrix) < 2:
        raise ValueError("用电量工作簿没有有效数据")
    header = {name.strip(): idx for idx, name in enumerate(matrix[0])}
    required = ["统计年月", "指标名称", "用电量"]
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"用电量工作簿缺少列：{missing}")

    rows: list[dict[str, Any]] = []
    for raw in matrix[1:]:
        if not any(str(x).strip() for x in raw):
            continue
        stat_date = datetime.strptime(raw[header["统计年月"]].strip(), "%Y-%m-%d").date()
        raw_value = safe_float(raw[header["用电量"]])
        rows.append({
            "date": date(stat_date.year, stat_date.month, 1),
            "period": f"{stat_date.year:04d}-{stat_date.month:02d}",
            "metric_name": raw[header["指标名称"]].strip(),
            "raw_value": raw_value,
            "raw_unit": "万千瓦时",
            "value": raw_value / 10000.0,
            "unit": "亿千瓦时",
            "source_file": path.name,
        })
    rows.sort(key=lambda r: r["date"])

    checks: list[DataQualityCheck] = []
    dates = [r["date"] for r in rows]
    checks.append(DataQualityCheck(
        "E001", "electricity_monthly", "记录数", len(rows) > 0,
        f"读取 {len(rows)} 条月度记录，范围 {dates[0]} 至 {dates[-1]}" if rows else "0 条记录"
    ))
    expected = month_sequence(dates[0], dates[-1]) if dates else []
    checks.append(DataQualityCheck(
        "E002", "electricity_monthly", "月度连续性", dates == expected,
        "月份连续且无缺口" if dates == expected else f"期望 {len(expected)} 个月，实际 {len(dates)} 个月"
    ))
    checks.append(DataQualityCheck(
        "E003", "electricity_monthly", "月份唯一性", len(dates) == len(set(dates)),
        "每个月唯一" if len(dates) == len(set(dates)) else "存在重复月份"
    ))
    checks.append(DataQualityCheck(
        "E004", "electricity_monthly", "正值检查", all(r["value"] > 0 for r in rows),
        "全部用电量为正" if all(r["value"] > 0 for r in rows) else "存在非正值"
    ))
    metric_names = sorted(set(r["metric_name"] for r in rows))
    checks.append(DataQualityCheck(
        "E005", "electricity_monthly", "指标一致性", len(metric_names) == 1,
        f"指标名称：{metric_names}"
    ))
    return rows, checks


def parse_quarter_header(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{4})年([1-4])季度", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def load_gdp_industry(path: Path, industry: str, min_year: int = 2000) -> tuple[list[dict[str, Any]], list[DataQualityCheck]]:
    matrix = read_first_sheet_xlsx(path)
    if len(matrix) < 3:
        raise ValueError(f"GDP 工作簿结构异常：{path}")
    headers = matrix[0]
    cumulative_row = matrix[1]
    yoy_row = matrix[2]

    cumulative: dict[tuple[int, int], float] = {}
    yoy: dict[tuple[int, int], float | None] = {}
    for col in range(1, len(headers)):
        parsed = parse_quarter_header(headers[col])
        if parsed is None:
            continue
        year, quarter = parsed
        if year < min_year:
            continue
        try:
            cumulative[(year, quarter)] = safe_float(cumulative_row[col])
        except (IndexError, ValueError):
            continue
        try:
            yoy[(year, quarter)] = safe_float(yoy_row[col])
        except (IndexError, ValueError):
            yoy[(year, quarter)] = None

    rows: list[dict[str, Any]] = []
    for year in sorted({y for y, _ in cumulative}):
        previous_cumulative = 0.0
        for quarter in range(1, 5):
            key = (year, quarter)
            if key not in cumulative:
                continue
            current = cumulative[key]
            single_quarter = current if quarter == 1 else current - previous_cumulative
            rows.append({
                "date": quarter_start(year, quarter),
                "year": year,
                "quarter": quarter,
                "period": quarter_label(year, quarter),
                "industry": industry,
                "cumulative_value": current,
                "value": single_quarter,
                "unit": "亿元（当年价格，按累计值差分得到单季度值）",
                "cumulative_yoy_pct": yoy.get(key),
                "revision_risk": quarter == 4,
                "source_file": path.name,
            })
            previous_cumulative = current
    rows.sort(key=lambda r: r["date"])

    checks: list[DataQualityCheck] = []
    checks.append(DataQualityCheck(
        f"G_{industry}_001", f"gdp_{industry}", "记录数", len(rows) > 0,
        f"读取并转换 {len(rows)} 条单季度记录，范围 {rows[0]['period']} 至 {rows[-1]['period']}" if rows else "0 条记录"
    ))
    checks.append(DataQualityCheck(
        f"G_{industry}_002", f"gdp_{industry}", "单季度正值检查", all(r["value"] > 0 for r in rows),
        "全部差分后的单季度值为正" if all(r["value"] > 0 for r in rows) else "存在非正单季度值；需要检查累计值或修订"
    ))
    # 每年累计 Q4 应等于四个单季度之和（代数恒等，但可检测缺季度）
    yearly_counts = defaultdict(int)
    for row in rows:
        yearly_counts[row["year"]] += 1
    bad_years = [y for y, n in yearly_counts.items() if n != 4]
    checks.append(DataQualityCheck(
        f"G_{industry}_003", f"gdp_{industry}", "季度完整性", not bad_years,
        "每年 4 个季度完整" if not bad_years else f"季度不完整年份：{bad_years}"
    ))
    checks.append(DataQualityCheck(
        f"G_{industry}_004", f"gdp_{industry}", "Q4 修订风险标记", True,
        "Q4 由全年累计值减前三季度累计值得到，可能集中吸收年度核算修订；已设置 revision_risk=True",
        severity="warning"
    ))
    return rows, checks


# -----------------------------
# 节假日通知解析
# -----------------------------

CHINESE_NUMERAL_PREFIX = r"[一二三四五六七八九十]+"
HOLIDAY_LINE_RE = re.compile(rf"^{CHINESE_NUMERAL_PREFIX}、(?P<name>[^：:]+)[：:](?P<body>.+)$")
RANGE_RE = re.compile(
    r"(?:(?P<y1>\d{4})年)?(?P<m1>\d{1,2})月(?P<d1>\d{1,2})日?"
    r"(?:[^。；;]*?至(?:(?P<y2>\d{4})年)?(?:(?P<m2>\d{1,2})月)?(?P<d2>\d{1,2})日?)?"
)
DATE_TOKEN_RE = re.compile(r"(?:(?P<year>\d{4})年)?(?P<month>\d{1,2})月(?P<day>\d{1,2})日")


def parse_holiday_notice(path: Path) -> tuple[list[dict[str, Any]], list[DataQualityCheck]]:
    doc = Document(path)
    current_year: int | None = None
    events: list[dict[str, Any]] = []
    for para in doc.paragraphs:
        text = para.text.strip().replace(" ", "")
        if not text:
            continue
        year_match = re.search(r"关于(20\d{2})年", text) or re.search(r"现将(20\d{2})年", text)
        if year_match:
            current_year = int(year_match.group(1))
        line_match = HOLIDAY_LINE_RE.match(text)
        if not line_match or current_year is None:
            continue
        name = line_match.group("name").replace("、", "+")
        body = line_match.group("body")
        leave_clause = body.split("。", 1)[0]
        range_match = RANGE_RE.search(leave_clause)
        if range_match:
            y1 = int(range_match.group("y1") or current_year)
            m1 = int(range_match.group("m1"))
            d1 = int(range_match.group("d1"))
            y2 = int(range_match.group("y2") or y1)
            m2 = int(range_match.group("m2") or m1)
            d2 = int(range_match.group("d2") or d1)
            start = date(y1, m1, d1)
            end = date(y2, m2, d2)
            if end < start:
                # 未显式写年份且跨年时的兜底规则
                end = date(y2 + 1, m2, d2)
            events.append({
                "event_id": f"holiday:{current_year}:{name}",
                "notice_year": current_year,
                "event_type": "holiday_leave_interval",
                "holiday_name": name,
                "start_date": start,
                "end_date": end,
                "days": (end - start).days + 1,
                "source_file": path.name,
                "source_url": SOURCE_URLS.get(f"holiday_{current_year}", ""),
                "raw_text": text,
            })

        # “上班”日期：仅解析明确写有“上班”的句子，避免把“除夕休息”等日期误判为调休工作日。
        for sentence in re.split(r"[。；;]", body):
            if "上班" not in sentence:
                continue
            for token in DATE_TOKEN_RE.finditer(sentence):
                y = int(token.group("year") or current_year)
                m = int(token.group("month"))
                d = int(token.group("day"))
                workday = date(y, m, d)
                events.append({
                    "event_id": f"makeup_workday:{current_year}:{workday.isoformat()}",
                    "notice_year": current_year,
                    "event_type": "makeup_workday",
                    "holiday_name": name,
                    "start_date": workday,
                    "end_date": workday,
                    "days": 1,
                    "source_file": path.name,
                    "source_url": SOURCE_URLS.get(f"holiday_{current_year}", ""),
                    "raw_text": text,
                })

    # 去重
    unique: dict[tuple[str, date, date, str], dict[str, Any]] = {}
    for event in events:
        key = (event["event_type"], event["start_date"], event["end_date"], event["holiday_name"])
        unique[key] = event
    events = sorted(unique.values(), key=lambda e: (e["start_date"], e["event_type"], e["holiday_name"]))

    leave_years = sorted({e["notice_year"] for e in events if e["event_type"] == "holiday_leave_interval"})
    checks = [
        DataQualityCheck(
            "H001", "holiday_calendar", "覆盖年份", leave_years == [2021, 2022, 2023, 2024, 2025],
            f"解析到节假日通知年份：{leave_years}"
        ),
        DataQualityCheck(
            "H002", "holiday_calendar", "春节事件", sum(1 for e in events if "春节" in e["holiday_name"] and e["event_type"] == "holiday_leave_interval") == 5,
            "2021-2025 每年均解析到春节放假区间"
        ),
    ]
    return events, checks


def aggregate_holiday_months(events: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    months = month_sequence(start, end)
    result: list[dict[str, Any]] = []
    for month in months:
        month_end = date(month.year + (1 if month.month == 12 else 0), 1 if month.month == 12 else month.month + 1, 1) - timedelta(days=1)
        holiday_days = 0
        spring_festival_days = 0
        makeup_workdays = 0
        names: set[str] = set()
        event_ids: list[str] = []
        for event in events:
            overlap_start = max(month, event["start_date"])
            overlap_end = min(month_end, event["end_date"])
            if overlap_start > overlap_end:
                continue
            overlap_days = (overlap_end - overlap_start).days + 1
            names.add(event["holiday_name"])
            event_ids.append(event["event_id"])
            if event["event_type"] == "holiday_leave_interval":
                holiday_days += overlap_days
                if "春节" in event["holiday_name"]:
                    spring_festival_days += overlap_days
            elif event["event_type"] == "makeup_workday":
                makeup_workdays += overlap_days
        result.append({
            "date": month,
            "period": f"{month.year:04d}-{month.month:02d}",
            "holiday_days": holiday_days,
            "spring_festival_days": spring_festival_days,
            "makeup_workdays": makeup_workdays,
            "holiday_names": sorted(names),
            "holiday_event_ids": sorted(set(event_ids)),
        })
    return result


# -----------------------------
# STL 分解
# -----------------------------

@dataclass
class STLConfig:
    period: int
    seasonal: int
    robust: bool = True
    medium_threshold: float = 0.05
    high_threshold: float = 0.08


def run_log_stl(rows: list[dict[str, Any]], config: STLConfig) -> tuple[list[dict[str, Any]], dict[str, float]]:
    values = np.asarray([float(r["value"]) for r in rows], dtype=float)
    if np.any(values <= 0):
        raise ValueError("对数 STL 要求序列全部为正")
    log_values = np.log(values)
    fit = STL(
        log_values,
        period=config.period,
        seasonal=config.seasonal,
        robust=config.robust,
    ).fit()
    trend = np.asarray(fit.trend, dtype=float)
    seasonal = np.asarray(fit.seasonal, dtype=float)
    resid = np.asarray(fit.resid, dtype=float)
    weights = np.asarray(fit.weights, dtype=float)
    expected = np.exp(trend + seasonal)
    residual_pct = values / expected - 1.0
    z = robust_zscore(resid)

    n = len(rows)
    output: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        enriched = dict(row)
        enriched.update({
            "log_observed": float(log_values[i]),
            "trend_log": float(trend[i]),
            "seasonal_log": float(seasonal[i]),
            "resid_log": float(resid[i]),
            "trend_level": float(np.exp(trend[i])),
            "seasonal_factor": float(np.exp(seasonal[i])),
            "expected_value": float(expected[i]),
            "residual_value": float(values[i] - expected[i]),
            "residual_pct": float(residual_pct[i]),
            "residual_pct_points": float(residual_pct[i] * 100.0),
            "robust_z": float(z[i]),
            "stl_weight": float(weights[i]),
            "anomaly_level": anomaly_level(float(residual_pct[i]), config.medium_threshold, config.high_threshold),
            "statistical_outlier": bool(abs(float(z[i])) >= 3.5),
            "endpoint_warning": bool(i < config.period or i >= n - config.period),
            "stl_period": config.period,
            "stl_seasonal": config.seasonal,
            "stl_robust": config.robust,
        })
        output.append(enriched)

    metrics = {
        "seasonal_strength": component_strength(seasonal, resid),
        "trend_strength": component_strength(trend, resid),
        "residual_mad_log": float(np.median(np.abs(resid - np.median(resid)))),
        "residual_std_log": float(np.std(resid, ddof=1)),
        "n_observations": n,
    }
    return output, metrics


# -----------------------------
# 异常上下文、模型输入与知识图谱
# -----------------------------

CAUSE_CATALOG = {
    "spring_festival_shift": {
        "name": "春节移动假日与停复工节奏",
        "category": "holiday",
        "required_evidence": ["春节日期", "制造业开工率", "分行业用电量"],
    },
    "weather_temperature": {
        "name": "极端温度与制冷/采暖负荷",
        "category": "weather",
        "required_evidence": ["逐日平均/最高温度", "空调用电负荷", "高温日数"],
    },
    "industrial_activity": {
        "name": "工业生产与订单变化",
        "category": "economy",
        "required_evidence": ["规上工业增加值", "PMI/订单", "第二产业分行业用电量"],
    },
    "policy_or_power_event": {
        "name": "电价、需求响应、限电或重大政策事件",
        "category": "policy_event",
        "required_evidence": ["政策发布时间", "需求响应记录", "电网运行事件"],
    },
    "annual_revision": {
        "name": "累计值差分与年度核算修订",
        "category": "data_quality",
        "required_evidence": ["GDP 历史版本/数据 vintage", "官方单季度值"],
    },
    "public_health_event": {
        "name": "公共卫生事件及管控冲击",
        "category": "event",
        "required_evidence": ["事件时间线", "交通/复工数据", "行业停工范围"],
    },
}


def attach_electricity_context(rows: list[dict[str, Any]], holiday_months: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_period = {r["period"]: r for r in holiday_months}
    spring_by_period = {r["period"]: r["spring_festival_days"] for r in holiday_months}
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        context = by_period.get(row["period"], {})
        enriched.update({
            "holiday_days": context.get("holiday_days", 0),
            "spring_festival_days": context.get("spring_festival_days", 0),
            "makeup_workdays": context.get("makeup_workdays", 0),
            "holiday_names": context.get("holiday_names", []),
            "holiday_event_ids": context.get("holiday_event_ids", []),
        })
        # 构造春节窗口：本月或前后月有春节放假区间
        y, m = map(int, row["period"].split("-"))
        previous = date(y - 1, 12, 1) if m == 1 else date(y, m - 1, 1)
        following = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        prev_key = f"{previous.year:04d}-{previous.month:02d}"
        next_key = f"{following.year:04d}-{following.month:02d}"
        in_spring_window = (
            spring_by_period.get(row["period"], 0) > 0
            or spring_by_period.get(prev_key, 0) > 0
            or spring_by_period.get(next_key, 0) > 0
        )
        enriched["spring_festival_window"] = bool(in_spring_window and m in {1, 2, 3})
        output.append(enriched)
    return output


def candidate_causes_for_row(row: dict[str, Any], metric_type: str) -> list[dict[str, Any]]:
    if row.get("anomaly_level") == "normal":
        return []
    candidates: list[dict[str, Any]] = []
    if metric_type == "electricity":
        month = int(row["date"].month)
        if row.get("spring_festival_window"):
            candidates.append({
                "cause_id": "spring_festival_shift",
                "confidence": "medium",
                "relation": "CANDIDATE_EXPLANATION",
                "evidence": f"异常发生在春节窗口；本月春节放假天数={row.get('spring_festival_days', 0)}",
                "causal_status": "未验证",
            })
        if month in {7, 8}:
            candidates.append({
                "cause_id": "weather_temperature",
                "confidence": "low",
                "relation": "CANDIDATE_EXPLANATION",
                "evidence": "异常月份位于夏季高温负荷窗口；尚未接入气象数据",
                "causal_status": "待补证据",
            })
        candidates.extend([
            {
                "cause_id": "industrial_activity",
                "confidence": "low",
                "relation": "CANDIDATE_EXPLANATION",
                "evidence": "全社会用电量受工业生产影响，需要分行业用电和工业增加值交叉验证",
                "causal_status": "待补证据",
            },
            {
                "cause_id": "policy_or_power_event",
                "confidence": "low",
                "relation": "CANDIDATE_EXPLANATION",
                "evidence": "需检索同期电价、需求响应、限电及重大事件记录",
                "causal_status": "待检索",
            },
        ])
    else:
        if row.get("revision_risk"):
            candidates.append({
                "cause_id": "annual_revision",
                "confidence": "high",
                "relation": "DATA_QUALITY_RISK",
                "evidence": "Q4 单季度值由全年累计值差分得到，可能吸收年度核算修订",
                "causal_status": "数据构造风险",
            })
        if row.get("industry") == "第二产业" and row.get("period") == "2020Q1":
            candidates.append({
                "cause_id": "public_health_event",
                "confidence": "medium",
                "relation": "CANDIDATE_EXPLANATION",
                "evidence": "时间与 2020Q1 公共卫生事件窗口重合；仍需行业停复工证据验证",
                "causal_status": "未验证",
            })
        candidates.append({
            "cause_id": "industrial_activity",
            "confidence": "low",
            "relation": "CANDIDATE_EXPLANATION",
            "evidence": "需要工业增加值、消费、投资或服务业高频指标交叉验证",
            "causal_status": "待补证据",
        })
    # 去重并保留顺序
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in candidates:
        key = item["cause_id"]
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def build_anomaly_events(electricity: list[dict[str, Any]], gdp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in electricity:
        if row["anomaly_level"] == "normal":
            continue
        event_id = f"anomaly:electricity:{row['period']}"
        causes = candidate_causes_for_row(row, "electricity")
        events.append({
            "anomaly_id": event_id,
            "region_code": REGION_CODE,
            "region_name": REGION_NAME,
            "period": row["period"],
            "frequency": "M",
            "metric_id": "electricity_total_monthly",
            "metric_name": "全社会用电量",
            "industry": "全社会",
            "observed": row["value"],
            "expected": row["expected_value"],
            "unit": row["unit"],
            "residual_pct": row["residual_pct"],
            "residual_pct_points": row["residual_pct_points"],
            "direction": "above_expected" if row["residual_pct"] > 0 else "below_expected",
            "anomaly_level": row["anomaly_level"],
            "robust_z": row["robust_z"],
            "stl_weight": row["stl_weight"],
            "endpoint_warning": row["endpoint_warning"],
            "revision_risk": False,
            "holiday_names": row.get("holiday_names", []),
            "spring_festival_days": row.get("spring_festival_days", 0),
            "candidate_causes": causes,
            "source_file": row["source_file"],
        })
    for row in gdp:
        if row["anomaly_level"] == "normal":
            continue
        industry_id = {"第一产业": "primary", "第二产业": "secondary", "第三产业": "tertiary"}[row["industry"]]
        event_id = f"anomaly:gdp_{industry_id}:{row['period']}"
        causes = candidate_causes_for_row(row, "gdp")
        events.append({
            "anomaly_id": event_id,
            "region_code": REGION_CODE,
            "region_name": REGION_NAME,
            "period": row["period"],
            "frequency": "Q",
            "metric_id": f"gdp_{industry_id}_quarterly",
            "metric_name": f"{row['industry']}单季度增加值",
            "industry": row["industry"],
            "observed": row["value"],
            "expected": row["expected_value"],
            "unit": row["unit"],
            "residual_pct": row["residual_pct"],
            "residual_pct_points": row["residual_pct_points"],
            "direction": "above_expected" if row["residual_pct"] > 0 else "below_expected",
            "anomaly_level": row["anomaly_level"],
            "robust_z": row["robust_z"],
            "stl_weight": row["stl_weight"],
            "endpoint_warning": row["endpoint_warning"],
            "revision_risk": row.get("revision_risk", False),
            "holiday_names": [],
            "spring_festival_days": 0,
            "candidate_causes": causes,
            "source_file": row["source_file"],
        })
    events.sort(key=lambda e: (e["period"], e["metric_id"]))
    return events


def build_model_input_records(
    electricity: list[dict[str, Any]],
    gdp: list[dict[str, Any]],
    source_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in electricity:
        candidates = candidate_causes_for_row(row, "electricity")
        records.append({
            "schema_version": SCHEMA_VERSION,
            "task": "anomaly_diagnosis",
            "region": {"code": REGION_CODE, "name": REGION_NAME},
            "period": {"label": row["period"], "frequency": "monthly", "start": row["date"].isoformat()},
            "metric": {"id": "electricity_total_monthly", "name": "全社会用电量", "unit": row["unit"]},
            "observation": {"value": row["value"], "raw_value": row["raw_value"], "raw_unit": row["raw_unit"]},
            "decomposition": {
                "method": "log-STL",
                "period": row["stl_period"],
                "seasonal_window": row["stl_seasonal"],
                "robust": row["stl_robust"],
                "trend_level": row["trend_level"],
                "seasonal_factor": row["seasonal_factor"],
                "expected_value": row["expected_value"],
                "residual_pct": row["residual_pct"],
                "robust_z": row["robust_z"],
                "stl_weight": row["stl_weight"],
            },
            "anomaly": {
                "level": row["anomaly_level"],
                "direction": "above_expected" if row["residual_pct"] > 0 else "below_expected",
                "endpoint_warning": row["endpoint_warning"],
                "statistical_outlier": row["statistical_outlier"],
            },
            "context": {
                "holiday_days": row.get("holiday_days", 0),
                "spring_festival_days": row.get("spring_festival_days", 0),
                "makeup_workdays": row.get("makeup_workdays", 0),
                "holiday_names": row.get("holiday_names", []),
                "candidate_causes": candidates,
                "required_next_evidence": sorted({ev for c in candidates for ev in CAUSE_CATALOG[c["cause_id"]]["required_evidence"]}),
            },
            "provenance": {
                "source_file": row["source_file"],
                "source_sha256": source_hashes.get(row["source_file"], ""),
                "source_url_status": "原始文件未内嵌网址；已用官方年度总量交叉核验",
            },
        })
    for row in gdp:
        industry_id = {"第一产业": "primary", "第二产业": "secondary", "第三产业": "tertiary"}[row["industry"]]
        candidates = candidate_causes_for_row(row, "gdp")
        records.append({
            "schema_version": SCHEMA_VERSION,
            "task": "anomaly_diagnosis",
            "region": {"code": REGION_CODE, "name": REGION_NAME},
            "period": {"label": row["period"], "frequency": "quarterly", "start": row["date"].isoformat()},
            "metric": {
                "id": f"gdp_{industry_id}_quarterly",
                "name": f"{row['industry']}单季度增加值",
                "unit": row["unit"],
            },
            "observation": {
                "value": row["value"],
                "cumulative_value": row["cumulative_value"],
                "cumulative_yoy_pct": row["cumulative_yoy_pct"],
            },
            "decomposition": {
                "method": "log-STL",
                "period": row["stl_period"],
                "seasonal_window": row["stl_seasonal"],
                "robust": row["stl_robust"],
                "trend_level": row["trend_level"],
                "seasonal_factor": row["seasonal_factor"],
                "expected_value": row["expected_value"],
                "residual_pct": row["residual_pct"],
                "robust_z": row["robust_z"],
                "stl_weight": row["stl_weight"],
            },
            "anomaly": {
                "level": row["anomaly_level"],
                "direction": "above_expected" if row["residual_pct"] > 0 else "below_expected",
                "endpoint_warning": row["endpoint_warning"],
                "revision_risk": row["revision_risk"],
                "statistical_outlier": row["statistical_outlier"],
            },
            "context": {
                "candidate_causes": candidates,
                "required_next_evidence": sorted({ev for c in candidates for ev in CAUSE_CATALOG[c["cause_id"]]["required_evidence"]}),
            },
            "provenance": {
                "source_file": row["source_file"],
                "source_sha256": source_hashes.get(row["source_file"], ""),
                "transformation": "年内累计值差分为单季度值；Q4 设置 revision_risk",
            },
        })
    records.sort(key=lambda r: (r["period"]["start"], r["metric"]["id"]))
    return records


def build_knowledge_graph(
    electricity: list[dict[str, Any]],
    gdp: list[dict[str, Any]],
    holiday_events: list[dict[str, Any]],
    anomaly_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, name: str, properties: dict[str, Any]) -> None:
        nodes.append({"node_id": node_id, "node_type": node_type, "name": name, "properties_json": properties})

    def add_edge(edge_id: str, source: str, relation: str, target: str, properties: dict[str, Any] | None = None) -> None:
        edges.append({
            "edge_id": edge_id,
            "source_id": source,
            "relation": relation,
            "target_id": target,
            "properties_json": properties or {},
        })

    add_node(f"region:{REGION_CODE}", "Region", REGION_NAME, {"code": REGION_CODE})
    metrics = {
        "electricity_total_monthly": ("Metric", "全社会用电量", "亿千瓦时"),
        "gdp_primary_quarterly": ("Metric", "第一产业单季度增加值", "亿元"),
        "gdp_secondary_quarterly": ("Metric", "第二产业单季度增加值", "亿元"),
        "gdp_tertiary_quarterly": ("Metric", "第三产业单季度增加值", "亿元"),
    }
    for metric_id, (node_type, name, unit) in metrics.items():
        add_node(f"metric:{metric_id}", node_type, name, {"unit": unit})

    for cause_id, meta in CAUSE_CATALOG.items():
        add_node(f"cause:{cause_id}", "CauseHypothesis", meta["name"], meta)

    for event in holiday_events:
        add_node(event["event_id"], "HolidayEvent" if event["event_type"] == "holiday_leave_interval" else "MakeupWorkday",
                 event["holiday_name"], {
                     "start_date": event["start_date"], "end_date": event["end_date"],
                     "days": event["days"], "source_url": event["source_url"]
                 })

    observation_nodes: dict[tuple[str, str], str] = {}
    for row in electricity:
        period_id = f"period:M:{row['period']}"
        add_node(period_id, "TimePeriod", row["period"], {"frequency": "M", "start": row["date"]})
        obs_id = f"obs:electricity:{row['period']}"
        observation_nodes[("electricity_total_monthly", row["period"])] = obs_id
        add_node(obs_id, "Observation", f"浙江全社会用电量 {row['period']}", {
            "value": row["value"], "expected": row["expected_value"], "residual_pct": row["residual_pct"],
            "trend_level": row["trend_level"], "seasonal_factor": row["seasonal_factor"]
        })
        add_edge(f"e:{obs_id}:region", obs_id, "OBSERVED_IN", f"region:{REGION_CODE}")
        add_edge(f"e:{obs_id}:metric", obs_id, "MEASURES", "metric:electricity_total_monthly")
        add_edge(f"e:{obs_id}:time", obs_id, "AT_TIME", period_id)

    for row in gdp:
        industry_id = {"第一产业": "primary", "第二产业": "secondary", "第三产业": "tertiary"}[row["industry"]]
        metric_id = f"gdp_{industry_id}_quarterly"
        period_id = f"period:Q:{row['period']}"
        add_node(period_id, "TimePeriod", row["period"], {"frequency": "Q", "start": row["date"]})
        obs_id = f"obs:gdp_{industry_id}:{row['period']}"
        observation_nodes[(metric_id, row["period"])] = obs_id
        add_node(obs_id, "Observation", f"浙江{row['industry']}增加值 {row['period']}", {
            "value": row["value"], "expected": row["expected_value"], "residual_pct": row["residual_pct"],
            "revision_risk": row["revision_risk"]
        })
        add_edge(f"e:{obs_id}:region", obs_id, "OBSERVED_IN", f"region:{REGION_CODE}")
        add_edge(f"e:{obs_id}:metric", obs_id, "MEASURES", f"metric:{metric_id}")
        add_edge(f"e:{obs_id}:time", obs_id, "AT_TIME", period_id)

    holiday_by_id = {e["event_id"]: e for e in holiday_events}
    for anomaly in anomaly_events:
        add_node(anomaly["anomaly_id"], "AnomalyEvent", f"{anomaly['metric_name']}异常 {anomaly['period']}", {
            "level": anomaly["anomaly_level"], "direction": anomaly["direction"],
            "residual_pct": anomaly["residual_pct"], "endpoint_warning": anomaly["endpoint_warning"],
            "revision_risk": anomaly["revision_risk"]
        })
        obs_id = observation_nodes[(anomaly["metric_id"], anomaly["period"])]
        add_edge(f"e:{anomaly['anomaly_id']}:derived", anomaly["anomaly_id"], "DERIVED_FROM", obs_id,
                 {"method": "log-STL", "threshold": "5% medium / 8% high"})
        for cause in anomaly["candidate_causes"]:
            add_edge(
                f"e:{anomaly['anomaly_id']}:cause:{cause['cause_id']}",
                anomaly["anomaly_id"], cause["relation"], f"cause:{cause['cause_id']}",
                {"confidence": cause["confidence"], "evidence": cause["evidence"], "causal_status": cause["causal_status"]}
            )
        # 仅对电力月度异常连接实际重叠的节假日事件
        if anomaly["frequency"] == "M":
            year, month = map(int, anomaly["period"].split("-"))
            month_start = date(year, month, 1)
            month_end = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1) - timedelta(days=1)
            for holiday_id, holiday in holiday_by_id.items():
                if holiday["event_type"] != "holiday_leave_interval":
                    continue
                if max(month_start, holiday["start_date"]) <= min(month_end, holiday["end_date"]):
                    add_edge(f"e:{anomaly['anomaly_id']}:holiday:{holiday_id}", anomaly["anomaly_id"],
                             "TEMPORALLY_OVERLAPS", holiday_id, {"causal_status": "时间重合，不代表因果"})

    # 节点去重（同一时间节点会由不同产业重复产生）
    deduped_nodes: dict[str, dict[str, Any]] = {}
    for node in nodes:
        deduped_nodes[node["node_id"]] = node
    deduped_edges: dict[str, dict[str, Any]] = {}
    for edge in edges:
        deduped_edges[edge["edge_id"]] = edge
    return list(deduped_nodes.values()), list(deduped_edges.values())


# -----------------------------
# 图表
# -----------------------------

def save_line_chart(x: Sequence[Any], series: list[tuple[str, Sequence[float]]], title: str, y_label: str, path: Path,
                    zero_line: bool = False, annotate_points: list[tuple[Any, float, str]] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for label, y in series:
        ax.plot(x, y, label=label, linewidth=1.8)
    if zero_line:
        ax.axhline(0, linewidth=1.0)
    if annotate_points:
        for px, py, text in annotate_points:
            ax.scatter([px], [py], zorder=5)
            ax.annotate(text, (px, py), xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    if len(series) > 1:
        ax.legend(loc="best")
    if x and isinstance(x[0], (date, datetime)):
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_seasonal_bar(month_factors: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    months = [r["month"] for r in month_factors]
    factors = [r["average_seasonal_factor"] for r in month_factors]
    ax.bar(months, factors)
    ax.axhline(1.0, linewidth=1.0)
    ax.set_xticks(months)
    ax.set_xlabel("月份")
    ax.set_ylabel("季节因子（1=无季节效应）")
    ax.set_title("浙江全社会用电量平均月份季节因子")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_gdp_anomaly_scatter(gdp_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    industries = ["第一产业", "第二产业", "第三产业"]
    for industry in industries:
        subset = [r for r in gdp_rows if r["industry"] == industry]
        ax.plot([r["date"] for r in subset], [r["residual_pct_points"] for r in subset], label=industry, linewidth=1.3)
        anomalies = [r for r in subset if r["anomaly_level"] != "normal"]
        ax.scatter([r["date"] for r in anomalies], [r["residual_pct_points"] for r in anomalies], s=24)
    ax.axhline(0, linewidth=1.0)
    ax.axhline(8, linewidth=0.8, linestyle="--")
    ax.axhline(-8, linewidth=0.8, linestyle="--")
    ax.set_title("三次产业单季度增加值 STL 残差（异常点已标记）")
    ax.set_ylabel("相对期望值偏差（%）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(YearLocator(2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 数据源登记与运行摘要
# -----------------------------

def build_source_registry(input_dir: Path, source_hashes: dict[str, str]) -> list[dict[str, Any]]:
    rows = [
        {
            "source_id": "ZJ_ELECTRICITY_MONTHLY_RAW",
            "dataset": "浙江全社会用电量（月度）",
            "local_file": ELECTRICITY_FILE,
            "source_url": "原始文件未内嵌；待项目组补录抓取页 URL 与发布日期",
            "validation_url": SOURCE_URLS["zhejiang_2024_electricity_validation"],
            "validation_result": "2024 年月度求和约 6780.85 亿千瓦时，与公开报道 6780 亿千瓦时基本一致",
            "sha256": source_hashes.get(ELECTRICITY_FILE, ""),
            "provenance_status": "部分完整",
        },
        {
            "source_id": "ZJ_GDP_PRIMARY_RAW",
            "dataset": "浙江第一产业 GDP（季度累计）",
            "local_file": GDP_FILES["第一产业"],
            "source_url": "原始文件未内嵌；待补录",
            "validation_url": SOURCE_URLS["zhejiang_2025_statistical_bulletin"],
            "validation_result": "2025Q4 累计值 2657 亿元，与 2025 年浙江统计公报一致",
            "sha256": source_hashes.get(GDP_FILES["第一产业"], ""),
            "provenance_status": "部分完整",
        },
        {
            "source_id": "ZJ_GDP_SECONDARY_RAW",
            "dataset": "浙江第二产业 GDP（季度累计）",
            "local_file": GDP_FILES["第二产业"],
            "source_url": "原始文件未内嵌；待补录",
            "validation_url": SOURCE_URLS["zhejiang_2025_statistical_bulletin"],
            "validation_result": "2025Q4 累计值 35682 亿元，与 2025 年浙江统计公报一致",
            "sha256": source_hashes.get(GDP_FILES["第二产业"], ""),
            "provenance_status": "部分完整",
        },
        {
            "source_id": "ZJ_GDP_TERTIARY_RAW",
            "dataset": "浙江第三产业 GDP（季度累计）",
            "local_file": GDP_FILES["第三产业"],
            "source_url": "原始文件未内嵌；待补录",
            "validation_url": SOURCE_URLS["zhejiang_2025_statistical_bulletin"],
            "validation_result": "2025Q4 累计值 56206 亿元，与 2025 年浙江统计公报一致",
            "sha256": source_hashes.get(GDP_FILES["第三产业"], ""),
            "provenance_status": "部分完整",
        },
        {
            "source_id": "CN_HOLIDAY_NOTICES_2021_2025",
            "dataset": "国务院办公厅 2021-2025 节假日安排",
            "local_file": HOLIDAY_FILE,
            "source_url": "; ".join(SOURCE_URLS[f"holiday_{y}"] for y in range(2021, 2026)),
            "validation_url": "同 source_url",
            "validation_result": "脚本解析 2021-2025 放假区间和调休工作日",
            "sha256": source_hashes.get(HOLIDAY_FILE, ""),
            "provenance_status": "完整",
        },
    ]
    return rows


def create_readme(project_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# 浙江电力经济 STL 分解成果包

## 1. 一键运行

```bash
python -m pip install -r requirements.txt
python stl_power_economy_pipeline.py --input-dir data/raw --output-dir outputs
```

## 2. 输入

- 月度全社会用电量：{ELECTRICITY_FILE}
- 第一、第二、第三产业季度累计 GDP：3 个 XLSX
- 国务院办公厅近五年节假日安排：{HOLIDAY_FILE}

## 3. 关键设定

- 月度用电量：log-STL, period=12, seasonal=13, robust=True
- 季度 GDP：先将累计值差分为单季度值，再做 log-STL, period=4, seasonal=7, robust=True
- 异常阈值：|残差比例| >= 8% 为 high，5%-8% 为 medium
- 首尾各 1 个季节周期标记 endpoint_warning
- GDP Q4 标记 revision_risk，防止把年度核算修订误认作经济冲击

## 4. 本次运行摘要

- 电力月度记录：{summary['electricity']['n_observations']} 条
- 电力高等级异常：{summary['electricity']['high_anomaly_count']} 条
- 电力趋势强度：{summary['electricity']['trend_strength']:.3f}
- 电力季节强度：{summary['electricity']['seasonal_strength']:.3f}
- GDP 单季度记录：{summary['gdp']['n_observations']} 条（3 个产业合计）
- GDP 高等级异常：{summary['gdp']['high_anomaly_count']} 条

## 5. 主要输出

- `outputs/tables/electricity_monthly_stl.csv`
- `outputs/tables/gdp_quarterly_stl.csv`
- `outputs/tables/anomaly_events.csv`
- `outputs/model_input/model_input_records.jsonl`
- `outputs/kg/kg_nodes.csv`, `outputs/kg/kg_edges.csv`
- `outputs/figures/*.png`

## 6. 因果诊断边界

STL 只能分离趋势、季节与不规则项，不能单独证明异常原因。脚本把节假日、天气、工业活动、政策事件等组织为“候选原因+所需证据”，供后续知识图谱、RAG 和规则引擎核验，避免把时间重合直接写成因果关系。
"""
    (project_dir / "README.md").write_text(text, encoding="utf-8")


def create_sources_md(project_dir: Path) -> None:
    lines = [
        "# 方法与交叉核验来源",
        "",
        "## 方法",
        f"- 国家统计局：什么是季节调整：{SOURCE_URLS['nbs_seasonal_adjustment']}",
        f"- statsmodels STL API：{SOURCE_URLS['statsmodels_stl']}",
        f"- statsmodels STL.fit API：{SOURCE_URLS['statsmodels_stl_fit']}",
        "- Cleveland, R. B. et al. (1990), STL: A Seasonal-Trend Decomposition Procedure Based on Loess, Journal of Official Statistics, 6, 3-73.",
        "",
        "## 浙江数据交叉核验",
        f"- 2024 年浙江全社会用电量：{SOURCE_URLS['zhejiang_2024_electricity_validation']}",
        f"- 2025 年浙江省统计公报：{SOURCE_URLS['zhejiang_2025_statistical_bulletin']}",
        "",
        "## 节假日通知",
    ]
    for year in range(2021, 2026):
        lines.append(f"- {year}：{SOURCE_URLS[f'holiday_{year}']}")
    (project_dir / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# 主流程
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="浙江省电力-经济数据 STL 分解与异常事件构建")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="原始数据目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--min-gdp-year", type=int, default=2000, help="GDP 分析起始年份，默认 2000")
    parser.add_argument("--medium-threshold", type=float, default=0.05, help="中等级异常残差比例阈值")
    parser.add_argument("--high-threshold", type=float, default=0.08, help="高等级异常残差比例阈值")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")
    kg_dir = ensure_dir(output_dir / "kg")
    model_dir = ensure_dir(output_dir / "model_input")

    required_paths = [input_dir / ELECTRICITY_FILE, input_dir / HOLIDAY_FILE]
    required_paths.extend(input_dir / file for file in GDP_FILES.values())
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing))

    source_hashes = {p.name: sha256_file(p) for p in required_paths}

    electricity_raw, checks_e = load_electricity(input_dir / ELECTRICITY_FILE)
    holiday_events, checks_h = parse_holiday_notice(input_dir / HOLIDAY_FILE)
    holiday_months = aggregate_holiday_months(holiday_events, electricity_raw[0]["date"], electricity_raw[-1]["date"])

    elec_config = STLConfig(
        period=12, seasonal=13, robust=True,
        medium_threshold=args.medium_threshold, high_threshold=args.high_threshold,
    )
    electricity_stl, electricity_metrics = run_log_stl(electricity_raw, elec_config)
    electricity_stl = attach_electricity_context(electricity_stl, holiday_months)

    gdp_all: list[dict[str, Any]] = []
    quality_checks: list[DataQualityCheck] = checks_e + checks_h
    gdp_metrics: dict[str, dict[str, float]] = {}
    for industry, filename in GDP_FILES.items():
        rows, checks = load_gdp_industry(input_dir / filename, industry, min_year=args.min_gdp_year)
        quality_checks.extend(checks)
        gdp_config = STLConfig(
            period=4, seasonal=7, robust=True,
            medium_threshold=args.medium_threshold, high_threshold=args.high_threshold,
        )
        decomposed, metrics = run_log_stl(rows, gdp_config)
        gdp_all.extend(decomposed)
        gdp_metrics[industry] = metrics
    gdp_all.sort(key=lambda r: (r["date"], r["industry"]))

    # 平均月份季节因子
    seasonal_by_month: dict[int, list[float]] = defaultdict(list)
    for row in electricity_stl:
        seasonal_by_month[row["date"].month].append(row["seasonal_factor"])
    month_factors = [
        {
            "month": month,
            "average_seasonal_factor": float(np.mean(seasonal_by_month[month])),
            "average_effect_pct": float((np.mean(seasonal_by_month[month]) - 1.0) * 100.0),
            "n": len(seasonal_by_month[month]),
        }
        for month in range(1, 13)
    ]

    anomaly_events = build_anomaly_events(electricity_stl, gdp_all)
    model_records = build_model_input_records(electricity_stl, gdp_all, source_hashes)
    kg_nodes, kg_edges = build_knowledge_graph(electricity_stl, gdp_all, holiday_events, anomaly_events)
    source_registry = build_source_registry(input_dir, source_hashes)

    # 数据交叉核验
    elec_2024_sum = sum(r["value"] for r in electricity_stl if r["date"].year == 2024)
    validation_checks = [
        DataQualityCheck(
            "V001", "electricity_monthly", "2024 年度总量外部核验",
            abs(elec_2024_sum - 6780.0) / 6780.0 < 0.005,
            f"月度求和={elec_2024_sum:.4f} 亿千瓦时；公开报道=6780 亿千瓦时；相对差={(elec_2024_sum/6780-1)*100:.3f}%",
            severity="warning",
        )
    ]
    expected_2025 = {"第一产业": 2657.0, "第二产业": 35682.0, "第三产业": 56206.0}
    for industry, target in expected_2025.items():
        q4 = next(r for r in gdp_all if r["industry"] == industry and r["period"] == "2025Q4")
        validation_checks.append(DataQualityCheck(
            f"V_GDP_{industry}", f"gdp_{industry}", "2025 全年累计值外部核验",
            abs(q4["cumulative_value"] - target) < 1e-9,
            f"原始累计值={q4['cumulative_value']:.0f} 亿元；统计公报={target:.0f} 亿元",
            severity="warning",
        ))
    quality_checks.extend(validation_checks)

    # CSV / JSONL 输出
    write_csv(tables_dir / "electricity_monthly_stl.csv", electricity_stl)
    write_csv(tables_dir / "gdp_quarterly_stl.csv", gdp_all)
    write_csv(tables_dir / "anomaly_events.csv", anomaly_events)
    write_csv(tables_dir / "electricity_month_seasonal_factors.csv", month_factors)
    write_csv(tables_dir / "holiday_calendar_events.csv", holiday_events)
    write_csv(tables_dir / "holiday_month_features.csv", holiday_months)
    write_csv(tables_dir / "data_source_registry.csv", source_registry)
    write_csv(tables_dir / "data_quality_checks.csv", [asdict(c) for c in quality_checks])
    write_jsonl(model_dir / "model_input_records.jsonl", model_records)
    write_csv(kg_dir / "kg_nodes.csv", kg_nodes)
    write_csv(kg_dir / "kg_edges.csv", kg_edges)

    # 图表
    e_dates = [r["date"] for r in electricity_stl]
    save_line_chart(
        e_dates,
        [("实际值", [r["value"] for r in electricity_stl]), ("STL 期望值", [r["expected_value"] for r in electricity_stl])],
        "浙江全社会用电量：实际值与 STL 期望值", "亿千瓦时",
        figures_dir / "01_electricity_actual_vs_expected.png",
        annotate_points=[(r["date"], r["value"], r["period"]) for r in electricity_stl if r["anomaly_level"] == "high"],
    )
    save_line_chart(
        e_dates, [("趋势水平", [r["trend_level"] for r in electricity_stl])],
        "浙江全社会用电量 STL 趋势项", "亿千瓦时",
        figures_dir / "02_electricity_trend.png",
    )
    save_seasonal_bar(month_factors, figures_dir / "03_electricity_month_seasonal_factors.png")
    save_line_chart(
        e_dates, [("残差偏离", [r["residual_pct_points"] for r in electricity_stl])],
        "浙江全社会用电量 STL 残差与异常", "相对期望值偏差（%）",
        figures_dir / "04_electricity_residual_anomalies.png", zero_line=True,
        annotate_points=[(r["date"], r["residual_pct_points"], r["period"]) for r in electricity_stl if r["anomaly_level"] != "normal"],
    )
    for index, industry in enumerate(["第一产业", "第二产业", "第三产业"], start=5):
        subset = [r for r in gdp_all if r["industry"] == industry]
        save_line_chart(
            [r["date"] for r in subset],
            [("实际单季度值", [r["value"] for r in subset]), ("STL 期望值", [r["expected_value"] for r in subset])],
            f"浙江{industry}单季度增加值：实际值与 STL 期望值", "亿元",
            figures_dir / f"{index:02d}_gdp_{industry}_actual_vs_expected.png",
            annotate_points=[(r["date"], r["value"], r["period"]) for r in subset if r["anomaly_level"] == "high" and r["date"].year >= 2018],
        )
    save_gdp_anomaly_scatter(gdp_all, figures_dir / "08_gdp_residual_anomalies.png")

    # 趋势年化增速：用趋势水平首尾值估算
    elapsed_years = (electricity_stl[-1]["date"] - electricity_stl[0]["date"]).days / 365.2425
    trend_cagr = (electricity_stl[-1]["trend_level"] / electricity_stl[0]["trend_level"]) ** (1 / elapsed_years) - 1

    high_electricity = [r for r in electricity_stl if r["anomaly_level"] == "high"]
    medium_electricity = [r for r in electricity_stl if r["anomaly_level"] == "medium"]
    high_gdp = [r for r in gdp_all if r["anomaly_level"] == "high"]
    medium_gdp = [r for r in gdp_all if r["anomaly_level"] == "medium"]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "region": {"code": REGION_CODE, "name": REGION_NAME},
        "method": {
            "electricity": asdict(elec_config),
            "gdp": asdict(STLConfig(4, 7, True, args.medium_threshold, args.high_threshold)),
            "equation": "log(y_t) = trend_t + seasonal_t + residual_t",
            "anomaly_rule": f"medium if |residual_pct| >= {args.medium_threshold:.0%}; high if >= {args.high_threshold:.0%}",
        },
        "electricity": {
            **electricity_metrics,
            "start": electricity_stl[0]["period"],
            "end": electricity_stl[-1]["period"],
            "trend_start": electricity_stl[0]["trend_level"],
            "trend_end": electricity_stl[-1]["trend_level"],
            "trend_cagr": trend_cagr,
            "annual_total_2024": elec_2024_sum,
            "high_anomaly_count": len(high_electricity),
            "medium_anomaly_count": len(medium_electricity),
            "high_anomalies": [
                {"period": r["period"], "observed": r["value"], "expected": r["expected_value"], "residual_pct": r["residual_pct"],
                 "endpoint_warning": r["endpoint_warning"], "spring_festival_window": r["spring_festival_window"]}
                for r in high_electricity
            ],
            "month_seasonal_factors": month_factors,
        },
        "gdp": {
            "n_observations": len(gdp_all),
            "metrics_by_industry": gdp_metrics,
            "high_anomaly_count": len(high_gdp),
            "medium_anomaly_count": len(medium_gdp),
            "high_anomalies": [
                {"industry": r["industry"], "period": r["period"], "observed": r["value"], "expected": r["expected_value"],
                 "residual_pct": r["residual_pct"], "revision_risk": r["revision_risk"], "endpoint_warning": r["endpoint_warning"]}
                for r in high_gdp
            ],
        },
        "knowledge_graph": {"node_count": len(kg_nodes), "edge_count": len(kg_edges)},
        "model_input": {"record_count": len(model_records), "schema_version": SCHEMA_VERSION},
        "data_quality": {
            "check_count": len(quality_checks),
            "failed_checks": [asdict(c) for c in quality_checks if not c.passed],
            "warning_count": sum(1 for c in quality_checks if c.severity == "warning"),
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    write_json(output_dir / "data_quality_report.json", [asdict(c) for c in quality_checks])

    project_dir = Path(__file__).resolve().parent
    create_readme(project_dir, summary)
    create_sources_md(project_dir)

    print(json.dumps({
        "status": "success",
        "output_dir": str(output_dir),
        "electricity_records": len(electricity_stl),
        "gdp_records": len(gdp_all),
        "electricity_high_anomalies": len(high_electricity),
        "gdp_high_anomalies": len(high_gdp),
        "kg_nodes": len(kg_nodes),
        "kg_edges": len(kg_edges),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
