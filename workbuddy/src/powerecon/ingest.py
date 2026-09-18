"""通用数据摄入层 —— "接入任意地区、任意频率"的落点。

这一层要解决的核心问题：浙江包和江苏包的列名完全不同
（浙江 assumed_value_yi_kwh / temperature_mean_c，江苏 consumption_month_yi_kwh / mean_temp_c），
知识图谱的列名也不一样（id/type vs node_id/relationship_type）。
如果每个地区都要改代码，这个产品就做不成 Claude Code。

所以这里全部走"声明 + 嗅探"双轨：
- region.yaml 里声明了就按声明来；
- 没声明（datasets: auto）就自动扫目录、猜日期列、猜数值列、推频率。
两条路产出同一种内部结构，上层工具完全不感知差异。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .contract import (
    DatasetInfo,
    DatasetKind,
    Frequency,
    RegionInfo,
    RegionInventory,
)
from .series_utils import parse_period

# ---------------------------------------------------------------- 列名启发式

DATE_ALIASES = (
    "date", "month", "period", "time", "datetime", "day", "quarter", "year",
    "月份", "日期", "时间", "年月", "季度", "年份",
)

VALUE_PREFERRED = (
    "consumption", "用电", "value_", "_value", "demand",
)

VALUE_HINTS = (
    "kwh", "generation", "temp", "capacity", "gdp", "yuan", "degree",
    "precipitation", "radiation", "load",
)

# 累计值列是最容易选错的坑：江苏表里 consumption_ytd_yi_kwh（年内累计）
# 和 consumption_month_yi_kwh（当月）并存，累计值单调增长、变异系数更大，
# 如果只按变异度排序就会选中累计值 —— 那样后面所有同比和季节分析都是错的。
VALUE_PENALTY = ("ytd", "cumulative", "_cum", "cum_", "累计", "year_to_date")

VALUE_BONUS = ("_month", "month_", "monthly", "当月", "单月")

VALUE_EXCLUDE = (
    "pct", "yoy", "mom", "rate", "flag", "url", "id", "note", "quality",
    "source", "description", "weekday", "day_type", "holiday_name", "is_",
    "name", "label", "unit", "remark",
)

NODE_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "node_id"),
    "name": ("name",),
    "label": ("label", "category"),
    "category": ("category", "label"),
    "description": ("description",),
    "unit": ("unit",),
    "frequency": ("frequency",),
    "source_id": ("source_id", "source_url"),
    "quality_flag": ("quality_flag",),
    "notes": ("notes",),
}

REL_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "relationship_id"),
    "source_id": ("source_id",),
    "target_id": ("target_id",),
    "type": ("type", "relationship_type"),
    "description": ("description",),
    "direction": ("direction",),
    "confidence": ("confidence", "evidence_level"),
    "lag": ("lag",),
    "provenance": ("provenance",),
}

OBS_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "observation_id"),
    "name": ("name",),
    "indicator_id": ("indicator_id",),
    "region_id": ("region_id",),
    "period_id": ("period_id",),
    "value": ("value",),
    "unit": ("unit",),
    "yoy_pct": ("yoy_pct",),
    "frequency": ("frequency",),
    "source_id": ("source_id",),
    "quality_flag": ("quality_flag",),
    "notes": ("notes",),
}


class RegionPackError(RuntimeError):
    """region pack 结构不合法时抛出，附带可操作的修复提示。"""


# ---------------------------------------------------------------- 基础读取


def read_table(path: Path) -> pd.DataFrame:
    """读 CSV/Excel，统一处理 BOM 与编码。"""
    if not path.exists():
        raise RegionPackError(f"数据文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def sniff_date_column(df: pd.DataFrame) -> str | None:
    """猜哪一列是期次。先看列名，再看内容能否解析成日期。"""
    lowered = {c: str(c).strip().lower() for c in df.columns}
    for col, low in lowered.items():
        if low in DATE_ALIASES:
            return col

    best: tuple[float, str] | None = None
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue
        hit = sum(1 for v in sample if parse_period(v) is not None) / len(sample)
        if hit >= 0.8 and (best is None or hit > best[0]):
            best = (hit, col)
    return best[1] if best else None


def sniff_value_column(df: pd.DataFrame, date_col: str | None) -> str | None:
    """猜哪一列是主数值列。

    打分维度（按重要性排序）：
    1. 语义优先级：consumption / 用电 / value_ 这类"我们想要的量"加权重分；
    2. 单位后缀：kwh / yuan / temp 等说明它是真实测量量；
    3. 当月值优先于累计值：累计列扣分，月内列加分 —— 这是最容易选错的地方；
    4. 变异度：数值本身要有波动，常数列直接排除。

    返回值可能是 None，调用方必须处理这种情况。
    """
    scored: list[tuple[float, str]] = []
    for col in df.columns:
        if col == date_col:
            continue
        low = str(col).strip().lower()
        if any(tok in low for tok in VALUE_EXCLUDE):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < max(3, len(df) * 0.5):
            continue
        std = float(series.std(skipna=True) or 0.0)
        if std == 0.0:
            continue
        mean = abs(float(series.mean(skipna=True)) or 0.0)
        cv = std / mean if mean > 1e-9 else std

        score = 0.0
        score += 3.0 * sum(1 for tok in VALUE_PREFERRED if tok in low)
        score += 1.5 * sum(1 for tok in VALUE_HINTS if tok in low)
        score += 3.0 * sum(1 for tok in VALUE_BONUS if tok in low)
        score -= 3.0 * sum(1 for tok in VALUE_PENALTY if tok in low)
        score += min(cv, 5.0) * 0.2
        scored.append((score, col))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def sniff_frequency(periods: list[pd.Timestamp]) -> Frequency:
    """从相邻间隔的中位数推频率。"""
    if len(periods) < 2:
        return Frequency.UNKNOWN
    ordered = sorted(periods)
    gaps = [
        (ordered[i + 1] - ordered[i]).total_seconds() / 86400.0
        for i in range(len(ordered) - 1)
    ]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return Frequency.UNKNOWN
    gaps.sort()
    median = gaps[len(gaps) // 2]
    if median <= 1.5:
        return Frequency.DAILY
    if median <= 3.0:
        return Frequency.DAILY
    if median <= 10.0:
        return Frequency.WEEKLY
    if median <= 45.0:
        return Frequency.MONTHLY
    if median <= 135.0:
        return Frequency.QUARTERLY
    return Frequency.ANNUAL


def normalize_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    group_col: str | None = None,
    unit: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """把任意表压成统一内部结构：[period, metric, industry, value, unit, source]。"""
    out = pd.DataFrame()
    out["period"] = df[date_col].map(parse_period)
    out["metric"] = df[value_col].astype(str) if group_col is None else df[group_col].astype(str)
    out["industry"] = df[group_col].astype(str) if group_col else None
    out["value"] = pd.to_numeric(df[value_col], errors="coerce")
    out["unit"] = unit
    out["source"] = source
    out = out.dropna(subset=["period", "value"])
    return out.sort_values("period").reset_index(drop=True)


# ---------------------------------------------------------------- 知识图谱读取


def _apply_aliases(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """把不同地区的列名映射到统一名字。找不到的列补空，保证下游列集合恒定。"""
    out = pd.DataFrame(index=df.index)
    for canonical, candidates in aliases.items():
        found = next((c for c in candidates if c in df.columns), None)
        out[canonical] = df[found] if found is not None else None
    return out


# ---------------------------------------------------------------- region pack


@dataclass
class RegionContext:
    """一个地区的全部可用资源。工具层从这里取数，不直接碰文件路径。"""

    info: RegionInfo
    pack_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, DatasetInfo] = field(default_factory=dict)
    _frames: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    kg_nodes: pd.DataFrame | None = None
    kg_relationships: pd.DataFrame | None = None
    kg_observations: pd.DataFrame | None = None
    knowledge_docs: list[dict[str, str]] = field(default_factory=list)

    def frame(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id not in self._frames:
            info = self.datasets.get(dataset_id)
            if info is None:
                raise RegionPackError(
                    f"地区 {self.info.name} 没有数据集 {dataset_id}。"
                    f"可用：{sorted(self.datasets)}"
                )
            self._frames[dataset_id] = read_table(self.pack_dir / info.path)
        return self._frames[dataset_id]

    def series(self, dataset_id: str, *, column: str | None = None) -> pd.DataFrame:
        """取一条标准化时间序列。column 可以覆盖 region.yaml 里声明的主数值列。"""
        info = self.datasets[dataset_id]
        df = self.frame(dataset_id)
        cfg = next(
            (d for d in self.config.get("datasets", []) if d.get("id") == dataset_id),
            {},
        ) if isinstance(self.config.get("datasets"), list) else {}

        date_col = cfg.get("date_column") or self._guess_date_col(df)
        value_col = column or cfg.get("value_column") or self._guess_value_col(df, date_col)
        group_col = cfg.get("group_column")
        if date_col is None or value_col is None:
            raise RegionPackError(
                f"数据集 {dataset_id} 无法识别日期列或数值列。"
                f"请在 region.yaml 中显式声明 date_column / value_column。"
                f"实际列：{list(df.columns)}"
            )
        return normalize_series(
            df,
            date_col,
            value_col,
            group_col=group_col,
            unit=info.unit,
            source=info.path,
        )

    @staticmethod
    def _guess_date_col(df: pd.DataFrame) -> str | None:
        return sniff_date_column(df)

    @staticmethod
    def _guess_value_col(df: pd.DataFrame, date_col: str | None) -> str | None:
        return sniff_value_column(df, date_col)


def load_region(pack_dir: str | Path, *, shared_knowledge: str | Path | None = None) -> RegionContext:
    """加载一个 region pack。这是整个系统的入口 —— 换地区就是换这个目录。

    shared_knowledge 指向跨地区共用的领域知识目录（春节错位、因果边界这类
    与地区无关的准则）。它和地区自有知识一起进检索库，避免每个 pack 复制一份。
    """
    pack_dir = Path(pack_dir).resolve()
    cfg_path = pack_dir / "region.yaml"
    if not cfg_path.exists():
        raise RegionPackError(
            f"{pack_dir} 下没有 region.yaml。一个 region pack 最少需要 "
            f"region.yaml（含 region.code / region.name）+ data/ 目录。"
        )
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw_region = config.get("region") or {}
    if not raw_region.get("code") or not raw_region.get("name"):
        raise RegionPackError("region.yaml 缺少 region.code 或 region.name。")

    info = RegionInfo(
        code=raw_region["code"],
        name=raw_region["name"],
        level=raw_region.get("level", "unknown"),
        timezone=raw_region.get("timezone", "Asia/Shanghai"),
        representative_point=raw_region.get("representative_point"),
        caveats=list(config.get("caveats") or []),
        pack_dir=str(pack_dir),
    )

    ctx = RegionContext(info=info, pack_dir=pack_dir, config=config)
    ctx.datasets = _collect_datasets(ctx, config)
    _load_kg(ctx, config)
    _load_knowledge_docs(ctx, config, _resolve_shared_knowledge(pack_dir, shared_knowledge))
    return ctx


def _resolve_shared_knowledge(
    pack_dir: Path, explicit: str | Path | None
) -> Path | None:
    """决定用哪个共享知识目录。

    显式传入优先；否则按标准布局向上找一层 ——
    region_packs/<name>/ 的同级目录 knowledge/ 就是共享知识库。
    这样 load_region 单独调用时也能拿到通用领域准则，不必依赖上层装配代码。
    """
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    candidate = pack_dir.parent.parent / "knowledge"
    return candidate if candidate.is_dir() else None


def _collect_datasets(ctx: RegionContext, config: dict[str, Any]) -> dict[str, DatasetInfo]:
    declared = config.get("datasets")
    result: dict[str, DatasetInfo] = {}

    if isinstance(declared, list):
        for entry in declared:
            ds = _describe_declared(ctx, entry)
            if ds:
                result[ds.id] = ds

    if declared in (None, "auto", []) or config.get("auto_discover", False):
        for ds in _discover_datasets(ctx):
            result.setdefault(ds.id, ds)

    if not result:
        raise RegionPackError(
            f"地区 {ctx.info.name} 没有可用数据集。"
            f"请把 CSV 放到 {ctx.pack_dir / 'data'} 下，或在 region.yaml 里声明 datasets。"
        )

    # overrides：只想改指标名/单位这类展示信息，不想关掉自动发现时用它。
    # 这样"列名嗅探"和"人类可读"可以分开，不必为了改个名字就写死整份声明。
    overrides = config.get("overrides") or {}
    for ds_id, patch in overrides.items():
        ds = result.get(ds_id)
        if ds is None or not isinstance(patch, dict):
            continue
        for key in ("metric", "unit", "industry", "id"):
            if key in patch:
                setattr(ds, key, patch[key])
        if "frequency" in patch:
            ds.frequency = Frequency(patch["frequency"])
    return result


def _describe_declared(ctx: RegionContext, entry: dict[str, Any]) -> DatasetInfo | None:
    rel = entry.get("path")
    if not rel:
        return None
    path = ctx.pack_dir / rel
    if not path.exists():
        return None
    kind = DatasetKind(entry.get("kind", "timeseries"))
    df = read_table(path)
    date_col = entry.get("date_column") or sniff_date_column(df)
    value_col = entry.get("value_column") or sniff_value_column(df, date_col)

    periods: list[pd.Timestamp] = []
    if kind in (DatasetKind.TIMESERIES, DatasetKind.CALENDAR) and date_col:
        periods = [p for p in (parse_period(v) for v in df[date_col]) if p is not None]

    freq = Frequency(entry["frequency"]) if entry.get("frequency") else sniff_frequency(periods)
    flags = sorted({str(v) for v in df.get("quality_flag", pd.Series(dtype=object)).dropna().unique()}) if "quality_flag" in df.columns else []

    return DatasetInfo(
        id=entry.get("id") or path.stem,
        kind=kind,
        path=rel,
        metric=entry.get("metric") or (value_col or path.stem),
        unit=entry.get("unit"),
        frequency=freq,
        industry=entry.get("industry"),
        n_rows=int(len(df)),
        period_start=str(min(periods).date()) if periods else None,
        period_end=str(max(periods).date()) if periods else None,
        columns=[str(c) for c in df.columns],
        quality_flags=flags,
        discovered=False,
    )


def _discover_datasets(ctx: RegionContext) -> list[DatasetInfo]:
    """扫 data/ 下所有 CSV，自动判定类型、日期列、数值列、频率。"""
    data_dir = ctx.pack_dir / "data"
    if not data_dir.is_dir():
        return []
    out: list[DatasetInfo] = []
    for path in sorted(data_dir.glob("*.csv")):
        name = path.stem.lower()
        if name.startswith("kg_"):
            continue
        try:
            df = read_table(path)
        except Exception:
            continue
        if df.empty:
            continue

        date_col = sniff_date_column(df)
        periods = [p for p in (parse_period(v) for v in df[date_col]) if p is not None] if date_col else []
        looks_like_time = bool(date_col) and len(periods) >= max(3, len(df) * 0.7)
        kind = DatasetKind.TIMESERIES if looks_like_time else DatasetKind.TABLE
        if "calendar" in name:
            kind = DatasetKind.CALENDAR

        value_col = sniff_value_column(df, date_col) if kind != DatasetKind.TABLE else None
        freq = sniff_frequency(periods) if looks_like_time else Frequency.UNKNOWN
        flags = sorted({str(v) for v in df["quality_flag"].dropna().unique()}) if "quality_flag" in df.columns else []

        out.append(
            DatasetInfo(
                id=path.stem,
                kind=kind,
                path=str(path.relative_to(ctx.pack_dir)).replace("\\", "/"),
                metric=value_col or path.stem,
                unit=None,
                frequency=freq,
                industry=None,
                n_rows=int(len(df)),
                period_start=str(min(periods).date()) if periods else None,
                period_end=str(max(periods).date()) if periods else None,
                columns=[str(c) for c in df.columns],
                quality_flags=flags,
                discovered=True,
            )
        )
    return out


def _load_kg(ctx: RegionContext, config: dict[str, Any]) -> None:
    """加载知识图谱。同时支持 region.yaml 里的 kg: 段和 datasets 里的 kg 条目。"""
    kg_cfg = config.get("kg")
    if not kg_cfg and isinstance(config.get("datasets"), list):
        kg_cfg = next((d for d in config["datasets"] if d.get("kind") == "knowledge_graph"), None)
    if not kg_cfg:
        return

    def _read(rel: str | None) -> pd.DataFrame | None:
        if not rel:
            return None
        p = ctx.pack_dir / rel
        return read_table(p) if p.exists() else None

    nodes = _read(kg_cfg.get("nodes"))
    rels = _read(kg_cfg.get("relationships"))
    obs = _read(kg_cfg.get("observations"))

    if nodes is not None:
        ctx.kg_nodes = _apply_aliases(nodes, NODE_ALIASES)
    if rels is not None:
        ctx.kg_relationships = _apply_aliases(rels, REL_ALIASES)
    if obs is not None:
        ctx.kg_observations = _apply_aliases(obs, OBS_ALIASES)


def _load_knowledge_docs(
    ctx: RegionContext, config: dict[str, Any], shared_dir: str | Path | None = None
) -> None:
    """把知识文档切块，供关键词检索用。零外部依赖，后续可换向量检索。

    加载顺序：先共享目录（通用领域准则），再地区目录（本地口径说明）。
    地区文档排后面，检索命中时能拿到更具体的上下文。
    """
    docs: list[dict[str, str]] = []
    roots: list[tuple[Path, str]] = []
    if shared_dir:
        roots.append((Path(shared_dir), "shared"))
    for rel in config.get("knowledge_docs") or []:
        roots.append((ctx.pack_dir / rel, "region"))

    for root, scope in roots:
        if root.is_dir():
            files = sorted(list(root.rglob("*.md")) + list(root.rglob("*.txt")))
        elif root.exists() and root.suffix.lower() in (".md", ".txt"):
            files = [root]
        else:
            files = []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            try:
                source = str(f.relative_to(ctx.pack_dir)).replace("\\", "/")
            except ValueError:
                source = f"{scope}:{f.name}"
            for i, chunk in enumerate(_chunk(text)):
                docs.append(
                    {
                        "doc_id": f"{scope}:{f.stem}#{i}",
                        "title": f.stem,
                        "text": chunk,
                        "source": source,
                    }
                )
    ctx.knowledge_docs = docs


def _chunk(text: str, size: int = 320, overlap: int = 60) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def build_inventory(ctx: RegionContext) -> RegionInventory:
    """给 LLM 看的"这个地区有什么"。它靠这个决定下一步调什么工具。"""
    notes: list[str] = []
    if ctx.kg_relationships is not None:
        n_rel = int(len(ctx.kg_relationships))
        notes.append(f"知识图谱含 {n_rel} 条关系、{len(ctx.kg_nodes) if ctx.kg_nodes is not None else 0} 个节点。")
    discovered = [d for d in ctx.datasets.values() if d.discovered]
    if discovered:
        notes.append(
            f"其中 {len(discovered)} 个数据集是自动发现并嗅探出来的（未在 region.yaml 中显式声明），"
            f"列名推断可能存在偏差，关键结论请用 load_series 指定 column 复核。"
        )
    return RegionInventory(
        region=ctx.info,
        datasets=sorted(ctx.datasets.values(), key=lambda d: d.id),
        knowledge_graph_available=ctx.kg_relationships is not None,
        knowledge_doc_count=len(ctx.knowledge_docs),
        notes=notes,
    )


@lru_cache(maxsize=16)
def load_region_cached(pack_dir: str, shared_knowledge: str | None = None) -> RegionContext:
    return load_region(pack_dir, shared_knowledge=shared_knowledge)


def available_packs(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "region.yaml").exists())
