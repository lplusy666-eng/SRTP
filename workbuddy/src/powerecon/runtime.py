"""运行时装配：把 region pack + 产物目录组装成工具上下文。

单独抽出来是因为 CLI、评测脚本、测试三处都要做同一件事，
而且都希望"换地区只改一个参数"。
"""

from __future__ import annotations

from pathlib import Path

from .ingest import RegionPackError, available_packs, load_region
from .tools.base import ToolContext

DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def default_packs_root() -> Path:
    return DEFAULT_ROOT / "region_packs"


def default_artifacts_root() -> Path:
    return DEFAULT_ROOT / "artifacts"


def default_shared_knowledge() -> Path:
    return DEFAULT_ROOT / "knowledge"


def make_context(
    pack: str,
    *,
    packs_root: Path | str | None = None,
    artifacts_root: Path | str | None = None,
    shared_knowledge: Path | str | None = None,
) -> ToolContext:
    """按 pack 名字装配一个工具上下文。pack 可以是目录名，也可以是地区代码。"""
    packs_root = Path(packs_root) if packs_root else default_packs_root()
    artifacts_root = Path(artifacts_root) if artifacts_root else default_artifacts_root()
    shared_knowledge = Path(shared_knowledge) if shared_knowledge else default_shared_knowledge()

    pack_dir = packs_root / pack
    if not (pack_dir / "region.yaml").exists():
        match = None
        for name in available_packs(packs_root):
            try:
                if load_region(packs_root / name).info.code.lower() == pack.lower():
                    match = packs_root / name
                    break
            except RegionPackError:
                continue
        if match is None:
            raise RegionPackError(
                f"找不到地区数据包 '{pack}'。可用：{available_packs(packs_root)}"
            )
        pack_dir = match

    region = load_region(pack_dir, shared_knowledge=shared_knowledge if shared_knowledge.is_dir() else None)
    charts = artifacts_root / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    return ToolContext(region=region, artifacts_dir=charts, packs_root=packs_root)
