"""工具层基座：注册器、schema 导出、调度、给 LLM 的紧凑渲染。

三条设计原则，都是为了让 LLM 真的会用这些工具：

1. **描述即文档。** 模型看不到代码，只看得到 name / description / JSON Schema。
   所以 description 必须写清楚"什么时候该用它、什么时候不该用、返回什么"，
   而不是写"查询数据"这种废话。
2. **schema 自动导出。** 入参用 Pydantic 定义一次，自动转成 OpenAI / Anthropic
   两种 function-calling 格式，杜绝手写 schema 和实现漂移。
3. **返回值必须紧凑。** 51 个点的 STL 结果原样塞进上下文会烧 token 且淹没有效信息，
   所以统一走 render_for_llm 做摘要 + 截断，只把关键统计量给模型，
   需要细节时模型再显式调带 full 参数的工具。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel

from ..ingest import RegionContext

MAX_LIST_ITEMS = 8
MAX_TEXT_CHARS = 4000


@dataclass
class ToolContext:
    """一次运行的全部外部依赖。工具只从这里取东西，不直接碰全局状态。"""

    region: RegionContext
    artifacts_dir: Path
    packs_root: Path
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def cache_get(self, key: str) -> Any:
        return self._cache.get(key)

    def cache_set(self, key: str, value: Any) -> Any:
        self._cache[key] = value
        return value

    def chart_path(self, name: str) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir / name


@dataclass
class ToolResult:
    ok: bool
    tool: str
    arguments: dict[str, Any]
    payload: BaseModel | None = None
    text: str = ""
    error: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "arguments": self.arguments,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[ToolContext, BaseModel], BaseModel]
    returns: str
    needs_region: bool = True
    tags: tuple[str, ...] = ()

    @property
    def input_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


REGISTRY: dict[str, ToolSpec] = {}


def tool(
    *,
    name: str,
    description: str,
    args_model: type[BaseModel],
    returns: str,
    needs_region: bool = True,
    tags: Iterable[str] = (),
) -> Callable:
    """把一个函数注册成工具。description 会被原样送进 LLM 上下文，请认真写。"""

    def decorator(fn: Callable[[ToolContext, BaseModel], BaseModel]) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"工具名重复：{name}")
        REGISTRY[name] = ToolSpec(
            name=name,
            description=description.strip(),
            args_model=args_model,
            fn=fn,
            returns=returns,
            needs_region=needs_region,
            tags=tuple(tags),
        )
        return fn

    return decorator


def get_spec(name: str) -> ToolSpec:
    if name not in REGISTRY:
        raise KeyError(
            f"没有名为 {name} 的工具。可用工具：{sorted(REGISTRY)}"
        )
    return REGISTRY[name]


def list_specs() -> list[ToolSpec]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def openai_tools() -> list[dict[str, Any]]:
    """直接可塞进 OpenAI / DeepSeek / 通义等兼容接口的 tools=[...] 参数。"""
    return [spec.to_openai() for spec in list_specs()]


def anthropic_tools() -> list[dict[str, Any]]:
    """直接可塞进 Anthropic messages API 的 tools=[...] 参数。"""
    return [spec.to_anthropic() for spec in list_specs()]


def tool_catalog_markdown() -> str:
    """给人和模型都能读的工具清单，也是 INTERFACE.md 里那张表的生成源。"""
    lines = ["| 工具 | 用途 | 主要入参 | 返回 |", "|---|---|---|---|"]
    for spec in list_specs():
        fields = list(spec.args_model.model_fields)
        params = ", ".join(f"`{f}`" for f in fields) or "—"
        lines.append(f"| `{spec.name}` | {spec.description.splitlines()[0]} | {params} | {spec.returns} |")
    return "\n".join(lines)


def dispatch(name: str, arguments: dict[str, Any] | None, ctx: ToolContext) -> ToolResult:
    """按名字执行工具。参数校验失败不抛异常，而是把错误文本回给模型让它自己改。"""
    started = time.perf_counter()
    arguments = arguments or {}
    try:
        spec = get_spec(name)
    except KeyError as exc:
        return ToolResult(
            ok=False,
            tool=name,
            arguments=arguments,
            error=str(exc),
            text=f"工具调用失败：{exc}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    try:
        args = spec.args_model.model_validate(arguments)
    except Exception as exc:
        return ToolResult(
            ok=False,
            tool=name,
            arguments=arguments,
            error=f"参数校验失败：{exc}",
            text=(
                f"参数校验失败，请修正后重试。\n"
                f"期望的 schema：{json.dumps(spec.input_schema, ensure_ascii=False)}\n"
                f"收到的参数：{json.dumps(arguments, ensure_ascii=False)}\n"
                f"错误详情：{exc}"
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    try:
        payload = spec.fn(ctx, args)
        return ToolResult(
            ok=True,
            tool=name,
            arguments=arguments,
            payload=payload,
            text=render_for_llm(payload),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # 工具内部错误也要变成可读文本，而不是让整条链路崩掉
        return ToolResult(
            ok=False,
            tool=name,
            arguments=arguments,
            error=f"{type(exc).__name__}: {exc}",
            text=(
                f"工具 `{name}` 执行出错：{type(exc).__name__}: {exc}\n"
                f"可以尝试换用其他工具，或调整参数后重试。"
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def render_for_llm(payload: BaseModel, *, max_list: int = MAX_LIST_ITEMS) -> str:
    """把工具返回值压成模型友好的紧凑文本：长列表只留头尾，其余字段全保留。"""
    raw = payload.model_dump(mode="json")
    trimmed = _trim(raw, max_list)
    text = json.dumps(trimmed, ensure_ascii=False, indent=2)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n…（已截断，完整结果共 {len(text)} 字符）"
    return text


def _trim(node: Any, max_list: int) -> Any:
    if isinstance(node, dict):
        return {k: _trim(v, max_list) for k, v in node.items()}
    if isinstance(node, list):
        if len(node) <= max_list:
            return [_trim(v, max_list) for v in node]
        head = node[: max_list // 2]
        tail = node[-(max_list - len(head)) :]
        return (
            [_trim(v, max_list) for v in head]
            + [f"…（省略 {len(node) - max_list} 项）"]
            + [_trim(v, max_list) for v in tail]
        )
    if isinstance(node, str) and len(node) > 600:
        return node[:600] + "…"
    return node
