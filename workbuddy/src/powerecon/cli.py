"""命令行入口。

设计上刻意只留很少几个命令 —— 这是一个分析工具，不是框架。
用户主要用的是 `ask`；`tools` / `inspect` 是给开发者对齐接口用的。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .agent import AgentLoop, LLMPlanner, RulePlanner
from .ingest import RegionPackError, available_packs
from .runtime import default_artifacts_root, default_packs_root, make_context
from .tools import anthropic_tools, openai_tools, tool_catalog_markdown
from .tools.base import list_specs

DEFAULT_QUESTION = "这个地区最近的电力数据有没有异常？能说明经济状况吗？"


def _load_callable(spec: str):
    """从 'module:function' 加载回调。这是接入大模型 API 的唯一接缝。"""
    if ":" not in spec:
        raise SystemExit(
            f"--llm-callable 需要写成 module:function 的形式，例如 my_llm:call_llm。收到：{spec}"
        )
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise SystemExit(f"模块 {module_name} 里没有 {func_name}。")
    return fn


def _build_planner(args, ctx):
    if args.planner == "llm":
        if not args.llm_callable:
            raise SystemExit(
                "选择 --planner llm 时必须提供 --llm-callable module:function。\n"
                "回调签名：call_llm(messages, tools) -> 模型响应对象或 dict。\n"
                "messages 是 OpenAI 兼容格式，tools 是 function-calling 清单，\n"
                "返回 None 或空 tool_calls 表示模型认为可以结束。"
            )
        return LLMPlanner(_load_callable(args.llm_callable))
    return RulePlanner(dataset_id=args.dataset)


def cmd_regions(args) -> int:
    packs_root = Path(args.packs_root) if args.packs_root else default_packs_root()
    names = available_packs(packs_root)
    if not names:
        print(f"在 {packs_root} 下没有找到任何 region pack。")
        return 1
    print(f"可用地区数据包（{packs_root}）：")
    for name in names:
        try:
            ctx = make_context(name, packs_root=packs_root, artifacts_root=default_artifacts_root())
            info = ctx.region.info
            n_ds = len(ctx.region.datasets)
            print(f"  - {name:12s} {info.code:8s} {info.name}  数据集 {n_ds} 个  数据边界 {len(info.caveats)} 条")
        except RegionPackError as exc:
            print(f"  - {name:12s} 加载失败：{exc}")
    return 0


def cmd_inspect(args) -> int:
    ctx = make_context(args.region, packs_root=args.packs_root)
    info = ctx.region.info
    print(f"地区：{info.name}（{info.code}，{info.level}）")
    print(f"数据包：{info.pack_dir}")
    print(f"代表性气象点：{info.representative_point}")
    print()
    print("数据集：")
    for ds in sorted(ctx.region.datasets.values(), key=lambda d: d.id):
        flag = "自动发现" if ds.discovered else "显式声明"
        print(
            f"  - {ds.id:22s} {ds.frequency.value:10s} {ds.metric[:20]:22s} "
            f"{ds.n_rows:5d} 行  {ds.period_start} ~ {ds.period_end}  [{flag}]"
        )
        if ds.quality_flags:
            print(f"      quality_flags: {ds.quality_flags}")
    print()
    print(f"知识图谱：{'已加载' if ctx.region.kg_relationships is not None else '无'}", end="")
    if ctx.region.kg_relationships is not None:
        print(
            f"（{len(ctx.region.kg_nodes)} 节点 / {len(ctx.region.kg_relationships)} 关系）"
        )
    else:
        print()
    print(f"知识文档片段：{len(ctx.region.knowledge_docs)} 条")
    print()
    print("数据边界声明（会原样进入模型上下文）：")
    for c in info.caveats:
        print(f"  - {c}")
    return 0


def cmd_tools(args) -> int:
    if args.format == "md":
        print(tool_catalog_markdown())
        return 0
    if args.format == "openai":
        payload = openai_tools()
    elif args.format == "anthropic":
        payload = anthropic_tools()
    elif args.format == "names":
        for spec in list_specs():
            print(f"{spec.name:24s} -> {spec.returns}")
        return 0
    else:
        payload = [
            {"name": s.name, "description": s.description, "args_schema": s.input_schema, "returns": s.returns}
            for s in list_specs()
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_ask(args) -> int:
    ctx = make_context(args.region, packs_root=args.packs_root)
    planner = _build_planner(args, ctx)
    loop = AgentLoop(
        ctx,
        planner,
        max_steps=args.max_steps,
        trace_dir=(Path(args.trace_dir) if args.trace_dir else default_artifacts_root() / "traces"),
    )
    result = loop.run(args.question)

    print("=" * 72)
    print(result.summary())
    if result.trace_path:
        print(f"留痕：{result.trace_path}")
    print("=" * 72)
    if result.conclusion is not None:
        print()
        print(result.conclusion.to_markdown())
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(result.conclusion.to_markdown(), encoding="utf-8")
            print(f"\n已写入 {out}")
    return 0 if result.ok else 1


def cmd_demo(args) -> int:
    ctx = make_context(args.region, packs_root=args.packs_root)
    planner = RulePlanner(dataset_id=args.dataset)
    loop = AgentLoop(
        ctx,
        planner,
        max_steps=args.max_steps,
        trace_dir=default_artifacts_root() / "traces",
    )
    questions = args.questions or [
        DEFAULT_QUESTION,
        "最近一期用电量同比变化多少？季节因素占多大比重？",
    ]
    for q in questions:
        print("#" * 72)
        print(f"# 问题：{q}")
        print("#" * 72)
        result = loop.run(q)
        print(result.summary())
        if result.conclusion is not None:
            print()
            print(result.conclusion.to_markdown())
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="powerecon",
        description="电力看经济智能体 —— 工具层与智能体循环",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--packs-root", default=None, help="region_packs 所在目录，默认仓库内 region_packs/")
        p.add_argument("--region", default="zhejiang", help="地区数据包名或地区代码，默认 zhejiang")
        p.add_argument("--dataset", default=None, help="指定数据集 id，默认自动选择用电量数据集")

    p_regions = sub.add_parser("regions", help="列出本机所有地区数据包")
    p_regions.add_argument("--packs-root", default=None)
    p_regions.set_defaults(func=cmd_regions)

    p_inspect = sub.add_parser("inspect", help="查看某个地区的数据清单与数据边界")
    add_common(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_tools = sub.add_parser("tools", help="导出工具清单与 JSON Schema")
    p_tools.add_argument(
        "--format",
        default="md",
        choices=["md", "json", "openai", "anthropic", "names"],
        help="md=表格说明，openai/anthropic=可直接塞进 API 的 function-calling 清单",
    )
    p_tools.set_defaults(func=cmd_tools)

    p_ask = sub.add_parser("ask", help="提一个自然语言问题，跑完整智能体循环")
    add_common(p_ask)
    p_ask.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    p_ask.add_argument("--planner", default="rule", choices=["rule", "llm"], help="rule=确定性规划器（默认），llm=接入大模型")
    p_ask.add_argument("--llm-callable", default=None, help="module:function，仅 --planner llm 时使用")
    p_ask.add_argument("--max-steps", type=int, default=12)
    p_ask.add_argument("--trace-dir", default=None)
    p_ask.add_argument("--out", default=None, help="把结论 Markdown 写到指定路径")
    p_ask.set_defaults(func=cmd_ask)

    p_demo = sub.add_parser("demo", help="跑一组示例问题，验证整条链路")
    add_common(p_demo)
    p_demo.add_argument("--max-steps", type=int, default=12)
    p_demo.add_argument("--questions", nargs="*", default=None)
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RegionPackError as exc:
        print(f"数据包错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
