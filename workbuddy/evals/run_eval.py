"""评测脚本。

用法：
    python evals/run_eval.py                       # 用规则规划器跑全部题目（基线）
    python evals/run_eval.py --planner llm --llm-callable my_llm:call_llm
    python evals/run_eval.py --id zj-01 --verbose  # 只跑一题，打印完整结论

输出：
    artifacts/evals/<timestamp>/report.md    人读的报告
    artifacts/evals/<timestamp>/results.json 机读的明细

为什么要有这个脚本：
"判断得准不准"必须有客观评分。没有评分集，所有"感觉变好了"都是自我安慰。
这个脚本同时服务两个目的 —— ① 回归测试：改了提示词或工具后看分数有没有掉；
② 基线对照：规则规划器是下限，LLM 规划器至少要明显好于它才有意义。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from powerecon.agent import AgentLoop, LLMPlanner, RulePlanner  # noqa: E402
from powerecon.runtime import make_context  # noqa: E402


@dataclass
class CaseResult:
    case_id: str
    region: str
    question: str
    ok: bool
    confidence: float
    causal_status: str
    tool_sequence: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conclusion_text: str = ""
    error: str | None = None


# 否定语境标记。出现在禁用短语之前，说明这句话是在否定/拒绝该说法，
# 而不是在主张它 —— 例如"本次分析不支持「用电下降说明经济走弱」"是正确表述。
_NEGATION_MARKERS = (
    "不支持", "不能", "不应", "无法", "否认", "排除", "拒绝",
    "不等于", "并非", "不是", "不可", "不宜", "避免",
)
_QUOTE_OPEN = ("「", "“", "『", "《", "'", '"')


def _find_asserted_phrase(text: str, phrase: str) -> str | None:
    """找出 phrase 被当作"主张"使用的上下文；若只出现在否定/引用语境则返回 None。"""
    idx = 0
    while True:
        idx = text.find(phrase, idx)
        if idx < 0:
            return None
        before = text[max(0, idx - 20) : idx]
        rejected = before.rstrip().endswith(_QUOTE_OPEN) or any(
            m in before for m in _NEGATION_MARKERS
        )
        if not rejected:
            return text[max(0, idx - 25) : idx + len(phrase) + 25].replace("\n", " ")
        idx += len(phrase)


def check_case(case: dict, result, meta: dict, strict_tools: bool) -> CaseResult:
    expect = case.get("expect") or {}
    c = result.conclusion
    cr = CaseResult(
        case_id=case["id"],
        region=case.get("region", "?"),
        question=case["question"],
        ok=True,
        confidence=0.0,
        causal_status="?",
        tool_sequence=list(result.tool_sequence),
    )
    if result.error:
        cr.ok = False
        cr.failures.append(f"运行出错：{result.error}")
        return cr
    if c is None:
        cr.ok = False
        cr.failures.append("没有产出结论")
        return cr

    cr.confidence = c.confidence
    cr.causal_status = c.causal_status.value
    cr.conclusion_text = c.conclusion
    # 数据边界（caveats）也是交付给用户的内容，计入可核查文本
    text = "\n".join(
        [c.conclusion]
        + [e.claim for e in c.evidence]
        + c.counter_evidence
        + c.uncertainty
        + c.caveats
    )

    # --- 硬检查：任何规划器都必须满足

    if cr.causal_status in (meta.get("forbidden_causal_status") or []):
        cr.ok = False
        cr.failures.append(f"因果强度越界：{cr.causal_status}")

    for phrase in meta.get("forbidden_phrases") or []:
        ctx = _find_asserted_phrase(text, phrase)
        if ctx is not None:
            cr.ok = False
            cr.failures.append(f"把「{phrase}」当作主张使用：…{ctx}…")

    if not c.counter_evidence:
        cr.ok = False
        cr.failures.append("缺少反证")
    if not c.uncertainty:
        cr.ok = False
        cr.failures.append("缺少不确定性说明")

    ceiling = float(meta.get("max_allowed_confidence", 0.85))
    if c.confidence > ceiling:
        cr.ok = False
        cr.failures.append(f"置信度 {c.confidence:.2f} 超过全局上限 {ceiling}")

    if "max_confidence" in expect and c.confidence > float(expect["max_confidence"]):
        cr.ok = False
        cr.failures.append(
            f"置信度 {c.confidence:.2f} 高于本题上限 {expect['max_confidence']}"
        )
    if "min_confidence" in expect and c.confidence < float(expect["min_confidence"]):
        cr.ok = False
        cr.failures.append(
            f"置信度 {c.confidence:.2f} 低于本题下限 {expect['min_confidence']}"
        )

    mentions = expect.get("must_mention_any") or []
    if mentions and not any(m in text for m in mentions):
        cr.ok = False
        cr.failures.append(f"未提及任何关键概念：{mentions}")

    period = case.get("period")
    if period:
        periods_in_evidence = " ".join(e.period or "" for e in c.evidence)
        if period not in periods_in_evidence and period not in text:
            cr.ok = False
            cr.failures.append(f"结论未涉及用户点明的期次 {period}")

    # --- 软检查：仅对 LLM 规划器生效

    must_call = expect.get("must_call") or []
    missing = [t for t in must_call if t not in result.tool_sequence]
    if missing:
        if strict_tools:
            cr.ok = False
            cr.failures.append(f"未调用必需工具：{missing}")
        else:
            cr.warnings.append(f"未调用工具 {missing}（规则规划器为固定链路，不计失败）")

    return cr


def main() -> int:
    parser = argparse.ArgumentParser(description="跑评测集")
    parser.add_argument("--questions", default=str(REPO_ROOT / "evals" / "questions.yaml"))
    parser.add_argument("--planner", default="rule", choices=["rule", "llm"])
    parser.add_argument("--llm-callable", default=None, help="module:function，仅 --planner llm")
    parser.add_argument("--id", default=None, help="只跑指定 id 的题目")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
    meta = spec.get("meta") or {}
    cases = spec["cases"]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            print(f"没有 id 为 {args.id} 的题目", file=sys.stderr)
            return 2

    if args.planner == "llm":
        if not args.llm_callable:
            print("--planner llm 需要 --llm-callable module:function", file=sys.stderr)
            return 2
        module_name, func_name = args.llm_callable.split(":", 1)
        call_llm = getattr(importlib.import_module(module_name), func_name)
        planner_factory = lambda: LLMPlanner(call_llm)  # noqa: E731
    else:
        planner_factory = RulePlanner

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "artifacts" / "evals" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for case in cases:
        ctx = make_context(case.get("region", "zhejiang"))
        loop = AgentLoop(ctx, planner_factory(), max_steps=args.max_steps)
        run = loop.run(case["question"])
        cr = check_case(case, run, meta, strict_tools=(args.planner == "llm"))
        results.append(cr)

        mark = "PASS" if cr.ok else "FAIL"
        print(f"[{mark}] {cr.case_id:10s} conf={cr.confidence:.2f} {cr.causal_status:16s} {case['question'][:34]}")
        for f in cr.failures:
            print(f"        × {f}")
        if args.verbose:
            for w in cr.warnings:
                print(f"        ! {w}")
            if cr.conclusion_text:
                print("        " + cr.conclusion_text.replace("\n", "\n        ")[:1200])

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    score = passed / total if total else 0.0

    print()
    print("=" * 68)
    print(f"规划器：{args.planner}　通过 {passed}/{total}　得分 {score:.1%}")
    by_region: dict[str, list[CaseResult]] = {}
    for r in results:
        by_region.setdefault(r.region, []).append(r)
    for region, rs in sorted(by_region.items()):
        p = sum(1 for r in rs if r.ok)
        print(f"  {region:10s} {p}/{len(rs)}")
    print("=" * 68)

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "planner": args.planner,
                "score": score,
                "passed": passed,
                "total": total,
                "cases": [
                    {
                        "id": r.case_id,
                        "region": r.region,
                        "question": r.question,
                        "ok": r.ok,
                        "confidence": r.confidence,
                        "causal_status": r.causal_status,
                        "tool_sequence": r.tool_sequence,
                        "failures": r.failures,
                        "warnings": r.warnings,
                        "conclusion": r.conclusion_text,
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = [
        f"# 评测报告 · {stamp}",
        "",
        f"- 规划器：`{args.planner}`",
        f"- 得分：**{score:.1%}**（{passed}/{total}）",
        "",
        "| 题目 | 地区 | 结果 | 置信度 | 因果强度 | 问题 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        report.append(
            f"| `{r.case_id}` | {r.region} | {'通过' if r.ok else '**失败**'} | "
            f"{r.confidence:.2f} | `{r.causal_status}` | {r.question} |"
        )
    fails = [r for r in results if not r.ok]
    if fails:
        report += ["", "## 失败明细", ""]
        for r in fails:
            report.append(f"### `{r.case_id}` {r.question}")
            for f in r.failures:
                report.append(f"- {f}")
            report.append("")
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n报告：{out_dir / 'report.md'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
