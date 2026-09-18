"""
一键运行端到端流水线
====================
用法：
    python scripts/run_pipeline.py            # 完整闭环 + 生成报告
    python scripts/run_pipeline.py --ask "观测期内共检测到多少起异常？"
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import load_config           # noqa
from agents.coordinator import PowerEconomyAgentSystem  # noqa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--no-rebuild", action="store_true", help="复用已有数据，不重新获取")
    parser.add_argument("--ask", default=None, help="运行后向智能体提问")
    args = parser.parse_args()

    cfg = load_config(args.config)
    system = PowerEconomyAgentSystem(cfg)
    out, report = system.run_pipeline(rebuild=not args.no_rebuild)
    trace = system.save_trace()

    print("\n" + "=" * 60)
    print("报告已生成:", out)
    print("协同留痕:", trace)
    print("=" * 60)

    if args.ask:
        print(f"\nQ: {args.ask}")
        print(f"A: {system.ask(args.ask)}")
    else:
        # 默认演示两个问答
        for q in ["观测期内共检测到多少起异常？", "幅度最大的异常是哪次？"]:
            print(f"\nQ: {q}\nA: {system.ask(q)}")


if __name__ == "__main__":
    main()
