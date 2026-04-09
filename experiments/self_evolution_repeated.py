"""
重复自进化实验 — 统计显著性验证

目标: 跑 N 次同一个研究问题，报告 mean±std，验证自进化效果是否稳定
新特性: Pipeline 现在包含 KnowledgeBase + Multi-LLM Critic

运行: python experiments/self_evolution_repeated.py [--runs 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
)

from research.core.config import PipelineConfig
from research.pipeline.research import ResearchPipeline

QUESTION = "What are the recent advances in retrieval-augmented generation (RAG) for large language models?"


async def single_run(run_id: int) -> dict:
    """单次 Pipeline 运行"""
    print(f"\n{'='*60}")
    print(f"RUN {run_id}")
    print(f"{'='*60}\n")

    config = PipelineConfig(
        max_iterations=2,
        satisfactory_threshold=8.0,
    )
    pipeline = ResearchPipeline(config)

    start = time.time()
    result = await pipeline.run(QUESTION)
    elapsed = time.time() - start

    rounds = []
    for record in result.evolution_log:
        round_data = {
            "iteration": record.iteration,
            "scores": {
                "coverage": record.scores.coverage,
                "depth": record.scores.depth,
                "coherence": record.scores.coherence,
                "accuracy": record.scores.accuracy,
                "overall": record.scores.overall,
            },
            "num_papers": record.num_papers,
            "num_notes": record.num_notes,
        }
        if record.scores.cross_model_spread:
            round_data["cross_model_spread"] = record.scores.cross_model_spread
        rounds.append(round_data)

    return {
        "run_id": run_id,
        "actual_iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "rounds": rounds,
    }


def compute_stats(runs: list[dict]) -> dict:
    """计算各维度的 mean±std"""
    import statistics

    # 取每次运行的最后一轮分数
    final_scores = []
    for run in runs:
        if run["rounds"]:
            final_scores.append(run["rounds"][-1]["scores"])

    if not final_scores:
        return {}

    stats = {}
    for dim in ["coverage", "depth", "coherence", "accuracy", "overall"]:
        values = [s[dim] for s in final_scores]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[dim] = {"mean": round(mean, 2), "std": round(std, 2), "values": values}

    # 自进化 Δ（如果有多轮的 run）
    deltas = []
    for run in runs:
        if len(run["rounds"]) >= 2:
            r0 = run["rounds"][0]["scores"]["overall"]
            r_last = run["rounds"][-1]["scores"]["overall"]
            deltas.append(round(r_last - r0, 2))

    if deltas:
        stats["evolution_delta"] = {
            "mean": round(statistics.mean(deltas), 2),
            "std": round(statistics.stdev(deltas) if len(deltas) > 1 else 0.0, 2),
            "values": deltas,
            "all_positive": all(d > 0 for d in deltas),
        }

    return stats


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print(f"Repeated Self-Evolution Experiment")
    print(f"Question: {QUESTION}")
    print(f"Runs: {args.runs}")
    print(f"Features: KnowledgeBase=ON, Multi-LLM Critic=ON")
    print(f"Timestamp: {datetime.now().isoformat()}")

    runs = []
    for i in range(args.runs):
        result = await single_run(i)
        runs.append(result)

        # 打印中间状态
        if result["rounds"]:
            last = result["rounds"][-1]["scores"]
            print(f"\n  Run {i} done: overall={last['overall']:.1f}, time={result['elapsed_seconds']}s")

    # 统计
    stats = compute_stats(runs)

    print(f"\n{'='*60}")
    print("REPEATED EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"Runs: {args.runs}")

    for dim in ["coverage", "depth", "coherence", "accuracy", "overall"]:
        if dim in stats:
            s = stats[dim]
            print(f"  {dim:12s}: {s['mean']:.2f} ± {s['std']:.2f}  ({s['values']})")

    if "evolution_delta" in stats:
        d = stats["evolution_delta"]
        print(f"\n  Evolution Δ: {d['mean']:+.2f} ± {d['std']:.2f}  ({d['values']})")
        print(f"  All positive: {d['all_positive']}")

    # 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)

    experiment_data = {
        "experiment": "self_evolution_repeated",
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "num_runs": args.runs,
        "config": {
            "model": "gpt-4o-mini",
            "critic_models": ["gpt-4o", "claude-sonnet-4-6-20250514"],
            "max_iterations": 2,
            "knowledge_base": True,
        },
        "runs": runs,
        "stats": stats,
    }

    output_path = results_dir / "self_evolution_repeated.json"
    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
