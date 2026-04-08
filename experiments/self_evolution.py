"""
自进化效果验证实验

目标: 验证 Critic 驱动的自进化机制确实能提升综述质量
方法: 同一个研究问题，对比 max_iterations=1/2/3 的效果
指标: 每轮的 CriticScores (coverage, depth, coherence, accuracy, overall)
输出: experiments/results/self_evolution.json

运行: python experiments/self_evolution.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 path
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


RESEARCH_QUESTION = "What are the recent advances in retrieval-augmented generation (RAG) for large language models?"


async def run_experiment(max_iterations: int) -> dict:
    """跑一次 Pipeline，返回结构化结果"""
    print(f"\n{'='*60}")
    print(f"Running with max_iterations={max_iterations}")
    print(f"{'='*60}\n")

    config = PipelineConfig(
        max_iterations=max_iterations,
        satisfactory_threshold=8.0,  # 设高一点，让它多迭代
    )
    pipeline = ResearchPipeline(config)

    start_time = time.time()
    result = await pipeline.run(RESEARCH_QUESTION)
    elapsed = time.time() - start_time

    # 提取每轮的数据
    rounds = []
    for record in result.evolution_log:
        rounds.append({
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
            "num_queries": len(record.strategy_snapshot.queries),
            "feedback_summary": record.feedback_summary,
        })

    return {
        "max_iterations": max_iterations,
        "actual_iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "final_title": result.report.title,
        "final_sections": len(result.report.sections),
        "final_references": len(result.report.references),
        "rounds": rounds,
    }


async def main() -> None:
    print(f"Research Question: {RESEARCH_QUESTION}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # 跑 max_iterations=3（包含了 1 轮和 2 轮的数据）
    result = await run_experiment(max_iterations=3)

    # 输出摘要
    print(f"\n{'='*60}")
    print("SELF-EVOLUTION RESULTS")
    print(f"{'='*60}")
    print(f"Total iterations: {result['actual_iterations']}")
    print(f"Total time: {result['elapsed_seconds']}s")
    print(f"Final: {result['final_sections']} sections, {result['final_references']} references")
    print()

    for r in result["rounds"]:
        s = r["scores"]
        print(
            f"  Round {r['iteration']}: "
            f"coverage={s['coverage']:.1f} depth={s['depth']:.1f} "
            f"coherence={s['coherence']:.1f} accuracy={s['accuracy']:.1f} "
            f"→ overall={s['overall']:.1f} "
            f"| papers={r['num_papers']} notes={r['num_notes']} queries={r['num_queries']}"
        )

    # 检查是否有分数提升
    if len(result["rounds"]) >= 2:
        r0 = result["rounds"][0]["scores"]["overall"]
        r_last = result["rounds"][-1]["scores"]["overall"]
        delta = r_last - r0
        print(f"\n  Improvement: {r0:.1f} → {r_last:.1f} (Δ{delta:+.1f})")
        if delta > 0:
            print("  ✓ Self-evolution is EFFECTIVE — scores improved!")
        elif delta == 0:
            print("  ~ Self-evolution had NO EFFECT — scores unchanged")
        else:
            print("  ✗ Self-evolution DECREASED scores — needs investigation")

    # 保存结果
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "self_evolution.json"

    experiment_data = {
        "experiment": "self_evolution",
        "timestamp": datetime.now().isoformat(),
        "question": RESEARCH_QUESTION,
        "config": {
            "model": "gpt-4o-mini",
            "critic_model": "gpt-4o",
            "temperature": 0.0,
            "satisfactory_threshold": 8.0,
        },
        "result": result,
    }

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
