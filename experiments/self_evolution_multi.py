"""
自进化多问题验证 — 验证自进化机制的普遍有效性

用 3 个不同领域的研究问题各跑 2 轮迭代，
验证自进化不是针对特定问题的个例，而是普遍有效的机制。

运行: python experiments/self_evolution_multi.py
"""

from __future__ import annotations

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

# 3 个不同领域的研究问题
QUESTIONS = [
    {
        "id": "nlp_agents",
        "domain": "NLP / AI Agents",
        "question": "What are the recent advances in tool-using capabilities of large language models?",
    },
    {
        "id": "ml_efficiency",
        "domain": "ML Systems / Efficiency",
        "question": "What are the latest techniques for efficient inference of large language models?",
    },
    {
        "id": "ai_safety",
        "domain": "AI Safety / Alignment",
        "question": "What are the current approaches to evaluating and mitigating hallucinations in large language models?",
    },
]


async def run_one_question(q: dict, max_iterations: int = 2) -> dict:
    """对单个问题跑自进化 Pipeline"""
    print(f"\n{'='*60}")
    print(f"[{q['id']}] Domain: {q['domain']}")
    print(f"Question: {q['question']}")
    print(f"{'='*60}\n")

    config = PipelineConfig(
        max_iterations=max_iterations,
        satisfactory_threshold=8.0,
    )
    pipeline = ResearchPipeline(config)

    start = time.time()
    result = await pipeline.run(q["question"])
    elapsed = time.time() - start

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
        })

    return {
        "question_id": q["id"],
        "domain": q["domain"],
        "question": q["question"],
        "actual_iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "final_sections": len(result.report.sections),
        "final_references": len(result.report.references),
        "rounds": rounds,
    }


async def main() -> None:
    print(f"Self-Evolution Multi-Question Experiment")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Questions: {len(QUESTIONS)}")

    all_results = []
    for q in QUESTIONS:
        result = await run_one_question(q)
        all_results.append(result)

        # 打印单个问题的结果
        for r in result["rounds"]:
            s = r["scores"]
            print(
                f"  Round {r['iteration']}: overall={s['overall']:.1f} "
                f"| papers={r['num_papers']} notes={r['num_notes']} queries={r['num_queries']}"
            )
        if len(result["rounds"]) >= 2:
            r0 = result["rounds"][0]["scores"]["overall"]
            r_last = result["rounds"][-1]["scores"]["overall"]
            print(f"  Δ = {r_last - r0:+.1f}")

    # 汇总
    print(f"\n{'='*60}")
    print("MULTI-QUESTION SELF-EVOLUTION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Question':<20} {'R0':<6} {'R1':<6} {'Δ':<6} {'Effective?'}")
    print("-" * 50)

    improvements = []
    for r in all_results:
        rounds = r["rounds"]
        r0 = rounds[0]["scores"]["overall"]
        r_last = rounds[-1]["scores"]["overall"] if len(rounds) > 1 else r0
        delta = r_last - r0
        improvements.append(delta)
        effective = "✓" if delta > 0 else ("~" if delta == 0 else "✗")
        print(f"{r['question_id']:<20} {r0:<6.1f} {r_last:<6.1f} {delta:<+6.1f} {effective}")

    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    effective_count = sum(1 for d in improvements if d > 0)
    print(f"\nAverage improvement: {avg_improvement:+.2f}")
    print(f"Effective in: {effective_count}/{len(improvements)} questions")

    # 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "self_evolution_multi.json"

    experiment_data = {
        "experiment": "self_evolution_multi",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_iterations": 2,
            "satisfactory_threshold": 8.0,
            "model": "gpt-4o-mini",
            "critic_model": "gpt-4o",
        },
        "summary": {
            "num_questions": len(QUESTIONS),
            "avg_improvement": round(avg_improvement, 3),
            "effective_count": effective_count,
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
