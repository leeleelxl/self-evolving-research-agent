"""
KnowledgeBase 消融实验 — 验证 RAG 集成到 Pipeline 的价值

对比两种模式:
1. KB enabled: Retriever → KB 过滤 → Reader 只读相关论文
2. KB disabled: Retriever → Reader 读全部论文（原始模式）

指标:
- Critic 评分（coverage, depth, coherence, accuracy, overall）
- Reader 精读论文数（效率）
- Pipeline 总耗时

运行: python experiments/kb_ablation.py
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

from research.core.config import KnowledgeBaseConfig, PipelineConfig
from research.pipeline.research import ResearchPipeline

RESEARCH_QUESTION = "What are the recent advances in retrieval-augmented generation (RAG) for large language models?"


async def run_pipeline(kb_enabled: bool) -> dict:
    """跑一次 Pipeline，返回结构化结果"""
    mode = "KB enabled" if kb_enabled else "KB disabled"
    print(f"\n{'='*60}")
    print(f"Running: {mode}")
    print(f"{'='*60}\n")

    config = PipelineConfig(
        max_iterations=2,
        satisfactory_threshold=8.0,
        knowledge_base=KnowledgeBaseConfig(enabled=kb_enabled),
    )
    pipeline = ResearchPipeline(config)

    start_time = time.time()
    result = await pipeline.run(RESEARCH_QUESTION)
    elapsed = time.time() - start_time

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
        "kb_enabled": kb_enabled,
        "actual_iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "final_sections": len(result.report.sections),
        "final_references": len(result.report.references),
        "rounds": rounds,
    }


async def main() -> None:
    print(f"KnowledgeBase Ablation Experiment")
    print(f"Question: {RESEARCH_QUESTION}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # 先跑 KB disabled（baseline），再跑 KB enabled
    result_without = await run_pipeline(kb_enabled=False)
    result_with = await run_pipeline(kb_enabled=True)

    # 汇总
    print(f"\n{'='*60}")
    print("KB ABLATION RESULTS")
    print(f"{'='*60}")

    for label, result in [("Without KB", result_without), ("With KB", result_with)]:
        last_round = result["rounds"][-1] if result["rounds"] else None
        if last_round:
            s = last_round["scores"]
            print(
                f"\n{label}:"
                f"\n  Overall: {s['overall']:.1f} "
                f"(cov={s['coverage']:.1f} dep={s['depth']:.1f} "
                f"coh={s['coherence']:.1f} acc={s['accuracy']:.1f})"
                f"\n  Papers read: {last_round['num_papers']} → {last_round['num_notes']} notes"
                f"\n  Time: {result['elapsed_seconds']}s"
            )

    # 对比
    if result_with["rounds"] and result_without["rounds"]:
        with_score = result_with["rounds"][-1]["scores"]["overall"]
        without_score = result_without["rounds"][-1]["scores"]["overall"]
        delta = with_score - without_score
        print(f"\n  Delta: {without_score:.1f} → {with_score:.1f} (Δ{delta:+.1f})")

    # 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "kb_ablation.json"

    experiment_data = {
        "experiment": "kb_ablation",
        "timestamp": datetime.now().isoformat(),
        "question": RESEARCH_QUESTION,
        "config": {
            "model": "gpt-4o-mini",
            "critic_model": "gpt-4o",
            "max_iterations": 2,
            "embedding": "BAAI/bge-small-en-v1.5",
            "retrieval_strategy": "hybrid",
            "chunk_strategy": "recursive",
        },
        "results": {
            "without_kb": result_without,
            "with_kb": result_with,
        },
    }

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
