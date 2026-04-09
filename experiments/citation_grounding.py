"""
引用质量验证实验

跑一次 Pipeline，然后用 embedding 相似度验证每条引用是否有支撑。
这是打破 "LLM 评 LLM" 闭环的客观指标。

运行: python experiments/citation_grounding.py
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
from research.evaluation.citation_verifier import CitationVerifier
from research.pipeline.research import ResearchPipeline

QUESTION = "What are the recent advances in retrieval-augmented generation (RAG) for large language models?"


async def main() -> None:
    print(f"Citation Grounding Experiment")
    print(f"Question: {QUESTION}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # 1. 跑 Pipeline
    print("--- Running Pipeline ---")
    config = PipelineConfig(max_iterations=2, satisfactory_threshold=8.0)
    pipeline = ResearchPipeline(config)

    start = time.time()
    result = await pipeline.run(QUESTION)
    elapsed = time.time() - start

    print(f"\nPipeline done: {result.total_iterations} iterations, {elapsed:.0f}s")
    print(f"Report: {len(result.report.sections)} sections, {len(result.report.references)} refs")
    print(f"Papers available: {len(result.papers)}")

    # 2. 引用验证
    print("\n--- Citation Verification ---")
    verifier = CitationVerifier()
    verification = verifier.verify(result)

    # 3. 输出结果
    print(f"\n{'='*60}")
    print("CITATION GROUNDING RESULTS")
    print(f"{'='*60}")
    print(f"Overall grounding rate: {verification['overall_grounding_rate']:.1%}")
    print(f"Overall avg similarity: {verification['overall_avg_similarity']:.3f}")
    print(f"Citations checked: {verification['num_citations_checked']}")
    print(f"  Grounded: {verification['num_citations_grounded']}")
    print(f"  Ungrounded: {verification['num_citations_ungrounded']}")
    print(f"  Missing (paper not found): {verification['num_citations_missing']}")
    print(f"Threshold: {verification['grounding_threshold']}")

    print(f"\nPer-section breakdown:")
    for sec in verification["sections"]:
        print(
            f"  {sec['section_title'][:50]:50s} "
            f"grounding={sec['grounding_rate']:.0%} "
            f"({sec.get('num_grounded', 0)}/{sec.get('num_cited', 0)})"
        )

    # 低相似度的引用（潜在问题）
    low_cites = []
    for sec in verification["sections"]:
        for cite in sec["citations"]:
            if cite["status"] == "checked" and not cite["grounded"]:
                low_cites.append((sec["section_title"][:40], cite["paper_id"], cite["similarity"]))

    if low_cites:
        print(f"\nLow-similarity citations (potential issues):")
        for sec_title, pid, sim in low_cites[:5]:
            print(f"  [{sec_title}] {pid} → sim={sim:.3f}")

    # 4. 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)

    experiment_data = {
        "experiment": "citation_grounding",
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "pipeline": {
            "iterations": result.total_iterations,
            "sections": len(result.report.sections),
            "references": len(result.report.references),
            "papers_available": len(result.papers),
            "elapsed_seconds": round(elapsed, 1),
        },
        "verification": verification,
    }

    output_path = results_dir / "citation_grounding.json"
    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
