"""
PDF 全文消融实验 — 验证全文精读 vs 仅 abstract 的效果差异

对比两种模式:
1. Abstract-only: 传统模式，Reader 只读 abstract（baseline）
2. Full-text: Reader 读 PDF 提取的全文

指标:
- Critic 评分（coverage, depth, coherence, accuracy, overall）
- Reader 精读论文数
- 全文提取成功率
- Pipeline 总耗时

运行: conda run -n base python experiments/pdf_ablation.py
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

from research.core.config import PDFConfig, PipelineConfig
from research.pipeline.research import ResearchPipeline

RESEARCH_QUESTION = "What are the recent advances in retrieval-augmented generation (RAG) for large language models?"


async def run_pipeline(pdf_enabled: bool) -> dict:
    """跑一次 Pipeline，返回结构化结果"""
    mode = "Full-text (PDF)" if pdf_enabled else "Abstract-only"
    print(f"\n{'='*60}")
    print(f"Running: {mode}")
    print(f"{'='*60}\n")

    config = PipelineConfig(
        max_iterations=2,
        satisfactory_threshold=8.0,
        pdf=PDFConfig(enabled=pdf_enabled),
    )
    pipeline = ResearchPipeline(config)

    start_time = time.time()
    result = await pipeline.run(RESEARCH_QUESTION)
    elapsed = time.time() - start_time

    # 统计全文提取情况
    total_papers = len(result.papers)
    papers_with_pdf_url = sum(1 for p in result.papers if p.pdf_url)
    papers_with_full_text = sum(1 for p in result.papers if p.full_text)

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
        })

    return {
        "pdf_enabled": pdf_enabled,
        "actual_iterations": result.total_iterations,
        "elapsed_seconds": round(elapsed, 1),
        "final_sections": len(result.report.sections),
        "final_references": len(result.report.references),
        "total_papers": total_papers,
        "papers_with_pdf_url": papers_with_pdf_url,
        "papers_with_full_text": papers_with_full_text,
        "pdf_extraction_rate": (
            round(papers_with_full_text / papers_with_pdf_url, 3)
            if papers_with_pdf_url > 0
            else 0
        ),
        "rounds": rounds,
    }


async def main() -> None:
    print("PDF Full-Text Ablation Experiment")
    print(f"Question: {RESEARCH_QUESTION}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # baseline: abstract-only
    result_abstract = await run_pipeline(pdf_enabled=False)
    # treatment: full-text PDF
    result_fulltext = await run_pipeline(pdf_enabled=True)

    # 汇总
    print(f"\n{'='*60}")
    print("PDF ABLATION RESULTS")
    print(f"{'='*60}")

    for label, result in [
        ("Abstract-only", result_abstract),
        ("Full-text (PDF)", result_fulltext),
    ]:
        last_round = result["rounds"][-1] if result["rounds"] else None
        if last_round:
            s = last_round["scores"]
            print(
                f"\n{label}:"
                f"\n  Overall: {s['overall']:.1f} "
                f"(cov={s['coverage']:.1f} dep={s['depth']:.1f} "
                f"coh={s['coherence']:.1f} acc={s['accuracy']:.1f})"
                f"\n  Papers: {result['total_papers']} total, "
                f"{result['papers_with_full_text']} with full text"
                f"\n  PDF extraction rate: {result['pdf_extraction_rate']:.1%}"
                f"\n  Time: {result['elapsed_seconds']}s"
            )

    # 对比
    if result_abstract["rounds"] and result_fulltext["rounds"]:
        abs_score = result_abstract["rounds"][-1]["scores"]["overall"]
        ft_score = result_fulltext["rounds"][-1]["scores"]["overall"]
        delta = ft_score - abs_score
        print(f"\n  Delta: {abs_score:.1f} → {ft_score:.1f} (Δ{delta:+.1f})")

    # 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "pdf_ablation.json"

    experiment_data = {
        "experiment": "pdf_ablation",
        "timestamp": datetime.now().isoformat(),
        "question": RESEARCH_QUESTION,
        "config": {
            "model": "gpt-4o-mini",
            "critic_model": "gpt-4o",
            "max_iterations": 2,
            "pdf_max_pages": 30,
            "pdf_max_text_length": 50000,
        },
        "results": {
            "abstract_only": result_abstract,
            "full_text": result_fulltext,
        },
    }

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
