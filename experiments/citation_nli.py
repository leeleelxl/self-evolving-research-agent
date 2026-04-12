"""
引用验证方法对比实验 — Embedding vs NLI

对同一个 Pipeline 结果，分别用两种方法验证引用质量:
1. Embedding: 余弦相似度（话题相关性）
2. NLI: 自然语言推理（蕴含关系）

核心问题: NLI 能否发现 embedding 漏掉的问题引用？

运行: conda run -n base python experiments/citation_nli.py
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
    print("Citation Verification: Embedding vs NLI")
    print(f"Question: {QUESTION}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # 1. 跑一次 Pipeline（共用结果）
    print("--- Running Pipeline ---")
    config = PipelineConfig(max_iterations=2, satisfactory_threshold=8.0)
    pipeline = ResearchPipeline(config)

    start = time.time()
    result = await pipeline.run(QUESTION)
    pipeline_elapsed = time.time() - start

    print(f"\nPipeline done: {result.total_iterations} iterations, {pipeline_elapsed:.0f}s")
    print(f"Report: {len(result.report.sections)} sections, {len(result.report.references)} refs")
    print(f"Papers available: {len(result.papers)}")

    # 2. Embedding 验证
    print("\n--- Embedding Verification ---")
    t0 = time.time()
    emb_verifier = CitationVerifier(method="embedding", grounding_threshold=0.3)
    emb_report = emb_verifier.verify(result)
    emb_elapsed = time.time() - t0
    print(f"Done in {emb_elapsed:.1f}s")

    # 3. NLI 验证
    print("\n--- NLI Verification ---")
    t0 = time.time()
    nli_verifier = CitationVerifier(method="nli", entailment_threshold=0.5)
    nli_report = nli_verifier.verify(result)
    nli_elapsed = time.time() - t0
    print(f"Done in {nli_elapsed:.1f}s")

    # 3.5. Hybrid 验证
    print("\n--- Hybrid Verification (Embedding + NLI) ---")
    t0 = time.time()
    hybrid_verifier = CitationVerifier(method="hybrid", grounding_threshold=0.3, contradiction_threshold=0.5)
    hybrid_report = hybrid_verifier.verify(result)
    hybrid_elapsed = time.time() - t0
    print(f"Done in {hybrid_elapsed:.1f}s")

    # 4. 对比输出
    print(f"\n{'='*60}")
    print("CITATION VERIFICATION COMPARISON")
    print(f"{'='*60}")

    for label, report, elapsed in [
        ("Embedding (cosine sim)", emb_report, emb_elapsed),
        ("NLI (sentence-level)", nli_report, nli_elapsed),
        ("Hybrid (emb + NLI)", hybrid_report, hybrid_elapsed),
    ]:
        print(f"\n{label}:")
        print(f"  Grounding rate: {report['overall_grounding_rate']:.1%}")
        print(f"  Avg score: {report['overall_avg_score']:.3f}")
        print(f"  Checked: {report['num_citations_checked']}")
        print(f"  Grounded: {report['num_citations_grounded']}")
        print(f"  Ungrounded: {report['num_citations_ungrounded']}")
        print(f"  Missing: {report['num_citations_missing']}")
        if "overall_contradiction_rate" in report and report["overall_contradiction_rate"] > 0:
            print(f"  Contradicted: {report['num_citations_contradicted']} "
                  f"({report['overall_contradiction_rate']:.1%})")
        print(f"  Time: {elapsed:.1f}s")

    # 5. 逐 section 对比
    print(f"\nPer-section comparison:")
    print(f"  {'Section':<45s} {'Emb':>6s} {'NLI':>6s} {'Delta':>7s}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*7}")

    for emb_sec, nli_sec in zip(emb_report["sections"], nli_report["sections"]):
        emb_rate = emb_sec["grounding_rate"]
        nli_rate = nli_sec["grounding_rate"]
        delta = nli_rate - emb_rate
        print(
            f"  {emb_sec['section_title'][:45]:<45s} "
            f"{emb_rate:>5.0%} {nli_rate:>5.0%} {delta:>+6.0%}"
        )

    # 6. NLI 发现的矛盾引用
    contradictions = []
    for sec in nli_report["sections"]:
        for cite in sec["citations"]:
            if cite.get("contradicted", False):
                contradictions.append({
                    "section": sec["section_title"][:50],
                    "paper_id": cite["paper_id"],
                    "title": cite.get("title", ""),
                    "probs": cite.get("nli_probs", {}),
                })

    if contradictions:
        print(f"\nContradiction alerts ({len(contradictions)} found):")
        for c in contradictions:
            print(f"  [{c['section']}]")
            print(f"    Paper: {c['title']}")
            print(f"    Probs: E={c['probs'].get('entailment',0):.2f} "
                  f"C={c['probs'].get('contradiction',0):.2f} "
                  f"N={c['probs'].get('neutral',0):.2f}")
    else:
        print(f"\nNo contradictions detected by NLI.")

    # 7. Embedding grounded 但 NLI 不认的（假阳性候选）
    false_positives = []
    for emb_sec, nli_sec in zip(emb_report["sections"], nli_report["sections"]):
        for emb_cite, nli_cite in zip(emb_sec["citations"], nli_sec["citations"]):
            if (
                emb_cite["status"] == "checked"
                and emb_cite["grounded"]
                and not nli_cite["grounded"]
            ):
                false_positives.append({
                    "section": emb_sec["section_title"][:50],
                    "paper_id": emb_cite["paper_id"],
                    "emb_score": emb_cite["score"],
                    "nli_entailment": nli_cite.get("nli_probs", {}).get("entailment", 0),
                })

    if false_positives:
        print(f"\nEmbedding false positives (grounded by emb, not by NLI): {len(false_positives)}")
        for fp in false_positives[:5]:
            print(f"  [{fp['section']}] {fp['paper_id']}")
            print(f"    Emb sim={fp['emb_score']:.3f}, NLI ent={fp['nli_entailment']:.3f}")

    # 8. 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)

    experiment_data = {
        "experiment": "citation_nli_comparison",
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "pipeline": {
            "iterations": result.total_iterations,
            "sections": len(result.report.sections),
            "references": len(result.report.references),
            "papers_available": len(result.papers),
            "elapsed_seconds": round(pipeline_elapsed, 1),
        },
        "embedding_verification": emb_report,
        "nli_verification": nli_report,
        "hybrid_verification": hybrid_report,
        "comparison": {
            "embedding_grounding_rate": emb_report["overall_grounding_rate"],
            "nli_grounding_rate": nli_report["overall_grounding_rate"],
            "hybrid_grounding_rate": hybrid_report["overall_grounding_rate"],
            "hybrid_contradiction_rate": hybrid_report["overall_contradiction_rate"],
            "contradictions_found": len(contradictions),
            "embedding_false_positives": len(false_positives),
            "embedding_elapsed": round(emb_elapsed, 1),
            "nli_elapsed": round(nli_elapsed, 1),
            "hybrid_elapsed": round(hybrid_elapsed, 1),
        },
    }

    output_path = results_dir / "citation_nli.json"
    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
