"""
Multi-LLM Critic 假设验证

假设：用不同模型（gpt-4o vs claude-sonnet）做 Critic，分数会有有意义的差异。
验证方式：对同一份综述，分别用两个模型评分，对比差异。

运行: python experiments/validate_multi_critic.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from research.core.config import LLMConfig, PipelineConfig
from research.core.models import (
    CriticFeedback,
    ResearchReport,
    ReportSection,
)
from research.agents.critic import CriticAgent

QUESTION = "What are the recent advances in retrieval-augmented generation (RAG) for large language models?"

# 用一份真实的综述来测试（模拟 Pipeline 输出）
TEST_REPORT = ResearchReport(
    title="Recent Advances in Retrieval-Augmented Generation for Large Language Models",
    abstract=(
        "This survey examines recent advances in RAG for LLMs, covering retrieval architectures, "
        "generation strategies, and evaluation approaches. We analyze dense, sparse, and hybrid "
        "retrieval methods, discuss the integration of retrieval with generation through various "
        "architectural patterns, and review emerging techniques for improving factual accuracy."
    ),
    sections=[
        ReportSection(
            section_title="Retrieval Architectures",
            content=(
                "Modern RAG systems employ three primary retrieval paradigms. Dense retrieval uses "
                "learned embeddings (e.g., DPR, Contriever) to capture semantic similarity. Sparse "
                "retrieval relies on term matching (BM25) for precise keyword lookup. Hybrid approaches "
                "combine both through reciprocal rank fusion (RRF), consistently outperforming either "
                "method alone. Recent work on ColBERT and multi-vector representations bridges the "
                "gap between dense and sparse methods."
            ),
            cited_papers=["p1", "p2", "p3"],
        ),
        ReportSection(
            section_title="Generation Integration Patterns",
            content=(
                "RAG architectures vary in how retrieval is integrated with generation. The simplest "
                "approach prepends retrieved passages to the prompt. More advanced methods include "
                "Fusion-in-Decoder (FiD), which encodes each passage separately before fusing in the "
                "decoder, and RETRO, which adds retrieval through cross-attention layers. Recent "
                "approaches like Self-RAG allow the model to decide when and what to retrieve."
            ),
            cited_papers=["p4", "p5", "p6"],
        ),
        ReportSection(
            section_title="Factual Accuracy and Hallucination Mitigation",
            content=(
                "A key motivation for RAG is reducing hallucinations. Studies show that RAG reduces "
                "factual errors by 30-50% compared to closed-book generation. However, challenges "
                "remain: models may ignore retrieved evidence, hallucinate citations, or fail to "
                "resolve contradictory information from multiple sources. Attribution and citation "
                "verification are active research areas."
            ),
            cited_papers=["p7", "p8"],
        ),
        ReportSection(
            section_title="Evaluation and Benchmarks",
            content=(
                "RAG evaluation spans retrieval quality (recall@k, MRR) and generation quality "
                "(EM, F1, ROUGE). Specialized benchmarks include KILT for knowledge-intensive tasks, "
                "ASQA for ambiguous questions, and RGB for robustness testing. Recent frameworks like "
                "RAGAS provide automated evaluation of faithfulness and relevancy without human labels."
            ),
            cited_papers=["p9", "p10"],
        ),
    ],
    references=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
)


async def evaluate_with_model(model: str) -> CriticFeedback:
    """用指定模型评估报告"""
    config = PipelineConfig(critic_model=model)
    critic = CriticAgent(pipeline_config=config)
    return await critic.run(TEST_REPORT, QUESTION)


async def main() -> None:
    models = ["gpt-4o", "claude-sonnet-4-6-20250514", "gpt-4o-mini"]

    print(f"Multi-LLM Critic Validation")
    print(f"Question: {QUESTION}")
    print(f"Report: {len(TEST_REPORT.sections)} sections, {len(TEST_REPORT.references)} refs")
    print(f"Models: {models}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    results = {}
    for model in models:
        print(f"--- Evaluating with {model} ---")
        try:
            feedback = await evaluate_with_model(model)
            results[model] = {
                "coverage": feedback.scores.coverage,
                "depth": feedback.scores.depth,
                "coherence": feedback.scores.coherence,
                "accuracy": feedback.scores.accuracy,
                "overall": feedback.scores.overall,
                "is_satisfactory": feedback.is_satisfactory,
                "missing_aspects": feedback.missing_aspects[:3],
                "num_new_queries": len(feedback.new_queries),
            }
            s = feedback.scores
            print(
                f"  cov={s.coverage:.1f} dep={s.depth:.1f} "
                f"coh={s.coherence:.1f} acc={s.accuracy:.1f} "
                f"→ overall={s.overall:.1f}"
            )
            print(f"  Missing: {feedback.missing_aspects[:2]}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[model] = {"error": str(e)[:200]}

    # 分析差异
    print(f"\n{'='*60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*60}")

    valid = {k: v for k, v in results.items() if "error" not in v}
    if len(valid) >= 2:
        model_names = list(valid.keys())
        for dim in ["coverage", "depth", "coherence", "accuracy", "overall"]:
            scores = [valid[m][dim] for m in model_names]
            spread = max(scores) - min(scores)
            print(f"  {dim:12s}: {' / '.join(f'{s:.1f}' for s in scores)}  (spread={spread:.1f})")

        # 计算两两差异
        print(f"\nPairwise overall delta:")
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                delta = abs(valid[m1]["overall"] - valid[m2]["overall"])
                print(f"  |{m1} - {m2}| = {delta:.2f}")

    # 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)

    experiment_data = {
        "experiment": "multi_critic_validation",
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "report_sections": len(TEST_REPORT.sections),
        "models": models,
        "results": results,
    }

    output_path = results_dir / "multi_critic_validation.json"
    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
