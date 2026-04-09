"""
Reference Recall 实验 — 外部 Ground Truth 对标

思路: 用学术界公认的 "essential papers" 列表作为 ground truth，
检查我们的 Pipeline 能找到多少。这是一个不依赖 LLM 判断的客观指标。

Ground truth 来源:
- RAG survey (Gao et al., 2023, arXiv:2312.10997) 的核心引文
- 领域专家公认的 landmark papers
- 通过标题模糊匹配检查我们的 Pipeline 是否检索到了这些论文

运行: python experiments/reference_recall.py
"""

from __future__ import annotations

import asyncio
import json
import re
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

# ============================================================
# Ground Truth: RAG 领域必引论文
# 来源: Gao et al. (2023) "RAG for LLMs: A Survey" + 领域共识
# ============================================================

RAG_ESSENTIAL_PAPERS = [
    # 核心 RAG 论文
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",  # Lewis et al. 2020, 原始 RAG
    "Dense Passage Retrieval for Open-Domain Question Answering",  # Karpukhin et al. 2020, DPR
    "REALM: Retrieval-Augmented Language Model Pre-Training",  # Guu et al. 2020
    "Improving Language Models by Retrieving from Trillions of Tokens",  # Borgeaud et al. 2022, RETRO
    "Atlas: Few-shot Learning with Retrieval Augmented Language Models",  # Izacard et al. 2022
    "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",  # FiD, Izacard & Grave 2021
    "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",  # Asai et al. 2023
    "Active Retrieval Augmented Generation",  # Jiang et al. 2023, FLARE
    # 检索模型
    "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction",  # Khattab & Zaharia 2020
    "Unsupervised Dense Information Retrieval with Contrastive Learning",  # Contriever, Izacard et al. 2022
    # 评估与幻觉
    "A Survey on Hallucination in Large Language Models",  # Huang et al. 2023
    "FActScore: Fine-grained Atomic Evaluation of Factual Precision",  # Min et al. 2023
    # 长上下文 vs RAG
    "Lost in the Middle: How Language Models Use Long Contexts",  # Liu et al. 2023
    # Chunking & Reranking
    "Precise Zero-Shot Dense Retrieval without Relevance Labels",  # HyDE, Gao et al. 2022
]

AGENT_ESSENTIAL_PAPERS = [
    "ReAct: Synergizing Reasoning and Acting in Language Models",  # Yao et al. 2022
    "Toolformer: Language Models Can Teach Themselves to Use Tools",  # Schick et al. 2023
    "Self-Refine: Iterative Refinement with Self-Feedback",  # Madaan et al. 2023
    "Reflexion: Language Agents with Verbal Reinforcement Learning",  # Shinn et al. 2023
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",  # Wei et al. 2022
    "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",  # Yao et al. 2023
    "Generative Agents: Interactive Simulacra of Human Behavior",  # Park et al. 2023
    "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",  # Wu et al. 2023
    "The Landscape of Emerging AI Agent Architectures",  # Masterman et al. 2024
]

BENCHMARKS = {
    "rag": {
        "question": "What are the recent advances in retrieval-augmented generation (RAG) for large language models?",
        "essential_papers": RAG_ESSENTIAL_PAPERS,
    },
    "agents": {
        "question": "What are the recent advances in tool-using capabilities of large language models?",
        "essential_papers": AGENT_ESSENTIAL_PAPERS,
    },
}


def normalize_title(title: str) -> str:
    """标题归一化用于模糊匹配"""
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def compute_recall(
    pipeline_papers: list[dict],
    essential_titles: list[str],
) -> dict:
    """计算 Reference Recall"""
    # 归一化 pipeline 检索到的论文标题
    retrieved_normalized = {normalize_title(p["title"]) for p in pipeline_papers}

    found = []
    missed = []

    for essential_title in essential_titles:
        norm_essential = normalize_title(essential_title)
        # 模糊匹配：检查 essential title 的关键词是否出现在 retrieved 的某个标题中
        matched = False
        for retrieved_title in retrieved_normalized:
            # 用关键词重叠检查（essential 标题的主要词是否出现）
            essential_words = set(norm_essential.split()) - {"the", "a", "an", "of", "for", "in", "and", "with", "to", "via", "by", "from"}
            retrieved_words = set(retrieved_title.split())
            overlap = essential_words & retrieved_words
            # 如果关键词重叠 >= 60%，认为匹配
            if len(overlap) >= len(essential_words) * 0.6:
                matched = True
                break

        if matched:
            found.append(essential_title)
        else:
            missed.append(essential_title)

    recall = len(found) / len(essential_titles) if essential_titles else 0.0

    return {
        "recall": round(recall, 4),
        "found": len(found),
        "total": len(essential_titles),
        "found_titles": found,
        "missed_titles": missed,
    }


async def run_benchmark(name: str, config: dict) -> dict:
    """对一个 benchmark 跑 Pipeline + 计算 recall"""
    question = config["question"]
    essential = config["essential_papers"]

    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Question: {question}")
    print(f"Essential papers: {len(essential)}")
    print(f"{'='*60}\n")

    pipeline_config = PipelineConfig(
        max_iterations=2,
        satisfactory_threshold=8.0,
    )
    pipeline = ResearchPipeline(pipeline_config)

    start = time.time()
    result = await pipeline.run(question)
    elapsed = time.time() - start

    # 提取 pipeline 检索到的论文
    papers_data = [{"title": p.title, "paper_id": p.paper_id} for p in result.papers]

    # 计算 recall
    recall_result = compute_recall(papers_data, essential)

    print(f"\nPipeline: {len(result.papers)} papers, {elapsed:.0f}s")
    print(f"Reference Recall: {recall_result['recall']:.1%} ({recall_result['found']}/{recall_result['total']})")
    print(f"\nFound: {recall_result['found_titles'][:5]}")
    if recall_result["missed_titles"]:
        print(f"Missed: {recall_result['missed_titles'][:5]}")

    return {
        "benchmark": name,
        "question": question,
        "pipeline_papers": len(result.papers),
        "elapsed_seconds": round(elapsed, 1),
        "recall": recall_result,
    }


async def main() -> None:
    print(f"Reference Recall Experiment")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {}
    for name, config in BENCHMARKS.items():
        result = await run_benchmark(name, config)
        results[name] = result

    # 汇总
    print(f"\n{'='*60}")
    print("REFERENCE RECALL SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        r = result["recall"]
        print(f"  {name:10s}: {r['recall']:.1%} ({r['found']}/{r['total']})")

    avg_recall = sum(r["recall"]["recall"] for r in results.values()) / len(results)
    print(f"\n  Average: {avg_recall:.1%}")

    # 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "reference_recall.json"

    experiment_data = {
        "experiment": "reference_recall",
        "timestamp": datetime.now().isoformat(),
        "description": "Reference Recall against curated essential papers lists",
        "benchmarks": results,
        "avg_recall": round(avg_recall, 4),
    }

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
