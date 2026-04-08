"""
RAG 消融实验 — chunk × retrieval × rerank 全组合对比

在 HotpotQA dev set 上评估不同 RAG 配置的效果。

HotpotQA 的结构:
- question: 多跳问题（需要综合多个段落回答）
- context: 10 个段落（含正确答案的段落 + 干扰段落）
- answer: ground truth 答案
- supporting_facts: 哪些段落是支撑事实

实验流程:
1. 把 context 段落 chunk 化
2. 建索引（dense / sparse / hybrid）
3. 用 question 检索 top-k chunk
4. 用 LLM 基于检索到的 chunk 回答问题
5. 计算 EM (Exact Match) 和 F1

运行: python experiments/rag_ablation.py [--num_samples 50]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import string
import sys
import time
from collections import Counter
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

from research.core.config import ChunkConfig
from research.core.llm import create_llm_client
from research.core.models import Chunk
from research.retrieval.chunking import create_chunker
from research.retrieval.indexing import DenseIndex, HybridIndex, SparseIndex

logger = structlog.get_logger()


# ── 评估指标 ──

def normalize_answer(s: str) -> str:
    """答案归一化：小写 + 去标点 + 去冠词 + 合并空格"""
    s = s.lower()
    # 去标点
    s = "".join(c for c in s if c not in string.punctuation)
    # 去冠词
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # 合并空格
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ── 数据加载 ──

def load_hotpotqa(path: str, num_samples: int = 50) -> list[dict]:
    """加载 HotpotQA 数据"""
    with open(path) as f:
        data = json.load(f)
    return data[:num_samples]


def extract_paragraphs(sample: dict) -> list[str]:
    """从 HotpotQA 样本提取所有段落文本"""
    paragraphs = []
    for title, sentences in sample["context"]:
        text = " ".join(sentences)
        paragraphs.append(f"{title}: {text}")
    return paragraphs


# ── Embedding 模型（全局单例，避免重复加载） ──

_embedding_model = None

def get_embedding_model():
    """延迟加载 embedding 模型（全局单例）"""
    global _embedding_model
    if _embedding_model is None:
        from research.retrieval.embedding import EmbeddingModel
        _embedding_model = EmbeddingModel()
    return _embedding_model


# ── 单次实验 ──

async def run_single_config(
    samples: list[dict],
    chunk_strategy: str,
    retrieval_strategy: str,
    llm_client,
    top_k: int = 5,
) -> dict:
    """用指定配置在数据集上评估"""

    chunk_config = ChunkConfig(strategy=chunk_strategy, chunk_size=300, chunk_overlap=30, max_chunk_size=500)
    chunker = create_chunker(chunk_config)

    em_scores = []
    f1_scores = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        answer = sample["answer"]
        paragraphs = extract_paragraphs(sample)

        # 1. Chunk
        all_chunks: list[Chunk] = []
        for j, para in enumerate(paragraphs):
            chunks = chunker.chunk(para, paper_id=f"doc_{j}")
            all_chunks.extend(chunks)

        # 批量 embedding（比逐条快很多）
        emb_model = get_embedding_model()
        all_embeddings = emb_model.embed([c.text for c in all_chunks])

        # 2. Index + Retrieve
        if retrieval_strategy == "dense":
            index = DenseIndex()
            index.add(all_chunks, all_embeddings)
            q_emb = emb_model.embed_single(question)
            results = index.search(q_emb, top_k=top_k)
        elif retrieval_strategy == "sparse":
            index = SparseIndex()
            index.add(all_chunks)
            results = index.search(question, top_k=top_k)
        else:  # hybrid
            index = HybridIndex()
            index.add(all_chunks, all_embeddings)
            q_emb = emb_model.embed_single(question)
            results = index.search(question, q_emb, top_k=top_k)

        # 3. 用 LLM 回答
        context = "\n\n".join(c.text for c, _ in results)
        prompt = (
            f"Answer the question based ONLY on the context below. "
            f"Be concise — answer in as few words as possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        try:
            pred = await llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
            )
            pred = pred.strip()
        except Exception:
            pred = ""

        em = exact_match(pred, answer)
        f1 = f1_score(pred, answer)
        em_scores.append(em)
        f1_scores.append(f1)

        if (i + 1) % 10 == 0:
            avg_em = sum(em_scores) / len(em_scores)
            avg_f1 = sum(f1_scores) / len(f1_scores)
            print(f"  [{chunk_strategy}×{retrieval_strategy}] {i+1}/{len(samples)} — EM={avg_em:.3f} F1={avg_f1:.3f}")

    return {
        "chunk_strategy": chunk_strategy,
        "retrieval_strategy": retrieval_strategy,
        "num_samples": len(samples),
        "exact_match": round(sum(em_scores) / len(em_scores), 4),
        "f1_score": round(sum(f1_scores) / len(f1_scores), 4),
    }


# ── 主实验 ──

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    data_path = project_root / "data" / "hotpotqa_dev_200.json"
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Run: python -c \"import httpx,json; ...\" to download first")
        return

    samples = load_hotpotqa(str(data_path), num_samples=args.num_samples)
    llm_client = create_llm_client("openai", model="gpt-4o-mini")

    print(f"RAG Ablation Experiment")
    print(f"Samples: {len(samples)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # 6 组关键消融组合
    configs = [
        # 固定 hybrid 检索，对比 chunk 策略
        ("fixed", "hybrid"),
        ("semantic", "hybrid"),
        ("recursive", "hybrid"),
        # 固定 recursive chunk，对比检索策略
        ("recursive", "dense"),
        ("recursive", "sparse"),
        # recursive + hybrid 已在上面，不重复
    ]

    results = []
    for chunk_s, retrieval_s in configs:
        print(f"\n--- Config: chunk={chunk_s}, retrieval={retrieval_s} ---")
        start = time.time()
        result = await run_single_config(samples, chunk_s, retrieval_s, llm_client)
        result["elapsed_seconds"] = round(time.time() - start, 1)
        results.append(result)
        print(f"  Result: EM={result['exact_match']:.4f} F1={result['f1_score']:.4f} ({result['elapsed_seconds']}s)")

    # 输出汇总表格
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Chunk':<12} {'Retrieval':<10} {'EM':<8} {'F1':<8} {'Time(s)':<8}")
    print("-" * 50)
    for r in results:
        print(f"{r['chunk_strategy']:<12} {r['retrieval_strategy']:<10} {r['exact_match']:<8.4f} {r['f1_score']:<8.4f} {r['elapsed_seconds']:<8.1f}")

    # 找最佳组合
    best = max(results, key=lambda x: x["f1_score"])
    print(f"\nBest: {best['chunk_strategy']} × {best['retrieval_strategy']} (F1={best['f1_score']:.4f})")

    # 保存结果
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "rag_ablation.json"

    experiment_data = {
        "experiment": "rag_ablation",
        "timestamp": datetime.now().isoformat(),
        "dataset": "hotpotqa_dev",
        "num_samples": len(samples),
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "top_k": 5,
            "embedding": "BAAI/bge-small-en-v1.5 (fastembed, 384d)",
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
