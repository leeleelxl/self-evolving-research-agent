"""RAG 基础设施测试 — chunking + indexing + reranker"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.core.config import ChunkConfig
from research.core.models import Chunk
from research.retrieval.chunking import (
    FixedChunker,
    RecursiveChunker,
    SemanticChunker,
    create_chunker,
)
from research.retrieval.indexing import DenseIndex, HybridIndex, SparseIndex
from research.retrieval.reranker import LLMReranker

SAMPLE_TEXT = """# Introduction

Retrieval-Augmented Generation (RAG) combines retrieval mechanisms with generative models. This approach has shown significant improvements in knowledge-intensive tasks.

# Related Work

Dense retrieval methods like DPR use dual encoders to learn dense representations. BM25 remains a strong baseline for sparse retrieval. Hybrid methods combine both approaches using reciprocal rank fusion.

# Method

Our approach uses a three-stage pipeline: retrieve, rerank, and generate. The retriever fetches candidate passages using hybrid search. The reranker filters and reorders candidates using a cross-encoder. The generator produces the final output conditioned on retrieved passages.

# Results

We evaluate on three benchmarks: NaturalQuestions, TriviaQA, and HotpotQA. Our hybrid approach outperforms dense-only by 3.2 F1 points and sparse-only by 5.1 F1 points.
"""


class TestChunking:
    """Chunking 策略测试"""

    def test_fixed_chunker(self) -> None:
        chunker = FixedChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT, "paper1")
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.paper_id == "paper1" for c in chunks)
        # chunk_index 应该是递增的
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_semantic_chunker(self) -> None:
        chunker = SemanticChunker(max_chunk_size=500)
        chunks = chunker.chunk(SAMPLE_TEXT, "paper2")
        assert len(chunks) >= 2  # 至少按段落切出 2 个
        # 每个 chunk 不应超过 max_size 太多
        for c in chunks:
            assert len(c.text) <= 600  # 允许一些余量

    def test_recursive_chunker(self) -> None:
        chunker = RecursiveChunker(max_chunk_size=300, min_chunk_size=50)
        chunks = chunker.chunk(SAMPLE_TEXT, "paper3")
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_create_chunker_factory(self) -> None:
        for strategy in ["fixed", "semantic", "recursive"]:
            config = ChunkConfig(strategy=strategy)
            chunker = create_chunker(config)
            assert chunker is not None


class TestIndexing:
    """索引测试"""

    def _make_chunks(self) -> list[Chunk]:
        return [
            Chunk(chunk_id="c1", paper_id="p1", text="RAG combines retrieval with generation", chunk_index=0),
            Chunk(chunk_id="c2", paper_id="p1", text="BM25 is a classic keyword matching algorithm", chunk_index=1),
            Chunk(chunk_id="c3", paper_id="p1", text="FAISS enables fast vector similarity search", chunk_index=2),
        ]

    def _make_embeddings(self) -> list[list[float]]:
        """简单的模拟 embedding（实际应该用 embedding 模型）"""
        return [
            [1.0, 0.0, 0.0, 0.0],  # RAG
            [0.0, 1.0, 0.0, 0.0],  # BM25
            [0.0, 0.0, 1.0, 0.0],  # FAISS
        ]

    def test_dense_index(self) -> None:
        index = DenseIndex()
        chunks = self._make_chunks()
        embeddings = self._make_embeddings()
        index.add(chunks, embeddings)

        assert index.size == 3

        # 查询和第一个 chunk 最相似
        results = index.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0].chunk_id == "c1"  # 最相似
        assert results[0][1] > results[1][1]   # 分数递减

    def test_sparse_index(self) -> None:
        index = SparseIndex()
        chunks = self._make_chunks()
        index.add(chunks)

        assert index.size == 3

        results = index.search("retrieval generation RAG", top_k=2)
        assert len(results) >= 1
        # 包含 "RAG" 和 "retrieval"、"generation" 的 chunk 应该排最前

    def test_hybrid_index(self) -> None:
        index = HybridIndex(weight=0.5)
        chunks = self._make_chunks()
        embeddings = self._make_embeddings()
        index.add(chunks, embeddings)

        results = index.search(
            query="retrieval augmented generation",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
        )
        assert len(results) >= 1
        # RRF 分数应该都是正数
        assert all(score > 0 for _, score in results)


@pytest.mark.integration
class TestReranker:
    """Reranker 集成测试"""

    @pytest.mark.asyncio
    async def test_rerank_orders_by_relevance(self) -> None:
        reranker = LLMReranker()
        chunks = [
            Chunk(chunk_id="c1", paper_id="p1", text="Python is a programming language used for web development.", chunk_index=0),
            Chunk(chunk_id="c2", paper_id="p1", text="Retrieval-Augmented Generation combines a retriever and generator for knowledge-intensive NLP tasks.", chunk_index=1),
            Chunk(chunk_id="c3", paper_id="p1", text="The weather in Tokyo is usually mild in spring.", chunk_index=2),
        ]

        results = await reranker.rerank("What is RAG?", chunks, top_k=3)
        assert len(results) == 3
        # RAG 相关的 chunk 应该排在前面
        assert results[0][0].chunk_id == "c2"
        assert results[0][1] > results[2][1]
