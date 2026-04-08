"""
LLM Reranker — 消融变量 C

对检索到的 top-k chunk 用 LLM 做精排。
LLM 判断每个 chunk 和 query 的相关度，重新排序。

面试要点:
- Rerank 是"两阶段检索"的第二阶段（recall → precision）
- 比向量相似度更准，但更慢更贵
- 工业界常用 cross-encoder 模型做 rerank（如 BGE-reranker）
- 我们用 LLM 做 rerank，效果更好但成本更高
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from research.core.llm import BaseLLMClient, create_llm_client
from research.core.models import Chunk


class RelevanceScore(BaseModel):
    """Reranker 的单次评分输出"""
    score: float = Field(description="Relevance score from 0.0 to 1.0")
    reason: str = Field(description="Brief reason for the score")


class LLMReranker:
    """LLM Reranker — 用 LLM 对候选 chunk 重排序

    Usage:
        reranker = LLMReranker()
        reranked = await reranker.rerank("What is RAG?", chunks, top_k=10)
    """

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        max_concurrent: int = 5,
    ) -> None:
        self._llm = llm or create_llm_client("openai", model="gpt-4o-mini")
        self._max_concurrent = max_concurrent

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """对 chunk 列表重排序，返回 top_k 个最相关的

        Args:
            query: 用户查询
            chunks: 粗排后的候选 chunk
            top_k: 保留的 chunk 数

        Returns:
            按相关度降序排列的 (chunk, score) 列表
        """
        if not chunks:
            return []

        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _score_one(chunk: Chunk) -> tuple[Chunk, float]:
            async with semaphore:
                score = await self._score_chunk(query, chunk)
                return (chunk, score)

        results = await asyncio.gather(
            *[_score_one(c) for c in chunks],
            return_exceptions=True,
        )

        # 收集成功的结果
        scored: list[tuple[Chunk, float]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            scored.append(result)

        # 按分数降序排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def _score_chunk(self, query: str, chunk: Chunk) -> float:
        """评估单个 chunk 与 query 的相关度"""
        prompt = (
            f"Query: {query}\n\n"
            f"Text passage:\n{chunk.text[:1000]}\n\n"
            "Rate the relevance of this text passage to the query. "
            "Score from 0.0 (completely irrelevant) to 1.0 (directly answers the query)."
        )

        try:
            result = await self._llm.generate_structured(
                messages=[{"role": "user", "content": prompt}],
                response_model=RelevanceScore,
            )
            return max(0.0, min(1.0, result.score))
        except Exception:
            # LLM 调用失败时给默认分数，不中断整体流程
            return 0.5
