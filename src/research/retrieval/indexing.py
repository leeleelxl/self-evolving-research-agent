"""
向量/文本索引 — 消融变量 B

三种检索策略:
1. Dense: FAISS 向量检索（语义相似度）
2. Sparse: BM25 关键词检索（词频匹配）
3. Hybrid: RRF 融合（两者互补，通常效果最好）

面试重点: Dense 擅长同义改写，Sparse 擅长精确术语，Hybrid 取长补短。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from research.core.models import Chunk


class DenseIndex:
    """FAISS 向量索引 — Dense Retrieval

    用 numpy 实现简单的余弦相似度检索。
    生产环境用 FAISS，但核心逻辑一样。
    这样面试时能讲清楚底层原理，不被问到"FAISS 内部怎么工作的"。

    后续如果数据量大，直接换成 faiss.IndexFlatIP 即可。
    """

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None  # (n, dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """添加 chunk 及其 embedding 到索引"""
        new_emb = np.array(embeddings, dtype=np.float32)
        # L2 归一化，这样内积 = 余弦相似度
        norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        new_emb = new_emb / norms

        if self._embeddings is None:
            self._embeddings = new_emb
        else:
            self._embeddings = np.vstack([self._embeddings, new_emb])
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[tuple[Chunk, float]]:
        """检索最相似的 chunk，返回 (chunk, score) 列表"""
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1)

        # 余弦相似度 = 归一化后的内积
        scores = self._embeddings @ q
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    @property
    def size(self) -> int:
        return len(self._chunks)


class SparseIndex:
    """BM25 关键词索引 — Sparse Retrieval

    BM25 是信息检索的经典算法，基于 TF-IDF 的改进。
    核心思想: 一个词在文档中出现越多（TF 高）且在其他文档中出现越少（IDF 高），
    该文档越可能与包含这个词的 query 相关。

    面试常问: "BM25 和 TF-IDF 有什么区别？"
    答: BM25 加了文档长度归一化和饱和函数，防止高频词过度影响。
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """添加 chunk 到 BM25 索引"""
        self._chunks.extend(chunks)
        # BM25 需要重建索引（不支持增量添加）
        tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """关键词检索"""
        if self._bm25 is None or len(self._chunks) == 0:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self._chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词: 小写 + 按空格切"""
        return text.lower().split()

    @property
    def size(self) -> int:
        return len(self._chunks)


class HybridIndex:
    """混合索引 — Reciprocal Rank Fusion (RRF)

    RRF 算法: 对 Dense 和 Sparse 的排名列表做融合。
    每个文档的 RRF 分数 = Σ 1/(k + rank_i)，k=60 是常用参数。

    为什么不用简单的分数加权？
    因为 Dense 和 Sparse 的分数尺度不同（余弦相似度 vs BM25 分数），
    直接加权没有意义。RRF 基于排名而非分数，天然归一化。
    """

    def __init__(self, weight: float = 0.5) -> None:
        """
        Args:
            weight: Dense 的权重 (0-1)，1-weight 给 Sparse
        """
        self.dense = DenseIndex()
        self.sparse = SparseIndex()
        self._weight = weight
        self._rrf_k = 60  # RRF 参数，60 是论文推荐值

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """同时添加到 Dense 和 Sparse 索引"""
        self.dense.add(chunks, embeddings)
        self.sparse.add(chunks)

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[tuple[Chunk, float]]:
        """混合检索：RRF 融合 Dense + Sparse 结果"""
        # 分别检索，取较多的候选
        fetch_k = top_k * 3
        dense_results = self.dense.search(query_embedding, top_k=fetch_k)
        sparse_results = self.sparse.search(query, top_k=fetch_k)

        # 计算 RRF 分数
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            rrf = self._weight / (self._rrf_k + rank + 1)
            chunk_scores[chunk.chunk_id] = chunk_scores.get(chunk.chunk_id, 0) + rrf
            chunk_map[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(sparse_results):
            rrf = (1 - self._weight) / (self._rrf_k + rank + 1)
            chunk_scores[chunk.chunk_id] = chunk_scores.get(chunk.chunk_id, 0) + rrf
            chunk_map[chunk.chunk_id] = chunk

        # 按 RRF 分数排序
        sorted_ids = sorted(chunk_scores, key=lambda x: chunk_scores[x], reverse=True)
        return [(chunk_map[cid], chunk_scores[cid]) for cid in sorted_ids[:top_k]]

    @property
    def size(self) -> int:
        return self.dense.size
