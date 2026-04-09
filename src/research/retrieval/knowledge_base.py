"""
KnowledgeBase — 论文知识库，连接 RAG 组件和 Pipeline

解决的核心问题:
  RAG 模块 (chunking/indexing/embedding/reranker) 之前和 Pipeline 脱节。
  KnowledgeBase 把它们组合成一个可用的整体，嵌入 Pipeline 的检索-精读流程。

工作流程:
  1. Retriever 搜到 50 篇论文 → add_papers() 把 abstract 建索引
  2. 对每个 sub-question → retrieve() 从索引中检索最相关的论文
  3. Reader 只精读检索到的论文，而非全部

设计决策:
  - 以 abstract 为文本源: 当前阶段没有 PDF 全文，abstract 足够支撑
  - 使用 chunking 模块: 虽然 abstract 短，但保持架构一致性，
    后续接 PDF 全文时同一套 KnowledgeBase 直接复用
  - 跨迭代累积: Pipeline 的多轮自进化中，知识库持续扩展，不丢弃
"""

from __future__ import annotations

import structlog

from research.core.config import PipelineConfig
from research.core.models import Chunk, Paper
from research.retrieval.chunking import create_chunker
from research.retrieval.embedding import EmbeddingModel
from research.retrieval.indexing import DenseIndex, HybridIndex, SparseIndex

logger = structlog.get_logger()


class KnowledgeBase:
    """论文知识库 — 封装 chunking + embedding + indexing

    Usage:
        kb = KnowledgeBase(config)
        kb.add_papers(papers)  # 索引论文
        relevant = kb.retrieve("What is dense retrieval?", top_k=10)
        relevant_multi = kb.retrieve_for_questions(
            ["What is RAG?", "How does BM25 work?"], top_k=10
        )
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._logger = logger.bind(component="knowledge_base")

        # 初始化 RAG 组件
        self._chunker = create_chunker(self._config.chunk)
        self._embedding = EmbeddingModel(
            model_name=self._config.knowledge_base.embedding_model,
        )

        # 根据配置选择索引类型
        retrieval_cfg = self._config.retrieval
        if retrieval_cfg.strategy == "dense":
            self._index = DenseIndex()
        elif retrieval_cfg.strategy == "sparse":
            self._index = SparseIndex()
        else:
            self._index = HybridIndex(weight=retrieval_cfg.hybrid_weight)

        # paper_id → Paper 映射，用于从 chunk 反查论文
        self._paper_map: dict[str, Paper] = {}
        # 已索引的 paper_id，防止重复索引
        self._indexed_ids: set[str] = set()

    def add_papers(self, papers: list[Paper]) -> None:
        """将论文 abstract 索引到知识库（增量式，跳过已有的）"""
        new_papers = [p for p in papers if p.paper_id not in self._indexed_ids]
        if not new_papers:
            self._logger.info("no_new_papers_to_index")
            return

        self._logger.info("indexing_papers", count=len(new_papers))

        # 1. Chunk: 将每篇论文的 abstract 切分
        all_chunks: list[Chunk] = []
        for paper in new_papers:
            text = f"{paper.title}\n\n{paper.abstract}"
            chunks = self._chunker.chunk(text, paper.paper_id)
            all_chunks.extend(chunks)
            self._paper_map[paper.paper_id] = paper
            self._indexed_ids.add(paper.paper_id)

        # 2. Embed: 批量生成 embedding
        texts = [c.text for c in all_chunks]
        embeddings = self._embedding.embed(texts)

        # 3. Index: 添加到索引
        if isinstance(self._index, HybridIndex):
            self._index.add(all_chunks, embeddings)
        elif isinstance(self._index, DenseIndex):
            self._index.add(all_chunks, embeddings)
        else:
            # SparseIndex 不需要 embedding
            self._index.add(all_chunks)

        self._logger.info(
            "indexing_done",
            new_papers=len(new_papers),
            new_chunks=len(all_chunks),
            total_index_size=self._index.size,
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[Paper]:
        """检索与 query 最相关的论文

        流程: query → embedding → 索引检索 → chunk → 反查 paper → 去重
        """
        if self._index.size == 0:
            return []

        retrieval_cfg = self._config.retrieval

        if isinstance(self._index, HybridIndex):
            query_emb = self._embedding.embed_single(query)
            results = self._index.search(
                query=query,
                query_embedding=query_emb,
                top_k=retrieval_cfg.top_k,
            )
        elif isinstance(self._index, DenseIndex):
            query_emb = self._embedding.embed_single(query)
            results = self._index.search(query_emb, top_k=retrieval_cfg.top_k)
        else:
            results = self._index.search(query, top_k=retrieval_cfg.top_k)

        # 从 chunk 反查 paper，保持排序，去重
        seen: set[str] = set()
        papers: list[Paper] = []
        for chunk, _score in results:
            if chunk.paper_id not in seen and chunk.paper_id in self._paper_map:
                seen.add(chunk.paper_id)
                papers.append(self._paper_map[chunk.paper_id])
                if len(papers) >= top_k:
                    break

        return papers

    def retrieve_for_questions(
        self,
        questions: list[str],
        top_k: int = 10,
    ) -> list[Paper]:
        """对多个子问题分别检索，合并去重，保持相关度排序

        这是 Pipeline 的主要入口: Planner 分解出 N 个子问题，
        每个子问题检索 top_k 篇，合并后去重。
        """
        if not questions:
            return []

        seen: set[str] = set()
        all_papers: list[Paper] = []

        for question in questions:
            papers = self.retrieve(question, top_k=top_k)
            for paper in papers:
                if paper.paper_id not in seen:
                    seen.add(paper.paper_id)
                    all_papers.append(paper)

        self._logger.info(
            "retrieve_for_questions",
            num_questions=len(questions),
            total_unique_papers=len(all_papers),
        )
        return all_papers

    @property
    def size(self) -> int:
        """当前知识库中的 chunk 数"""
        return self._index.size

    @property
    def num_papers(self) -> int:
        """当前知识库中的论文数"""
        return len(self._indexed_ids)
