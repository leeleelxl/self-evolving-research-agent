"""
Retriever Agent — 文献检索执行

职责:
1. 接收 Planner 生成的 SearchStrategy
2. 并发执行多个检索 query（Semantic Scholar / arXiv）
3. 去重、按引用数排序、截断

设计要点:
- Retriever 不做"思考"，只做"执行"。检索策略由 Planner 决定。
- 用 asyncio.Semaphore 控制并发数，避免触发 API 速率限制。
- 去重策略: paper_id 精确匹配 + 标题归一化模糊匹配。
"""

from __future__ import annotations

import asyncio
import re

import structlog

from research.core.agent import BaseAgent
from research.core.config import LLMConfig, PipelineConfig
from research.core.models import Paper, ResearchPlan
from research.retrieval.search import create_search_client

logger = structlog.get_logger()


class RetrieverAgent(BaseAgent):
    """文献检索 Agent

    Usage:
        retriever = RetrieverAgent(config)
        papers = await retriever.run(plan)
    """

    name = "Retriever"
    role = "Execute search queries and retrieve academic papers from Semantic Scholar and arXiv."

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        super().__init__(llm_config)
        config = pipeline_config or PipelineConfig()
        self._max_papers = config.max_papers_total
        self._max_concurrent = 3  # 最多同时 3 个 API 请求，避免限速

    async def run(self, plan: ResearchPlan) -> list[Paper]:
        """执行检索策略，返回去重排序后的论文列表"""
        queries = plan.search_strategy.queries
        self.logger.info("retrieval_start", num_queries=len(queries))

        # 并发执行所有 query，用 Semaphore 控制并发数
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _search_with_limit(query):
            async with semaphore:
                client = create_search_client(query.source)
                return await client.search(query.query, max_results=query.max_results)

        results = await asyncio.gather(
            *[_search_with_limit(q) for q in queries],
            return_exceptions=True,
        )

        # 收集所有论文，跳过失败的 query
        all_papers: list[Paper] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(
                    "query_failed",
                    query=queries[i].query,
                    error=str(result),
                )
                continue
            all_papers.extend(result)

        self.logger.info("raw_results", total=len(all_papers))

        # 去重
        unique_papers = self._deduplicate(all_papers)
        self.logger.info("after_dedup", count=len(unique_papers))

        # 按引用数降序排列，截断
        unique_papers.sort(key=lambda p: p.citations, reverse=True)
        final = unique_papers[: self._max_papers]

        self.logger.info("retrieval_done", final_count=len(final))
        return final

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """去重：paper_id 精确匹配 + 标题归一化模糊匹配

        为什么需要标题匹配？
        同一篇论文在 Semantic Scholar 和 arXiv 上有不同的 ID，
        但标题基本相同。归一化后对比可以识别跨源重复。
        """
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[Paper] = []

        for paper in papers:
            # 跳过 ID 重复
            if paper.paper_id in seen_ids:
                continue

            # 跳过标题重复（归一化：小写 + 去除标点和多余空格）
            normalized_title = self._normalize_title(paper.title)
            if normalized_title in seen_titles:
                continue

            seen_ids.add(paper.paper_id)
            seen_titles.add(normalized_title)
            unique.append(paper)

        return unique

    @staticmethod
    def _normalize_title(title: str) -> str:
        """标题归一化：小写 + 去标点 + 合并空格"""
        title = title.lower()
        title = re.sub(r"[^\w\s]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title
