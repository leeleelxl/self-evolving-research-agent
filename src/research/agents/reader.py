"""
Reader Agent — 论文精读与结构化信息提取

职责:
1. 对每篇论文做 LLM 调用（优先全文，降级到 abstract）
2. 提取结构化笔记 (PaperNote)
3. 评估与研究问题的相关度
4. 并发处理，错误隔离

支持两种模式:
- Full-text: paper.full_text 有值时，使用 PDF 提取的全文（截断保护）
- Abstract-only: 降级模式，仅用 abstract
"""

from __future__ import annotations

import asyncio

from research.core.agent import BaseAgent
from research.core.config import LLMConfig
from research.core.models import Paper, PaperNote

SYSTEM_PROMPT = """\
You are the Reader Agent in an academic research system.

Your job is to read an academic paper and extract structured information. \
You may receive either a full paper text or just an abstract. \
Be concise but accurate. Focus on what the paper actually says, not what you think it should say.

For relevance scoring:
- 1.0: Directly addresses the research question
- 0.7-0.9: Highly relevant, covers key sub-topics
- 0.4-0.6: Partially relevant, tangentially related
- 0.1-0.3: Marginally relevant
- 0.0: Not relevant at all
"""

READ_PROMPT_TEMPLATE = """\
## Research Question
{question}

## Paper to Analyze
**Title:** {title}
**Authors:** {authors}
**Year:** {year}

{content_section}

Extract a structured PaperNote with:
1. core_contribution: One sentence summarizing the main contribution
2. methodology: Brief description of the approach/method
3. key_findings: List of 2-4 key findings or results
4. limitations: Any limitations mentioned or apparent (can be empty)
5. relevance_score: How relevant is this paper to the research question (0.0 to 1.0)
6. relevance_reason: Why is it relevant or not relevant (one sentence)

Set paper_id to "{paper_id}" and title to "{title}".
"""

# 全文模式下，截断到此长度（约 12k tokens），避免超出 context window
_MAX_FULL_TEXT_CHARS = 30000


class ReaderAgent(BaseAgent):
    """论文精读 Agent

    Usage:
        reader = ReaderAgent()
        notes = await reader.run(papers, "What is RAG?")
    """

    name = "Reader"
    role = SYSTEM_PROMPT

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        super().__init__(llm_config)
        self._max_concurrent = 5  # 最多同时精读 5 篇
        self._min_relevance = 0.3  # 低于此分数的论文过滤掉

    async def run(self, papers: list[Paper], question: str) -> list[PaperNote]:
        """并发精读论文列表，返回过滤后的结构化笔记"""
        if not papers:
            return []

        self.logger.info("reading_start", num_papers=len(papers))

        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _read_one(paper: Paper) -> PaperNote | None:
            async with semaphore:
                return await self._read_paper(paper, question)

        results = await asyncio.gather(
            *[_read_one(p) for p in papers],
            return_exceptions=True,
        )

        # 收集成功的结果，跳过失败和低相关度的
        notes: list[PaperNote] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning("read_failed", paper=papers[i].title[:60], error=str(result))
                continue
            if result is None:
                continue
            if result.relevance_score < self._min_relevance:
                self.logger.debug("low_relevance", paper=result.title[:60], score=result.relevance_score)
                continue
            notes.append(result)

        # 按相关度排序
        notes.sort(key=lambda n: n.relevance_score, reverse=True)

        self.logger.info("reading_done", total=len(papers), kept=len(notes))
        return notes

    async def _read_paper(self, paper: Paper, question: str) -> PaperNote:
        """精读单篇论文 → 结构化笔记

        有全文则用全文（截断保护），无全文则降级到 abstract。
        """
        if paper.full_text:
            text = paper.full_text[:_MAX_FULL_TEXT_CHARS]
            content_section = f"**Full Text (extracted from PDF):**\n{text}"
        else:
            content_section = f"**Abstract:** {paper.abstract}"

        prompt = READ_PROMPT_TEMPLATE.format(
            question=question,
            title=paper.title,
            authors=", ".join(paper.authors[:5]),
            year=paper.year,
            content_section=content_section,
            paper_id=paper.paper_id,
        )
        return await self.generate_structured(prompt, PaperNote)
