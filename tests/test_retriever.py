"""RetrieverAgent 测试"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.agents.retriever import RetrieverAgent
from research.core.models import Paper, ResearchPlan, SearchQuery, SearchStrategy


class TestRetrieverUnit:
    """RetrieverAgent 单元测试 — 不调用 API"""

    def test_deduplicate_by_id(self) -> None:
        """paper_id 精确去重"""
        retriever = RetrieverAgent()
        papers = [
            Paper(paper_id="a", title="Paper A", abstract="abs", authors=[], year=2024, url="", source="arxiv"),
            Paper(paper_id="a", title="Paper A dup", abstract="abs", authors=[], year=2024, url="", source="arxiv"),
            Paper(paper_id="b", title="Paper B", abstract="abs", authors=[], year=2024, url="", source="arxiv"),
        ]
        result = retriever._deduplicate(papers)
        assert len(result) == 2

    def test_deduplicate_by_title(self) -> None:
        """标题归一化去重（跨源同一篇论文）"""
        retriever = RetrieverAgent()
        papers = [
            Paper(paper_id="ss_1", title="Attention Is All You Need", abstract="abs", authors=[], year=2017, url="", source="semantic_scholar"),
            Paper(paper_id="arxiv:1706.03762", title="Attention is All You Need.", abstract="abs", authors=[], year=2017, url="", source="arxiv"),
        ]
        result = retriever._deduplicate(papers)
        assert len(result) == 1  # 标题归一化后相同

    def test_normalize_title(self) -> None:
        assert RetrieverAgent._normalize_title("Hello, World!") == "hello world"
        assert RetrieverAgent._normalize_title("  Attention  Is  All ") == "attention is all"


@pytest.mark.integration
class TestRetrieverIntegration:
    """RetrieverAgent 集成测试 — 真实 API 调用"""

    @pytest.mark.asyncio
    async def test_retrieve_from_arxiv(self) -> None:
        """从 arXiv 检索论文"""
        retriever = RetrieverAgent()
        plan = ResearchPlan(
            original_question="test",
            sub_questions=["test"],
            search_strategy=SearchStrategy(
                queries=[SearchQuery(query="LLM agent", source="arxiv", max_results=5)],
                focus_areas=[],
            ),
        )
        papers = await retriever.run(plan)
        assert len(papers) > 0
        assert all(isinstance(p, Paper) for p in papers)
