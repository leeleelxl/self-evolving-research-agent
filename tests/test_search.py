"""检索客户端测试"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.core.models import Paper
from research.retrieval.search import ArxivClient, SemanticScholarClient, create_search_client


class TestSearchFactory:
    """工厂函数测试"""

    def test_create_semantic_scholar(self) -> None:
        client = create_search_client("semantic_scholar")
        assert isinstance(client, SemanticScholarClient)

    def test_create_arxiv(self) -> None:
        client = create_search_client("arxiv")
        assert isinstance(client, ArxivClient)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            create_search_client("unknown")


@pytest.mark.integration
class TestSemanticScholarIntegration:
    """Semantic Scholar API 集成测试 — 真实调用"""

    @pytest.mark.asyncio
    async def test_search_returns_papers(self) -> None:
        client = SemanticScholarClient()
        papers = await client.search("retrieval augmented generation", max_results=5)

        # Semantic Scholar 无 key 时容易 429，返回空列表不算失败
        if len(papers) == 0:
            pytest.skip("Semantic Scholar rate limited (429), skipping")

        assert all(isinstance(p, Paper) for p in papers)

        paper = papers[0]
        assert paper.paper_id
        assert paper.title
        assert paper.abstract
        assert paper.source == "semantic_scholar"

    @pytest.mark.asyncio
    async def test_search_empty_query(self) -> None:
        client = SemanticScholarClient()
        papers = await client.search("xyznonexistentquery12345", max_results=5)
        # 可能返回空，也可能返回少量结果，不应报错
        assert isinstance(papers, list)


@pytest.mark.integration
class TestArxivIntegration:
    """arXiv API 集成测试 — 真实调用"""

    @pytest.mark.asyncio
    async def test_search_returns_papers(self) -> None:
        client = ArxivClient()
        papers = await client.search("transformer attention mechanism", max_results=5)

        assert len(papers) > 0
        assert all(isinstance(p, Paper) for p in papers)

        paper = papers[0]
        assert paper.paper_id.startswith("arxiv:")
        assert paper.title
        assert paper.abstract
        assert paper.source == "arxiv"
        assert paper.year > 0

    @pytest.mark.asyncio
    async def test_search_with_few_results(self) -> None:
        client = ArxivClient()
        papers = await client.search("LLM agents", max_results=3)
        assert len(papers) <= 3
