"""KnowledgeBase 测试 — RAG 组件与 Pipeline 的桥梁"""

from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.core.config import KnowledgeBaseConfig, PipelineConfig
from research.core.models import Paper
from research.retrieval.knowledge_base import KnowledgeBase


def _make_papers() -> list[Paper]:
    """构造测试用论文"""
    return [
        Paper(
            paper_id="p1",
            title="Dense Passage Retrieval for Open-Domain QA",
            abstract="We propose DPR, using dense representations for passage retrieval. "
            "DPR uses a dual-encoder architecture with BERT to learn dense embeddings "
            "for questions and passages, achieving strong results on open-domain QA.",
            authors=["Karpukhin"],
            year=2020,
            url="https://arxiv.org/abs/2004.04906",
            source="arxiv",
            citations=3000,
        ),
        Paper(
            paper_id="p2",
            title="BM25 The Next Generation of Lucene Relevance",
            abstract="BM25 is a ranking function used by search engines to estimate "
            "the relevance of documents to a given search query. It is based on "
            "the probabilistic retrieval framework using term frequency and inverse "
            "document frequency.",
            authors=["Robertson"],
            year=2009,
            url="https://example.com/bm25",
            source="semantic_scholar",
            citations=5000,
        ),
        Paper(
            paper_id="p3",
            title="Attention Is All You Need",
            abstract="We propose the Transformer, a model architecture based entirely "
            "on attention mechanisms, dispensing with recurrence and convolutions. "
            "The Transformer allows for significantly more parallelization and "
            "achieves state-of-the-art results in machine translation.",
            authors=["Vaswani"],
            year=2017,
            url="https://arxiv.org/abs/1706.03762",
            source="arxiv",
            citations=80000,
        ),
        Paper(
            paper_id="p4",
            title="Reciprocal Rank Fusion for Information Retrieval",
            abstract="We present reciprocal rank fusion (RRF), a method for combining "
            "rankings from multiple information retrieval systems. RRF yields "
            "consistently better results than any individual system and most other "
            "combination methods on the TREC datasets.",
            authors=["Cormack"],
            year=2009,
            url="https://example.com/rrf",
            source="semantic_scholar",
            citations=800,
        ),
    ]


def _mock_embed(texts: list[str]) -> list[list[float]]:
    """简单的 mock embedding：基于关键词生成固定向量

    retrieval 关键词 → 第 0 维高
    BM25 关键词 → 第 1 维高
    transformer 关键词 → 第 2 维高
    fusion 关键词 → 第 3 维高
    """
    result = []
    for text in texts:
        text_lower = text.lower()
        vec = [0.1, 0.1, 0.1, 0.1]
        if "retrieval" in text_lower or "passage" in text_lower or "dpr" in text_lower:
            vec[0] = 0.9
        if "bm25" in text_lower or "term frequency" in text_lower:
            vec[1] = 0.9
        if "transformer" in text_lower or "attention" in text_lower:
            vec[2] = 0.9
        if "fusion" in text_lower or "combining" in text_lower or "rrf" in text_lower:
            vec[3] = 0.9
        result.append(vec)
    return result


class TestKnowledgeBase:
    """KnowledgeBase 单元测试（mock embedding）"""

    def _create_kb(self, **overrides) -> KnowledgeBase:
        """创建 KnowledgeBase，mock 掉 embedding 模型"""
        config = PipelineConfig(**overrides)
        with patch("research.retrieval.knowledge_base.EmbeddingModel") as mock_cls:
            mock_model = MagicMock()
            mock_model.embed.side_effect = _mock_embed
            mock_model.embed_single.side_effect = lambda t: _mock_embed([t])[0]
            mock_model.dimension = 4
            mock_cls.return_value = mock_model
            kb = KnowledgeBase(config)
        return kb

    def test_add_papers(self) -> None:
        """测试论文索引"""
        kb = self._create_kb()
        papers = _make_papers()

        kb.add_papers(papers)

        assert kb.num_papers == 4
        assert kb.size > 0  # chunk 数 >= 论文��

    def test_add_papers_incremental(self) -> None:
        """测试增量索引：重复添加不会重复索引"""
        kb = self._create_kb()
        papers = _make_papers()

        kb.add_papers(papers[:2])
        size_after_first = kb.size
        num_after_first = kb.num_papers

        kb.add_papers(papers)  # 包含前 2 篇 + 新的 2 篇
        assert kb.num_papers == 4
        assert kb.size > size_after_first  # 新 chunk 被添加
        assert num_after_first == 2

    def test_retrieve_relevance(self) -> None:
        """测试检索相关性：retrieval 相关的 query 应该返回 retrieval 论文"""
        kb = self._create_kb()
        kb.add_papers(_make_papers())

        results = kb.retrieve("dense passage retrieval methods", top_k=2)
        assert len(results) >= 1
        # DPR 论文应该排在前面
        paper_ids = [p.paper_id for p in results]
        assert "p1" in paper_ids

    def test_retrieve_transformer_query(self) -> None:
        """测试 Transformer 相关的 query"""
        kb = self._create_kb()
        kb.add_papers(_make_papers())

        results = kb.retrieve("transformer attention mechanism", top_k=2)
        assert len(results) >= 1
        paper_ids = [p.paper_id for p in results]
        assert "p3" in paper_ids

    def test_retrieve_empty_kb(self) -> None:
        """空知识库返回空列表"""
        kb = self._create_kb()
        results = kb.retrieve("anything", top_k=5)
        assert results == []

    def test_retrieve_for_questions_dedup(self) -> None:
        """多问题检索：结果应去重"""
        kb = self._create_kb()
        kb.add_papers(_make_papers())

        results = kb.retrieve_for_questions(
            ["dense retrieval", "passage retrieval DPR"],
            top_k=10,
        )
        # 去重：同一篇论文不应出现多次
        paper_ids = [p.paper_id for p in results]
        assert len(paper_ids) == len(set(paper_ids))

    def test_retrieve_for_questions_coverage(self) -> None:
        """多问题检索：不同主题的问题应覆盖更多论文"""
        kb = self._create_kb()
        kb.add_papers(_make_papers())

        results = kb.retrieve_for_questions(
            ["dense passage retrieval", "transformer attention architecture"],
            top_k=3,
        )
        paper_ids = [p.paper_id for p in results]
        # 两个不同主题应该命中不同的论文
        assert "p1" in paper_ids  # retrieval 论文
        assert "p3" in paper_ids  # transformer 论文

    def test_retrieve_for_empty_questions(self) -> None:
        """空问题列表返回空"""
        kb = self._create_kb()
        kb.add_papers(_make_papers())

        results = kb.retrieve_for_questions([], top_k=5)
        assert results == []

    def test_sparse_index_mode(self) -> None:
        """Sparse-only 模式也能正常工作"""
        config = PipelineConfig(
            retrieval={"strategy": "sparse"},
        )
        with patch("research.retrieval.knowledge_base.EmbeddingModel") as mock_cls:
            mock_model = MagicMock()
            mock_model.embed.side_effect = _mock_embed
            mock_model.embed_single.side_effect = lambda t: _mock_embed([t])[0]
            mock_cls.return_value = mock_model
            kb = KnowledgeBase(config)

        kb.add_papers(_make_papers())
        results = kb.retrieve("BM25 term frequency", top_k=2)
        assert len(results) >= 1


@pytest.mark.integration
class TestKnowledgeBaseIntegration:
    """KnowledgeBase 集成测试（真实 fastembed）"""

    def test_real_embedding_pipeline(self) -> None:
        """使用真实 embedding 模型的端到端测试"""
        config = PipelineConfig()
        kb = KnowledgeBase(config)
        papers = _make_papers()

        kb.add_papers(papers)
        assert kb.num_papers == 4

        # Retrieval 相关的 query
        results = kb.retrieve("dense retrieval passage ranking", top_k=2)
        assert len(results) >= 1

        # 多问题检索
        results = kb.retrieve_for_questions(
            ["how does BM25 work", "what is the transformer architecture"],
            top_k=3,
        )
        assert len(results) >= 2
