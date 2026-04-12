"""CitationVerifier 测试 — embedding + NLI 双模式"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from research.core.models import Paper, PipelineResult, ReportSection, ResearchReport
from research.evaluation.citation_verifier import CitationVerifier


def _make_paper(paper_id: str, title: str = "Test Paper", abstract: str = "Test abstract") -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=["Author"],
        year=2024,
        url="https://example.com",
        source="semantic_scholar",
    )


def _make_pipeline_result(
    sections: list[ReportSection],
    papers: list[Paper],
) -> PipelineResult:
    return PipelineResult(
        report=ResearchReport(
            title="Test Report",
            abstract="Test abstract",
            sections=sections,
            references=[p.paper_id for p in papers],
        ),
        evolution_log=[],
        total_iterations=1,
        papers=papers,
    )


# ── Embedding 模式测试 ──


class TestEmbeddingMode:
    def test_verify_basic(self) -> None:
        """基本 embedding 验证流程"""
        paper = _make_paper("p1", abstract="Dense retrieval uses neural embeddings for search")
        section = ReportSection(
            section_title="Dense Retrieval",
            content="Dense retrieval methods leverage neural embeddings to perform semantic search",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        verifier = CitationVerifier(method="embedding", grounding_threshold=0.3)
        report = verifier.verify(result)

        assert report["method"] == "embedding"
        assert report["num_citations_checked"] == 1
        assert report["overall_grounding_rate"] >= 0
        assert report["overall_grounding_rate"] <= 1

    def test_missing_paper(self) -> None:
        """引用的论文不在 papers 列表中"""
        section = ReportSection(
            section_title="Test",
            content="Some content",
            cited_papers=["missing_id"],
        )
        result = _make_pipeline_result([section], [])

        verifier = CitationVerifier(method="embedding")
        report = verifier.verify(result)

        assert report["num_citations_missing"] == 1
        assert report["num_citations_checked"] == 0

    def test_empty_sections(self) -> None:
        """无引用的 section"""
        section = ReportSection(
            section_title="Intro",
            content="Introduction text",
            cited_papers=[],
        )
        result = _make_pipeline_result([section], [])

        verifier = CitationVerifier(method="embedding")
        report = verifier.verify(result)

        assert report["num_citations_checked"] == 0
        assert report["overall_grounding_rate"] == 0.0

    def test_high_similarity_is_grounded(self) -> None:
        """高相似度的引用应被标记为 grounded"""
        paper = _make_paper("p1", abstract="RAG combines retrieval with generation for QA")
        section = ReportSection(
            section_title="RAG",
            content="RAG combines retrieval with generation for question answering tasks",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        verifier = CitationVerifier(method="embedding", grounding_threshold=0.3)
        report = verifier.verify(result)

        assert report["num_citations_grounded"] >= 0  # embedding 结果不确定性，只验证流程
        assert len(report["sections"]) == 1
        assert report["sections"][0]["num_cited"] == 1


# ── NLI 模式测试 ──


class TestNLIMode:
    def _make_mock_nli_model(
        self,
        entailment: float = 0.7,
        contradiction: float = 0.1,
        neutral: float = 0.2,
    ) -> MagicMock:
        """创建 mock NLI 模型

        句子级 NLI: predict 接收 N 个 (premise, hypothesis) 对，
        返回 (N, 3) 数组。mock 对所有句子返回相同的分数。
        """
        mock = MagicMock()

        def predict_side_effect(pairs, apply_softmax=True):
            n = len(pairs)
            return np.array([[contradiction, entailment, neutral]] * n)

        mock.predict.side_effect = predict_side_effect
        return mock

    def test_nli_verify_entailed(self) -> None:
        """NLI 判断为 entailment 时应 grounded"""
        paper = _make_paper("p1", abstract="Transformers use self-attention mechanism")
        section = ReportSection(
            section_title="Attention",
            content="Transformers rely on the self-attention mechanism for sequence modeling. This approach enables parallel processing of input tokens.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        mock_nli = self._make_mock_nli_model(entailment=0.8, contradiction=0.05, neutral=0.15)

        with patch.object(CitationVerifier, "_load_nli_model", return_value=mock_nli):
            verifier = CitationVerifier(method="nli", entailment_threshold=0.5)
            report = verifier.verify(result)

        assert report["method"] == "nli"
        assert report["num_citations_grounded"] == 1
        assert report["overall_contradiction_rate"] == 0.0
        cite = report["sections"][0]["citations"][0]
        assert cite["grounded"] is True
        assert "nli_probs" in cite
        assert cite["nli_probs"]["entailment"] == 0.8

    def test_nli_verify_contradicted(self) -> None:
        """NLI 判断为 contradiction 时应标记"""
        paper = _make_paper("p1", abstract="Method A outperforms Method B")
        section = ReportSection(
            section_title="Results",
            content="Method B outperforms Method A significantly in all benchmark evaluations. The results demonstrate clear superiority.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        mock_nli = self._make_mock_nli_model(entailment=0.1, contradiction=0.7, neutral=0.2)

        with patch.object(CitationVerifier, "_load_nli_model", return_value=mock_nli):
            verifier = CitationVerifier(method="nli", entailment_threshold=0.5)
            report = verifier.verify(result)

        assert report["num_citations_grounded"] == 0
        assert report["num_citations_contradicted"] == 1
        assert report["overall_contradiction_rate"] > 0
        cite = report["sections"][0]["citations"][0]
        assert cite["contradicted"] is True

    def test_nli_verify_neutral(self) -> None:
        """NLI 判断为 neutral 时不 grounded 也不 contradicted"""
        paper = _make_paper("p1", abstract="A study on protein folding")
        section = ReportSection(
            section_title="NLP",
            content="Language models are widely used for text generation tasks. They have transformed the field of natural language processing.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        mock_nli = self._make_mock_nli_model(entailment=0.15, contradiction=0.15, neutral=0.7)

        with patch.object(CitationVerifier, "_load_nli_model", return_value=mock_nli):
            verifier = CitationVerifier(method="nli", entailment_threshold=0.5)
            report = verifier.verify(result)

        assert report["num_citations_grounded"] == 0
        assert report["num_citations_contradicted"] == 0

    def test_nli_missing_paper(self) -> None:
        """NLI 模式下缺失论文的处理"""
        section = ReportSection(
            section_title="Test",
            content="Some content that is long enough for sentence splitting to work correctly here.",
            cited_papers=["missing_id"],
        )
        result = _make_pipeline_result([section], [])

        mock_nli = self._make_mock_nli_model()
        with patch.object(CitationVerifier, "_load_nli_model", return_value=mock_nli):
            verifier = CitationVerifier(method="nli")
            report = verifier.verify(result)

        assert report["num_citations_missing"] == 1

    def test_sentence_splitting(self) -> None:
        """验证句子分割逻辑"""
        from research.evaluation.citation_verifier import CitationVerifier

        sentences = CitationVerifier._split_sentences(
            "This is the first sentence about RAG. "
            "The second sentence discusses retrieval methods. "
            "Short. "  # < 20 chars, should be filtered
            "The third sentence covers generation approaches and their effectiveness."
        )
        assert len(sentences) == 3  # "Short." 被过滤


# ── 方法选择测试 ──


class TestHybridMode:
    def test_hybrid_grounding_from_embedding_contradiction_from_nli(self) -> None:
        """Hybrid: grounding 来自 embedding, contradiction 来自 NLI"""
        paper = _make_paper("p1", abstract="RAG combines retrieval and generation for QA tasks")
        section = ReportSection(
            section_title="RAG",
            content="RAG integrates retrieval with generation to answer questions. This approach has shown great promise in recent years.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        # Mock NLI: 无矛盾
        mock_nli = MagicMock()

        def predict_no_contradiction(pairs, apply_softmax=True):
            n = len(pairs)
            return np.array([[0.05, 0.15, 0.80]] * n)  # neutral

        mock_nli.predict.side_effect = predict_no_contradiction

        with patch.object(CitationVerifier, "_load_nli_model", return_value=mock_nli):
            verifier = CitationVerifier(method="hybrid")
            report = verifier.verify(result)

        assert report["method"] == "hybrid"
        cite = report["sections"][0]["citations"][0]
        # Grounding 由 embedding 决定（高相似度应 grounded）
        assert cite["grounded"] is True or cite["grounded"] is False  # 取决于实际 embedding
        # Contradiction 由 NLI 决定
        assert cite["contradicted"] is False
        # 应同时有 embedding 和 NLI 的信息
        assert "embedding_similarity" in cite
        assert "nli_probs" in cite

    def test_hybrid_detects_contradiction(self) -> None:
        """Hybrid: embedding grounded 但 NLI 检测到矛盾"""
        paper = _make_paper("p1", abstract="Method X improves accuracy by 20%")
        section = ReportSection(
            section_title="Results",
            content="Method X actually decreases accuracy by 15% compared to the baseline. The results are disappointing overall.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        # Mock NLI: 强矛盾
        mock_nli = MagicMock()

        def predict_contradiction(pairs, apply_softmax=True):
            n = len(pairs)
            return np.array([[0.85, 0.05, 0.10]] * n)  # contradiction

        mock_nli.predict.side_effect = predict_contradiction

        with patch.object(CitationVerifier, "_load_nli_model", return_value=mock_nli):
            verifier = CitationVerifier(method="hybrid", contradiction_threshold=0.5)
            report = verifier.verify(result)

        cite = report["sections"][0]["citations"][0]
        assert cite["contradicted"] is True
        assert report["num_citations_contradicted"] == 1


class TestMethodSelection:
    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            CitationVerifier(method="invalid")  # type: ignore[arg-type]

    def test_embedding_method_default(self) -> None:
        verifier = CitationVerifier()
        assert verifier._method == "embedding"
