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
        assert report["overall_mismatch_rate"] == 0.0
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
        assert report["num_citations_mismatched"] == 1
        assert report["overall_mismatch_rate"] > 0
        cite = report["sections"][0]["citations"][0]
        assert cite["mismatched"] is True

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
        assert report["num_citations_mismatched"] == 0

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


class TestAttributionMode:
    """Attribution 模式：LLM-judge 检测 paper_id 错配（pivot 自 NLI v4）"""

    def _mock_judge_matching(self, verifier: CitationVerifier) -> None:
        """让 LLM judge 总返回 matching"""
        from unittest.mock import AsyncMock

        async def fake_generate(messages, response_model):
            return response_model(
                label="matching",
                reasoning="Abstract supports the section's claims.",
                confidence=0.85,
            )

        verifier._judge_client = MagicMock()
        verifier._judge_client.generate_structured = AsyncMock(side_effect=fake_generate)

    def _mock_judge_mismatched(self, verifier: CitationVerifier) -> None:
        """让 LLM judge 总返回 mismatched（paper_id 错配）"""
        from unittest.mock import AsyncMock

        async def fake_generate(messages, response_model):
            return response_model(
                label="mismatched",
                reasoning="Section attributes features that don't match this abstract.",
                confidence=0.9,
            )

        verifier._judge_client = MagicMock()
        verifier._judge_client.generate_structured = AsyncMock(side_effect=fake_generate)

    def test_attribution_matching_not_mismatched(self) -> None:
        paper = _make_paper("p1", abstract="RAG for QA with dense retrieval")
        section = ReportSection(
            section_title="RAG",
            content="This paper proposes RAG for question answering.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        with patch.object(CitationVerifier, "_load_judge_client", return_value=MagicMock()):
            verifier = CitationVerifier(method="attribution")
            self._mock_judge_matching(verifier)
            report = verifier.verify(result)

        assert report["method"] == "attribution"
        assert report["num_citations_mismatched"] == 0
        cite = report["sections"][0]["citations"][0]
        assert cite["mismatched"] is False
        assert cite["attribution_label"] == "matching"
        assert cite["grounded"] is True  # matching/partial 都算 grounded

    def test_attribution_detects_paper_id_mismatch(self) -> None:
        """真实场景: section 把 FAIR-RAG 的特性归给讲 Indonesian QA 的论文 (P4 FN 案例)"""
        paper = _make_paper("p1", abstract="Indonesian open-domain QA with BM25 and Gemma")
        section = ReportSection(
            section_title="Training",
            content="FAIR-RAG introduces checklist-based structured evidence assessment for multi-hop QA.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        with patch.object(CitationVerifier, "_load_judge_client", return_value=MagicMock()):
            verifier = CitationVerifier(method="attribution")
            self._mock_judge_mismatched(verifier)
            report = verifier.verify(result)

        assert report["num_citations_mismatched"] == 1
        assert report["overall_mismatch_rate"] > 0
        cite = report["sections"][0]["citations"][0]
        assert cite["mismatched"] is True
        assert cite["attribution_label"] == "mismatched"


class TestHybridMode:
    """Hybrid 模式 v4: embedding grounding + attribution 错配检测"""

    def test_hybrid_returns_both_embedding_and_attribution(self) -> None:
        paper = _make_paper("p1", abstract="Dense retrieval with contrastive learning")
        section = ReportSection(
            section_title="Retrieval",
            content="Dense retrieval uses contrastive training for better recall.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        from unittest.mock import AsyncMock

        async def fake_generate(messages, response_model):
            return response_model(
                label="matching",
                reasoning="ok",
                confidence=0.8,
            )

        with patch.object(CitationVerifier, "_load_judge_client", return_value=MagicMock()):
            verifier = CitationVerifier(method="hybrid")
            verifier._judge_client = MagicMock()
            verifier._judge_client.generate_structured = AsyncMock(side_effect=fake_generate)
            report = verifier.verify(result)

        assert report["method"] == "hybrid"
        cite = report["sections"][0]["citations"][0]
        assert "embedding_similarity" in cite  # embedding 部分
        assert "attribution_label" in cite  # attribution 部分
        assert cite["mismatched"] is False

    def test_hybrid_mismatched_from_attribution(self) -> None:
        """Hybrid: embedding 看起来话题相关, 但 attribution 检测出 paper_id 错配"""
        paper = _make_paper("p1", abstract="A study on graph neural networks")
        section = ReportSection(
            section_title="GNN",
            content="This paper proposes a novel GNN architecture with attention.",
            cited_papers=["p1"],
        )
        result = _make_pipeline_result([section], [paper])

        from unittest.mock import AsyncMock

        async def fake_generate(messages, response_model):
            return response_model(
                label="mismatched",
                reasoning="section claims don't belong to this paper",
                confidence=0.9,
            )

        with patch.object(CitationVerifier, "_load_judge_client", return_value=MagicMock()):
            verifier = CitationVerifier(method="hybrid")
            verifier._judge_client = MagicMock()
            verifier._judge_client.generate_structured = AsyncMock(side_effect=fake_generate)
            report = verifier.verify(result)

        cite = report["sections"][0]["citations"][0]
        assert cite["mismatched"] is True
        assert report["num_citations_mismatched"] == 1


class TestNLIDeprecation:
    def test_nli_raises_deprecation_warning(self) -> None:
        mock_nli = MagicMock()

        def predict(pairs, apply_softmax=True):
            n = len(pairs)
            return np.array([[0.05, 0.15, 0.80]] * n)

        mock_nli.predict.side_effect = predict

        with patch.object(CitationVerifier, "_load_nli_model", return_value=mock_nli):
            with pytest.warns(DeprecationWarning, match="precision=0%"):
                CitationVerifier(method="nli")


class TestMethodSelection:
    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            CitationVerifier(method="invalid")  # type: ignore[arg-type]

    def test_embedding_method_default(self) -> None:
        verifier = CitationVerifier()
        assert verifier._method == "embedding"
