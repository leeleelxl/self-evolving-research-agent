"""Pipeline 集成 CitationVerifier 测试

验证 P2 的 wiring 逻辑（不真调 LLM）:
- verify_citations=False (默认): PipelineResult.citation_verification is None
- verify_citations=True: Pipeline 末尾调 CitationVerifier.verify() 并填充字段
- Verifier 失败: citation_verification 保持 None，不影响主结果
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.core.config import (
    KnowledgeBaseConfig,
    LLMConfig,
    PipelineConfig,
)
from research.core.models import (
    CriticFeedback,
    CriticScores,
    Paper,
    PaperNote,
    ResearchPlan,
    ResearchReport,
    SearchQuery,
    SearchStrategy,
)
from research.pipeline.research import ResearchPipeline


def _build_mocked_pipeline(verify_citations: bool = False) -> ResearchPipeline:
    config = PipelineConfig(
        max_iterations=1,
        llm=LLMConfig(),
        knowledge_base=KnowledgeBaseConfig(enabled=False),
        verify_citations=verify_citations,
    )
    pipeline = ResearchPipeline(config)

    # Mock Agent 返回固定值
    pipeline._planner = MagicMock()
    pipeline._planner.run = AsyncMock(return_value=ResearchPlan(
        original_question="test",
        sub_questions=["sub1"],
        search_strategy=SearchStrategy(
            queries=[SearchQuery(query="q1")],
            focus_areas=[],
        ),
        iteration=0,
    ))
    pipeline._retriever = MagicMock()
    pipeline._retriever.run = AsyncMock(return_value=[
        Paper(paper_id="p1", title="Paper 1", abstract="abs", authors=[], year=2024, url="", source="arxiv"),
    ])
    pipeline._reader = MagicMock()
    pipeline._reader.run = AsyncMock(return_value=[
        PaperNote(paper_id="p1", title="Paper 1", core_contribution="c", methodology="m",
                  key_findings=["f"], relevance_score=0.9, relevance_reason="r"),
    ])
    pipeline._writer = MagicMock()
    pipeline._writer.run = AsyncMock(return_value=ResearchReport(
        title="Report", abstract="abs", sections=[], references=["p1"],
    ))
    pipeline._critic = MagicMock()
    pipeline._critic.run = AsyncMock(return_value=CriticFeedback(
        scores=CriticScores(coverage=8, depth=8, coherence=8, accuracy=8),
        missing_aspects=[], improvement_suggestions=[], is_satisfactory=True,
    ))

    return pipeline


class TestCitationVerificationWiring:
    @pytest.mark.asyncio
    async def test_default_disabled_field_is_none(self) -> None:
        """默认 verify_citations=False → citation_verification 是 None"""
        pipeline = _build_mocked_pipeline(verify_citations=False)
        result = await pipeline.run("test")
        assert result.citation_verification is None

    @pytest.mark.asyncio
    async def test_enabled_populates_field(self) -> None:
        """verify_citations=True → Pipeline 调 CitationVerifier 填充字段"""
        pipeline = _build_mocked_pipeline(verify_citations=True)

        fake_result = {
            "method": "hybrid",
            "sections": [],
            "overall_grounding_rate": 1.0,
            "overall_avg_score": 0.85,
            "overall_mismatch_rate": 0.1,
            "num_citations_checked": 10,
            "num_citations_grounded": 10,
            "num_citations_ungrounded": 0,
            "num_citations_mismatched": 1,
            "num_citations_missing": 0,
            "threshold": 0.3,
        }

        mock_verifier_cls = MagicMock()
        mock_verifier_instance = MagicMock()
        # Pipeline 现在用 await verify_async()，mock async 版本
        mock_verifier_instance.verify_async = AsyncMock(return_value=fake_result)
        mock_verifier_cls.return_value = mock_verifier_instance

        with patch(
            "research.evaluation.citation_verifier.CitationVerifier",
            mock_verifier_cls,
        ):
            result = await pipeline.run("test")

        assert result.citation_verification is not None
        assert result.citation_verification["method"] == "hybrid"
        assert result.citation_verification["overall_grounding_rate"] == 1.0
        assert result.citation_verification["num_citations_mismatched"] == 1

    @pytest.mark.asyncio
    async def test_verifier_failure_does_not_crash(self) -> None:
        """Verifier 抛异常时，citation_verification 保持 None，主结果仍返回"""
        pipeline = _build_mocked_pipeline(verify_citations=True)

        mock_verifier_cls = MagicMock()
        mock_verifier_cls.side_effect = RuntimeError("embedding model failed to load")

        with patch(
            "research.evaluation.citation_verifier.CitationVerifier",
            mock_verifier_cls,
        ):
            result = await pipeline.run("test")

        # Pipeline 返回了，不 crash
        assert result is not None
        assert result.report is not None
        # citation_verification 保持 None
        assert result.citation_verification is None
