"""Pipeline 编排测试"""

import pytest
from dotenv import load_dotenv
from unittest.mock import AsyncMock, patch

load_dotenv()

from research.core.config import PipelineConfig
from research.core.models import (
    CriticFeedback,
    CriticScores,
    PaperNote,
    ResearchPlan,
    ResearchReport,
    ReportSection,
    Paper,
    SearchQuery,
    SearchStrategy,
)
from research.pipeline.research import ResearchPipeline


class TestPipelineMocked:
    """Pipeline 单元测试 — mock 所有 Agent，验证编排逻辑"""

    @pytest.mark.asyncio
    async def test_pipeline_single_iteration(self) -> None:
        """单轮迭代：所有 Agent 被正确调用"""
        config = PipelineConfig(max_iterations=1)
        pipeline = ResearchPipeline(config)

        # Mock 所有 Agent
        pipeline._planner.run = AsyncMock(return_value=ResearchPlan(
            original_question="test",
            sub_questions=["q1"],
            search_strategy=SearchStrategy(queries=[SearchQuery(query="test")], focus_areas=[]),
        ))
        pipeline._retriever.run = AsyncMock(return_value=[
            Paper(paper_id="p1", title="Paper 1", abstract="abs", authors=[], year=2024, url="", source="arxiv"),
        ])
        pipeline._reader.run = AsyncMock(return_value=[
            PaperNote(paper_id="p1", title="Paper 1", core_contribution="c", methodology="m",
                      key_findings=["f1"], relevance_score=0.9, relevance_reason="relevant"),
        ])
        pipeline._writer.run = AsyncMock(return_value=ResearchReport(
            title="Survey", abstract="abs", sections=[], references=["p1"],
        ))
        pipeline._critic.run = AsyncMock(return_value=CriticFeedback(
            scores=CriticScores(coverage=8, depth=8, coherence=8, accuracy=8),
            missing_aspects=[], improvement_suggestions=[], is_satisfactory=True,
        ))

        result = await pipeline.run("test question")

        assert result.total_iterations == 1
        assert result.report.title == "Survey"
        assert len(result.evolution_log) == 1
        assert result.evolution_log[0].scores.overall == 8.0

    @pytest.mark.asyncio
    async def test_pipeline_self_evolution_loop(self) -> None:
        """自进化：Critic 不满意 → 第二轮"""
        config = PipelineConfig(max_iterations=3)
        pipeline = ResearchPipeline(config)

        plan = ResearchPlan(
            original_question="test", sub_questions=["q1"],
            search_strategy=SearchStrategy(queries=[SearchQuery(query="test")], focus_areas=[]),
        )
        paper = Paper(paper_id="p1", title="P1", abstract="abs", authors=[], year=2024, url="", source="arxiv")
        note = PaperNote(paper_id="p1", title="P1", core_contribution="c", methodology="m",
                         key_findings=["f1"], relevance_score=0.9, relevance_reason="r")
        report = ResearchReport(title="S", abstract="a", sections=[], references=[])

        # 第一轮不满意，第二轮满意
        unsatisfied = CriticFeedback(
            scores=CriticScores(coverage=4, depth=4, coherence=4, accuracy=4),
            missing_aspects=["X"], improvement_suggestions=["add X"], is_satisfactory=False,
        )
        satisfied = CriticFeedback(
            scores=CriticScores(coverage=8, depth=8, coherence=8, accuracy=8),
            missing_aspects=[], improvement_suggestions=[], is_satisfactory=True,
        )

        pipeline._planner.run = AsyncMock(return_value=plan)
        pipeline._retriever.run = AsyncMock(return_value=[paper])
        pipeline._reader.run = AsyncMock(return_value=[note])
        pipeline._writer.run = AsyncMock(return_value=report)
        pipeline._critic.run = AsyncMock(side_effect=[unsatisfied, satisfied])

        result = await pipeline.run("test")

        assert result.total_iterations == 2
        # 验证进化分数递增
        assert result.evolution_log[0].scores.overall == 4.0
        assert result.evolution_log[1].scores.overall == 8.0

    @pytest.mark.asyncio
    async def test_pipeline_deduplicates_notes(self) -> None:
        """跨轮去重：同一篇论文不会重复出现在笔记中"""
        config = PipelineConfig(max_iterations=2)
        pipeline = ResearchPipeline(config)

        plan = ResearchPlan(
            original_question="test", sub_questions=["q1"],
            search_strategy=SearchStrategy(queries=[SearchQuery(query="test")], focus_areas=[]),
        )
        paper = Paper(paper_id="same_paper", title="P", abstract="a", authors=[], year=2024, url="", source="arxiv")
        note = PaperNote(paper_id="same_paper", title="P", core_contribution="c", methodology="m",
                         key_findings=["f"], relevance_score=0.8, relevance_reason="r")
        report = ResearchReport(title="S", abstract="a", sections=[], references=[])

        pipeline._planner.run = AsyncMock(return_value=plan)
        pipeline._retriever.run = AsyncMock(return_value=[paper])
        pipeline._reader.run = AsyncMock(return_value=[note])
        pipeline._writer.run = AsyncMock(return_value=report)
        pipeline._critic.run = AsyncMock(side_effect=[
            CriticFeedback(scores=CriticScores(coverage=4, depth=4, coherence=4, accuracy=4),
                           missing_aspects=["X"], improvement_suggestions=["Y"], is_satisfactory=False),
            CriticFeedback(scores=CriticScores(coverage=8, depth=8, coherence=8, accuracy=8),
                           missing_aspects=[], improvement_suggestions=[], is_satisfactory=True),
        ])

        result = await pipeline.run("test")

        # Writer 应该收到的 notes 列表中 same_paper 只出现一次
        # 第二轮 writer.run 调用时，all_notes 只有 1 条（去重了）
        writer_calls = pipeline._writer.run.call_args_list
        last_call_notes = writer_calls[-1][0][0]  # 第一个位置参数
        assert len(last_call_notes) == 1
