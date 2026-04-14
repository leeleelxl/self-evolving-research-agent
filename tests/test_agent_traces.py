"""Agent IO 追踪功能测试 — 验证 Agent 实际 IO 能被保存

做 Agent 项目的核心纪律: 评估必须以 Agent IO 为证据，不能只看聚合数字。
本测试验证 trace 记录功能在 minimal/standard/full 三种级别下的行为。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.core.config import (
    KnowledgeBaseConfig,
    LLMConfig,
    PipelineConfig,
)
from research.core.models import (
    AgentTrace,
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


def _make_plan(iteration: int = 0) -> ResearchPlan:
    return ResearchPlan(
        original_question="test",
        sub_questions=["sub1", "sub2"],
        search_strategy=SearchStrategy(
            queries=[SearchQuery(query="q1"), SearchQuery(query="q2")],
            focus_areas=["area1"],
            exclude_terms=[],
        ),
        iteration=iteration,
    )


def _make_paper(pid: str = "p1") -> Paper:
    return Paper(
        paper_id=pid,
        title=f"Paper {pid}",
        abstract="test abstract",
        authors=["A"],
        year=2024,
        url="https://x.com",
        source="arxiv",
    )


def _make_note(pid: str = "p1") -> PaperNote:
    return PaperNote(
        paper_id=pid,
        title=f"Paper {pid}",
        core_contribution="This paper contributes X",
        methodology="They use Y",
        key_findings=["finding 1", "finding 2"],
        limitations=[],
        relevance_score=0.8,
        relevance_reason="Directly addresses the question",
    )


def _make_report() -> ResearchReport:
    return ResearchReport(
        title="Test Report",
        abstract="Abstract",
        sections=[],
        references=["p1"],
    )


def _make_feedback(satisfactory: bool = True) -> CriticFeedback:
    return CriticFeedback(
        scores=CriticScores(coverage=8, depth=7, coherence=8, accuracy=7),
        missing_aspects=["aspect1"],
        improvement_suggestions=["specific improvement 1"],
        new_queries=["new query 1"],
        is_satisfactory=satisfactory,
    )


def _build_mocked_pipeline(trace_level: str = "full") -> ResearchPipeline:
    """构造 mock Pipeline: 所有 Agent 都返回固定值，只测 trace 记录逻辑"""
    config = PipelineConfig(
        max_iterations=1,
        llm=LLMConfig(),
        knowledge_base=KnowledgeBaseConfig(enabled=False),
        trace_level=trace_level,  # type: ignore[arg-type]
    )
    pipeline = ResearchPipeline(config)

    pipeline._planner = MagicMock()
    pipeline._planner.run = AsyncMock(return_value=_make_plan(0))

    pipeline._retriever = MagicMock()
    pipeline._retriever.run = AsyncMock(return_value=[_make_paper("p1"), _make_paper("p2")])

    pipeline._reader = MagicMock()
    pipeline._reader.run = AsyncMock(return_value=[_make_note("p1")])

    pipeline._writer = MagicMock()
    pipeline._writer.run = AsyncMock(return_value=_make_report())

    pipeline._critic = MagicMock()
    pipeline._critic.run = AsyncMock(return_value=_make_feedback(satisfactory=True))

    return pipeline


class TestTraceRecording:
    @pytest.mark.asyncio
    async def test_full_level_records_all_agents(self) -> None:
        pipeline = _build_mocked_pipeline(trace_level="full")
        result = await pipeline.run("test question")

        agent_names = [t.agent_name for t in result.agent_traces]
        assert "Planner" in agent_names
        assert "Retriever" in agent_names
        assert "Reader" in agent_names
        assert "Writer" in agent_names
        assert "Critic" in agent_names
        assert len(result.agent_traces) == 5

    @pytest.mark.asyncio
    async def test_minimal_level_only_decision_agents(self) -> None:
        """minimal 模式只记录 Planner 和 Critic（决策证据）"""
        pipeline = _build_mocked_pipeline(trace_level="minimal")
        result = await pipeline.run("test question")

        agent_names = [t.agent_name for t in result.agent_traces]
        assert agent_names == ["Planner", "Critic"]

    @pytest.mark.asyncio
    async def test_standard_level_reader_is_slimmed(self) -> None:
        """standard 模式下 Reader 的 notes 只保留关键字段"""
        pipeline = _build_mocked_pipeline(trace_level="standard")
        result = await pipeline.run("test question")

        reader_traces = [t for t in result.agent_traces if t.agent_name == "Reader"]
        assert len(reader_traces) == 1
        note = reader_traces[0].output["notes"][0]
        assert "core_contribution" in note  # 保留
        assert "methodology" not in note  # 精简掉
        assert "key_findings" not in note  # 精简掉

    @pytest.mark.asyncio
    async def test_full_level_reader_keeps_all_fields(self) -> None:
        """full 模式下 Reader 的 notes 保留所有字段"""
        pipeline = _build_mocked_pipeline(trace_level="full")
        result = await pipeline.run("test question")

        reader_traces = [t for t in result.agent_traces if t.agent_name == "Reader"]
        note = reader_traces[0].output["notes"][0]
        assert "core_contribution" in note
        assert "methodology" in note
        assert "key_findings" in note
        assert "limitations" in note
        assert "relevance_reason" in note


class TestTraceContent:
    @pytest.mark.asyncio
    async def test_planner_trace_preserves_queries_text(self) -> None:
        """Planner trace 必须保留完整 queries 文本（自进化验证的关键证据）"""
        pipeline = _build_mocked_pipeline()
        result = await pipeline.run("test question")

        planner_trace = next(t for t in result.agent_traces if t.agent_name == "Planner")
        queries = planner_trace.output["search_strategy"]["queries"]
        assert len(queries) == 2
        assert queries[0]["query"] == "q1"
        assert queries[1]["query"] == "q2"

    @pytest.mark.asyncio
    async def test_critic_trace_preserves_suggestions_text(self) -> None:
        """Critic trace 必须保留完整 improvement_suggestions（套话 vs 具体的证据）"""
        pipeline = _build_mocked_pipeline()
        result = await pipeline.run("test question")

        critic_trace = next(t for t in result.agent_traces if t.agent_name == "Critic")
        assert critic_trace.output["improvement_suggestions"] == ["specific improvement 1"]
        assert critic_trace.output["new_queries"] == ["new query 1"]

    @pytest.mark.asyncio
    async def test_trace_has_iteration_and_timestamp(self) -> None:
        pipeline = _build_mocked_pipeline()
        result = await pipeline.run("test question")

        for trace in result.agent_traces:
            assert trace.iteration == 0
            assert trace.timestamp_ms > 0
            assert trace.input_summary  # 非空

    @pytest.mark.asyncio
    async def test_retriever_trace_has_paper_summaries(self) -> None:
        pipeline = _build_mocked_pipeline()
        result = await pipeline.run("test question")

        retriever_trace = next(t for t in result.agent_traces if t.agent_name == "Retriever")
        papers = retriever_trace.output["papers"]
        assert len(papers) == 2
        assert papers[0]["paper_id"] == "p1"
        assert papers[0]["title"] == "Paper p1"


class TestAgentTraceModel:
    def test_trace_model_validation(self) -> None:
        trace = AgentTrace(
            agent_name="Planner",
            iteration=0,
            input_summary="test",
            output={"key": "value"},
            timestamp_ms=1234567890,
        )
        assert trace.agent_name == "Planner"

    def test_trace_rejects_invalid_agent_name(self) -> None:
        with pytest.raises(Exception):  # Pydantic ValidationError
            AgentTrace(
                agent_name="InvalidAgent",  # type: ignore[arg-type]
                iteration=0,
                input_summary="test",
                output={},
                timestamp_ms=0,
            )
