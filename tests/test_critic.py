"""CriticAgent 测试"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.agents.critic import CriticAgent
from research.core.config import PipelineConfig
from research.core.models import CriticFeedback, ReportSection, ResearchReport


@pytest.mark.integration
class TestCriticIntegration:
    """CriticAgent 集成测试 — 真实 LLM 调用"""

    @pytest.mark.asyncio
    async def test_evaluate_empty_report(self) -> None:
        """评估一份空报告 — 应该给低分"""
        config = PipelineConfig(satisfactory_threshold=7.0)
        critic = CriticAgent(pipeline_config=config)

        report = ResearchReport(
            title="Survey: LLM Agents",
            abstract="This is a placeholder abstract.",
            sections=[],
            references=[],
        )

        feedback = await critic.run(report, "What are recent advances in LLM agents?")

        assert isinstance(feedback, CriticFeedback)
        # 空报告应该拿低分
        assert feedback.scores.overall < 7.0
        assert not feedback.is_satisfactory
        # 应该给出改进建议
        assert len(feedback.missing_aspects) > 0
        assert len(feedback.improvement_suggestions) > 0

    @pytest.mark.asyncio
    async def test_evaluate_decent_report(self) -> None:
        """评估一份有内容的报告"""
        config = PipelineConfig(satisfactory_threshold=5.0)  # 降低阈值
        critic = CriticAgent(pipeline_config=config)

        report = ResearchReport(
            title="Survey: Recent Advances in LLM Agents",
            abstract="This survey covers recent advances in LLM-based agents, including tool use, planning, and multi-agent systems.",
            sections=[
                ReportSection(
                    section_title="Tool Use in LLM Agents",
                    content="Recent work has shown that LLMs can effectively use external tools through function calling mechanisms. Toolformer demonstrated that language models can learn to use APIs. This has been extended in systems like GPT-4 and Claude.",
                    cited_papers=["paper1", "paper2"],
                ),
                ReportSection(
                    section_title="Multi-Agent Collaboration",
                    content="Multi-agent systems like AutoGen and CrewAI enable multiple LLM agents to collaborate on complex tasks. These systems use orchestration patterns including sequential chains, DAGs, and coordinator-based approaches.",
                    cited_papers=["paper3"],
                ),
            ],
            references=["paper1", "paper2", "paper3"],
        )

        feedback = await critic.run(report, "What are recent advances in LLM agents?")

        assert isinstance(feedback, CriticFeedback)
        # 有内容的报告，各维度应在合理范围 (0-10)
        assert 0 <= feedback.scores.coverage <= 10
        assert 0 <= feedback.scores.depth <= 10
        # 应该给出具体的改进建议
        assert len(feedback.improvement_suggestions) > 0

    @pytest.mark.asyncio
    async def test_threshold_override(self) -> None:
        """代码层面重算 is_satisfactory，不依赖 LLM"""
        config = PipelineConfig(satisfactory_threshold=10.0)  # 不可能达到
        critic = CriticAgent(pipeline_config=config)

        report = ResearchReport(title="Test", abstract="Test", sections=[], references=[])
        feedback = await critic.run(report, "test")

        # 阈值 10.0，任何报告都不应该满意
        assert not feedback.is_satisfactory
