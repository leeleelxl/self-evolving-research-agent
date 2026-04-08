"""PlannerAgent 测试"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.agents.planner import PlannerAgent
from research.core.models import CriticFeedback, CriticScores, ResearchPlan


@pytest.mark.integration
class TestPlannerIntegration:
    """PlannerAgent 集成测试 — 真实调用 LLM"""

    @pytest.mark.asyncio
    async def test_initial_plan(self) -> None:
        """首轮：问题分解 + 策略生成"""
        planner = PlannerAgent()
        plan = await planner.run("What are recent advances in LLM agents?")

        assert isinstance(plan, ResearchPlan)
        assert plan.iteration == 0
        assert len(plan.sub_questions) >= 2
        assert len(plan.search_strategy.queries) >= 2
        assert plan.original_question == "What are recent advances in LLM agents?"

    @pytest.mark.asyncio
    async def test_refine_plan_with_feedback(self) -> None:
        """迭代轮：根据 Critic 反馈调整策略"""
        planner = PlannerAgent()

        feedback = CriticFeedback(
            scores=CriticScores(coverage=4.0, depth=5.0, coherence=7.0, accuracy=6.0),
            missing_aspects=["multi-agent collaboration patterns", "agent memory mechanisms"],
            improvement_suggestions=["Add papers about agent memory and state management"],
            new_queries=["agent memory architecture 2024", "multi-agent collaboration survey"],
            is_satisfactory=False,
        )

        plan = await planner.run(
            "What are recent advances in LLM agents?",
            feedback=feedback,
            iteration=1,
        )

        assert isinstance(plan, ResearchPlan)
        assert plan.iteration == 1
        # 迭代轮应该比首轮有更多/不同的 query
        assert len(plan.search_strategy.queries) >= 2
