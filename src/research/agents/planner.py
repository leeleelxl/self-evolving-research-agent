"""
Planner Agent — 研究规划与策略生成

职责:
1. 首轮: 将研究问题分解为可检索的子问题 + 生成 SearchStrategy
2. 迭代轮: 分析 Critic 反馈，针对性调整检索策略

自进化的起点: Planner 根据 Critic 反馈"学习"如何更好地检索。
这不是简单重试——它会根据反馈增加新 query、调整 focus、排除已覆盖内容。
"""

from __future__ import annotations

from research.core.agent import BaseAgent
from research.core.config import LLMConfig
from research.core.models import CriticFeedback, ResearchPlan

# ── System Prompt ──
# 角色定义要具体，不要泛泛地说"你是一个助手"
SYSTEM_PROMPT = """\
You are the Planner Agent in an academic research system.

Your job is to decompose a research question into sub-questions and create a search strategy \
for retrieving relevant academic papers from Semantic Scholar and arXiv.

Guidelines:
- Break the question into 3-5 specific, searchable sub-questions
- For each sub-question, generate 1-2 search queries (concise keyword phrases, not full sentences)
- Assign each query to the most appropriate source: "semantic_scholar" for published papers, "arxiv" for recent preprints
- Focus areas should capture the key themes to guide paper selection
- Search queries should be in English (academic databases work best with English queries)
"""

# ── 首轮 Prompt ──
INITIAL_PROMPT_TEMPLATE = """\
Research question: {question}

Decompose this research question into sub-questions and create a search strategy.
Think about:
1. What are the key concepts and their relationships?
2. What sub-topics need to be covered for a comprehensive survey?
3. What search terms would find the most relevant papers?

Generate a ResearchPlan with sub_questions, search queries, and focus areas.
Set iteration to 0.
"""

# ── 迭代轮 Prompt（带 Critic 反馈） ──
REFINE_PROMPT_TEMPLATE = """\
Research question: {question}

This is iteration {iteration}. The Critic Agent evaluated the previous survey and found issues.

## Critic Feedback
- Coverage score: {coverage}/10
- Depth score: {depth}/10
- Coherence score: {coherence}/10
- Accuracy score: {accuracy}/10

### Missing aspects:
{missing_aspects}

### Improvement suggestions:
{improvement_suggestions}

### Suggested new queries:
{new_queries}

## Your Task
Based on the feedback above, create an IMPROVED search strategy:
1. Add new sub-questions to address missing aspects
2. Generate new search queries targeting the weak areas
3. Update focus areas to emphasize what's lacking
4. Add previously covered topics to exclude_terms to avoid redundant retrieval

Keep the original sub-questions that are still relevant, and ADD new ones.
Set iteration to {iteration}.
"""


class PlannerAgent(BaseAgent):
    """研究规划 Agent

    Usage:
        planner = PlannerAgent()

        # 首轮
        plan = await planner.run("What are recent advances in LLM agents?")

        # 迭代轮（带 Critic 反馈）
        plan = await planner.run(question, feedback=critic_feedback, iteration=1)
    """

    name = "Planner"
    role = SYSTEM_PROMPT

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        super().__init__(llm_config)

    async def run(
        self,
        question: str,
        feedback: CriticFeedback | None = None,
        iteration: int = 0,
    ) -> ResearchPlan:
        """生成或更新研究计划

        Args:
            question: 研究问题
            feedback: 上一轮 Critic 的反馈（首轮为 None）
            iteration: 当前迭代轮次
        """
        if feedback is None or iteration == 0:
            prompt = self._build_initial_prompt(question)
        else:
            prompt = self._build_refine_prompt(question, feedback, iteration)

        self.logger.info("planning", iteration=iteration, has_feedback=feedback is not None)

        plan = await self.generate_structured(prompt, ResearchPlan)

        # 确保 iteration 字段正确（LLM 可能搞错）
        plan.iteration = iteration
        plan.original_question = question

        self.logger.info(
            "plan_generated",
            sub_questions=len(plan.sub_questions),
            queries=len(plan.search_strategy.queries),
        )
        return plan

    def _build_initial_prompt(self, question: str) -> str:
        return INITIAL_PROMPT_TEMPLATE.format(question=question)

    def _build_refine_prompt(
        self,
        question: str,
        feedback: CriticFeedback,
        iteration: int,
    ) -> str:
        return REFINE_PROMPT_TEMPLATE.format(
            question=question,
            iteration=iteration,
            coverage=feedback.scores.coverage,
            depth=feedback.scores.depth,
            coherence=feedback.scores.coherence,
            accuracy=feedback.scores.accuracy,
            missing_aspects="\n".join(f"- {a}" for a in feedback.missing_aspects),
            improvement_suggestions="\n".join(f"- {s}" for s in feedback.improvement_suggestions),
            new_queries="\n".join(f"- {q}" for q in feedback.new_queries) if feedback.new_queries else "- (none suggested)",
        )
