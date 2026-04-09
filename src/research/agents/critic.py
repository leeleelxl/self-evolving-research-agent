"""
Critic Agent — 质量评估与自进化驱动

职责:
1. 评估 ResearchReport 的质量（4 个维度）
2. 生成结构化反馈（缺什么、怎么改、该搜什么）
3. 判断是否达标 → 驱动自进化循环

这是整个系统的"裁判"——Agent-as-Judge 模式。
Critic 的评估质量直接决定自进化是否有效。

设计选择:
- 可以用比其他 Agent 更强的模型（config.critic_model）
- is_satisfactory 在代码层面基于 threshold 重算，不完全信任 LLM 判断
- 评分 prompt 要求给出具体证据，而不只是数字
"""

from __future__ import annotations

from research.core.agent import BaseAgent
from research.core.config import LLMConfig, PipelineConfig
from research.core.llm import create_llm_client
from research.core.models import CriticFeedback, ResearchReport

SYSTEM_PROMPT = """\
You are the Critic Agent in an academic research system.

Your job is to evaluate the quality of a research survey and provide structured feedback \
for improvement. You must be rigorous and specific — vague feedback like "needs improvement" \
is useless. Point out exactly what's missing and suggest concrete search queries to fix it.

Scoring guidelines (0-10 scale):
- Coverage (覆盖度): Are all major sub-topics covered? Are there obvious gaps?
  - 8-10: Comprehensive, covers all key aspects
  - 5-7: Covers most topics but missing some important areas
  - 0-4: Major gaps, key topics entirely missing

- Depth (深度): Is the analysis superficial or does it go into technical details?
  - 8-10: Deep technical analysis with methodology details
  - 5-7: Reasonable depth but some topics only surface-level
  - 0-4: Mostly superficial, no technical details

- Coherence (连贯性): Does the survey flow logically? Are sections well-connected?
  - 8-10: Clear logical flow, smooth transitions
  - 5-7: Generally coherent but some disconnected parts
  - 0-4: Disjointed, no clear narrative

- Accuracy (准确性): Are claims consistent with what the cited papers state?
  Note: This survey is based on paper abstracts, not full text. \
  Score accuracy based on whether claims are consistent with abstract-level information. \
  Do NOT penalize for lacking details that would only be in the full paper.
  - 8-10: Claims accurately reflect the cited papers' abstracts
  - 5-7: Mostly accurate, some claims slightly overstate abstract content
  - 0-4: Clear contradictions with cited papers or fabricated claims

Be honest and critical. A score of 5 is average, not bad. Reserve 8+ for genuinely good work.
"""

EVALUATE_PROMPT_TEMPLATE = """\
## Original Research Question
{question}

## Survey to Evaluate

### Title
{title}

### Abstract
{abstract}

### Sections
{sections}

### References
{references}

## Your Task
1. Score each dimension (coverage, depth, coherence, accuracy) from 0-10
2. List specific aspects that are MISSING from the survey (be concrete)
3. Suggest specific improvements (actionable, not vague)
4. Suggest new search queries that would help fill the gaps (ready-to-execute keyword phrases)
5. Set is_satisfactory based on whether the overall quality is acceptable

Remember: be specific and honest. "Add more papers" is bad feedback. \
"Missing coverage of retrieval-augmented generation techniques, \
especially dense passage retrieval and hybrid search methods" is good feedback.
"""


class CriticAgent(BaseAgent):
    """质量评估 Agent — 自进化机制的驱动力

    Usage:
        critic = CriticAgent(pipeline_config)
        feedback = await critic.run(report, question)

        if not feedback.is_satisfactory:
            # 反馈回 Planner → 自进化
            new_plan = await planner.run(question, feedback, iteration=1)
    """

    name = "Critic"
    role = SYSTEM_PROMPT

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        # Critic 可以用更强的模型
        config = pipeline_config or PipelineConfig()
        critic_llm_config = llm_config or LLMConfig(model=config.critic_model)
        super().__init__(critic_llm_config)

        self._threshold = config.satisfactory_threshold

    async def run(self, report: ResearchReport, question: str) -> CriticFeedback:
        """评估综述质量，返回结构化反馈"""
        prompt = self._build_prompt(report, question)

        self.logger.info("evaluating", title=report.title)

        feedback = await self.generate_structured(prompt, CriticFeedback)

        # 关键：在代码层面重算 is_satisfactory
        # 不完全依赖 LLM 的判断，确保与 config 阈值一致
        feedback.is_satisfactory = feedback.scores.overall >= self._threshold

        self.logger.info(
            "evaluation_done",
            coverage=feedback.scores.coverage,
            depth=feedback.scores.depth,
            coherence=feedback.scores.coherence,
            accuracy=feedback.scores.accuracy,
            overall=feedback.scores.overall,
            is_satisfactory=feedback.is_satisfactory,
        )
        return feedback

    def _build_prompt(self, report: ResearchReport, question: str) -> str:
        # 拼接 sections 内容
        sections_text = ""
        if report.sections:
            for s in report.sections:
                sections_text += f"\n#### {s.section_title}\n{s.content}\n"
        else:
            sections_text = "(No sections yet)"

        # 拼接引用
        refs_text = ", ".join(report.references) if report.references else "(No references)"

        return EVALUATE_PROMPT_TEMPLATE.format(
            question=question,
            title=report.title,
            abstract=report.abstract,
            sections=sections_text,
            references=refs_text,
        )
