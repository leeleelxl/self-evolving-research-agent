"""
Critic Agent — 质量评估与自进化驱动

职责:
1. 评估 ResearchReport 的质量（4 个维度）
2. 生成结构化反馈（缺什么、怎么改、该搜什么）
3. 判断是否达标 → 驱动自进化循环

设计选择:
- Multi-LLM 交叉评估（参考 AutoSurvey, arXiv:2406.10252）
  - 主模型 + 副模型分别打分，取均值
  - 记录分歧度（cross_model_spread），分歧大 = 评估不稳定
  - missing_aspects 合并去重，覆盖更全面
- is_satisfactory 在代码层面基于 threshold 重算，不完全信任 LLM 判断
- 评分 prompt 要求给出具体证据，而不只是数字
"""

from __future__ import annotations

from research.core.agent import BaseAgent
from research.core.config import LLMConfig, PipelineConfig
from research.core.llm import BaseLLMClient, create_llm_client
from research.core.models import CriticFeedback, CriticScores, ResearchReport

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

    支持 Multi-LLM 交叉评估：用两个不同的 LLM 分别打分，
    取均值作为最终评分，记录分歧度作为评估置信度指标。

    Usage:
        critic = CriticAgent(pipeline_config)
        feedback = await critic.run(report, question)

        # 查看跨模型分歧度
        spread = feedback.scores.cross_model_spread
        # {'coverage': 2.0, 'depth': 1.0, ...} → coverage 评估最不稳定
    """

    name = "Critic"
    role = SYSTEM_PROMPT

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        config = pipeline_config or PipelineConfig()
        critic_llm_config = llm_config or LLMConfig(model=config.critic_model)
        super().__init__(critic_llm_config)

        self._threshold = config.satisfactory_threshold

        # 副模型（Multi-LLM 交叉验证）
        self._secondary_llm: BaseLLMClient | None = None
        secondary_model = config.critic_secondary_model
        if secondary_model:
            self._secondary_llm = create_llm_client(
                provider="openai",
                model=secondary_model,
                max_tokens=critic_llm_config.max_tokens,
                temperature=critic_llm_config.temperature,
            )
            self.logger.info(
                "multi_llm_critic",
                primary=config.critic_model,
                secondary=secondary_model,
            )

    async def run(self, report: ResearchReport, question: str) -> CriticFeedback:
        """评估综述质量，返回结构化反馈

        如果配置了副模型，会做交叉评估：
        - 分数 = 两个模型的均值
        - cross_model_spread = 各维度的分歧度
        - missing_aspects = 两个模型的合并去重
        """
        prompt = self._build_prompt(report, question)
        self.logger.info("evaluating", title=report.title)

        # 主模型评估
        primary_feedback = await self.generate_structured(prompt, CriticFeedback)

        # 副模型评估（如果启用）
        if self._secondary_llm is not None:
            feedback = await self._cross_evaluate(
                prompt, primary_feedback,
            )
        else:
            feedback = primary_feedback

        # 在代码层面重算 is_satisfactory
        feedback.is_satisfactory = feedback.scores.overall >= self._threshold

        self.logger.info(
            "evaluation_done",
            coverage=feedback.scores.coverage,
            depth=feedback.scores.depth,
            coherence=feedback.scores.coherence,
            accuracy=feedback.scores.accuracy,
            overall=feedback.scores.overall,
            is_satisfactory=feedback.is_satisfactory,
            cross_model=bool(feedback.scores.cross_model_spread),
        )
        return feedback

    async def _cross_evaluate(
        self,
        prompt: str,
        primary: CriticFeedback,
    ) -> CriticFeedback:
        """副模型评估 + 融合结果"""
        assert self._secondary_llm is not None

        try:
            secondary = await self._secondary_llm.generate_structured(
                messages=[{"role": "user", "content": prompt}],
                response_model=CriticFeedback,
                system=self._build_system_prompt(),
            )
        except Exception as e:
            self.logger.warning("secondary_critic_failed", error=str(e)[:200])
            return primary

        p, s = primary.scores, secondary.scores

        # 各维度分歧度
        spread = {
            "coverage": abs(p.coverage - s.coverage),
            "depth": abs(p.depth - s.depth),
            "coherence": abs(p.coherence - s.coherence),
            "accuracy": abs(p.accuracy - s.accuracy),
        }

        # 均值分数
        merged_scores = CriticScores(
            coverage=round((p.coverage + s.coverage) / 2, 1),
            depth=round((p.depth + s.depth) / 2, 1),
            coherence=round((p.coherence + s.coherence) / 2, 1),
            accuracy=round((p.accuracy + s.accuracy) / 2, 1),
            cross_model_spread=spread,
        )

        # missing_aspects 合并去重（两个模型可能指出不同的缺失方向）
        seen = set()
        merged_missing: list[str] = []
        for aspect in primary.missing_aspects + secondary.missing_aspects:
            key = aspect[:50].lower()
            if key not in seen:
                seen.add(key)
                merged_missing.append(aspect)

        # new_queries 同理
        seen_q: set[str] = set()
        merged_queries: list[str] = []
        for q in primary.new_queries + secondary.new_queries:
            key = q.lower().strip()
            if key not in seen_q:
                seen_q.add(key)
                merged_queries.append(q)

        self.logger.info(
            "cross_evaluation",
            primary_overall=p.overall,
            secondary_overall=s.overall,
            merged_overall=merged_scores.overall,
            max_spread=max(spread.values()),
            spread=spread,
        )

        return CriticFeedback(
            scores=merged_scores,
            missing_aspects=merged_missing,
            improvement_suggestions=primary.improvement_suggestions,
            new_queries=merged_queries,
            is_satisfactory=False,  # 下面会重算
        )

    def _build_prompt(self, report: ResearchReport, question: str) -> str:
        sections_text = ""
        if report.sections:
            for s in report.sections:
                sections_text += f"\n#### {s.section_title}\n{s.content}\n"
        else:
            sections_text = "(No sections yet)"

        refs_text = ", ".join(report.references) if report.references else "(No references)"

        return EVALUATE_PROMPT_TEMPLATE.format(
            question=question,
            title=report.title,
            abstract=report.abstract,
            sections=sections_text,
            references=refs_text,
        )
