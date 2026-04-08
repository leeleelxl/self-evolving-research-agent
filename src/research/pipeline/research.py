"""
研究 Pipeline — 多 Agent 编排与自进化循环

编排模式: Sequential + 条件循环
- 不用 LangGraph/DAG，因为 Agent 间有严格数据依赖，无法并行
- 自进化循环: Critic 不满意 → 带反馈回 Planner → 新一轮迭代
- 状态累积: 每轮新检索的论文累加到知识库，不是从头开始

核心流程:
  for each iteration:
    1. Planner: 分解问题 / 根据反馈调整策略
    2. Retriever: 执行检索
    3. Reader: 精读论文
    4. Writer: 生成综述
    5. Critic: 评估质量
    → 达标则输出，否则带反馈进入下一轮
"""

from __future__ import annotations

import structlog

from research.agents.critic import CriticAgent
from research.agents.planner import PlannerAgent
from research.agents.reader import ReaderAgent
from research.agents.retriever import RetrieverAgent
from research.agents.writer import WriterAgent
from research.core.config import PipelineConfig
from research.core.models import (
    CriticFeedback,
    EvolutionRecord,
    Paper,
    PaperNote,
    PipelineResult,
    ResearchPlan,
    ResearchReport,
)

logger = structlog.get_logger()


class ResearchPipeline:
    """多 Agent 研究 Pipeline

    Usage:
        config = PipelineConfig(max_iterations=3)
        pipeline = ResearchPipeline(config)
        result = await pipeline.run("Transformer 架构的最新改进有哪些？")

        # 查看自进化效果
        for record in result.evolution_log:
            print(f"Round {record.iteration}: {record.scores.overall:.1f}")
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = logger.bind(component="pipeline")

        # 创建 Agent 实例
        llm_config = self.config.llm
        self._planner = PlannerAgent(llm_config)
        self._retriever = RetrieverAgent(pipeline_config=self.config, llm_config=llm_config)
        self._reader = ReaderAgent(llm_config)
        self._writer = WriterAgent(llm_config)
        self._critic = CriticAgent(pipeline_config=self.config)

        self._evolution_log: list[EvolutionRecord] = []

    async def run(self, question: str) -> PipelineResult:
        """执行完整的研究 Pipeline"""
        self.logger.info("pipeline_start", question=question)

        self._evolution_log = []
        all_notes: list[PaperNote] = []
        seen_paper_ids: set[str] = set()  # 跨轮去重
        feedback: CriticFeedback | None = None
        report: ResearchReport | None = None

        for iteration in range(self.config.max_iterations):
            self.logger.info("iteration_start", iteration=iteration)

            # ── Step 1: 规划 ──
            plan = await self._planner.run(question, feedback, iteration)

            # ── Step 2: 检索 ──
            papers = await self._retriever.run(plan)

            # ── Step 3: 精读 ──
            notes = await self._reader.run(papers, question)

            # 跨轮去重：只保留新论文的笔记
            new_notes = []
            for note in notes:
                if note.paper_id not in seen_paper_ids:
                    seen_paper_ids.add(note.paper_id)
                    new_notes.append(note)
            all_notes.extend(new_notes)

            # ── Step 4: 写作（用所有轮次的笔记） ──
            report = await self._writer.run(all_notes, plan)

            # ── Step 5: 评估 ──
            feedback = await self._critic.run(report, question)

            # ── 记录进化数据 ──
            self._evolution_log.append(
                EvolutionRecord(
                    iteration=iteration,
                    scores=feedback.scores,
                    num_papers=len(papers),
                    num_notes=len(new_notes),
                    strategy_snapshot=plan.search_strategy,
                    feedback_summary="; ".join(feedback.improvement_suggestions[:3]),
                )
            )

            self.logger.info(
                "iteration_end",
                iteration=iteration,
                overall_score=feedback.scores.overall,
                is_satisfactory=feedback.is_satisfactory,
                total_notes=len(all_notes),
            )

            if feedback.is_satisfactory:
                self.logger.info("pipeline_converged", iterations=iteration + 1)
                break

        assert report is not None, "Pipeline must produce a report"

        return PipelineResult(
            report=report,
            evolution_log=self._evolution_log,
            total_iterations=len(self._evolution_log),
        )
