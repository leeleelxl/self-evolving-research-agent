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
    3. KnowledgeBase: 索引论文，按子问题检索最相关的子集
    4. Reader: 只精读相关论文（而非全部）
    5. Writer: 生成综述
    6. Critic: 评估质量
    → 达标则输出，否则带反馈进入下一轮
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from research.agents.critic import CriticAgent
from research.agents.planner import PlannerAgent
from research.agents.reader import ReaderAgent
from research.agents.retriever import RetrieverAgent
from research.agents.writer import WriterAgent
from research.core.config import PipelineConfig
from research.core.models import (
    AgentTrace,
    CriticFeedback,
    EvolutionRecord,
    Paper,
    PaperNote,
    PipelineResult,
    ResearchPlan,
    ResearchReport,
)
from research.retrieval.knowledge_base import KnowledgeBase
from research.retrieval.pdf import fetch_full_texts

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

        # KnowledgeBase: 按需延迟初始化（embedding 模型加载慢）
        self._kb: KnowledgeBase | None = None
        self._kb_enabled = self.config.knowledge_base.enabled

        self._evolution_log: list[EvolutionRecord] = []
        self._agent_traces: list[AgentTrace] = []

    def _record_trace(
        self,
        agent_name: str,
        iteration: int,
        input_summary: str,
        output: dict[str, Any],
    ) -> None:
        """记录单次 Agent 调用的完整 IO，用于事后验证 Agent 是否按预期行动

        trace_level 控制粒度:
        - minimal: 只记录 Planner + Critic（决策证据）
        - standard: + Reader 的精简字段
        - full: 所有 Agent 完整 output
        """
        level = self.config.trace_level

        if level == "minimal" and agent_name not in ("Planner", "Critic"):
            return

        # standard 模式下 Reader 输出精简（不存 methodology/key_findings/limitations 全文）
        if level == "standard" and agent_name == "Reader" and "notes" in output:
            output = {
                "notes": [
                    {
                        "paper_id": n["paper_id"],
                        "title": n["title"],
                        "core_contribution": n["core_contribution"],
                        "relevance_score": n["relevance_score"],
                        "relevance_reason": n["relevance_reason"],
                    }
                    for n in output["notes"]
                ],
                "kept": output.get("kept", len(output.get("notes", []))),
            }

        self._agent_traces.append(
            AgentTrace(
                agent_name=agent_name,  # type: ignore[arg-type]
                iteration=iteration,
                input_summary=input_summary,
                output=output,
                timestamp_ms=int(time.time() * 1000),
            )
        )

    async def run(self, question: str) -> PipelineResult:
        """执行完整的研究 Pipeline"""
        self.logger.info("pipeline_start", question=question)

        self._evolution_log = []
        self._agent_traces = []
        all_notes: list[PaperNote] = []
        all_papers_map: dict[str, Paper] = {}  # 跨轮累积所有论文
        seen_paper_ids: set[str] = set()  # 跨轮去重
        previous_queries: list[str] = []  # 已执行的 query，传给 Planner 避免重复
        feedback: CriticFeedback | None = None
        report: ResearchReport | None = None

        for iteration in range(self.config.max_iterations):
            self.logger.info("iteration_start", iteration=iteration)

            try:
                # ── Step 1: 规划 ──
                plan = await self._planner.run(
                    question, feedback, iteration, previous_queries,
                )
                self._record_trace(
                    agent_name="Planner",
                    iteration=iteration,
                    input_summary=(
                        f"question='{question[:80]}', "
                        f"has_feedback={feedback is not None}, "
                        f"previous_queries={len(previous_queries)}"
                    ),
                    output=plan.model_dump(),
                )

                # 记录本轮的 query（供下一轮去重）
                for sq in plan.search_strategy.queries:
                    previous_queries.append(sq.query)

                # ── Step 2: 检索 ──
                papers = await self._retriever.run(plan)
                self._record_trace(
                    agent_name="Retriever",
                    iteration=iteration,
                    input_summary=f"{len(plan.search_strategy.queries)} queries",
                    output={
                        "papers": [
                            {
                                "paper_id": p.paper_id,
                                "title": p.title,
                                "year": p.year,
                                "citations": p.citations,
                                "has_pdf_url": p.pdf_url is not None,
                            }
                            for p in papers
                        ],
                        "total": len(papers),
                    },
                )

                # 检索到 0 篇 → 跳过本轮，保留已有数据
                if not papers:
                    self.logger.warning("no_papers_found", iteration=iteration)
                    continue

                # 收集论文（用于 post-hoc 引用验证）
                for p in papers:
                    all_papers_map[p.paper_id] = p

                # ── Step 2.5: PDF 全文提取（启用时） ──
                if self.config.pdf.enabled:
                    await fetch_full_texts(papers, self.config.pdf)

                # ── Step 3: KnowledgeBase 过滤 ──
                papers_to_read = self._filter_by_knowledge_base(papers, plan)

                # ── Step 4: 精读（只读 KB 筛选后的论文） ──
                notes = await self._reader.run(papers_to_read, question)
                self._record_trace(
                    agent_name="Reader",
                    iteration=iteration,
                    input_summary=(
                        f"{len(papers_to_read)} papers to read, "
                        f"pdf_enabled={self.config.pdf.enabled}"
                    ),
                    output={
                        "notes": [n.model_dump() for n in notes],
                        "kept": len(notes),
                        "filtered_out": len(papers_to_read) - len(notes),
                    },
                )

                # 跨轮去重：只保留新论文的笔记
                new_notes = []
                for note in notes:
                    if note.paper_id not in seen_paper_ids:
                        seen_paper_ids.add(note.paper_id)
                        new_notes.append(note)
                all_notes.extend(new_notes)

                # ── Step 5: 写作（用所有轮次的笔记） ──
                report = await self._writer.run(all_notes, plan)
                self._record_trace(
                    agent_name="Writer",
                    iteration=iteration,
                    input_summary=f"{len(all_notes)} total notes (new={len(new_notes)})",
                    output=report.model_dump(),
                )

                # ── Step 6: 评估 ──
                feedback = await self._critic.run(report, question)
                self._record_trace(
                    agent_name="Critic",
                    iteration=iteration,
                    input_summary=(
                        f"report: {len(report.sections)} sections, "
                        f"{len(report.references)} refs"
                    ),
                    output=feedback.model_dump(),
                )

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
                    kb_filtered=self._kb_enabled,
                    papers_retrieved=len(papers),
                    papers_to_read=len(papers_to_read),
                )

                if feedback.is_satisfactory:
                    self.logger.info("pipeline_converged", iterations=iteration + 1)
                    break

            except Exception as e:
                self.logger.error(
                    "iteration_failed",
                    iteration=iteration,
                    error=str(e)[:300],
                )
                # 单轮失败不终止整个 Pipeline，继续下一轮
                continue

        if report is None:
            self.logger.error("pipeline_no_report", question=question[:100])
            # 兜底：返回空报告，而非 crash
            report = ResearchReport(
                title="Research Report (incomplete)",
                abstract=f"Pipeline failed to generate a complete report for: {question}",
                sections=[],
                references=[],
            )

        return PipelineResult(
            report=report,
            evolution_log=self._evolution_log,
            total_iterations=len(self._evolution_log),
            papers=list(all_papers_map.values()),
            agent_traces=self._agent_traces,
        )

    def _filter_by_knowledge_base(
        self,
        papers: list[Paper],
        plan: ResearchPlan,
    ) -> list[Paper]:
        """用 KnowledgeBase 过滤论文：对每个子问题检索最相关的论文

        如果 KB 未启用，直接返回全部论文（兼容消融实验）。
        """
        if not self._kb_enabled:
            return papers

        # 延迟初始化：第一次调用时才加载 embedding 模型
        if self._kb is None:
            self._kb = KnowledgeBase(self.config)

        # 增量索引新论文
        self._kb.add_papers(papers)

        # 按子问题检索
        kb_config = self.config.knowledge_base
        relevant = self._kb.retrieve_for_questions(
            plan.sub_questions,
            top_k=kb_config.top_k_per_question,
        )

        self.logger.info(
            "kb_filter",
            total_papers=len(papers),
            kb_total_indexed=self._kb.num_papers,
            relevant_papers=len(relevant),
        )

        return relevant
