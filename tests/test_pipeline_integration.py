"""Pipeline 端到端集成测试 — 真 API + IO-aware 断言

设计原则（源自 memory/feedback_agent_io_observation.md）:
- 不只 assert "不 crash"
- 必须 assert Agent 实际行为的合理性（sub_questions >= 2、queries 真正 diverge 等）
- 用 agent_traces 作为证据，不依赖 Critic 评分

默认 skip（`pytest` 不跑，需 `pytest -m integration` 显式触发）。
每个测试 ~2-3min 真 API，会产生少量 API 费用。
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.core.config import KnowledgeBaseConfig, PipelineConfig
from research.pipeline.research import ResearchPipeline


pytestmark = pytest.mark.integration


class TestPipelineEndToEnd:
    """真调 API 的端到端测试，验证 Pipeline 在真实环境工作 + Agent IO 合理性"""

    @pytest.mark.asyncio
    async def test_single_iteration_with_real_api(self) -> None:
        """单轮真 API 跑通 + 5 个 Agent 的 IO 都合理

        这个测试消除了"测试虚高"的担忧 — 它真的调了所有外部服务:
        Planner (LLM) → Retriever (Semantic Scholar + arXiv) → Reader (LLM)
        → Writer (LLM) → Critic (双 LLM)。
        """
        config = PipelineConfig(
            max_iterations=1,
            satisfactory_threshold=10.0,  # 不可达 → 必跑满 1 轮
            trace_level="full",
            knowledge_base=KnowledgeBaseConfig(enabled=False),  # 加速，避免 embedding 模型加载
        )
        pipeline = ResearchPipeline(config)
        result = await pipeline.run("What is retrieval-augmented generation?")

        # ── 基础结构 ──
        assert result is not None
        assert result.total_iterations == 1
        assert result.report is not None

        # ── Agent IO 完整性：5 个 Agent 各 1 次 ──
        assert len(result.agent_traces) == 5
        agent_names = [t.agent_name for t in result.agent_traces]
        for expected in ["Planner", "Retriever", "Reader", "Writer", "Critic"]:
            assert expected in agent_names, f"{expected} trace missing"

        # ── Planner: 分解出多个子问题 + 多样 query ──
        planner_trace = next(t for t in result.agent_traces if t.agent_name == "Planner")
        sub_qs = planner_trace.output["sub_questions"]
        assert len(sub_qs) >= 2, f"Planner only gave {len(sub_qs)} sub_questions"
        queries = planner_trace.output["search_strategy"]["queries"]
        assert len(queries) >= 2, f"Planner only gave {len(queries)} queries"
        # 每个 query 非空字符串
        for q in queries:
            assert isinstance(q["query"], str) and len(q["query"]) > 5

        # ── Retriever: 真的搜到论文 ──
        retriever_trace = next(t for t in result.agent_traces if t.agent_name == "Retriever")
        total_papers = retriever_trace.output["total"]
        assert total_papers >= 3, f"Retriever only got {total_papers} papers"

        # ── Reader: 精读不是 abstract 复述 ──
        reader_trace = next(t for t in result.agent_traces if t.agent_name == "Reader")
        notes = reader_trace.output["notes"]
        assert reader_trace.output["kept"] >= 1
        for note in notes:
            # core_contribution 至少 20 字符（避免"A paper about X"这种空壳）
            assert len(note["core_contribution"]) >= 20
            # relevance_reason 有实质内容
            assert len(note["relevance_reason"]) >= 10
            # relevance_score 在合理区间
            assert 0 <= note["relevance_score"] <= 1

        # ── Writer: 生成有效 report ──
        writer_trace = next(t for t in result.agent_traces if t.agent_name == "Writer")
        assert len(writer_trace.output["title"]) > 0
        assert len(writer_trace.output["sections"]) >= 1
        # 至少引用了一些论文
        assert len(writer_trace.output["references"]) >= 1

        # ── Critic: 打分合理 + 给出具体建议（如果未达标） ──
        critic_trace = next(t for t in result.agent_traces if t.agent_name == "Critic")
        scores = critic_trace.output["scores"]
        for dim in ["coverage", "depth", "coherence", "accuracy"]:
            assert 0 <= scores[dim] <= 10, f"{dim}={scores[dim]} out of range"
        # 因为阈值设为 10.0，is_satisfactory 应该 False → 有 suggestions
        if not critic_trace.output["is_satisfactory"]:
            assert len(critic_trace.output["improvement_suggestions"]) >= 1
            # 建议内容不是空话（每条至少 20 字符）
            for sug in critic_trace.output["improvement_suggestions"]:
                assert len(sug) >= 20

    @pytest.mark.asyncio
    async def test_self_evolution_actually_diverges(self) -> None:
        """真 2 轮 Pipeline → Planner Round 1 必须给出和 Round 0 不同的 queries

        这是"自进化是真的"的自动化证据。之前只在 trace_demo.py 人工观察到
        Round 0→1 38 个新 query 0 重复，现在用 assertion 保证每次都满足。
        """
        config = PipelineConfig(
            max_iterations=2,
            satisfactory_threshold=10.0,  # 必跑 2 轮
            trace_level="full",
            knowledge_base=KnowledgeBaseConfig(enabled=False),
        )
        pipeline = ResearchPipeline(config)
        result = await pipeline.run(
            "What are the recent advances in retrieval-augmented generation?"
        )

        # 至少跑通 2 轮
        planner_traces = [t for t in result.agent_traces if t.agent_name == "Planner"]
        assert len(planner_traces) >= 2, f"Only {len(planner_traces)} Planner iterations"

        round0_queries = {
            q["query"] for q in planner_traces[0].output["search_strategy"]["queries"]
        }
        round1_queries = {
            q["query"] for q in planner_traces[1].output["search_strategy"]["queries"]
        }

        # ── 核心断言：自进化真的 diverge ──
        new_in_round1 = round1_queries - round0_queries
        # Round 1 至少要有一些新 query（否则说明 Planner 没有响应 Critic feedback）
        assert len(new_in_round1) >= 3, (
            f"Self-evolution failed: Round 1 only added {len(new_in_round1)} new queries.\n"
            f"Round 0: {round0_queries}\n"
            f"Round 1 new: {new_in_round1}"
        )

        # Critic 给出了 improvement_suggestions（feedback loop 的起点）
        critic_traces = [t for t in result.agent_traces if t.agent_name == "Critic"]
        assert len(critic_traces[0].output["improvement_suggestions"]) >= 1
