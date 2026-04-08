"""核心模块的单元测试"""

from research.core.config import ChunkConfig, PipelineConfig, RetrievalConfig
from research.core.models import (
    CriticFeedback,
    CriticScores,
    EvolutionRecord,
    Paper,
    PaperNote,
    PipelineResult,
    ResearchPlan,
    ResearchReport,
    SearchQuery,
    SearchStrategy,
)


class TestModels:
    """数据模型的基础测试 — 确保 Pydantic 校验正常"""

    def test_paper_creation(self) -> None:
        paper = Paper(
            paper_id="abc123",
            title="Test Paper",
            abstract="A test abstract",
            authors=["Author A"],
            year=2024,
            url="https://example.com",
            source="semantic_scholar",
        )
        assert paper.paper_id == "abc123"
        assert paper.source == "semantic_scholar"
        assert paper.citations == 0  # 默认值

    def test_critic_scores_overall(self) -> None:
        scores = CriticScores(coverage=8.0, depth=6.0, coherence=7.0, accuracy=9.0)
        assert scores.overall == 7.5  # (8+6+7+9) / 4

    def test_critic_feedback(self) -> None:
        feedback = CriticFeedback(
            scores=CriticScores(coverage=8.0, depth=6.0, coherence=7.0, accuracy=9.0),
            missing_aspects=["recent work on X"],
            improvement_suggestions=["add more papers from 2024"],
            new_queries=["X method 2024 survey"],
            is_satisfactory=True,
        )
        assert feedback.is_satisfactory
        assert len(feedback.new_queries) == 1

    def test_research_plan(self) -> None:
        plan = ResearchPlan(
            original_question="What is RAG?",
            sub_questions=["What is retrieval augmented generation?", "How does RAG work?"],
            search_strategy=SearchStrategy(
                queries=[SearchQuery(query="RAG survey 2024")],
                focus_areas=["retrieval methods"],
            ),
        )
        assert plan.iteration == 0
        assert len(plan.sub_questions) == 2

    def test_pipeline_result(self) -> None:
        result = PipelineResult(
            report=ResearchReport(
                title="Test",
                abstract="Test abstract",
            ),
            evolution_log=[
                EvolutionRecord(
                    iteration=0,
                    scores=CriticScores(coverage=5.0, depth=5.0, coherence=5.0, accuracy=5.0),
                    num_papers=10,
                    num_notes=8,
                    strategy_snapshot=SearchStrategy(queries=[], focus_areas=[]),
                ),
            ],
            total_iterations=1,
        )
        assert result.total_iterations == 1
        assert result.evolution_log[0].scores.overall == 5.0


class TestConfig:
    """配置模块的测试"""

    def test_default_config(self) -> None:
        config = PipelineConfig()
        assert config.max_iterations == 3
        assert config.satisfactory_threshold == 7.0
        assert config.llm.temperature == 0.0

    def test_chunk_config_strategies(self) -> None:
        for strategy in ["fixed", "semantic", "recursive"]:
            config = ChunkConfig(strategy=strategy)
            assert config.strategy == strategy

    def test_retrieval_config_strategies(self) -> None:
        for strategy in ["dense", "sparse", "hybrid"]:
            config = RetrievalConfig(strategy=strategy)
            assert config.strategy == strategy

    def test_custom_config(self) -> None:
        """消融实验场景: 自定义配置组合"""
        config = PipelineConfig(
            chunk=ChunkConfig(strategy="semantic", chunk_size=256),
            retrieval=RetrievalConfig(strategy="dense", top_k=10),
            rerank={"enabled": False},
            max_iterations=5,
        )
        assert config.chunk.strategy == "semantic"
        assert config.retrieval.strategy == "dense"
        assert not config.rerank.enabled
        assert config.max_iterations == 5
