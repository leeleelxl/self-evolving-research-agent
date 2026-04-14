"""
核心数据模型 — Agent 之间的通信契约

设计原则:
1. 每个模型对应 Pipeline 中的一个数据流转节点
2. 用 Pydantic v2 做运行时校验 + LLM structured output 的 schema
3. 模型之间通过 paper_id 等 ID 字段关联，而非嵌套引用（避免循环依赖）
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================
# 论文与知识
# ============================================================


class Paper(BaseModel):
    """从学术 API 获取的原始论文元信息"""

    paper_id: str = Field(description="Semantic Scholar 或 arXiv ID")
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    year: int
    url: str
    source: Literal["semantic_scholar", "arxiv"]
    citations: int = 0
    pdf_url: str | None = None
    full_text: str | None = Field(
        default=None,
        description="PDF 全文（提取后填入），为 None 时降级到 abstract",
    )


class Chunk(BaseModel):
    """论文的文本片段 — 检索的最小单位

    为什么要 chunk？LLM 的 context window 有限，
    且检索时整篇论文粒度太粗，无法定位具体段落。
    chunk 是 RAG 的核心概念：把长文档切成小片段，
    每个片段独立做 embedding 和检索。
    """

    chunk_id: str
    paper_id: str = Field(description="所属论文的 ID")
    text: str
    chunk_index: int = Field(description="在原文中的顺序位置")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="附加信息，如 section_title, page_num",
    )


class PaperNote(BaseModel):
    """Reader Agent 对单篇论文的结构化精读笔记

    这不是论文原文，而是 Reader Agent 提取的关键信息。
    结构化格式确保 Writer Agent 能高效消费，
    不需要重新阅读原文。
    """

    paper_id: str
    title: str
    core_contribution: str = Field(description="一句话总结核心贡献")
    methodology: str = Field(description="方法论概述")
    key_findings: list[str] = Field(description="关键发现列表")
    limitations: list[str] = Field(default_factory=list, description="局限性")
    relevance_score: float = Field(ge=0, le=1, description="与研究问题的相关度")
    relevance_reason: str = Field(description="为什么相关/不相关")


# ============================================================
# 研究规划
# ============================================================


class SearchQuery(BaseModel):
    """单次检索请求 — Retriever Agent 的执行单元"""

    query: str
    source: Literal["semantic_scholar", "arxiv"] = "semantic_scholar"
    max_results: int = Field(default=20, ge=1, le=100)
    year_min: int | None = Field(default=None, description="最早发表年份，如 2020")
    year_max: int | None = Field(default=None, description="最晚发表年份，如 2026")


class SearchStrategy(BaseModel):
    """一轮检索的完整策略 — Planner Agent 的核心输出

    自进化的关键：Critic 反馈后，Planner 会修改这个策略。
    比如 coverage 低 → 增加 queries；depth 低 → 增加 focus_areas。
    """

    queries: list[SearchQuery] = Field(description="要执行的检索列表")
    focus_areas: list[str] = Field(
        default_factory=list,
        description="应重点关注的研究方向",
    )
    exclude_terms: list[str] = Field(
        default_factory=list,
        description="应排除的内容（避免重复检索）",
    )


class ResearchPlan(BaseModel):
    """Planner Agent 的完整输出 — 一轮研究的行动计划"""

    original_question: str
    sub_questions: list[str] = Field(description="分解后的子问题")
    search_strategy: SearchStrategy
    iteration: int = Field(default=0, description="当前迭代轮次")


# ============================================================
# 输出
# ============================================================


class ReportSection(BaseModel):
    """综述报告的一个章节"""

    section_title: str
    content: str
    cited_papers: list[str] = Field(
        default_factory=list,
        description="本节引用的论文 ID",
    )


class ResearchReport(BaseModel):
    """Writer Agent 的输出 — 完整的研究综述"""

    title: str
    abstract: str
    sections: list[ReportSection] = Field(default_factory=list)
    references: list[str] = Field(
        default_factory=list,
        description="所有引用的论文 ID",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 评估与反馈 — 自进化的驱动力
# ============================================================


class CriticScores(BaseModel):
    """Critic Agent 的多维度评分

    四个维度的设计参考了学术论文审稿的评估标准。
    每个维度 0-10 分，便于量化追踪自进化效果。

    支持 Multi-LLM 交叉评估：当使用双模型时，
    scores 是均值，cross_model_spread 记录各维度分歧度。
    """

    coverage: float = Field(ge=0, le=10, description="文献覆盖度")
    depth: float = Field(ge=0, le=10, description="分析深度")
    coherence: float = Field(ge=0, le=10, description="逻辑连贯性")
    accuracy: float = Field(ge=0, le=10, description="事实准确性")
    cross_model_spread: dict[str, float] = Field(
        default_factory=dict,
        description="双模型评分分歧度（各维度的 |model1 - model2|），为空表示单模型",
    )

    @property
    def overall(self) -> float:
        """综合评分 — 四个维度的均值"""
        return (self.coverage + self.depth + self.coherence + self.accuracy) / 4


class CriticFeedback(BaseModel):
    """Critic Agent 的完整输出 — 结构化评估反馈

    这是自进化机制的核心数据结构。
    不只是打分，还要给出可执行的改进建议。
    new_queries 字段尤其重要：直接告诉 Planner 该搜什么。
    """

    scores: CriticScores
    missing_aspects: list[str] = Field(description="综述中缺失的研究方向")
    improvement_suggestions: list[str] = Field(description="具体改进建议")
    new_queries: list[str] = Field(
        default_factory=list,
        description="建议的新检索 query，可直接执行",
    )
    is_satisfactory: bool = Field(description="综述质量是否达标")


# ============================================================
# Pipeline 追踪 — 用于消融实验和效果分析
# ============================================================


class EvolutionRecord(BaseModel):
    """单轮迭代的快照 — 记录自进化过程

    每轮迭代后保存一条记录，用于:
    1. 画自进化效果曲线（分数 vs 迭代轮次）
    2. 分析策略变化（每轮加了什么 query、改了什么 focus）
    3. 消融实验的数据支撑
    """

    iteration: int
    scores: CriticScores
    num_papers: int = Field(description="本轮检索到的论文数")
    num_notes: int = Field(description="本轮精读的论文数")
    strategy_snapshot: SearchStrategy = Field(description="本轮使用的策略快照")
    feedback_summary: str = Field(
        default="",
        description="Critic 反馈摘要",
    )


class AgentTrace(BaseModel):
    """单次 Agent 调用的完整 IO 记录 — Agent 项目的核心证据

    做 Agent 项目时，聚合数字（分数、计数）能说谎：
    - "自进化 Δ+0.30" 可能来自 Critic 随机抖动，非真进化
    - "queries 从 8→38" 可能是同义词换皮，非真 diverge
    - "100% grounding" 可能因 Reader 只是 paraphrase abstract

    必须保存每次 Agent 调用的完整文本 IO，才能验证 Agent 是否按预期行动。
    """

    agent_name: Literal["Planner", "Retriever", "Reader", "Writer", "Critic"]
    iteration: int = Field(description="属于哪一轮自进化迭代")
    input_summary: str = Field(description="输入的简要描述，如 'question + feedback'")
    output: dict[str, Any] = Field(
        description="Agent 的完整结构化输出（Pydantic .model_dump()）",
    )
    timestamp_ms: int = Field(description="调用开始时的毫秒时间戳")


class PipelineResult(BaseModel):
    """Pipeline 的最终输出 — 包含报告、进化过程、Agent IO 追踪"""

    report: ResearchReport
    evolution_log: list[EvolutionRecord] = Field(default_factory=list)
    total_iterations: int
    papers: list[Paper] = Field(
        default_factory=list,
        description="Pipeline 中索引的所有论文（用于 post-hoc 引用质量验证）",
    )
    agent_traces: list[AgentTrace] = Field(
        default_factory=list,
        description="每次 Agent 调用的完整 IO 记录，用于验证 Agent 是否按预期行动",
    )
