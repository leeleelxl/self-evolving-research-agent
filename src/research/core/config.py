"""
配置管理 — 消融实验的基础

所有可调参数集中在这里。消融实验 = 遍历配置组合。
用 Pydantic Settings 管理，支持环境变量覆盖。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM 调用配置"""

    provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="LLM 提供商: openai=中转站GPT, anthropic=Claude",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="默认模型，gpt-4o-mini 性价比最高",
    )
    max_tokens: int = 4096
    temperature: float = Field(
        default=0.0,
        description="学术场景用 0，减少随机性，保证可复现",
    )


class ChunkConfig(BaseModel):
    """Chunk 策略配置 — 消融变量 A"""

    strategy: Literal["fixed", "semantic", "recursive"] = Field(
        default="recursive",
        description="fixed=固定大小, semantic=按语义边界, recursive=递归切分",
    )
    chunk_size: int = Field(default=512, description="目标 chunk 大小 (token 数)")
    chunk_overlap: int = Field(default=64, description="相邻 chunk 的重叠 token 数")
    max_chunk_size: int = Field(default=1024, description="递归策略的最大 chunk 大小")


class RetrievalConfig(BaseModel):
    """检索策略配置 — 消融变量 B"""

    strategy: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid",
        description="dense=FAISS向量, sparse=BM25关键词, hybrid=混合",
    )
    top_k: int = Field(default=20, description="检索返回的 chunk 数")
    hybrid_weight: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="混合检索中 dense 的权重 (1-weight 给 sparse)",
    )


class RerankConfig(BaseModel):
    """Rerank 策略配置 — 消融变量 C"""

    enabled: bool = Field(default=True, description="是否启用 rerank")
    top_k: int = Field(default=10, description="rerank 后保留的 chunk 数")


class KnowledgeBaseConfig(BaseModel):
    """KnowledgeBase 配置 — 控制论文知识库的构建和检索"""

    enabled: bool = Field(
        default=True,
        description="是否启用 KnowledgeBase 过滤（关闭则 Reader 读所有论文）",
    )
    top_k_per_question: int = Field(
        default=10,
        ge=1,
        description="每个子问题从知识库检索的论文数",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding 模型名称",
    )


class PDFConfig(BaseModel):
    """PDF 全文处理配置"""

    enabled: bool = Field(
        default=False,
        description="是否启用 PDF 全文提取（关闭则仅用 abstract）",
    )
    max_pages: int = Field(
        default=30,
        ge=1,
        le=100,
        description="每篇 PDF 最多提取的页数",
    )
    max_text_length: int = Field(
        default=50000,
        ge=1000,
        description="提取后全文的最大字符数（截断保护）",
    )
    download_timeout: float = Field(
        default=30.0,
        description="PDF 下载超时（秒）",
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        description="最大并发下载数",
    )


class PipelineConfig(BaseModel):
    """Pipeline 总配置 — 一次实验的完整参数"""

    # LLM
    llm: LLMConfig = Field(default_factory=LLMConfig)
    critic_model: str = Field(
        default="gpt-4o",
        description="Critic 主模型",
    )
    critic_secondary_model: str = Field(
        default="claude-sonnet-4-6-20250514",
        description="Critic 副模型（交叉验证，设为空字符串禁用）",
    )

    # RAG pipeline
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)

    # 自进化
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="最大自进化轮数",
    )
    satisfactory_threshold: float = Field(
        default=7.0,
        ge=0,
        le=10,
        description="Critic 评分达到此阈值即停止迭代",
    )

    # 检索
    max_papers_per_query: int = Field(default=20, description="每个 query 最多返回的论文数")
    max_papers_total: int = Field(default=50, description="单轮迭代最多保留的论文总数")

    # Agent IO 追踪
    trace_level: Literal["minimal", "standard", "full"] = Field(
        default="full",
        description=(
            "Agent IO 追踪粒度: "
            "minimal=仅存 Planner/Critic 的决策文本; "
            "standard=含 Reader 的 core_contribution/relevance_reason; "
            "full=全部 Agent 完整 output（研究项目推荐）"
        ),
    )

    # 引用验证（opt-in）
    verify_citations: bool = Field(
        default=False,
        description=(
            "是否启用引用质量验证。启用后 Pipeline 末尾跑 CitationVerifier，"
            "结果写入 PipelineResult.citation_verification。"
            "Hybrid 方法会调用独立 LLM judge（attribution），有 API 成本。"
        ),
    )
    citation_verification_method: Literal["embedding", "attribution", "hybrid"] = Field(
        default="hybrid",
        description=(
            "引用验证方法: embedding (快速非 LLM) / attribution (LLM-judge) / "
            "hybrid (embedding grounding + attribution, 推荐)"
        ),
    )
    citation_verification_judge: str = Field(
        default="claude-sonnet-4-6-20250514",
        description=(
            "Attribution LLM judge 模型（走中转站 OpenAI API）。"
            "推荐用和主 Pipeline Critic 不同的模型避免相关性偏差。"
        ),
    )
