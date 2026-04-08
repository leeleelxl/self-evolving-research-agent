"""核心基础设施 — 数据模型、配置、LLM 封装、Agent 基类"""

from research.core.agent import BaseAgent
from research.core.config import LLMConfig, PipelineConfig
from research.core.llm import BaseLLMClient, create_llm_client
from research.core.models import (
    CriticFeedback,
    CriticScores,
    Paper,
    PaperNote,
    PipelineResult,
    ResearchPlan,
    ResearchReport,
)

__all__ = [
    "BaseAgent",
    "LLMConfig",
    "PipelineConfig",
    "BaseLLMClient",
    "create_llm_client",
    "CriticFeedback",
    "CriticScores",
    "Paper",
    "PaperNote",
    "PipelineResult",
    "ResearchPlan",
    "ResearchReport",
]
