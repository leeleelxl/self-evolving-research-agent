"""
Agent 基类 — 多 Agent 系统的统一接口

设计选择:
- 不强制 ReAct 循环。每个 Agent 的执行模式不同:
  - Planner/Writer/Critic: 单次 structured output 调用
  - Retriever: 多步 tool calling 循环
  - Reader: 批量并发处理
- 基类只提供公共能力: LLM 调用、日志、配置
- 每个 Agent 子类自定义 run() 逻辑

为什么不用 LangChain Agent?
1. LangChain Agent 强制 ReAct 循环，不适合所有场景
2. 黑盒太多，面试时讲不清内部机制
3. 依赖链太深（LangChain → LangGraph → ...），一个版本不兼容全挂
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel

from research.core.config import LLMConfig
from research.core.llm import BaseLLMClient, create_llm_client

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """Agent 基类

    子类只需实现:
    1. name, role 属性
    2. run() 方法
    """

    name: str  # Agent 名称，如 "Planner"
    role: str  # 角色描述，会作为 system prompt 的一部分

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        config = llm_config or LLMConfig()
        self.llm: BaseLLMClient = create_llm_client(
            provider=config.provider,
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        self.logger = logger.bind(agent=self.name)

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """执行 Agent 的核心逻辑 — 子类必须实现

        不同 Agent 的签名不同:
        - Planner.run(question, feedback?) → ResearchPlan
        - Retriever.run(plan) → list[Paper]
        - Reader.run(papers, question) → list[PaperNote]
        - Writer.run(notes, plan) → ResearchReport
        - Critic.run(report, question) → CriticFeedback
        """
        ...

    # ── 公共能力 ──

    async def generate_text(self, user_message: str) -> str:
        """简单的文本生成 — 传入用户消息，返回回复文本"""
        self.logger.info("generating_text", message_preview=user_message[:100])
        return await self.llm.generate(
            messages=[{"role": "user", "content": user_message}],
            system=self._build_system_prompt(),
        )

    async def generate_structured(
        self,
        user_message: str,
        response_model: type[T],
    ) -> T:
        """结构化输出 — 让 LLM 直接返回 Pydantic 模型实例

        大部分 Agent 的核心调用方式。示例:
            plan = await self.generate_structured(
                "分解这个研究问题: ...",
                ResearchPlan,
            )
        """
        self.logger.info(
            "generating_structured",
            model=response_model.__name__,
            message_preview=user_message[:100],
        )
        return await self.llm.generate_structured(
            messages=[{"role": "user", "content": user_message}],
            response_model=response_model,
            system=self._build_system_prompt(),
        )

    def _build_system_prompt(self) -> str:
        """构建 system prompt — 包含角色描述"""
        return f"You are {self.name}, a specialized agent in an academic research system.\n\nRole: {self.role}"
