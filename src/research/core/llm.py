"""
LLM 封装层 — 统一的多后端 LLM 调用接口

核心设计：
1. 统一接口: generate() / generate_structured() 对上层 Agent 透明
2. 多后端: 支持 OpenAI 兼容 API（中转站）和 Anthropic Claude
3. 流式累积: 中转站只支持流式，内部用流式收集完整结果再返回
4. structured output: 通过 tool/function calling 让 LLM 直接返回 Pydantic 模型

不依赖 LangChain，直接用官方 SDK。
好处：面试时能讲清楚每一层在做什么，不会被问到框架内部细节答不上来。
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """LLM 客户端抽象接口

    所有 Agent 通过这个接口调用 LLM，不感知底层是 Claude 还是 GPT。
    这样：
    - 切换 provider 只需改配置，不改 Agent 代码
    - 消融实验可以对比不同模型的效果
    """

    def __init__(self, model: str, max_tokens: int, temperature: float) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> str:
        """生成自由文本"""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        """生成结构化输出（Pydantic 模型实例）"""
        ...

    @property
    def total_tokens(self) -> dict[str, int]:
        return {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
            "total": self._total_input_tokens + self._total_output_tokens,
        }

    @staticmethod
    def _dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """内联展开 Pydantic 生成的 JSON Schema 中的 $defs/$ref

        Pydantic 对嵌套模型会生成 $defs + $ref 引用:
          {"properties": {"strategy": {"$ref": "#/$defs/SearchStrategy"}}, "$defs": {...}}

        OpenAI API 不支持这种格式，需要把引用替换为实际定义。
        """
        schema = deepcopy(schema)
        defs = schema.pop("$defs", {})
        if not defs:
            return schema

        def resolve(obj: Any) -> Any:
            if isinstance(obj, dict):
                # 处理 {"$ref": "#/$defs/ModelName"}
                if "$ref" in obj:
                    ref_path = obj["$ref"]  # e.g. "#/$defs/SearchStrategy"
                    ref_name = ref_path.split("/")[-1]
                    if ref_name in defs:
                        return resolve(deepcopy(defs[ref_name]))
                    return obj
                # 处理 {"allOf": [{"$ref": "..."}]}（Pydantic 常生成这种）
                if "allOf" in obj and len(obj["allOf"]) == 1:
                    resolved = resolve(obj["allOf"][0])
                    # 保留 allOf 同级的其他字段（如 description）
                    extra = {k: v for k, v in obj.items() if k != "allOf"}
                    if isinstance(resolved, dict):
                        resolved.update(extra)
                    return resolved
                return {k: resolve(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [resolve(item) for item in obj]
            return obj

        return resolve(schema)


# ============================================================
# OpenAI 兼容客户端（中转站用这个）
# ============================================================


class OpenAIClient(BaseLLMClient):
    """OpenAI 兼容 API 客户端

    特点：
    - 必须用流式（中转站非流式返回空 content）
    - 内部用流式收集完整结果，对外接口仍是同步返回
    - structured output 通过 function calling 实现

    Usage:
        client = OpenAIClient(model="gpt-4o-mini")
        text = await client.generate(messages=[...])
        plan = await client.generate_structured(messages=[...], response_model=ResearchPlan)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, max_tokens, temperature)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> str:
        """流式收集文本生成结果"""
        all_messages = self._prepend_system(messages, system)

        stream = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=all_messages,
            stream=True,
        )

        chunks: list[str] = []
        async for chunk in stream:
            choice = chunk.choices[0]
            if choice.delta.content:
                chunks.append(choice.delta.content)
            # 从最后一个 chunk 获取 usage（如果有）
            if hasattr(chunk, "usage") and chunk.usage:
                self._track_usage_raw(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)

        result = "".join(chunks)
        logger.debug("llm_call", model=self.model, response_len=len(result))
        return result

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        """通过 function calling 实现 structured output

        实现原理（和 Claude 的 tool_use 思路一致）：
        1. 把 Pydantic schema 包装成 OpenAI function 定义
        2. 用 tool_choice="required" 强制 LLM 调用该 function
        3. 流式收集 function arguments JSON
        4. 用 Pydantic 解析 → 类型安全的模型实例
        """
        tool_name = response_model.__name__
        # 内联展开 $defs 引用，否则 OpenAI API 会报错
        tool_schema = self._dereference_schema(response_model.model_json_schema())

        all_messages = self._prepend_system(messages, system)

        stream = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=all_messages,
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Output structured data as {tool_name}",
                        "parameters": tool_schema,
                    },
                }
            ],
            tool_choice="required",
        )

        # 流式收集 function call 的 arguments
        arg_chunks: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        arg_chunks.append(tc.function.arguments)

        raw_json = "".join(arg_chunks)
        logger.debug("structured_output", model=self.model, tool=tool_name, json_len=len(raw_json))

        return response_model.model_validate(json.loads(raw_json))

    @staticmethod
    def _prepend_system(
        messages: list[dict[str, str]],
        system: str | None,
    ) -> list[dict[str, str]]:
        """OpenAI 格式：system prompt 作为第一条 message"""
        if system:
            return [{"role": "system", "content": system}, *messages]
        return messages

    def _track_usage_raw(self, input_tokens: int, output_tokens: int) -> None:
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens


# ============================================================
# Anthropic Claude 客户端
# ============================================================


class AnthropicClient(BaseLLMClient):
    """Claude API 客户端

    Usage:
        client = AnthropicClient(model="claude-sonnet-4-6-20250514")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(model, max_tokens, temperature)
        import anthropic

        self._client = anthropic.AsyncAnthropic()

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        self._track_usage_anthropic(response)
        return response.content[0].text

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        tool_name = response_model.__name__
        tool_schema = response_model.model_json_schema()

        tools = [
            {
                "name": tool_name,
                "description": f"Output structured data as {tool_name}",
                "input_schema": tool_schema,
            }
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        self._track_usage_anthropic(response)

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return response_model.model_validate(block.input)

        raise ValueError(f"LLM did not return expected tool call: {tool_name}")

    def _track_usage_anthropic(self, response: Any) -> None:
        usage = response.usage
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        logger.debug(
            "llm_call",
            model=self.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )


# ============================================================
# 工厂函数 — 根据配置创建对应客户端
# ============================================================


def create_llm_client(
    provider: str = "openai",
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseLLMClient:
    """根据 provider 创建 LLM 客户端

    Args:
        provider: "openai" 或 "anthropic"
        model: 模型名。默认 openai→gpt-4o-mini, anthropic→claude-sonnet
        **kwargs: 传给具体客户端的额外参数（如 base_url）

    Usage:
        # 用中转站的 GPT
        client = create_llm_client("openai", model="gpt-4o-mini")

        # 用 Claude
        client = create_llm_client("anthropic", model="claude-sonnet-4-6-20250514")
    """
    if provider == "openai":
        return OpenAIClient(
            model=model or "gpt-4o-mini",
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    elif provider == "anthropic":
        return AnthropicClient(
            model=model or "claude-sonnet-4-6-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
