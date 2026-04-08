"""LLM 集成测试 — 验证中转站 API 可用

这些测试会真实调用 API，需要有 .env 中的 API key。
用 pytest -m integration 单独运行。
"""

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from research.core.llm import OpenAIClient, create_llm_client


class SimpleOutput(BaseModel):
    """测试用的简单结构化输出"""

    answer: str = Field(description="The answer")
    confidence: float = Field(ge=0, le=1, description="Confidence score")


@pytest.mark.integration
class TestOpenAIClient:
    """中转站 API 集成测试"""

    @pytest.mark.asyncio
    async def test_generate_text(self) -> None:
        """流式文本生成"""
        client = create_llm_client("openai", model="gpt-4o-mini")
        result = await client.generate(
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )
        assert "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_structured(self) -> None:
        """流式 structured output"""
        client = create_llm_client("openai", model="gpt-4o-mini")
        result = await client.generate_structured(
            messages=[{"role": "user", "content": "What is 2+2? Be confident."}],
            response_model=SimpleOutput,
        )
        assert isinstance(result, SimpleOutput)
        assert "4" in result.answer
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        """system prompt 生效"""
        client = create_llm_client("openai", model="gpt-4o-mini")
        result = await client.generate(
            messages=[{"role": "user", "content": "What is your role?"}],
            system="You are a helpful math tutor. Always mention you are a math tutor.",
        )
        assert "math" in result.lower() or "tutor" in result.lower()
