"""CLI 入口 — python -m research "你的研究问题" """

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
import structlog

load_dotenv()  # 从 .env 加载 API key

from research.core.config import PipelineConfig
from research.pipeline.research import ResearchPipeline

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)


async def main(question: str) -> None:
    config = PipelineConfig()
    pipeline = ResearchPipeline(config)
    result = await pipeline.run(question)

    print("\n" + "=" * 60)
    print(f"Title: {result.report.title}")
    print(f"Iterations: {result.total_iterations}")
    print("=" * 60)

    for record in result.evolution_log:
        scores = record.scores
        print(
            f"  Round {record.iteration}: "
            f"coverage={scores.coverage:.1f} depth={scores.depth:.1f} "
            f"coherence={scores.coherence:.1f} accuracy={scores.accuracy:.1f} "
            f"→ overall={scores.overall:.1f}"
        )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are recent advances in LLM agents?"
    asyncio.run(main(question))
