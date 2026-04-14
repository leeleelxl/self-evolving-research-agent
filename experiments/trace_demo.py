"""
Agent IO 追踪验证 Demo — 跑一个最小 Pipeline 并保存完整 agent_traces

目的: 验证 P0 改动后，一次真实 pipeline 运行能否保留所有 Agent 的完整 IO，
特别是 Planner Round 0 vs Round 1 的 queries diverge 是否可观察。

运行后产物:
- experiments/results/trace_demo.json (含 agent_traces)
- 用 `python experiments/inspect_agent_io.py experiments/results/trace_demo.json --diff-queries` 对比
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
)

from research.core.config import PipelineConfig
from research.pipeline.research import ResearchPipeline

QUESTION = "What are the recent advances in retrieval-augmented generation (RAG) for large language models?"


async def main() -> None:
    print(f"Agent IO Trace Demo")
    print(f"Question: {QUESTION}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    # 2 轮迭代是观察自进化的最小需求
    config = PipelineConfig(
        max_iterations=2,
        satisfactory_threshold=9.5,  # 故意高阈值，逼系统必跑 2 轮
        trace_level="full",
    )
    pipeline = ResearchPipeline(config)

    result = await pipeline.run(QUESTION)

    print(f"\nPipeline done: {result.total_iterations} iterations")
    print(f"Agent traces recorded: {len(result.agent_traces)}")

    # 分类计数
    by_agent: dict[str, int] = {}
    for t in result.agent_traces:
        by_agent[t.agent_name] = by_agent.get(t.agent_name, 0) + 1
    print(f"By agent: {by_agent}")

    # 保存
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "trace_demo.json"

    experiment_data = {
        "experiment": "trace_demo",
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "config": {
            "max_iterations": 2,
            "trace_level": "full",
        },
        "total_iterations": result.total_iterations,
        "agent_traces": [t.model_dump() for t in result.agent_traces],
        "final_report_title": result.report.title,
        "final_sections": len(result.report.sections),
        "final_references": len(result.report.references),
    }

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)

    size_kb = output_path.stat().st_size / 1024
    print(f"\nResults saved to: {output_path} ({size_kb:.1f} KB)")
    print(f"\nNext step:")
    print(f"  python experiments/inspect_agent_io.py {output_path} --diff-queries")
    print(f"  python experiments/inspect_agent_io.py {output_path} --agent Critic")


if __name__ == "__main__":
    asyncio.run(main())
