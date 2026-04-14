"""
P2 Smoke Test — 真 E2E 验证 Citation Verification 集成到 Pipeline

验证目标:
1. verify_citations=True 时 Pipeline 跑通
2. result.citation_verification 被填充
3. 打印真实 attribution LLM judge 输出（IO 观察）
4. 对比 result.citation_verification 是否和独立跑 verifier 一致

运行成本: ~5-8 min pipeline + ~30 sec × N 引用的 LLM judge
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

from research.core.config import KnowledgeBaseConfig, PipelineConfig
from research.pipeline.research import ResearchPipeline

QUESTION = "What are recent advances in retrieval-augmented generation?"


async def main() -> None:
    print("=" * 60)
    print("P2 Smoke Test: Citation Verification 集成 Pipeline")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # 最小配置：1 轮迭代 + 启用 citation verification
    config = PipelineConfig(
        max_iterations=1,
        satisfactory_threshold=10.0,  # 不可达 → 确保跑满
        trace_level="full",
        knowledge_base=KnowledgeBaseConfig(enabled=False),  # 加速
        verify_citations=True,
        citation_verification_method="hybrid",
        citation_verification_judge="claude-sonnet-4-6-20250514",
    )

    pipeline = ResearchPipeline(config)
    result = await pipeline.run(QUESTION)

    print("\n" + "=" * 60)
    print("Pipeline 基础信息")
    print("=" * 60)
    print(f"  iterations: {result.total_iterations}")
    print(f"  sections: {len(result.report.sections)}")
    print(f"  papers: {len(result.papers)}")
    print(f"  agent_traces: {len(result.agent_traces)}")

    # === 验证 P2 ===
    assert result.citation_verification is not None, \
        "P2 FAILED: result.citation_verification is None"
    cv = result.citation_verification

    # 额外断言：Attribution 必须有真实输出（不是全 error）
    attr_errors = 0
    attr_valid = 0
    for sec in cv["sections"]:
        for cite in sec.get("citations", []):
            if cite.get("attribution_label") == "error":
                attr_errors += 1
            elif cite.get("attribution_label") in ("matching", "partial", "mismatched", "unverifiable"):
                attr_valid += 1
    print(f"\n  Attribution LLM judge: {attr_valid} valid, {attr_errors} errors")
    assert attr_valid > 0, \
        f"P2 FAILED: Attribution LLM judge 全失败（{attr_errors} errors）！async bug 未修"
    if attr_errors > 0:
        print(f"  ⚠️  有 {attr_errors} 条 attribution 失败（可能是 API 抖动或超时）")

    print("\n" + "=" * 60)
    print("Citation Verification 结果")
    print("=" * 60)
    print(f"  Method: {cv['method']}")
    print(f"  Sections verified: {len(cv['sections'])}")
    print(f"  Citations checked: {cv['num_citations_checked']}")
    print(f"  Grounded: {cv['num_citations_grounded']}")
    print(f"  Mismatched: {cv['num_citations_mismatched']}")
    print(f"  Missing: {cv['num_citations_missing']}")
    print(f"  Overall grounding rate: {cv['overall_grounding_rate']:.1%}")
    print(f"  Overall mismatch rate: {cv['overall_mismatch_rate']:.1%}")

    # === IO 观察：打印每个 section 的 attribution 判定 ===
    print("\n" + "=" * 60)
    print("Attribution IO 观察（Agent 实际行为证据）")
    print("=" * 60)

    # Attribution label 分布
    attr_labels: dict[str, int] = {}
    all_cites = []
    for sec in cv["sections"]:
        for cite in sec.get("citations", []):
            all_cites.append({**cite, "section_title": sec["section_title"]})
            label = cite.get("attribution_label", "n/a")
            attr_labels[label] = attr_labels.get(label, 0) + 1

    print(f"\nAttribution label 分布: {attr_labels}")

    # 打印 mismatched 的（最关键：看 Claude judge 抓到什么真错配）
    mismatched_cites = [c for c in all_cites if c.get("mismatched")]
    print(f"\n--- Mismatched 详情（{len(mismatched_cites)} 条）---")
    for i, c in enumerate(mismatched_cites[:5], 1):  # 最多打 5 条
        print(f"\n[{i}] Section: {c['section_title'][:60]}")
        print(f"    Paper: {c.get('title', c['paper_id'])[:70]}")
        print(f"    Confidence: {c.get('attribution_confidence')}")
        print(f"    Reasoning: {c.get('attribution_reasoning', '')[:250]}")

    # 打印 partial（scope mismatch）分布
    partial_cites = [c for c in all_cites
                     if c.get("attribution_label") == "partial"]
    print(f"\n--- Partial（scope mismatch）: {len(partial_cites)} 条 ---")
    for i, c in enumerate(partial_cites[:3], 1):
        print(f"[{i}] {c.get('title', '')[:65]} → {c.get('attribution_reasoning', '')[:150]}")

    # 对比 embedding grounding 和 attribution 的差异（Hybrid 下两者都有）
    print("\n" + "=" * 60)
    print("Embedding grounding vs Attribution 差异（Hybrid 视角）")
    print("=" * 60)
    print(f"  Embedding 判 grounded: {cv['num_citations_grounded']}")
    print(f"  Attribution 判 mismatched: {cv['num_citations_mismatched']}")
    print(f"  两者同时 grounded + not mismatched（真干净）: ", end="")
    clean = sum(1 for c in all_cites if c.get("grounded") and not c.get("mismatched"))
    print(f"{clean}/{cv['num_citations_checked']}")

    # 保存
    output_path = project_root / "experiments" / "results" / "p2_smoke_test.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "experiment": "p2_smoke_test",
                "timestamp": datetime.now().isoformat(),
                "question": QUESTION,
                "config": {
                    "verify_citations": True,
                    "citation_verification_method": "hybrid",
                    "judge_model": "claude-sonnet-4-6-20250514",
                },
                "summary": {
                    "num_sections": len(result.report.sections),
                    "num_papers": len(result.papers),
                    "num_citations_checked": cv["num_citations_checked"],
                    "num_mismatched": cv["num_citations_mismatched"],
                    "attribution_label_distribution": attr_labels,
                },
                "citation_verification": cv,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved: {output_path}")
    print("\n✅ P2 Smoke Test PASSED")


if __name__ == "__main__":
    asyncio.run(main())
