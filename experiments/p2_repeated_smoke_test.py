"""
P2 Smoke Test Repeated (n=3) — 给 Writer 错配率加 Bootstrap CI

动机:
  P2 单次 smoke test 报告 "Writer 69.1% 引用错配"，但 n=1 不能说明
  这是系统性问题还是 LLM 随机抖动。本实验跑 3 次完整 Pipeline，
  对 mismatch rate 和 label 分布算 95% bootstrap CI。

设计:
- 每次跑 1-iteration Pipeline + hybrid citation verification
- 记录: mismatch_rate, label 分布, raw attribution reasoning 样本
- 聚合: paired_bootstrap_ci（小样本诚实声明）

成本: ~60 min (3 × 20 min), ~$0.15 API
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
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
from research.evaluation.statistics import format_ci, paired_bootstrap_ci
from research.pipeline.research import ResearchPipeline

QUESTION = "What are recent advances in retrieval-augmented generation?"
N_RUNS = 3


async def run_one(run_id: int) -> dict:
    """跑一次完整 Pipeline + verification"""
    print(f"\n{'='*60}")
    print(f"Run {run_id + 1}/{N_RUNS}")
    print(f"{'='*60}")

    config = PipelineConfig(
        max_iterations=1,
        satisfactory_threshold=10.0,
        knowledge_base=KnowledgeBaseConfig(enabled=False),
        trace_level="full",
        verify_citations=True,
        citation_verification_method="hybrid",
        citation_verification_judge="claude-sonnet-4-6-20250514",
    )
    pipeline = ResearchPipeline(config)

    t_start = time.time()
    result = await pipeline.run(QUESTION)
    elapsed = time.time() - t_start

    cv = result.citation_verification
    if cv is None:
        print(f"⚠️  Run {run_id+1}: citation_verification is None")
        return {
            "run_id": run_id,
            "status": "failed_no_verification",
            "elapsed_sec": round(elapsed, 1),
        }

    # 聚合 attribution label 分布
    label_dist: dict[str, int] = {}
    all_cites = []
    for sec in cv["sections"]:
        for cite in sec.get("citations", []):
            label = cite.get("attribution_label", "n/a")
            label_dist[label] = label_dist.get(label, 0) + 1
            all_cites.append(cite)

    # 检查 attribution 是否真跑通（不是全 error）
    n_valid_attr = sum(c for l, c in label_dist.items()
                       if l in ("matching", "partial", "mismatched", "unverifiable"))
    n_err_attr = label_dist.get("error", 0)

    # 挑 2-3 个 mismatched 的 reasoning 样本保留（IO 证据）
    mismatched_samples = [
        {
            "paper_id": c["paper_id"],
            "section_title": c.get("section_title", "")[:60],
            "paper_title": c.get("title", "")[:80],
            "reasoning": c.get("attribution_reasoning", "")[:300],
            "confidence": c.get("attribution_confidence", 0),
        }
        for c in all_cites if c.get("mismatched")
    ][:3]

    summary = {
        "run_id": run_id,
        "status": "ok" if n_valid_attr > 0 else "attribution_all_failed",
        "elapsed_sec": round(elapsed, 1),
        "num_sections": len(result.report.sections),
        "num_papers": len(result.papers),
        "num_citations_checked": cv["num_citations_checked"],
        "num_mismatched": cv["num_citations_mismatched"],
        "mismatch_rate": cv["overall_mismatch_rate"],
        "attribution_label_distribution": label_dist,
        "attribution_valid_count": n_valid_attr,
        "attribution_error_count": n_err_attr,
        "mismatched_samples": mismatched_samples,  # IO 观察
    }

    print(f"\n  elapsed: {summary['elapsed_sec']}s")
    print(f"  citations: {summary['num_citations_checked']}")
    print(f"  mismatch_rate: {summary['mismatch_rate']:.1%}")
    print(f"  label distribution: {label_dist}")

    return summary


async def main() -> None:
    print("P7: P2 Smoke Test Repeated (n=3)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Question: {QUESTION}\n")

    all_runs = []
    for i in range(N_RUNS):
        try:
            run_result = await run_one(i)
            all_runs.append(run_result)
        except Exception as e:
            print(f"\n⚠️  Run {i+1} CRASHED: {e}")
            all_runs.append({
                "run_id": i,
                "status": "crashed",
                "error": str(e)[:500],
            })

    # === 聚合 ===
    ok_runs = [r for r in all_runs if r.get("status") == "ok"]

    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"Successful runs: {len(ok_runs)}/{N_RUNS}")

    if len(ok_runs) < 2:
        print("⚠️  不足 2 个成功 run，无法算 CI")
        return

    # Bootstrap CI for mismatch rate
    rates = [r["mismatch_rate"] for r in ok_runs]
    r_mean, r_low, r_high = paired_bootstrap_ci(rates, seed=42)

    print(f"\n--- Mismatch Rate ---")
    print(f"  Values across runs: {[f'{r:.1%}' for r in rates]}")
    print(f"  {format_ci(r_mean * 100, r_low * 100, r_high * 100)}  (in %)")

    # Label 分布聚合（每个 label 的比率 mean+CI）
    print(f"\n--- Attribution Label Distribution (rate per run) ---")
    all_labels = set()
    for r in ok_runs:
        all_labels.update(r["attribution_label_distribution"].keys())

    label_stats = {}
    for label in sorted(all_labels):
        ratios = []
        for r in ok_runs:
            cnt = r["attribution_label_distribution"].get(label, 0)
            total = r["num_citations_checked"]
            ratios.append(cnt / total if total > 0 else 0)
        if len(ratios) >= 2:
            mean, low, high = paired_bootstrap_ci(ratios, seed=42)
            print(f"  {label:15s}: {format_ci(mean * 100, low * 100, high * 100, precision=1)}%  "
                  f"(values: {[f'{r:.1%}' for r in ratios]})")
            label_stats[label] = {
                "values_per_run": ratios,
                "mean": round(mean, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
            }

    # Citations per run
    counts = [r["num_citations_checked"] for r in ok_runs]
    print(f"\n--- Pipeline Stats ---")
    print(f"  Citations per run: {counts}")
    print(f"  Sections per run: {[r['num_sections'] for r in ok_runs]}")
    print(f"  Elapsed per run (sec): {[r['elapsed_sec'] for r in ok_runs]}")

    # IO 观察：展示 mismatched 样本
    print(f"\n--- Mismatched Samples (IO 证据，每轮前 2 条) ---")
    for r in ok_runs:
        print(f"\n  Run {r['run_id']+1}:")
        for i, s in enumerate(r.get("mismatched_samples", [])[:2], 1):
            print(f"    [{i}] Section: {s['section_title'][:50]}")
            print(f"        Paper: {s['paper_title'][:65]}")
            print(f"        Reasoning: {s['reasoning'][:200]}")

    # === 保存 ===
    output = {
        "experiment": "p2_repeated_smoke_test",
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "n_runs": N_RUNS,
        "n_successful": len(ok_runs),
        "config": {
            "max_iterations": 1,
            "verify_citations": True,
            "citation_verification_method": "hybrid",
            "judge_model": "claude-sonnet-4-6-20250514",
            "trace_level": "full",
        },
        "mismatch_rate_stats": {
            "values_per_run": rates,
            "mean": round(r_mean, 4),
            "ci_95_low": round(r_low, 4),
            "ci_95_high": round(r_high, 4),
            "format": format_ci(r_mean, r_low, r_high, precision=4),
        },
        "attribution_label_stats": label_stats,
        "raw_runs": all_runs,
    }

    output_path = project_root / "experiments" / "results" / "p2_repeated_smoke_test.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"Saved: {output_path}")

    # 最终结论
    print(f"\n🎯 CONCLUSION:")
    print(f"   Mismatch rate (n={len(ok_runs)}): {r_mean:.1%}, "
          f"95% CI [{r_low:.1%}, {r_high:.1%}]")
    if r_low > 0.3:
        print(f"   ✅ 大于 30%，Writer 错配是系统性问题（不是随机抖动）")
    else:
        print(f"   ⚠️  CI 下界 {r_low:.1%} 较低，可能存在随机抖动")


if __name__ == "__main__":
    asyncio.run(main())
