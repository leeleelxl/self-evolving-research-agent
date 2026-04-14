"""
Post-hoc 统计分析 — 对已有实验数据计算 95% bootstrap CI

目的:
  之前实验只给了 mean ± std，用户追问"Δ+0.30 显著吗"答不上。
  此脚本对现有 experiments/results/*.json 应用 paired_bootstrap_ci，
  输出 confidence_intervals.json，让 README 数字可被严谨查证。

诚实声明:
  对于 n<10 的小样本，bootstrap CI 数学上受限于 [min, max] of 观测值。
  这不能替代大样本显著性检验，但仍比 "mean ± std" 更严谨。

用法:
  conda run -n base python experiments/compute_ci.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from research.evaluation.statistics import (
    format_ci,
    is_significant,
    paired_bootstrap_ci,
)

RESULTS_DIR = project_root / "experiments" / "results"


def analyze_self_evolution() -> dict:
    """对 self_evolution_repeated.json 的 n=3 数据算 CI"""
    path = RESULTS_DIR / "self_evolution_repeated.json"
    with open(path) as f:
        data = json.load(f)

    stats = data["stats"]
    analysis: dict = {"source": str(path.name)}

    # Evolution Δ (核心指标)
    deltas = stats["evolution_delta"]["values"]
    mean, low, high = paired_bootstrap_ci(deltas, seed=42)
    analysis["evolution_delta"] = {
        "values": deltas,
        "n": len(deltas),
        "mean": round(mean, 4),
        "ci_95_low": round(low, 4),
        "ci_95_high": round(high, 4),
        "format": format_ci(mean, low, high),
        "is_significant": is_significant(low, high),
    }

    # Overall 各 run 分布
    overalls = stats["overall"]["values"]
    mean, low, high = paired_bootstrap_ci(overalls, seed=42)
    analysis["overall"] = {
        "values": overalls,
        "n": len(overalls),
        "mean": round(mean, 4),
        "ci_95_low": round(low, 4),
        "ci_95_high": round(high, 4),
        "format": format_ci(mean, low, high),
    }

    return analysis


def main() -> None:
    print("Bootstrap CI Post-hoc Analysis")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    all_analyses: dict = {
        "experiment": "bootstrap_ci_posthoc",
        "timestamp": datetime.now().isoformat(),
        "method": "paired_bootstrap_ci (n_bootstrap=10000, alpha=0.05, seed=42)",
        "caveat": (
            "小样本（n<10）的 bootstrap CI 数学上受限于观测值的 [min, max]。"
            "CI 不含 0 只能说明所有观测方向一致，"
            "不等价于大样本下的严格显著性检验。"
            "大样本（n≥30）验证是未来工作（见 README 已知局限）。"
        ),
        "analyses": {},
    }

    print("=" * 60)
    print("Self-Evolution Experiment (n=3)")
    print("=" * 60)
    se_analysis = analyze_self_evolution()
    all_analyses["analyses"]["self_evolution"] = se_analysis

    for metric_name, metric in se_analysis.items():
        if metric_name == "source":
            continue
        print(f"\n{metric_name}:")
        print(f"  n = {metric['n']}, values = {metric['values']}")
        print(f"  → {metric['format']}")
        if metric.get("is_significant"):
            print(f"  → 显著（CI 不含 0）")
        elif "is_significant" in metric:
            print(f"  → 未显著（CI 含 0）")

    # 保存
    output_path = RESULTS_DIR / "bootstrap_ci_posthoc.json"
    with open(output_path, "w") as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
