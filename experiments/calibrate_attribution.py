"""
Attribution Method Calibration — 验证 v5 pivot 的实际能力

用 P4 已有的 24 条 (section, paper, rater_label) 数据，跑真实 attribution
验证，对比 Claude judge 判定 vs P4 gpt-4o rater 的 ground truth。

为什么不重跑 pipeline:
1. 省时间（~10 min 省掉）
2. 用相同数据对比，能直接看 attribution vs NLI 在相同 24 条上的表现差异
3. P4 数据已有 rater_label，是 ground truth

为什么用 Claude 做 judge:
1. P4 rater 是 gpt-4o（ground truth），judge 用 gpt-4o 会有 self-favoring 偏差
2. Claude (claude-sonnet-4-6) 是独立家族，和 gpt-4o 视角不同
3. 风险：Claude 也是主 Pipeline Critic 副模型，但 Critic 任务（打分）和 Judge 任务（判断 attribution）不同，相关性低

诚实声明:
- Single-LLM judge vs single-LLM rater，仍不是人工真值
- 只能说 "attribution 和 rater 一致度"，不能说 "attribution precision 真值"
- 但比不跑验证强 100 倍
"""

from __future__ import annotations

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

from research.core.models import Paper, ReportSection
from research.evaluation.citation_verifier import CitationVerifier

JUDGE_MODEL = "claude-sonnet-4-6-20250514"  # 独立视角，避免 gpt-4o 评 gpt-4o
INPUT_JSON = project_root / "experiments" / "results" / "nli_calibration.json"


def main() -> None:
    print("=" * 60)
    print("Attribution Method Calibration")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Ground truth rater: gpt-4o (from P4)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # 1. Load P4 data
    with open(INPUT_JSON) as f:
        p4_data = json.load(f)
    samples = p4_data["rated_samples"]
    print(f"\nLoaded {len(samples)} samples from P4 calibration")

    # 2. 初始化 attribution verifier（Claude judge）
    verifier = CitationVerifier(
        method="attribution",
        attribution_judge_model=JUDGE_MODEL,
    )

    # 3. 对每条跑 attribution + IO 观察
    results = []
    for i, s in enumerate(samples, 1):
        paper = Paper(
            paper_id=s["paper_id"],
            title=s["paper_title"],
            abstract=s["paper_abstract"],
            authors=[],
            year=2024,
            url="",
            source="semantic_scholar",
        )
        section = ReportSection(
            section_title=s["section_title"],
            content=s["section_content"],
            cited_papers=[s["paper_id"]],
        )

        # 跑 attribution（内部 async，sync 封装）
        cite_result = verifier._verify_one_attribution(section, paper)

        # IO 观察：打印每条的 LLM judge reasoning
        attr_label = cite_result.get("attribution_label", "error")
        attr_conf = cite_result.get("attribution_confidence", 0)
        attr_reasoning = cite_result.get("attribution_reasoning", "")

        # 标记特殊: rater 判矛盾的（ground truth 真矛盾）
        is_true_contra = s["rater_label"] == "contradiction"
        marker = " ⚠️ TRUE-CONTRA" if is_true_contra else ""

        print(f"\n[{i}/{len(samples)}] {s['paper_id'][:12]}{marker}")
        print(f"  Paper: {s['paper_title'][:65]}")
        print(f"  Rater (ground truth): {s['rater_label']}")
        print(f"  Attribution: {attr_label} (conf={attr_conf})")
        print(f"  Judge reasoning: {attr_reasoning[:250]}")

        results.append({
            "paper_id": s["paper_id"],
            "paper_title": s["paper_title"],
            "section_title": s["section_title"],
            "rater_label": s["rater_label"],  # P4 ground truth
            "rater_reasoning": s["rater_reasoning"],
            "attribution_label": attr_label,
            "attribution_confidence": attr_conf,
            "attribution_reasoning": attr_reasoning,
            "attribution_mismatched": cite_result.get("mismatched", False),
        })

    # 4. 计算指标（attribution=mismatched vs rater=contradiction）
    valid = [r for r in results if r["attribution_label"] != "error"]

    tp = sum(1 for r in valid if r["attribution_mismatched"] and r["rater_label"] == "contradiction")
    fp = sum(1 for r in valid if r["attribution_mismatched"] and r["rater_label"] != "contradiction")
    fn = sum(1 for r in valid if not r["attribution_mismatched"] and r["rater_label"] == "contradiction")
    tn = sum(1 for r in valid if not r["attribution_mismatched"] and r["rater_label"] != "contradiction")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 多分类对比 attribution label 分布
    attr_label_dist: dict[str, int] = {}
    for r in valid:
        attr_label_dist[r["attribution_label"]] = attr_label_dist.get(r["attribution_label"], 0) + 1

    # Agreement: attribution label 和 rater label 对应
    # matching  ↔ support
    # mismatched ↔ contradiction
    # partial   ↔ neutral (partial = scope mismatch)
    # unverifiable ↔ neutral
    label_map = {"matching": "support", "mismatched": "contradiction",
                 "partial": "neutral", "unverifiable": "neutral"}
    agreements = sum(1 for r in valid if label_map.get(r["attribution_label"]) == r["rater_label"])
    agreement_rate = agreements / len(valid) if valid else 0

    print(f"\n{'='*60}")
    print("ATTRIBUTION CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"Valid samples: {len(valid)} / {len(results)}")
    print(f"\nAttribution label distribution: {attr_label_dist}")
    print(f"\nBinary: mismatched vs contradiction")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1: {f1:.1%}")
    print(f"\nMulti-class agreement (attr label ↔ rater label): "
          f"{agreements}/{len(valid)} = {agreement_rate:.1%}")

    # 5. 特别关注 FAIR-RAG 案例（P4 的 1 个 FN，真矛盾）
    print(f"\n{'='*60}")
    print("Critical Case: P4 rater-identified true contradiction")
    print(f"{'='*60}")
    true_contras = [r for r in valid if r["rater_label"] == "contradiction"]
    for r in true_contras:
        judged_right = r["attribution_mismatched"]
        print(f"\n  Paper: {r['paper_title'][:65]}")
        print(f"  Rater reasoning: {r['rater_reasoning'][:200]}")
        print(f"  Attribution judgement: {r['attribution_label']}")
        print(f"  Judge reasoning: {r['attribution_reasoning'][:200]}")
        print(f"  Attribution {'✅ CAUGHT' if judged_right else '❌ MISSED'} this true contradiction")

    # 6. 对比 NLI vs Attribution 在同 24 条数据
    nli_precision = p4_data["metrics"]["precision"]
    nli_recall = p4_data["metrics"]["recall"]
    print(f"\n{'='*60}")
    print("NLI (P4) vs Attribution (P6) — Same 24 Samples")
    print(f"{'='*60}")
    print(f"              NLI              Attribution")
    print(f"  Precision   {nli_precision:.1%}           {precision:.1%}")
    print(f"  Recall      {nli_recall:.1%}           {recall:.1%}")

    # 7. 保存
    output = {
        "experiment": "attribution_calibration",
        "timestamp": datetime.now().isoformat(),
        "judge": {
            "model": JUDGE_MODEL,
            "method": "single-LLM judge (claude independent from gpt-4o rater)",
        },
        "ground_truth_source": {
            "experiment": "P4 nli_calibration",
            "rater_model": "gpt-4o",
            "caveat": "single-LLM rater, not human",
        },
        "n_valid": len(valid),
        "n_errors": len(results) - len(valid),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "multi_class_agreement": round(agreement_rate, 4),
        "attribution_label_distribution": attr_label_dist,
        "nli_comparison": {
            "nli_precision": nli_precision,
            "nli_recall": nli_recall,
        },
        "results": results,
    }

    output_path = project_root / "experiments" / "results" / "attribution_calibration.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
