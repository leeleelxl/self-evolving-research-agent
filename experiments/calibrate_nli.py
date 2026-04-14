"""
NLI 矛盾检测 Calibration — 用独立 LLM rater 验证 precision/recall

设计:
1. 跑 1 轮 pipeline → 拿到完整 sections + papers
2. 对每个 (section, paper) 对跑 NLI → 完整数据 + 文本
3. Stratified 抽样 30 条 (10 NLI 矛盾 + 20 分层抽样非矛盾)
4. 用 gpt-4o 作为独立 LLM rater 判定每条真实标签
5. 算 precision/recall/F1

为什么选 gpt-4o 做 rater:
- 避免 Claude 作为 rater（主 Pipeline 用 Claude 做 Multi-LLM Critic 副模型，
  Claude-rater-of-Claude-Critic 会引入相关性偏差）
- gpt-4o 和主 Pipeline 用的 gpt-4o-mini 是同家族，但不同 scale，相对独立
- 未来真人审核可用 experiments/results/nli_calibration_sample.csv re-review

诚实声明:
- 这是 single-LLM-rater calibration，不是人工
- 严格 calibration 需要多 human rater + inter-rater agreement (κ>0.6)
- 但比 0 calibration 强 10 倍
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

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

from pydantic import BaseModel, Field

from research.core.config import KnowledgeBaseConfig, PipelineConfig
from research.core.llm import create_llm_client
from research.evaluation.citation_verifier import CitationVerifier
from research.pipeline.research import ResearchPipeline

QUESTION = "What are the recent advances in retrieval-augmented generation (RAG) for large language models?"
N_SAMPLE_CONTRADICTIONS = 10  # 全取 NLI 检测为矛盾的（验证 precision）
N_SAMPLE_NON_CONTRADICTIONS = 20  # 分层抽样非矛盾（验证 recall）
RATER_MODEL = "gpt-4o"  # 独立 LLM rater，不能是 Claude（避免相关性偏差）
RANDOM_SEED = 42


class CalibrationJudgment(BaseModel):
    """Rater 对一条 (section_claim, paper_abstract) 的判定"""

    label: Literal["contradiction", "support", "neutral"] = Field(
        description=(
            "'contradiction': section 声称和 paper abstract 矛盾; "
            "'support': section 声称被 abstract 清晰支持; "
            "'neutral': section 声称和 abstract 相关但不能直接蕴含"
        )
    )
    reasoning: str = Field(description="判定理由，1-2 句话")
    confidence: float = Field(ge=0, le=1, description="判定置信度")


RATER_PROMPT_TEMPLATE = """\
You are a scientific citation accuracy rater. Read a SECTION from a research survey and a PAPER's abstract that the section cites. Determine the logical relationship.

SECTION (from a RAG survey):
{section_content}

CITED PAPER:
Title: {paper_title}
Abstract: {paper_abstract}

Task: Does the section's discussion of this paper CONTRADICT the abstract, SUPPORT it, or is it NEUTRAL (related topic but not directly entailed)?

Definitions:
- contradiction: The section states something that directly conflicts with what the abstract says.
- support: The section's claims about this paper can be reasonably derived from the abstract.
- neutral: The section and abstract are about related topics but there is no direct entailment or contradiction. This is the most common case for survey citations.

Provide your label, reasoning (1-2 sentences), and confidence (0-1).
"""


async def collect_verification_data() -> tuple[list[dict], dict]:
    """跑 pipeline + 对每个 (section, paper) pair 跑 NLI，返回完整数据"""
    print("[1/5] Running pipeline (1 iteration)...")
    config = PipelineConfig(
        max_iterations=1,
        satisfactory_threshold=10.0,
        knowledge_base=KnowledgeBaseConfig(enabled=True),
        trace_level="full",
    )
    pipeline = ResearchPipeline(config)
    result = await pipeline.run(QUESTION)
    print(f"  Pipeline done: {len(result.papers)} papers, "
          f"{len(result.report.sections)} sections")

    print("[2/5] Running hybrid verification with full text preservation...")
    verifier = CitationVerifier(
        method="hybrid",
        grounding_threshold=0.3,
        contradiction_threshold=0.5,
    )
    paper_map = {p.paper_id: p for p in result.papers}

    # 逐对跑，保留文本
    all_pairs: list[dict] = []
    for sec in result.report.sections:
        for pid in sec.cited_papers:
            paper = paper_map.get(pid)
            if paper is None:
                continue
            # 调用 hybrid verifier 的单对验证
            cite = verifier._verify_one_hybrid(sec, paper)  # type: ignore[attr-defined]
            all_pairs.append({
                "section_title": sec.section_title,
                "section_content": sec.content,
                "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "paper_abstract": paper.abstract,
                "nli_entailment": cite["nli_probs"].get("entailment", 0),
                "nli_contradiction": cite["nli_probs"].get("contradiction", 0),
                "nli_neutral": cite["nli_probs"].get("neutral", 0),
                "embedding_similarity": cite.get("embedding_similarity", 0),
                "nli_contradicted": cite.get("contradicted", False),
                "nli_grounded": cite.get("grounded", False),
            })

    summary = {
        "total_pairs": len(all_pairs),
        "nli_contradicted": sum(1 for p in all_pairs if p["nli_contradicted"]),
        "embedding_grounded": sum(1 for p in all_pairs if p["nli_grounded"]),
    }
    print(f"  Total pairs: {summary['total_pairs']}, "
          f"NLI contradicted: {summary['nli_contradicted']}, "
          f"Embedding grounded: {summary['embedding_grounded']}")
    return all_pairs, summary


def stratified_sample(pairs: list[dict]) -> list[dict]:
    """抽样策略: 全取 NLI 矛盾 + 分层抽非矛盾"""
    random.seed(RANDOM_SEED)
    contras = [p for p in pairs if p["nli_contradicted"]]
    non_contras = [p for p in pairs if not p["nli_contradicted"]]

    # 取全部矛盾（不超过 N_SAMPLE_CONTRADICTIONS）
    sample_contras = contras[:N_SAMPLE_CONTRADICTIONS]

    # 非矛盾按 nli_entailment 分层抽 N_SAMPLE_NON_CONTRADICTIONS 条
    non_contras_sorted = sorted(non_contras, key=lambda p: p["nli_entailment"])
    if len(non_contras_sorted) <= N_SAMPLE_NON_CONTRADICTIONS:
        sample_non_contras = non_contras_sorted
    else:
        # 均匀分成 N 层，每层随机取一个
        step = len(non_contras_sorted) / N_SAMPLE_NON_CONTRADICTIONS
        indices = [int(i * step) for i in range(N_SAMPLE_NON_CONTRADICTIONS)]
        sample_non_contras = [non_contras_sorted[i] for i in indices]

    return sample_contras + sample_non_contras


async def rate_with_llm(pair: dict, client) -> dict:
    """调独立 LLM rater 给一对样本打标签"""
    prompt = RATER_PROMPT_TEMPLATE.format(
        section_content=pair["section_content"][:2000],  # 截断避免超 context
        paper_title=pair["paper_title"],
        paper_abstract=pair["paper_abstract"][:1500],
    )
    judgment = await client.generate_structured(
        messages=[{"role": "user", "content": prompt}],
        response_model=CalibrationJudgment,
    )
    return {
        **pair,
        "rater_label": judgment.label,
        "rater_reasoning": judgment.reasoning,
        "rater_confidence": judgment.confidence,
    }


async def run_calibration(samples: list[dict]) -> list[dict]:
    print(f"[4/5] Rating {len(samples)} samples with {RATER_MODEL}...")
    client = create_llm_client("openai", model=RATER_MODEL)

    results = []
    for i, pair in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] Rating {pair['paper_id'][:12]}...", end="", flush=True)
        try:
            rated = await rate_with_llm(pair, client)
            results.append(rated)
            print(f" → {rated['rater_label']}")
        except Exception as e:
            print(f" FAILED: {e!r}")
            results.append({**pair, "rater_label": "error", "rater_reasoning": str(e)})
    return results


def compute_metrics(rated: list[dict]) -> dict:
    """以 rater 判定为 ground truth，算 NLI contradiction detector 的 precision/recall"""
    # 预测: NLI 判为 contradicted
    # 真实: rater 判为 contradiction
    rated_valid = [r for r in rated if r["rater_label"] != "error"]

    tp = sum(1 for r in rated_valid if r["nli_contradicted"] and r["rater_label"] == "contradiction")
    fp = sum(1 for r in rated_valid if r["nli_contradicted"] and r["rater_label"] != "contradiction")
    fn = sum(1 for r in rated_valid if not r["nli_contradicted"] and r["rater_label"] == "contradiction")
    tn = sum(1 for r in rated_valid if not r["nli_contradicted"] and r["rater_label"] != "contradiction")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_rated": len(rated_valid),
        "n_errors": len(rated) - len(rated_valid),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "rater_label_distribution": {
            "contradiction": sum(1 for r in rated_valid if r["rater_label"] == "contradiction"),
            "support": sum(1 for r in rated_valid if r["rater_label"] == "support"),
            "neutral": sum(1 for r in rated_valid if r["rater_label"] == "neutral"),
        },
    }


async def main() -> None:
    print(f"NLI Contradiction Detection Calibration")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    # 1-2: 采集数据
    all_pairs, summary = await collect_verification_data()

    # 3: 抽样
    print(f"[3/5] Stratified sampling "
          f"(up to {N_SAMPLE_CONTRADICTIONS} contradictions + "
          f"{N_SAMPLE_NON_CONTRADICTIONS} non-contradictions)...")
    samples = stratified_sample(all_pairs)
    print(f"  Sampled: {len(samples)} pairs")

    # 4: LLM rater
    rated = await run_calibration(samples)

    # 5: 指标
    print("[5/5] Computing metrics...")
    metrics = compute_metrics(rated)

    # 输出
    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"Sampled: {metrics['n_rated']} valid / {metrics['n_errors']} errors")
    cm = metrics["confusion_matrix"]
    print(f"\nConfusion matrix (NLI claims contradicted):")
    print(f"  TP (NLI=contra, rater=contra): {cm['tp']}")
    print(f"  FP (NLI=contra, rater=other):  {cm['fp']}")
    print(f"  FN (NLI=clean, rater=contra):  {cm['fn']}")
    print(f"  TN (NLI=clean, rater=other):   {cm['tn']}")
    print(f"\nMetrics:")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall: {metrics['recall']:.1%}")
    print(f"  F1: {metrics['f1']:.1%}")
    print(f"\nRater label distribution: {metrics['rater_label_distribution']}")

    # 保存
    results_dir = project_root / "experiments" / "results"
    calibration_dir = project_root / "experiments" / "calibration"
    calibration_dir.mkdir(exist_ok=True)

    output_data = {
        "experiment": "nli_calibration",
        "timestamp": datetime.now().isoformat(),
        "question": QUESTION,
        "rater": {
            "model": RATER_MODEL,
            "method": "single-LLM rater (not human)",
            "caveat": (
                "This is LLM-based single-rater calibration. "
                "Strict calibration requires multi-human raters with "
                "inter-rater agreement (Cohen's κ > 0.6). "
                "Produces precision/recall estimates, not ground truth."
            ),
        },
        "summary": summary,
        "sampled": len(rated),
        "metrics": metrics,
        "rated_samples": rated,
    }

    json_path = results_dir / "nli_calibration.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # CSV 给用户手动 re-review（只放关键字段）
    import csv
    csv_path = calibration_dir / "nli_calibration_sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paper_id", "section_title", "paper_title",
            "nli_entailment", "nli_contradiction", "nli_neutral",
            "nli_contradicted", "rater_label", "rater_confidence",
            "rater_reasoning",
        ])
        writer.writeheader()
        for r in rated:
            writer.writerow({
                "paper_id": r["paper_id"],
                "section_title": r["section_title"][:60],
                "paper_title": r["paper_title"][:60],
                "nli_entailment": r["nli_entailment"],
                "nli_contradiction": r["nli_contradiction"],
                "nli_neutral": r["nli_neutral"],
                "nli_contradicted": r["nli_contradicted"],
                "rater_label": r.get("rater_label", ""),
                "rater_confidence": r.get("rater_confidence", ""),
                "rater_reasoning": r.get("rater_reasoning", "")[:200],
            })

    print(f"\nResults saved:")
    print(f"  JSON (full): {json_path}")
    print(f"  CSV (for re-review): {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
