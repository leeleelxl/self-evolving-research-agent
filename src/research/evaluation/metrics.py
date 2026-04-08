"""
评估指标 — EM / F1 / Critic Scores 分析

标准 NLP 评估指标，用于：
1. HotpotQA 消融实验（EM + F1）
2. 自进化效果分析（Critic Scores 趋势）
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """答案归一化：小写 + 去标点 + 去冠词 + 合并空格

    这是 SQuAD / HotpotQA 标准的归一化方法。
    """
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match — 预测和答案完全一致则 1，否则 0"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 — 预测和答案的 token 重叠度

    F1 = 2 * precision * recall / (precision + recall)
    - precision = 预测中有多少 token 在答案里
    - recall = 答案中有多少 token 在预测里
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: list[str],
    ground_truths: list[str],
) -> dict[str, float]:
    """批量计算 EM 和 F1

    Returns:
        {"exact_match": float, "f1": float, "num_samples": int}
    """
    assert len(predictions) == len(ground_truths)

    em_scores = [exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    f1_scores = [f1_score(p, g) for p, g in zip(predictions, ground_truths)]

    return {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "num_samples": len(predictions),
    }
