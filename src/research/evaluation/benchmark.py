"""
Benchmark 数据加载 — HotpotQA

加载和预处理 HotpotQA 数据集，供消融实验使用。

HotpotQA 数据结构:
- question: 多跳问题
- answer: ground truth 答案
- context: list of (title, sentences) — 10 个段落
- supporting_facts: list of (title, sent_idx) — 支撑事实
- type: "bridge" 或 "comparison"
- level: "easy", "medium", "hard"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HotpotQASample:
    """单个 HotpotQA 样本"""

    id: str
    question: str
    answer: str
    paragraphs: list[str]  # 预处理后的段落文本
    question_type: str  # "bridge" or "comparison"
    level: str  # "easy", "medium", "hard"


def load_hotpotqa(
    path: str | Path,
    num_samples: int | None = None,
    level: str | None = None,
) -> list[HotpotQASample]:
    """加载 HotpotQA 数据集

    Args:
        path: JSON 文件路径
        num_samples: 最多加载的样本数（None = 全部）
        level: 过滤难度（"easy" / "medium" / "hard" / None = 全部）
    """
    with open(path) as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        if level and item.get("level") != level:
            continue

        paragraphs = []
        for title, sentences in item["context"]:
            text = " ".join(sentences)
            paragraphs.append(f"{title}: {text}")

        samples.append(
            HotpotQASample(
                id=item["_id"],
                question=item["question"],
                answer=item["answer"],
                paragraphs=paragraphs,
                question_type=item.get("type", "unknown"),
                level=item.get("level", "unknown"),
            )
        )

        if num_samples and len(samples) >= num_samples:
            break

    return samples
