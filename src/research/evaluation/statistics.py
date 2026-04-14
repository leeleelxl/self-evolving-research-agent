"""
统计工具 — Bootstrap 置信区间

目的: 把 "mean ± std" 升级为 "mean + 95% CI"，支持严谨的显著性声明。
特别针对小样本（n=3~10）实验，bootstrap 比 parametric t-test 更鲁棒。

面试追问应对: "Δ+0.30 显著吗？" → 这里算出的 CI 可以直接回答。

参考: Efron & Tibshirani (1993) An Introduction to the Bootstrap.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    deltas: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """配对 Bootstrap 置信区间

    适用场景: 有 n 对 (before, after) 观测，关心 Δ = after - before 的均值。
    给定 n 个 delta 值，用 bootstrap 估计 delta 均值的 (1-alpha) 置信区间。

    Args:
        deltas: 配对差值列表，如 [0.33, 0.47, 0.10]
        n_bootstrap: bootstrap 采样次数
        alpha: 显著性水平（0.05 → 95% CI）
        seed: 随机种子，保证可复现

    Returns:
        (mean, ci_low, ci_high)

    显著性判定:
        - ci_low > 0   → 方向显著为正
        - ci_high < 0  → 方向显著为负
        - 0 ∈ [low, high] → 未达显著度（小样本常见）
    """
    if not deltas:
        raise ValueError("deltas 不能为空")
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap 至少 100，得到 {n_bootstrap}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0, 1)，得到 {alpha}")

    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)

    # bootstrap: 生成 n_bootstrap × n 的重采样矩阵
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[indices].mean(axis=1)

    ci_low = float(np.percentile(means, 100 * alpha / 2))
    ci_high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    mean = float(arr.mean())

    return mean, ci_low, ci_high


def is_significant(
    ci_low: float, ci_high: float, threshold: float = 0.0
) -> bool:
    """判断 CI 是否完全落在 threshold 的一侧（即显著偏离）"""
    return ci_low > threshold or ci_high < threshold


def format_ci(mean: float, ci_low: float, ci_high: float, precision: int = 2) -> str:
    """格式化 CI 为 README-friendly 字符串

    返回如 "+0.30 (95% CI: [-0.05, +0.65])"
    """
    sign = "+" if mean >= 0 else ""
    return (
        f"{sign}{mean:.{precision}f} "
        f"(95% CI: [{ci_low:+.{precision}f}, {ci_high:+.{precision}f}])"
    )
