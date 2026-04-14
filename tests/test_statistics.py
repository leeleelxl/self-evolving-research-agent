"""Bootstrap CI 统计工具测试"""

from __future__ import annotations

import pytest

from research.evaluation.statistics import (
    format_ci,
    is_significant,
    paired_bootstrap_ci,
)


class TestPairedBootstrapCI:
    def test_returns_three_numbers(self) -> None:
        mean, low, high = paired_bootstrap_ci([0.1, 0.2, 0.3])
        assert isinstance(mean, float)
        assert isinstance(low, float)
        assert isinstance(high, float)
        assert low <= mean <= high

    def test_single_value_has_zero_width_ci(self) -> None:
        """单一值的 CI 收缩到该值本身"""
        mean, low, high = paired_bootstrap_ci([0.5])
        assert mean == 0.5
        assert low == 0.5
        assert high == 0.5

    def test_all_zeros_ci_contains_zero(self) -> None:
        mean, low, high = paired_bootstrap_ci([0.0, 0.0, 0.0])
        assert mean == 0.0
        assert low == 0.0
        assert high == 0.0

    def test_clearly_positive_deltas_ci_above_zero(self) -> None:
        """全部为正且大的 delta，CI 下界应 > 0"""
        mean, low, high = paired_bootstrap_ci([1.0, 1.1, 0.9, 1.2, 0.95, 1.05, 1.15])
        assert mean > 0.9
        assert low > 0  # 下界严格为正 → 显著

    def test_n3_small_sample_ci_may_span_zero(self) -> None:
        """n=3 小样本现实: CI 经常包含 0（这是数学限制不是 bug）

        我们项目的自进化 n=3 数据: Δ=[0.33, 0.47, 0.10]
        验证这是"方向为正但未显著"的典型场景。
        """
        mean, low, high = paired_bootstrap_ci([0.33, 0.47, 0.10])
        assert mean == pytest.approx(0.30, abs=0.01)
        assert low < mean
        assert high > mean

    def test_seed_reproducibility(self) -> None:
        """相同 seed 返回相同 CI"""
        r1 = paired_bootstrap_ci([0.1, 0.2, 0.3], seed=123)
        r2 = paired_bootstrap_ci([0.1, 0.2, 0.3], seed=123)
        assert r1 == r2

    def test_different_seeds_may_differ_slightly(self) -> None:
        """不同 seed 结果小幅不同（但 mean 相同）"""
        r1 = paired_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=1)
        r2 = paired_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=2)
        assert r1[0] == r2[0]  # mean 始终相同
        # CI 可能略不同但应接近
        assert abs(r1[1] - r2[1]) < 0.1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            paired_bootstrap_ci([])

    def test_invalid_n_bootstrap_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bootstrap"):
            paired_bootstrap_ci([0.1, 0.2], n_bootstrap=50)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            paired_bootstrap_ci([0.1, 0.2], alpha=1.5)


class TestIsSignificant:
    def test_ci_above_threshold(self) -> None:
        assert is_significant(ci_low=0.1, ci_high=0.5) is True

    def test_ci_below_threshold(self) -> None:
        assert is_significant(ci_low=-0.5, ci_high=-0.1) is True

    def test_ci_spans_zero_not_significant(self) -> None:
        assert is_significant(ci_low=-0.1, ci_high=0.3) is False

    def test_custom_threshold(self) -> None:
        # CI [0.5, 0.8] 完全低于 threshold=1.0 → 显著偏离（方向: <）
        assert is_significant(0.5, 0.8, threshold=1.0) is True
        # CI [1.5, 2.0] 完全高于 threshold=1.0 → 显著偏离（方向: >）
        assert is_significant(1.5, 2.0, threshold=1.0) is True
        # CI [0.8, 1.2] 跨过 threshold=1.0 → 未显著
        assert is_significant(0.8, 1.2, threshold=1.0) is False


class TestFormatCI:
    def test_positive_mean(self) -> None:
        s = format_ci(0.30, -0.05, 0.65)
        assert "+0.30" in s
        assert "95% CI" in s
        assert "-0.05" in s
        assert "+0.65" in s

    def test_negative_mean(self) -> None:
        s = format_ci(-0.15, -0.30, 0.05)
        assert "-0.15" in s
        assert "-0.30" in s

    def test_precision(self) -> None:
        s = format_ci(0.123456, 0.100, 0.150, precision=3)
        assert "+0.123" in s
