"""评估模块测试"""

from research.evaluation.metrics import compute_metrics, exact_match, f1_score, normalize_answer


class TestMetrics:

    def test_normalize_answer(self) -> None:
        assert normalize_answer("The Cat.") == "cat"
        assert normalize_answer("  a  big  dog  ") == "big dog"
        assert normalize_answer("Hello, World!") == "hello world"

    def test_exact_match(self) -> None:
        assert exact_match("yes", "yes") == 1.0
        assert exact_match("Yes", "yes") == 1.0  # 大小写不敏感
        assert exact_match("The answer", "answer") == 1.0  # 去冠词
        assert exact_match("no", "yes") == 0.0

    def test_f1_score(self) -> None:
        # 完全匹配
        assert f1_score("the cat sat", "the cat sat") == 1.0
        # 部分匹配
        f1 = f1_score("the cat sat on the mat", "the cat sat")
        assert 0.5 < f1 < 1.0
        # 无匹配
        assert f1_score("hello world", "foo bar") == 0.0

    def test_compute_metrics(self) -> None:
        preds = ["yes", "Paris", "hello"]
        truths = ["yes", "paris", "world"]
        result = compute_metrics(preds, truths)

        assert result["num_samples"] == 3
        assert result["exact_match"] > 0  # 至少 "yes" 和 "Paris" 匹配
        assert 0 < result["f1"] < 1
