import pytest

from hm_recsys.metrics import (
    average_precision_at_k,
    dedupe_preserve_order,
    mean_average_precision_at_k,
    recall_at_k,
)


def test_dedupe_preserve_order_keeps_first_occurrence() -> None:
    assert dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ("a", "b", "c")


def test_average_precision_at_k_known_example() -> None:
    score = average_precision_at_k(actual=["a", "b"], predicted=["a", "c", "b"], k=12)

    assert score == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_at_k_duplicate_predictions_consume_rank_slots() -> None:
    score = average_precision_at_k(actual=["a", "b"], predicted=["a", "a", "b"], k=12)

    assert score == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_at_k_does_not_rescue_items_beyond_k_after_dedupe() -> None:
    score = average_precision_at_k(actual=["b"], predicted=["a", "a", "b"], k=2)

    assert score == 0.0


def test_average_precision_at_k_deduplicates_repeated_actual_purchases() -> None:
    assert average_precision_at_k(actual=["a", "a", "b"], predicted=["b", "a"], k=12) == 1.0


def test_mean_average_precision_excludes_empty_actuals_by_default() -> None:
    score = mean_average_precision_at_k(
        actual_by_customer={"c1": ["a"], "c2": []},
        predicted_by_customer={"c1": ["a"], "c2": ["z"]},
        k=12,
    )

    assert score == 1.0


def test_recall_at_k_uses_unique_actuals_and_predictions() -> None:
    assert recall_at_k(actual=["a", "a", "b", "c"], predicted=["a", "a", "z", "b"], k=3) == 1 / 3


def test_recall_at_k_does_not_rescue_items_beyond_k_after_dedupe() -> None:
    assert recall_at_k(actual=["b"], predicted=["a", "a", "b"], k=2) == 0.0


def test_metric_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        average_precision_at_k(actual=["a"], predicted=["a"], k=0)
