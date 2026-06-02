"""Leakage-safe ranking metrics for H&M MAP@12 validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def dedupe_preserve_order(items: Iterable[str]) -> tuple[str, ...]:
    """Remove duplicate IDs while preserving first occurrence order.

    Args:
        items: Ordered iterable of item identifiers.

    Returns:
        Tuple containing the first occurrence of each item.
    """

    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return tuple(deduped)


def average_precision_at_k(actual: Iterable[str], predicted: Iterable[str], k: int = 12) -> float:
    """Compute average precision at ``k`` for one customer.

    Duplicate predictions consume rank slots but receive credit only once, matching
    the H&M/Kaggle MAP@12 contract.

    Args:
        actual: Relevant article IDs for one customer; repeated purchases are
            treated as a unique relevance set.
        predicted: Ranked article IDs for the same customer.
        k: Maximum rank depth to evaluate.

    Returns:
        Average precision at ``k``. Returns ``0.0`` when ``actual`` is empty.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    actual_set = set(actual)
    if not actual_set:
        return 0.0

    denominator = min(len(actual_set), k)
    hits = 0
    score = 0.0
    seen_predictions: set[str] = set()
    for rank, article_id in enumerate(tuple(predicted)[:k], start=1):
        if article_id in actual_set and article_id not in seen_predictions:
            hits += 1
            score += hits / rank
        seen_predictions.add(article_id)

    return score / denominator


def mean_average_precision_at_k(
    actual_by_customer: Mapping[str, Iterable[str]],
    predicted_by_customer: Mapping[str, Iterable[str]],
    k: int = 12,
    exclude_empty_actuals: bool = True,
) -> float:
    """Compute mean average precision at ``k`` across customers.

    Args:
        actual_by_customer: Mapping from customer ID to relevant article IDs.
        predicted_by_customer: Mapping from customer ID to ranked predictions.
        k: Maximum rank depth to evaluate.
        exclude_empty_actuals: Whether to skip customers with no relevant labels,
            matching Kaggle's scoring behavior for the hidden target week.

    Returns:
        Mean AP@K over evaluated customers, or ``0.0`` when no customers remain.
    """

    scores: list[float] = []
    for customer_id, actual in actual_by_customer.items():
        actual_tuple = tuple(actual)
        if exclude_empty_actuals and not actual_tuple:
            continue
        scores.append(
            average_precision_at_k(
                actual=actual_tuple,
                predicted=predicted_by_customer.get(customer_id, ()),
                k=k,
            )
        )
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(actual: Iterable[str], predicted: Iterable[str], k: int) -> float:
    """Compute set recall at ``k`` for one customer.

    Args:
        actual: Relevant article IDs for one customer.
        predicted: Ranked predictions for the same customer.
        k: Maximum rank depth to evaluate.

    Returns:
        Fraction of unique relevant articles present in the top ``k`` predictions.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    actual_set = set(actual)
    if not actual_set:
        return 0.0
    predicted_set = set(tuple(predicted)[:k])
    return len(actual_set & predicted_set) / len(actual_set)
