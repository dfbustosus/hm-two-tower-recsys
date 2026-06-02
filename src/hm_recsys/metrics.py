from __future__ import annotations

from collections.abc import Iterable, Mapping


def dedupe_preserve_order(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return tuple(deduped)


def average_precision_at_k(actual: Iterable[str], predicted: Iterable[str], k: int = 12) -> float:
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
    if k <= 0:
        raise ValueError("k must be positive")

    actual_set = set(actual)
    if not actual_set:
        return 0.0
    predicted_set = set(tuple(predicted)[:k])
    return len(actual_set & predicted_set) / len(actual_set)
