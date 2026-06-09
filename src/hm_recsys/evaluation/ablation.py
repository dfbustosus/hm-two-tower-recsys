"""Ablation harness for quantifying per-feature MAP@K contributions.

The harness is intentionally tiny and pure-Python so it can be exercised
in CI. Given a ranker callable that maps ``(customer_id, feature_set)``
to a ranked article list, and a ground-truth label mapping, it reports:

* baseline MAP@K with the canonical feature set;
* MAP@K after removing each feature one at a time (leave-one-out);
* MAP@K after adding each candidate feature on top of the baseline.

These three signals together pin down whether a new ranker input (e.g.
``two_tower_score``) is genuinely contributing to the lift target
specified in the SDD plan, rather than merely correlating with an
existing feature.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from hm_recsys.evaluation.metrics import average_precision_at_k

RankerFn = Callable[[str, frozenset[str]], Sequence[str]]


@dataclass(frozen=True)
class AblationOutcome:
    """One ablation comparison vs. the baseline feature set."""

    variant_name: str
    feature_set: tuple[str, ...]
    map_at_k: float
    delta_vs_baseline: float


@dataclass(frozen=True)
class AblationReport:
    """Container for a single ablation experiment."""

    k: int
    baseline_feature_names: tuple[str, ...]
    baseline_map_at_k: float
    leave_one_out: tuple[AblationOutcome, ...]
    add_one_in: tuple[AblationOutcome, ...]


def _mean_map_at_k(
    ranker: RankerFn,
    feature_set: frozenset[str],
    labels_by_customer: Mapping[str, Sequence[str]],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not labels_by_customer:
        return 0.0
    total = 0.0
    evaluated = 0
    for customer_id, labels in labels_by_customer.items():
        if not labels:
            continue
        predictions = ranker(customer_id, feature_set)
        total += average_precision_at_k(predictions, tuple(labels), k)
        evaluated += 1
    if evaluated == 0:
        return 0.0
    return total / evaluated


def run_ablation(
    *,
    ranker: RankerFn,
    baseline_features: Iterable[str],
    candidate_features_to_add: Iterable[str] = (),
    labels_by_customer: Mapping[str, Sequence[str]],
    k: int = 12,
) -> AblationReport:
    """Run a leave-one-out and add-one-in ablation experiment.

    Args:
        ranker: Callable that takes a customer ID and the active feature
            set (as a ``frozenset``) and returns its ranked predictions.
        baseline_features: Names of features active in the baseline ranker.
        candidate_features_to_add: Names of features that are *not* in the
            baseline but should be evaluated as additions.
        labels_by_customer: Held-out ground-truth labels for evaluation.
        k: MAP cut-off depth.

    Returns:
        Structured :class:`AblationReport` for downstream analysis.
    """

    baseline_tuple = tuple(dict.fromkeys(baseline_features))
    baseline_set = frozenset(baseline_tuple)
    baseline_map = _mean_map_at_k(ranker, baseline_set, labels_by_customer, k)

    leave_one_out: list[AblationOutcome] = []
    for feature in baseline_tuple:
        reduced = baseline_set - {feature}
        score = _mean_map_at_k(ranker, reduced, labels_by_customer, k)
        leave_one_out.append(
            AblationOutcome(
                variant_name=f"-{feature}",
                feature_set=tuple(sorted(reduced)),
                map_at_k=score,
                delta_vs_baseline=score - baseline_map,
            )
        )

    add_one_in: list[AblationOutcome] = []
    for feature in dict.fromkeys(candidate_features_to_add):
        if feature in baseline_set:
            continue
        augmented = baseline_set | {feature}
        score = _mean_map_at_k(ranker, augmented, labels_by_customer, k)
        add_one_in.append(
            AblationOutcome(
                variant_name=f"+{feature}",
                feature_set=tuple(sorted(augmented)),
                map_at_k=score,
                delta_vs_baseline=score - baseline_map,
            )
        )

    return AblationReport(
        k=k,
        baseline_feature_names=baseline_tuple,
        baseline_map_at_k=baseline_map,
        leave_one_out=tuple(leave_one_out),
        add_one_in=tuple(add_one_in),
    )


__all__ = (
    "AblationOutcome",
    "AblationReport",
    "RankerFn",
    "run_ablation",
)
