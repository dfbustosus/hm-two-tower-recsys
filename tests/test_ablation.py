"""Tests for the ablation harness."""

from __future__ import annotations

from collections.abc import Sequence

from hm_recsys.evaluation.ablation import AblationReport, run_ablation


def _toy_ranker(customer_id: str, feature_set: frozenset[str]) -> Sequence[str]:
    """Deterministic stub ranker:

    * always emits article ``A`` first;
    * adds ``B`` only when ``two_tower`` is enabled;
    * adds ``C`` only when ``segment_pop`` is enabled.
    """

    _ = customer_id  # ranker is customer-agnostic in this toy example
    ranked: list[str] = ["A"]
    if "two_tower" in feature_set:
        ranked.append("B")
    if "segment_pop" in feature_set:
        ranked.append("C")
    return ranked


def test_run_ablation_reports_baseline_and_leave_one_out_deltas() -> None:
    labels = {"u1": ("B", "C"), "u2": ("A",)}

    report = run_ablation(
        ranker=_toy_ranker,
        baseline_features=("two_tower", "segment_pop"),
        candidate_features_to_add=(),
        labels_by_customer=labels,
        k=12,
    )

    assert isinstance(report, AblationReport)
    assert report.baseline_feature_names == ("two_tower", "segment_pop")
    assert report.baseline_map_at_k > 0.0
    leave_out_names = {outcome.variant_name for outcome in report.leave_one_out}
    assert leave_out_names == {"-two_tower", "-segment_pop"}
    # Removing two_tower removes article B, dropping precision for u1.
    minus_two_tower = next(o for o in report.leave_one_out if o.variant_name == "-two_tower")
    assert minus_two_tower.delta_vs_baseline < 0


def test_run_ablation_add_one_in_increases_score_when_feature_helps() -> None:
    labels = {"u1": ("B",)}

    report = run_ablation(
        ranker=_toy_ranker,
        baseline_features=("segment_pop",),
        candidate_features_to_add=("two_tower",),
        labels_by_customer=labels,
        k=12,
    )

    assert report.baseline_map_at_k == 0.0
    plus_two_tower = next(o for o in report.add_one_in if o.variant_name == "+two_tower")
    assert plus_two_tower.delta_vs_baseline > 0


def test_run_ablation_handles_empty_labels() -> None:
    report = run_ablation(
        ranker=_toy_ranker,
        baseline_features=("two_tower",),
        candidate_features_to_add=(),
        labels_by_customer={},
        k=12,
    )

    assert report.baseline_map_at_k == 0.0
    assert report.leave_one_out[0].delta_vs_baseline == 0.0
