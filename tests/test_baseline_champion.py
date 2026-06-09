"""Unit tests for the baseline-pinning module."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hm_recsys.baselines.champion import (
    DEFAULT_BASELINE_TARGET_MAP_AT_K,
    DEFAULT_BASELINE_TARGET_TOLERANCE,
    BaselineCandidate,
    OfflineRollingMetrics,
    build_baseline_champion_report,
    compute_config_hash,
    discover_rolling_validation_candidates,
    load_baseline_champion_report,
    merge_candidates,
    pick_champion_candidate,
    render_baseline_champion_markdown,
    write_baseline_champion_markdown,
    write_baseline_champion_report,
)

CUTOFFS = ("2020-09-02", "2020-09-09", "2020-09-16")


def _offline(mean_value: float) -> OfflineRollingMetrics:
    return OfflineRollingMetrics(
        cutoffs=CUTOFFS,
        per_cutoff_map_at_k={
            "2020-09-02": mean_value - 0.001,
            "2020-09-09": mean_value,
            "2020-09-16": mean_value + 0.001,
        },
        mean_map_at_k=mean_value,
        min_map_at_k=mean_value - 0.001,
        max_map_at_k=mean_value + 0.001,
        k=12,
        candidate_k=100,
        rolling_report_path="artifacts/ranker-baselines/rolling.json",
    )


def _candidate(
    name: str,
    ranker_kind: str = "deterministic",
    offline_mean: float | None = 0.020,
    lb_score: float | None = None,
    extra_config: dict[str, object] | None = None,
) -> BaselineCandidate:
    config = {"name": name, "ranker_kind": ranker_kind, **(extra_config or {})}
    return BaselineCandidate(
        name=name,
        ranker_kind=ranker_kind,
        config_hash=compute_config_hash(config),
        config=config,
        offline_metrics=_offline(offline_mean) if offline_mean is not None else None,
        leaderboard_public_map_at_k=lb_score,
    )


def test_compute_config_hash_is_deterministic_and_sensitive() -> None:
    first = compute_config_hash({"alpha": 1, "beta": [2, 3]})
    second = compute_config_hash({"beta": [2, 3], "alpha": 1})
    different = compute_config_hash({"alpha": 1, "beta": [2, 4]})
    assert first == second
    assert first != different
    assert len(first) == 8


def test_pick_champion_prefers_lb_within_tolerance() -> None:
    candidates = (
        _candidate("a", lb_score=0.018),
        _candidate("b", lb_score=DEFAULT_BASELINE_TARGET_MAP_AT_K + 0.0005),
        _candidate("c", lb_score=0.030),
    )
    index, rationale, warnings = pick_champion_candidate(candidates)
    assert index == 1
    assert "matches target LB" in rationale
    assert warnings == ()


def test_pick_champion_falls_back_to_highest_lb_when_target_missed() -> None:
    candidates = (
        _candidate("a", lb_score=0.030),
        _candidate("b", lb_score=0.045),
    )
    index, rationale, warnings = pick_champion_candidate(candidates)
    assert index == 1
    assert "no_candidate_within_target_tolerance" in warnings
    assert "highest-LB" in rationale


def test_pick_champion_falls_back_to_highest_offline_when_no_lb() -> None:
    candidates = (
        _candidate("a", offline_mean=0.015),
        _candidate("b", offline_mean=0.020),
        _candidate("c", offline_mean=0.018),
    )
    index, rationale, warnings = pick_champion_candidate(candidates)
    assert index == 1
    assert "no_leaderboard_data_yet" in warnings
    assert "highest offline rolling mean" in rationale


def test_pick_champion_returns_none_when_no_candidates() -> None:
    index, rationale, warnings = pick_champion_candidate(())
    assert index is None
    assert warnings == ("no_candidates",)
    assert rationale == "No baseline candidates registered."


def test_pick_champion_returns_none_when_candidates_have_no_metrics() -> None:
    candidate = BaselineCandidate(
        name="empty",
        ranker_kind="deterministic",
        config_hash="abc12345",
        config={},
        offline_metrics=None,
    )
    index, rationale, warnings = pick_champion_candidate((candidate,))
    assert index is None
    assert "no_offline_or_leaderboard_data" in warnings
    assert rationale.startswith("No candidate")


def test_discover_rolling_validation_candidates_skips_missing_dir(tmp_path: Path) -> None:
    discovered = discover_rolling_validation_candidates(tmp_path / "missing")
    assert discovered == ()


def test_discover_rolling_validation_candidates_extracts_three_rankers(
    tmp_path: Path,
) -> None:
    payload = {
        "cutoffs": list(CUTOFFS),
        "k": 12,
        "candidate_k": 50,
        "include_co_visitation": True,
        "max_target_customers": 1000,
        "config": {"epochs": 3, "learning_rate": 0.01, "l2": 0.001},
        "windows": [
            {
                "evaluation_cutoff": "2020-09-02",
                "source_order_map_at_k": 0.014,
                "deterministic_map_at_k": 0.020,
                "learned_map_at_k": 0.018,
            },
            {
                "evaluation_cutoff": "2020-09-09",
                "source_order_map_at_k": 0.015,
                "deterministic_map_at_k": 0.021,
                "learned_map_at_k": 0.019,
            },
            {
                "evaluation_cutoff": "2020-09-16",
                "source_order_map_at_k": 0.013,
                "deterministic_map_at_k": 0.022,
                "learned_map_at_k": 0.017,
            },
        ],
    }
    report_path = tmp_path / "rolling_linear_ranker_a.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    candidates = discover_rolling_validation_candidates(tmp_path, cutoffs=CUTOFFS)
    assert len(candidates) == 3
    kinds = {candidate.ranker_kind for candidate in candidates}
    assert kinds == {"source_order", "deterministic", "learned_linear"}
    deterministic = next(
        candidate for candidate in candidates if candidate.ranker_kind == "deterministic"
    )
    assert deterministic.offline_metrics is not None
    assert deterministic.offline_metrics.mean_map_at_k == pytest.approx(0.021, abs=1e-6)
    assert deterministic.offline_metrics.per_cutoff_map_at_k["2020-09-16"] == pytest.approx(
        0.022, abs=1e-6
    )


def test_discover_rolling_validation_candidates_filters_by_cutoffs(tmp_path: Path) -> None:
    payload = {
        "cutoffs": ["2020-09-02", "2020-09-09"],
        "k": 12,
        "candidate_k": 12,
        "config": {},
        "windows": [
            {
                "evaluation_cutoff": "2020-09-02",
                "source_order_map_at_k": 0.014,
                "deterministic_map_at_k": 0.020,
                "learned_map_at_k": 0.018,
            },
        ],
    }
    report_path = tmp_path / "rolling_linear_ranker_b.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert discover_rolling_validation_candidates(tmp_path, cutoffs=CUTOFFS) == ()


def test_build_and_write_report_round_trips(tmp_path: Path) -> None:
    candidates = (
        _candidate("deterministic_smoke", offline_mean=0.020),
        _candidate("learned_smoke", ranker_kind="learned_linear", offline_mean=0.017),
    )
    report = build_baseline_champion_report(candidates)
    output_path = tmp_path / "champion.json"
    write_baseline_champion_report(report, output_path)
    parsed = load_baseline_champion_report(output_path)
    assert parsed.target_leaderboard_map_at_k == DEFAULT_BASELINE_TARGET_MAP_AT_K
    assert parsed.target_tolerance == DEFAULT_BASELINE_TARGET_TOLERANCE
    assert len(parsed.candidates) == 2
    assert parsed.candidates[0].offline_metrics is not None
    assert parsed.candidates[0].offline_metrics.mean_map_at_k == pytest.approx(0.020)
    assert parsed.champion_index == 0


def test_merge_candidates_preserves_user_supplied_lb_scores() -> None:
    existing = (
        replace(
            _candidate("deterministic_smoke", offline_mean=0.020),
            leaderboard_public_map_at_k=0.022,
            notes="Submitted on Kaggle on day 3",
        ),
    )
    incoming = (_candidate("deterministic_smoke", offline_mean=0.0205),)
    merged = merge_candidates(existing, incoming)
    assert len(merged) == 1
    candidate = merged[0]
    assert candidate.leaderboard_public_map_at_k == 0.022
    assert candidate.notes == "Submitted on Kaggle on day 3"
    assert candidate.offline_metrics is not None
    assert candidate.offline_metrics.mean_map_at_k == pytest.approx(0.0205)


def test_merge_candidates_adds_new_candidates_and_preserves_submissions() -> None:
    existing_candidate = replace(
        _candidate("existing", offline_mean=0.020),
        submission_paths=("submissions/existing.csv",),
    )
    incoming_candidate = replace(
        _candidate("existing", offline_mean=0.020),
        submission_paths=("submissions/existing.csv", "submissions/extra.csv"),
    )
    extra = _candidate("brand_new", offline_mean=0.018)
    merged = merge_candidates((existing_candidate,), (incoming_candidate, extra))
    assert {candidate.name for candidate in merged} == {"existing", "brand_new"}
    existing_merged = next(candidate for candidate in merged if candidate.name == "existing")
    assert existing_merged.submission_paths == (
        "submissions/existing.csv",
        "submissions/extra.csv",
    )


def test_render_baseline_champion_markdown_includes_champion_marker(tmp_path: Path) -> None:
    candidates = (
        _candidate("deterministic_smoke", offline_mean=0.020),
        _candidate("learned_smoke", offline_mean=0.017),
    )
    report = build_baseline_champion_report(candidates)
    markdown_path = tmp_path / "champion.md"
    write_baseline_champion_markdown(report, markdown_path)
    body = markdown_path.read_text(encoding="utf-8")
    assert "# Baseline Champion Pinning" in body
    assert "deterministic_smoke" in body
    assert "★" in body
    assert "## Registered candidates" in body


def test_extra_warnings_are_merged_and_deduplicated() -> None:
    candidates = (_candidate("only", offline_mean=0.020),)
    report = build_baseline_champion_report(
        candidates,
        extra_warnings=("no_rolling_reports_discovered", "no_leaderboard_data_yet"),
    )
    assert "no_rolling_reports_discovered" in report.warnings
    assert "no_leaderboard_data_yet" in report.warnings


def test_render_baseline_champion_markdown_handles_no_champion() -> None:
    candidate = BaselineCandidate(
        name="empty",
        ranker_kind="deterministic",
        config_hash="abc12345",
        config={},
        offline_metrics=None,
    )
    report = build_baseline_champion_report((candidate,))
    body = render_baseline_champion_markdown(report)
    assert "Champion: **(none selected)**" in body
