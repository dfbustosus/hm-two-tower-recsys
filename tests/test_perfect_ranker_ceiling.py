"""Unit and integration tests for the perfect-ranker ceiling oracle."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hm_recsys.evaluation.metrics import mean_average_precision_at_k
from hm_recsys.evaluation.perfect_ranker import (
    PerfectRankerCutoffInput,
    build_perfect_ranker_ceiling,
    build_perfect_ranker_ceiling_report,
    compute_perfect_ranker_ceiling,
    iter_candidates_grouped_by_customer,
    load_perfect_ranker_ceiling_report,
    oracle_average_precision_at_k,
    oracle_recall_at_k,
    render_perfect_ranker_ceiling_markdown,
    write_perfect_ranker_ceiling_markdown,
    write_perfect_ranker_ceiling_report,
)
from hm_recsys.retrieval.candidate_export import (
    CANDIDATE_EXPORT_HEADER,
    CandidateRecord,
)


def _write_candidate_csv(path: Path, rows: list[CandidateRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        for row in rows:
            writer.writerow(
                [
                    row.customer_id,
                    row.article_id,
                    row.source,
                    row.source_rank,
                    f"{row.source_score:g}",
                ]
            )


def _candidate(
    customer_id: str,
    article_id: str,
    *,
    source: str = "repeat",
    source_rank: int = 1,
    source_score: float = 1.0,
) -> CandidateRecord:
    return CandidateRecord(
        customer_id=customer_id,
        article_id=article_id,
        source=source,
        source_rank=source_rank,
        source_score=source_score,
    )


def test_oracle_average_precision_matches_brute_force_for_two_labels() -> None:
    labels = ("a", "b", "c")
    candidates = ("x", "a", "y", "b", "z")
    reachable = len(set(labels) & set(candidates))
    oracle = oracle_average_precision_at_k(
        reachable_label_count=reachable, label_count=len(labels), k=12
    )
    perfect_ranking = (
        "a",
        "b",
        *(article for article in candidates if article not in {"a", "b"}),
    )
    brute_force = mean_average_precision_at_k(
        actual_by_customer={"c1": labels},
        predicted_by_customer={"c1": perfect_ranking},
        k=12,
    )
    assert oracle == pytest.approx(brute_force)
    assert oracle == pytest.approx(2 / 3)


def test_oracle_recall_caps_at_one_when_all_labels_reachable() -> None:
    assert oracle_recall_at_k(reachable_label_count=3, label_count=3, k=12) == 1.0


def test_oracle_recall_uses_label_denominator_not_k() -> None:
    assert oracle_recall_at_k(reachable_label_count=2, label_count=20, k=12) == pytest.approx(
        2 / 20
    )


def test_oracle_zero_labels_returns_zero() -> None:
    assert oracle_average_precision_at_k(reachable_label_count=0, label_count=0, k=12) == 0.0
    assert oracle_recall_at_k(reachable_label_count=0, label_count=0, k=12) == 0.0


def test_oracle_input_validation() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        oracle_average_precision_at_k(reachable_label_count=1, label_count=1, k=0)
    with pytest.raises(ValueError, match="reachable_label_count must be non-negative"):
        oracle_average_precision_at_k(reachable_label_count=-1, label_count=1, k=1)
    with pytest.raises(ValueError, match="reachable_label_count cannot exceed label_count"):
        oracle_average_precision_at_k(reachable_label_count=3, label_count=2, k=1)
    with pytest.raises(ValueError, match="k must be positive"):
        oracle_recall_at_k(reachable_label_count=1, label_count=1, k=0)


def test_iter_candidates_grouped_by_customer_dedupes_within_customer() -> None:
    grouped = list(
        iter_candidates_grouped_by_customer(
            iter(
                [
                    _candidate("c1", "a"),
                    _candidate("c1", "b"),
                    _candidate("c1", "a"),
                    _candidate("c2", "x"),
                ]
            )
        )
    )
    assert grouped == [("c1", ("a", "b")), ("c2", ("x",))]


def test_iter_candidates_grouped_by_customer_rejects_non_contiguous_customers() -> None:
    with pytest.raises(ValueError, match="customer 'c1' reappears non-contiguously"):
        list(
            iter_candidates_grouped_by_customer(
                iter(
                    [
                        _candidate("c1", "a"),
                        _candidate("c2", "b"),
                        _candidate("c1", "c"),
                    ]
                )
            )
        )


def test_compute_perfect_ranker_ceiling_matches_brute_force(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.csv"
    _write_candidate_csv(
        candidate_path,
        [
            _candidate("c1", "a"),
            _candidate("c1", "b"),
            _candidate("c1", "x"),
            _candidate("c2", "z"),
            _candidate("c2", "y"),
            _candidate("c3", "p"),
            _candidate("c3", "q"),
            _candidate("c3", "r"),
        ],
    )
    labels = {
        "c1": ("a", "b", "c"),
        "c2": ("y",),
        "c3": ("nope",),
        "c4_no_candidates": ("anything",),
        "c5_no_labels": (),
    }
    ceiling, per_customer = compute_perfect_ranker_ceiling(
        candidate_path=candidate_path, validation_labels=labels, cutoff="2020-09-16", k=3
    )

    assert ceiling.evaluated_customers == 4
    assert ceiling.customers_in_candidates == 3
    assert ceiling.customers_without_any_candidate == 1
    expected_mean = (oracle_average_precision_at_k(2, 3, 3) + 1.0 + 0.0 + 0.0) / 4
    assert ceiling.mean_oracle_map_at_k == pytest.approx(expected_mean)
    assert ceiling.candidate_label_coverage == pytest.approx(3 / 6)
    assert ceiling.candidate_rows == 8

    c1_record = next(record for record in per_customer if record.customer_id == "c1")
    assert c1_record.label_count == 3
    assert c1_record.candidate_count == 3
    assert c1_record.reachable_label_count == 2


def test_build_perfect_ranker_ceiling_aggregates_across_cutoffs(tmp_path: Path) -> None:
    cutoff_one = tmp_path / "c1.csv"
    cutoff_two = tmp_path / "c2.csv"
    _write_candidate_csv(cutoff_one, [_candidate("u1", "a"), _candidate("u1", "b")])
    _write_candidate_csv(cutoff_two, [_candidate("u1", "c"), _candidate("u2", "d")])

    report, per_customer_by_cutoff = build_perfect_ranker_ceiling(
        inputs=(
            PerfectRankerCutoffInput(
                cutoff="2020-09-09",
                candidate_path=cutoff_one,
                validation_labels={"u1": ("a",)},
            ),
            PerfectRankerCutoffInput(
                cutoff="2020-09-16",
                candidate_path=cutoff_two,
                validation_labels={"u1": ("nope",), "u2": ("d",)},
            ),
        ),
        k=12,
    )

    assert tuple(report.cutoffs) == ("2020-09-09", "2020-09-16")
    first_cutoff = report.per_cutoff[0]
    second_cutoff = report.per_cutoff[1]
    assert first_cutoff.mean_oracle_map_at_k == pytest.approx(1.0)
    assert second_cutoff.mean_oracle_map_at_k == pytest.approx(0.5)
    assert report.mean_oracle_map_at_k == pytest.approx(0.75)
    assert report.min_oracle_map_at_k == pytest.approx(0.5)
    assert report.max_oracle_map_at_k == pytest.approx(1.0)
    assert "small_sample_evaluation_lt_100_customers" in report.warnings
    assert set(per_customer_by_cutoff.keys()) == {"2020-09-09", "2020-09-16"}


def test_build_perfect_ranker_ceiling_rejects_duplicate_cutoffs(tmp_path: Path) -> None:
    candidate_path = tmp_path / "c.csv"
    _write_candidate_csv(candidate_path, [_candidate("u1", "a")])
    bundle = PerfectRankerCutoffInput(
        cutoff="2020-09-09",
        candidate_path=candidate_path,
        validation_labels={"u1": ("a",)},
    )
    with pytest.raises(ValueError, match="cutoff inputs must have unique cutoff dates"):
        build_perfect_ranker_ceiling((bundle, bundle))


def test_build_perfect_ranker_ceiling_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="at least one cutoff input"):
        build_perfect_ranker_ceiling(())


def test_compute_perfect_ranker_ceiling_rejects_non_positive_k(tmp_path: Path) -> None:
    candidate_path = tmp_path / "c.csv"
    _write_candidate_csv(candidate_path, [_candidate("u1", "a")])
    with pytest.raises(ValueError, match="k must be positive"):
        compute_perfect_ranker_ceiling(
            candidate_path=candidate_path,
            validation_labels={"u1": ("a",)},
            cutoff="2020-09-09",
            k=0,
        )


def test_report_round_trips_json_and_markdown(tmp_path: Path) -> None:
    candidate_path = tmp_path / "c.csv"
    _write_candidate_csv(
        candidate_path,
        [_candidate("u1", "a"), _candidate("u2", "b"), _candidate("u3", "c")],
    )
    report = build_perfect_ranker_ceiling_report(
        inputs=(
            PerfectRankerCutoffInput(
                cutoff="2020-09-16",
                candidate_path=candidate_path,
                validation_labels={"u1": ("a",), "u2": ("z",), "u3": ("c",)},
            ),
        ),
        k=12,
    )
    json_path = tmp_path / "ceiling.json"
    markdown_path = tmp_path / "ceiling.md"
    write_perfect_ranker_ceiling_report(report, json_path)
    write_perfect_ranker_ceiling_markdown(report, markdown_path)
    parsed = load_perfect_ranker_ceiling_report(json_path)
    assert parsed.mean_oracle_map_at_k == pytest.approx(report.mean_oracle_map_at_k)
    assert parsed.per_cutoff[0].evaluated_customers == 3
    assert "Perfect-Ranker (Oracle) Ceiling" in markdown_path.read_text(encoding="utf-8")


def test_render_perfect_ranker_ceiling_markdown_includes_per_cutoff_paths(tmp_path: Path) -> None:
    candidate_path = tmp_path / "c.csv"
    _write_candidate_csv(candidate_path, [_candidate("u1", "a")])
    report = build_perfect_ranker_ceiling_report(
        inputs=(
            PerfectRankerCutoffInput(
                cutoff="2020-09-16",
                candidate_path=candidate_path,
                validation_labels={"u1": ("a",)},
            ),
        )
    )
    body = render_perfect_ranker_ceiling_markdown(report)
    assert "## Candidate paths" in body
    assert candidate_path.name in body
