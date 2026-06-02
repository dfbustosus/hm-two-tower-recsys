import csv
from pathlib import Path

import pytest

from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.linear import (
    LearnedLinearRankerReport,
    LinearRankerConfig,
    build_learned_linear_ranker_report,
    evaluate_linear_ranker_from_csv,
    train_linear_ranker_from_csv,
)
from hm_recsys.ranking.rolling import (
    build_rolling_ranker_validation_report,
    learned_report_to_rolling_window,
    write_rolling_ranker_validation_report,
)
from hm_recsys.retrieval.candidate_export import CANDIDATE_EXPORT_HEADER
from hm_recsys.retrieval.source_names import RECENT_POPULARITY_SOURCE, REPEAT_SOURCE

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"


def test_build_rolling_ranker_validation_report_aggregates_windows(tmp_path: Path) -> None:
    first_report = build_learned_report(
        tmp_path=tmp_path,
        train_cutoff="2020-01-01",
        evaluation_cutoff="2020-01-08",
    )
    second_report = build_learned_report(
        tmp_path=tmp_path,
        train_cutoff="2020-01-08",
        evaluation_cutoff="2020-01-15",
    )

    report = build_rolling_ranker_validation_report(
        (first_report, second_report),
        max_target_customers=2,
        candidate_summary_paths=("artifacts/candidate-exports/example.json",),
    )

    assert report.cutoffs == ("2020-01-08", "2020-01-15")
    assert report.window_count == 2
    assert report.max_target_customers == 2
    assert report.candidate_summary_paths == ("artifacts/candidate-exports/example.json",)
    assert report.aggregate.mean_learned_map_at_k == pytest.approx(
        (first_report.evaluation.map_at_k + second_report.evaluation.map_at_k) / 2
    )
    assert report.aggregate.windows_improved_vs_source_order == 2
    assert report.aggregate.windows_improved_vs_deterministic >= 0
    assert report.windows[0].training_positive_pairs == 2
    assert "bias" in report.windows[0].model_weights

    report_path = write_rolling_ranker_validation_report(report, tmp_path / "rolling.json")

    assert report_path.exists()
    assert '"aggregate"' in report_path.read_text(encoding="utf-8")


def test_learned_report_to_rolling_window_exposes_key_metrics(tmp_path: Path) -> None:
    learned_report = build_learned_report(
        tmp_path=tmp_path,
        train_cutoff="2020-01-01",
        evaluation_cutoff="2020-01-08",
    )

    window = learned_report_to_rolling_window(learned_report)

    assert window.train_validation_end_exclusive == "2020-01-08"
    assert window.evaluation_cutoff == "2020-01-08"
    assert window.training_unique_candidate_pairs == 4
    assert window.evaluation_unique_candidate_pairs == 4
    assert window.learned_map_at_k == pytest.approx(learned_report.evaluation.map_at_k)


def test_build_rolling_ranker_validation_report_rejects_bad_inputs(tmp_path: Path) -> None:
    first_report = build_learned_report(
        tmp_path=tmp_path,
        train_cutoff="2020-01-01",
        evaluation_cutoff="2020-01-08",
    )
    second_report = build_learned_report(
        tmp_path=tmp_path,
        train_cutoff="2020-01-08",
        evaluation_cutoff="2020-01-15",
    )

    with pytest.raises(ValueError, match="at least one"):
        build_rolling_ranker_validation_report(())

    with pytest.raises(ValueError, match="unique"):
        build_rolling_ranker_validation_report((first_report, first_report))

    with pytest.raises(ValueError, match="ascending"):
        build_rolling_ranker_validation_report((second_report, first_report))


def build_learned_report(
    tmp_path: Path,
    train_cutoff: str,
    evaluation_cutoff: str,
) -> LearnedLinearRankerReport:
    train_path = tmp_path / f"train_{train_cutoff}.csv"
    eval_path = tmp_path / f"eval_{evaluation_cutoff}.csv"
    rows = [
        (CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
        (CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_SOURCE, 1, 1.0),
        (SECOND_CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
        (SECOND_CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_SOURCE, 1, 1.0),
    ]
    write_candidate_csv(train_path, rows)
    write_candidate_csv(eval_path, rows)

    labels = {CUSTOMER_ID: (ARTICLE_2,), SECOND_CUSTOMER_ID: (ARTICLE_2,)}
    config = LinearRankerConfig(
        epochs=30,
        learning_rate=0.25,
        l2=0.0,
        positive_weight=1.0,
    )
    training = train_linear_ranker_from_csv(train_path, validation_labels=labels, config=config)
    evaluation = evaluate_linear_ranker_from_csv(
        eval_path,
        validation_labels=labels,
        model=training.model,
        k=2,
    )
    return build_learned_linear_ranker_report(
        train_split=TemporalSplit.from_isoformat(train_cutoff),
        evaluation_split=TemporalSplit.from_isoformat(evaluation_cutoff),
        k=2,
        candidate_k=2,
        config=config,
        training_result=training,
        evaluation=evaluation,
    )


def write_candidate_csv(path: Path, rows: list[tuple[str, str, str, int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        writer.writerows(rows)
