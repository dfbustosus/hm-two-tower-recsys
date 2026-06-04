import csv
import json
from pathlib import Path

import pytest

from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.deterministic_tuning import (
    DeterministicRankerTuningGrid,
    tune_deterministic_ranker_from_csv,
    write_deterministic_ranker_tuning_report,
)
from hm_recsys.retrieval.candidate_export import CANDIDATE_EXPORT_HEADER
from hm_recsys.retrieval.source_names import (
    GARMENT_GROUP_POPULARITY_SOURCE,
    RECENT_POPULARITY_SOURCE,
)

CUSTOMER_ID = "a" * 64
ARTICLE_POPULAR = "0000000001"
ARTICLE_GARMENT = "0000000002"


def test_tune_deterministic_ranker_selects_weights_on_train_window(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train_candidates.csv"
    eval_path = tmp_path / "eval_candidates.csv"
    _write_candidates(train_path)
    _write_candidates(eval_path)
    grid = DeterministicRankerTuningGrid(
        garment_group_popularity_presence_weights=(0.0, 3.0),
        garment_group_popularity_score_weights=(0.0,),
        age_segment_popularity_presence_weights=(0.0,),
        age_segment_popularity_score_weights=(0.0,),
        source_count_weights=(0.0,),
        best_rank_score_weights=(0.0,),
    )

    report = tune_deterministic_ranker_from_csv(
        train_candidate_path=train_path,
        train_validation_labels={CUSTOMER_ID: (ARTICLE_GARMENT,)},
        train_split=TemporalSplit.from_isoformat("2020-09-09"),
        evaluation_candidate_path=eval_path,
        evaluation_validation_labels={CUSTOMER_ID: (ARTICLE_GARMENT,)},
        evaluation_split=TemporalSplit.from_isoformat("2020-09-16"),
        k=1,
        candidate_k=2,
        grid=grid,
    )

    assert report.trial_count == 2
    assert report.default_evaluation.map_at_k == 0.0
    assert report.selected_evaluation.map_at_k == 1.0
    assert report.delta_selected_vs_default_map_at_k == 1.0
    assert report.selected_weights.garment_group_popularity_presence_weight == 3.0
    assert report.top_train_trials[0].train_map_at_k == 1.0


def test_write_deterministic_ranker_tuning_report(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.csv"
    _write_candidates(candidate_path)
    grid = DeterministicRankerTuningGrid(
        garment_group_popularity_presence_weights=(3.0,),
        garment_group_popularity_score_weights=(0.0,),
        age_segment_popularity_presence_weights=(0.0,),
        age_segment_popularity_score_weights=(0.0,),
        source_count_weights=(0.0,),
        best_rank_score_weights=(0.0,),
    )
    report = tune_deterministic_ranker_from_csv(
        train_candidate_path=candidate_path,
        train_validation_labels={CUSTOMER_ID: (ARTICLE_GARMENT,)},
        train_split=TemporalSplit.from_isoformat("2020-09-09"),
        evaluation_candidate_path=candidate_path,
        evaluation_validation_labels={CUSTOMER_ID: (ARTICLE_GARMENT,)},
        evaluation_split=TemporalSplit.from_isoformat("2020-09-16"),
        k=1,
        candidate_k=2,
        grid=grid,
    )

    output_path = write_deterministic_ranker_tuning_report(report, tmp_path / "report.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["selected_weights"]["garment_group_popularity_presence_weight"] == 3.0
    assert payload["selected_evaluation"]["map_at_k"] == 1.0


def test_tuning_grid_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must contain at least one value"):
        DeterministicRankerTuningGrid(garment_group_popularity_presence_weights=())
    with pytest.raises(ValueError, match="negative"):
        DeterministicRankerTuningGrid(garment_group_popularity_presence_weights=(-0.1,))


def test_tuning_rejects_overlapping_windows(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.csv"
    _write_candidates(candidate_path)

    with pytest.raises(ValueError, match="training label window"):
        tune_deterministic_ranker_from_csv(
            train_candidate_path=candidate_path,
            train_validation_labels={CUSTOMER_ID: (ARTICLE_GARMENT,)},
            train_split=TemporalSplit.from_isoformat("2020-09-16"),
            evaluation_candidate_path=candidate_path,
            evaluation_validation_labels={CUSTOMER_ID: (ARTICLE_GARMENT,)},
            evaluation_split=TemporalSplit.from_isoformat("2020-09-16"),
        )


def _write_candidates(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        writer.writerow((CUSTOMER_ID, ARTICLE_POPULAR, RECENT_POPULARITY_SOURCE, 1, 1.0))
        writer.writerow((CUSTOMER_ID, ARTICLE_GARMENT, GARMENT_GROUP_POPULARITY_SOURCE, 1, 1.0))
