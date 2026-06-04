import csv
import json
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.training.two_tower_export import (
    TWO_TOWER_ARTICLE_MAPPING_HEADER,
    TWO_TOWER_CUSTOMER_MAPPING_HEADER,
    TWO_TOWER_EXAMPLE_HEADER,
)
from hm_recsys.training.two_tower_retrieval import (
    TwoTowerSmokeModel,
    TwoTowerSmokeTrainingConfig,
    build_article_popularity_score_prior,
    build_two_tower_retrieval_report,
    evaluate_two_tower_retrieval,
    rank_two_tower_candidates,
    train_two_tower_smoke_model_from_csv,
    write_two_tower_retrieval_report,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
UNMAPPED_CUSTOMER_ID = "c" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
ARTICLE_3 = "0000000003"


def test_two_tower_smoke_training_ranks_candidates_and_evaluates(tmp_path: Path) -> None:
    paths = write_two_tower_smoke_artifacts(tmp_path)

    model, summary = train_two_tower_smoke_model_from_csv(
        examples_path=paths["examples"],
        customer_mapping_path=paths["customers"],
        article_mapping_path=paths["articles"],
        config=TwoTowerSmokeTrainingConfig(
            embedding_dim=4,
            epochs=20,
            learning_rate=0.2,
            seed=7,
        ),
    )
    ranked = rank_two_tower_candidates(model, CUSTOMER_ID, k=2)
    evaluation = evaluate_two_tower_retrieval(
        model,
        validation_labels={
            CUSTOMER_ID: (ARTICLE_1,),
            SECOND_CUSTOMER_ID: (ARTICLE_2,),
            UNMAPPED_CUSTOMER_ID: (ARTICLE_3,),
        },
        k=2,
        evaluation_ks=(1, 2, 3),
    )

    assert summary.rows_read == 4
    assert summary.positive_examples == 2
    assert summary.negative_examples == 2
    assert ranked
    assert evaluation.total_labeled_customers == 3
    assert evaluation.mapped_labeled_customers == 2
    assert evaluation.evaluated_customers == 2
    assert evaluation.duplicate_prediction_rows == 0
    assert 0.0 <= evaluation.map_at_k <= 1.0
    assert 0.0 <= evaluation.recall_at_k <= 1.0
    assert evaluation.evaluation_ks == (1, 2, 3)
    assert evaluation.prediction_k == 3
    assert set(evaluation.recall_by_k) == {"1", "2", "3"}
    assert evaluation.recall_at_k == evaluation.recall_by_k["2"]
    assert evaluation.score_prior_weight == 0.0
    assert evaluation.score_prior_articles == 0


def test_two_tower_retrieval_report_writes_json(tmp_path: Path) -> None:
    paths = write_two_tower_smoke_artifacts(tmp_path)
    model, summary = train_two_tower_smoke_model_from_csv(
        paths["examples"],
        paths["customers"],
        paths["articles"],
        config=TwoTowerSmokeTrainingConfig(embedding_dim=2, epochs=1),
    )
    evaluation = evaluate_two_tower_retrieval(model, {CUSTOMER_ID: (ARTICLE_1,)}, k=1)
    report = build_two_tower_retrieval_report(
        cutoff="2020-01-08",
        validation_end_exclusive="2020-01-15",
        horizon_days=7,
        training=summary,
        evaluation=evaluation,
    )

    report_path = write_two_tower_retrieval_report(report, tmp_path / "report.json")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["training"]["rows_read"] == 4
    assert payload["evaluation"]["total_labeled_customers"] == 1
    assert payload["evaluation"]["recall_by_k"]["1"] >= 0.0


def test_two_tower_bpr_training_runs_on_anchor_negatives(tmp_path: Path) -> None:
    paths = write_two_tower_smoke_artifacts(tmp_path)

    model, summary = train_two_tower_smoke_model_from_csv(
        paths["examples"],
        paths["customers"],
        paths["articles"],
        config=TwoTowerSmokeTrainingConfig(
            embedding_dim=4,
            epochs=3,
            learning_rate=0.1,
            seed=11,
            loss="bpr",
        ),
    )

    assert summary.config.loss == "bpr"
    assert summary.final_average_loss >= 0.0
    assert rank_two_tower_candidates(model, CUSTOMER_ID, k=2)


def test_two_tower_smoke_rejects_invalid_inputs(tmp_path: Path) -> None:
    paths = write_two_tower_smoke_artifacts(tmp_path)

    with pytest.raises(ValueError, match="embedding_dim"):
        TwoTowerSmokeTrainingConfig(embedding_dim=0)
    with pytest.raises(ValueError, match="positive_recency_half_life_days"):
        TwoTowerSmokeTrainingConfig(positive_recency_half_life_days=0)
    model, _ = train_two_tower_smoke_model_from_csv(
        paths["examples"],
        paths["customers"],
        paths["articles"],
    )
    with pytest.raises(ValueError, match="k"):
        rank_two_tower_candidates(model, CUSTOMER_ID, k=0)
    with pytest.raises(ValueError, match="score_prior_weight"):
        rank_two_tower_candidates(model, CUSTOMER_ID, k=1, score_prior_weight=-0.1)
    with pytest.raises(ValueError, match="evaluation_ks"):
        evaluate_two_tower_retrieval(model, {CUSTOMER_ID: (ARTICLE_1,)}, evaluation_ks=(0,))


def test_two_tower_popularity_prior_is_cutoff_safe() -> None:
    split = TemporalSplit.from_isoformat("2020-01-10")

    prior = build_article_popularity_score_prior(
        (
            TransactionEvent(date(2020, 1, 9), CUSTOMER_ID, ARTICLE_1),
            TransactionEvent(date(2020, 1, 8), SECOND_CUSTOMER_ID, ARTICLE_1),
            TransactionEvent(date(2020, 1, 1), SECOND_CUSTOMER_ID, ARTICLE_2),
            TransactionEvent(date(2020, 1, 10), CUSTOMER_ID, ARTICLE_3),
        ),
        split,
        lookback_days=7,
    )

    assert ARTICLE_1 in prior
    assert ARTICLE_2 not in prior
    assert ARTICLE_3 not in prior
    assert prior[ARTICLE_1] == 1.0


def test_two_tower_ranking_uses_configured_score_prior() -> None:
    model = TwoTowerSmokeModel(
        customer_ids=(CUSTOMER_ID,),
        article_ids=(ARTICLE_1, ARTICLE_2, ARTICLE_3),
        customer_embeddings=((1.0, 0.0),),
        article_embeddings=((1.0, 0.0), (0.4, 0.0), (0.0, 0.0)),
        article_positive_counts=(1, 1, 1),
    )

    uncalibrated = rank_two_tower_candidates(model, CUSTOMER_ID, k=3)
    calibrated = rank_two_tower_candidates(
        model,
        CUSTOMER_ID,
        k=3,
        article_score_prior={ARTICLE_3: 1.0},
        score_prior_weight=2.0,
    )

    assert uncalibrated[0] == ARTICLE_1
    assert calibrated[0] == ARTICLE_3


def write_two_tower_smoke_artifacts(base: Path) -> dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    examples_path = base / "examples.csv"
    customer_path = base / "customers.csv"
    article_path = base / "articles.csv"
    with customer_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TWO_TOWER_CUSTOMER_MAPPING_HEADER)
        writer.writerow((0, CUSTOMER_ID))
        writer.writerow((1, SECOND_CUSTOMER_ID))
    with article_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TWO_TOWER_ARTICLE_MAPPING_HEADER)
        writer.writerow((0, ARTICLE_1))
        writer.writerow((1, ARTICLE_2))
        writer.writerow((2, ARTICLE_3))
    with examples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TWO_TOWER_EXAMPLE_HEADER)
        writer.writerow((0, 0, 1, CUSTOMER_ID, ARTICLE_1, "positive", "2020-01-01", 2, "", ""))
        writer.writerow(
            (0, 1, 0, CUSTOMER_ID, ARTICLE_2, "random_negative", "2020-01-01", 0, 1, ARTICLE_1)
        )
        writer.writerow(
            (1, 1, 1, SECOND_CUSTOMER_ID, ARTICLE_2, "positive", "2020-01-02", 1, "", "")
        )
        writer.writerow(
            (
                1,
                0,
                0,
                SECOND_CUSTOMER_ID,
                ARTICLE_1,
                "random_negative",
                "2020-01-02",
                0,
                1,
                ARTICLE_2,
            )
        )
    return {"examples": examples_path, "customers": customer_path, "articles": article_path}
