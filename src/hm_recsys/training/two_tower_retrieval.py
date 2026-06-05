"""Minimal dependency-light two-tower retrieval smoke training and evaluation."""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.metrics import mean_average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.training.two_tower_export import (
    TWO_TOWER_ARTICLE_MAPPING_HEADER,
    TWO_TOWER_CUSTOMER_MAPPING_HEADER,
    TWO_TOWER_EXAMPLE_HEADER,
)

TwoTowerSmokeLoss = Literal["logistic", "bpr"]


@dataclass(frozen=True)
class TwoTowerSmokeTrainingConfig:
    """Configuration for the lightweight two-tower smoke model.

    Attributes:
        embedding_dim: Shared customer/item embedding dimension.
        epochs: Deterministic passes over exported examples.
        learning_rate: SGD step size for logistic dot-product loss.
        l2: L2 penalty applied to updated embeddings.
        seed: Non-negative random seed for initial embeddings.
        loss: Training objective. ``logistic`` is pointwise binary loss;
            ``bpr`` is pairwise Bayesian Personalized Ranking loss over exported
            positive-anchor/negative rows.
        max_training_examples: Optional deterministic cap on example rows read.
        positive_recency_half_life_days: Optional half-life for downweighting older
            anchor dates. When set, examples closer to ``recency_reference_date``
            receive larger SGD weights.
        recency_reference_date: Optional ISO date used as the recency anchor. If
            omitted while recency weighting is enabled, the latest anchor date in
            the examples file is used.
    """

    embedding_dim: int = 16
    epochs: int = 3
    learning_rate: float = 0.05
    l2: float = 0.0
    seed: int = 42
    loss: TwoTowerSmokeLoss = "logistic"
    max_training_examples: int | None = None
    positive_recency_half_life_days: float | None = None
    recency_reference_date: str | None = None

    def __post_init__(self) -> None:
        """Validate training hyperparameters."""

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.loss not in {"logistic", "bpr"}:
            raise ValueError(f"unsupported loss: {self.loss!r}")
        if self.max_training_examples is not None and self.max_training_examples <= 0:
            raise ValueError("max_training_examples must be positive when provided")
        if (
            self.positive_recency_half_life_days is not None
            and self.positive_recency_half_life_days <= 0.0
        ):
            raise ValueError("positive_recency_half_life_days must be positive when provided")
        if self.recency_reference_date is not None:
            _parse_iso_date(self.recency_reference_date, field_name="recency_reference_date")


@dataclass(frozen=True)
class TwoTowerTrainingExample:
    """One exported two-tower training example row."""

    customer_index: int
    article_index: int
    label: int
    positive_count: int
    anchor_t_dat: date
    positive_anchor_article_index: int | None = None


@dataclass(frozen=True)
class TwoTowerSmokeModel:
    """Trained lightweight two-tower embedding model."""

    customer_ids: tuple[str, ...]
    article_ids: tuple[str, ...]
    customer_embeddings: tuple[tuple[float, ...], ...]
    article_embeddings: tuple[tuple[float, ...], ...]
    article_positive_counts: tuple[int, ...]

    @property
    def customer_id_to_index(self) -> dict[str, int]:
        """Return customer ID to embedding-row mapping."""

        return {customer_id: index for index, customer_id in enumerate(self.customer_ids)}


@dataclass(frozen=True)
class TwoTowerSmokeTrainingSummary:
    """Summary of a lightweight two-tower training run."""

    examples_path: str
    customer_mapping_path: str
    article_mapping_path: str
    rows_read: int
    positive_examples: int
    negative_examples: int
    unique_customers: int
    unique_articles: int
    config: TwoTowerSmokeTrainingConfig
    final_average_loss: float
    runtime_seconds: float


@dataclass(frozen=True)
class TwoTowerRetrievalEvaluation:
    """Offline retrieval metrics for a two-tower candidate source."""

    total_labeled_customers: int
    mapped_labeled_customers: int
    evaluated_customers: int
    k: int
    evaluation_ks: tuple[int, ...]
    prediction_k: int
    article_pool_size: int
    unique_label_articles: int
    label_articles_in_pool: int
    label_article_pool_coverage: float
    score_prior_weight: float
    score_prior_articles: int
    map_at_k: float
    recall_at_k: float
    recall_by_k: dict[str, float]
    duplicate_prediction_rows: int
    sample_predictions: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class TwoTowerRetrievalReport:
    """Train/evaluate report for the lightweight two-tower retrieval smoke."""

    generated_at_utc: str
    cutoff: str
    validation_end_exclusive: str
    horizon_days: int
    training: TwoTowerSmokeTrainingSummary
    evaluation: TwoTowerRetrievalEvaluation


def train_two_tower_smoke_model_from_csv(
    examples_path: Path | str,
    customer_mapping_path: Path | str,
    article_mapping_path: Path | str,
    config: TwoTowerSmokeTrainingConfig | None = None,
) -> tuple[TwoTowerSmokeModel, TwoTowerSmokeTrainingSummary]:
    """Train a small two-tower dot-product model from exported examples.

    This is deliberately dependency-light and correctness-first. It provides a
    measurable retrieval challenger before adding PyTorch, ANN indexes, or richer
    tower features.
    """

    training_config = config or TwoTowerSmokeTrainingConfig()
    started_at = perf_counter()
    resolved_examples_path = Path(examples_path).expanduser().resolve()
    resolved_customer_mapping_path = Path(customer_mapping_path).expanduser().resolve()
    resolved_article_mapping_path = Path(article_mapping_path).expanduser().resolve()
    customer_ids = _read_mapping(
        resolved_customer_mapping_path,
        expected_header=TWO_TOWER_CUSTOMER_MAPPING_HEADER,
    )
    article_ids = _read_mapping(
        resolved_article_mapping_path,
        expected_header=TWO_TOWER_ARTICLE_MAPPING_HEADER,
    )
    examples = _read_training_examples(
        resolved_examples_path,
        article_id_to_index={article_id: index for index, article_id in enumerate(article_ids)},
        max_training_examples=training_config.max_training_examples,
    )
    if not examples:
        raise ValueError("examples_path must contain at least one data row")

    _validate_example_indices(
        examples,
        customer_count=len(customer_ids),
        article_count=len(article_ids),
    )
    rng = random.Random(training_config.seed)
    scale = 1.0 / math.sqrt(training_config.embedding_dim)
    customer_embeddings = [
        [rng.uniform(-scale, scale) for _ in range(training_config.embedding_dim)]
        for _ in customer_ids
    ]
    article_embeddings = [
        [rng.uniform(-scale, scale) for _ in range(training_config.embedding_dim)]
        for _ in article_ids
    ]
    article_positive_counts = [0] * len(article_ids)
    for example in examples:
        if example.label == 1:
            article_positive_counts[example.article_index] += max(1, example.positive_count)

    recency_reference_date = _resolve_recency_reference_date(training_config, examples)
    final_loss = _train_embedding_tables(
        examples=examples,
        customer_embeddings=customer_embeddings,
        article_embeddings=article_embeddings,
        config=training_config,
        recency_reference_date=recency_reference_date,
    )

    positive_examples = sum(1 for example in examples if example.label == 1)
    negative_examples = sum(1 for example in examples if example.label == 0)
    model = TwoTowerSmokeModel(
        customer_ids=customer_ids,
        article_ids=article_ids,
        customer_embeddings=tuple(tuple(vector) for vector in customer_embeddings),
        article_embeddings=tuple(tuple(vector) for vector in article_embeddings),
        article_positive_counts=tuple(article_positive_counts),
    )
    summary = TwoTowerSmokeTrainingSummary(
        examples_path=str(resolved_examples_path),
        customer_mapping_path=str(resolved_customer_mapping_path),
        article_mapping_path=str(resolved_article_mapping_path),
        rows_read=len(examples),
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        unique_customers=len(customer_ids),
        unique_articles=len(article_ids),
        config=training_config,
        final_average_loss=final_loss,
        runtime_seconds=perf_counter() - started_at,
    )
    return model, summary


def rank_two_tower_candidates(
    model: TwoTowerSmokeModel,
    customer_id: str,
    k: int,
    max_retrieval_articles: int | None = None,
    article_score_prior: Mapping[str, float] | None = None,
    score_prior_weight: float = 0.0,
) -> tuple[str, ...]:
    """Rank article IDs for one customer using two-tower dot-product scores."""

    return tuple(
        article_id
        for article_id, _ in score_two_tower_candidates(
            model,
            customer_id,
            k=k,
            max_retrieval_articles=max_retrieval_articles,
            article_score_prior=article_score_prior,
            score_prior_weight=score_prior_weight,
        )
    )


def score_two_tower_candidates(
    model: TwoTowerSmokeModel,
    customer_id: str,
    k: int,
    max_retrieval_articles: int | None = None,
    article_score_prior: Mapping[str, float] | None = None,
    score_prior_weight: float = 0.0,
) -> tuple[tuple[str, float], ...]:
    """Rank article IDs with their two-tower retrieval scores for one customer."""

    if k <= 0:
        raise ValueError("k must be positive")
    if max_retrieval_articles is not None and max_retrieval_articles <= 0:
        raise ValueError("max_retrieval_articles must be positive when provided")
    if score_prior_weight < 0.0:
        raise ValueError("score_prior_weight must be non-negative")
    customer_index = model.customer_id_to_index.get(customer_id)
    if customer_index is None:
        return ()
    user_vector = model.customer_embeddings[customer_index]
    resolved_prior = article_score_prior if score_prior_weight > 0.0 else None
    article_indices = _candidate_article_indices(
        model,
        max_retrieval_articles,
        article_score_prior=resolved_prior,
    )
    ranked = sorted(
        (
            (
                model.article_ids[article_index],
                _dot(user_vector, model.article_embeddings[article_index])
                + score_prior_weight
                * _article_prior_score(
                    model.article_ids[article_index],
                    resolved_prior,
                ),
            )
            for article_index in article_indices
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(ranked[:k])


def evaluate_two_tower_retrieval(
    model: TwoTowerSmokeModel,
    validation_labels: Mapping[str, Iterable[str]],
    *,
    k: int = 12,
    evaluation_ks: Sequence[int] = (12, 50, 100),
    max_eval_customers: int | None = None,
    max_retrieval_articles: int | None = 5000,
    article_score_prior: Mapping[str, float] | None = None,
    score_prior_weight: float = 0.0,
    sample_prediction_count: int = 5,
) -> TwoTowerRetrievalEvaluation:
    """Evaluate two-tower retrieval against validation labels.

    ``k`` remains the MAP cutoff used for the H&M ranking objective. The
    separate ``evaluation_ks`` sequence controls candidate-generation recall
    diagnostics such as Recall@50 and Recall@100.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    metric_ks = _normalize_evaluation_ks((*evaluation_ks, k))
    prediction_k = max(metric_ks)
    if max_eval_customers is not None and max_eval_customers <= 0:
        raise ValueError("max_eval_customers must be positive when provided")
    if max_retrieval_articles is not None and max_retrieval_articles <= 0:
        raise ValueError("max_retrieval_articles must be positive when provided")
    if score_prior_weight < 0.0:
        raise ValueError("score_prior_weight must be non-negative")
    if sample_prediction_count < 0:
        raise ValueError("sample_prediction_count must be non-negative")
    resolved_prior = article_score_prior if score_prior_weight > 0.0 else None

    customer_id_to_index = model.customer_id_to_index
    mapped_customers = tuple(
        sorted(
            customer_id for customer_id in validation_labels if customer_id in customer_id_to_index
        )
    )
    evaluated_customers = (
        mapped_customers[:max_eval_customers] if max_eval_customers else mapped_customers
    )
    labels_for_eval = {
        customer_id: tuple(validation_labels[customer_id]) for customer_id in evaluated_customers
    }
    predictions = {
        customer_id: rank_two_tower_candidates(
            model,
            customer_id,
            k=prediction_k,
            max_retrieval_articles=max_retrieval_articles,
            article_score_prior=resolved_prior,
            score_prior_weight=score_prior_weight,
        )
        for customer_id in evaluated_customers
    }
    article_pool = {
        model.article_ids[index]
        for index in _candidate_article_indices(
            model,
            max_retrieval_articles,
            article_score_prior=resolved_prior,
        )
    }
    unique_label_articles = {
        article_id for labels in labels_for_eval.values() for article_id in labels
    }
    label_articles_in_pool = unique_label_articles & article_pool
    duplicate_prediction_rows = sum(
        1 for article_ids in predictions.values() if len(set(article_ids)) != len(article_ids)
    )
    recall_by_k = {
        str(metric_k): _mean(
            recall_at_k(labels, predictions.get(customer_id, ()), k=metric_k)
            for customer_id, labels in labels_for_eval.items()
        )
        for metric_k in metric_ks
    }
    return TwoTowerRetrievalEvaluation(
        total_labeled_customers=len(validation_labels),
        mapped_labeled_customers=len(mapped_customers),
        evaluated_customers=len(evaluated_customers),
        k=k,
        evaluation_ks=metric_ks,
        prediction_k=prediction_k,
        article_pool_size=len(article_pool),
        unique_label_articles=len(unique_label_articles),
        label_articles_in_pool=len(label_articles_in_pool),
        label_article_pool_coverage=(
            len(label_articles_in_pool) / len(unique_label_articles)
            if unique_label_articles
            else 0.0
        ),
        score_prior_weight=score_prior_weight,
        score_prior_articles=len(resolved_prior or {}),
        map_at_k=mean_average_precision_at_k(labels_for_eval, predictions, k=k),
        recall_at_k=recall_by_k[str(k)],
        recall_by_k=recall_by_k,
        duplicate_prediction_rows=duplicate_prediction_rows,
        sample_predictions=dict(tuple(predictions.items())[:sample_prediction_count]),
    )


def build_two_tower_retrieval_report(
    cutoff: str,
    validation_end_exclusive: str,
    horizon_days: int,
    training: TwoTowerSmokeTrainingSummary,
    evaluation: TwoTowerRetrievalEvaluation,
) -> TwoTowerRetrievalReport:
    """Build a train/evaluate report for the two-tower smoke model."""

    return TwoTowerRetrievalReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoff=cutoff,
        validation_end_exclusive=validation_end_exclusive,
        horizon_days=horizon_days,
        training=training,
        evaluation=evaluation,
    )


def build_article_popularity_score_prior(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    lookback_days: int = 7,
) -> dict[str, float]:
    """Build a normalized recent-popularity article score prior.

    Only rows with ``split.cutoff - lookback_days <= t_dat < split.cutoff`` are
    used, so the prior is safe for next-week validation.
    """

    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    start_date = split.cutoff - timedelta(days=lookback_days)
    counts: dict[str, int] = {}
    for transaction in transactions:
        if start_date <= transaction.t_dat < split.cutoff:
            counts[transaction.article_id] = counts.get(transaction.article_id, 0) + 1
    if not counts:
        return {}
    max_score = max(math.log1p(count) for count in counts.values())
    return {
        article_id: math.log1p(count) / max_score
        for article_id, count in counts.items()
        if max_score > 0.0
    }


def two_tower_retrieval_report_to_dict(report: TwoTowerRetrievalReport) -> dict[str, Any]:
    """Convert a two-tower retrieval report to JSON-serializable primitives."""

    return asdict(report)


def write_two_tower_retrieval_report(report: TwoTowerRetrievalReport, path: Path | str) -> Path:
    """Write a two-tower retrieval report as deterministic JSON."""

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(two_tower_retrieval_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _read_mapping(path: Path, expected_header: tuple[str, str]) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, ()))
        if header != expected_header:
            raise ValueError(f"mapping header must be exactly {','.join(expected_header)}")
        ids_by_index: dict[int, str] = {}
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"line {line_number}: invalid mapping row")
            try:
                index = int(row[0])
            except ValueError as exc:
                raise ValueError(f"line {line_number}: invalid mapping index") from exc
            ids_by_index[index] = row[1]
    if not ids_by_index:
        return ()
    expected_indices = tuple(range(max(ids_by_index) + 1))
    if tuple(sorted(ids_by_index)) != expected_indices:
        raise ValueError("mapping indices must be contiguous from zero")
    return tuple(ids_by_index[index] for index in expected_indices)


def _read_training_examples(
    path: Path,
    article_id_to_index: Mapping[str, int],
    max_training_examples: int | None,
) -> tuple[TwoTowerTrainingExample, ...]:
    examples: list[TwoTowerTrainingExample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TWO_TOWER_EXAMPLE_HEADER:
            raise ValueError(
                f"examples header must be exactly {','.join(TWO_TOWER_EXAMPLE_HEADER)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if max_training_examples is not None and len(examples) >= max_training_examples:
                break
            try:
                label = int(row["label"])
                if label not in {0, 1}:
                    raise ValueError
                examples.append(
                    TwoTowerTrainingExample(
                        customer_index=int(row["customer_index"]),
                        article_index=int(row["article_index"]),
                        label=label,
                        positive_count=int(row["positive_count"]),
                        anchor_t_dat=_parse_iso_date(
                            row["anchor_t_dat"],
                            field_name="anchor_t_dat",
                        ),
                        positive_anchor_article_index=_positive_anchor_index(
                            row.get("positive_article_id_anchor", ""),
                            article_id_to_index,
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: invalid two-tower example row") from exc
    return tuple(examples)


def _positive_anchor_index(
    article_id: str,
    article_id_to_index: Mapping[str, int],
) -> int | None:
    if not article_id:
        return None
    try:
        return article_id_to_index[article_id]
    except KeyError as exc:
        raise ValueError("positive_article_id_anchor is outside article mapping") from exc


def _validate_example_indices(
    examples: Iterable[TwoTowerTrainingExample],
    customer_count: int,
    article_count: int,
) -> None:
    for example in examples:
        if example.customer_index < 0 or example.customer_index >= customer_count:
            raise ValueError("example customer_index is outside customer mapping")
        if example.article_index < 0 or example.article_index >= article_count:
            raise ValueError("example article_index is outside article mapping")
        if example.positive_anchor_article_index is not None and (
            example.positive_anchor_article_index < 0
            or example.positive_anchor_article_index >= article_count
        ):
            raise ValueError("example positive_anchor_article_index is outside article mapping")


def _train_embedding_tables(
    examples: Sequence[TwoTowerTrainingExample],
    customer_embeddings: list[list[float]],
    article_embeddings: list[list[float]],
    config: TwoTowerSmokeTrainingConfig,
    recency_reference_date: date | None,
) -> float:
    if config.loss == "bpr":
        return _train_embedding_tables_bpr(
            examples,
            customer_embeddings,
            article_embeddings,
            config,
            recency_reference_date,
        )
    return _train_embedding_tables_logistic(
        examples,
        customer_embeddings,
        article_embeddings,
        config,
        recency_reference_date,
    )


def _train_embedding_tables_logistic(
    examples: Sequence[TwoTowerTrainingExample],
    customer_embeddings: list[list[float]],
    article_embeddings: list[list[float]],
    config: TwoTowerSmokeTrainingConfig,
    recency_reference_date: date | None,
) -> float:
    final_loss = 0.0
    for _ in range(config.epochs):
        total_loss = 0.0
        for example in examples:
            user_vector = customer_embeddings[example.customer_index]
            item_vector = article_embeddings[example.article_index]
            score = _dot(user_vector, item_vector)
            probability = _sigmoid(score)
            weight = _example_training_weight(example, config, recency_reference_date)
            error = (probability - example.label) * weight
            old_user = tuple(user_vector)
            old_item = tuple(item_vector)
            for dimension in range(config.embedding_dim):
                user_vector[dimension] -= config.learning_rate * (
                    error * old_item[dimension] + config.l2 * old_user[dimension]
                )
                item_vector[dimension] -= config.learning_rate * (
                    error * old_user[dimension] + config.l2 * old_item[dimension]
                )
            total_loss += _logistic_loss(example.label, probability) * weight
        final_loss = total_loss / len(examples)
    return final_loss


def _train_embedding_tables_bpr(
    examples: Sequence[TwoTowerTrainingExample],
    customer_embeddings: list[list[float]],
    article_embeddings: list[list[float]],
    config: TwoTowerSmokeTrainingConfig,
    recency_reference_date: date | None,
) -> float:
    pair_examples = tuple(
        example
        for example in examples
        if example.label == 0 and example.positive_anchor_article_index is not None
    )
    if not pair_examples:
        raise ValueError("bpr loss requires negative examples with positive anchors")

    final_loss = 0.0
    for _ in range(config.epochs):
        total_loss = 0.0
        for example in pair_examples:
            user_vector = customer_embeddings[example.customer_index]
            positive_vector = article_embeddings[example.positive_anchor_article_index or 0]
            negative_vector = article_embeddings[example.article_index]
            score_diff = _dot(user_vector, positive_vector) - _dot(user_vector, negative_vector)
            probability = _sigmoid(score_diff)
            weight = _example_training_weight(example, config, recency_reference_date)
            error = (probability - 1.0) * weight
            old_user = tuple(user_vector)
            old_positive = tuple(positive_vector)
            old_negative = tuple(negative_vector)
            for dimension in range(config.embedding_dim):
                user_gradient = error * (old_positive[dimension] - old_negative[dimension])
                positive_gradient = error * old_user[dimension]
                negative_gradient = -error * old_user[dimension]
                user_vector[dimension] -= config.learning_rate * (
                    user_gradient + config.l2 * old_user[dimension]
                )
                positive_vector[dimension] -= config.learning_rate * (
                    positive_gradient + config.l2 * old_positive[dimension]
                )
                negative_vector[dimension] -= config.learning_rate * (
                    negative_gradient + config.l2 * old_negative[dimension]
                )
            total_loss += _logistic_loss(1, probability) * weight
        final_loss = total_loss / len(pair_examples)
    return final_loss


def _candidate_article_indices(
    model: TwoTowerSmokeModel,
    max_retrieval_articles: int | None,
    article_score_prior: Mapping[str, float] | None = None,
) -> tuple[int, ...]:
    ranked_indices = sorted(
        range(len(model.article_ids)),
        key=lambda index: (
            -_article_prior_score(model.article_ids[index], article_score_prior),
            -model.article_positive_counts[index],
            model.article_ids[index],
        ),
    )
    if max_retrieval_articles is not None:
        return tuple(ranked_indices[:max_retrieval_articles])
    return tuple(ranked_indices)


def _article_prior_score(
    article_id: str,
    article_score_prior: Mapping[str, float] | None,
) -> float:
    return article_score_prior.get(article_id, 0.0) if article_score_prior is not None else 0.0


def _resolve_recency_reference_date(
    config: TwoTowerSmokeTrainingConfig,
    examples: Sequence[TwoTowerTrainingExample],
) -> date | None:
    if config.positive_recency_half_life_days is None:
        return None
    if config.recency_reference_date is not None:
        return _parse_iso_date(
            config.recency_reference_date,
            field_name="recency_reference_date",
        )
    return max(example.anchor_t_dat for example in examples)


def _example_training_weight(
    example: TwoTowerTrainingExample,
    config: TwoTowerSmokeTrainingConfig,
    recency_reference_date: date | None,
) -> float:
    weight = 1.0 + math.log1p(max(example.positive_count - 1, 0)) if example.label else 1.0
    if config.positive_recency_half_life_days is None or recency_reference_date is None:
        return weight
    age_days = max((recency_reference_date - example.anchor_t_dat).days, 0)
    recency_multiplier = 0.5 ** (age_days / config.positive_recency_half_life_days)
    return float(weight * recency_multiplier)


def _parse_iso_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc


def _normalize_evaluation_ks(evaluation_ks: Sequence[int]) -> tuple[int, ...]:
    if not evaluation_ks:
        raise ValueError("evaluation_ks must contain at least one cutoff")
    normalized = tuple(sorted(set(evaluation_ks)))
    if any(k <= 0 for k in normalized):
        raise ValueError("evaluation_ks values must be positive")
    return normalized


def _mean(values: Iterable[float]) -> float:
    observed = tuple(values)
    return sum(observed) / len(observed) if observed else 0.0


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logistic_loss(label: int, probability: float) -> float:
    epsilon = 1e-12
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    return -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))
