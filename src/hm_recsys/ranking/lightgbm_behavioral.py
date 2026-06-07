"""Optional LightGBM ranker using source and cutoff-safe behavioral features."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.metrics import average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.behavioral import (
    BEHAVIORAL_FEATURE_NAMES,
    CutoffBehavioralFeatures,
    build_cutoff_behavioral_features,
)
from hm_recsys.ranking.deterministic import (
    DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    CandidateFeatures,
    DeterministicRankerWeights,
    aggregate_candidate_features,
    iter_candidate_records_from_csv,
    score_candidate,
)
from hm_recsys.ranking.linear import LINEAR_FEATURE_NAMES, feature_vector

LIGHTGBM_BEHAVIORAL_FEATURE_NAMES = (
    *LINEAR_FEATURE_NAMES,
    "deterministic_score",
    *BEHAVIORAL_FEATURE_NAMES,
)


@dataclass(frozen=True)
class LightGBMBehavioralRankerConfig:
    """Configuration for optional LightGBM behavioral-ranker evaluation."""

    k: int = 12
    negative_per_positive: int = 50
    blend_lambda: float = 0.75
    num_boost_round: int = 120
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 5.0
    seed: int = 42
    num_threads: int = 4
    chunk_customers: int = 2000
    deterministic_weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS

    def __post_init__(self) -> None:
        """Validate numeric configuration values."""

        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.negative_per_positive <= 0:
            raise ValueError("negative_per_positive must be positive")
        if self.blend_lambda < 0:
            raise ValueError("blend_lambda must be non-negative")
        if self.num_boost_round <= 0:
            raise ValueError("num_boost_round must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_leaves <= 1:
            raise ValueError("num_leaves must be greater than one")
        if self.min_data_in_leaf <= 0:
            raise ValueError("min_data_in_leaf must be positive")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction must be in (0, 1]")
        if not 0 < self.bagging_fraction <= 1:
            raise ValueError("bagging_fraction must be in (0, 1]")
        if self.bagging_freq < 0:
            raise ValueError("bagging_freq must be non-negative")
        if self.lambda_l2 < 0:
            raise ValueError("lambda_l2 must be non-negative")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.chunk_customers <= 0:
            raise ValueError("chunk_customers must be positive")

    def lightgbm_params(self) -> dict[str, Any]:
        """Return LightGBM LambdaRank parameters."""

        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [self.k],
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "verbosity": -1,
            "seed": self.seed,
            "num_threads": self.num_threads,
        }


DEFAULT_LIGHTGBM_BEHAVIORAL_RANKER_CONFIG = LightGBMBehavioralRankerConfig()


@dataclass(frozen=True)
class LightGBMBehavioralRankerReport:
    """Leakage-safe optional LightGBM behavioral-ranker evaluation report."""

    generated_at_utc: str
    train_cutoff: str
    train_validation_end_exclusive: str
    evaluation_cutoff: str
    evaluation_end_exclusive: str
    horizon_days: int
    train_candidate_path: str
    evaluation_candidate_path: str
    train_label_customers: int
    evaluation_label_customers: int
    evaluated_customers: int
    missing_evaluation_label_customers: int
    train_unique_candidate_pairs: int
    train_positive_pairs: int
    train_negative_pairs_sampled: int
    evaluation_unique_candidate_pairs: int
    deterministic_map_at_k: float
    deterministic_recall_at_k: float
    model_only_map_at_k: float
    model_only_recall_at_k: float
    blend_map_at_k: float
    blend_recall_at_k: float
    delta_vs_deterministic_map_at_k: float
    delta_vs_deterministic_recall_at_k: float
    blend_normalization: str
    feature_names: tuple[str, ...]
    config: LightGBMBehavioralRankerConfig
    valid_report: bool = True


def lightgbm_behavioral_feature_vector(
    candidate_features: CandidateFeatures,
    behavioral_features: CutoffBehavioralFeatures,
    weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
) -> tuple[float, ...]:
    """Return the feature vector used by the optional LightGBM ranker."""

    return (
        *feature_vector(candidate_features),
        score_candidate(candidate_features, weights),
        *behavioral_features.vector_for(
            candidate_features.customer_id,
            candidate_features.article_id,
        ),
    )


def evaluate_lightgbm_behavioral_ranker_from_csv(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    train_split: TemporalSplit,
    evaluation_split: TemporalSplit,
    train_candidate_path: Path | str,
    evaluation_candidate_path: Path | str,
    train_validation_labels: Mapping[str, Iterable[str]],
    evaluation_validation_labels: Mapping[str, Iterable[str]],
    config: LightGBMBehavioralRankerConfig = DEFAULT_LIGHTGBM_BEHAVIORAL_RANKER_CONFIG,
) -> LightGBMBehavioralRankerReport:
    """Train and evaluate an optional LightGBM behavioral ranker.

    LightGBM and NumPy are imported lazily so the base package stays dependency
    light. All behavioral features are computed with transactions strictly
    before each split cutoff.
    """

    np, lgb = _import_lightgbm_runtime()
    train_labels = _labels_as_sets(train_validation_labels)
    evaluation_labels = _labels_as_sets(evaluation_validation_labels)
    resolved_train_candidate_path = Path(train_candidate_path).expanduser().resolve()
    resolved_evaluation_candidate_path = Path(evaluation_candidate_path).expanduser().resolve()

    train_features_by_customer = aggregate_candidate_features(
        iter_candidate_records_from_csv(resolved_train_candidate_path),
        train_labels,
    )
    train_behavioral_features = build_cutoff_behavioral_features(
        transaction_iter_factory(),
        train_split.cutoff,
        target_customer_ids=train_labels,
    )
    x_train, y_train, train_groups, train_pair_count, train_positive_count = _build_train_matrix(
        np=np,
        features_by_customer=train_features_by_customer,
        validation_labels=train_labels,
        behavioral_features=train_behavioral_features,
        config=config,
    )
    if not train_groups:
        raise ValueError("training candidate file produced no grouped training examples")
    train_dataset = lgb.Dataset(
        x_train,
        label=y_train,
        group=train_groups,
        feature_name=list(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES),
    )
    model = lgb.train(
        config.lightgbm_params(),
        train_dataset,
        num_boost_round=config.num_boost_round,
    )

    evaluation_behavioral_features = build_cutoff_behavioral_features(
        transaction_iter_factory(),
        evaluation_split.cutoff,
        target_customer_ids=evaluation_labels,
    )
    metrics = _evaluate_streaming(
        np=np,
        model=model,
        candidate_path=resolved_evaluation_candidate_path,
        validation_labels=evaluation_labels,
        behavioral_features=evaluation_behavioral_features,
        config=config,
    )

    return LightGBMBehavioralRankerReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        train_cutoff=train_split.cutoff.isoformat(),
        train_validation_end_exclusive=train_split.validation_end.isoformat(),
        evaluation_cutoff=evaluation_split.cutoff.isoformat(),
        evaluation_end_exclusive=evaluation_split.validation_end.isoformat(),
        horizon_days=evaluation_split.horizon_days,
        train_candidate_path=str(resolved_train_candidate_path),
        evaluation_candidate_path=str(resolved_evaluation_candidate_path),
        train_label_customers=len(train_labels),
        evaluation_label_customers=len(evaluation_labels),
        evaluated_customers=metrics.evaluated_customers,
        missing_evaluation_label_customers=len(evaluation_labels) - metrics.evaluated_customers,
        train_unique_candidate_pairs=train_pair_count,
        train_positive_pairs=train_positive_count,
        train_negative_pairs_sampled=len(y_train) - train_positive_count,
        evaluation_unique_candidate_pairs=metrics.unique_candidate_pairs,
        deterministic_map_at_k=metrics.deterministic_map_at_k,
        deterministic_recall_at_k=metrics.deterministic_recall_at_k,
        model_only_map_at_k=metrics.model_only_map_at_k,
        model_only_recall_at_k=metrics.model_only_recall_at_k,
        blend_map_at_k=metrics.blend_map_at_k,
        blend_recall_at_k=metrics.blend_recall_at_k,
        delta_vs_deterministic_map_at_k=(metrics.blend_map_at_k - metrics.deterministic_map_at_k),
        delta_vs_deterministic_recall_at_k=(
            metrics.blend_recall_at_k - metrics.deterministic_recall_at_k
        ),
        blend_normalization="per_customer_zscore",
        feature_names=LIGHTGBM_BEHAVIORAL_FEATURE_NAMES,
        config=config,
    )


def iter_grouped_candidate_features_from_csv(
    candidate_path: Path | str,
    validation_labels: Mapping[str, Iterable[str]],
    max_customers: int | None = None,
) -> Iterable[tuple[str, dict[str, CandidateFeatures]]]:
    """Yield grouped candidate features by customer and reject repeated groups."""

    if max_customers is not None and max_customers <= 0:
        raise ValueError("max_customers must be positive when provided")
    label_sets = _labels_as_sets(validation_labels)
    current_customer_id: str | None = None
    current_features: dict[str, CandidateFeatures] = {}
    seen_customers: set[str] = set()
    yielded_customers = 0

    for record in iter_candidate_records_from_csv(candidate_path):
        if record.customer_id not in label_sets:
            continue
        if current_customer_id is None:
            current_customer_id = record.customer_id
        if record.customer_id != current_customer_id:
            if current_customer_id in seen_customers:
                raise _repeated_customer_block_error(current_customer_id)
            seen_customers.add(current_customer_id)
            yield current_customer_id, current_features
            yielded_customers += 1
            if max_customers is not None and yielded_customers >= max_customers:
                return
            current_customer_id = record.customer_id
            current_features = {}
            if current_customer_id in seen_customers:
                raise _repeated_customer_block_error(current_customer_id)

        features = current_features.get(record.article_id)
        if features is None:
            features = CandidateFeatures(
                customer_id=record.customer_id,
                article_id=record.article_id,
                label=int(record.article_id in label_sets.get(record.customer_id, set())),
            )
            current_features[record.article_id] = features
        features.update_from_record(record)

    if current_customer_id is not None:
        if current_customer_id in seen_customers:
            raise _repeated_customer_block_error(current_customer_id)
        yield current_customer_id, current_features


def _repeated_customer_block_error(customer_id: str) -> ValueError:
    return ValueError(f"candidate CSV contains repeated customer block: {customer_id}")


def write_lightgbm_behavioral_ranker_report(
    report: LightGBMBehavioralRankerReport,
    path: Path | str,
) -> Path:
    """Write a LightGBM behavioral-ranker report as JSON."""

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


@dataclass(frozen=True)
class _StreamingMetrics:
    evaluated_customers: int
    unique_candidate_pairs: int
    deterministic_map_at_k: float
    deterministic_recall_at_k: float
    model_only_map_at_k: float
    model_only_recall_at_k: float
    blend_map_at_k: float
    blend_recall_at_k: float


def _build_train_matrix(
    *,
    np: Any,
    features_by_customer: Mapping[str, Mapping[str, CandidateFeatures]],
    validation_labels: Mapping[str, set[str]],
    behavioral_features: CutoffBehavioralFeatures,
    config: LightGBMBehavioralRankerConfig,
) -> tuple[Any, Any, list[int], int, int]:
    x_rows: list[tuple[float, ...]] = []
    y_rows: list[int] = []
    groups: list[int] = []
    unique_pair_count = 0
    positive_pair_count = 0
    for customer_id in sorted(validation_labels):
        candidate_features = features_by_customer.get(customer_id, {})
        if not candidate_features:
            continue
        actual = validation_labels[customer_id]
        values = list(candidate_features.values())
        positives = [features for features in values if features.article_id in actual]
        positive_pair_count += len(positives)
        negative_cap = config.negative_per_positive * max(1, len(positives))
        negatives = sorted(
            (features for features in values if features.article_id not in actual),
            key=lambda features: (
                -score_candidate(features, config.deterministic_weights),
                features.article_id,
            ),
        )[:negative_cap]
        sampled_values = positives + negatives
        if not sampled_values:
            continue
        unique_pair_count += len(values)
        groups.append(len(sampled_values))
        for features in sampled_values:
            x_rows.append(
                lightgbm_behavioral_feature_vector(
                    features,
                    behavioral_features,
                    config.deterministic_weights,
                )
            )
            y_rows.append(int(features.article_id in actual))
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int32),
        groups,
        unique_pair_count,
        positive_pair_count,
    )


def _evaluate_streaming(
    *,
    np: Any,
    model: Any,
    candidate_path: Path,
    validation_labels: Mapping[str, set[str]],
    behavioral_features: CutoffBehavioralFeatures,
    config: LightGBMBehavioralRankerConfig,
) -> _StreamingMetrics:
    deterministic_ap_sum = 0.0
    deterministic_recall_sum = 0.0
    model_ap_sum = 0.0
    model_recall_sum = 0.0
    blend_ap_sum = 0.0
    blend_recall_sum = 0.0
    evaluated_customers = 0
    unique_candidate_pairs = 0
    chunk: list[tuple[str, dict[str, CandidateFeatures]]] = []
    for grouped_features in iter_grouped_candidate_features_from_csv(
        candidate_path,
        validation_labels,
    ):
        chunk.append(grouped_features)
        if len(chunk) >= config.chunk_customers:
            chunk_metrics = _score_grouped_chunk(
                np=np,
                model=model,
                chunk=chunk,
                validation_labels=validation_labels,
                behavioral_features=behavioral_features,
                config=config,
            )
            (
                deterministic_ap_sum,
                deterministic_recall_sum,
                model_ap_sum,
                model_recall_sum,
                blend_ap_sum,
                blend_recall_sum,
                evaluated_customers,
                unique_candidate_pairs,
            ) = _add_chunk_metrics(
                chunk_metrics,
                deterministic_ap_sum,
                deterministic_recall_sum,
                model_ap_sum,
                model_recall_sum,
                blend_ap_sum,
                blend_recall_sum,
                evaluated_customers,
                unique_candidate_pairs,
            )
            chunk = []
    if chunk:
        chunk_metrics = _score_grouped_chunk(
            np=np,
            model=model,
            chunk=chunk,
            validation_labels=validation_labels,
            behavioral_features=behavioral_features,
            config=config,
        )
        (
            deterministic_ap_sum,
            deterministic_recall_sum,
            model_ap_sum,
            model_recall_sum,
            blend_ap_sum,
            blend_recall_sum,
            evaluated_customers,
            unique_candidate_pairs,
        ) = _add_chunk_metrics(
            chunk_metrics,
            deterministic_ap_sum,
            deterministic_recall_sum,
            model_ap_sum,
            model_recall_sum,
            blend_ap_sum,
            blend_recall_sum,
            evaluated_customers,
            unique_candidate_pairs,
        )
    if evaluated_customers == 0:
        raise ValueError("evaluation candidate file produced no labeled customer groups")
    return _StreamingMetrics(
        evaluated_customers=evaluated_customers,
        unique_candidate_pairs=unique_candidate_pairs,
        deterministic_map_at_k=deterministic_ap_sum / evaluated_customers,
        deterministic_recall_at_k=deterministic_recall_sum / evaluated_customers,
        model_only_map_at_k=model_ap_sum / evaluated_customers,
        model_only_recall_at_k=model_recall_sum / evaluated_customers,
        blend_map_at_k=blend_ap_sum / evaluated_customers,
        blend_recall_at_k=blend_recall_sum / evaluated_customers,
    )


@dataclass(frozen=True)
class _ChunkMetrics:
    deterministic_ap_sum: float
    deterministic_recall_sum: float
    model_ap_sum: float
    model_recall_sum: float
    blend_ap_sum: float
    blend_recall_sum: float
    customers: int
    unique_candidate_pairs: int


def _score_grouped_chunk(
    *,
    np: Any,
    model: Any,
    chunk: list[tuple[str, dict[str, CandidateFeatures]]],
    validation_labels: Mapping[str, set[str]],
    behavioral_features: CutoffBehavioralFeatures,
    config: LightGBMBehavioralRankerConfig,
) -> _ChunkMetrics:
    all_values: list[tuple[str, list[CandidateFeatures]]] = []
    offsets: list[tuple[int, int]] = []
    x_rows: list[tuple[float, ...]] = []
    deterministic_scores: list[float] = []
    cursor = 0
    for customer_id, article_features in chunk:
        values = list(article_features.values())
        all_values.append((customer_id, values))
        offsets.append((cursor, cursor + len(values)))
        cursor += len(values)
        for features in values:
            x_rows.append(
                lightgbm_behavioral_feature_vector(
                    features,
                    behavioral_features,
                    config.deterministic_weights,
                )
            )
            deterministic_scores.append(score_candidate(features, config.deterministic_weights))
    model_scores = model.predict(np.asarray(x_rows, dtype=np.float32)) if x_rows else []

    deterministic_ap_sum = 0.0
    deterministic_recall_sum = 0.0
    model_ap_sum = 0.0
    model_recall_sum = 0.0
    blend_ap_sum = 0.0
    blend_recall_sum = 0.0
    unique_candidate_pairs = 0
    for (customer_id, values), (start, end) in zip(all_values, offsets, strict=True):
        actual = tuple(validation_labels[customer_id])
        customer_deterministic_scores = deterministic_scores[start:end]
        customer_model_scores = tuple(float(score) for score in model_scores[start:end])
        customer_blend_scores = _per_customer_zscore_blend(
            customer_deterministic_scores,
            customer_model_scores,
            config.blend_lambda,
        )
        deterministic_predictions = _rank_by_scores(values, customer_deterministic_scores, config.k)
        model_predictions = _rank_by_scores(values, customer_model_scores, config.k)
        blend_predictions = _rank_by_scores(values, customer_blend_scores, config.k)
        deterministic_ap_sum += average_precision_at_k(actual, deterministic_predictions, config.k)
        deterministic_recall_sum += recall_at_k(actual, deterministic_predictions, config.k)
        model_ap_sum += average_precision_at_k(actual, model_predictions, config.k)
        model_recall_sum += recall_at_k(actual, model_predictions, config.k)
        blend_ap_sum += average_precision_at_k(actual, blend_predictions, config.k)
        blend_recall_sum += recall_at_k(actual, blend_predictions, config.k)
        unique_candidate_pairs += len(values)
    return _ChunkMetrics(
        deterministic_ap_sum=deterministic_ap_sum,
        deterministic_recall_sum=deterministic_recall_sum,
        model_ap_sum=model_ap_sum,
        model_recall_sum=model_recall_sum,
        blend_ap_sum=blend_ap_sum,
        blend_recall_sum=blend_recall_sum,
        customers=len(all_values),
        unique_candidate_pairs=unique_candidate_pairs,
    )


def _add_chunk_metrics(
    chunk_metrics: _ChunkMetrics,
    deterministic_ap_sum: float,
    deterministic_recall_sum: float,
    model_ap_sum: float,
    model_recall_sum: float,
    blend_ap_sum: float,
    blend_recall_sum: float,
    evaluated_customers: int,
    unique_candidate_pairs: int,
) -> tuple[float, float, float, float, float, float, int, int]:
    return (
        deterministic_ap_sum + chunk_metrics.deterministic_ap_sum,
        deterministic_recall_sum + chunk_metrics.deterministic_recall_sum,
        model_ap_sum + chunk_metrics.model_ap_sum,
        model_recall_sum + chunk_metrics.model_recall_sum,
        blend_ap_sum + chunk_metrics.blend_ap_sum,
        blend_recall_sum + chunk_metrics.blend_recall_sum,
        evaluated_customers + chunk_metrics.customers,
        unique_candidate_pairs + chunk_metrics.unique_candidate_pairs,
    )


def _rank_by_scores(
    values: Iterable[CandidateFeatures], scores: Iterable[float], k: int
) -> tuple[str, ...]:
    return tuple(
        features.article_id
        for _, features in sorted(
            zip(scores, values, strict=True),
            key=lambda item: (-float(item[0]), item[1].article_id),
        )[:k]
    )


def _per_customer_zscore_blend(
    deterministic_scores: Iterable[float],
    model_scores: Iterable[float],
    blend_lambda: float,
) -> tuple[float, ...]:
    deterministic = tuple(float(score) for score in deterministic_scores)
    model = tuple(float(score) for score in model_scores)
    deterministic_mean, deterministic_std = _mean_and_std(deterministic)
    model_mean, model_std = _mean_and_std(model)
    return tuple(
        ((deterministic_score - deterministic_mean) / deterministic_std)
        + blend_lambda * ((model_score - model_mean) / model_std)
        for deterministic_score, model_score in zip(deterministic, model, strict=True)
    )


def _mean_and_std(values: tuple[float, ...]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance**0.5
    return mean, std if std > 1e-9 else 1.0


def _labels_as_sets(labels: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    return {customer_id: set(article_ids) for customer_id, article_ids in labels.items()}


def _import_lightgbm_runtime() -> tuple[Any, Any]:
    try:
        lgb = import_module("lightgbm")
        np = import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "The optional LightGBM behavioral ranker requires numpy and lightgbm. "
            "Install them in the local environment, e.g. `python -m pip install lightgbm`."
        ) from exc
    return np, lgb
