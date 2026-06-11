"""Optional LightGBM ranker using source and cutoff-safe behavioral features."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from hm_recsys.evaluation.metrics import average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.behavioral import (
    BEHAVIORAL_FEATURE_NAMES,
    ArticleAttributeMap,
    BehavioralTransaction,
    CutoffBehavioralFeatures,
    build_cutoff_behavioral_features,
)
from hm_recsys.ranking.deterministic import (
    DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    CandidateFeatures,
    DeterministicRankerWeights,
    iter_candidate_records_from_csv,
    score_candidate,
)
from hm_recsys.ranking.linear import LINEAR_FEATURE_NAMES, feature_vector

LIGHTGBM_BEHAVIORAL_FEATURE_NAMES = (
    *LINEAR_FEATURE_NAMES,
    "deterministic_score",
    *BEHAVIORAL_FEATURE_NAMES,
    "two_tower_score",
    "content_user_cosine",
)
"""Feature-name tuple consumed by the LightGBM behavioral ranker.

Optional sidecar features (``two_tower_score``, ``content_user_cosine``)
are appended at the end so legacy candidate CSVs without those columns
produce identical feature vectors for every other column, keeping
booster splits backwards-compatible with checkpoints trained before
the columns existed (LightGBM saved boosters use feature names, not
positions; new features simply get zero variance during inference on
legacy CSVs).
"""


LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS: DeterministicRankerWeights = replace(
    DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    repeat_presence_weight=3.0,
    repeat_score_weight=1.0,
    recent_popularity_presence_weight=0.0,
    age_segment_popularity_presence_weight=0.45,
    age_segment_popularity_score_weight=0.1,
    garment_group_popularity_presence_weight=0.8,
    garment_group_popularity_score_weight=0.55,
    co_visitation_presence_weight=1.0,
    co_visitation_score_weight=0.0,
    source_count_weight=0.15,
    best_rank_score_weight=0.0,
    two_tower_retrieval_presence_weight=1.5,
    two_tower_retrieval_score_weight=0.0,
    two_tower_retrieval_rank_weight=0.0,
    two_tower_retrieval_latest_customer_presence_weight=1.0,
    two_tower_retrieval_latest_customer_score_weight=0.0,
    two_tower_retrieval_latest_customer_rank_weight=1.0,
)
"""Curated deterministic-prior weights used by the LightGBM behavioral CLI.

These are NOT the bare ``DEFAULT_DETERMINISTIC_RANKER_WEIGHTS``; they were
hand-tuned against the full-validation diagnostic prior before behavioral
features were introduced. The LightGBM blend uses this prior as the
deterministic side of the per-customer z-score blend, so it is the right
baseline to compare any other ranker (CatBoost, two-tower, ensemble)
against. See ``cli/_legacy.py::_lightgbm_behavioral_ranker_weights_from_args``
for the original definition; this constant exists so the same weights can
be reused from scripts and tests without depending on the legacy CLI.
"""


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
    objective: str = "lambdarank"

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
        if self.objective not in {"lambdarank", "rank_xendcg"}:
            raise ValueError("objective must be one of {'lambdarank', 'rank_xendcg'}")

    def lightgbm_params(self) -> dict[str, Any]:
        """Return LightGBM ranking parameters.

        ``rank_xendcg`` (XE-NDCG) typically outperforms ``lambdarank`` on
        very large candidate lists because its gradient is well-behaved on
        long-tail positions; ``lambdarank`` is kept as a fallback for
        reproducing legacy reports.
        """

        return {
            "objective": self.objective,
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


@dataclass(frozen=True)
class LightGBMBehavioralTrainingSummary:
    """Training summary for a cutoff-safe LightGBM behavioral ranker."""

    train_cutoff: str
    train_validation_end_exclusive: str
    horizon_days: int
    train_candidate_path: str
    train_label_customers: int
    train_grouped_candidate_customers: int
    train_unique_candidate_pairs: int
    train_positive_pairs: int
    train_negative_pairs_sampled: int
    train_matrix_rows: int
    feature_names: tuple[str, ...]
    config: LightGBMBehavioralRankerConfig


@dataclass(frozen=True)
class LightGBMBehavioralTrainingWindow:
    """One leakage-safe training window for multi-window LightGBM training."""

    split: TemporalSplit
    candidate_path: Path | str
    validation_labels: Mapping[str, Iterable[str]]


@dataclass(frozen=True)
class LightGBMBehavioralTrainingWindowSummary:
    """Per-window training-matrix summary."""

    train_cutoff: str
    train_validation_end_exclusive: str
    train_candidate_path: str
    train_label_customers: int
    train_grouped_candidate_customers: int
    train_unique_candidate_pairs: int
    train_positive_pairs: int
    train_negative_pairs_sampled: int
    train_matrix_rows: int


def lightgbm_behavioral_feature_vector(
    candidate_features: CandidateFeatures,
    behavioral_features: CutoffBehavioralFeatures,
    weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
) -> tuple[float, ...]:
    """Return the feature vector used by the optional LightGBM ranker.

    Layout (must stay in lock-step with ``LIGHTGBM_BEHAVIORAL_FEATURE_NAMES``):

    1. ``LINEAR_FEATURE_NAMES`` columns from the deterministic-feature
       linearization;
    2. the scalar deterministic score (sum of weighted source rows);
    3. ``BEHAVIORAL_FEATURE_NAMES`` columns from the cutoff-safe
       behavioral feature builder;
    4. ``two_tower_score`` — pair-level two-tower cosine, ``0.0`` when
       absent;
    5. ``content_user_cosine`` — pair-level customer/article FashionCLIP
       cosine, ``0.0`` when absent or cold-start.

    Trailing optional features default to ``0.0``, so this function is a
    no-op for legacy CSVs and only contributes signal once a real score
    is plumbed in by their respective augmentation scripts.
    """

    return (
        *feature_vector(candidate_features),
        score_candidate(candidate_features, weights),
        *behavioral_features.vector_for(
            candidate_features.customer_id,
            candidate_features.article_id,
        ),
        float(candidate_features.two_tower_score),
        float(candidate_features.content_user_cosine),
    )


def train_lightgbm_behavioral_ranker_from_csv(
    transaction_iter_factory: Callable[[], Iterable[BehavioralTransaction]],
    train_split: TemporalSplit,
    train_candidate_path: Path | str,
    train_validation_labels: Mapping[str, Iterable[str]],
    article_attributes_by_id: ArticleAttributeMap | None = None,
    config: LightGBMBehavioralRankerConfig = DEFAULT_LIGHTGBM_BEHAVIORAL_RANKER_CONFIG,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Any, Any, LightGBMBehavioralTrainingSummary]:
    """Train the optional LightGBM behavioral ranker and return runtime objects.

    LightGBM/NumPy are returned with the model so callers can score final-data
    candidates without importing optional dependencies at module import time.
    """

    return train_lightgbm_behavioral_ranker_from_windows(
        transaction_iter_factory=transaction_iter_factory,
        training_windows=(
            LightGBMBehavioralTrainingWindow(
                split=train_split,
                candidate_path=train_candidate_path,
                validation_labels=train_validation_labels,
            ),
        ),
        article_attributes_by_id=article_attributes_by_id,
        config=config,
        progress_callback=progress_callback,
    )


def train_lightgbm_behavioral_ranker_from_windows(
    transaction_iter_factory: Callable[[], Iterable[BehavioralTransaction]],
    training_windows: Sequence[LightGBMBehavioralTrainingWindow],
    article_attributes_by_id: ArticleAttributeMap | None = None,
    config: LightGBMBehavioralRankerConfig = DEFAULT_LIGHTGBM_BEHAVIORAL_RANKER_CONFIG,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Any, Any, LightGBMBehavioralTrainingSummary]:
    """Train the optional LightGBM behavioral ranker on one or more windows."""

    if not training_windows:
        raise ValueError("at least one training window is required")

    np, lgb = _import_lightgbm_runtime()
    x_arrays: list[Any] = []
    y_arrays: list[Any] = []
    train_groups: list[int] = []
    window_summaries: list[LightGBMBehavioralTrainingWindowSummary] = []

    for window_index, window in enumerate(training_windows, start=1):
        train_labels = _labels_as_sets(window.validation_labels)
        resolved_candidate_path = Path(window.candidate_path).expanduser().resolve()
        prefix = f"training window {window_index}/{len(training_windows)} {window.split.cutoff}"

        _notify(progress_callback, f"{prefix}: loading grouped training candidates")
        train_features_by_customer = dict(
            iter_grouped_candidate_features_from_csv(
                resolved_candidate_path,
                train_labels,
            )
        )
        _notify(
            progress_callback,
            f"{prefix}: loaded grouped training candidates for "
            f"{len(train_features_by_customer)} customers",
        )
        _notify(progress_callback, f"{prefix}: building cutoff-safe behavioral features")
        train_behavioral_features = build_cutoff_behavioral_features(
            transaction_iter_factory(),
            window.split.cutoff,
            target_customer_ids=train_labels,
            article_attributes_by_id=article_attributes_by_id,
        )
        _notify(progress_callback, f"{prefix}: building LightGBM training matrix")
        matrix_progress_callback: Callable[[str], None] | None = None
        if progress_callback is not None:

            def matrix_progress_callback(message: str, prefix: str = prefix) -> None:
                progress_callback(f"{prefix}: {message}")

        x_train, y_train, groups, pair_count, positive_count = _build_train_matrix(
            np=np,
            features_by_customer=train_features_by_customer,
            validation_labels=train_labels,
            behavioral_features=train_behavioral_features,
            config=config,
            progress_callback=matrix_progress_callback,
        )
        if not groups:
            raise ValueError(
                f"training candidate file produced no grouped examples: "
                f"{resolved_candidate_path}"
            )
        x_arrays.append(x_train)
        y_arrays.append(y_train)
        train_groups.extend(groups)
        window_summaries.append(
            LightGBMBehavioralTrainingWindowSummary(
                train_cutoff=window.split.cutoff.isoformat(),
                train_validation_end_exclusive=window.split.validation_end.isoformat(),
                train_candidate_path=str(resolved_candidate_path),
                train_label_customers=len(train_labels),
                train_grouped_candidate_customers=len(train_features_by_customer),
                train_unique_candidate_pairs=pair_count,
                train_positive_pairs=positive_count,
                train_negative_pairs_sampled=len(y_train) - positive_count,
                train_matrix_rows=len(y_train),
            )
        )

    x_train_all = x_arrays[0] if len(x_arrays) == 1 else np.concatenate(x_arrays, axis=0)
    y_train_all = y_arrays[0] if len(y_arrays) == 1 else np.concatenate(y_arrays, axis=0)
    total_positive_count = sum(summary.train_positive_pairs for summary in window_summaries)
    _notify(
        progress_callback,
        f"training LightGBM on {len(y_train_all)} sampled rows, "
        f"{total_positive_count} positives, {len(train_groups)} groups, "
        f"{len(window_summaries)} windows",
    )
    train_dataset = lgb.Dataset(
        x_train_all,
        label=y_train_all,
        group=train_groups,
        feature_name=list(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES),
    )
    model = lgb.train(
        config.lightgbm_params(),
        train_dataset,
        num_boost_round=config.num_boost_round,
    )
    train_cutoffs = tuple(summary.train_cutoff for summary in window_summaries)
    train_validation_ends = tuple(
        summary.train_validation_end_exclusive for summary in window_summaries
    )
    train_candidate_paths = tuple(summary.train_candidate_path for summary in window_summaries)
    summary = LightGBMBehavioralTrainingSummary(
        train_cutoff=",".join(train_cutoffs),
        train_validation_end_exclusive=",".join(train_validation_ends),
        horizon_days=training_windows[0].split.horizon_days,
        train_candidate_path=";".join(train_candidate_paths),
        train_label_customers=sum(summary.train_label_customers for summary in window_summaries),
        train_grouped_candidate_customers=sum(
            summary.train_grouped_candidate_customers for summary in window_summaries
        ),
        train_unique_candidate_pairs=sum(
            summary.train_unique_candidate_pairs for summary in window_summaries
        ),
        train_positive_pairs=total_positive_count,
        train_negative_pairs_sampled=sum(
            summary.train_negative_pairs_sampled for summary in window_summaries
        ),
        train_matrix_rows=len(y_train_all),
        feature_names=LIGHTGBM_BEHAVIORAL_FEATURE_NAMES,
        config=config,
    )
    return np, model, summary


def evaluate_lightgbm_behavioral_ranker_from_csv(
    transaction_iter_factory: Callable[[], Iterable[BehavioralTransaction]],
    train_split: TemporalSplit,
    evaluation_split: TemporalSplit,
    train_candidate_path: Path | str,
    evaluation_candidate_path: Path | str,
    train_validation_labels: Mapping[str, Iterable[str]],
    evaluation_validation_labels: Mapping[str, Iterable[str]],
    article_attributes_by_id: ArticleAttributeMap | None = None,
    config: LightGBMBehavioralRankerConfig = DEFAULT_LIGHTGBM_BEHAVIORAL_RANKER_CONFIG,
    progress_callback: Callable[[str], None] | None = None,
) -> LightGBMBehavioralRankerReport:
    """Train and evaluate an optional LightGBM behavioral ranker.

    LightGBM and NumPy are imported lazily so the base package stays dependency
    light. All behavioral features are computed with transactions strictly
    before each split cutoff.
    """

    evaluation_labels = _labels_as_sets(evaluation_validation_labels)
    resolved_evaluation_candidate_path = Path(evaluation_candidate_path).expanduser().resolve()
    np, model, training_summary = train_lightgbm_behavioral_ranker_from_csv(
        transaction_iter_factory=transaction_iter_factory,
        train_split=train_split,
        train_candidate_path=train_candidate_path,
        train_validation_labels=train_validation_labels,
        article_attributes_by_id=article_attributes_by_id,
        config=config,
        progress_callback=progress_callback,
    )

    _notify(progress_callback, "building cutoff-safe evaluation behavioral features")
    evaluation_behavioral_features = build_cutoff_behavioral_features(
        transaction_iter_factory(),
        evaluation_split.cutoff,
        target_customer_ids=evaluation_labels,
        article_attributes_by_id=article_attributes_by_id,
    )
    _notify(progress_callback, "scoring evaluation candidates")
    metrics = _evaluate_streaming(
        np=np,
        model=model,
        candidate_path=resolved_evaluation_candidate_path,
        validation_labels=evaluation_labels,
        behavioral_features=evaluation_behavioral_features,
        config=config,
        progress_callback=progress_callback,
    )

    return LightGBMBehavioralRankerReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        train_cutoff=train_split.cutoff.isoformat(),
        train_validation_end_exclusive=train_split.validation_end.isoformat(),
        evaluation_cutoff=evaluation_split.cutoff.isoformat(),
        evaluation_end_exclusive=evaluation_split.validation_end.isoformat(),
        horizon_days=evaluation_split.horizon_days,
        train_candidate_path=training_summary.train_candidate_path,
        evaluation_candidate_path=str(resolved_evaluation_candidate_path),
        train_label_customers=training_summary.train_label_customers,
        evaluation_label_customers=len(evaluation_labels),
        evaluated_customers=metrics.evaluated_customers,
        missing_evaluation_label_customers=len(evaluation_labels) - metrics.evaluated_customers,
        train_unique_candidate_pairs=training_summary.train_unique_candidate_pairs,
        train_positive_pairs=training_summary.train_positive_pairs,
        train_negative_pairs_sampled=training_summary.train_negative_pairs_sampled,
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


@dataclass(frozen=True)
class LightGBMBehavioralRankerAdapter:
    """Concrete :class:`hm_recsys.ranking.protocol.Ranker` for LightGBM behavioral models.

    The adapter wraps a trained LightGBM booster plus precomputed cutoff-safe
    behavioral features and exposes the protocol's batch-ranking entry point.

    Attributes:
        np: The numpy module reference (injected to avoid eager imports in
            modules that do not depend on numpy).
        model: Trained LightGBM booster supporting ``model.predict``.
        behavioral_features: Cutoff-safe behavioral features keyed by customer.
        config: Ranker configuration including ``k`` and blend hyperparameters.
        name: Stable short identifier used in JSON reports. Defaults to
            ``"lightgbm_behavioral"``.
    """

    np: Any
    model: Any
    behavioral_features: CutoffBehavioralFeatures
    config: LightGBMBehavioralRankerConfig
    name: str = "lightgbm_behavioral"

    def rank_customer_batch(
        self,
        features_by_customer: Mapping[str, Mapping[str, CandidateFeatures]],
        *,
        k: int,
    ) -> Mapping[str, tuple[str, ...]]:
        """Batch-score all customers with one LightGBM prediction call."""

        if k != self.config.k:
            raise ValueError(
                f"k={k} must match the configured ranker depth config.k={self.config.k}"
            )
        grouped = tuple(features_by_customer.items())
        return rank_lightgbm_behavioral_candidate_groups(
            np=self.np,
            model=self.model,
            grouped_candidate_features=grouped,
            behavioral_features=self.behavioral_features,
            config=self.config,
        )


def rank_lightgbm_behavioral_candidate_features(
    *,
    np: Any,
    model: Any,
    candidate_features: Mapping[str, CandidateFeatures],
    behavioral_features: CutoffBehavioralFeatures,
    config: LightGBMBehavioralRankerConfig,
) -> tuple[str, ...]:
    """Rank one customer's candidates with LightGBM blended with deterministic prior."""

    values = list(candidate_features.values())
    if not values:
        return ()
    x_rows = [
        lightgbm_behavioral_feature_vector(
            features,
            behavioral_features,
            config.deterministic_weights,
        )
        for features in values
    ]
    deterministic_scores = [
        score_candidate(features, config.deterministic_weights) for features in values
    ]
    model_scores = tuple(
        float(score) for score in model.predict(np.asarray(x_rows, dtype=np.float32))
    )
    blend_scores = _per_customer_zscore_blend(
        deterministic_scores,
        model_scores,
        config.blend_lambda,
    )
    return _rank_by_scores(values, blend_scores, config.k)


def rank_lightgbm_behavioral_candidate_groups(
    *,
    np: Any,
    model: Any,
    grouped_candidate_features: Sequence[tuple[str, Mapping[str, CandidateFeatures]]],
    behavioral_features: CutoffBehavioralFeatures,
    config: LightGBMBehavioralRankerConfig,
) -> dict[str, tuple[str, ...]]:
    """Batch-rank multiple customers' candidates with one LightGBM prediction call."""

    all_values: list[tuple[str, list[CandidateFeatures]]] = []
    offsets: list[tuple[int, int]] = []
    x_rows: list[tuple[float, ...]] = []
    deterministic_scores: list[float] = []
    cursor = 0
    for customer_id, article_features in grouped_candidate_features:
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
    ranked_by_customer: dict[str, tuple[str, ...]] = {}
    for (customer_id, values), (start, end) in zip(all_values, offsets, strict=True):
        customer_deterministic_scores = deterministic_scores[start:end]
        customer_model_scores = tuple(float(score) for score in model_scores[start:end])
        customer_blend_scores = _per_customer_zscore_blend(
            customer_deterministic_scores,
            customer_model_scores,
            config.blend_lambda,
        )
        ranked_by_customer[customer_id] = _rank_by_scores(values, customer_blend_scores, config.k)
    return ranked_by_customer


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
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Any, Any, list[int], int, int]:
    x_rows: list[tuple[float, ...]] = []
    y_rows: list[int] = []
    groups: list[int] = []
    unique_pair_count = 0
    positive_pair_count = 0
    for customer_index, customer_id in enumerate(sorted(validation_labels), start=1):
        candidate_features = features_by_customer.get(customer_id, {})
        if not candidate_features:
            if customer_index % config.chunk_customers == 0:
                _notify(
                    progress_callback,
                    f"processed training customers: {customer_index}/{len(validation_labels)}",
                )
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
        if customer_index % config.chunk_customers == 0:
            _notify(
                progress_callback,
                f"processed training customers: {customer_index}/{len(validation_labels)} "
                f"rows={len(y_rows)} positives={positive_pair_count}",
            )
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
    progress_callback: Callable[[str], None] | None = None,
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
            _notify(
                progress_callback,
                f"scored evaluation customers: {evaluated_customers}/{len(validation_labels)} "
                f"pairs={unique_candidate_pairs}",
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
        _notify(
            progress_callback,
            f"scored evaluation customers: {evaluated_customers}/{len(validation_labels)} "
            f"pairs={unique_candidate_pairs}",
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


def _notify(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


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
