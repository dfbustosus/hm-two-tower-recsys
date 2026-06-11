"""Multi-seed bagged LightGBM evaluator.

Trains N LightGBM models with different seeds (and optionally
slightly different ``num_leaves`` / ``feature_fraction`` /
``bagging_fraction``) on the same candidate matrix, then averages
per-candidate predictions. The averaged model is blended with the
curated deterministic prior using the same per-customer z-score scheme
as the single-model evaluator.

Why this typically helps:

* Each LightGBM seed lands in a different local minimum of the
  pairwise-rank loss because feature/sample bagging and split tie-
  breaking are stochastic.
* Averaging across seeds reduces the variance component of the model's
  predictions without changing the bias, so the averaged ranking tends
  to be slightly sharper on held-out customers.

Reuses ``_build_train_matrix`` / ``_evaluate_streaming`` and the
``LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS`` constant so the result is
directly comparable to the single-model baselines under
``artifacts/ranker-baselines/p4-runs/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from hm_recsys.data.io import TransactionEvent, iter_csv_rows
from hm_recsys.evaluation.metrics import average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import TemporalSplit, collect_validation_labels
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.ranking.behavioral import (
    ArticleAttributeMap,
    build_cutoff_behavioral_features,
)
from hm_recsys.ranking.deterministic import CandidateFeatures, score_candidate
from hm_recsys.ranking.lightgbm_behavioral import (
    LIGHTGBM_BEHAVIORAL_FEATURE_NAMES,
    LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS,
    LightGBMBehavioralRankerConfig,
    _build_train_matrix,
    _labels_as_sets,
    _notify,
    iter_grouped_candidate_features_from_csv,
    lightgbm_behavioral_feature_vector,
)


@dataclass(frozen=True)
class BaggedLightGBMConfig:
    """Hyperparameters for the bagged-LightGBM eval."""

    k: int = 12
    negative_per_positive: int = 50
    blend_lambda: float = 0.75
    objective: str = "lambdarank"
    num_boost_round: int = 200
    base_num_leaves: int = 63
    min_data_in_leaf: int = 100
    learning_rate: float = 0.03
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 5.0
    num_threads: int = 4
    seeds: tuple[int, ...] = (42, 137, 271, 314, 415)
    leaf_jitter: tuple[int, ...] = (0, +8, -8, +16, -16)
    feature_fraction_jitter: tuple[float, ...] = (0.0, +0.05, -0.05, +0.10, -0.10)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    eval_cutoff = date.fromisoformat(args.cutoff)
    train_cutoff = date.fromisoformat(
        args.train_cutoff or (eval_cutoff - timedelta(days=args.horizon_days)).isoformat()
    )
    train_split = TemporalSplit(cutoff=train_cutoff, horizon_days=args.horizon_days)
    eval_split = TemporalSplit(cutoff=eval_cutoff, horizon_days=args.horizon_days)

    config = _config_from_args(args)
    lgbm_config = _lgbm_config_for_train_matrix(config)
    progress = _make_progress("[bagged]") if args.progress else None

    _notify(progress, "loading article metadata attributes")
    article_attributes = _maybe_load_article_attributes(paths.raw_data_dir / "articles.csv")
    _notify(progress, f"loaded article metadata for {len(article_attributes)} articles")

    _notify(progress, "loading validation labels (train + eval)")
    train_labels = collect_validation_labels(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"), train_split
    )
    eval_labels = collect_validation_labels(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"), eval_split
    )
    _notify(progress, f"labels train={len(train_labels)} eval={len(eval_labels)}")

    _notify(progress, "streaming grouped training candidates")
    train_features_by_customer = dict(
        iter_grouped_candidate_features_from_csv(
            Path(args.train_candidate_path), _labels_as_sets(train_labels)
        )
    )
    _notify(
        progress,
        f"loaded grouped training candidates for {len(train_features_by_customer)} customers",
    )

    _notify(progress, "building cutoff-safe training behavioral features")
    train_behavioral_features = build_cutoff_behavioral_features(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"),
        train_split.cutoff,
        target_customer_ids=_labels_as_sets(train_labels),
        article_attributes_by_id=article_attributes,
    )

    np, lgb = _import_lightgbm_runtime()
    _notify(progress, "building training matrix")
    x_train, y_train, groups, pair_count, positive_count = _build_train_matrix(
        np=np,
        features_by_customer=train_features_by_customer,
        validation_labels=_labels_as_sets(train_labels),
        behavioral_features=train_behavioral_features,
        config=lgbm_config,
        progress_callback=progress,
    )
    if not groups:
        raise SystemExit("training candidate file produced no grouped examples")
    _notify(
        progress,
        f"training matrix: rows={len(y_train)} positives={positive_count} groups={len(groups)} "
        f"features={x_train.shape[1]}",
    )

    seed_count = len(config.seeds)
    leaf_jitter = _pad_to(config.leaf_jitter, seed_count, fill=0)
    feature_fraction_jitter = _pad_to(config.feature_fraction_jitter, seed_count, fill=0.0)
    bag_models: list[Any] = []
    seed_train_seconds: list[float] = []
    for index, seed in enumerate(config.seeds):
        seed_leaves = max(8, config.base_num_leaves + leaf_jitter[index])
        seed_ff = max(0.5, min(1.0, config.feature_fraction + feature_fraction_jitter[index]))
        params = {
            "objective": config.objective,
            "metric": "ndcg",
            "ndcg_eval_at": [config.k],
            "learning_rate": config.learning_rate,
            "num_leaves": seed_leaves,
            "min_data_in_leaf": config.min_data_in_leaf,
            "feature_fraction": seed_ff,
            "bagging_fraction": config.bagging_fraction,
            "bagging_freq": config.bagging_freq,
            "lambda_l2": config.lambda_l2,
            "verbosity": -1,
            "seed": seed,
            "num_threads": config.num_threads,
        }
        _notify(
            progress,
            f"training bag {index + 1}/{seed_count} seed={seed} leaves={seed_leaves} "
            f"ff={seed_ff:.2f}",
        )
        start = time.perf_counter()
        train_dataset = lgb.Dataset(
            x_train,
            label=y_train,
            group=groups,
            feature_name=list(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES),
        )
        model = lgb.train(params, train_dataset, num_boost_round=config.num_boost_round)
        elapsed = time.perf_counter() - start
        seed_train_seconds.append(elapsed)
        bag_models.append(model)
        _notify(progress, f"bag {index + 1}/{seed_count} trained in {elapsed:.1f}s")

    del x_train, y_train, train_behavioral_features, train_features_by_customer

    _notify(progress, "building cutoff-safe evaluation behavioral features")
    eval_behavioral_features = build_cutoff_behavioral_features(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"),
        eval_split.cutoff,
        target_customer_ids=_labels_as_sets(eval_labels),
        article_attributes_by_id=article_attributes,
    )

    _notify(progress, "scoring evaluation candidates (streaming)")
    summary = _evaluate_streaming_bagged(
        np=np,
        bag_models=bag_models,
        candidate_path=Path(args.eval_candidate_path).expanduser().resolve(),
        validation_labels=_labels_as_sets(eval_labels),
        behavioral_features=eval_behavioral_features,
        lgbm_config=lgbm_config,
        config=config,
        progress_callback=progress,
    )

    report = _build_report(
        summary=summary,
        config=config,
        train_split=train_split,
        eval_split=eval_split,
        train_candidate_path=Path(args.train_candidate_path).expanduser().resolve(),
        eval_candidate_path=Path(args.eval_candidate_path).expanduser().resolve(),
        train_label_customers=len(train_labels),
        eval_label_customers=len(eval_labels),
        train_unique_candidate_pairs=pair_count,
        train_positive_pairs=positive_count,
        seed_train_seconds=seed_train_seconds,
    )
    if args.report_path:
        out = Path(args.report_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        _notify(progress, f"wrote report: {out}")

    _print_summary(summary)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _StrategyMetrics:
    ap_sum: float = 0.0
    recall_sum: float = 0.0


@dataclass
class _BaggedSummary:
    evaluated_customers: int
    unique_candidate_pairs: int
    metrics: dict[str, _StrategyMetrics]

    def map_at_k(self, name: str) -> float:
        return (
            self.metrics[name].ap_sum / self.evaluated_customers
            if self.evaluated_customers
            else 0.0
        )

    def recall_at_k(self, name: str) -> float:
        return (
            self.metrics[name].recall_sum / self.evaluated_customers
            if self.evaluated_customers
            else 0.0
        )


_STRATEGY_NAMES: tuple[str, ...] = (
    "deterministic",
    "bag_mean_blend",
    "bag_mean_only",
)


def _evaluate_streaming_bagged(
    *,
    np: Any,
    bag_models: list[Any],
    candidate_path: Path,
    validation_labels: dict[str, set[str]],
    behavioral_features: Any,
    lgbm_config: LightGBMBehavioralRankerConfig,
    config: BaggedLightGBMConfig,
    progress_callback: Any,
) -> _BaggedSummary:
    chunk_size = 2000
    metrics = {name: _StrategyMetrics() for name in _STRATEGY_NAMES}
    evaluated_customers = 0
    unique_candidate_pairs = 0
    chunk: list[tuple[str, dict[str, CandidateFeatures]]] = []
    for grouped in iter_grouped_candidate_features_from_csv(candidate_path, validation_labels):
        chunk.append(grouped)
        if len(chunk) >= chunk_size:
            evaluated_customers, unique_candidate_pairs = _score_chunk(
                np=np,
                bag_models=bag_models,
                chunk=chunk,
                validation_labels=validation_labels,
                behavioral_features=behavioral_features,
                lgbm_config=lgbm_config,
                config=config,
                metrics=metrics,
                evaluated_customers=evaluated_customers,
                unique_candidate_pairs=unique_candidate_pairs,
            )
            chunk = []
            _notify(
                progress_callback,
                f"scored evaluation customers: {evaluated_customers}/{len(validation_labels)} "
                f"pairs={unique_candidate_pairs}",
            )
    if chunk:
        evaluated_customers, unique_candidate_pairs = _score_chunk(
            np=np,
            bag_models=bag_models,
            chunk=chunk,
            validation_labels=validation_labels,
            behavioral_features=behavioral_features,
            lgbm_config=lgbm_config,
            config=config,
            metrics=metrics,
            evaluated_customers=evaluated_customers,
            unique_candidate_pairs=unique_candidate_pairs,
        )
        _notify(
            progress_callback,
            f"scored evaluation customers: {evaluated_customers}/{len(validation_labels)} "
            f"pairs={unique_candidate_pairs}",
        )
    return _BaggedSummary(
        evaluated_customers=evaluated_customers,
        unique_candidate_pairs=unique_candidate_pairs,
        metrics=metrics,
    )


def _score_chunk(
    *,
    np: Any,
    bag_models: list[Any],
    chunk: list[tuple[str, dict[str, CandidateFeatures]]],
    validation_labels: dict[str, set[str]],
    behavioral_features: Any,
    lgbm_config: LightGBMBehavioralRankerConfig,
    config: BaggedLightGBMConfig,
    metrics: dict[str, _StrategyMetrics],
    evaluated_customers: int,
    unique_candidate_pairs: int,
) -> tuple[int, int]:
    weights = lgbm_config.deterministic_weights
    k = lgbm_config.k
    grouped_values: list[tuple[str, list[CandidateFeatures]]] = []
    offsets: list[tuple[int, int]] = []
    x_rows: list[tuple[float, ...]] = []
    deterministic_scores: list[float] = []
    cursor = 0
    for customer_id, article_features in chunk:
        values = list(article_features.values())
        grouped_values.append((customer_id, values))
        offsets.append((cursor, cursor + len(values)))
        cursor += len(values)
        for features in values:
            x_rows.append(
                lightgbm_behavioral_feature_vector(features, behavioral_features, weights)
            )
            deterministic_scores.append(score_candidate(features, weights))
    if not x_rows:
        return evaluated_customers, unique_candidate_pairs
    x_matrix = np.asarray(x_rows, dtype=np.float32)
    bag_predictions = np.zeros(x_matrix.shape[0], dtype=np.float64)
    for model in bag_models:
        bag_predictions += model.predict(x_matrix)
    bag_predictions /= len(bag_models)
    for (customer_id, values), (start, end) in zip(grouped_values, offsets, strict=True):
        actual = tuple(validation_labels[customer_id])
        det_slice = deterministic_scores[start:end]
        bag_slice = [float(score) for score in bag_predictions[start:end]]
        articles = [features.article_id for features in values]
        det_pred = _rank_articles(articles, det_slice, k)
        metrics["deterministic"].ap_sum += average_precision_at_k(actual, det_pred, k=k)
        metrics["deterministic"].recall_sum += recall_at_k(actual, det_pred, k=k)
        bag_only_pred = _rank_articles(articles, bag_slice, k)
        metrics["bag_mean_only"].ap_sum += average_precision_at_k(actual, bag_only_pred, k=k)
        metrics["bag_mean_only"].recall_sum += recall_at_k(actual, bag_only_pred, k=k)
        blend = _zscore_blend(det_slice, bag_slice, config.blend_lambda)
        bag_blend_pred = _rank_articles(articles, blend, k)
        metrics["bag_mean_blend"].ap_sum += average_precision_at_k(actual, bag_blend_pred, k=k)
        metrics["bag_mean_blend"].recall_sum += recall_at_k(actual, bag_blend_pred, k=k)
        evaluated_customers += 1
        unique_candidate_pairs += len(values)
    return evaluated_customers, unique_candidate_pairs


def _rank_articles(
    articles: list[str], scores: list[float], k: int
) -> tuple[str, ...]:
    return tuple(
        article
        for article, _ in sorted(
            zip(articles, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )[:k]
    )


def _zscore_blend(det: list[float], model: list[float], lam: float) -> list[float]:
    det_mean, det_std = _mean_std(det)
    model_mean, model_std = _mean_std(model)
    return [
        ((d - det_mean) / det_std) + lam * ((m - model_mean) / model_std)
        for d, m in zip(det, model, strict=True)
    ]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    if len(values) < 2:
        return values[0], 1.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    return mean, std if std > 1e-9 else 1.0


def _pad_to(seq: tuple[Any, ...], target_len: int, *, fill: Any) -> tuple[Any, ...]:
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + (fill,) * (target_len - len(seq))


def _config_from_args(args: argparse.Namespace) -> BaggedLightGBMConfig:
    seeds = tuple(args.seeds)
    leaf_jitter = (
        tuple(args.leaf_jitter)
        if args.leaf_jitter
        else (0,) * len(seeds)
    )
    feature_fraction_jitter = (
        tuple(args.feature_fraction_jitter)
        if args.feature_fraction_jitter
        else (0.0,) * len(seeds)
    )
    return BaggedLightGBMConfig(
        k=args.k,
        negative_per_positive=args.negative_per_positive,
        blend_lambda=args.blend_lambda,
        objective=args.objective,
        num_boost_round=args.num_boost_round,
        base_num_leaves=args.base_num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        learning_rate=args.learning_rate,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l2=args.lambda_l2,
        num_threads=args.num_threads,
        seeds=seeds,
        leaf_jitter=leaf_jitter,
        feature_fraction_jitter=feature_fraction_jitter,
    )


def _lgbm_config_for_train_matrix(config: BaggedLightGBMConfig) -> LightGBMBehavioralRankerConfig:
    return LightGBMBehavioralRankerConfig(
        k=config.k,
        negative_per_positive=config.negative_per_positive,
        blend_lambda=config.blend_lambda,
        objective=config.objective,
        deterministic_weights=LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS,
    )


def _maybe_load_article_attributes(articles_csv_path: Path) -> ArticleAttributeMap:
    try:
        from hm_recsys.ranking.behavioral import load_article_attribute_maps

        return load_article_attribute_maps(articles_csv_path.parent)
    except (FileNotFoundError, OSError):
        return {}


def _iter_transaction_events(path: Path) -> Iterator[TransactionEvent]:
    for row in iter_csv_rows(path, ("t_dat", "customer_id", "article_id")):
        yield TransactionEvent(
            t_dat=date.fromisoformat(row["t_dat"]),
            customer_id=row["customer_id"],
            article_id=row["article_id"],
        )


def _import_lightgbm_runtime() -> tuple[Any, Any]:
    import numpy as np

    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LightGBM not installed; install via `pip install lightgbm`."
        ) from exc
    return np, lgb


def _make_progress(prefix: str):
    def cb(message: str) -> None:
        print(f"{prefix} {message}", flush=True)

    return cb


def _build_report(
    *,
    summary: _BaggedSummary,
    config: BaggedLightGBMConfig,
    train_split: TemporalSplit,
    eval_split: TemporalSplit,
    train_candidate_path: Path,
    eval_candidate_path: Path,
    train_label_customers: int,
    eval_label_customers: int,
    train_unique_candidate_pairs: int,
    train_positive_pairs: int,
    seed_train_seconds: list[float],
) -> dict[str, Any]:
    return {
        "model": "lightgbm_bagged_multi_seed",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_cutoff": train_split.cutoff.isoformat(),
        "train_validation_end_exclusive": train_split.validation_end.isoformat(),
        "evaluation_cutoff": eval_split.cutoff.isoformat(),
        "evaluation_end_exclusive": eval_split.validation_end.isoformat(),
        "horizon_days": eval_split.horizon_days,
        "train_candidate_path": str(train_candidate_path),
        "evaluation_candidate_path": str(eval_candidate_path),
        "train_label_customers": train_label_customers,
        "evaluation_label_customers": eval_label_customers,
        "evaluated_customers": summary.evaluated_customers,
        "train_unique_candidate_pairs": train_unique_candidate_pairs,
        "train_positive_pairs": train_positive_pairs,
        "evaluation_unique_candidate_pairs": summary.unique_candidate_pairs,
        "metrics": {
            name: {
                "map_at_k": summary.map_at_k(name),
                "recall_at_k": summary.recall_at_k(name),
            }
            for name in _STRATEGY_NAMES
        },
        "config": asdict(config),
        "seed_train_seconds": seed_train_seconds,
        "feature_names": list(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES),
    }


def _print_summary(summary: _BaggedSummary) -> None:
    print()
    print("[bagged] === MAP@12 / recall@12 by strategy ===")
    for name in _STRATEGY_NAMES:
        m = summary.map_at_k(name)
        r = summary.recall_at_k(name)
        print(f"[bagged] {name:20s} MAP@12={m:.5f}  recall@12={r:.5f}")
    base = summary.map_at_k("deterministic")
    print()
    print("[bagged] === Lift vs deterministic ===")
    for name in _STRATEGY_NAMES:
        if name == "deterministic":
            continue
        m = summary.map_at_k(name)
        lift = m - base
        rel = (lift / base * 100.0) if base > 0 else float("nan")
        print(f"[bagged] {name:20s} delta_MAP={lift:+.5f}  rel={rel:+6.2f}%")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--train-cutoff", default=None)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--train-candidate-path", type=Path, required=True)
    parser.add_argument("--eval-candidate-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--negative-per-positive", type=int, default=50)
    parser.add_argument("--blend-lambda", type=float, default=0.75)
    parser.add_argument(
        "--objective", default="lambdarank", choices=("lambdarank", "rank_xendcg")
    )
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--base-num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l2", type=float, default=5.0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 137, 271, 314, 415]
    )
    parser.add_argument("--leaf-jitter", type=int, nargs="*", default=None)
    parser.add_argument(
        "--feature-fraction-jitter", type=float, nargs="*", default=None
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    return parser


if __name__ == "__main__":
    sys.exit(main())
