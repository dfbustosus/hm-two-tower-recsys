"""Train LightGBM + CatBoost in one process and evaluate blended MAP@12.

Loads candidates and behavioral features **once**, trains both gradient-
boosted rankers on the same feature matrix, then computes per-customer
predictions and blends them via:

* per-customer z-score weighted sum (with deterministic prior), and
* Reciprocal Rank Fusion (RRF)

Reports MAP@12 / recall@12 for:

* deterministic baseline
* LightGBM blended with deterministic
* CatBoost blended with deterministic
* (LightGBM + CatBoost) z-score ensemble blended with deterministic
* (LightGBM + CatBoost) RRF ensemble (no deterministic blend; pure rank)

The ensemble is the Phase 4 deliverable; the per-model numbers are
diagnostic so we can attribute lift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable, Iterator
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
from hm_recsys.ranking.deterministic import (
    CandidateFeatures,
    score_candidate,
)
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
class EnsembleConfig:
    """Hyperparameters for the dual-model ensemble eval."""

    k: int = 12
    negative_per_positive: int = 50
    blend_lambda: float = 0.75

    # LightGBM
    lgbm_objective: str = "lambdarank"
    lgbm_num_boost_round: int = 200
    lgbm_num_leaves: int = 63
    lgbm_min_data_in_leaf: int = 100
    lgbm_learning_rate: float = 0.03
    lgbm_feature_fraction: float = 0.9
    lgbm_bagging_fraction: float = 0.9
    lgbm_bagging_freq: int = 1
    lgbm_lambda_l2: float = 5.0
    lgbm_seed: int = 42
    lgbm_num_threads: int = 4

    # CatBoost
    catboost_loss_function: str = "YetiRank"
    catboost_iterations: int = 400
    catboost_depth: int = 8
    catboost_learning_rate: float = 0.05
    catboost_l2_leaf_reg: float = 5.0
    catboost_rsm: float = 0.9
    catboost_bagging_temperature: float = 1.0
    catboost_seed: int = 42
    catboost_thread_count: int = 4

    # Ensemble blends
    rrf_constant: float = 60.0
    ensemble_lambda_lgbm: float = 1.0
    ensemble_lambda_catboost: float = 1.0


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
    progress = _make_progress("[ensemble]") if args.progress else None

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
    _notify(
        progress,
        f"loaded labels train={len(train_labels)} eval={len(eval_labels)}",
    )

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
    cb = _import_catboost_runtime()

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

    _notify(progress, "training LightGBM")
    lgbm_t0 = time.perf_counter()
    lgbm_train_dataset = lgb.Dataset(
        x_train,
        label=y_train,
        group=groups,
        feature_name=list(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES),
    )
    lgbm_model = lgb.train(
        {
            "objective": config.lgbm_objective,
            "metric": "ndcg",
            "ndcg_eval_at": [config.k],
            "learning_rate": config.lgbm_learning_rate,
            "num_leaves": config.lgbm_num_leaves,
            "min_data_in_leaf": config.lgbm_min_data_in_leaf,
            "feature_fraction": config.lgbm_feature_fraction,
            "bagging_fraction": config.lgbm_bagging_fraction,
            "bagging_freq": config.lgbm_bagging_freq,
            "lambda_l2": config.lgbm_lambda_l2,
            "verbosity": -1,
            "seed": config.lgbm_seed,
            "num_threads": config.lgbm_num_threads,
        },
        lgbm_train_dataset,
        num_boost_round=config.lgbm_num_boost_round,
    )
    _notify(progress, f"LightGBM trained in {time.perf_counter() - lgbm_t0:.1f}s")

    _notify(progress, "training CatBoost YetiRank")
    cat_t0 = time.perf_counter()
    catboost_group_id = _group_id_from_sizes(np, groups)
    catboost_train_pool = cb.Pool(data=x_train, label=y_train, group_id=catboost_group_id)
    catboost_model = cb.CatBoostRanker(
        loss_function=config.catboost_loss_function,
        iterations=config.catboost_iterations,
        depth=config.catboost_depth,
        learning_rate=config.catboost_learning_rate,
        l2_leaf_reg=config.catboost_l2_leaf_reg,
        rsm=config.catboost_rsm,
        bagging_temperature=config.catboost_bagging_temperature,
        random_seed=config.catboost_seed,
        thread_count=config.catboost_thread_count,
        verbose=False,
    )
    catboost_model.fit(catboost_train_pool)
    _notify(progress, f"CatBoost trained in {time.perf_counter() - cat_t0:.1f}s")

    # Free training matrices before eval pass.
    del x_train, y_train, train_behavioral_features, train_features_by_customer

    _notify(progress, "building cutoff-safe evaluation behavioral features")
    eval_behavioral_features = build_cutoff_behavioral_features(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"),
        eval_split.cutoff,
        target_customer_ids=_labels_as_sets(eval_labels),
        article_attributes_by_id=article_attributes,
    )

    _notify(progress, "scoring evaluation candidates (streaming)")
    summary = _evaluate_streaming_ensemble(
        np=np,
        lgbm_model=lgbm_model,
        catboost_model=catboost_model,
        candidate_path=Path(args.eval_candidate_path).expanduser().resolve(),
        validation_labels=_labels_as_sets(eval_labels),
        behavioral_features=eval_behavioral_features,
        lgbm_config=lgbm_config,
        ensemble_config=config,
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
        train_matrix_rows=positive_count + (positive_count * config.negative_per_positive),
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

    def add(self, predicted: tuple[str, ...], labels: Iterable[str], k: int) -> None:
        self.ap_sum += average_precision_at_k(labels, predicted, k=k)
        self.recall_sum += recall_at_k(labels, predicted, k=k)


@dataclass
class _EnsembleSummary:
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
    "lgbm_blend",
    "catboost_blend",
    "ensemble_zscore_blend",
    "ensemble_rrf",
)


def _evaluate_streaming_ensemble(
    *,
    np: Any,
    lgbm_model: Any,
    catboost_model: Any,
    candidate_path: Path,
    validation_labels: dict[str, set[str]],
    behavioral_features: Any,
    lgbm_config: LightGBMBehavioralRankerConfig,
    ensemble_config: EnsembleConfig,
    progress_callback: Any,
) -> _EnsembleSummary:
    chunk_size = 2000
    metrics = {name: _StrategyMetrics() for name in _STRATEGY_NAMES}
    evaluated_customers = 0
    unique_candidate_pairs = 0
    chunk: list[tuple[str, dict[str, CandidateFeatures]]] = []
    for grouped in iter_grouped_candidate_features_from_csv(candidate_path, validation_labels):
        chunk.append(grouped)
        if len(chunk) >= chunk_size:
            evaluated_customers, unique_candidate_pairs = _score_chunk_into_metrics(
                np=np,
                lgbm_model=lgbm_model,
                catboost_model=catboost_model,
                chunk=chunk,
                validation_labels=validation_labels,
                behavioral_features=behavioral_features,
                lgbm_config=lgbm_config,
                ensemble_config=ensemble_config,
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
        evaluated_customers, unique_candidate_pairs = _score_chunk_into_metrics(
            np=np,
            lgbm_model=lgbm_model,
            catboost_model=catboost_model,
            chunk=chunk,
            validation_labels=validation_labels,
            behavioral_features=behavioral_features,
            lgbm_config=lgbm_config,
            ensemble_config=ensemble_config,
            metrics=metrics,
            evaluated_customers=evaluated_customers,
            unique_candidate_pairs=unique_candidate_pairs,
        )
        _notify(
            progress_callback,
            f"scored evaluation customers: {evaluated_customers}/{len(validation_labels)} "
            f"pairs={unique_candidate_pairs}",
        )
    return _EnsembleSummary(
        evaluated_customers=evaluated_customers,
        unique_candidate_pairs=unique_candidate_pairs,
        metrics=metrics,
    )


def _score_chunk_into_metrics(
    *,
    np: Any,
    lgbm_model: Any,
    catboost_model: Any,
    chunk: list[tuple[str, dict[str, CandidateFeatures]]],
    validation_labels: dict[str, set[str]],
    behavioral_features: Any,
    lgbm_config: LightGBMBehavioralRankerConfig,
    ensemble_config: EnsembleConfig,
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
    lgbm_scores = lgbm_model.predict(x_matrix)
    catboost_scores = catboost_model.predict(x_matrix)
    for (customer_id, values), (start, end) in zip(grouped_values, offsets, strict=True):
        actual = tuple(validation_labels[customer_id])
        det_slice = deterministic_scores[start:end]
        lgbm_slice = [float(score) for score in lgbm_scores[start:end]]
        catboost_slice = [float(score) for score in catboost_scores[start:end]]
        articles = [features.article_id for features in values]

        # deterministic
        det_predictions = _rank_articles(articles, det_slice, k)
        metrics["deterministic"].add(det_predictions, actual, k)

        # det + lgbm
        lgbm_blend = _zscore_blend(det_slice, lgbm_slice, ensemble_config.blend_lambda)
        lgbm_predictions = _rank_articles(articles, lgbm_blend, k)
        metrics["lgbm_blend"].add(lgbm_predictions, actual, k)

        # det + catboost
        cat_blend = _zscore_blend(det_slice, catboost_slice, ensemble_config.blend_lambda)
        cat_predictions = _rank_articles(articles, cat_blend, k)
        metrics["catboost_blend"].add(cat_predictions, actual, k)

        # det + lambda_lgbm * z(lgbm) + lambda_cb * z(catboost)
        ensemble_blend = _multi_zscore_blend(
            det_slice,
            (
                (ensemble_config.ensemble_lambda_lgbm, lgbm_slice),
                (ensemble_config.ensemble_lambda_catboost, catboost_slice),
            ),
        )
        ensemble_predictions = _rank_articles(articles, ensemble_blend, k)
        metrics["ensemble_zscore_blend"].add(ensemble_predictions, actual, k)

        # RRF of lgbm + catboost (no determinitic; pure model fusion)
        rrf_scores = _rrf_scores(
            [_score_rank(lgbm_slice), _score_rank(catboost_slice)],
            constant=ensemble_config.rrf_constant,
        )
        rrf_predictions = _rank_articles(articles, rrf_scores, k)
        metrics["ensemble_rrf"].add(rrf_predictions, actual, k)

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


def _multi_zscore_blend(
    det: list[float], extra: tuple[tuple[float, list[float]], ...]
) -> list[float]:
    det_mean, det_std = _mean_std(det)
    base = [(d - det_mean) / det_std for d in det]
    for lam, model in extra:
        m_mean, m_std = _mean_std(model)
        z = [(m - m_mean) / m_std for m in model]
        base = [b + lam * zi for b, zi in zip(base, z, strict=True)]
    return base


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    if len(values) < 2:
        return values[0], 1.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    return mean, std if std > 1e-9 else 1.0


def _score_rank(scores: list[float]) -> list[int]:
    """Return 1-based ranks (highest score = rank 1) used by RRF."""

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(scores)
    for rank_index, position in enumerate(order, start=1):
        ranks[position] = rank_index
    return ranks


def _rrf_scores(rank_lists: list[list[int]], *, constant: float) -> list[float]:
    if not rank_lists:
        return []
    n = len(rank_lists[0])
    fused = [0.0] * n
    for ranks in rank_lists:
        for index, rank in enumerate(ranks):
            fused[index] += 1.0 / (constant + rank)
    return fused


def _group_id_from_sizes(np: Any, group_sizes: list[int]) -> Any:
    group_ids: list[int] = []
    for group_index, size in enumerate(group_sizes):
        group_ids.extend([group_index] * size)
    return np.asarray(group_ids, dtype=np.int64)


def _config_from_args(args: argparse.Namespace) -> EnsembleConfig:
    return EnsembleConfig(
        k=args.k,
        negative_per_positive=args.negative_per_positive,
        blend_lambda=args.blend_lambda,
        lgbm_objective=args.lgbm_objective,
        lgbm_num_boost_round=args.lgbm_num_boost_round,
        lgbm_num_leaves=args.lgbm_num_leaves,
        lgbm_min_data_in_leaf=args.lgbm_min_data_in_leaf,
        lgbm_learning_rate=args.lgbm_learning_rate,
        catboost_loss_function=args.catboost_loss_function,
        catboost_iterations=args.catboost_iterations,
        catboost_depth=args.catboost_depth,
        catboost_learning_rate=args.catboost_learning_rate,
        rrf_constant=args.rrf_constant,
        ensemble_lambda_lgbm=args.ensemble_lambda_lgbm,
        ensemble_lambda_catboost=args.ensemble_lambda_catboost,
    )


def _lgbm_config_for_train_matrix(config: EnsembleConfig) -> LightGBMBehavioralRankerConfig:
    """Wrap fields needed by ``_build_train_matrix`` into the upstream config."""

    return LightGBMBehavioralRankerConfig(
        k=config.k,
        negative_per_positive=config.negative_per_positive,
        blend_lambda=config.blend_lambda,
        objective=config.lgbm_objective,
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


def _import_catboost_runtime() -> Any:
    try:
        import catboost as cb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CatBoost not installed; install via `pip install 'catboost>=1.2'`."
        ) from exc
    return cb


def _make_progress(prefix: str):
    def cb(message: str) -> None:
        print(f"{prefix} {message}", flush=True)

    return cb


def _build_report(
    *,
    summary: _EnsembleSummary,
    config: EnsembleConfig,
    train_split: TemporalSplit,
    eval_split: TemporalSplit,
    train_candidate_path: Path,
    eval_candidate_path: Path,
    train_label_customers: int,
    eval_label_customers: int,
    train_unique_candidate_pairs: int,
    train_positive_pairs: int,
    train_matrix_rows: int,
) -> dict[str, Any]:
    return {
        "model": "lgbm_catboost_ensemble",
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
        "train_matrix_rows": train_matrix_rows,
        "evaluation_unique_candidate_pairs": summary.unique_candidate_pairs,
        "metrics": {
            name: {
                "map_at_k": summary.map_at_k(name),
                "recall_at_k": summary.recall_at_k(name),
            }
            for name in _STRATEGY_NAMES
        },
        "config": asdict(config),
        "feature_names": list(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES),
    }


def _print_summary(summary: _EnsembleSummary) -> None:
    print()
    print("[ensemble] === MAP@12 / recall@12 by strategy ===")
    for name in _STRATEGY_NAMES:
        m = summary.map_at_k(name)
        r = summary.recall_at_k(name)
        print(f"[ensemble] {name:25s} MAP@12={m:.5f}  recall@12={r:.5f}")
    base = summary.map_at_k("deterministic")
    print()
    print("[ensemble] === Lift vs deterministic ===")
    for name in _STRATEGY_NAMES:
        if name == "deterministic":
            continue
        m = summary.map_at_k(name)
        lift = m - base
        rel = (lift / base * 100.0) if base > 0 else float("nan")
        print(f"[ensemble] {name:25s} delta_MAP={lift:+.5f}  rel={rel:+6.2f}%")


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
        "--lgbm-objective", default="lambdarank", choices=("lambdarank", "rank_xendcg")
    )
    parser.add_argument("--lgbm-num-boost-round", type=int, default=200)
    parser.add_argument("--lgbm-num-leaves", type=int, default=63)
    parser.add_argument("--lgbm-min-data-in-leaf", type=int, default=100)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)

    parser.add_argument(
        "--catboost-loss-function",
        default="YetiRank",
        choices=("YetiRank", "YetiRankPairwise"),
    )
    parser.add_argument("--catboost-iterations", type=int, default=400)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)

    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--ensemble-lambda-lgbm", type=float, default=1.0)
    parser.add_argument("--ensemble-lambda-catboost", type=float, default=1.0)

    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    return parser


if __name__ == "__main__":
    sys.exit(main())
