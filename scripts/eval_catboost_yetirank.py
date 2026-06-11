"""Train + evaluate CatBoost YetiRank on the LightGBM behavioral feature set.

Re-uses the same feature pipeline and per-customer z-score blend logic as
the LightGBM ranker so the two models are A/B-comparable. The only
substitution is the gradient-boosted model itself.

Produces a JSON report with the same metric shape as the LightGBM
reports plus a ``model`` field naming the booster. Intended for fast
iteration before promoting CatBoost into a proper module under
``src/hm_recsys/ranking/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from hm_recsys.data.io import TransactionEvent, iter_csv_rows
from hm_recsys.evaluation.metrics import recall_at_k  # noqa: F401 - kept for parity with LGBM eval
from hm_recsys.evaluation.temporal import TemporalSplit, collect_validation_labels
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.ranking.behavioral import (
    ArticleAttributeMap,
    build_cutoff_behavioral_features,
)

# Re-using the private helpers from lightgbm_behavioral is intentional:
# the goal is to keep the CatBoost evaluator bit-for-bit comparable to the
# LightGBM one. We only swap the booster.
from hm_recsys.ranking.lightgbm_behavioral import (
    LIGHTGBM_BEHAVIORAL_FEATURE_NAMES,
    LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS,
    LightGBMBehavioralRankerConfig,
    _build_train_matrix,
    _evaluate_streaming,
    _labels_as_sets,
    _notify,
)


@dataclass(frozen=True)
class CatBoostYetiRankConfig:
    """CatBoost-side configuration parameters."""

    iterations: int = 400
    depth: int = 8
    learning_rate: float = 0.05
    l2_leaf_reg: float = 5.0
    rsm: float = 0.9
    bagging_temperature: float = 1.0
    seed: int = 42
    thread_count: int = 4
    loss_function: str = "YetiRank"  # or "YetiRankPairwise"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "rsm": self.rsm,
            "bagging_temperature": self.bagging_temperature,
            "seed": self.seed,
            "thread_count": self.thread_count,
            "loss_function": self.loss_function,
        }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    eval_cutoff = date.fromisoformat(args.cutoff)
    train_cutoff_str = args.train_cutoff or (
        eval_cutoff.replace(day=eval_cutoff.day) - _safe_timedelta(args.horizon_days)
    ).isoformat()
    train_cutoff = date.fromisoformat(train_cutoff_str)
    train_split = TemporalSplit(cutoff=train_cutoff, horizon_days=args.horizon_days)
    eval_split = TemporalSplit(cutoff=eval_cutoff, horizon_days=args.horizon_days)

    lgbm_config = LightGBMBehavioralRankerConfig(
        k=args.k,
        negative_per_positive=args.negative_per_positive,
        blend_lambda=args.blend_lambda,
        deterministic_weights=LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS,
    )
    catboost_config = CatBoostYetiRankConfig(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        rsm=args.rsm,
        bagging_temperature=args.bagging_temperature,
        seed=args.seed,
        thread_count=args.thread_count,
        loss_function=args.loss_function,
    )

    progress = _make_progress("[catboost]") if args.progress else None
    _notify(progress, "loading article metadata attributes")
    article_attributes = _maybe_load_article_attributes(paths.raw_data_dir / "articles.csv")
    _notify(progress, f"loaded article metadata for {len(article_attributes)} articles")

    _notify(progress, "loading training validation labels")
    train_labels = collect_validation_labels(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"), train_split
    )
    _notify(progress, f"loaded training labels for {len(train_labels)} customers")

    _notify(progress, "loading evaluation validation labels")
    eval_labels = collect_validation_labels(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"), eval_split
    )
    _notify(progress, f"loaded evaluation labels for {len(eval_labels)} customers")

    _notify(progress, "loading grouped training candidates")
    from hm_recsys.ranking.lightgbm_behavioral import (
        iter_grouped_candidate_features_from_csv,
    )

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
    # TransactionEvent already satisfies the BehavioralTransaction Protocol
    # (t_dat / customer_id / article_id) so we can pass it straight through.
    train_behavioral_features = build_cutoff_behavioral_features(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"),
        train_split.cutoff,
        target_customer_ids=_labels_as_sets(train_labels),
        article_attributes_by_id=article_attributes,
    )

    _notify(progress, "importing CatBoost runtime")
    np, cb = _import_catboost_runtime()
    _notify(progress, "building CatBoost training matrix")
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

    group_id = _group_id_from_sizes(np, groups)
    train_pool = cb.Pool(data=x_train, label=y_train, group_id=group_id)
    _notify(
        progress,
        f"training CatBoost YetiRank: {len(y_train)} rows, {positive_count} positives, "
        f"{len(groups)} groups, iterations={catboost_config.iterations}",
    )
    train_start = time.perf_counter()
    model = cb.CatBoostRanker(
        loss_function=catboost_config.loss_function,
        iterations=catboost_config.iterations,
        depth=catboost_config.depth,
        learning_rate=catboost_config.learning_rate,
        l2_leaf_reg=catboost_config.l2_leaf_reg,
        rsm=catboost_config.rsm,
        bagging_temperature=catboost_config.bagging_temperature,
        random_seed=catboost_config.seed,
        thread_count=catboost_config.thread_count,
        verbose=False,
    )
    model.fit(train_pool)
    _notify(
        progress, f"trained CatBoost in {time.perf_counter() - train_start:.1f}s"
    )

    _notify(progress, "building cutoff-safe evaluation behavioral features")
    eval_behavioral_features = build_cutoff_behavioral_features(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"),
        eval_split.cutoff,
        target_customer_ids=_labels_as_sets(eval_labels),
        article_attributes_by_id=article_attributes,
    )

    _notify(progress, "scoring evaluation candidates")
    metrics = _evaluate_streaming(
        np=np,
        model=model,
        candidate_path=Path(args.eval_candidate_path).expanduser().resolve(),
        validation_labels=_labels_as_sets(eval_labels),
        behavioral_features=eval_behavioral_features,
        config=lgbm_config,
        progress_callback=progress,
    )

    report = {
        "model": "catboost_yetirank",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_cutoff": train_split.cutoff.isoformat(),
        "train_validation_end_exclusive": train_split.validation_end.isoformat(),
        "evaluation_cutoff": eval_split.cutoff.isoformat(),
        "evaluation_end_exclusive": eval_split.validation_end.isoformat(),
        "horizon_days": eval_split.horizon_days,
        "train_candidate_path": str(Path(args.train_candidate_path).expanduser().resolve()),
        "evaluation_candidate_path": str(
            Path(args.eval_candidate_path).expanduser().resolve()
        ),
        "train_label_customers": len(train_labels),
        "evaluation_label_customers": len(eval_labels),
        "evaluated_customers": metrics.evaluated_customers,
        "missing_evaluation_label_customers": (
            len(eval_labels) - metrics.evaluated_customers
        ),
        "train_unique_candidate_pairs": pair_count,
        "train_positive_pairs": positive_count,
        "train_negative_pairs_sampled": len(y_train) - positive_count,
        "train_matrix_rows": len(y_train),
        "evaluation_unique_candidate_pairs": metrics.unique_candidate_pairs,
        "deterministic_map_at_k": metrics.deterministic_map_at_k,
        "deterministic_recall_at_k": metrics.deterministic_recall_at_k,
        "model_only_map_at_k": metrics.model_only_map_at_k,
        "model_only_recall_at_k": metrics.model_only_recall_at_k,
        "blend_map_at_k": metrics.blend_map_at_k,
        "blend_recall_at_k": metrics.blend_recall_at_k,
        "delta_vs_deterministic_map_at_k": (
            metrics.blend_map_at_k - metrics.deterministic_map_at_k
        ),
        "delta_vs_deterministic_recall_at_k": (
            metrics.blend_recall_at_k - metrics.deterministic_recall_at_k
        ),
        "blend_normalization": "per_customer_zscore",
        "feature_names": list(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES),
        "catboost_config": catboost_config.to_dict(),
        "lightgbm_config_for_comparison": {
            "k": lgbm_config.k,
            "negative_per_positive": lgbm_config.negative_per_positive,
            "blend_lambda": lgbm_config.blend_lambda,
        },
    }

    if args.report_path:
        out = Path(args.report_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        _notify(progress, f"wrote report: {out}")

    print(
        f"[catboost] CatBoost YetiRank: deterministic={metrics.deterministic_map_at_k:.5f} "
        f"model_only={metrics.model_only_map_at_k:.5f} "
        f"blend={metrics.blend_map_at_k:.5f} "
        f"delta={(metrics.blend_map_at_k - metrics.deterministic_map_at_k):+.5f}",
        flush=True,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="Evaluation cutoff (YYYY-MM-DD).")
    parser.add_argument(
        "--train-cutoff", default=None, help="Training cutoff (default: cutoff - horizon_days)."
    )
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--train-candidate-path", type=Path, required=True)
    parser.add_argument("--eval-candidate-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--negative-per-positive", type=int, default=50)
    parser.add_argument("--blend-lambda", type=float, default=0.75)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--rsm", type=float, default=0.9)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thread-count", type=int, default=4)
    parser.add_argument(
        "--loss-function",
        default="YetiRank",
        choices=("YetiRank", "YetiRankPairwise"),
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    return parser


def _make_progress(prefix: str):
    def cb(message: str) -> None:
        print(f"{prefix} {message}", flush=True)

    return cb


def _iter_transaction_events(path: Path) -> Iterable[TransactionEvent]:
    for row in iter_csv_rows(path, ("t_dat", "customer_id", "article_id")):
        yield TransactionEvent(
            t_dat=date.fromisoformat(row["t_dat"]),
            customer_id=row["customer_id"],
            article_id=row["article_id"],
        )


def _maybe_load_article_attributes(articles_csv_path: Path) -> ArticleAttributeMap:
    """Best-effort article attributes loader; returns empty map on failure."""

    try:
        from hm_recsys.ranking.behavioral import (
            load_article_attribute_maps,
        )

        return load_article_attribute_maps(articles_csv_path.parent)
    except (FileNotFoundError, OSError):
        return {}


def _group_id_from_sizes(np, group_sizes: list[int]):
    """Convert LightGBM-style group sizes to CatBoost-style group_id array."""

    group_ids = []
    for group_index, size in enumerate(group_sizes):
        group_ids.extend([group_index] * size)
    return np.asarray(group_ids, dtype=np.int64)


def _import_catboost_runtime():
    import numpy as np

    try:
        import catboost as cb
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "CatBoost runtime missing; install via `pip install 'catboost>=1.2'`."
        ) from exc
    return np, cb


def _safe_timedelta(days: int) -> timedelta:
    return timedelta(days=days)


if __name__ == "__main__":
    sys.exit(main())
