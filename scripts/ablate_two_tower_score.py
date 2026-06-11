"""Quick A/B ablation: does two_tower_score lift MAP@12 over source-only?

Given a candidate CSV already scored by ``score-two-tower-candidates``
(i.e. contains a ``two_tower_score`` column), compute MAP@12 for five
ranking strategies:

* ``source_only``     - baseline; rank by per-pair max(source_score)
* ``two_tower_only``  - rank by two_tower_score
* ``blend_alpha=0.25`` / ``0.50`` / ``0.75`` - z-normalized blends

Reads validation labels from ``transactions_train.csv`` for the 7 days
following ``--cutoff``. Restricts evaluation to customers that appear in
the candidate CSV (i.e. the 1k/10k slice) AND have at least one
validation purchase.

This is a *directional* read on whether the two-tower contributes
ranking signal as a standalone feature; it is not a full LightGBM
retraining. See SDD plan Phase 3 exit gate.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from hm_recsys.data.io import TransactionEvent, iter_csv_rows
from hm_recsys.evaluation.metrics import average_precision_at_k
from hm_recsys.evaluation.temporal import TemporalSplit, collect_validation_labels
from hm_recsys.infrastructure.paths import ProjectPaths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-candidates",
        type=Path,
        required=True,
        help="CSV produced by 'score-two-tower-candidates' (with 'two_tower_score' column).",
    )
    parser.add_argument("--cutoff", type=str, required=True, help="Training cutoff (YYYY-MM-DD).")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
        help="Z-blend weights (alpha * source + (1-alpha) * two_tower).",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit(cutoff=date.fromisoformat(args.cutoff), horizon_days=args.horizon_days)

    print(f"[ablate] reading scored candidates: {args.scored_candidates}", flush=True)
    pair_to_features = _collect_pair_features(args.scored_candidates)
    customers_in_slice = {customer for customer, _ in pair_to_features}
    print(
        f"[ablate] unique_pairs={len(pair_to_features)} "
        f"customers_in_slice={len(customers_in_slice)}",
        flush=True,
    )

    print(f"[ablate] loading validation labels for cutoff={split.cutoff}...", flush=True)
    validation_labels = collect_validation_labels(
        _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv"), split
    )
    eval_customers = sorted(customers_in_slice & validation_labels.keys())
    print(
        f"[ablate] customers_with_validation_labels_in_slice={len(eval_customers)}"
        f" (full_validation_universe={len(validation_labels)})",
        flush=True,
    )
    if not eval_customers:
        print("[ablate] no overlap between slice and validation; nothing to evaluate", flush=True)
        return 1

    pairs_by_customer: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for (customer, article), (source_score, two_tower_score) in pair_to_features.items():
        pairs_by_customer[customer].append((article, source_score, two_tower_score))

    source_stats = _global_zscore_stats(
        feature_index=1, pairs_by_customer=pairs_by_customer, eval_customers=eval_customers
    )
    two_tower_stats = _global_zscore_stats(
        feature_index=2, pairs_by_customer=pairs_by_customer, eval_customers=eval_customers
    )
    print(
        f"[ablate] source_score mean={source_stats[0]:.4f} stdev={source_stats[1]:.4f} | "
        f"two_tower_score mean={two_tower_stats[0]:.4f} stdev={two_tower_stats[1]:.4f}",
        flush=True,
    )

    strategies: dict[str, object] = {
        "source_only": lambda src, tt: src,
        "two_tower_only": lambda src, tt: tt,
    }
    for alpha in args.alphas:
        strategies[f"blend_alpha={alpha:.2f}"] = _make_blend(
            alpha=alpha, source_stats=source_stats, two_tower_stats=two_tower_stats
        )

    results: dict[str, float] = {}
    for name, score_fn in strategies.items():
        map_at_k = _evaluate_strategy(
            pairs_by_customer=pairs_by_customer,
            eval_customers=eval_customers,
            validation_labels=validation_labels,
            score_fn=score_fn,
            k=args.k,
        )
        results[name] = map_at_k
        print(f"[ablate] {name:24s} MAP@{args.k}={map_at_k:.5f}", flush=True)

    baseline = results["source_only"]
    print()
    print("[ablate] === lift vs source_only baseline ===", flush=True)
    for name, score in results.items():
        lift = score - baseline
        rel = (lift / baseline * 100.0) if baseline > 0 else float("nan")
        marker = " <-- best" if score == max(results.values()) else ""
        print(
            f"[ablate] {name:24s} MAP@{args.k}={score:.5f}  "
            f"abs_lift={lift:+.5f}  rel_lift={rel:+.2f}%{marker}",
            flush=True,
        )
    return 0


def _collect_pair_features(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Aggregate scored candidate rows into one entry per (customer, article).

    For each pair, keeps the max source_score (since multiple sources can
    emit the same pair) and the two_tower_score (constant per pair).
    """

    pair_to_features: dict[tuple[str, str], tuple[float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"customer_id", "article_id", "source_score", "two_tower_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"scored CSV missing columns: {sorted(missing)}")
        for row in reader:
            key = (row["customer_id"], row["article_id"])
            source_score = float(row["source_score"])
            two_tower_score = float(row["two_tower_score"])
            existing = pair_to_features.get(key)
            if existing is None:
                pair_to_features[key] = (source_score, two_tower_score)
            else:
                pair_to_features[key] = (
                    max(existing[0], source_score),
                    two_tower_score,
                )
    return pair_to_features


def _iter_transaction_events(path: Path):
    for row in iter_csv_rows(path, ("t_dat", "customer_id", "article_id")):
        yield TransactionEvent(
            t_dat=date.fromisoformat(row["t_dat"]),
            customer_id=row["customer_id"],
            article_id=row["article_id"],
        )


def _global_zscore_stats(
    *,
    feature_index: int,
    pairs_by_customer: dict[str, list[tuple[str, float, float]]],
    eval_customers: list[str],
) -> tuple[float, float]:
    """Mean and (population) stdev over all pair-feature values in the slice."""

    values: list[float] = []
    for customer in eval_customers:
        for entry in pairs_by_customer[customer]:
            values.append(entry[feature_index])
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 1.0
    stdev = statistics.pstdev(values)
    return mean, stdev if stdev > 0 else 1.0


def _make_blend(
    *, alpha: float, source_stats: tuple[float, float], two_tower_stats: tuple[float, float]
):
    src_mean, src_std = source_stats
    tt_mean, tt_std = two_tower_stats

    def blend(src: float, tt: float) -> float:
        z_src = (src - src_mean) / src_std
        z_tt = (tt - tt_mean) / tt_std
        return alpha * z_src + (1.0 - alpha) * z_tt

    return blend


def _evaluate_strategy(
    *,
    pairs_by_customer: dict[str, list[tuple[str, float, float]]],
    eval_customers: list[str],
    validation_labels: dict[str, tuple[str, ...]],
    score_fn,
    k: int,
) -> float:
    """Compute mean MAP@k over the supplied eval customers."""

    total_score = 0.0
    for customer in eval_customers:
        entries = pairs_by_customer[customer]
        ranked = sorted(
            entries,
            key=lambda entry: score_fn(entry[1], entry[2]),
            reverse=True,
        )
        predicted = [article for article, _, _ in ranked[:k]]
        labels = validation_labels.get(customer, ())
        total_score += average_precision_at_k(labels, predicted, k=k)
    return total_score / len(eval_customers) if eval_customers else 0.0


if __name__ == "__main__":
    sys.exit(main())
