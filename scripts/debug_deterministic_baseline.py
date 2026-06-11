"""Ground-truth deterministic-baseline MAP@12 for a candidate CSV."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from hm_recsys.data.io import TransactionEvent, iter_csv_rows
from hm_recsys.evaluation.metrics import average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import TemporalSplit, collect_validation_labels
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.ranking.deterministic import DEFAULT_DETERMINISTIC_RANKER_WEIGHTS, score_candidate
from hm_recsys.ranking.lightgbm_behavioral import (
    _labels_as_sets,
    iter_grouped_candidate_features_from_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--max-customers", type=int, default=None)
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    split = TemporalSplit(cutoff=date.fromisoformat(args.cutoff), horizon_days=args.horizon_days)

    print("[debug] loading validation labels", flush=True)

    def _events():
        for row in iter_csv_rows(
            paths.raw_data_dir / "transactions_train.csv",
            ("t_dat", "customer_id", "article_id"),
        ):
            yield TransactionEvent(
                t_dat=date.fromisoformat(row["t_dat"]),
                customer_id=row["customer_id"],
                article_id=row["article_id"],
            )

    eval_labels = collect_validation_labels(_events(), split)
    print(f"[debug] eval_labels={len(eval_labels)} customers", flush=True)

    weights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS
    label_sets = _labels_as_sets(eval_labels)
    ap_sum = 0.0
    recall_sum = 0.0
    customers = 0
    pairs = 0
    first_dump = True
    for customer_id, article_features in iter_grouped_candidate_features_from_csv(
        args.candidate_path, label_sets
    ):
        if args.max_customers is not None and customers >= args.max_customers:
            break
        values = list(article_features.values())
        scored = [(score_candidate(f, weights), f.article_id) for f in values]
        scored.sort(key=lambda x: (-x[0], x[1]))
        top_k_articles = [a for _, a in scored[: args.k]]
        actual = label_sets[customer_id]
        ap_sum += average_precision_at_k(actual, top_k_articles, k=args.k)
        recall_sum += recall_at_k(actual, top_k_articles, k=args.k)
        customers += 1
        pairs += len(values)
        if first_dump:
            print(
                f"[debug] FIRST customer={customer_id} candidates={len(values)} "
                f"labels={len(actual)} top12={top_k_articles[:5]}...",
                flush=True,
            )
            first_dump = False
        if customers % 2000 == 0:
            print(
                f"[debug] processed customers={customers} "
                f"running_MAP@12={ap_sum/customers:.5f}",
                flush=True,
            )
    print(
        f"[debug] FINAL customers={customers} pairs={pairs} "
        f"MAP@12={ap_sum/customers:.5f} recall@12={recall_sum/customers:.5f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
