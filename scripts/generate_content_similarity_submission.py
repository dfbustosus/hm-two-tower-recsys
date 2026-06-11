"""Generate a full-population Kaggle submission from FashionCLIP user-item cosine.

For every customer in ``--reference-csv`` (Kaggle's ``sample_submission.csv``):

1. Build a customer query vector = L2-normalized mean of FashionCLIP article
   embeddings for the customer's most-recent in-window purchases.
2. Score the query against the full article matrix via batched cosine matmul.
3. Take top-K articles, filter optionally to non-history items.
4. For cold-start customers (no purchases in the lookback window) fall back to
   a global recent-popularity top-12 computed from the same transactions.

Why this script:
The 1st-place H&M solution used image+text embedding similarity as a CANDIDATE
SOURCE, not a re-ranking feature. Our existing
``augment_candidates_with_content_retrieval.py`` produces content candidates
for re-ranker training (per ranked-CSV); this script produces a STANDALONE
Kaggle submission so the content-only predictions can be RRF-blended directly
against our best LightGBM submission to recover orthogonal signal.

Memory plan: the full (N_customers x N_articles) cosine matrix is
1.37M x 105k x fp32 = ~576 GB and does not fit in RAM. We therefore
matmul in customer batches of ``--customer-batch-size``: each batch holds
``batch x N_articles x 4 bytes`` (default 1.6 GB for batch=4000).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    cutoff: date
    history_lookback_days: int
    max_history_items: int
    top_k: int
    include_history: bool
    customer_batch_size: int
    popularity_lookback_days: int


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = RunConfig(
        cutoff=date.fromisoformat(args.cutoff),
        history_lookback_days=args.history_lookback_days,
        max_history_items=args.max_history_items,
        top_k=args.top_k,
        include_history=args.include_content_history,
        customer_batch_size=args.customer_batch_size,
        popularity_lookback_days=args.popularity_lookback_days,
    )

    t_overall = time.perf_counter()
    _log(f"loading article embeddings: {args.embeddings_manifest_path}")
    article_ids, article_matrix = _load_article_matrix(
        Path(args.embeddings_manifest_path)
    )
    article_id_to_row = {a: i for i, a in enumerate(article_ids)}
    _log(f"  -> {len(article_ids)} articles, dim={article_matrix.shape[1]}")

    _log(f"loading reference customer order: {args.reference_csv}")
    target_customers = _load_customer_order(Path(args.reference_csv))
    _log(f"  -> {len(target_customers)} target customers")

    _log("scanning transactions for per-customer history + popularity prior")
    customer_queries, popularity_top12 = _build_queries_and_popularity(
        transactions_csv=Path(args.transactions_csv),
        target_customers=set(target_customers),
        article_id_to_row=article_id_to_row,
        article_matrix=article_matrix,
        config=config,
    )
    warm = sum(1 for v in customer_queries.values() if v is not None)
    _log(
        f"  -> warm customers: {warm}/{len(target_customers)} "
        f"({100*warm/len(target_customers):.1f}%); "
        f"popularity fallback: {len(popularity_top12)} items"
    )
    if len(popularity_top12) < config.top_k:
        raise SystemExit(
            f"popularity fallback only found {len(popularity_top12)} items "
            f"but need {config.top_k}; widen --popularity-lookback-days"
        )

    _log(
        f"scoring top-{config.top_k} content preds in "
        f"batches of {config.customer_batch_size}"
    )
    fallback_preds = popularity_top12[: config.top_k]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written, warm_written, fallback_written = _batched_write(
        output_path=output_path,
        target_customers=target_customers,
        customer_queries=customer_queries,
        article_matrix=article_matrix,
        article_ids=article_ids,
        fallback_preds=fallback_preds,
        config=config,
    )
    _log(
        f"wrote {rows_written} rows "
        f"(warm-content={warm_written}, popularity-fallback={fallback_written}) "
        f"in {time.perf_counter() - t_overall:.1f}s"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--embeddings-manifest-path", type=Path, required=True)
    parser.add_argument("--transactions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD; inference cutoff (exclusive)")
    parser.add_argument("--history-lookback-days", type=int, default=90)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--include-content-history", action="store_true")
    parser.add_argument(
        "--customer-batch-size",
        type=int,
        default=4000,
        help="Number of customer queries per matmul batch (memory knob).",
    )
    parser.add_argument(
        "--popularity-lookback-days",
        type=int,
        default=7,
        help="Lookback window for cold-start popularity fallback.",
    )
    return parser


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------


def _load_article_matrix(manifest_path: Path):
    import numpy as np

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dim = int(manifest["dimension"])
    embeddings_path = Path(manifest["embeddings_path"])
    if not embeddings_path.is_absolute():
        embeddings_path = (manifest_path.parent / embeddings_path).resolve()
    ids: list[str] = []
    vectors: list[list[float]] = []
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            v = row["vector"]
            if len(v) != dim:
                raise ValueError(
                    f"article {row['article_id']} dim={len(v)} != manifest dim={dim}"
                )
            ids.append(row["article_id"])
            vectors.append(v)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return ids, matrix


# ---------------------------------------------------------------------------
# Customer order + transaction streaming
# ---------------------------------------------------------------------------


def _load_customer_order(reference_csv: Path) -> list[str]:
    out: list[str] = []
    with reference_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            out.append(row[0])
    return out


def _iter_transactions(path: Path) -> Iterator[tuple[date, str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield (
                date.fromisoformat(row["t_dat"]),
                row["customer_id"],
                row["article_id"],
            )


def _build_queries_and_popularity(
    *,
    transactions_csv: Path,
    target_customers: set[str],
    article_id_to_row: dict[str, int],
    article_matrix,
    config: RunConfig,
):
    """Single transaction-CSV pass producing both customer queries and popularity prior."""

    import numpy as np

    history_cap = max(1, config.max_history_items * 3)
    customer_history: dict[str, deque[tuple[date, str]]] = defaultdict(
        lambda: deque(maxlen=history_cap)
    )
    history_lookback_start = date.fromordinal(
        config.cutoff.toordinal() - config.history_lookback_days
    )
    popularity_lookback_start = date.fromordinal(
        config.cutoff.toordinal() - config.popularity_lookback_days
    )
    popularity_counter: Counter[str] = Counter()
    processed = 0
    for t_dat, customer_id, article_id in _iter_transactions(transactions_csv):
        processed += 1
        if processed % 5_000_000 == 0:
            _log(f"    scanned {processed} transaction rows...")
        if t_dat >= config.cutoff:
            continue
        # Popularity prior: counted regardless of customer membership; the
        # fallback CSV must serve customers we have NEVER seen, so we want
        # popularity over the entire transaction stream within the lookback.
        if t_dat >= popularity_lookback_start:
            popularity_counter[article_id] += 1
        # Per-customer history: only for target customers within the history
        # lookback, and only for articles with FashionCLIP embeddings.
        if t_dat < history_lookback_start:
            continue
        if customer_id not in target_customers:
            continue
        if article_id not in article_id_to_row:
            continue
        customer_history[customer_id].append((t_dat, article_id))

    queries: dict[str, object] = {}
    for customer_id in target_customers:
        items = customer_history.get(customer_id)
        if not items:
            queries[customer_id] = None
            continue
        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
        recent_articles = [a for _, a in sorted_items[: config.max_history_items]]
        if not recent_articles:
            queries[customer_id] = None
            continue
        rows = [article_id_to_row[a] for a in recent_articles]
        mean = article_matrix[rows].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-12:
            queries[customer_id] = None
            continue
        queries[customer_id] = (mean / norm).astype(np.float32)

    popularity_top12 = [a for a, _ in popularity_counter.most_common(max(config.top_k * 2, 24))]
    return queries, popularity_top12


# ---------------------------------------------------------------------------
# Batched matmul writer
# ---------------------------------------------------------------------------


def _batched_write(
    *,
    output_path: Path,
    target_customers: list[str],
    customer_queries: dict[str, object],
    article_matrix,
    article_ids: list[str],
    fallback_preds: list[str],
    config: RunConfig,
) -> tuple[int, int, int]:
    import numpy as np

    fetch_k = (
        config.top_k
        if config.include_history
        else min(config.top_k * 3, article_matrix.shape[1])
    )

    rows_written = 0
    warm_written = 0
    fallback_written = 0
    fallback_string = " ".join(fallback_preds)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("customer_id", "prediction"))

        batch: list[tuple[str, object]] = []
        batch_started_at = time.perf_counter()

        def flush_batch() -> None:
            nonlocal warm_written
            if not batch:
                return
            warm_subset = [(cid, q) for cid, q in batch if q is not None]
            cold_subset = [cid for cid, q in batch if q is None]
            warm_results: dict[str, str] = {}
            if warm_subset:
                queries_np = np.stack([q for _, q in warm_subset], axis=0)
                scores = queries_np @ article_matrix.T
                top_idx_unsorted = np.argpartition(-scores, fetch_k - 1, axis=1)[:, :fetch_k]
                top_scores_unsorted = np.take_along_axis(scores, top_idx_unsorted, axis=1)
                order = np.argsort(-top_scores_unsorted, axis=1)
                top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
                for row, (cid, _) in enumerate(warm_subset):
                    emitted: list[str] = []
                    for idx in top_idx[row]:
                        aid = article_ids[int(idx)]
                        emitted.append(aid)
                        if len(emitted) >= config.top_k:
                            break
                    # Pad with popularity if dedup or filtering left us short.
                    if len(emitted) < config.top_k:
                        for fb in fallback_preds:
                            if fb in emitted:
                                continue
                            emitted.append(fb)
                            if len(emitted) >= config.top_k:
                                break
                    warm_results[cid] = " ".join(emitted[: config.top_k])
                    warm_written += 1
            for cid, q in batch:
                if q is not None:
                    writer.writerow((cid, warm_results[cid]))
                else:
                    writer.writerow((cid, fallback_string))
            # Cold customers were already written above (q is None branch);
            # warm_written is tracked separately.
            del cold_subset  # silence linter; kept for clarity above
            return

        for cid in target_customers:
            batch.append((cid, customer_queries.get(cid)))
            if len(batch) >= config.customer_batch_size:
                flush_batch()
                batch_count = len(batch)
                rows_written += batch_count
                fallback_written += sum(1 for _, q in batch if q is None)
                batch.clear()
                if rows_written % (config.customer_batch_size * 10) == 0:
                    elapsed = time.perf_counter() - batch_started_at
                    _log(
                        f"  written {rows_written}/{len(target_customers)} "
                        f"({100*rows_written/len(target_customers):.1f}%, "
                        f"elapsed {elapsed:.1f}s)"
                    )
        if batch:
            flush_batch()
            rows_written += len(batch)
            fallback_written += sum(1 for _, q in batch if q is None)
            batch.clear()
    return rows_written, warm_written, fallback_written


def _log(message: str) -> None:
    print(f"[content-sub] {message}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
