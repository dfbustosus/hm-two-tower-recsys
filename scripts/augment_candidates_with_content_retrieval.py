"""Append content-similarity retrieval rows to an existing candidate CSV.

Why this script (vs ``hm-recsys export-candidates --content-similarity-manifest-path``):
the official CLI regenerates *all* sources (co-vis, age-segment, garment, etc.)
which dominates wall time. This script is purpose-built to add ONLY the
``text_similarity`` source on top of an already-exported canonical CSV.
It uses a single vectorized BLAS matmul (~1 s for 10k customers x 105k
articles x 512 dims on M4 Pro) and a single streaming pass to append rows.

Pipeline:

1. Load FashionCLIP article embeddings from a JSONL manifest.
2. Build per-customer query vectors from the most recent purchases within
   ``--history-lookback-days``, capped at ``--max-history-items``.
3. Run a single cosine matmul (customers x articles) to score every
   (customer, article) pair, then take top ``--k`` per customer.
4. Append the new rows to the existing CSV with source name ``text_similarity``
   (must match ``TEXT_SIMILARITY_SOURCE`` in :mod:`hm_recsys.retrieval.source_names`).

Customer-history articles are excluded from content candidates by default
to match the published ``--no-include-content-history`` behavior in the
canonical CLI; pass ``--include-content-history`` to keep them.

The output CSV's column schema is the canonical
:data:`CANDIDATE_EXPORT_HEADER` plus any pre-existing optional augmented
columns: those columns are preserved verbatim for original rows and
zero-filled for new content-similarity rows (a deliberate KISS choice;
the LightGBM pipeline already treats missing augmented columns as zero
defaults via :data:`CandidateFeatures.update_from_record`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from math import log1p
from pathlib import Path

from hm_recsys.retrieval.candidate_export import (
    CANDIDATE_EXPORT_HEADER,
    KNOWN_AUGMENTED_COLUMNS,
)
from hm_recsys.retrieval.source_names import TEXT_SIMILARITY_SOURCE


@dataclass(frozen=True)
class AugmentConfig:
    cutoff: date
    history_lookback_days: int
    max_history_items: int
    k: int
    include_history: bool
    popularity_prior_alpha: float
    popularity_prior_lookback_days: int
    source_name: str


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = AugmentConfig(
        cutoff=date.fromisoformat(args.cutoff),
        history_lookback_days=args.history_lookback_days,
        max_history_items=args.max_history_items,
        k=args.k,
        include_history=args.include_content_history,
        popularity_prior_alpha=args.popularity_prior_alpha,
        popularity_prior_lookback_days=args.popularity_prior_lookback_days,
        source_name=args.source_name,
    )

    started = time.perf_counter()
    _log(f"loading article embeddings manifest: {args.embeddings_manifest_path}")
    article_ids, article_matrix = _load_article_embedding_matrix(
        Path(args.embeddings_manifest_path)
    )
    _log(f"loaded {len(article_ids)} article vectors (dim={article_matrix.shape[1]})")

    _log(f"collecting target customers from {args.candidate_csv}")
    target_customers = _collect_target_customers(Path(args.candidate_csv))
    _log(f"target customers: {len(target_customers)}")

    _log("building per-customer query embeddings from transactions")
    (
        customer_queries,
        customer_history_articles,
        article_popularity_weights,
    ) = _build_customer_queries(
        transactions_csv=Path(args.transactions_csv),
        target_customers=target_customers,
        article_id_to_row=_invert(article_ids),
        article_matrix=article_matrix,
        config=config,
    )
    warm = sum(1 for v in customer_queries.values() if v is not None)
    _log(f"warm customers: {warm}/{len(target_customers)}")

    _log("scoring content candidates via single BLAS matmul")
    customer_topk = _score_topk(
        customer_queries=customer_queries,
        article_matrix=article_matrix,
        article_ids=article_ids,
        customer_history=customer_history_articles,
        k=config.k,
        include_history=config.include_history,
        article_popularity_weights=article_popularity_weights,
    )
    _log(
        f"generated content candidates for {sum(1 for v in customer_topk.values() if v)} "
        f"customers"
    )

    _log(f"streaming augmented CSV: {args.output_csv}")
    appended = _stream_augment(
        candidate_csv=Path(args.candidate_csv),
        output_csv=Path(args.output_csv),
        customer_topk=customer_topk,
        source_name=config.source_name,
    )
    metadata_written = _write_augmented_metadata(
        candidate_csv=Path(args.candidate_csv),
        output_csv=Path(args.output_csv),
        appended_rows=appended,
        config=config,
        embeddings_manifest_path=Path(args.embeddings_manifest_path),
    )
    elapsed = time.perf_counter() - started
    _log(f"done in {elapsed:.1f}s: appended {appended} content-similarity rows")
    if metadata_written is not None:
        _log(f"metadata sidecar written: {metadata_written}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--embeddings-manifest-path", type=Path, required=True)
    parser.add_argument("--transactions-csv", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--history-lookback-days", type=int, default=90)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument(
        "--popularity-prior-alpha",
        type=float,
        default=0.0,
        help=(
            "Exponent for multiplicative popularity-prior retrieval. Scores become "
            "cosine * normalized_recent_popularity ** alpha. Use 0.5 for the "
            "Senkin-style popularity-priored content source. Default 0 preserves "
            "raw cosine behavior."
        ),
    )
    parser.add_argument(
        "--popularity-prior-lookback-days",
        type=int,
        default=7,
        help="Pre-cutoff lookback window used to compute popularity priors.",
    )
    parser.add_argument(
        "--source-name",
        default=TEXT_SIMILARITY_SOURCE,
        help="Candidate source name to emit for appended rows.",
    )
    parser.add_argument(
        "--include-content-history",
        action="store_true",
        help="Do NOT filter pre-cutoff history articles from content candidates.",
    )
    return parser


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------


def _load_article_embedding_matrix(manifest_path: Path):
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
            vector = row["vector"]
            if len(vector) != dim:
                raise ValueError(
                    f"vector for {row['article_id']} dim={len(vector)} != {dim}"
                )
            ids.append(row["article_id"])
            vectors.append(vector)

    matrix = np.asarray(vectors, dtype=np.float32)
    # L2-normalize defensively even when the manifest claims ``normalized``.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return ids, matrix


def _invert(article_ids: list[str]) -> dict[str, int]:
    return {a: i for i, a in enumerate(article_ids)}


# ---------------------------------------------------------------------------
# Customer query construction
# ---------------------------------------------------------------------------


def _collect_target_customers(candidate_csv: Path) -> set[str]:
    customers: set[str] = set()
    with candidate_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "customer_id" not in reader.fieldnames:
            raise ValueError(f"missing customer_id in {candidate_csv}")
        for row in reader:
            customers.add(row["customer_id"])
    return customers


def _iter_transactions(path: Path) -> Iterator[tuple[date, str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield (
                date.fromisoformat(row["t_dat"]),
                row["customer_id"],
                row["article_id"],
            )


def _build_customer_queries(
    *,
    transactions_csv: Path,
    target_customers: set[str],
    article_id_to_row: dict[str, int],
    article_matrix,
    config: AugmentConfig,
):
    import numpy as np

    history_cap = max(1, config.max_history_items * 3)
    customer_history: dict[str, deque[tuple[date, str]]] = defaultdict(
        lambda: deque(maxlen=history_cap)
    )
    lookback_start = date.fromordinal(
        config.cutoff.toordinal() - config.history_lookback_days
    )
    if config.popularity_prior_alpha < 0:
        raise ValueError("popularity_prior_alpha must be non-negative")
    if config.popularity_prior_lookback_days <= 0:
        raise ValueError("popularity_prior_lookback_days must be positive")

    popularity_lookback_start = date.fromordinal(
        config.cutoff.toordinal() - config.popularity_prior_lookback_days
    )
    article_popularity_counts: dict[str, int] = defaultdict(int)
    processed = 0
    for t_dat, customer_id, article_id in _iter_transactions(transactions_csv):
        processed += 1
        if processed % 5_000_000 == 0:
            _log(f"  scanned {processed} transaction rows...")
        if t_dat >= config.cutoff:
            continue
        if (
            config.popularity_prior_alpha > 0.0
            and t_dat >= popularity_lookback_start
            and article_id in article_id_to_row
        ):
            article_popularity_counts[article_id] += 1
        if t_dat < lookback_start:
            continue
        if customer_id not in target_customers:
            continue
        if article_id not in article_id_to_row:
            continue
        customer_history[customer_id].append((t_dat, article_id))

    queries: dict[str, object] = {}
    history_articles: dict[str, set[str]] = {}
    for customer_id in target_customers:
        items = customer_history.get(customer_id)
        if not items:
            queries[customer_id] = None
            history_articles[customer_id] = set()
            continue
        sorted_items = sorted(items, key=lambda item: item[0], reverse=True)
        recent_articles = [a for _, a in sorted_items[: config.max_history_items]]
        history_articles[customer_id] = set(recent_articles)
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
    article_popularity_weights = _build_article_popularity_weights(
        article_ids_by_row=article_id_to_row,
        popularity_counts=article_popularity_counts,
        alpha=config.popularity_prior_alpha,
    )
    return queries, history_articles, article_popularity_weights


def _build_article_popularity_weights(
    *,
    article_ids_by_row: dict[str, int],
    popularity_counts: dict[str, int],
    alpha: float,
):
    import numpy as np

    article_count = len(article_ids_by_row)
    if alpha == 0.0:
        return np.ones(article_count, dtype=np.float32)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    weights = np.zeros(article_count, dtype=np.float32)
    if not popularity_counts:
        return weights
    max_log_count = max(log1p(count) for count in popularity_counts.values())
    if max_log_count <= 0.0:
        return weights
    for article_id, count in popularity_counts.items():
        row = article_ids_by_row.get(article_id)
        if row is None:
            continue
        normalized_popularity = log1p(count) / max_log_count
        weights[row] = float(normalized_popularity**alpha)
    return weights


# ---------------------------------------------------------------------------
# Top-K scoring
# ---------------------------------------------------------------------------


def _score_topk(
    *,
    customer_queries: dict[str, object],
    article_matrix,
    article_ids: list[str],
    customer_history: dict[str, set[str]],
    k: int,
    include_history: bool,
    article_popularity_weights,
) -> dict[str, list[tuple[int, str, float]]]:
    import numpy as np

    warm_ids = [cid for cid, q in customer_queries.items() if q is not None]
    if not warm_ids:
        return {cid: [] for cid in customer_queries}
    query_matrix = np.stack(
        [customer_queries[cid] for cid in warm_ids],  # type: ignore[misc]
        axis=0,
    )
    _log(
        f"matmul: queries={query_matrix.shape} x articles={article_matrix.shape} "
        f"= scores={query_matrix.shape[0]}x{article_matrix.shape[0]}"
    )
    # Single big BLAS matmul. Both inputs are L2-normalized so the result
    # is the cosine similarity. Float32 to keep memory linear.
    scores = query_matrix @ article_matrix.T
    if article_popularity_weights is not None:
        popularity_weights = np.asarray(article_popularity_weights, dtype=np.float32)
        if popularity_weights.shape != (article_matrix.shape[0],):
            raise ValueError(
                "article_popularity_weights must align to article_matrix rows; "
                f"got {popularity_weights.shape}, expected {(article_matrix.shape[0],)}"
            )
        scores = scores * popularity_weights.reshape(1, -1)
    _log(f"scores tensor: shape={scores.shape} dtype={scores.dtype}")

    # If we need to over-fetch then filter history, take a larger top
    # candidate pool first.
    fetch_k = k if include_history else min(k * 3, scores.shape[1])
    # ``np.argpartition`` is O(n) per row; much faster than full argsort.
    top_idx_unsorted = np.argpartition(-scores, fetch_k - 1, axis=1)[:, :fetch_k]
    top_scores_unsorted = np.take_along_axis(scores, top_idx_unsorted, axis=1)
    # Sort the candidate pool descending per row.
    order = np.argsort(-top_scores_unsorted, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
    top_scores = np.take_along_axis(top_scores_unsorted, order, axis=1)

    out: dict[str, list[tuple[int, str, float]]] = {cid: [] for cid in customer_queries}
    for row, cid in enumerate(warm_ids):
        history = customer_history.get(cid, set())
        emitted: list[tuple[int, str, float]] = []
        rank = 1
        for idx, score in zip(top_idx[row], top_scores[row], strict=False):
            aid = article_ids[int(idx)]
            if (not include_history) and aid in history:
                continue
            emitted.append((rank, aid, float(score)))
            rank += 1
            if rank > k:
                break
        out[cid] = emitted
    return out


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _stream_augment(
    *,
    candidate_csv: Path,
    output_csv: Path,
    customer_topk: dict[str, list[tuple[int, str, float]]],
    source_name: str,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with (
        candidate_csv.open("r", encoding="utf-8", newline="") as input_handle,
        output_csv.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty candidate CSV: {candidate_csv}")
        input_fields = list(reader.fieldnames)
        _validate_header(input_fields)
        writer = csv.DictWriter(output_handle, fieldnames=input_fields)
        writer.writeheader()
        # Stream all original rows verbatim.
        last_customer: str | None = None
        for row in reader:
            customer_id = row["customer_id"]
            if last_customer is not None and customer_id != last_customer:
                # Customer boundary in original CSV - append content-sim rows for previous customer.
                appended += _emit_content_rows(
                    writer=writer,
                    customer_id=last_customer,
                    content_rows=customer_topk.get(last_customer, []),
                    input_fields=input_fields,
                    source_name=source_name,
                )
            writer.writerow(row)
            last_customer = customer_id
        # Flush the trailing customer's content rows.
        if last_customer is not None:
            appended += _emit_content_rows(
                writer=writer,
                customer_id=last_customer,
                content_rows=customer_topk.get(last_customer, []),
                input_fields=input_fields,
                source_name=source_name,
            )
    return appended


def _emit_content_rows(
    *,
    writer: csv.DictWriter,
    customer_id: str,
    content_rows: list[tuple[int, str, float]],
    input_fields: list[str],
    source_name: str,
) -> int:
    if not content_rows:
        return 0
    augmented_columns = [c for c in input_fields if c in KNOWN_AUGMENTED_COLUMNS]
    written = 0
    for rank, article_id, score in content_rows:
        record = {
            "customer_id": customer_id,
            "article_id": article_id,
            "source": source_name,
            "source_rank": rank,
            "source_score": f"{score:.6f}",
        }
        # Zero-fill any pre-existing optional augmented columns.
        for column in augmented_columns:
            record[column] = "0"
        writer.writerow(record)
        written += 1
    return written


def _validate_header(input_fields: list[str]) -> None:
    canonical = list(CANDIDATE_EXPORT_HEADER)
    if input_fields[: len(canonical)] != canonical:
        raise ValueError(
            f"input CSV header must start with {','.join(canonical)}; got "
            f"{','.join(input_fields)}"
        )
    trailing = input_fields[len(canonical) :]
    for column in trailing:
        if column not in KNOWN_AUGMENTED_COLUMNS:
            raise ValueError(
                f"input CSV has unknown trailing column {column!r}; known "
                f"augmented columns: {','.join(KNOWN_AUGMENTED_COLUMNS)}"
            )


def _write_augmented_metadata(
    *,
    candidate_csv: Path,
    output_csv: Path,
    appended_rows: int,
    config: AugmentConfig,
    embeddings_manifest_path: Path,
) -> Path | None:
    metadata_path = candidate_csv.with_suffix(".json")
    if not metadata_path.exists():
        _log(f"input metadata sidecar not found; skipping JSON sidecar: {metadata_path}")
        return None
    output_metadata_path = output_csv.with_suffix(".json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_counts = dict(payload.get("source_row_counts") or {})
    source_counts[config.source_name] = source_counts.get(config.source_name, 0) + appended_rows
    payload["source_row_counts"] = dict(sorted(source_counts.items()))
    if isinstance(payload.get("rows_written"), int):
        payload["rows_written"] = int(payload["rows_written"]) + appended_rows
    payload["output_path"] = str(output_csv)
    payload["content_similarity_manifest_path"] = str(
        embeddings_manifest_path.expanduser().resolve()
    )
    payload["content_similarity_source_name"] = config.source_name
    payload["content_similarity_max_history_items"] = config.max_history_items
    payload["content_similarity_popularity_prior_weight"] = None
    payload["content_similarity_popularity_lookback_days"] = (
        config.popularity_prior_lookback_days
        if config.popularity_prior_alpha > 0.0
        else None
    )
    payload["augmentation"] = {
        "kind": "multiplicative_content_popularity_prior",
        "source_name": config.source_name,
        "history_lookback_days": config.history_lookback_days,
        "max_history_items": config.max_history_items,
        "k": config.k,
        "include_history": config.include_history,
        "popularity_prior_alpha": config.popularity_prior_alpha,
        "popularity_prior_lookback_days": config.popularity_prior_lookback_days,
        "embeddings_manifest_path": str(embeddings_manifest_path.expanduser().resolve()),
        "appended_rows": appended_rows,
    }
    output_metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_metadata_path


def _log(message: str) -> None:
    print(f"[content-retrieval] {message}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
