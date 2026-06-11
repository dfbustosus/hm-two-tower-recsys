"""Weighted Reciprocal Rank Fusion (RRF) blender for Kaggle submission CSVs.

For each candidate (customer, article), the fused score is

    score = sum_m weight_m / (k_offset + rank_m_in_model)

where ``rank_m_in_model`` is the 1-indexed position of the article in model
``m``'s top-12 prediction for the customer (or ``inf`` if missing, giving
zero contribution). Top-12 by aggregated score becomes the final prediction.

Why weighted RRF (vs simple RRF / arithmetic-rank average):
- The previous unweighted ensemble produced LB 0.01846 - WORSE than every
  input - because it gave equal vote to a weak two-tower-only submission
  (LB ~0.015). Performance-proportional weights prevent that failure mode.
- RRF is monotone in ranks (immune to score scale), uses every model's
  full top-12, and degrades gracefully if a model omits an article: the
  weak model can NUDGE but cannot OVERRIDE a strong model's consensus.
- ``k_offset = 60`` is the long-standing Cormack et al. (2009) default.

Tie-break: within equal aggregated RRF score, articles are sorted by the
SUM of (weight_m * presence_m), then by lowest min(rank). This favors
broader consensus and avoids non-deterministic dict-order tie-breaking.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInput:
    name: str
    path: Path
    weight: float


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if len(args.model_paths) != len(args.model_weights):
        raise SystemExit(
            "--model-path and --model-weight must be provided the same number "
            f"of times (got {len(args.model_paths)} paths and "
            f"{len(args.model_weights)} weights)"
        )
    if any(w <= 0 for w in args.model_weights):
        raise SystemExit("--model-weight must be strictly positive")

    models = [
        ModelInput(
            name=path.name.replace(".csv", ""), path=path, weight=weight,
        )
        for path, weight in zip(args.model_paths, args.model_weights, strict=True)
    ]
    _log(f"blending {len(models)} models with RRF k_offset={args.k_offset}")
    for m in models:
        _log(f"  model: weight={m.weight:.2f}  path={m.path}")

    iterators = [_iter_model_rows(m.path) for m in models]
    customer_order = _load_customer_order(args.reference_csv)
    _log(f"reference customer order: {len(customer_order)} customers from {args.reference_csv}")

    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("customer_id", "prediction"))
        # All model CSVs MUST share the same customer order as the reference
        # so we can stream them in lockstep. Validated by the assertion below.
        current_rows = [next(it) for it in iterators]
        for customer_id in customer_order:
            per_model_rows: list[tuple[str, list[str]]] = []
            for model_idx, (cust, preds) in enumerate(current_rows):
                if cust != customer_id:
                    raise ValueError(
                        f"customer order mismatch in model "
                        f"{models[model_idx].name}: expected {customer_id} got {cust}"
                    )
                per_model_rows.append((cust, preds))
                try:
                    current_rows[model_idx] = next(iterators[model_idx])
                except StopIteration:
                    # Sentinel so the next outer-loop step trips the mismatch
                    # check rather than silently exhausting; in practice all
                    # models have ~1.37M rows so this only fires on bugs.
                    current_rows[model_idx] = ("__exhausted__", [])
            fused = _fuse_one_customer(
                customer_id=customer_id,
                per_model_rows=per_model_rows,
                weights=[m.weight for m in models],
                k_offset=args.k_offset,
                top_k=args.top_k,
            )
            writer.writerow((customer_id, " ".join(fused)))
            written += 1
            if written % 200_000 == 0:
                _log(f"  fused {written}/{len(customer_order)} customers...")

    _log(f"wrote {written} rows to {output_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        action="append",
        dest="model_paths",
        type=Path,
        required=True,
        help="Path to a Kaggle-shaped submission CSV. Repeat for each model.",
    )
    parser.add_argument(
        "--model-weight",
        action="append",
        dest="model_weights",
        type=float,
        required=True,
        help="Positive RRF weight for the corresponding --model-path. Repeat.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
        help="CSV whose customer_id column defines the output row order.",
    )
    parser.add_argument("--k-offset", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=12)
    return parser


def _iter_model_rows(path: Path) -> Iterator[tuple[str, list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["customer_id", "prediction"]:
            raise ValueError(
                f"{path} has unexpected header {header}; expected "
                "['customer_id','prediction']"
            )
        for row in reader:
            yield row[0], row[1].split()


def _load_customer_order(reference_csv: Path) -> list[str]:
    out: list[str] = []
    with reference_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            out.append(row[0])
    return out


def _fuse_one_customer(
    *,
    customer_id: str,
    per_model_rows: list[tuple[str, list[str]]],
    weights: list[float],
    k_offset: int,
    top_k: int,
) -> list[str]:
    del customer_id  # unused; kept for clearer call-site diagnostics
    # Aggregate weighted RRF scores + secondary keys.
    scores: dict[str, float] = {}
    consensus: dict[str, float] = {}
    min_rank: dict[str, int] = {}
    for weight, (_, preds) in zip(weights, per_model_rows, strict=True):
        for rank_zero, article_id in enumerate(preds):
            rank = rank_zero + 1
            scores[article_id] = scores.get(article_id, 0.0) + weight / (k_offset + rank)
            consensus[article_id] = consensus.get(article_id, 0.0) + weight
            existing = min_rank.get(article_id, math.inf)
            if rank < existing:
                min_rank[article_id] = rank
    ranked = sorted(
        scores.keys(),
        # Higher fused score, higher total weight-presence, lower min rank.
        key=lambda a: (-scores[a], -consensus[a], min_rank.get(a, math.inf), a),
    )
    return ranked[:top_k]


def _log(message: str) -> None:
    print(f"[rrf-blend] {message}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
