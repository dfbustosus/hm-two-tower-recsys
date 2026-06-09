"""Perfect-ranker (oracle) ceiling on a fixed candidate set.

The perfect-ranker ceiling answers a single auditable question: given the
candidate articles currently emitted by our retrieval stack, what is the
maximum MAP@K an *omniscient* ranker could achieve if it sorted the labels
to the top? The ceiling decomposes the MAP@K gap into

* an unreachable component caused by candidate-coverage loss
  (labels missing from the candidate set), and
* a reachable component caused by ranker quality
  (labels present in the candidate set but misranked).

This decomposition lets Phase 0/1/2 of the plan decide where to invest:
ranker tuning when the ceiling is far above the achieved score, and richer
retrieval when the ceiling itself is too low.

The implementation is intentionally streaming-friendly so it can process the
multi-million-row candidate CSVs without loading them entirely into memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hm_recsys.evaluation.metrics import dedupe_preserve_order
from hm_recsys.ranking.deterministic import iter_candidate_records_from_csv
from hm_recsys.retrieval.candidate_export import CandidateRecord

__all__ = (
    "PerCustomerOracle",
    "PerfectRankerCeiling",
    "PerfectRankerCeilingReport",
    "PerfectRankerCutoffInput",
    "build_perfect_ranker_ceiling",
    "build_perfect_ranker_ceiling_report",
    "compute_perfect_ranker_ceiling",
    "iter_candidates_grouped_by_customer",
    "load_perfect_ranker_ceiling_report",
    "oracle_average_precision_at_k",
    "oracle_recall_at_k",
    "render_perfect_ranker_ceiling_markdown",
    "write_perfect_ranker_ceiling_markdown",
    "write_perfect_ranker_ceiling_report",
)


@dataclass(frozen=True)
class PerfectRankerCutoffInput:
    """Per-cutoff input bundle for the perfect-ranker ceiling computation.

    Attributes:
        cutoff: Evaluation cutoff date string (``YYYY-MM-DD``).
        candidate_path: Resolved path to the ranker-ready candidate CSV.
        validation_labels: Validation labels keyed by customer ID for this
            cutoff. Customers with no labels are dropped to mirror Kaggle's
            evaluation contract.
    """

    cutoff: str
    candidate_path: Path
    validation_labels: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class PerCustomerOracle:
    """Per-customer oracle decomposition for a single evaluation customer.

    Attributes:
        customer_id: H&M customer identifier.
        label_count: Number of distinct labels for the customer.
        candidate_count: Number of distinct candidate articles for the customer.
        reachable_label_count: ``|labels ∩ candidates|`` (capped to ``k``).
        oracle_average_precision_at_k: Best AP@K achievable by sorting reachable
            labels into the top positions.
        oracle_recall_at_k: ``reachable_label_count / label_count``.
    """

    customer_id: str
    label_count: int
    candidate_count: int
    reachable_label_count: int
    oracle_average_precision_at_k: float
    oracle_recall_at_k: float


@dataclass(frozen=True)
class PerfectRankerCeiling:
    """Aggregate perfect-ranker ceiling metrics for one cutoff.

    Attributes:
        cutoff: Evaluation cutoff date string.
        candidate_path: Resolved candidate CSV path used.
        k: Recommendation depth.
        evaluated_customers: Number of customers with ≥1 label that were
            present in the candidate CSV (denominator for the mean MAP/recall).
        labeled_customers: Number of customers with ≥1 label in the validation
            labels (before intersecting with the candidate CSV).
        customers_in_candidates: Number of customers that appear in the
            candidate CSV (whether or not they have labels).
        customers_without_any_candidate: Labeled customers that have *zero*
            candidate articles. These customers contribute ``0.0`` to the
            ceiling and are the strongest signal that retrieval is incomplete.
        mean_oracle_map_at_k: Mean AP@K oracle across evaluated customers.
        mean_oracle_recall_at_k: Mean Recall@K oracle across evaluated
            customers.
        candidate_label_coverage: Fraction of total labels present in the
            candidate set across evaluated customers. Distinct from
            ``mean_oracle_recall_at_k`` because it ignores the ``k`` cap.
        mean_candidate_count: Mean number of distinct candidates per evaluated
            customer.
        mean_label_count: Mean number of distinct labels per evaluated
            customer.
        candidate_rows: Total raw rows scanned in the candidate CSV.
    """

    cutoff: str
    candidate_path: str
    k: int
    evaluated_customers: int
    labeled_customers: int
    customers_in_candidates: int
    customers_without_any_candidate: int
    mean_oracle_map_at_k: float
    mean_oracle_recall_at_k: float
    candidate_label_coverage: float
    mean_candidate_count: float
    mean_label_count: float
    candidate_rows: int


@dataclass(frozen=True)
class PerfectRankerCeilingReport:
    """Multi-cutoff perfect-ranker ceiling report.

    Attributes:
        generated_at_utc: UTC timestamp for the report (``isoformat`` seconds).
        cutoffs: Evaluation cutoffs included.
        k: Recommendation depth used across cutoffs.
        per_cutoff: Per-cutoff ceiling metrics.
        mean_oracle_map_at_k: Arithmetic mean of per-cutoff oracle MAP@K.
        min_oracle_map_at_k: Minimum per-cutoff oracle MAP@K.
        max_oracle_map_at_k: Maximum per-cutoff oracle MAP@K.
        mean_oracle_recall_at_k: Arithmetic mean of per-cutoff oracle recall.
        mean_candidate_label_coverage: Arithmetic mean of per-cutoff candidate
            label coverage (label-set intersection, unbounded by ``k``).
        warnings: Sorted unique advisory strings (e.g. small-sample warnings).
    """

    generated_at_utc: str
    cutoffs: tuple[str, ...]
    k: int
    per_cutoff: tuple[PerfectRankerCeiling, ...]
    mean_oracle_map_at_k: float
    min_oracle_map_at_k: float
    max_oracle_map_at_k: float
    mean_oracle_recall_at_k: float
    mean_candidate_label_coverage: float
    warnings: tuple[str, ...]


def oracle_average_precision_at_k(reachable_label_count: int, label_count: int, k: int) -> float:
    """Return the oracle AP@K for one customer.

    Reordering reachable labels into the first ``min(reachable_label_count, k)``
    positions yields a closed-form AP@K because every contribution is
    ``rank / rank = 1.0``.

    Args:
        reachable_label_count: ``|labels ∩ candidates|``.
        label_count: Total distinct labels for the customer.
        k: Recommendation depth.

    Returns:
        Oracle AP@K. Returns ``0.0`` if ``label_count`` is zero.

    Raises:
        ValueError: If ``k`` is not positive or ``reachable_label_count`` is
            negative or exceeds ``label_count``.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if reachable_label_count < 0:
        raise ValueError("reachable_label_count must be non-negative")
    if reachable_label_count > label_count:
        raise ValueError("reachable_label_count cannot exceed label_count")
    if label_count == 0:
        return 0.0
    hits = min(reachable_label_count, k)
    denominator = min(label_count, k)
    return hits / denominator


def oracle_recall_at_k(reachable_label_count: int, label_count: int, k: int) -> float:
    """Return the oracle recall@K for one customer.

    Args:
        reachable_label_count: ``|labels ∩ candidates|``.
        label_count: Total distinct labels for the customer.
        k: Recommendation depth.

    Returns:
        Oracle recall@K, or ``0.0`` if ``label_count`` is zero.

    Raises:
        ValueError: If ``k`` is not positive or ``reachable_label_count`` is
            negative or exceeds ``label_count``.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if reachable_label_count < 0:
        raise ValueError("reachable_label_count must be non-negative")
    if reachable_label_count > label_count:
        raise ValueError("reachable_label_count cannot exceed label_count")
    if label_count == 0:
        return 0.0
    return min(reachable_label_count, k) / label_count


def iter_candidates_grouped_by_customer(
    records: Iterable[CandidateRecord],
) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Group sorted candidate records by customer and emit deduplicated articles.

    The ranker-ready candidate CSVs are written customer-by-customer, so a
    single linear scan is sufficient to materialize each customer's distinct
    candidate list without buffering the whole file.

    Args:
        records: Candidate records ordered by customer ID.

    Yields:
        Tuples of ``(customer_id, distinct_article_ids_in_first_seen_order)``.

    Raises:
        ValueError: If two non-contiguous customer-ID groups are detected,
            which would indicate an unsorted candidate CSV.
    """

    current_customer: str | None = None
    current_articles: list[str] = []
    seen_customers: set[str] = set()
    for record in records:
        if current_customer is None:
            current_customer = record.customer_id
        if record.customer_id != current_customer:
            yield current_customer, dedupe_preserve_order(current_articles)
            if record.customer_id in seen_customers:
                raise ValueError(
                    "candidate CSV must be sorted by customer_id; "
                    f"customer {record.customer_id!r} reappears non-contiguously"
                )
            seen_customers.add(current_customer)
            current_customer = record.customer_id
            current_articles = []
        current_articles.append(record.article_id)
    if current_customer is not None:
        yield current_customer, dedupe_preserve_order(current_articles)


def _scan_candidates(candidate_path: Path) -> tuple[dict[str, tuple[str, ...]], int]:
    """Stream a candidate CSV into ``{customer_id: distinct_articles}``.

    Args:
        candidate_path: Ranker-ready candidate CSV path.

    Returns:
        Tuple of ``(candidates_by_customer, candidate_rows_scanned)``.
    """

    records = list(iter_candidate_records_from_csv(candidate_path))
    candidates_by_customer: dict[str, tuple[str, ...]] = {}
    for customer_id, articles in iter_candidates_grouped_by_customer(iter(records)):
        candidates_by_customer[customer_id] = articles
    return candidates_by_customer, len(records)


def compute_perfect_ranker_ceiling(
    candidate_path: Path | str,
    validation_labels: Mapping[str, Iterable[str]],
    *,
    cutoff: str,
    k: int = 12,
) -> tuple[PerfectRankerCeiling, tuple[PerCustomerOracle, ...]]:
    """Compute the perfect-ranker ceiling for one cutoff.

    Args:
        candidate_path: Candidate CSV produced by the retrieval stage.
        validation_labels: Validation labels keyed by customer ID.
        cutoff: Evaluation cutoff identifier (``YYYY-MM-DD``).
        k: Recommendation depth.

    Returns:
        Tuple of ``(aggregate ceiling, per-customer oracle records)``.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    resolved_path = Path(candidate_path).expanduser().resolve()
    candidates_by_customer, candidate_rows = _scan_candidates(resolved_path)

    labels_by_customer: dict[str, tuple[str, ...]] = {}
    for customer_id, labels in validation_labels.items():
        deduped = dedupe_preserve_order(labels)
        if deduped:
            labels_by_customer[customer_id] = deduped

    per_customer: list[PerCustomerOracle] = []
    customers_without_any_candidate = 0
    total_labels = 0
    reachable_labels = 0
    sum_candidate_count = 0
    sum_label_count = 0
    for customer_id, labels in labels_by_customer.items():
        candidates = candidates_by_customer.get(customer_id, ())
        if not candidates:
            customers_without_any_candidate += 1
        reachable = len(set(labels) & set(candidates))
        oracle_map = oracle_average_precision_at_k(
            reachable_label_count=reachable, label_count=len(labels), k=k
        )
        oracle_recall = oracle_recall_at_k(
            reachable_label_count=reachable, label_count=len(labels), k=k
        )
        per_customer.append(
            PerCustomerOracle(
                customer_id=customer_id,
                label_count=len(labels),
                candidate_count=len(candidates),
                reachable_label_count=reachable,
                oracle_average_precision_at_k=oracle_map,
                oracle_recall_at_k=oracle_recall,
            )
        )
        total_labels += len(labels)
        reachable_labels += reachable
        sum_candidate_count += len(candidates)
        sum_label_count += len(labels)

    evaluated_customers = len(per_customer)
    if evaluated_customers == 0:
        mean_oracle_map = 0.0
        mean_oracle_recall = 0.0
        mean_candidate_count = 0.0
        mean_label_count = 0.0
        candidate_label_coverage = 0.0
    else:
        mean_oracle_map = (
            sum(record.oracle_average_precision_at_k for record in per_customer)
            / evaluated_customers
        )
        mean_oracle_recall = (
            sum(record.oracle_recall_at_k for record in per_customer) / evaluated_customers
        )
        mean_candidate_count = sum_candidate_count / evaluated_customers
        mean_label_count = sum_label_count / evaluated_customers
        candidate_label_coverage = reachable_labels / total_labels if total_labels else 0.0

    ceiling = PerfectRankerCeiling(
        cutoff=cutoff,
        candidate_path=str(resolved_path),
        k=k,
        evaluated_customers=evaluated_customers,
        labeled_customers=len(labels_by_customer),
        customers_in_candidates=len(candidates_by_customer),
        customers_without_any_candidate=customers_without_any_candidate,
        mean_oracle_map_at_k=mean_oracle_map,
        mean_oracle_recall_at_k=mean_oracle_recall,
        candidate_label_coverage=candidate_label_coverage,
        mean_candidate_count=mean_candidate_count,
        mean_label_count=mean_label_count,
        candidate_rows=candidate_rows,
    )
    return ceiling, tuple(per_customer)


def build_perfect_ranker_ceiling(
    inputs: Sequence[PerfectRankerCutoffInput], *, k: int = 12
) -> tuple[PerfectRankerCeilingReport, dict[str, tuple[PerCustomerOracle, ...]]]:
    """Compute per-cutoff ceilings and aggregate them into a report.

    Args:
        inputs: One bundle per evaluation cutoff.
        k: Recommendation depth.

    Returns:
        Tuple of ``(aggregate report, per-cutoff per-customer oracle records)``.

    Raises:
        ValueError: If ``inputs`` is empty, contains duplicate cutoffs, or
            ``k`` is not positive.
    """

    if not inputs:
        raise ValueError("at least one cutoff input is required")
    if k <= 0:
        raise ValueError("k must be positive")
    cutoffs = tuple(item.cutoff for item in inputs)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoff inputs must have unique cutoff dates")

    per_cutoff: list[PerfectRankerCeiling] = []
    per_customer_by_cutoff: dict[str, tuple[PerCustomerOracle, ...]] = {}
    warnings: set[str] = set()
    for bundle in inputs:
        ceiling, per_customer = compute_perfect_ranker_ceiling(
            candidate_path=bundle.candidate_path,
            validation_labels=bundle.validation_labels,
            cutoff=bundle.cutoff,
            k=k,
        )
        per_cutoff.append(ceiling)
        per_customer_by_cutoff[bundle.cutoff] = per_customer
        if ceiling.evaluated_customers < 100:
            warnings.add("small_sample_evaluation_lt_100_customers")
        if ceiling.customers_without_any_candidate > 0:
            warnings.add("customers_without_any_candidate_present")

    oracle_maps = tuple(ceiling.mean_oracle_map_at_k for ceiling in per_cutoff)
    oracle_recalls = tuple(ceiling.mean_oracle_recall_at_k for ceiling in per_cutoff)
    label_coverages = tuple(ceiling.candidate_label_coverage for ceiling in per_cutoff)

    report = PerfectRankerCeilingReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoffs=cutoffs,
        k=k,
        per_cutoff=tuple(per_cutoff),
        mean_oracle_map_at_k=sum(oracle_maps) / len(oracle_maps),
        min_oracle_map_at_k=min(oracle_maps),
        max_oracle_map_at_k=max(oracle_maps),
        mean_oracle_recall_at_k=sum(oracle_recalls) / len(oracle_recalls),
        mean_candidate_label_coverage=sum(label_coverages) / len(label_coverages),
        warnings=tuple(sorted(warnings)),
    )
    return report, per_customer_by_cutoff


def build_perfect_ranker_ceiling_report(
    inputs: Sequence[PerfectRankerCutoffInput], *, k: int = 12
) -> PerfectRankerCeilingReport:
    """Compute the aggregate perfect-ranker ceiling report only.

    A thin wrapper for callers that do not need the per-customer breakdown.

    Args:
        inputs: One bundle per evaluation cutoff.
        k: Recommendation depth.

    Returns:
        Aggregate perfect-ranker ceiling report.
    """

    report, _ = build_perfect_ranker_ceiling(inputs, k=k)
    return report


def perfect_ranker_ceiling_report_to_dict(
    report: PerfectRankerCeilingReport,
) -> dict[str, Any]:
    """Convert a perfect-ranker ceiling report to JSON-serializable primitives.

    Args:
        report: Report object to convert.

    Returns:
        Dictionary suitable for ``json.dumps``.
    """

    return asdict(report)


def write_perfect_ranker_ceiling_report(
    report: PerfectRankerCeilingReport, path: Path | str
) -> Path:
    """Write a perfect-ranker ceiling report as deterministic JSON.

    Args:
        report: Ceiling report to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(perfect_ranker_ceiling_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def load_perfect_ranker_ceiling_report(path: Path | str) -> PerfectRankerCeilingReport:
    """Read a previously written ceiling report from JSON.

    Args:
        path: JSON path produced by :func:`write_perfect_ranker_ceiling_report`.

    Returns:
        Parsed perfect-ranker ceiling report.
    """

    parsed = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    per_cutoff = tuple(
        PerfectRankerCeiling(
            cutoff=item["cutoff"],
            candidate_path=item["candidate_path"],
            k=item["k"],
            evaluated_customers=item["evaluated_customers"],
            labeled_customers=item["labeled_customers"],
            customers_in_candidates=item["customers_in_candidates"],
            customers_without_any_candidate=item["customers_without_any_candidate"],
            mean_oracle_map_at_k=item["mean_oracle_map_at_k"],
            mean_oracle_recall_at_k=item["mean_oracle_recall_at_k"],
            candidate_label_coverage=item["candidate_label_coverage"],
            mean_candidate_count=item["mean_candidate_count"],
            mean_label_count=item["mean_label_count"],
            candidate_rows=item["candidate_rows"],
        )
        for item in parsed["per_cutoff"]
    )
    return PerfectRankerCeilingReport(
        generated_at_utc=parsed["generated_at_utc"],
        cutoffs=tuple(parsed["cutoffs"]),
        k=parsed["k"],
        per_cutoff=per_cutoff,
        mean_oracle_map_at_k=parsed["mean_oracle_map_at_k"],
        min_oracle_map_at_k=parsed["min_oracle_map_at_k"],
        max_oracle_map_at_k=parsed["max_oracle_map_at_k"],
        mean_oracle_recall_at_k=parsed["mean_oracle_recall_at_k"],
        mean_candidate_label_coverage=parsed["mean_candidate_label_coverage"],
        warnings=tuple(parsed["warnings"]),
    )


def render_perfect_ranker_ceiling_markdown(report: PerfectRankerCeilingReport) -> str:
    """Render a perfect-ranker ceiling report as Markdown.

    Args:
        report: Ceiling report to render.

    Returns:
        Markdown body suitable for writing to ``.md`` artifacts.
    """

    lines: list[str] = [
        "# Perfect-Ranker (Oracle) Ceiling",
        "",
        f"- Generated: `{report.generated_at_utc}`",
        f"- Recommendation depth: `k = {report.k}`",
        f"- Cutoffs evaluated: `{', '.join(report.cutoffs)}`",
        (
            f"- Mean oracle MAP@K: `{report.mean_oracle_map_at_k:.5f}` "
            f"(min `{report.min_oracle_map_at_k:.5f}`, max `{report.max_oracle_map_at_k:.5f}`)"
        ),
        f"- Mean oracle Recall@K: `{report.mean_oracle_recall_at_k:.5f}`",
        f"- Mean candidate label coverage: `{report.mean_candidate_label_coverage:.5f}`",
        "",
    ]
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in report.warnings)
        lines.append("")

    lines.extend(
        [
            "## Per-cutoff ceiling",
            "",
            "| Cutoff | Evaluated | No candidates | Oracle MAP@K | Oracle Recall@K"
            " | Label coverage | Mean candidates/cust | Mean labels/cust |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ceiling in report.per_cutoff:
        lines.append(
            "| "
            + " | ".join(
                [
                    ceiling.cutoff,
                    str(ceiling.evaluated_customers),
                    str(ceiling.customers_without_any_candidate),
                    f"{ceiling.mean_oracle_map_at_k:.5f}",
                    f"{ceiling.mean_oracle_recall_at_k:.5f}",
                    f"{ceiling.candidate_label_coverage:.5f}",
                    f"{ceiling.mean_candidate_count:.2f}",
                    f"{ceiling.mean_label_count:.2f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.extend(["## Candidate paths", ""])
    for ceiling in report.per_cutoff:
        lines.append(f"- `{ceiling.cutoff}` → `{ceiling.candidate_path}`")
    lines.append("")
    return "\n".join(lines)


def write_perfect_ranker_ceiling_markdown(
    report: PerfectRankerCeilingReport, path: Path | str
) -> Path:
    """Write the Markdown rendering of a perfect-ranker ceiling report.

    Args:
        report: Ceiling report to render.
        path: Destination Markdown path.

    Returns:
        Resolved path written to disk.
    """

    markdown_path = Path(path).expanduser().resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_perfect_ranker_ceiling_markdown(report), encoding="utf-8")
    return markdown_path
