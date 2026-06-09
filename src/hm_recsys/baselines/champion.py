"""Baseline-candidate registry used to pin the MAP@12 reproducibility champion."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

DEFAULT_BASELINE_TARGET_MAP_AT_K: float = 0.02205
"""The Kaggle public leaderboard score the gap-closure plan must reproduce."""

DEFAULT_BASELINE_TARGET_TOLERANCE: float = 0.001
"""Tolerance (in absolute MAP@12 units) within which a candidate is considered a match."""


@dataclass(frozen=True)
class OfflineRollingMetrics:
    """Locally-measured MAP@K across rolling validation cutoffs.

    Attributes:
        cutoffs: Evaluation cutoff dates in ascending order.
        per_cutoff_map_at_k: MAP@K keyed by cutoff date.
        mean_map_at_k: Arithmetic mean across cutoffs.
        min_map_at_k: Worst per-cutoff score.
        max_map_at_k: Best per-cutoff score.
        k: Recommendation depth used for MAP.
        candidate_k: Maximum candidates per source used to build features.
        rolling_report_path: JSON report this metrics block was derived from.
    """

    cutoffs: tuple[str, ...]
    per_cutoff_map_at_k: dict[str, float]
    mean_map_at_k: float
    min_map_at_k: float
    max_map_at_k: float
    k: int
    candidate_k: int
    rolling_report_path: str


@dataclass(frozen=True)
class BaselineCandidate:
    """One ranker configuration we are considering as the champion baseline.

    Attributes:
        name: Human-readable identifier (printed in summaries and reports).
        ranker_kind: One of ``deterministic``, ``learned_linear``,
            ``lightgbm_behavioral``. Other strings are accepted but reserved.
        config_hash: Short canonical-JSON SHA-256 digest of ``config``.
        config: Full configuration captured for reproducibility.
        offline_metrics: Locally measured rolling metrics (when available).
        submission_paths: Submission CSVs generated from this candidate.
        leaderboard_public_map_at_k: User-supplied Kaggle public-LB MAP@12.
        leaderboard_private_map_at_k: User-supplied Kaggle private-LB MAP@12.
        submitted_at_utc: Timestamp the user uploaded to Kaggle (manual).
        kaggle_submission_message: Optional Kaggle message text recorded by user.
        notes: Free-form notes (e.g. why this candidate exists, caveats).
    """

    name: str
    ranker_kind: str
    config_hash: str
    config: dict[str, Any]
    offline_metrics: OfflineRollingMetrics | None = None
    submission_paths: tuple[str, ...] = ()
    leaderboard_public_map_at_k: float | None = None
    leaderboard_private_map_at_k: float | None = None
    submitted_at_utc: str | None = None
    kaggle_submission_message: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class BaselineChampionReport:
    """Champion-pinning report combining all known baseline candidates.

    Attributes:
        generated_at_utc: UTC timestamp when the report was rendered.
        target_leaderboard_map_at_k: The Kaggle LB number to reproduce
            (defaults to the user-asserted 0.02205).
        target_tolerance: Absolute MAP@12 tolerance for the match check.
        candidates: All registered baseline candidates.
        champion_index: Index into ``candidates`` of the chosen champion,
            or ``None`` when no candidate is eligible.
        champion_rationale: One-line explanation of the champion choice.
        warnings: Plan-level warnings (e.g. missing LB data).
    """

    generated_at_utc: str
    target_leaderboard_map_at_k: float
    target_tolerance: float
    candidates: tuple[BaselineCandidate, ...]
    champion_index: int | None
    champion_rationale: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Discovery


def discover_rolling_validation_candidates(
    rolling_reports_dir: Path | str,
    cutoffs: tuple[str, ...] | None = None,
    glob_pattern: str = "rolling_linear_ranker_*.json",
) -> tuple[BaselineCandidate, ...]:
    """Discover learned-linear baseline candidates from rolling-validation reports.

    The current codebase only emits rolling reports for the learned linear
    ranker, but each report also records the source-order and deterministic
    MAP@K on the same windows. We surface all three as separate candidates
    so the champion-selection logic has a complete picture.

    Args:
        rolling_reports_dir: Directory containing rolling-validation JSON.
        cutoffs: Optional cutoff filter; reports whose cutoffs do not match
            exactly are skipped.
        glob_pattern: Glob applied within ``rolling_reports_dir``.

    Returns:
        Tuple of :class:`BaselineCandidate` instances in deterministic order.
    """

    directory = Path(rolling_reports_dir).expanduser().resolve()
    if not directory.exists():
        return ()
    candidates: list[BaselineCandidate] = []
    for report_path in sorted(directory.glob(glob_pattern)):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        report_cutoffs = tuple(payload.get("cutoffs", ()))
        if cutoffs is not None and report_cutoffs != cutoffs:
            continue
        candidates.extend(_candidates_from_rolling_payload(payload, report_path))
    return tuple(candidates)


def _candidates_from_rolling_payload(
    payload: dict[str, Any], report_path: Path
) -> tuple[BaselineCandidate, ...]:
    """Convert one rolling-validation payload into baseline candidates."""

    cutoffs = tuple(payload.get("cutoffs", ()))
    windows = payload.get("windows", ())
    if not cutoffs or not windows:
        return ()
    metric_keys = (
        ("source_order", "source_order_map_at_k"),
        ("deterministic", "deterministic_map_at_k"),
        ("learned_linear", "learned_map_at_k"),
    )
    config_payload = payload.get("config") or {}
    k = int(payload.get("k", 12))
    candidate_k = int(payload.get("candidate_k", 12))
    candidates: list[BaselineCandidate] = []
    for ranker_kind, metric_field in metric_keys:
        per_cutoff: dict[str, float] = {}
        for window in windows:
            cutoff = window.get("evaluation_cutoff")
            value = window.get(metric_field)
            if cutoff is None or value is None:
                continue
            per_cutoff[str(cutoff)] = float(value)
        if not per_cutoff:
            continue
        values = list(per_cutoff.values())
        offline = OfflineRollingMetrics(
            cutoffs=cutoffs,
            per_cutoff_map_at_k=per_cutoff,
            mean_map_at_k=round(sum(values) / len(values), 6),
            min_map_at_k=round(min(values), 6),
            max_map_at_k=round(max(values), 6),
            k=k,
            candidate_k=candidate_k,
            rolling_report_path=str(report_path),
        )
        config_dict = {
            "ranker_kind": ranker_kind,
            "rolling_report": report_path.name,
            "cutoffs": list(cutoffs),
            "k": k,
            "candidate_k": candidate_k,
            "training_config": config_payload,
            "include_co_visitation": payload.get("include_co_visitation"),
            "max_target_customers": payload.get("max_target_customers"),
        }
        config_hash = compute_config_hash(config_dict)
        candidates.append(
            BaselineCandidate(
                name=(f"{ranker_kind}@{','.join(cutoffs)}" f"[k={k},candidate_k={candidate_k}]"),
                ranker_kind=ranker_kind,
                config_hash=config_hash,
                config=config_dict,
                offline_metrics=offline,
                submission_paths=(),
                notes=(
                    "Auto-discovered from rolling-ranker-validation report. "
                    "Kaggle leaderboard score not yet recorded."
                ),
            )
        )
    return tuple(candidates)


def compute_config_hash(config: dict[str, Any]) -> str:
    """Return the first 8 characters of a SHA-256 over a canonical-JSON config."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Champion selection


def pick_champion_candidate(
    candidates: Iterable[BaselineCandidate],
    target_leaderboard_map_at_k: float = DEFAULT_BASELINE_TARGET_MAP_AT_K,
    target_tolerance: float = DEFAULT_BASELINE_TARGET_TOLERANCE,
) -> tuple[int | None, str, tuple[str, ...]]:
    """Pick the best champion candidate using the following priority order.

    The plan specifies a deterministic, defensible champion choice.
    Priority:

    1. If any candidate has a Kaggle leaderboard score within
       ``target_tolerance`` of ``target_leaderboard_map_at_k``, pick the
       one with the highest LB score (ties broken by highest offline mean,
       then by name for determinism).
    2. Otherwise, if any candidate has a Kaggle LB score, pick the highest
       LB score and warn that the target was not matched.
    3. Otherwise, pick the candidate with the highest offline rolling-mean
       MAP@K and warn that LB calibration is missing.
    4. If no candidate has either, return ``(None, ...)`` and warn.

    Args:
        candidates: Iterable of registered baseline candidates.
        target_leaderboard_map_at_k: The leaderboard score being reproduced.
        target_tolerance: Absolute tolerance considered a match.

    Returns:
        Tuple ``(champion_index, rationale, warnings)``.
    """

    candidate_list = list(candidates)
    warnings: list[str] = []
    if not candidate_list:
        return None, "No baseline candidates registered.", ("no_candidates",)

    def lb_key(item: tuple[int, BaselineCandidate]) -> tuple[float, float, str]:
        _, candidate = item
        offline_mean = (
            candidate.offline_metrics.mean_map_at_k
            if candidate.offline_metrics is not None
            else 0.0
        )
        lb_score = candidate.leaderboard_public_map_at_k or 0.0
        return (lb_score, offline_mean, candidate.name)

    indexed = list(enumerate(candidate_list))
    within_tolerance = [
        (index, candidate)
        for index, candidate in indexed
        if candidate.leaderboard_public_map_at_k is not None
        and abs(candidate.leaderboard_public_map_at_k - target_leaderboard_map_at_k)
        <= target_tolerance
    ]
    if within_tolerance:
        index, candidate = max(within_tolerance, key=lb_key)
        rationale = (
            f"Candidate '{candidate.name}' matches target LB MAP@K "
            f"{target_leaderboard_map_at_k:.5f} within ±{target_tolerance:.4f} "
            f"(observed {candidate.leaderboard_public_map_at_k:.5f})."
        )
        return index, rationale, tuple(warnings)

    with_lb = [
        (index, candidate)
        for index, candidate in indexed
        if candidate.leaderboard_public_map_at_k is not None
    ]
    if with_lb:
        index, candidate = max(with_lb, key=lb_key)
        warnings.append("no_candidate_within_target_tolerance")
        rationale = (
            f"No candidate matched target LB MAP@K {target_leaderboard_map_at_k:.5f} "
            f"within ±{target_tolerance:.4f}. Picked highest-LB candidate "
            f"'{candidate.name}' "
            f"(observed {candidate.leaderboard_public_map_at_k:.5f})."
        )
        return index, rationale, tuple(warnings)

    with_offline = [
        (index, candidate) for index, candidate in indexed if candidate.offline_metrics is not None
    ]
    if with_offline:
        warnings.append("no_leaderboard_data_yet")
        index, candidate = max(
            with_offline,
            key=lambda item: (
                (
                    item[1].offline_metrics.mean_map_at_k
                    if item[1].offline_metrics is not None
                    else 0.0
                ),
                item[1].name,
            ),
        )
        mean_value = (
            candidate.offline_metrics.mean_map_at_k
            if candidate.offline_metrics is not None
            else 0.0
        )
        rationale = (
            "No Kaggle leaderboard data available yet. "
            f"Picked candidate '{candidate.name}' with highest offline rolling "
            f"mean MAP@K {mean_value:.5f}."
        )
        return index, rationale, tuple(warnings)

    warnings.append("no_offline_or_leaderboard_data")
    return None, "No candidate has offline or leaderboard metrics.", tuple(warnings)


# ---------------------------------------------------------------------------
# Building, serialization, and rendering


def build_baseline_champion_report(
    candidates: Iterable[BaselineCandidate],
    target_leaderboard_map_at_k: float = DEFAULT_BASELINE_TARGET_MAP_AT_K,
    target_tolerance: float = DEFAULT_BASELINE_TARGET_TOLERANCE,
    extra_warnings: Iterable[str] = (),
) -> BaselineChampionReport:
    """Assemble a champion report from a collection of candidates.

    Args:
        candidates: Candidates to evaluate.
        target_leaderboard_map_at_k: LB number being reproduced.
        target_tolerance: Tolerance for the LB match check.
        extra_warnings: Additional warnings to merge into the report.

    Returns:
        Fully populated :class:`BaselineChampionReport`.
    """

    candidate_tuple = tuple(candidates)
    champion_index, rationale, warnings = pick_champion_candidate(
        candidate_tuple,
        target_leaderboard_map_at_k=target_leaderboard_map_at_k,
        target_tolerance=target_tolerance,
    )
    merged_warnings = tuple(dict.fromkeys((*warnings, *extra_warnings)))
    return BaselineChampionReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        target_leaderboard_map_at_k=target_leaderboard_map_at_k,
        target_tolerance=target_tolerance,
        candidates=candidate_tuple,
        champion_index=champion_index,
        champion_rationale=rationale,
        warnings=merged_warnings,
    )


def baseline_champion_report_to_dict(report: BaselineChampionReport) -> dict[str, Any]:
    """Convert a champion report into JSON-serializable primitives."""

    return asdict(report)


def write_baseline_champion_report(report: BaselineChampionReport, path: Path | str) -> Path:
    """Write a champion report as deterministic JSON."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(baseline_champion_report_to_dict(report), indent=2, sort_keys=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    return output_path


def load_baseline_champion_report(path: Path | str) -> BaselineChampionReport:
    """Load a previously-written champion report from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed :class:`BaselineChampionReport`.
    """

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    candidates = tuple(
        _candidate_from_dict(candidate_payload)
        for candidate_payload in payload.get("candidates", ())
    )
    return BaselineChampionReport(
        generated_at_utc=str(payload.get("generated_at_utc", "")),
        target_leaderboard_map_at_k=float(
            payload.get("target_leaderboard_map_at_k", DEFAULT_BASELINE_TARGET_MAP_AT_K)
        ),
        target_tolerance=float(payload.get("target_tolerance", DEFAULT_BASELINE_TARGET_TOLERANCE)),
        candidates=candidates,
        champion_index=(
            int(payload["champion_index"]) if payload.get("champion_index") is not None else None
        ),
        champion_rationale=str(payload.get("champion_rationale", "")),
        warnings=tuple(payload.get("warnings", ())),
    )


def _candidate_from_dict(payload: dict[str, Any]) -> BaselineCandidate:
    """Reconstruct a :class:`BaselineCandidate` from a JSON dictionary."""

    offline_payload = payload.get("offline_metrics")
    offline = None
    if offline_payload is not None:
        offline = OfflineRollingMetrics(
            cutoffs=tuple(offline_payload.get("cutoffs", ())),
            per_cutoff_map_at_k={
                str(cutoff): float(value)
                for cutoff, value in (offline_payload.get("per_cutoff_map_at_k", {}).items())
            },
            mean_map_at_k=float(offline_payload.get("mean_map_at_k", 0.0)),
            min_map_at_k=float(offline_payload.get("min_map_at_k", 0.0)),
            max_map_at_k=float(offline_payload.get("max_map_at_k", 0.0)),
            k=int(offline_payload.get("k", 12)),
            candidate_k=int(offline_payload.get("candidate_k", 12)),
            rolling_report_path=str(offline_payload.get("rolling_report_path", "")),
        )
    return BaselineCandidate(
        name=str(payload.get("name", "")),
        ranker_kind=str(payload.get("ranker_kind", "")),
        config_hash=str(payload.get("config_hash", "")),
        config=dict(payload.get("config", {})),
        offline_metrics=offline,
        submission_paths=tuple(payload.get("submission_paths", ())),
        leaderboard_public_map_at_k=_optional_float(payload.get("leaderboard_public_map_at_k")),
        leaderboard_private_map_at_k=_optional_float(payload.get("leaderboard_private_map_at_k")),
        submitted_at_utc=_optional_str(payload.get("submitted_at_utc")),
        kaggle_submission_message=_optional_str(payload.get("kaggle_submission_message")),
        notes=str(payload.get("notes", "")),
    )


def _optional_float(value: Any) -> float | None:
    """Return ``float(value)`` or ``None`` for falsy/None inputs."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    """Return ``str(value)`` or ``None`` when value is ``None``."""

    if value is None:
        return None
    return str(value)


def merge_candidates(
    existing: Iterable[BaselineCandidate],
    incoming: Iterable[BaselineCandidate],
) -> tuple[BaselineCandidate, ...]:
    """Merge incoming candidates into an existing list keyed by config hash.

    When an existing candidate shares a ``config_hash`` with an incoming
    one, fields the user has set (LB scores, notes, submission paths) are
    preserved while computed offline metrics are refreshed from the
    incoming entry.

    Args:
        existing: Previously registered candidates (e.g. from disk).
        incoming: Newly discovered candidates.

    Returns:
        Merged, deterministic tuple of candidates.
    """

    by_hash: dict[str, BaselineCandidate] = {}
    for candidate in existing:
        by_hash[candidate.config_hash] = candidate
    for candidate in incoming:
        existing_candidate = by_hash.get(candidate.config_hash)
        if existing_candidate is None:
            by_hash[candidate.config_hash] = candidate
            continue
        merged_submissions = tuple(
            dict.fromkeys((*existing_candidate.submission_paths, *candidate.submission_paths))
        )
        by_hash[candidate.config_hash] = replace(
            existing_candidate,
            offline_metrics=candidate.offline_metrics or existing_candidate.offline_metrics,
            submission_paths=merged_submissions,
            notes=existing_candidate.notes or candidate.notes,
            config=candidate.config or existing_candidate.config,
        )
    return tuple(sorted(by_hash.values(), key=lambda c: (c.ranker_kind, c.name)))


def render_baseline_champion_markdown(report: BaselineChampionReport) -> str:
    """Render the champion report as a Markdown summary."""

    lines: list[str] = []
    lines.append("# Baseline Champion Pinning")
    lines.append("")
    lines.append(f"- Generated: `{report.generated_at_utc}`")
    lines.append(
        "- Target leaderboard MAP@K: "
        f"`{report.target_leaderboard_map_at_k:.5f}` "
        f"(tolerance `±{report.target_tolerance:.4f}`)"
    )
    if report.champion_index is None:
        lines.append("- Champion: **(none selected)**")
    else:
        champion = report.candidates[report.champion_index]
        lines.append(f"- Champion: **{champion.name}**")
    lines.append(f"- Rationale: {report.champion_rationale}")
    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- `{warning}`")
    lines.append("")
    lines.append("## Registered candidates")
    lines.append("")
    lines.append("| # | Ranker | Name | Config hash | Offline mean | LB public | LB private |")
    lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: |")
    for index, candidate in enumerate(report.candidates):
        offline_mean = (
            f"{candidate.offline_metrics.mean_map_at_k:.5f}"
            if candidate.offline_metrics is not None
            else "—"
        )
        lb_public = (
            f"{candidate.leaderboard_public_map_at_k:.5f}"
            if candidate.leaderboard_public_map_at_k is not None
            else "—"
        )
        lb_private = (
            f"{candidate.leaderboard_private_map_at_k:.5f}"
            if candidate.leaderboard_private_map_at_k is not None
            else "—"
        )
        marker = "★" if index == report.champion_index else ""
        lines.append(
            f"| {index} | {candidate.ranker_kind} | {candidate.name} {marker} | "
            f"`{candidate.config_hash}` | {offline_mean} | {lb_public} | {lb_private} |"
        )
    lines.append("")
    lines.append("## Per-candidate details")
    lines.append("")
    for index, candidate in enumerate(report.candidates):
        lines.append(f"### {index}. {candidate.name}")
        lines.append("")
        lines.append(f"- Ranker kind: `{candidate.ranker_kind}`")
        lines.append(f"- Config hash: `{candidate.config_hash}`")
        if candidate.offline_metrics is not None:
            lines.append(
                "- Offline rolling MAP@K (mean / min / max): "
                f"{candidate.offline_metrics.mean_map_at_k:.5f} / "
                f"{candidate.offline_metrics.min_map_at_k:.5f} / "
                f"{candidate.offline_metrics.max_map_at_k:.5f}"
            )
            lines.append("- Per-cutoff offline MAP@K:")
            for cutoff, value in candidate.offline_metrics.per_cutoff_map_at_k.items():
                lines.append(f"  - {cutoff}: {value:.5f}")
            lines.append(f"- Rolling report: `{candidate.offline_metrics.rolling_report_path}`")
        if candidate.submission_paths:
            lines.append("- Submission paths:")
            for path in candidate.submission_paths:
                lines.append(f"  - `{path}`")
        if candidate.leaderboard_public_map_at_k is not None:
            lines.append(
                "- Kaggle public LB MAP@K: " f"{candidate.leaderboard_public_map_at_k:.5f}"
            )
        if candidate.leaderboard_private_map_at_k is not None:
            lines.append(
                "- Kaggle private LB MAP@K: " f"{candidate.leaderboard_private_map_at_k:.5f}"
            )
        if candidate.submitted_at_utc is not None:
            lines.append(f"- Submitted at: `{candidate.submitted_at_utc}`")
        if candidate.kaggle_submission_message:
            lines.append(f"- Kaggle message: {candidate.kaggle_submission_message}")
        if candidate.notes:
            lines.append(f"- Notes: {candidate.notes}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_baseline_champion_markdown(report: BaselineChampionReport, path: Path | str) -> Path:
    """Write the Markdown summary of the champion report to disk."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_baseline_champion_markdown(report), encoding="utf-8")
    return output_path
