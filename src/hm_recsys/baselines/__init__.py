"""Baseline-pinning utilities for the Phase -1 reproducibility audit.

This package centralizes the bookkeeping that lets the gap-closure plan
make auditable improvement claims. It records:

    * Each baseline candidate (deterministic, learned-linear, LightGBM,
      ...) with its configuration hash.
    * The locally-computed rolling-mean MAP@12 (the only number we can
      compute without Kaggle access).
    * The user-supplied Kaggle leaderboard score (filled in after the
      user uploads a submission and reports the number back).
    * Whichever candidate is currently designated as the champion, with
      an explicit rationale.

The plan calls this the "champion_022_config.json" file. We additionally
support an arbitrary number of registered candidates so the same
artifact also drives the P-1.3 offline-to-leaderboard calibration.
"""

from hm_recsys.baselines.champion import (
    DEFAULT_BASELINE_TARGET_MAP_AT_K,
    DEFAULT_BASELINE_TARGET_TOLERANCE,
    BaselineCandidate,
    BaselineChampionReport,
    OfflineRollingMetrics,
    baseline_champion_report_to_dict,
    discover_rolling_validation_candidates,
    load_baseline_champion_report,
    pick_champion_candidate,
    render_baseline_champion_markdown,
    write_baseline_champion_markdown,
    write_baseline_champion_report,
)

__all__ = [
    "DEFAULT_BASELINE_TARGET_MAP_AT_K",
    "DEFAULT_BASELINE_TARGET_TOLERANCE",
    "BaselineCandidate",
    "BaselineChampionReport",
    "OfflineRollingMetrics",
    "baseline_champion_report_to_dict",
    "discover_rolling_validation_candidates",
    "load_baseline_champion_report",
    "pick_champion_candidate",
    "render_baseline_champion_markdown",
    "write_baseline_champion_markdown",
    "write_baseline_champion_report",
]
