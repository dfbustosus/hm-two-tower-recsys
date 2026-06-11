"""Command-line interface for H&M recommender validation utilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from typing import cast

from hm_recsys.baselines.champion import (
    DEFAULT_BASELINE_TARGET_MAP_AT_K,
    DEFAULT_BASELINE_TARGET_TOLERANCE,
    build_baseline_champion_report,
    discover_rolling_validation_candidates,
    load_baseline_champion_report,
    merge_candidates,
    write_baseline_champion_markdown,
    write_baseline_champion_report,
)
from hm_recsys.data.contracts import validate_hm_data_contract, write_data_contract_report
from hm_recsys.data.io import (
    iter_transaction_events,
    iter_transactions,
    load_article_ids,
    load_submission_customer_ids,
    load_submission_customer_ids_in_order,
)
from hm_recsys.eda import (
    DEFAULT_ROLLING_CUTOFFS,
    EdaReportConfig,
    EdaSegmentThresholds,
    build_eda_report,
    write_eda_report,
    write_eda_report_markdown,
)
from hm_recsys.embeddings.article_content import (
    build_article_popularity_priority,
    write_article_content_export,
)
from hm_recsys.embeddings.cache_manifest import EmbeddingCacheKind
from hm_recsys.embeddings.generation import (
    ArticleEmbeddingCacheWriteConfig,
    write_article_embedding_cache_from_content_export,
)
from hm_recsys.embeddings.image_inventory import write_article_image_inventory
from hm_recsys.evaluation.perfect_ranker import (
    PerfectRankerCutoffInput,
    build_perfect_ranker_ceiling_report,
    write_perfect_ranker_ceiling_markdown,
    write_perfect_ranker_ceiling_report,
)
from hm_recsys.evaluation.submission import (
    validate_submission_file,
    write_submission_file,
    write_submission_validation_report,
)
from hm_recsys.evaluation.temporal import (
    TemporalSplit,
    collect_validation_labels_for_splits,
    summarize_temporal_split,
    summarize_temporal_split_with_labels,
    write_temporal_split_summary,
)
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.ranking.behavioral import load_article_attribute_maps
from hm_recsys.ranking.deterministic import (
    DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    DeterministicRankerWeights,
    evaluate_deterministic_ranker_from_csv,
    write_deterministic_ranker_report,
)
from hm_recsys.ranking.deterministic_tuning import (
    DEFAULT_DETERMINISTIC_RANKER_TUNING_GRID,
    DeterministicRankerTuningGrid,
    select_deterministic_ranker_weights_from_csv,
    tune_deterministic_ranker_from_csv,
    write_deterministic_ranker_tuning_report,
)
from hm_recsys.ranking.lightgbm_behavioral import (
    LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS,
    LightGBMBehavioralRankerConfig,
    LightGBMBehavioralTrainingWindow,
    evaluate_lightgbm_behavioral_ranker_from_csv,
    train_lightgbm_behavioral_ranker_from_windows,
    write_lightgbm_behavioral_ranker_report,
)
from hm_recsys.ranking.linear import (
    LinearRankerConfig,
    build_learned_linear_ranker_report,
    evaluate_linear_ranker_from_csv,
    previous_window_split,
    train_linear_ranker_from_csv,
    write_learned_linear_ranker_report,
)
from hm_recsys.ranking.rolling import (
    build_rolling_ranker_validation_report,
    write_rolling_ranker_validation_report,
)
from hm_recsys.ranking.submission import (
    build_deterministic_ranker_submission_predictions,
    build_deterministic_ranker_submission_report,
    build_learned_linear_ranker_submission_report,
    build_lightgbm_behavioral_ranker_submission_predictions,
    build_linear_ranker_submission_predictions,
    write_deterministic_ranker_submission_report,
    write_learned_linear_ranker_submission_report,
)
from hm_recsys.retrieval.baselines import (
    build_repeat_popularity_submission_baseline,
    evaluate_repeat_popularity_baseline,
    find_max_transaction_date,
    write_baseline_evaluation_report,
)
from hm_recsys.retrieval.candidate_diagnostics import (
    DEFAULT_EVALUATION_KS,
    evaluate_baseline_candidate_diagnostics,
    write_candidate_diagnostics_report,
)
from hm_recsys.retrieval.candidate_export import (
    select_validation_label_customer_ids,
    write_candidate_export_summary,
    write_validation_candidate_export,
)
from hm_recsys.retrieval.co_visitation import (
    DEFAULT_MAX_HISTORY_ITEMS,
    DEFAULT_MAX_NEIGHBORS_PER_ITEM,
)
from hm_recsys.retrieval.content_similarity_diagnostics import (
    DEFAULT_CONTENT_SIMILARITY_EVALUATION_KS,
    evaluate_cached_content_similarity,
    write_content_similarity_diagnostics_report,
)
from hm_recsys.retrieval.metadata_affinity import load_article_attribute_values
from hm_recsys.retrieval.seasonality import (
    DEFAULT_SEASONAL_SHIFT_DAYS,
    DEFAULT_SEASONAL_WINDOW_DAYS,
)
from hm_recsys.retrieval.segment_popularity import (
    DEFAULT_AGE_SEGMENT_BUCKET_SIZE,
    load_customer_age_segments,
)
from hm_recsys.retrieval.source_names import (
    MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE,
    MULTIMODAL_SIMILARITY_SOURCE,
    TWO_TOWER_RETRIEVAL_SOURCE,
)
from hm_recsys.training.two_tower_export import (
    TwoTowerExampleExportConfig,
    write_two_tower_example_export,
    write_two_tower_example_export_summary,
)
from hm_recsys.training.two_tower_retrieval import (
    TwoTowerSmokeModel,
    TwoTowerSmokeTrainingConfig,
    build_article_popularity_score_prior,
    build_two_tower_retrieval_report,
    evaluate_two_tower_retrieval,
    train_two_tower_smoke_model_from_csv,
    write_two_tower_retrieval_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``hm-recsys`` argument parser.

    Returns:
        Configured ``ArgumentParser`` with all supported subcommands.
    """

    parser = argparse.ArgumentParser(prog="hm-recsys")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-data-contract",
        help="Validate local H&M raw data files and write a JSON report.",
    )
    validate_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to auto-detection from the current directory.",
    )
    validate_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help=(
            "Raw H&M data directory. Defaults to "
            "data/raw/h-and-m-personalized-fashion-recommendations/."
        ),
    )
    validate_parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Report path. Defaults to artifacts/data-contract/data_contract_report.json.",
    )
    validate_parser.set_defaults(handler=_handle_validate_data_contract)

    eda_parser = subparsers.add_parser(
        "eda-report",
        help=(
            "Compute the Phase -1 EDA report covering transaction volume, channel "
            "mix, customer history depth, article hierarchy fanout, repeat-purchase "
            "structure, and cold-user share across rolling cutoffs."
        ),
    )
    eda_parser.add_argument(
        "--rolling-cutoffs",
        nargs="+",
        default=list(DEFAULT_ROLLING_CUTOFFS),
        metavar="YYYY-MM-DD",
        help=(
            "Rolling validation cutoffs used to compute cold-user share. "
            f"Defaults to {' '.join(DEFAULT_ROLLING_CUTOFFS)}."
        ),
    )
    eda_parser.add_argument(
        "--cold-max-transactions",
        type=int,
        default=EdaSegmentThresholds().cold_max_transactions,
        help="Maximum transaction count for the cold segment. Defaults to 0.",
    )
    eda_parser.add_argument(
        "--sparse-max-transactions",
        type=int,
        default=EdaSegmentThresholds().sparse_max_transactions,
        help="Maximum transaction count for the sparse segment. Defaults to 4.",
    )
    eda_parser.add_argument(
        "--top-hierarchy-values",
        type=int,
        default=20,
        help="Maximum number of values surfaced per hierarchy column. Defaults to 20.",
    )
    eda_parser.add_argument(
        "--top-busy-days",
        type=int,
        default=30,
        help="Number of busiest transaction days surfaced. Defaults to 30.",
    )
    eda_parser.add_argument("--project-root", type=Path, default=None)
    eda_parser.add_argument("--raw-data-dir", type=Path, default=None)
    eda_parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON report path. Defaults to artifacts/eda/eda_report.json.",
    )
    eda_parser.add_argument(
        "--markdown-path",
        type=Path,
        default=None,
        help="Markdown report path. Defaults to artifacts/eda/eda_report.md.",
    )
    eda_parser.set_defaults(handler=_handle_eda_report)

    pin_baseline_parser = subparsers.add_parser(
        "pin-baseline-champion",
        help=(
            "Pin the current MAP@12 baseline champion by inventorying existing "
            "rolling-validation reports and (optionally) attaching user-supplied "
            "Kaggle leaderboard scores."
        ),
    )
    pin_baseline_parser.add_argument(
        "--rolling-reports-dir",
        type=Path,
        default=None,
        help="Directory of rolling-validation JSON reports to scan.",
    )
    pin_baseline_parser.add_argument(
        "--rolling-cutoffs",
        nargs="+",
        default=list(DEFAULT_ROLLING_CUTOFFS),
        metavar="YYYY-MM-DD",
        help=(
            "Restrict discovery to rolling reports whose cutoffs match exactly. "
            f"Defaults to {' '.join(DEFAULT_ROLLING_CUTOFFS)}."
        ),
    )
    pin_baseline_parser.add_argument(
        "--target-leaderboard-map-at-k",
        type=float,
        default=DEFAULT_BASELINE_TARGET_MAP_AT_K,
        help=(
            "Kaggle leaderboard MAP@K to reproduce. "
            f"Defaults to {DEFAULT_BASELINE_TARGET_MAP_AT_K}."
        ),
    )
    pin_baseline_parser.add_argument(
        "--target-tolerance",
        type=float,
        default=DEFAULT_BASELINE_TARGET_TOLERANCE,
        help=(
            "Absolute MAP@K tolerance for the leaderboard match check. "
            f"Defaults to {DEFAULT_BASELINE_TARGET_TOLERANCE}."
        ),
    )
    pin_baseline_parser.add_argument(
        "--merge-existing",
        action="store_true",
        help=(
            "Preserve user-supplied fields (LB scores, notes) from the existing "
            "champion report at the destination path when it already exists."
        ),
    )
    pin_baseline_parser.add_argument("--project-root", type=Path, default=None)
    pin_baseline_parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON output path. Defaults to artifacts/baselines/champion_022_config.json.",
    )
    pin_baseline_parser.add_argument(
        "--markdown-path",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to artifacts/baselines/champion_022_config.md.",
    )
    pin_baseline_parser.set_defaults(handler=_handle_pin_baseline_champion)

    perfect_ranker_parser = subparsers.add_parser(
        "compute-perfect-ranker-ceiling",
        help=(
            "Compute the perfect-ranker (oracle) MAP@K ceiling on the existing "
            "candidate set for a rolling cutoff schedule. Quantifies the share of "
            "the MAP gap that is reachable by ranker improvements alone."
        ),
    )
    perfect_ranker_parser.add_argument(
        "--cutoffs",
        nargs="+",
        default=list(DEFAULT_ROLLING_CUTOFFS),
        metavar="YYYY-MM-DD",
        help=(
            "Evaluation cutoffs. Defaults to the canonical rolling schedule "
            f"({' '.join(DEFAULT_ROLLING_CUTOFFS)})."
        ),
    )
    perfect_ranker_parser.add_argument(
        "--candidate-path",
        action="append",
        type=Path,
        metavar="PATH",
        help=(
            "Candidate CSV path for one cutoff. Repeat the flag to provide one "
            "path per cutoff. Order must match --cutoffs."
        ),
    )
    perfect_ranker_parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="Validation label horizon in days. Defaults to 7.",
    )
    perfect_ranker_parser.add_argument(
        "--k",
        type=int,
        default=12,
        help="Recommendation depth. Defaults to 12.",
    )
    perfect_ranker_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help=(
            "Optional deterministic smoke cap mirrored from the candidate-export "
            "command. Restricts the validation-label sample so MAP/ceiling are "
            "computed on the same customer universe used to build the candidates."
        ),
    )
    perfect_ranker_parser.add_argument("--project-root", type=Path, default=None)
    perfect_ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    perfect_ranker_parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help=(
            "JSON output path. Defaults to "
            "artifacts/ranker-baselines/perfect_ranker_ceiling.json."
        ),
    )
    perfect_ranker_parser.add_argument(
        "--markdown-path",
        type=Path,
        default=None,
        help=(
            "Markdown output path. Defaults to "
            "artifacts/ranker-baselines/perfect_ranker_ceiling.md."
        ),
    )
    perfect_ranker_parser.set_defaults(handler=_handle_compute_perfect_ranker_ceiling)

    split_parser = subparsers.add_parser(
        "summarize-temporal-split",
        help="Summarize train/validation row counts for a cutoff-based split.",
    )
    split_parser.add_argument("--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD.")
    split_parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="Validation horizon in days. Defaults to 7.",
    )
    split_parser.add_argument("--project-root", type=Path, default=None)
    split_parser.add_argument("--raw-data-dir", type=Path, default=None)
    split_parser.add_argument("--report-path", type=Path, default=None)
    split_parser.set_defaults(handler=_handle_summarize_temporal_split)

    submission_parser = subparsers.add_parser(
        "validate-submission",
        help="Validate a Kaggle submission CSV against sample_submission and articles.",
    )
    submission_parser.add_argument("--submission-path", type=Path, required=True)
    submission_parser.add_argument("--project-root", type=Path, default=None)
    submission_parser.add_argument("--raw-data-dir", type=Path, default=None)
    submission_parser.add_argument("--report-path", type=Path, default=None)
    submission_parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow fewer than 12 predictions per customer.",
    )
    submission_parser.set_defaults(handler=_handle_validate_submission)

    baseline_parser = subparsers.add_parser(
        "evaluate-baseline",
        help="Evaluate repeat-purchase plus recent-popularity baseline on a temporal split.",
    )
    baseline_parser.add_argument(
        "--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD."
    )
    baseline_parser.add_argument("--horizon-days", type=int, default=7)
    baseline_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    baseline_parser.add_argument("--k", type=int, default=12)
    baseline_parser.add_argument("--project-root", type=Path, default=None)
    baseline_parser.add_argument("--raw-data-dir", type=Path, default=None)
    baseline_parser.add_argument("--report-path", type=Path, default=None)
    baseline_parser.set_defaults(handler=_handle_evaluate_baseline)

    baseline_submission_parser = subparsers.add_parser(
        "generate-baseline-submission",
        help="Generate and validate a repeat-purchase plus popularity submission CSV.",
    )
    baseline_submission_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    baseline_submission_parser.add_argument("--k", type=int, default=12)
    baseline_submission_parser.add_argument("--project-root", type=Path, default=None)
    baseline_submission_parser.add_argument("--raw-data-dir", type=Path, default=None)
    baseline_submission_parser.add_argument("--output-path", type=Path, default=None)
    baseline_submission_parser.add_argument("--validation-report-path", type=Path, default=None)
    baseline_submission_parser.set_defaults(handler=_handle_generate_baseline_submission)

    candidate_parser = subparsers.add_parser(
        "candidate-diagnostics",
        help="Evaluate repeat and popularity candidate-source diagnostics.",
    )
    candidate_parser.add_argument(
        "--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD."
    )
    candidate_parser.add_argument("--horizon-days", type=int, default=7)
    candidate_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    candidate_parser.add_argument(
        "--evaluation-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVALUATION_KS),
        help="Candidate recall cutoffs. Defaults to 12 50 100.",
    )
    candidate_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate diagnostics.",
    )
    candidate_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    candidate_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    candidate_parser.add_argument("--project-root", type=Path, default=None)
    candidate_parser.add_argument("--raw-data-dir", type=Path, default=None)
    candidate_parser.add_argument("--report-path", type=Path, default=None)
    candidate_parser.set_defaults(handler=_handle_candidate_diagnostics)

    candidate_export_parser = subparsers.add_parser(
        "export-candidates",
        help="Export ranker-ready candidate source rows for validation-label customers.",
    )
    candidate_export_parser.add_argument(
        "--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD."
    )
    candidate_export_parser.add_argument("--horizon-days", type=int, default=7)
    candidate_export_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    candidate_export_parser.add_argument("--k", type=int, default=12)
    candidate_export_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    candidate_export_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    candidate_export_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    candidate_export_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke exports.",
    )
    _add_seasonal_popularity_arguments(candidate_export_parser)
    _add_age_segment_popularity_arguments(candidate_export_parser)
    _add_garment_group_popularity_arguments(candidate_export_parser)
    _add_product_code_popularity_arguments(candidate_export_parser)
    _add_content_similarity_candidate_arguments(candidate_export_parser)
    _add_two_tower_candidate_arguments(candidate_export_parser)
    candidate_export_parser.add_argument("--project-root", type=Path, default=None)
    candidate_export_parser.add_argument("--raw-data-dir", type=Path, default=None)
    candidate_export_parser.add_argument("--output-path", type=Path, default=None)
    candidate_export_parser.add_argument("--report-path", type=Path, default=None)
    candidate_export_parser.set_defaults(handler=_handle_export_candidates)

    ranker_parser = subparsers.add_parser(
        "evaluate-ranker-baseline",
        help="Evaluate deterministic source-feature ranker on temporal validation.",
    )
    ranker_parser.add_argument(
        "--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD."
    )
    ranker_parser.add_argument("--horizon-days", type=int, default=7)
    ranker_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    ranker_parser.add_argument("--candidate-k", type=int, default=12)
    ranker_parser.add_argument("--k", type=int, default=12)
    ranker_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    ranker_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    ranker_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    ranker_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke evaluations.",
    )
    _add_seasonal_popularity_arguments(ranker_parser)
    _add_age_segment_popularity_arguments(ranker_parser)
    _add_garment_group_popularity_arguments(ranker_parser)
    _add_product_code_popularity_arguments(ranker_parser)
    _add_content_similarity_candidate_arguments(ranker_parser)
    _add_two_tower_candidate_arguments(ranker_parser)
    _add_two_tower_ranker_weight_arguments(ranker_parser)
    ranker_parser.add_argument("--project-root", type=Path, default=None)
    ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    ranker_parser.add_argument("--candidate-output-path", type=Path, default=None)
    ranker_parser.add_argument("--candidate-report-path", type=Path, default=None)
    ranker_parser.add_argument("--report-path", type=Path, default=None)
    ranker_parser.set_defaults(handler=_handle_evaluate_ranker_baseline)

    tuned_ranker_parser = subparsers.add_parser(
        "tune-deterministic-ranker",
        help="Tune deterministic ranker weights on the previous window and evaluate them.",
    )
    tuned_ranker_parser.add_argument(
        "--cutoff", required=True, help="Evaluation cutoff date, YYYY-MM-DD."
    )
    tuned_ranker_parser.add_argument(
        "--train-cutoff",
        default=None,
        help="Optional tuning-label cutoff. Defaults to previous non-overlapping window.",
    )
    tuned_ranker_parser.add_argument("--horizon-days", type=int, default=7)
    tuned_ranker_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    tuned_ranker_parser.add_argument("--candidate-k", type=int, default=12)
    tuned_ranker_parser.add_argument("--k", type=int, default=12)
    tuned_ranker_parser.add_argument(
        "--top-trials",
        type=int,
        default=10,
        help="Number of top tuning-window weight trials to retain in the report.",
    )
    tuned_ranker_parser.add_argument(
        "--research-weight-grid",
        action="store_true",
        help=(
            "Use a broader fast in-memory grid over repeat, recent popularity, "
            "co-visitation, metadata, and two-tower rank weights."
        ),
    )
    tuned_ranker_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    tuned_ranker_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    tuned_ranker_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    tuned_ranker_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap applied separately to tune and eval windows.",
    )
    _add_seasonal_popularity_arguments(tuned_ranker_parser)
    _add_age_segment_popularity_arguments(tuned_ranker_parser)
    _add_garment_group_popularity_arguments(tuned_ranker_parser)
    _add_product_code_popularity_arguments(tuned_ranker_parser)
    _add_content_similarity_candidate_arguments(tuned_ranker_parser)
    _add_two_tower_candidate_arguments(tuned_ranker_parser)
    _add_two_tower_ranker_weight_arguments(tuned_ranker_parser)
    tuned_ranker_parser.add_argument("--project-root", type=Path, default=None)
    tuned_ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    tuned_ranker_parser.add_argument("--train-candidate-output-path", type=Path, default=None)
    tuned_ranker_parser.add_argument("--eval-candidate-output-path", type=Path, default=None)
    tuned_ranker_parser.add_argument("--report-path", type=Path, default=None)
    tuned_ranker_parser.set_defaults(handler=_handle_tune_deterministic_ranker)

    learned_ranker_parser = subparsers.add_parser(
        "evaluate-learned-ranker-baseline",
        help="Train on the previous window and evaluate a learned linear ranker.",
    )
    learned_ranker_parser.add_argument(
        "--cutoff", required=True, help="Evaluation cutoff date, YYYY-MM-DD."
    )
    learned_ranker_parser.add_argument(
        "--train-cutoff",
        default=None,
        help="Optional training-label cutoff. Defaults to previous non-overlapping window.",
    )
    learned_ranker_parser.add_argument("--horizon-days", type=int, default=7)
    learned_ranker_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    learned_ranker_parser.add_argument("--candidate-k", type=int, default=12)
    learned_ranker_parser.add_argument("--k", type=int, default=12)
    learned_ranker_parser.add_argument("--epochs", type=int, default=3)
    learned_ranker_parser.add_argument("--learning-rate", type=float, default=0.01)
    learned_ranker_parser.add_argument("--l2", type=float, default=0.001)
    learned_ranker_parser.add_argument("--positive-weight", type=float, default=None)
    learned_ranker_parser.add_argument("--max-auto-positive-weight", type=float, default=10.0)
    learned_ranker_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    learned_ranker_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    learned_ranker_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    learned_ranker_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap applied separately to train and eval windows.",
    )
    _add_seasonal_popularity_arguments(learned_ranker_parser)
    _add_age_segment_popularity_arguments(learned_ranker_parser)
    _add_garment_group_popularity_arguments(learned_ranker_parser)
    _add_product_code_popularity_arguments(learned_ranker_parser)
    _add_content_similarity_candidate_arguments(learned_ranker_parser)
    learned_ranker_parser.add_argument("--project-root", type=Path, default=None)
    learned_ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    learned_ranker_parser.add_argument("--train-candidate-output-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--eval-candidate-output-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--train-candidate-report-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--eval-candidate-report-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--report-path", type=Path, default=None)
    learned_ranker_parser.set_defaults(handler=_handle_evaluate_learned_ranker_baseline)

    lightgbm_ranker_parser = subparsers.add_parser(
        "evaluate-lightgbm-behavioral-ranker",
        help=(
            "Train optional LightGBM LambdaRank on source plus cutoff-safe behavioral "
            "features and evaluate a blended ranker. Requires local lightgbm."
        ),
    )
    lightgbm_ranker_parser.add_argument(
        "--cutoff", required=True, help="Evaluation cutoff date, YYYY-MM-DD."
    )
    lightgbm_ranker_parser.add_argument(
        "--train-cutoff",
        default=None,
        help="Training-label cutoff. Defaults to previous non-overlapping window.",
    )
    lightgbm_ranker_parser.add_argument("--horizon-days", type=int, default=7)
    lightgbm_ranker_parser.add_argument("--k", type=int, default=12)
    lightgbm_ranker_parser.add_argument("--negative-per-positive", type=int, default=50)
    lightgbm_ranker_parser.add_argument("--blend-lambda", type=float, default=0.75)
    lightgbm_ranker_parser.add_argument("--num-boost-round", type=int, default=120)
    lightgbm_ranker_parser.add_argument("--learning-rate", type=float, default=0.03)
    lightgbm_ranker_parser.add_argument("--num-leaves", type=int, default=31)
    lightgbm_ranker_parser.add_argument("--min-data-in-leaf", type=int, default=100)
    lightgbm_ranker_parser.add_argument("--feature-fraction", type=float, default=0.9)
    lightgbm_ranker_parser.add_argument("--bagging-fraction", type=float, default=0.9)
    lightgbm_ranker_parser.add_argument("--bagging-freq", type=int, default=1)
    lightgbm_ranker_parser.add_argument("--lambda-l2", type=float, default=5.0)
    lightgbm_ranker_parser.add_argument("--seed", type=int, default=42)
    lightgbm_ranker_parser.add_argument("--num-threads", type=int, default=4)
    lightgbm_ranker_parser.add_argument("--chunk-customers", type=int, default=2000)
    lightgbm_ranker_parser.add_argument(
        "--objective",
        choices=("lambdarank", "rank_xendcg"),
        default="lambdarank",
        help=(
            "LightGBM ranking objective. 'rank_xendcg' (XE-NDCG) typically "
            "outperforms 'lambdarank' on long candidate lists."
        ),
    )
    lightgbm_ranker_parser.add_argument(
        "--progress",
        action="store_true",
        help="Print monitorable progress messages for long LightGBM validation runs.",
    )
    lightgbm_ranker_parser.add_argument(
        "--max-train-customers",
        type=int,
        default=None,
        help="Optional deterministic cap for training-label customers.",
    )
    lightgbm_ranker_parser.add_argument(
        "--max-eval-customers",
        type=int,
        default=None,
        help="Optional deterministic cap for evaluation-label customers.",
    )
    _add_two_tower_ranker_weight_arguments(lightgbm_ranker_parser)
    lightgbm_ranker_parser.add_argument("--project-root", type=Path, default=None)
    lightgbm_ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    lightgbm_ranker_parser.add_argument("--train-candidate-path", type=Path, required=True)
    lightgbm_ranker_parser.add_argument("--eval-candidate-path", type=Path, required=True)
    lightgbm_ranker_parser.add_argument("--report-path", type=Path, default=None)
    lightgbm_ranker_parser.set_defaults(handler=_handle_evaluate_lightgbm_behavioral_ranker)

    lightgbm_submission_parser = subparsers.add_parser(
        "generate-lightgbm-behavioral-submission",
        help=(
            "Train the optional rich LightGBM behavioral ranker on a leakage-safe "
            "label window and generate a final-data submission. Requires local lightgbm."
        ),
    )
    lightgbm_submission_parser.add_argument(
        "--train-cutoff",
        required=True,
        help="Training-label cutoff date, YYYY-MM-DD.",
    )
    lightgbm_submission_parser.add_argument("--horizon-days", type=int, default=7)
    lightgbm_submission_parser.add_argument("--k", type=int, default=12)
    lightgbm_submission_parser.add_argument("--candidate-k", type=int, default=100)
    lightgbm_submission_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    lightgbm_submission_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    lightgbm_submission_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
    )
    lightgbm_submission_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
    )
    lightgbm_submission_parser.add_argument("--negative-per-positive", type=int, default=50)
    lightgbm_submission_parser.add_argument("--blend-lambda", type=float, default=0.75)
    lightgbm_submission_parser.add_argument("--num-boost-round", type=int, default=120)
    lightgbm_submission_parser.add_argument("--learning-rate", type=float, default=0.03)
    lightgbm_submission_parser.add_argument("--num-leaves", type=int, default=31)
    lightgbm_submission_parser.add_argument("--min-data-in-leaf", type=int, default=100)
    lightgbm_submission_parser.add_argument("--feature-fraction", type=float, default=0.9)
    lightgbm_submission_parser.add_argument("--bagging-fraction", type=float, default=0.9)
    lightgbm_submission_parser.add_argument("--bagging-freq", type=int, default=1)
    lightgbm_submission_parser.add_argument("--lambda-l2", type=float, default=5.0)
    lightgbm_submission_parser.add_argument("--seed", type=int, default=42)
    lightgbm_submission_parser.add_argument("--num-threads", type=int, default=4)
    lightgbm_submission_parser.add_argument("--chunk-customers", type=int, default=2000)
    lightgbm_submission_parser.add_argument(
        "--progress",
        action="store_true",
        help="Print monitorable progress messages for long training/submission runs.",
    )
    lightgbm_submission_parser.add_argument(
        "--max-train-customers",
        type=int,
        default=None,
        help="Optional deterministic cap for training-label customers.",
    )
    lightgbm_submission_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional smoke cap for final target customers; skips full submission validation.",
    )
    lightgbm_submission_parser.add_argument(
        "--transaction-progress-interval",
        type=int,
        default=None,
        help="Optional raw transaction scan progress interval for source builders.",
    )
    lightgbm_submission_parser.add_argument(
        "--prediction-progress-interval",
        type=int,
        default=10000,
        help="Customer interval for final scoring progress messages.",
    )
    _add_two_tower_ranker_weight_arguments(lightgbm_submission_parser)
    _add_two_tower_candidate_arguments(lightgbm_submission_parser)
    _add_seasonal_popularity_arguments(lightgbm_submission_parser)
    _add_age_segment_popularity_arguments(lightgbm_submission_parser)
    _add_garment_group_popularity_arguments(lightgbm_submission_parser)
    _add_product_code_popularity_arguments(lightgbm_submission_parser)
    lightgbm_submission_parser.add_argument("--project-root", type=Path, default=None)
    lightgbm_submission_parser.add_argument("--raw-data-dir", type=Path, default=None)
    lightgbm_submission_parser.add_argument("--train-candidate-path", type=Path, required=True)
    lightgbm_submission_parser.add_argument(
        "--extra-train-window",
        nargs=2,
        action="append",
        default=None,
        metavar=("CUTOFF", "CANDIDATE_CSV"),
        help=(
            "Additional leakage-safe training window as CUTOFF CANDIDATE_CSV. "
            "May be repeated for multi-week LightGBM training."
        ),
    )
    lightgbm_submission_parser.add_argument("--output-path", type=Path, default=None)
    lightgbm_submission_parser.add_argument("--validation-report-path", type=Path, default=None)
    lightgbm_submission_parser.add_argument("--report-path", type=Path, default=None)
    lightgbm_submission_parser.set_defaults(handler=_handle_generate_lightgbm_behavioral_submission)

    rolling_ranker_parser = subparsers.add_parser(
        "rolling-ranker-validation",
        help="Evaluate learned and deterministic rankers across rolling temporal windows.",
    )
    rolling_ranker_parser.add_argument(
        "--cutoffs",
        nargs="+",
        required=True,
        help="Evaluation cutoff dates, e.g. 2020-09-02 2020-09-09 2020-09-16.",
    )
    rolling_ranker_parser.add_argument("--horizon-days", type=int, default=7)
    rolling_ranker_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    rolling_ranker_parser.add_argument("--candidate-k", type=int, default=12)
    rolling_ranker_parser.add_argument("--k", type=int, default=12)
    rolling_ranker_parser.add_argument("--epochs", type=int, default=3)
    rolling_ranker_parser.add_argument("--learning-rate", type=float, default=0.01)
    rolling_ranker_parser.add_argument("--l2", type=float, default=0.001)
    rolling_ranker_parser.add_argument("--positive-weight", type=float, default=None)
    rolling_ranker_parser.add_argument("--max-auto-positive-weight", type=float, default=10.0)
    rolling_ranker_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    rolling_ranker_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    rolling_ranker_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    rolling_ranker_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap applied separately to all rolling windows.",
    )
    _add_seasonal_popularity_arguments(rolling_ranker_parser)
    _add_age_segment_popularity_arguments(rolling_ranker_parser)
    _add_garment_group_popularity_arguments(rolling_ranker_parser)
    _add_product_code_popularity_arguments(rolling_ranker_parser)
    _add_content_similarity_candidate_arguments(rolling_ranker_parser)
    rolling_ranker_parser.add_argument("--project-root", type=Path, default=None)
    rolling_ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    rolling_ranker_parser.add_argument("--report-path", type=Path, default=None)
    rolling_ranker_parser.set_defaults(handler=_handle_rolling_ranker_validation)

    learned_submission_parser = subparsers.add_parser(
        "generate-learned-ranker-submission",
        help="Train latest-window linear ranker and generate a validated Kaggle CSV.",
    )
    learned_submission_parser.add_argument("--horizon-days", type=int, default=7)
    learned_submission_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    learned_submission_parser.add_argument("--candidate-k", type=int, default=12)
    learned_submission_parser.add_argument("--k", type=int, default=12)
    learned_submission_parser.add_argument("--epochs", type=int, default=3)
    learned_submission_parser.add_argument("--learning-rate", type=float, default=0.01)
    learned_submission_parser.add_argument("--l2", type=float, default=0.001)
    learned_submission_parser.add_argument("--positive-weight", type=float, default=None)
    learned_submission_parser.add_argument("--max-auto-positive-weight", type=float, default=10.0)
    learned_submission_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    learned_submission_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    learned_submission_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    learned_submission_parser.add_argument("--project-root", type=Path, default=None)
    learned_submission_parser.add_argument("--raw-data-dir", type=Path, default=None)
    learned_submission_parser.add_argument("--train-candidate-output-path", type=Path, default=None)
    learned_submission_parser.add_argument("--train-candidate-report-path", type=Path, default=None)
    learned_submission_parser.add_argument("--output-path", type=Path, default=None)
    learned_submission_parser.add_argument("--validation-report-path", type=Path, default=None)
    learned_submission_parser.add_argument("--report-path", type=Path, default=None)
    learned_submission_parser.set_defaults(handler=_handle_generate_learned_ranker_submission)

    deterministic_submission_parser = subparsers.add_parser(
        "generate-deterministic-ranker-submission",
        help="Tune deterministic weights on the latest visible week and generate a CSV.",
    )
    deterministic_submission_parser.add_argument("--horizon-days", type=int, default=7)
    deterministic_submission_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    deterministic_submission_parser.add_argument("--candidate-k", type=int, default=12)
    deterministic_submission_parser.add_argument("--k", type=int, default=12)
    deterministic_submission_parser.add_argument(
        "--top-trials",
        type=int,
        default=10,
        help="Number of top tuning-window weight trials to retain in the report.",
    )
    deterministic_submission_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    deterministic_submission_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    deterministic_submission_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    deterministic_submission_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap for the latest-window weight-selection customers only.",
    )
    _add_seasonal_popularity_arguments(deterministic_submission_parser)
    _add_age_segment_popularity_arguments(deterministic_submission_parser)
    _add_garment_group_popularity_arguments(deterministic_submission_parser)
    _add_product_code_popularity_arguments(deterministic_submission_parser)
    _add_two_tower_candidate_arguments(deterministic_submission_parser)
    _add_two_tower_ranker_weight_arguments(deterministic_submission_parser)
    deterministic_submission_parser.add_argument("--project-root", type=Path, default=None)
    deterministic_submission_parser.add_argument("--raw-data-dir", type=Path, default=None)
    deterministic_submission_parser.add_argument(
        "--train-candidate-output-path", type=Path, default=None
    )
    deterministic_submission_parser.add_argument(
        "--train-candidate-report-path", type=Path, default=None
    )
    deterministic_submission_parser.add_argument("--output-path", type=Path, default=None)
    deterministic_submission_parser.add_argument(
        "--validation-report-path", type=Path, default=None
    )
    deterministic_submission_parser.add_argument("--report-path", type=Path, default=None)
    deterministic_submission_parser.set_defaults(
        handler=_handle_generate_deterministic_ranker_submission
    )

    two_tower_export_parser = subparsers.add_parser(
        "export-two-tower-examples",
        help="Export cutoff-safe two-tower positives, random negatives, and ID mappings.",
    )
    two_tower_export_parser.add_argument(
        "--cutoff",
        required=True,
        help="Exclusive training cutoff date, YYYY-MM-DD.",
    )
    two_tower_export_parser.add_argument("--horizon-days", type=int, default=7)
    two_tower_export_parser.add_argument("--negatives-per-positive", type=int, default=1)
    two_tower_export_parser.add_argument("--seed", type=int, default=42)
    two_tower_export_parser.add_argument(
        "--max-positive-examples",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke exports.",
    )
    two_tower_export_parser.add_argument(
        "--negative-sampling",
        default="random",
        choices=("random", "popularity", "mixed"),
        help="Negative sampling strategy for exported two-tower negatives.",
    )
    two_tower_export_parser.add_argument(
        "--positive-selection",
        default="first",
        choices=("first", "latest", "latest_customer"),
        help="Positive-pair selection strategy when --max-positive-examples is set.",
    )
    two_tower_export_parser.add_argument("--project-root", type=Path, default=None)
    two_tower_export_parser.add_argument("--raw-data-dir", type=Path, default=None)
    two_tower_export_parser.add_argument("--examples-path", type=Path, default=None)
    two_tower_export_parser.add_argument("--customer-mapping-path", type=Path, default=None)
    two_tower_export_parser.add_argument("--article-mapping-path", type=Path, default=None)
    two_tower_export_parser.add_argument("--report-path", type=Path, default=None)
    two_tower_export_parser.set_defaults(handler=_handle_export_two_tower_examples)

    two_tower_eval_parser = subparsers.add_parser(
        "evaluate-two-tower-retrieval",
        help="Train/evaluate the lightweight two-tower retrieval smoke model.",
    )
    two_tower_eval_parser.add_argument(
        "--cutoff",
        required=True,
        help="Validation cutoff date matching the exported example cutoff, YYYY-MM-DD.",
    )
    two_tower_eval_parser.add_argument("--horizon-days", type=int, default=7)
    two_tower_eval_parser.add_argument("--negatives-per-positive", type=int, default=1)
    two_tower_eval_parser.add_argument(
        "--negative-sampling",
        default="random",
        choices=("random", "popularity", "mixed"),
    )
    two_tower_eval_parser.add_argument(
        "--positive-selection",
        default="first",
        choices=("first", "latest", "latest_customer"),
        help="Positive-pair selection strategy used to infer default artifact paths.",
    )
    two_tower_eval_parser.add_argument("--seed", type=int, default=42)
    two_tower_eval_parser.add_argument("--embedding-dim", type=int, default=16)
    two_tower_eval_parser.add_argument("--epochs", type=int, default=3)
    two_tower_eval_parser.add_argument("--learning-rate", type=float, default=0.05)
    two_tower_eval_parser.add_argument("--l2", type=float, default=0.0)
    two_tower_eval_parser.add_argument(
        "--loss",
        default="logistic",
        choices=("logistic", "bpr"),
        help="Two-tower training objective. Defaults to pointwise logistic.",
    )
    two_tower_eval_parser.add_argument("--logq-correction-alpha", type=float, default=0.0)
    two_tower_eval_parser.add_argument(
        "--positive-recency-half-life-days",
        type=float,
        default=None,
        help="Optional half-life for recency-weighting exported training examples.",
    )
    two_tower_eval_parser.add_argument(
        "--recency-reference-date",
        default=None,
        help="Optional ISO date for recency weighting. Defaults to the validation cutoff.",
    )
    two_tower_eval_parser.add_argument("--k", type=int, default=12)
    two_tower_eval_parser.add_argument(
        "--evaluation-ks",
        type=int,
        nargs="+",
        default=[12, 50, 100],
        help="Candidate-recall cutoffs. Defaults to 12 50 100.",
    )
    two_tower_eval_parser.add_argument(
        "--max-positive-examples",
        type=int,
        default=None,
        help="Positive-example cap used to infer default exported artifact paths.",
    )
    two_tower_eval_parser.add_argument(
        "--max-training-examples",
        type=int,
        default=None,
        help="Optional deterministic cap on example rows consumed by SGD.",
    )
    two_tower_eval_parser.add_argument(
        "--max-eval-customers",
        type=int,
        default=1000,
        help=(
            "Optional deterministic cap for mapped validation customers. "
            "Use 0 for all mapped customers."
        ),
    )
    two_tower_eval_parser.add_argument(
        "--max-retrieval-articles",
        type=int,
        default=5000,
        help="Optional top-popular article pool cap for exact retrieval.",
    )
    two_tower_eval_parser.add_argument(
        "--popularity-prior-weight",
        type=float,
        default=0.0,
        help="Optional additive weight for a cutoff-safe recent-popularity score prior.",
    )
    two_tower_eval_parser.add_argument(
        "--popularity-prior-lookback-days",
        type=int,
        default=7,
        help="Pre-cutoff lookback window for the two-tower popularity score prior.",
    )
    two_tower_eval_parser.add_argument("--project-root", type=Path, default=None)
    two_tower_eval_parser.add_argument("--raw-data-dir", type=Path, default=None)
    two_tower_eval_parser.add_argument("--examples-path", type=Path, default=None)
    two_tower_eval_parser.add_argument("--customer-mapping-path", type=Path, default=None)
    two_tower_eval_parser.add_argument("--article-mapping-path", type=Path, default=None)
    two_tower_eval_parser.add_argument("--report-path", type=Path, default=None)
    two_tower_eval_parser.set_defaults(handler=_handle_evaluate_two_tower_retrieval)

    image_inventory_parser = subparsers.add_parser(
        "inventory-article-images",
        help="Map articles.csv IDs to local H&M image paths and report missing/malformed images.",
    )
    image_inventory_parser.add_argument("--project-root", type=Path, default=None)
    image_inventory_parser.add_argument("--raw-data-dir", type=Path, default=None)
    image_inventory_parser.add_argument("--manifest-path", type=Path, default=None)
    image_inventory_parser.add_argument("--report-path", type=Path, default=None)
    image_inventory_parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum missing/extra/malformed examples retained in the JSON report.",
    )
    image_inventory_parser.set_defaults(handler=_handle_inventory_article_images)

    article_content_parser = subparsers.add_parser(
        "export-article-content",
        help="Export article text fields and image paths for open-source embedding providers.",
    )
    article_content_parser.add_argument("--project-root", type=Path, default=None)
    article_content_parser.add_argument("--raw-data-dir", type=Path, default=None)
    article_content_parser.add_argument("--output-path", type=Path, default=None)
    article_content_parser.add_argument("--report-path", type=Path, default=None)
    article_content_parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Optional cap after applying article priority order.",
    )
    article_content_parser.add_argument(
        "--priority-cutoff",
        default=None,
        help="Exclusive YYYY-MM-DD cutoff for transaction-popularity article ordering.",
    )
    article_content_parser.add_argument(
        "--priority-lookback-days",
        type=int,
        default=None,
        help="Optional popularity lookback ending at --priority-cutoff.",
    )
    article_content_parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum missing-image/empty-text examples retained in the JSON report.",
    )
    article_content_parser.set_defaults(handler=_handle_export_article_content)

    article_embedding_parser = subparsers.add_parser(
        "generate-article-embeddings",
        help="Generate cached article embeddings with an optional open-source provider.",
    )
    article_embedding_parser.add_argument("--project-root", type=Path, default=None)
    article_embedding_parser.add_argument("--raw-data-dir", type=Path, default=None)
    article_embedding_parser.add_argument(
        "--provider",
        choices=("hf-clip",),
        default="hf-clip",
        help="Embedding provider backend. hf-clip uses transformers AutoModel/AutoProcessor.",
    )
    article_embedding_parser.add_argument(
        "--model-id",
        default="patrickjohncyh/fashion-clip",
        help="Open-source HuggingFace model ID. Defaults to FashionCLIP.",
    )
    article_embedding_parser.add_argument("--model-revision", default="main")
    article_embedding_parser.add_argument(
        "--embedding-kind",
        choices=("image", "text", "multimodal"),
        default="multimodal",
    )
    article_embedding_parser.add_argument("--device", default="auto")
    article_embedding_parser.add_argument("--batch-size", type=int, default=32)
    article_embedding_parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke embedding generation.",
    )
    article_embedding_parser.add_argument("--article-content-path", type=Path, default=None)
    article_embedding_parser.add_argument("--embeddings-path", type=Path, default=None)
    article_embedding_parser.add_argument("--article-mapping-path", type=Path, default=None)
    article_embedding_parser.add_argument("--manifest-path", type=Path, default=None)
    article_embedding_parser.add_argument(
        "--preprocessing",
        default=(
            "HuggingFace AutoProcessor defaults; JSONL float vectors; "
            "multimodal vectors average available text/image embeddings"
        ),
    )
    article_embedding_parser.add_argument(
        "--license-note",
        default="Check upstream HuggingFace model card/license before competition use.",
    )
    article_embedding_parser.set_defaults(handler=_handle_generate_article_embeddings)

    content_similarity_parser = subparsers.add_parser(
        "content-similarity-diagnostics",
        help="Evaluate cached article embeddings as a leakage-safe content candidate source.",
    )
    content_similarity_parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD")
    content_similarity_parser.add_argument("--horizon-days", type=int, default=7)
    content_similarity_parser.add_argument("--manifest-path", type=Path, required=True)
    content_similarity_parser.add_argument(
        "--source-name",
        default=MULTIMODAL_SIMILARITY_SOURCE,
    )
    content_similarity_parser.add_argument(
        "--evaluation-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTENT_SIMILARITY_EVALUATION_KS),
    )
    content_similarity_parser.add_argument(
        "--max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
    )
    content_similarity_parser.add_argument(
        "--popularity-prior-weight",
        type=float,
        default=0.0,
        help="Blend weight for pre-cutoff article popularity when reranking content candidates.",
    )
    content_similarity_parser.add_argument(
        "--popularity-lookback-days",
        type=int,
        default=None,
        help="Optional pre-cutoff popularity-prior lookback window.",
    )
    content_similarity_parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Optional content neighbor pool size before popularity-prior reranking.",
    )
    content_similarity_parser.add_argument(
        "--include-history",
        action="store_true",
        help="Do not filter articles already present in the customer's pre-cutoff history.",
    )
    content_similarity_parser.add_argument("--max-target-customers", type=int, default=None)
    content_similarity_parser.add_argument("--project-root", type=Path, default=None)
    content_similarity_parser.add_argument("--raw-data-dir", type=Path, default=None)
    content_similarity_parser.add_argument("--report-path", type=Path, default=None)
    content_similarity_parser.set_defaults(handler=_handle_content_similarity_diagnostics)
    return parser


def _add_content_similarity_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional cached content-similarity candidate-source arguments."""

    parser.add_argument(
        "--content-similarity-manifest-path",
        type=Path,
        default=None,
        help="Optional cached article-embedding manifest for content candidate rows.",
    )
    parser.add_argument(
        "--content-similarity-source-name",
        default=MULTIMODAL_SIMILARITY_SOURCE,
        help="Source name emitted for cached content-similarity candidates.",
    )
    parser.add_argument(
        "--content-similarity-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help="Recent unique history length for content query vectors.",
    )
    parser.add_argument(
        "--content-similarity-popularity-prior-weight",
        type=float,
        default=0.0,
        help="Blend weight for pre-cutoff article popularity in content reranking.",
    )
    parser.add_argument(
        "--content-similarity-popularity-lookback-days",
        type=int,
        default=None,
        help="Optional pre-cutoff popularity-prior lookback window for content reranking.",
    )
    parser.add_argument(
        "--content-similarity-candidate-pool-size",
        type=int,
        default=None,
        help="Optional content neighbor pool size before popularity-prior reranking.",
    )
    parser.add_argument(
        "--include-content-history",
        action="store_true",
        help="Do not filter pre-cutoff history articles from content candidates.",
    )


def _add_two_tower_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional trained two-tower candidate-source arguments."""

    parser.add_argument(
        "--include-two-tower-retrieval",
        action="store_true",
        help="Train the lightweight two-tower model and append two-tower candidate rows.",
    )
    parser.add_argument(
        "--two-tower-source-name",
        default=TWO_TOWER_RETRIEVAL_SOURCE,
        help="Source name emitted for two-tower retrieval candidate rows.",
    )
    parser.add_argument("--two-tower-examples-path", type=Path, default=None)
    parser.add_argument("--two-tower-customer-mapping-path", type=Path, default=None)
    parser.add_argument("--two-tower-article-mapping-path", type=Path, default=None)
    parser.add_argument("--two-tower-negatives-per-positive", type=int, default=1)
    parser.add_argument(
        "--two-tower-negative-sampling",
        default="random",
        choices=("random", "popularity", "mixed"),
    )
    parser.add_argument("--two-tower-seed", type=int, default=42)
    parser.add_argument(
        "--two-tower-positive-selection",
        default="latest",
        choices=("first", "latest", "latest_customer"),
    )
    parser.add_argument("--two-tower-max-positive-examples", type=int, default=100000)
    parser.add_argument("--two-tower-embedding-dim", type=int, default=32)
    parser.add_argument("--two-tower-epochs", type=int, default=80)
    parser.add_argument("--two-tower-learning-rate", type=float, default=0.05)
    parser.add_argument("--two-tower-l2", type=float, default=0.0)
    parser.add_argument(
        "--two-tower-loss",
        default="bpr",
        choices=("logistic", "bpr"),
    )
    parser.add_argument("--two-tower-logq-correction-alpha", type=float, default=0.0)
    parser.add_argument("--two-tower-max-training-examples", type=int, default=None)
    parser.add_argument("--two-tower-max-retrieval-articles", type=int, default=5000)


def _add_two_tower_ranker_weight_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional deterministic ranker weights for two-tower rows."""

    parser.add_argument(
        "--two-tower-ranker-presence-weight",
        type=float,
        default=None,
        help="Optional deterministic ranker additive weight for two-tower rows.",
    )
    parser.add_argument(
        "--two-tower-ranker-score-weight",
        type=float,
        default=None,
        help="Optional deterministic ranker score weight for two-tower rows.",
    )


def _add_seasonal_popularity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional shifted-window seasonal popularity source arguments."""

    parser.add_argument(
        "--include-seasonal-popularity",
        action="store_true",
        help=(
            "Include global article popularity from a historical shifted window, "
            "for example same-week-last-year popularity."
        ),
    )
    parser.add_argument(
        "--seasonal-shift-days",
        type=int,
        default=DEFAULT_SEASONAL_SHIFT_DAYS,
        help=(
            "Days between the cutoff and historical seasonal window start. "
            f"Defaults to {DEFAULT_SEASONAL_SHIFT_DAYS} for 2020 leap-year "
            "same-date-last-year H&M cutoffs."
        ),
    )
    parser.add_argument(
        "--seasonal-window-days",
        type=int,
        default=DEFAULT_SEASONAL_WINDOW_DAYS,
        help=(
            "Length of the shifted historical popularity window. "
            f"Defaults to {DEFAULT_SEASONAL_WINDOW_DAYS} days."
        ),
    )


def _add_age_segment_popularity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional age-segment popularity candidate-source arguments."""

    parser.add_argument(
        "--include-age-segment-popularity",
        action="store_true",
        help="Include recent article popularity among customers in the same age bucket.",
    )
    parser.add_argument(
        "--age-segment-bucket-size",
        type=int,
        default=DEFAULT_AGE_SEGMENT_BUCKET_SIZE,
        help="Age bucket width for segment popularity. Defaults to 10 years.",
    )
    parser.add_argument(
        "--age-segment-popularity-lookback-days",
        type=int,
        default=None,
        help=(
            "Optional pre-cutoff lookback for age-segment popularity. "
            "Defaults to global lookback."
        ),
    )


def _add_garment_group_popularity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional garment-group affinity popularity candidate-source arguments."""

    parser.add_argument(
        "--include-garment-group-popularity",
        action="store_true",
        help=(
            "Include recent article popularity from garment groups seen in each "
            "customer's pre-cutoff history."
        ),
    )
    parser.add_argument(
        "--garment-group-popularity-lookback-days",
        type=int,
        default=None,
        help=(
            "Optional pre-cutoff lookback for garment-group popularity. "
            "Defaults to global lookback."
        ),
    )
    parser.add_argument(
        "--garment-group-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique customer-history articles used for garment-group affinities. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )


def _add_product_code_popularity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional product-code affinity popularity candidate-source arguments."""

    parser.add_argument(
        "--include-product-code-popularity",
        action="store_true",
        help=(
            "Include recent article popularity from product codes seen in each "
            "customer's pre-cutoff history (same-product color/size variants)."
        ),
    )
    parser.add_argument(
        "--product-code-popularity-lookback-days",
        type=int,
        default=None,
        help=(
            "Optional pre-cutoff lookback for product-code popularity. "
            "Defaults to global lookback."
        ),
    )
    parser.add_argument(
        "--product-code-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique customer-history articles used for product-code affinities. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code from the selected command handler.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_pin_baseline_champion(args: argparse.Namespace) -> int:
    """Handle the ``pin-baseline-champion`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` when at least one candidate is registered; ``1`` otherwise.
    """

    paths = ProjectPaths.from_root(root=args.project_root)
    rolling_reports_dir = args.rolling_reports_dir or (paths.artifacts_dir / "ranker-baselines")
    if not rolling_reports_dir.is_absolute():
        rolling_reports_dir = paths.root / rolling_reports_dir
    cutoffs = tuple(args.rolling_cutoffs) if args.rolling_cutoffs else None
    report_path = args.report_path or paths.baseline_champion_report_path
    markdown_path = args.markdown_path or paths.baseline_champion_markdown_path
    if not report_path.is_absolute():
        report_path = paths.root / report_path
    if not markdown_path.is_absolute():
        markdown_path = paths.root / markdown_path

    discovered = discover_rolling_validation_candidates(
        rolling_reports_dir=rolling_reports_dir,
        cutoffs=cutoffs,
    )
    candidates = discovered
    extra_warnings: list[str] = []
    if args.merge_existing and report_path.exists():
        existing = load_baseline_champion_report(report_path)
        candidates = merge_candidates(existing.candidates, discovered)
    if not discovered:
        extra_warnings.append("no_rolling_reports_discovered")

    report = build_baseline_champion_report(
        candidates=candidates,
        target_leaderboard_map_at_k=args.target_leaderboard_map_at_k,
        target_tolerance=args.target_tolerance,
        extra_warnings=extra_warnings,
    )
    written_json = write_baseline_champion_report(report, report_path)
    written_markdown = write_baseline_champion_markdown(report, markdown_path)

    print(f"Rolling reports directory: {rolling_reports_dir}")
    print(f"Discovered candidates: {len(discovered)}")
    print(f"Total registered candidates: {len(report.candidates)}")
    if report.champion_index is None:
        print("Champion: (none selected)")
    else:
        champion = report.candidates[report.champion_index]
        print(f"Champion: {champion.name}")
        if champion.offline_metrics is not None:
            print(f"  Offline rolling mean MAP@K: " f"{champion.offline_metrics.mean_map_at_k:.5f}")
        if champion.leaderboard_public_map_at_k is not None:
            print(f"  Kaggle public LB MAP@K: {champion.leaderboard_public_map_at_k:.5f}")
    print(f"Rationale: {report.champion_rationale}")
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    print(f"JSON report written to: {written_json}")
    print(f"Markdown report written to: {written_markdown}")
    return 0 if report.candidates else 1


def _handle_compute_perfect_ranker_ceiling(args: argparse.Namespace) -> int:
    """Handle the ``compute-perfect-ranker-ceiling`` subcommand.

    The handler refuses to silently mismatch ``--cutoffs`` and ``--candidate-path``
    so the reported ceiling is always traceable to a specific candidate file.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    cutoffs = tuple(args.cutoffs)
    if not cutoffs:
        print("error: at least one --cutoff is required")
        return 1
    candidate_paths_arg = list(args.candidate_path or ())
    if not candidate_paths_arg:
        print("error: at least one --candidate-path is required")
        return 1
    if len(candidate_paths_arg) != len(cutoffs):
        print(
            "error: number of --candidate-path values must match number of "
            f"--cutoffs (got {len(candidate_paths_arg)} paths for {len(cutoffs)} cutoffs)"
        )
        return 1

    submission_customer_ids = set(load_submission_customer_ids(paths.raw_data_dir))
    resolved_candidate_paths: list[Path] = []
    splits: list[TemporalSplit] = []
    for cutoff, raw_candidate_path in zip(cutoffs, candidate_paths_arg, strict=True):
        candidate_path = (
            raw_candidate_path
            if raw_candidate_path.is_absolute()
            else (paths.root / raw_candidate_path)
        )
        if not candidate_path.exists():
            print(f"error: candidate CSV not found: {candidate_path}")
            return 1
        resolved_candidate_paths.append(candidate_path)
        splits.append(
            TemporalSplit(cutoff=date.fromisoformat(cutoff), horizon_days=int(args.horizon_days))
        )

    labels_by_cutoff = collect_validation_labels_for_splits(
        iter_transaction_events(paths.raw_data_dir), splits
    )
    bundles: list[PerfectRankerCutoffInput] = []
    for cutoff, candidate_path, split in zip(
        cutoffs, resolved_candidate_paths, splits, strict=True
    ):
        raw_labels = labels_by_cutoff[split.cutoff]
        target_ids = select_validation_label_customer_ids(
            validation_labels=raw_labels,
            submission_customer_ids=submission_customer_ids,
            max_target_customers=args.max_target_customers,
        )
        filtered_labels = {customer_id: raw_labels[customer_id] for customer_id in target_ids}
        bundles.append(
            PerfectRankerCutoffInput(
                cutoff=cutoff,
                candidate_path=candidate_path,
                validation_labels=filtered_labels,
            )
        )

    report = build_perfect_ranker_ceiling_report(bundles, k=int(args.k))

    report_path = args.report_path or paths.perfect_ranker_ceiling_report_path
    markdown_path = args.markdown_path or paths.perfect_ranker_ceiling_markdown_path
    if not report_path.is_absolute():
        report_path = paths.root / report_path
    if not markdown_path.is_absolute():
        markdown_path = paths.root / markdown_path
    written_json = write_perfect_ranker_ceiling_report(report, report_path)
    written_markdown = write_perfect_ranker_ceiling_markdown(report, markdown_path)

    print(f"Cutoffs evaluated: {', '.join(report.cutoffs)}")
    print(f"Mean oracle MAP@{report.k}: {report.mean_oracle_map_at_k:.5f}")
    print(f"  min: {report.min_oracle_map_at_k:.5f}, max: {report.max_oracle_map_at_k:.5f}")
    print(f"Mean oracle Recall@{report.k}: {report.mean_oracle_recall_at_k:.5f}")
    print(
        "Mean candidate label coverage (unbounded by k): "
        f"{report.mean_candidate_label_coverage:.5f}"
    )
    for ceiling in report.per_cutoff:
        print(
            f"  {ceiling.cutoff}: oracle MAP@{report.k}="
            f"{ceiling.mean_oracle_map_at_k:.5f} "
            f"recall={ceiling.mean_oracle_recall_at_k:.5f} "
            f"evaluated={ceiling.evaluated_customers} "
            f"no-candidates={ceiling.customers_without_any_candidate}"
        )
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    print(f"JSON report written to: {written_json}")
    print(f"Markdown report written to: {written_markdown}")
    return 0


def _handle_eda_report(args: argparse.Namespace) -> int:
    """Handle the ``eda-report`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the EDA report to disk.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    rolling_cutoffs = tuple(args.rolling_cutoffs)
    config = EdaReportConfig(
        rolling_cutoffs=rolling_cutoffs,
        top_hierarchy_values=int(args.top_hierarchy_values),
        top_busy_days=int(args.top_busy_days),
        segment_thresholds=EdaSegmentThresholds(
            cold_max_transactions=int(args.cold_max_transactions),
            sparse_max_transactions=int(args.sparse_max_transactions),
        ),
    )
    report_path = args.report_path or paths.eda_report_path
    markdown_path = args.markdown_path or paths.eda_report_markdown_path
    if not report_path.is_absolute():
        report_path = paths.root / report_path
    if not markdown_path.is_absolute():
        markdown_path = paths.root / markdown_path

    report = build_eda_report(paths.raw_data_dir, config=config)
    written_json = write_eda_report(report, report_path)
    written_markdown = write_eda_report_markdown(report, markdown_path)

    print(f"Raw data directory: {paths.raw_data_dir}")
    print(f"Transactions analyzed: {report.transactions.total_rows:,}")
    print("Date range: " f"{report.transactions.date_min} to {report.transactions.date_max}")
    print(f"Distinct customers (transactions): {report.transactions.distinct_customers:,}")
    print(f"Distinct articles (transactions): {report.transactions.distinct_articles:,}")
    print(
        "Submission segmentation: cold="
        f"{report.customers.submission_segment_counts.get('cold', 0):,}"
        ", sparse="
        f"{report.customers.submission_segment_counts.get('sparse', 0):,}"
        ", dense="
        f"{report.customers.submission_segment_counts.get('dense', 0):,}"
    )
    for cutoff, share in report.customers.cold_user_share_by_cutoff.items():
        cold_count = report.customers.cold_customer_counts_by_cutoff.get(cutoff, 0)
        print(f"Cold-user share @ {cutoff}: {share:.4f} ({cold_count:,} customers)")
    print(f"JSON report written to: {written_json}")
    print(f"Markdown report written to: {written_markdown}")
    return 0


def _handle_validate_data_contract(args: argparse.Namespace) -> int:
    """Handle the ``validate-data-contract`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` when the data contract is valid; otherwise ``1``.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    report_path = _resolve_report_path(paths, args.report_path)
    report = validate_hm_data_contract(paths.raw_data_dir)
    written_report_path = write_data_contract_report(report, report_path)

    print(f"Raw data directory: {paths.raw_data_dir}")
    print(f"Data contract valid: {report.valid}")
    print(f"Report written to: {written_report_path}")
    for result in report.files:
        status = "valid" if result.valid else "invalid"
        print(f"- {result.file_name}: {status}, rows={result.row_count}")
        for failure in result.failures:
            print(f"  - {failure}")
    if report.optional_images["exists"] and not report.optional_images["is_directory"]:
        print(f"- images: invalid optional path {report.optional_images['path']}")
    return 0 if report.valid else 1


def _handle_summarize_temporal_split(args: argparse.Namespace) -> int:
    """Handle the ``summarize-temporal-split`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the temporal split report.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    report_path = args.report_path or paths.temporal_split_report_path(args.cutoff)
    if not report_path.is_absolute():
        report_path = paths.root / report_path

    summary = summarize_temporal_split(iter_transaction_events(paths.raw_data_dir), split)
    written_report_path = write_temporal_split_summary(summary, report_path)
    print(f"Raw data directory: {paths.raw_data_dir}")
    print(f"Cutoff: {summary.cutoff}")
    print(f"Validation end exclusive: {summary.validation_end_exclusive}")
    print(f"Train rows: {summary.train_rows}")
    print(f"Validation rows: {summary.validation_rows}")
    print(f"Future rows: {summary.future_rows}")
    print(f"Report written to: {written_report_path}")
    return 0


def _handle_validate_submission(args: argparse.Namespace) -> int:
    """Handle the ``validate-submission`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` when the submission is valid; otherwise ``1``.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    report_path = args.report_path or paths.submission_validation_report_path(args.submission_path)
    if not report_path.is_absolute():
        report_path = paths.root / report_path

    result = validate_submission_file(
        submission_path=args.submission_path,
        expected_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        valid_article_ids=load_article_ids(paths.raw_data_dir),
        require_full_length=not args.allow_short,
    )
    written_report_path = write_submission_validation_report(result, report_path)
    print(f"Submission valid: {result.valid}")
    print(f"Rows: {result.row_count}")
    print(f"Expected customers: {result.expected_customer_count}")
    print(f"Report written to: {written_report_path}")
    for failure in result.failures:
        print(f"- {failure}")
    return 0 if result.valid else 1


def _handle_evaluate_baseline(args: argparse.Namespace) -> int:
    """Handle the ``evaluate-baseline`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after evaluating the baseline and writing its report.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    report_path = args.report_path or paths.baseline_report_path(
        cutoff=args.cutoff,
        lookback_days=args.popularity_lookback_days,
        k=args.k,
    )
    if not report_path.is_absolute():
        report_path = paths.root / report_path

    report = evaluate_repeat_popularity_baseline(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        target_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        k=args.k,
        popularity_lookback_days=args.popularity_lookback_days,
    )
    written_report_path = write_baseline_evaluation_report(report, report_path)
    print(f"Cutoff: {report.cutoff}")
    print(f"Validation end exclusive: {report.validation_end_exclusive}")
    print(f"MAP@{report.k}: {report.map_at_k:.8f}")
    print(f"Recall@{report.k}: {report.recall_at_k:.8f}")
    print(f"Prediction target customers: {report.diagnostics.target_customers}")
    print(f"Evaluated label customers: {report.diagnostics.evaluated_customers}")
    print(
        "Full-length prediction coverage: "
        f"{report.diagnostics.customers_with_full_length_predictions}/"
        f"{report.diagnostics.target_customers}"
    )
    print(f"Duplicate prediction rows: {report.diagnostics.duplicate_prediction_rows}")
    print(f"Report written to: {written_report_path}")
    return 0


def _handle_generate_baseline_submission(args: argparse.Namespace) -> int:
    """Handle the ``generate-baseline-submission`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` when the generated submission passes validation; otherwise ``1``.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    output_path = args.output_path or paths.baseline_submission_path(
        lookback_days=args.popularity_lookback_days,
        k=args.k,
    )
    output_path = _resolve_path_under_root(paths, output_path)

    target_customer_ids = load_submission_customer_ids_in_order(paths.raw_data_dir)
    submission = build_repeat_popularity_submission_baseline(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        target_customer_ids=target_customer_ids,
        k=args.k,
        popularity_lookback_days=args.popularity_lookback_days,
    )
    written_submission_path = write_submission_file(
        predictions_by_customer=submission.predictions.predictions,
        customer_ids=target_customer_ids,
        path=output_path,
        max_predictions=args.k,
    )

    validation_report_path = args.validation_report_path or paths.submission_validation_report_path(
        written_submission_path
    )
    validation_report_path = _resolve_path_under_root(paths, validation_report_path)
    validation_result = validate_submission_file(
        submission_path=written_submission_path,
        expected_customer_ids=set(target_customer_ids),
        valid_article_ids=load_article_ids(paths.raw_data_dir),
        max_predictions=args.k,
        require_full_length=True,
    )
    written_validation_report_path = write_submission_validation_report(
        validation_result,
        validation_report_path,
    )

    print(f"Max transaction date: {submission.max_transaction_date.isoformat()}")
    print(f"Training cutoff: {submission.training_cutoff.isoformat()}")
    print(f"Prediction target customers: {submission.diagnostics.target_customers}")
    print(
        "Full-length prediction coverage: "
        f"{submission.diagnostics.customers_with_full_length_predictions}/"
        f"{submission.diagnostics.target_customers}"
    )
    print(f"Duplicate prediction rows: {submission.diagnostics.duplicate_prediction_rows}")
    print(f"Submission written to: {written_submission_path}")
    print(f"Submission valid: {validation_result.valid}")
    print(f"Validation report written to: {written_validation_report_path}")
    for failure in validation_result.failures:
        print(f"- {failure}")
    return 0 if validation_result.valid else 1


def _handle_candidate_diagnostics(args: argparse.Namespace) -> int:
    """Handle the ``candidate-diagnostics`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the candidate diagnostics report.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    max_k = max(set(args.evaluation_ks))
    report_path = args.report_path or paths.candidate_diagnostics_report_path(
        cutoff=args.cutoff,
        lookback_days=args.popularity_lookback_days,
        max_k=max_k,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
    )
    report_path = _resolve_path_under_root(paths, report_path)

    report = evaluate_baseline_candidate_diagnostics(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        target_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        popularity_lookback_days=args.popularity_lookback_days,
        evaluation_ks=args.evaluation_ks,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
    )
    written_report_path = write_candidate_diagnostics_report(report, report_path)
    print(f"Cutoff: {report.cutoff}")
    print(f"Validation end exclusive: {report.validation_end_exclusive}")
    print(f"Target customers: {report.target_customers}")
    print(f"Evaluated label customers: {report.evaluated_customers}")
    for source in report.sources:
        recall_summary = ", ".join(
            f"recall@{k}={value:.6f}" for k, value in source.recall_at_k.items()
        )
        print(
            f"- {source.source}: MAP@12={source.map_at_12:.8f}, "
            f"{recall_summary}, coverage={source.candidate_coverage:.6f}"
        )
    print(f"Report written to: {written_report_path}")
    return 0


def _handle_export_candidates(args: argparse.Namespace) -> int:
    """Handle the ``export-candidates`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the candidate CSV and summary JSON.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    content_source_name = _effective_content_similarity_source_name(args)
    customer_age_segments = _load_customer_age_segments_if_enabled(paths, args)
    article_garment_groups = _load_article_garment_groups_if_enabled(paths, args)
    article_product_codes = _load_article_product_codes_if_enabled(paths, args)
    output_path = args.output_path or paths.candidate_export_path(
        cutoff=args.cutoff,
        k=args.k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        content_similarity_source_name=(
            content_source_name if args.content_similarity_manifest_path is not None else None
        ),
        content_similarity_manifest_path=args.content_similarity_manifest_path,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            args.content_similarity_candidate_pool_size
            if args.content_similarity_manifest_path is not None
            else None
        ),
        include_two_tower_retrieval=args.include_two_tower_retrieval,
        two_tower_config_slug=_two_tower_config_slug(args),
        max_target_customers=args.max_target_customers,
    )
    output_path = _resolve_path_under_root(paths, output_path)
    report_path = args.report_path or paths.candidate_export_report_path(output_path)
    report_path = _resolve_path_under_root(paths, report_path)
    content_manifest_path = _resolve_optional_path_under_root(
        paths, args.content_similarity_manifest_path
    )
    two_tower_model = _train_two_tower_candidate_model_if_enabled(paths, split, args)

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        submission_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        output_path=output_path,
        k=args.k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=args.seasonal_shift_days,
        seasonal_window_days=args.seasonal_window_days,
        include_age_segment_popularity=args.include_age_segment_popularity,
        customer_segment_by_id=customer_age_segments,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=_age_segment_popularity_lookback_days(args),
        include_garment_group_popularity=args.include_garment_group_popularity,
        article_garment_group_by_id=article_garment_groups,
        garment_group_popularity_lookback_days=_garment_group_popularity_lookback_days(args),
        garment_group_max_history_items=args.garment_group_max_history_items,
        include_product_code_popularity=args.include_product_code_popularity,
        article_product_code_by_id=article_product_codes,
        product_code_popularity_lookback_days=_product_code_popularity_lookback_days(args),
        product_code_max_history_items=args.product_code_max_history_items,
        content_similarity_manifest_path=content_manifest_path,
        content_similarity_source_name=content_source_name,
        content_similarity_max_history_items=args.content_similarity_max_history_items,
        content_similarity_exclude_history=not args.include_content_history,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
        ),
        content_similarity_candidate_pool_size=args.content_similarity_candidate_pool_size,
        two_tower_model=two_tower_model,
        two_tower_source_name=args.two_tower_source_name,
        two_tower_max_retrieval_articles=args.two_tower_max_retrieval_articles,
        max_target_customers=args.max_target_customers,
    )
    written_report_path = write_candidate_export_summary(summary, report_path)
    print(f"Cutoff: {summary.cutoff}")
    print(f"Validation end exclusive: {summary.validation_end_exclusive}")
    print(f"Target scope: {summary.target_scope}")
    print(f"Target customers: {summary.target_customers}")
    print(f"Rows written: {summary.rows_written}")
    for source, row_count in summary.source_row_counts.items():
        print(f"- {source}: rows={row_count}")
    print(f"Candidate CSV written to: {summary.output_path}")
    print(f"Summary report written to: {written_report_path}")
    return 0


def _handle_evaluate_ranker_baseline(args: argparse.Namespace) -> int:
    """Handle the ``evaluate-ranker-baseline`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing candidate export artifacts and ranker report.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    content_source_name = _effective_content_similarity_source_name(args)
    customer_age_segments = _load_customer_age_segments_if_enabled(paths, args)
    article_garment_groups = _load_article_garment_groups_if_enabled(paths, args)
    article_product_codes = _load_article_product_codes_if_enabled(paths, args)
    ranker_weights = _deterministic_ranker_weights_from_args(args)
    candidate_output_path = args.candidate_output_path or paths.candidate_export_path(
        cutoff=args.cutoff,
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        content_similarity_source_name=(
            content_source_name if args.content_similarity_manifest_path is not None else None
        ),
        content_similarity_manifest_path=args.content_similarity_manifest_path,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            args.content_similarity_candidate_pool_size
            if args.content_similarity_manifest_path is not None
            else None
        ),
        include_two_tower_retrieval=args.include_two_tower_retrieval,
        two_tower_config_slug=_two_tower_config_slug(args),
        max_target_customers=args.max_target_customers,
    )
    candidate_output_path = _resolve_path_under_root(paths, candidate_output_path)
    candidate_report_path = args.candidate_report_path or paths.candidate_export_report_path(
        candidate_output_path
    )
    candidate_report_path = _resolve_path_under_root(paths, candidate_report_path)
    content_manifest_path = _resolve_optional_path_under_root(
        paths, args.content_similarity_manifest_path
    )
    two_tower_model = _train_two_tower_candidate_model_if_enabled(paths, split, args)
    ranker_report_path = args.report_path or paths.ranker_baseline_report_path(
        cutoff=args.cutoff,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        content_similarity_source_name=(
            content_source_name if args.content_similarity_manifest_path is not None else None
        ),
        content_similarity_manifest_path=args.content_similarity_manifest_path,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            args.content_similarity_candidate_pool_size
            if args.content_similarity_manifest_path is not None
            else None
        ),
        include_two_tower_retrieval=args.include_two_tower_retrieval,
        two_tower_config_slug=_two_tower_config_slug(args),
    )
    ranker_report_path = _resolve_path_under_root(paths, ranker_report_path)

    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    export_summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        submission_customer_ids=submission_customer_ids,
        output_path=candidate_output_path,
        k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=args.seasonal_shift_days,
        seasonal_window_days=args.seasonal_window_days,
        include_age_segment_popularity=args.include_age_segment_popularity,
        customer_segment_by_id=customer_age_segments,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=_age_segment_popularity_lookback_days(args),
        include_garment_group_popularity=args.include_garment_group_popularity,
        article_garment_group_by_id=article_garment_groups,
        garment_group_popularity_lookback_days=_garment_group_popularity_lookback_days(args),
        garment_group_max_history_items=args.garment_group_max_history_items,
        include_product_code_popularity=args.include_product_code_popularity,
        article_product_code_by_id=article_product_codes,
        product_code_popularity_lookback_days=_product_code_popularity_lookback_days(args),
        product_code_max_history_items=args.product_code_max_history_items,
        content_similarity_manifest_path=content_manifest_path,
        content_similarity_source_name=content_source_name,
        content_similarity_max_history_items=args.content_similarity_max_history_items,
        content_similarity_exclude_history=not args.include_content_history,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
        ),
        content_similarity_candidate_pool_size=args.content_similarity_candidate_pool_size,
        two_tower_model=two_tower_model,
        two_tower_source_name=args.two_tower_source_name,
        two_tower_max_retrieval_articles=args.two_tower_max_retrieval_articles,
        max_target_customers=args.max_target_customers,
    )
    written_candidate_report_path = write_candidate_export_summary(
        export_summary, candidate_report_path
    )

    validation_data = summarize_temporal_split_with_labels(
        iter_transaction_events(paths.raw_data_dir), split
    )
    target_customer_ids = select_validation_label_customer_ids(
        validation_labels=validation_data.validation_labels,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_target_customers,
    )
    validation_labels = {
        customer_id: validation_data.validation_labels[customer_id]
        for customer_id in target_customer_ids
    }
    ranker_report = evaluate_deterministic_ranker_from_csv(
        candidate_path=candidate_output_path,
        validation_labels=validation_labels,
        split=split,
        k=args.k,
        weights=ranker_weights,
    )
    written_ranker_report_path = write_deterministic_ranker_report(
        ranker_report, ranker_report_path
    )
    print(f"Cutoff: {ranker_report.cutoff}")
    print(f"Validation end exclusive: {ranker_report.validation_end_exclusive}")
    print(f"Candidate rows: {ranker_report.candidate_rows}")
    print(f"Unique candidate pairs: {ranker_report.unique_candidate_pairs}")
    print(f"Evaluated customers: {ranker_report.evaluated_customers}")
    print(f"Ranker MAP@{ranker_report.k}: {ranker_report.map_at_k:.8f}")
    print(f"Baseline MAP@{ranker_report.k}: {ranker_report.baseline_map_at_k:.8f}")
    print(f"Delta MAP@{ranker_report.k}: {ranker_report.delta_map_at_k:.8f}")
    print(f"Ranker recall@{ranker_report.k}: {ranker_report.recall_at_k:.8f}")
    print(f"Candidate CSV written to: {export_summary.output_path}")
    print(f"Candidate summary written to: {written_candidate_report_path}")
    print(f"Ranker report written to: {written_ranker_report_path}")
    return 0


def _handle_tune_deterministic_ranker(args: argparse.Namespace) -> int:
    """Handle the ``tune-deterministic-ranker`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing candidate artifacts and the tuning report.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    evaluation_split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    train_split = (
        TemporalSplit.from_isoformat(args.train_cutoff, horizon_days=args.horizon_days)
        if args.train_cutoff is not None
        else previous_window_split(evaluation_split)
    )
    if train_split.validation_end > evaluation_split.cutoff:
        raise ValueError("training label window must end before the evaluation cutoff")

    train_candidate_path = (
        _resolve_path_under_root(paths, args.train_candidate_output_path)
        if args.train_candidate_output_path is not None
        else _ranker_candidate_export_path(paths, train_split, args)
    )
    eval_candidate_path = (
        _resolve_path_under_root(paths, args.eval_candidate_output_path)
        if args.eval_candidate_output_path is not None
        else _ranker_candidate_export_path(paths, evaluation_split, args)
    )
    content_source_name = _effective_content_similarity_source_name(args)
    report_path = args.report_path or paths.deterministic_ranker_tuning_report_path(
        train_cutoff=train_split.cutoff.isoformat(),
        evaluation_cutoff=args.cutoff,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        content_similarity_source_name=(
            content_source_name if args.content_similarity_manifest_path is not None else None
        ),
        content_similarity_manifest_path=args.content_similarity_manifest_path,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            args.content_similarity_candidate_pool_size
            if args.content_similarity_manifest_path is not None
            else None
        ),
        include_two_tower_retrieval=args.include_two_tower_retrieval,
        two_tower_config_slug=_two_tower_config_slug(args),
    )
    report_path = _resolve_path_under_root(paths, report_path)

    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    candidate_export_cache: dict[Path, Path] = {}
    written_train_candidate_report_path = _write_cached_ranker_candidate_export(
        candidate_export_cache,
        paths,
        train_split,
        submission_customer_ids,
        train_candidate_path,
        args,
    )
    written_eval_candidate_report_path = _write_cached_ranker_candidate_export(
        candidate_export_cache,
        paths,
        evaluation_split,
        submission_customer_ids,
        eval_candidate_path,
        args,
    )
    train_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=train_split,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_target_customers,
    )
    eval_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=evaluation_split,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_target_customers,
    )
    report = tune_deterministic_ranker_from_csv(
        train_candidate_path=train_candidate_path,
        train_validation_labels=train_labels,
        train_split=train_split,
        evaluation_candidate_path=eval_candidate_path,
        evaluation_validation_labels=eval_labels,
        evaluation_split=evaluation_split,
        k=args.k,
        candidate_k=args.candidate_k,
        grid=_deterministic_tuning_grid_from_args(args),
        default_weights=_deterministic_ranker_weights_from_args(args),
        top_n=args.top_trials,
    )
    written_report_path = write_deterministic_ranker_tuning_report(report, report_path)

    print(f"Tuning cutoff: {report.train_cutoff}")
    print(f"Tuning label end exclusive: {report.train_validation_end_exclusive}")
    print(f"Evaluation cutoff: {report.evaluation_cutoff}")
    print(f"Evaluation end exclusive: {report.evaluation_end_exclusive}")
    print(f"Weight trials: {report.trial_count}")
    print(f"Default train MAP@{report.k}: {report.default_train.map_at_k:.8f}")
    print(f"Selected train MAP@{report.k}: {report.selected_train.map_at_k:.8f}")
    print(f"Default eval MAP@{report.k}: {report.default_evaluation.map_at_k:.8f}")
    print(f"Selected eval MAP@{report.k}: {report.selected_evaluation.map_at_k:.8f}")
    print(
        f"Delta selected vs default eval MAP@{report.k}: "
        f"{report.delta_selected_vs_default_map_at_k:.8f}"
    )
    print(f"Selected eval recall@{report.k}: {report.selected_evaluation.recall_at_k:.8f}")
    print(f"Train candidate CSV: {train_candidate_path}")
    print(f"Eval candidate CSV: {eval_candidate_path}")
    print(f"Train candidate summary: {written_train_candidate_report_path}")
    print(f"Eval candidate summary: {written_eval_candidate_report_path}")
    print(f"Tuning report written to: {written_report_path}")
    return 0


def _handle_evaluate_learned_ranker_baseline(args: argparse.Namespace) -> int:
    """Handle the ``evaluate-learned-ranker-baseline`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after training/evaluating and writing artifacts.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    evaluation_split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    content_source_name = _effective_content_similarity_source_name(args)
    customer_age_segments = _load_customer_age_segments_if_enabled(paths, args)
    article_garment_groups = _load_article_garment_groups_if_enabled(paths, args)
    article_product_codes = _load_article_product_codes_if_enabled(paths, args)
    train_split = (
        TemporalSplit.from_isoformat(args.train_cutoff, horizon_days=args.horizon_days)
        if args.train_cutoff is not None
        else previous_window_split(evaluation_split)
    )
    if train_split.validation_end > evaluation_split.cutoff:
        raise ValueError("training label window must end before the evaluation cutoff")

    train_candidate_path = args.train_candidate_output_path or paths.candidate_export_path(
        cutoff=train_split.cutoff.isoformat(),
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        content_similarity_source_name=(
            content_source_name if args.content_similarity_manifest_path is not None else None
        ),
        content_similarity_manifest_path=args.content_similarity_manifest_path,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            args.content_similarity_candidate_pool_size
            if args.content_similarity_manifest_path is not None
            else None
        ),
        max_target_customers=args.max_target_customers,
    )
    eval_candidate_path = args.eval_candidate_output_path or paths.candidate_export_path(
        cutoff=args.cutoff,
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        content_similarity_source_name=(
            content_source_name if args.content_similarity_manifest_path is not None else None
        ),
        content_similarity_manifest_path=args.content_similarity_manifest_path,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            args.content_similarity_candidate_pool_size
            if args.content_similarity_manifest_path is not None
            else None
        ),
        max_target_customers=args.max_target_customers,
    )
    train_candidate_path = _resolve_path_under_root(paths, train_candidate_path)
    eval_candidate_path = _resolve_path_under_root(paths, eval_candidate_path)
    train_candidate_report_path = (
        args.train_candidate_report_path or paths.candidate_export_report_path(train_candidate_path)
    )
    eval_candidate_report_path = (
        args.eval_candidate_report_path or paths.candidate_export_report_path(eval_candidate_path)
    )
    train_candidate_report_path = _resolve_path_under_root(paths, train_candidate_report_path)
    eval_candidate_report_path = _resolve_path_under_root(paths, eval_candidate_report_path)
    content_manifest_path = _resolve_optional_path_under_root(
        paths, args.content_similarity_manifest_path
    )
    report_path = args.report_path or paths.learned_ranker_baseline_report_path(
        train_cutoff=train_split.cutoff.isoformat(),
        evaluation_cutoff=args.cutoff,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
        config_slug=_learned_ranker_config_slug(args),
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        content_similarity_source_name=(
            content_source_name if args.content_similarity_manifest_path is not None else None
        ),
        content_similarity_manifest_path=args.content_similarity_manifest_path,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            args.content_similarity_candidate_pool_size
            if args.content_similarity_manifest_path is not None
            else None
        ),
    )
    report_path = _resolve_path_under_root(paths, report_path)

    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    train_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=train_split,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_target_customers,
    )
    eval_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=evaluation_split,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_target_customers,
    )

    train_export_summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=train_split,
        submission_customer_ids=submission_customer_ids,
        output_path=train_candidate_path,
        k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=args.seasonal_shift_days,
        seasonal_window_days=args.seasonal_window_days,
        include_age_segment_popularity=args.include_age_segment_popularity,
        customer_segment_by_id=customer_age_segments,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=_age_segment_popularity_lookback_days(args),
        include_garment_group_popularity=args.include_garment_group_popularity,
        article_garment_group_by_id=article_garment_groups,
        garment_group_popularity_lookback_days=_garment_group_popularity_lookback_days(args),
        garment_group_max_history_items=args.garment_group_max_history_items,
        include_product_code_popularity=args.include_product_code_popularity,
        article_product_code_by_id=article_product_codes,
        product_code_popularity_lookback_days=_product_code_popularity_lookback_days(args),
        product_code_max_history_items=args.product_code_max_history_items,
        content_similarity_manifest_path=content_manifest_path,
        content_similarity_source_name=content_source_name,
        content_similarity_max_history_items=args.content_similarity_max_history_items,
        content_similarity_exclude_history=not args.include_content_history,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
        ),
        content_similarity_candidate_pool_size=args.content_similarity_candidate_pool_size,
        max_target_customers=args.max_target_customers,
    )
    eval_export_summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=evaluation_split,
        submission_customer_ids=submission_customer_ids,
        output_path=eval_candidate_path,
        k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=args.seasonal_shift_days,
        seasonal_window_days=args.seasonal_window_days,
        include_age_segment_popularity=args.include_age_segment_popularity,
        customer_segment_by_id=customer_age_segments,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=_age_segment_popularity_lookback_days(args),
        include_garment_group_popularity=args.include_garment_group_popularity,
        article_garment_group_by_id=article_garment_groups,
        garment_group_popularity_lookback_days=_garment_group_popularity_lookback_days(args),
        garment_group_max_history_items=args.garment_group_max_history_items,
        include_product_code_popularity=args.include_product_code_popularity,
        article_product_code_by_id=article_product_codes,
        product_code_popularity_lookback_days=_product_code_popularity_lookback_days(args),
        product_code_max_history_items=args.product_code_max_history_items,
        content_similarity_manifest_path=content_manifest_path,
        content_similarity_source_name=content_source_name,
        content_similarity_max_history_items=args.content_similarity_max_history_items,
        content_similarity_exclude_history=not args.include_content_history,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
        ),
        content_similarity_candidate_pool_size=args.content_similarity_candidate_pool_size,
        max_target_customers=args.max_target_customers,
    )
    written_train_candidate_report_path = write_candidate_export_summary(
        train_export_summary, train_candidate_report_path
    )
    written_eval_candidate_report_path = write_candidate_export_summary(
        eval_export_summary, eval_candidate_report_path
    )

    config = LinearRankerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        positive_weight=args.positive_weight,
        max_auto_positive_weight=args.max_auto_positive_weight,
    )
    training_result = train_linear_ranker_from_csv(
        candidate_path=train_candidate_path,
        validation_labels=train_labels,
        config=config,
    )
    evaluation = evaluate_linear_ranker_from_csv(
        candidate_path=eval_candidate_path,
        validation_labels=eval_labels,
        model=training_result.model,
        k=args.k,
    )
    report = build_learned_linear_ranker_report(
        train_split=train_split,
        evaluation_split=evaluation_split,
        k=args.k,
        candidate_k=args.candidate_k,
        config=config,
        training_result=training_result,
        evaluation=evaluation,
    )
    written_report_path = write_learned_linear_ranker_report(report, report_path)

    print(f"Training cutoff: {report.train_cutoff}")
    print(f"Training label end exclusive: {report.train_validation_end_exclusive}")
    print(f"Evaluation cutoff: {report.evaluation_cutoff}")
    print(f"Evaluation end exclusive: {report.evaluation_end_exclusive}")
    print(f"Training pairs: {report.training.unique_candidate_pairs}")
    print(f"Training positives: {report.training.positive_pairs}")
    print(f"Evaluation pairs: {report.evaluation.unique_candidate_pairs}")
    print(f"Evaluated customers: {report.evaluation.evaluated_customers}")
    print(f"Learned MAP@{report.k}: {report.evaluation.map_at_k:.8f}")
    print(f"Deterministic MAP@{report.k}: {report.evaluation.deterministic_map_at_k:.8f}")
    print(f"Source-order MAP@{report.k}: {report.evaluation.baseline_map_at_k:.8f}")
    print(
        f"Delta vs deterministic MAP@{report.k}: "
        f"{report.evaluation.delta_vs_deterministic_map_at_k:.8f}"
    )
    print(f"Train candidate CSV written to: {train_export_summary.output_path}")
    print(f"Eval candidate CSV written to: {eval_export_summary.output_path}")
    print(f"Train candidate summary written to: {written_train_candidate_report_path}")
    print(f"Eval candidate summary written to: {written_eval_candidate_report_path}")
    print(f"Learned ranker report written to: {written_report_path}")
    return 0


def _handle_evaluate_lightgbm_behavioral_ranker(args: argparse.Namespace) -> int:
    """Handle the ``evaluate-lightgbm-behavioral-ranker`` subcommand."""

    progress = (
        (lambda message: print(f"[lightgbm] {message}", flush=True)) if args.progress else None
    )
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    evaluation_split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    train_split = (
        TemporalSplit.from_isoformat(args.train_cutoff, horizon_days=args.horizon_days)
        if args.train_cutoff is not None
        else previous_window_split(evaluation_split)
    )
    if train_split.validation_end > evaluation_split.cutoff:
        raise ValueError("training label window must end before the evaluation cutoff")

    train_candidate_path = _resolve_path_under_root(paths, args.train_candidate_path)
    eval_candidate_path = _resolve_path_under_root(paths, args.eval_candidate_path)
    if progress is not None:
        progress("validating candidate export metadata")
    _validate_candidate_export_metadata(train_candidate_path, train_split, "training")
    _validate_candidate_export_metadata(eval_candidate_path, evaluation_split, "evaluation")
    report_path = args.report_path or _lightgbm_behavioral_ranker_report_path(paths, args)
    report_path = _resolve_path_under_root(paths, report_path)

    if progress is not None:
        progress("loading sample submission customer universe")
    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    if progress is not None:
        progress("loading training validation labels")
    train_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=train_split,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_train_customers,
    )
    if progress is not None:
        progress(f"loaded training labels for {len(train_labels)} customers")
        progress("loading evaluation validation labels")
    eval_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=evaluation_split,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_eval_customers,
    )
    if progress is not None:
        progress(f"loaded evaluation labels for {len(eval_labels)} customers")
        progress("loading article metadata attributes")
    article_attributes_by_id = load_article_attribute_maps(paths.raw_data_dir)
    if progress is not None:
        progress(f"loaded article metadata for {len(article_attributes_by_id)} articles")
    config = LightGBMBehavioralRankerConfig(
        k=args.k,
        negative_per_positive=args.negative_per_positive,
        blend_lambda=args.blend_lambda,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l2=args.lambda_l2,
        seed=args.seed,
        num_threads=args.num_threads,
        chunk_customers=args.chunk_customers,
        deterministic_weights=_lightgbm_behavioral_ranker_weights_from_args(args),
        objective=args.objective,
    )
    report = evaluate_lightgbm_behavioral_ranker_from_csv(
        transaction_iter_factory=lambda: iter_transactions(paths.raw_data_dir),
        train_split=train_split,
        evaluation_split=evaluation_split,
        train_candidate_path=train_candidate_path,
        evaluation_candidate_path=eval_candidate_path,
        train_validation_labels=train_labels,
        evaluation_validation_labels=eval_labels,
        article_attributes_by_id=article_attributes_by_id,
        config=config,
        progress_callback=progress,
    )
    written_report_path = write_lightgbm_behavioral_ranker_report(report, report_path)

    print(f"Training cutoff: {report.train_cutoff}")
    print(f"Training label end exclusive: {report.train_validation_end_exclusive}")
    print(f"Evaluation cutoff: {report.evaluation_cutoff}")
    print(f"Evaluation end exclusive: {report.evaluation_end_exclusive}")
    print(f"Training pairs: {report.train_unique_candidate_pairs}")
    print(f"Training positives: {report.train_positive_pairs}")
    print(f"Evaluation pairs: {report.evaluation_unique_candidate_pairs}")
    print(f"Evaluated customers: {report.evaluated_customers}")
    print(f"Deterministic MAP@{config.k}: {report.deterministic_map_at_k:.8f}")
    print(f"LightGBM-only MAP@{config.k}: {report.model_only_map_at_k:.8f}")
    print(f"Blend MAP@{config.k}: {report.blend_map_at_k:.8f}")
    print(
        f"Delta blend vs deterministic MAP@{config.k}: "
        f"{report.delta_vs_deterministic_map_at_k:.8f}"
    )
    print(f"Blend recall@{config.k}: {report.blend_recall_at_k:.8f}")
    print(f"Train candidate CSV: {train_candidate_path}")
    print(f"Eval candidate CSV: {eval_candidate_path}")
    print(f"LightGBM behavioral ranker report written to: {written_report_path}")
    return 0


def _handle_generate_lightgbm_behavioral_submission(args: argparse.Namespace) -> int:
    """Handle the rich LightGBM behavioral-ranker submission command."""

    progress = (
        (lambda message: print(f"[lightgbm-submission] {message}", flush=True))
        if args.progress
        else None
    )
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    train_split = TemporalSplit.from_isoformat(args.train_cutoff, horizon_days=args.horizon_days)
    train_candidate_path = _resolve_path_under_root(paths, args.train_candidate_path)
    training_window_specs = [(train_split, train_candidate_path)]
    for extra_cutoff, extra_candidate_path in args.extra_train_window or ():
        extra_split = TemporalSplit.from_isoformat(extra_cutoff, horizon_days=args.horizon_days)
        training_window_specs.append(
            (extra_split, _resolve_path_under_root(paths, Path(extra_candidate_path)))
        )
    if progress is not None:
        progress("validating training candidate export metadata")
    for window_index, (window_split, window_candidate_path) in enumerate(
        training_window_specs,
        start=1,
    ):
        _validate_candidate_export_metadata(
            window_candidate_path,
            window_split,
            f"training window {window_index}",
        )

    slug = _lightgbm_behavioral_submission_slug(args)
    output_path = _resolve_path_under_root(
        paths,
        args.output_path or (paths.submissions_dir / f"{slug}.csv"),
    )
    validation_report_path = _resolve_path_under_root(
        paths,
        args.validation_report_path or paths.submission_validation_report_path(output_path),
    )
    report_path = _resolve_path_under_root(
        paths,
        args.report_path or (paths.artifacts_dir / "ranker-submissions" / f"{slug}.json"),
    )

    if progress is not None:
        progress("loading sample submission customer universe")
    target_customer_ids = load_submission_customer_ids_in_order(paths.raw_data_dir)
    if args.max_target_customers is not None:
        if args.max_target_customers <= 0:
            raise ValueError("max-target-customers must be positive when provided")
        target_customer_ids = target_customer_ids[: args.max_target_customers]
    submission_customer_id_set = load_submission_customer_ids(paths.raw_data_dir)

    training_windows: list[LightGBMBehavioralTrainingWindow] = []
    for window_index, (window_split, window_candidate_path) in enumerate(
        training_window_specs,
        start=1,
    ):
        if progress is not None:
            progress(
                f"loading training validation labels for window {window_index}: "
                f"{window_split.cutoff}"
            )
        window_labels = _validation_labels_for_split(
            raw_data_dir=paths.raw_data_dir,
            split=window_split,
            submission_customer_ids=submission_customer_id_set,
            max_target_customers=args.max_train_customers,
        )
        if progress is not None:
            progress(
                f"loaded training labels for window {window_index}: "
                f"{len(window_labels)} customers"
            )
        training_windows.append(
            LightGBMBehavioralTrainingWindow(
                split=window_split,
                candidate_path=window_candidate_path,
                validation_labels=window_labels,
            )
        )
    if progress is not None:
        progress("loading article metadata attributes")
    article_attributes_by_id = load_article_attribute_maps(paths.raw_data_dir)
    if progress is not None:
        progress(f"loaded article metadata for {len(article_attributes_by_id)} articles")

    config = LightGBMBehavioralRankerConfig(
        k=args.k,
        negative_per_positive=args.negative_per_positive,
        blend_lambda=args.blend_lambda,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l2=args.lambda_l2,
        seed=args.seed,
        num_threads=args.num_threads,
        chunk_customers=args.chunk_customers,
        deterministic_weights=_lightgbm_behavioral_ranker_weights_from_args(args),
    )
    np, model, training_summary = train_lightgbm_behavioral_ranker_from_windows(
        transaction_iter_factory=lambda: iter_transactions(paths.raw_data_dir),
        training_windows=tuple(training_windows),
        article_attributes_by_id=article_attributes_by_id,
        config=config,
        progress_callback=progress,
    )

    if progress is not None:
        progress("finding final training cutoff")
    max_transaction_date = find_max_transaction_date(
        iter_transaction_events(paths.raw_data_dir),
        progress_interval=args.transaction_progress_interval,
        progress_callback=(
            None
            if progress is None
            else lambda rows: progress(f"scanned transactions for max date: {rows}")
        ),
    )
    final_split = TemporalSplit(
        cutoff=max_transaction_date + timedelta(days=1),
        horizon_days=args.horizon_days,
    )
    final_two_tower_model = _train_two_tower_candidate_model_if_enabled(paths, final_split, args)
    customer_segment_by_id = _load_customer_age_segments_if_enabled(paths, args)
    article_garment_group_by_id = _load_article_garment_groups_if_enabled(paths, args)
    article_product_code_by_id = _load_article_product_codes_if_enabled(paths, args)
    submission = build_lightgbm_behavioral_ranker_submission_predictions(
        transaction_iter_factory=lambda: iter_transactions(paths.raw_data_dir),
        split=final_split,
        target_customer_ids=target_customer_ids,
        np=np,
        model=model,
        config=config,
        article_attributes_by_id=article_attributes_by_id,
        k=args.k,
        candidate_k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=args.seasonal_shift_days,
        seasonal_window_days=args.seasonal_window_days,
        include_age_segment_popularity=args.include_age_segment_popularity,
        customer_segment_by_id=customer_segment_by_id,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        article_garment_group_by_id=article_garment_group_by_id,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=args.garment_group_max_history_items,
        include_product_code_popularity=args.include_product_code_popularity,
        article_product_code_by_id=article_product_code_by_id,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=args.product_code_max_history_items,
        two_tower_model=final_two_tower_model,
        two_tower_source_name=TWO_TOWER_RETRIEVAL_SOURCE,
        two_tower_max_retrieval_articles=args.two_tower_max_retrieval_articles,
        max_transaction_date=max_transaction_date,
        transaction_progress_interval=args.transaction_progress_interval,
        transaction_progress_callback=(
            None
            if progress is None
            else lambda phase, rows: progress(f"{phase} scanned transactions: {rows}")
        ),
        status_callback=progress,
        progress_interval=args.prediction_progress_interval,
        progress_callback=(
            None
            if progress is None
            else lambda completed, total: progress(f"scored final customers: {completed}/{total}")
        ),
    )
    written_submission_path = write_submission_file(
        submission.predictions,
        target_customer_ids,
        output_path,
        max_predictions=args.k,
    )

    validation_result = None
    written_validation_report_path = None
    if args.max_target_customers is None:
        validation_result = validate_submission_file(
            submission_path=written_submission_path,
            expected_customer_ids=submission_customer_id_set,
            valid_article_ids=load_article_ids(paths.raw_data_dir),
            require_full_length=True,
        )
        written_validation_report_path = write_submission_validation_report(
            validation_result,
            validation_report_path,
        )
    report = {
        "experiment": slug,
        "train_cutoff": train_split.cutoff.isoformat(),
        "train_validation_end_exclusive": train_split.validation_end.isoformat(),
        "train_window_count": len(training_windows),
        "extra_train_windows": tuple(
            {
                "cutoff": window.split.cutoff.isoformat(),
                "validation_end_exclusive": window.split.validation_end.isoformat(),
                "candidate_path": str(window.candidate_path),
            }
            for window in training_windows[1:]
        ),
        "final_training_cutoff": final_split.cutoff.isoformat(),
        "max_transaction_date": max_transaction_date.isoformat(),
        "k": args.k,
        "candidate_k": args.candidate_k,
        "popularity_lookback_days": args.popularity_lookback_days,
        "include_co_visitation": not args.no_co_visitation,
        "include_seasonal_popularity": args.include_seasonal_popularity,
        "seasonal_shift_days": (
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        "seasonal_window_days": (
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        "include_age_segment_popularity": args.include_age_segment_popularity,
        "include_garment_group_popularity": args.include_garment_group_popularity,
        "include_two_tower_retrieval": args.include_two_tower_retrieval,
        "two_tower_source_name": (
            TWO_TOWER_RETRIEVAL_SOURCE if args.include_two_tower_retrieval else None
        ),
        "two_tower_max_retrieval_articles": (
            args.two_tower_max_retrieval_articles if args.include_two_tower_retrieval else None
        ),
        "max_train_customers": args.max_train_customers,
        "max_target_customers": args.max_target_customers,
        "training": asdict(training_summary),
        "submission": asdict(submission.diagnostics),
        "submission_path": str(written_submission_path),
        "validation_report_path": (
            str(written_validation_report_path) if written_validation_report_path else None
        ),
        "validation": asdict(validation_result) if validation_result is not None else None,
        "valid": bool(validation_result.valid) if validation_result is not None else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"LightGBM behavioral submission written to: {written_submission_path}")
    if validation_result is not None:
        print(f"Submission valid: {validation_result.valid}")
        print(f"Validation report written to: {written_validation_report_path}")
    else:
        print("Submission validation skipped because --max-target-customers was set")
    print(f"LightGBM behavioral submission report written to: {report_path}")
    return 0 if validation_result is None or validation_result.valid else 1


def _validate_candidate_export_metadata(
    candidate_path: Path,
    split: TemporalSplit,
    role: str,
) -> None:
    """Validate that a candidate CSV's sidecar metadata matches the split.

    This guards the optional LightGBM evaluator from combining cutoff-safe
    behavioral features with stale or future-leaking precomputed source ranks.
    """

    metadata_path = candidate_path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError(f"{role} candidate metadata JSON is required: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cutoff = metadata.get("cutoff")
    if cutoff != split.cutoff.isoformat():
        raise ValueError(
            f"{role} candidate metadata cutoff {cutoff!r} does not match "
            f"requested split cutoff {split.cutoff.isoformat()!r}"
        )
    horizon_days = metadata.get("horizon_days")
    if horizon_days != split.horizon_days:
        raise ValueError(
            f"{role} candidate metadata horizon_days {horizon_days!r} does not match "
            f"requested horizon_days {split.horizon_days!r}"
        )
    validation_end = metadata.get("validation_end_exclusive")
    if validation_end != split.validation_end.isoformat():
        raise ValueError(
            f"{role} candidate metadata validation_end_exclusive {validation_end!r} "
            f"does not match requested validation end {split.validation_end.isoformat()!r}"
        )


def _handle_rolling_ranker_validation(args: argparse.Namespace) -> int:
    """Handle the ``rolling-ranker-validation`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the rolling validation report and candidate artifacts.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    cutoffs = tuple(args.cutoffs)
    content_source_name = _effective_content_similarity_source_name(args)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("rolling evaluation cutoffs must be unique")
    if tuple(sorted(cutoffs)) != cutoffs:
        raise ValueError("rolling evaluation cutoffs must be in ascending order")
    report_path = args.report_path or paths.rolling_ranker_validation_report_path(
        cutoffs=cutoffs,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
        config_slug=_learned_ranker_config_slug(args),
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        content_similarity_source_name=(
            content_source_name if args.content_similarity_manifest_path is not None else None
        ),
        content_similarity_manifest_path=args.content_similarity_manifest_path,
        content_similarity_popularity_prior_weight=(
            args.content_similarity_popularity_prior_weight
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            args.content_similarity_popularity_lookback_days
            if args.content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            args.content_similarity_candidate_pool_size
            if args.content_similarity_manifest_path is not None
            else None
        ),
    )
    report_path = _resolve_path_under_root(paths, report_path)

    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    config = LinearRankerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        positive_weight=args.positive_weight,
        max_auto_positive_weight=args.max_auto_positive_weight,
    )

    candidate_export_cache: dict[Path, Path] = {}
    labels_cache: dict[tuple[str, int], dict[str, tuple[str, ...]]] = {}
    learned_reports = []

    print(
        "Starting rolling ranker validation. "
        f"cutoffs={', '.join(cutoffs)}, "
        f"max_target_customers={args.max_target_customers}, "
        f"include_co_visitation={not args.no_co_visitation}. "
        "This command scans transactions for each unique train/eval window.",
        flush=True,
    )

    for window_index, cutoff in enumerate(cutoffs, start=1):
        evaluation_split = TemporalSplit.from_isoformat(cutoff, horizon_days=args.horizon_days)
        train_split = previous_window_split(evaluation_split)
        if train_split.validation_end > evaluation_split.cutoff:
            raise ValueError("training label window must end before the evaluation cutoff")
        print(
            f"[{window_index}/{len(cutoffs)}] evaluation={evaluation_split.cutoff.isoformat()} "
            f"train={train_split.cutoff.isoformat()}.."
            f"{train_split.validation_end.isoformat()} exclusive",
            flush=True,
        )

        train_candidate_path = _ranker_candidate_export_path(
            paths=paths,
            split=train_split,
            args=args,
        )
        eval_candidate_path = _ranker_candidate_export_path(
            paths=paths,
            split=evaluation_split,
            args=args,
        )
        print(
            f"[{window_index}/{len(cutoffs)}] writing/reusing train candidates: "
            f"{train_candidate_path}",
            flush=True,
        )
        _write_cached_ranker_candidate_export(
            cache=candidate_export_cache,
            paths=paths,
            split=train_split,
            submission_customer_ids=submission_customer_ids,
            output_path=train_candidate_path,
            args=args,
        )
        print(
            f"[{window_index}/{len(cutoffs)}] writing/reusing eval candidates: "
            f"{eval_candidate_path}",
            flush=True,
        )
        _write_cached_ranker_candidate_export(
            cache=candidate_export_cache,
            paths=paths,
            split=evaluation_split,
            submission_customer_ids=submission_customer_ids,
            output_path=eval_candidate_path,
            args=args,
        )
        print(
            f"[{window_index}/{len(cutoffs)}] collecting labels and training linear ranker",
            flush=True,
        )
        train_labels = _cached_validation_labels_for_split(
            cache=labels_cache,
            raw_data_dir=paths.raw_data_dir,
            split=train_split,
            submission_customer_ids=submission_customer_ids,
            max_target_customers=args.max_target_customers,
        )
        eval_labels = _cached_validation_labels_for_split(
            cache=labels_cache,
            raw_data_dir=paths.raw_data_dir,
            split=evaluation_split,
            submission_customer_ids=submission_customer_ids,
            max_target_customers=args.max_target_customers,
        )

        training_result = train_linear_ranker_from_csv(
            candidate_path=train_candidate_path,
            validation_labels=train_labels,
            config=config,
        )
        print(
            f"[{window_index}/{len(cutoffs)}] evaluating learned, deterministic, "
            "and source-order rankers",
            flush=True,
        )
        evaluation = evaluate_linear_ranker_from_csv(
            candidate_path=eval_candidate_path,
            validation_labels=eval_labels,
            model=training_result.model,
            k=args.k,
        )
        learned_reports.append(
            build_learned_linear_ranker_report(
                train_split=train_split,
                evaluation_split=evaluation_split,
                k=args.k,
                candidate_k=args.candidate_k,
                config=config,
                training_result=training_result,
                evaluation=evaluation,
            )
        )
        print(
            f"[{window_index}/{len(cutoffs)}] done: "
            f"learned_MAP@{args.k}={evaluation.map_at_k:.8f}, "
            f"deterministic_MAP@{args.k}={evaluation.deterministic_map_at_k:.8f}, "
            f"source_order_MAP@{args.k}={evaluation.baseline_map_at_k:.8f}",
            flush=True,
        )

    candidate_summary_paths = tuple(
        str(summary_path) for summary_path in sorted(candidate_export_cache.values(), key=str)
    )
    rolling_report = build_rolling_ranker_validation_report(
        learned_reports,
        max_target_customers=args.max_target_customers,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        candidate_summary_paths=candidate_summary_paths,
    )
    written_report_path = write_rolling_ranker_validation_report(rolling_report, report_path)

    print(f"Rolling cutoffs: {', '.join(rolling_report.cutoffs)}")
    print(f"Windows evaluated: {rolling_report.window_count}")
    print(
        f"Mean learned MAP@{rolling_report.k}: "
        f"{rolling_report.aggregate.mean_learned_map_at_k:.8f}"
    )
    print(
        f"Mean deterministic MAP@{rolling_report.k}: "
        f"{rolling_report.aggregate.mean_deterministic_map_at_k:.8f}"
    )
    print(
        f"Mean source-order MAP@{rolling_report.k}: "
        f"{rolling_report.aggregate.mean_source_order_map_at_k:.8f}"
    )
    print(
        f"Mean delta vs deterministic MAP@{rolling_report.k}: "
        f"{rolling_report.aggregate.mean_delta_learned_vs_deterministic_map_at_k:.8f}"
    )
    print(
        "Windows improved vs deterministic: "
        f"{rolling_report.aggregate.windows_improved_vs_deterministic}/"
        f"{rolling_report.window_count}"
    )
    for window in rolling_report.windows:
        print(
            f"- {window.evaluation_cutoff}: learned={window.learned_map_at_k:.8f}, "
            f"deterministic={window.deterministic_map_at_k:.8f}, "
            f"source_order={window.source_order_map_at_k:.8f}, "
            f"delta_det={window.delta_learned_vs_deterministic_map_at_k:.8f}"
        )
    print(f"Candidate summary reports written: {len(candidate_summary_paths)}")
    print(f"Rolling ranker report written to: {written_report_path}")
    return 0


def _handle_generate_learned_ranker_submission(args: argparse.Namespace) -> int:
    """Handle the ``generate-learned-ranker-submission`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` when the generated submission passes validation; otherwise ``1``.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    config = LinearRankerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        positive_weight=args.positive_weight,
        max_auto_positive_weight=args.max_auto_positive_weight,
    )
    max_transaction_date = find_max_transaction_date(
        iter_transaction_events(paths.raw_data_dir),
        progress_interval=5_000_000,
        progress_callback=lambda rows: print(
            f"max-date transaction scan progress: {rows} rows",
            flush=True,
        ),
    )
    final_split = TemporalSplit(
        cutoff=max_transaction_date + timedelta(days=1),
        horizon_days=args.horizon_days,
    )
    train_split = previous_window_split(final_split)
    train_candidate_path = args.train_candidate_output_path or paths.candidate_export_path(
        cutoff=train_split.cutoff.isoformat(),
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
    )
    train_candidate_path = _resolve_path_under_root(paths, train_candidate_path)
    train_candidate_report_path = (
        args.train_candidate_report_path or paths.candidate_export_report_path(train_candidate_path)
    )
    train_candidate_report_path = _resolve_path_under_root(paths, train_candidate_report_path)
    output_path = args.output_path or paths.learned_ranker_submission_path(
        k=args.k,
        candidate_k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        config_slug=_learned_ranker_config_slug(args),
    )
    output_path = _resolve_path_under_root(paths, output_path)
    validation_report_path = args.validation_report_path or paths.submission_validation_report_path(
        output_path
    )
    validation_report_path = _resolve_path_under_root(paths, validation_report_path)
    report_path = args.report_path or paths.learned_ranker_submission_report_path(output_path)
    report_path = _resolve_path_under_root(paths, report_path)

    target_customer_ids = load_submission_customer_ids_in_order(paths.raw_data_dir)
    submission_customer_set = set(target_customer_ids)
    print(
        "Training learned ranker for final submission on latest label window: "
        f"{train_split.cutoff.isoformat()}..{train_split.validation_end.isoformat()} exclusive",
        flush=True,
    )
    train_candidate_summary_path = _write_cached_ranker_candidate_export(
        cache={},
        paths=paths,
        split=train_split,
        submission_customer_ids=submission_customer_set,
        output_path=train_candidate_path,
        args=args,
    )
    train_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=train_split,
        submission_customer_ids=submission_customer_set,
        max_target_customers=None,
    )
    training_result = train_linear_ranker_from_csv(
        candidate_path=train_candidate_path,
        validation_labels=train_labels,
        config=config,
    )

    print(
        "Generating final-data learned-ranker predictions for "
        f"{len(target_customer_ids)} customers. include_co_visitation={not args.no_co_visitation}",
        flush=True,
    )
    submission = build_linear_ranker_submission_predictions(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=final_split,
        target_customer_ids=target_customer_ids,
        model=training_result.model,
        k=args.k,
        candidate_k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        max_transaction_date=max_transaction_date,
        transaction_progress_interval=5_000_000,
        transaction_progress_callback=lambda phase, rows: print(
            f"{phase} transaction scan progress: {rows} rows",
            flush=True,
        ),
        status_callback=lambda message: print(f"Final source build: {message}", flush=True),
        progress_interval=100_000,
        progress_callback=lambda completed, total: print(
            f"Final scoring progress: {completed}/{total} customers",
            flush=True,
        ),
    )
    written_submission_path = write_submission_file(
        predictions_by_customer=submission.predictions,
        customer_ids=target_customer_ids,
        path=output_path,
        max_predictions=args.k,
    )
    validation_result = validate_submission_file(
        submission_path=written_submission_path,
        expected_customer_ids=submission_customer_set,
        valid_article_ids=load_article_ids(paths.raw_data_dir),
        max_predictions=args.k,
        require_full_length=True,
    )
    written_validation_report_path = write_submission_validation_report(
        validation_result,
        validation_report_path,
    )
    report = build_learned_linear_ranker_submission_report(
        train_split=train_split,
        final_split=final_split,
        k=args.k,
        candidate_k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        config=config,
        model=training_result.model,
        training=training_result.summary,
        submission=submission,
        submission_path=written_submission_path,
        validation_report_path=written_validation_report_path,
        validation=validation_result,
    )
    written_report_path = write_learned_linear_ranker_submission_report(report, report_path)

    print(f"Max transaction date: {max_transaction_date.isoformat()}")
    print(f"Final training cutoff: {final_split.cutoff.isoformat()}")
    print(f"Training pairs: {training_result.summary.unique_candidate_pairs}")
    print(f"Training positives: {training_result.summary.positive_pairs}")
    print(f"Prediction target customers: {submission.diagnostics.target_customers}")
    print(
        "Full-length prediction coverage: "
        f"{submission.diagnostics.customers_with_full_length_predictions}/"
        f"{submission.diagnostics.target_customers}"
    )
    print(f"Duplicate prediction rows: {submission.diagnostics.duplicate_prediction_rows}")
    print(f"Train candidate summary: {train_candidate_summary_path}")
    print(f"Submission written to: {written_submission_path}")
    print(f"Submission valid: {validation_result.valid}")
    print(f"Validation report written to: {written_validation_report_path}")
    print(f"Learned-ranker submission report written to: {written_report_path}")
    for failure in validation_result.failures:
        print(f"- {failure}")
    return 0 if validation_result.valid else 1


def _handle_generate_deterministic_ranker_submission(args: argparse.Namespace) -> int:
    """Handle the ``generate-deterministic-ranker-submission`` subcommand."""

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    max_transaction_date = find_max_transaction_date(
        iter_transaction_events(paths.raw_data_dir),
        progress_interval=5_000_000,
        progress_callback=lambda rows: print(
            f"max-date transaction scan progress: {rows} rows",
            flush=True,
        ),
    )
    final_split = TemporalSplit(
        cutoff=max_transaction_date + timedelta(days=1),
        horizon_days=args.horizon_days,
    )
    tuning_split = previous_window_split(final_split)
    train_candidate_path = (
        _resolve_path_under_root(paths, args.train_candidate_output_path)
        if args.train_candidate_output_path is not None
        else _ranker_candidate_export_path(paths, tuning_split, args)
    )
    train_candidate_report_path = (
        args.train_candidate_report_path or paths.candidate_export_report_path(train_candidate_path)
    )
    train_candidate_report_path = _resolve_path_under_root(paths, train_candidate_report_path)
    output_path = args.output_path or paths.deterministic_ranker_submission_path(
        k=args.k,
        candidate_k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_product_code_popularity=args.include_product_code_popularity,
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if args.include_product_code_popularity
            else None
        ),
        product_code_max_history_items=(
            args.product_code_max_history_items if args.include_product_code_popularity else None
        ),
        tuning_slug=(
            f"tune_first_{args.max_target_customers}_customers"
            if args.max_target_customers is not None
            else None
        ),
    )
    output_path = _resolve_path_under_root(paths, output_path)
    validation_report_path = args.validation_report_path or paths.submission_validation_report_path(
        output_path
    )
    validation_report_path = _resolve_path_under_root(paths, validation_report_path)
    report_path = args.report_path or paths.deterministic_ranker_submission_report_path(output_path)
    report_path = _resolve_path_under_root(paths, report_path)

    target_customer_ids = load_submission_customer_ids_in_order(paths.raw_data_dir)
    submission_customer_set = set(target_customer_ids)
    print(
        "Selecting deterministic weights on latest visible label window: "
        f"{tuning_split.cutoff.isoformat()}..{tuning_split.validation_end.isoformat()} exclusive",
        flush=True,
    )
    train_candidate_summary_path = _write_cached_ranker_candidate_export(
        cache={},
        paths=paths,
        split=tuning_split,
        submission_customer_ids=submission_customer_set,
        output_path=train_candidate_path,
        args=args,
    )
    train_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=tuning_split,
        submission_customer_ids=submission_customer_set,
        max_target_customers=None,
    )
    weight_selection = select_deterministic_ranker_weights_from_csv(
        candidate_path=train_candidate_path,
        validation_labels=train_labels,
        k=args.k,
        grid=_deterministic_tuning_grid_from_args(args),
        default_weights=_deterministic_ranker_weights_from_args(args),
        top_n=args.top_trials,
    )

    print(
        "Generating final-data deterministic-ranker predictions for "
        f"{len(target_customer_ids)} customers. include_co_visitation={not args.no_co_visitation}, "
        f"include_seasonal={args.include_seasonal_popularity}, "
        f"include_age_segment={args.include_age_segment_popularity}, "
        f"include_garment_group={args.include_garment_group_popularity}, "
        f"include_two_tower={args.include_two_tower_retrieval}",
        flush=True,
    )
    customer_age_segments = _load_customer_age_segments_if_enabled(paths, args)
    article_garment_groups = _load_article_garment_groups_if_enabled(paths, args)
    article_product_codes = _load_article_product_codes_if_enabled(paths, args)
    final_two_tower_model = _train_two_tower_candidate_model_if_enabled(paths, final_split, args)
    submission = build_deterministic_ranker_submission_predictions(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=final_split,
        target_customer_ids=target_customer_ids,
        weights=weight_selection.selected_weights,
        k=args.k,
        candidate_k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=args.seasonal_shift_days,
        seasonal_window_days=args.seasonal_window_days,
        include_age_segment_popularity=args.include_age_segment_popularity,
        customer_segment_by_id=customer_age_segments,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=_age_segment_popularity_lookback_days(args),
        include_garment_group_popularity=args.include_garment_group_popularity,
        article_garment_group_by_id=article_garment_groups,
        garment_group_popularity_lookback_days=_garment_group_popularity_lookback_days(args),
        garment_group_max_history_items=args.garment_group_max_history_items,
        include_product_code_popularity=args.include_product_code_popularity,
        article_product_code_by_id=article_product_codes,
        product_code_popularity_lookback_days=_product_code_popularity_lookback_days(args),
        product_code_max_history_items=args.product_code_max_history_items,
        two_tower_model=final_two_tower_model,
        two_tower_source_name=args.two_tower_source_name,
        two_tower_max_retrieval_articles=args.two_tower_max_retrieval_articles,
        max_transaction_date=max_transaction_date,
        transaction_progress_interval=5_000_000,
        transaction_progress_callback=lambda phase, rows: print(
            f"{phase} transaction scan progress: {rows} rows",
            flush=True,
        ),
        status_callback=lambda message: print(f"Final source build: {message}", flush=True),
        progress_interval=100_000,
        progress_callback=lambda completed, total: print(
            f"Final scoring progress: {completed}/{total} customers",
            flush=True,
        ),
    )
    written_submission_path = write_submission_file(
        predictions_by_customer=submission.predictions,
        customer_ids=target_customer_ids,
        path=output_path,
        max_predictions=args.k,
    )
    validation_result = validate_submission_file(
        submission_path=written_submission_path,
        expected_customer_ids=submission_customer_set,
        valid_article_ids=load_article_ids(paths.raw_data_dir),
        max_predictions=args.k,
        require_full_length=True,
    )
    written_validation_report_path = write_submission_validation_report(
        validation_result,
        validation_report_path,
    )
    report = build_deterministic_ranker_submission_report(
        tuning_split=tuning_split,
        final_split=final_split,
        k=args.k,
        candidate_k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        include_seasonal_popularity=args.include_seasonal_popularity,
        seasonal_shift_days=(
            args.seasonal_shift_days if args.include_seasonal_popularity else None
        ),
        seasonal_window_days=(
            args.seasonal_window_days if args.include_seasonal_popularity else None
        ),
        include_age_segment_popularity=args.include_age_segment_popularity,
        age_segment_bucket_size=args.age_segment_bucket_size,
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if args.include_age_segment_popularity
            else None
        ),
        include_garment_group_popularity=args.include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if args.include_garment_group_popularity
            else None
        ),
        garment_group_max_history_items=(
            args.garment_group_max_history_items if args.include_garment_group_popularity else None
        ),
        include_two_tower_retrieval=args.include_two_tower_retrieval,
        two_tower_source_name=(
            args.two_tower_source_name if args.include_two_tower_retrieval else None
        ),
        two_tower_max_retrieval_articles=(
            args.two_tower_max_retrieval_articles if args.include_two_tower_retrieval else None
        ),
        weight_selection=weight_selection,
        submission=submission,
        submission_path=written_submission_path,
        validation_report_path=written_validation_report_path,
        validation=validation_result,
    )
    written_report_path = write_deterministic_ranker_submission_report(report, report_path)

    print(f"Max transaction date: {max_transaction_date.isoformat()}")
    print(f"Final training cutoff: {final_split.cutoff.isoformat()}")
    print(f"Tuning pairs: {weight_selection.selected_metrics.unique_candidate_pairs}")
    print(f"Tuning default MAP@{args.k}: {weight_selection.default_metrics.map_at_k:.8f}")
    print(f"Tuning selected MAP@{args.k}: {weight_selection.selected_metrics.map_at_k:.8f}")
    print(f"Prediction target customers: {submission.diagnostics.target_customers}")
    print(
        "Full-length prediction coverage: "
        f"{submission.diagnostics.customers_with_full_length_predictions}/"
        f"{submission.diagnostics.target_customers}"
    )
    print(f"Duplicate prediction rows: {submission.diagnostics.duplicate_prediction_rows}")
    print(f"Train candidate summary: {train_candidate_summary_path}")
    print(f"Submission written to: {written_submission_path}")
    print(f"Submission valid: {validation_result.valid}")
    print(f"Validation report written to: {written_validation_report_path}")
    print(f"Deterministic-ranker submission report written to: {written_report_path}")
    for failure in validation_result.failures:
        print(f"- {failure}")
    return 0 if validation_result.valid else 1


def _handle_export_two_tower_examples(args: argparse.Namespace) -> int:
    """Handle the ``export-two-tower-examples`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing examples, mappings, and summary metadata.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    config = TwoTowerExampleExportConfig(
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        negative_sampling=args.negative_sampling,
        positive_selection=args.positive_selection,
        max_positive_examples=args.max_positive_examples,
    )
    examples_path = args.examples_path or paths.two_tower_examples_path(
        cutoff=args.cutoff,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        max_positive_examples=args.max_positive_examples,
        positive_selection=args.positive_selection,
        negative_sampling=args.negative_sampling,
    )
    examples_path = _resolve_path_under_root(paths, examples_path)
    customer_mapping_path = args.customer_mapping_path or paths.two_tower_customer_mapping_path(
        examples_path
    )
    customer_mapping_path = _resolve_path_under_root(paths, customer_mapping_path)
    article_mapping_path = args.article_mapping_path or paths.two_tower_article_mapping_path(
        examples_path
    )
    article_mapping_path = _resolve_path_under_root(paths, article_mapping_path)
    report_path = args.report_path or paths.two_tower_example_export_report_path(examples_path)
    report_path = _resolve_path_under_root(paths, report_path)

    print(
        "Exporting two-tower examples: "
        f"cutoff={split.cutoff.isoformat()}, "
        f"negatives_per_positive={config.negatives_per_positive}, "
        f"seed={config.seed}, "
        f"max_positive_examples={config.max_positive_examples}, "
        f"positive_selection={config.positive_selection}",
        flush=True,
    )
    summary = write_two_tower_example_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        examples_path=examples_path,
        customer_mapping_path=customer_mapping_path,
        article_mapping_path=article_mapping_path,
        config=config,
        progress_interval=5_000_000,
        progress_callback=lambda phase, rows: print(
            f"{phase} progress: {rows} transaction rows",
            flush=True,
        ),
    )
    written_report_path = write_two_tower_example_export_summary(summary, report_path)

    print(f"Cutoff: {summary.cutoff}")
    print(f"Validation end exclusive: {summary.validation_end_exclusive}")
    print(f"Train rows seen: {summary.train_rows_seen}")
    print(f"Positive examples: {summary.positive_examples_written}")
    print(f"Negative examples: {summary.negative_examples_written}")
    print(f"Rows written: {summary.rows_written}")
    print(f"Mapped customers: {summary.unique_customers}")
    print(f"Mapped articles: {summary.unique_articles}")
    print(f"Skipped negative examples: {summary.skipped_negative_examples}")
    print(f"Examples CSV written to: {summary.examples_path}")
    print(f"Customer mapping written to: {summary.customer_mapping_path}")
    print(f"Article mapping written to: {summary.article_mapping_path}")
    print(f"Summary report written to: {written_report_path}")
    return 0


def _handle_evaluate_two_tower_retrieval(args: argparse.Namespace) -> int:
    """Handle the ``evaluate-two-tower-retrieval`` subcommand."""

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    examples_path = args.examples_path or paths.two_tower_examples_path(
        cutoff=args.cutoff,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        max_positive_examples=args.max_positive_examples,
        positive_selection=args.positive_selection,
        negative_sampling=args.negative_sampling,
    )
    examples_path = _resolve_path_under_root(paths, examples_path)
    customer_mapping_path = args.customer_mapping_path or paths.two_tower_customer_mapping_path(
        examples_path
    )
    article_mapping_path = args.article_mapping_path or paths.two_tower_article_mapping_path(
        examples_path
    )
    customer_mapping_path = _resolve_path_under_root(paths, customer_mapping_path)
    article_mapping_path = _resolve_path_under_root(paths, article_mapping_path)
    max_eval_customers = args.max_eval_customers if args.max_eval_customers != 0 else None
    report_path = args.report_path or paths.two_tower_retrieval_report_path(
        examples_path=examples_path,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        k=args.k,
        evaluation_ks=args.evaluation_ks,
        loss=args.loss,
        logq_correction_alpha=args.logq_correction_alpha,
        positive_recency_half_life_days=args.positive_recency_half_life_days,
        popularity_prior_weight=args.popularity_prior_weight,
        popularity_prior_lookback_days=(
            args.popularity_prior_lookback_days if args.popularity_prior_weight > 0.0 else None
        ),
        max_eval_customers=max_eval_customers,
        max_retrieval_articles=args.max_retrieval_articles,
    )
    report_path = _resolve_path_under_root(paths, report_path)

    recency_reference_date = args.recency_reference_date
    if args.positive_recency_half_life_days is not None and recency_reference_date is None:
        recency_reference_date = split.cutoff.isoformat()
    config = TwoTowerSmokeTrainingConfig(
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
        loss=args.loss,
        max_training_examples=args.max_training_examples,
        positive_recency_half_life_days=args.positive_recency_half_life_days,
        recency_reference_date=recency_reference_date,
        logq_correction_alpha=args.logq_correction_alpha,
    )
    print(
        "Training lightweight two-tower smoke model from exported examples: " f"{examples_path}",
        flush=True,
    )
    model, training = train_two_tower_smoke_model_from_csv(
        examples_path=examples_path,
        customer_mapping_path=customer_mapping_path,
        article_mapping_path=article_mapping_path,
        config=config,
    )
    validation_data = summarize_temporal_split_with_labels(
        iter_transaction_events(paths.raw_data_dir),
        split,
    )
    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    validation_labels = {
        customer_id: validation_data.validation_labels[customer_id]
        for customer_id in select_validation_label_customer_ids(
            validation_labels=validation_data.validation_labels,
            submission_customer_ids=submission_customer_ids,
            max_target_customers=None,
        )
    }
    article_score_prior = (
        build_article_popularity_score_prior(
            iter_transaction_events(paths.raw_data_dir),
            split,
            lookback_days=args.popularity_prior_lookback_days,
        )
        if args.popularity_prior_weight > 0.0
        else None
    )
    evaluation = evaluate_two_tower_retrieval(
        model,
        validation_labels,
        k=args.k,
        evaluation_ks=args.evaluation_ks,
        max_eval_customers=max_eval_customers,
        max_retrieval_articles=args.max_retrieval_articles,
        article_score_prior=article_score_prior,
        score_prior_weight=args.popularity_prior_weight,
    )
    report = build_two_tower_retrieval_report(
        cutoff=split.cutoff.isoformat(),
        validation_end_exclusive=split.validation_end.isoformat(),
        horizon_days=split.horizon_days,
        training=training,
        evaluation=evaluation,
    )
    written_report_path = write_two_tower_retrieval_report(report, report_path)

    print(f"Cutoff: {report.cutoff}")
    print(f"Validation end exclusive: {report.validation_end_exclusive}")
    print(f"Training rows read: {training.rows_read}")
    print(f"Training positives: {training.positive_examples}")
    print(f"Training negatives: {training.negative_examples}")
    print(f"Mapped validation customers: {evaluation.mapped_labeled_customers}")
    print(f"Evaluated customers: {evaluation.evaluated_customers}")
    print(f"Article pool size: {evaluation.article_pool_size}")
    print(f"Label article pool coverage: {evaluation.label_article_pool_coverage:.8f}")
    print(f"Score prior weight: {evaluation.score_prior_weight:.8f}")
    print(f"Score prior articles: {evaluation.score_prior_articles}")
    print(f"Two-tower MAP@{evaluation.k}: {evaluation.map_at_k:.8f}")
    for metric_k in evaluation.evaluation_ks:
        print(f"Two-tower recall@{metric_k}: {evaluation.recall_by_k[str(metric_k)]:.8f}")
    print(f"Retrieval report written to: {written_report_path}")
    return 0


def _handle_inventory_article_images(args: argparse.Namespace) -> int:
    """Handle the ``inventory-article-images`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        Zero when the image path is usable; non-zero when a blocking inventory
        failure is detected.
    """

    paths = ProjectPaths.from_root(args.project_root, raw_data_dir=args.raw_data_dir)
    manifest_path = args.manifest_path or paths.article_image_inventory_manifest_path
    manifest_path = _resolve_path_under_root(paths, manifest_path)
    report_path = args.report_path or paths.article_image_inventory_report_path
    report_path = _resolve_path_under_root(paths, report_path)

    print(
        "Inventorying article images: "
        f"raw_data_dir={paths.raw_data_dir}, max_examples={args.max_examples}",
        flush=True,
    )
    summary = write_article_image_inventory(
        raw_data_dir=paths.raw_data_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        max_examples=args.max_examples,
    )

    print(f"Image inventory valid: {summary.valid}")
    print(f"Articles: {summary.article_count}")
    print(f"Scanned image files: {summary.scanned_image_file_count}")
    print(f"Canonical image files: {summary.canonical_image_file_count}")
    print(f"Matched articles: {summary.matched_article_count}")
    print(f"Missing article images: {summary.missing_article_count}")
    print(f"Extra canonical images: {summary.extra_canonical_image_count}")
    print(f"Malformed image files: {summary.malformed_image_file_count}")
    print(f"Image coverage: {summary.image_coverage:.6f}")
    if summary.failures:
        print("Failures:")
        for failure in summary.failures:
            print(f"- {failure}")
    print(f"Manifest CSV written to: {summary.manifest_path}")
    print(f"Summary report written to: {summary.report_path}")
    return 0 if summary.valid else 1


def _handle_export_article_content(args: argparse.Namespace) -> int:
    """Handle the ``export-article-content`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing article content inputs and diagnostics.
    """

    paths = ProjectPaths.from_root(args.project_root, raw_data_dir=args.raw_data_dir)
    has_subset_config = (
        args.max_articles is not None
        or args.priority_cutoff is not None
        or args.priority_lookback_days is not None
    )
    output_path = args.output_path or (
        paths.article_content_export_path_for_config(
            max_articles=args.max_articles,
            priority_cutoff=args.priority_cutoff,
            priority_lookback_days=args.priority_lookback_days,
        )
        if has_subset_config
        else paths.article_content_export_path
    )
    output_path = _resolve_path_under_root(paths, output_path)
    report_path = args.report_path or paths.article_content_export_report_path_for_path(output_path)
    report_path = _resolve_path_under_root(paths, report_path)
    if args.priority_lookback_days is not None and args.priority_cutoff is None:
        raise ValueError("--priority-cutoff is required with --priority-lookback-days")
    priority_cutoff = date.fromisoformat(args.priority_cutoff) if args.priority_cutoff else None
    article_id_order = (
        build_article_popularity_priority(
            iter_transaction_events(paths.raw_data_dir),
            cutoff=priority_cutoff,
            lookback_days=args.priority_lookback_days,
        )
        if priority_cutoff is not None
        else None
    )

    print(
        "Exporting article content for embedding providers: "
        f"raw_data_dir={paths.raw_data_dir}, max_examples={args.max_examples}, "
        f"max_articles={args.max_articles}, priority_cutoff={args.priority_cutoff}, "
        f"priority_lookback_days={args.priority_lookback_days}",
        flush=True,
    )
    summary = write_article_content_export(
        raw_data_dir=paths.raw_data_dir,
        output_path=output_path,
        report_path=report_path,
        max_examples=args.max_examples,
        article_id_order=article_id_order,
        max_articles=args.max_articles,
    )

    print(f"Articles: {summary.article_count}")
    print(f"Records written: {summary.records_written}")
    print(f"Priority articles: {summary.priority_article_count}")
    print(f"Image-available records: {summary.image_available_count}")
    print(f"Image-missing records: {summary.image_missing_count}")
    print(f"Empty combined-text records: {summary.empty_combined_text_count}")
    print(f"Content CSV written to: {summary.output_path}")
    print(f"Summary report written to: {summary.report_path}")
    return 0


def _handle_generate_article_embeddings(args: argparse.Namespace) -> int:
    """Handle the ``generate-article-embeddings`` subcommand."""

    paths = ProjectPaths.from_root(args.project_root, raw_data_dir=args.raw_data_dir)
    default_article_content_path = _resolve_path_under_root(
        paths,
        paths.article_content_export_path,
    )
    article_content_path = args.article_content_path or paths.article_content_export_path
    article_content_path = _resolve_path_under_root(paths, article_content_path)
    provider_slug = f"{args.provider}_{args.model_id}_{args.model_revision}"
    cache_provider_slug = provider_slug
    if article_content_path != default_article_content_path:
        cache_provider_slug = f"{cache_provider_slug}_{article_content_path.stem}"
    if args.max_articles is not None:
        cache_provider_slug = f"{cache_provider_slug}_first_{args.max_articles}_articles"
    embeddings_path = args.embeddings_path or paths.article_embedding_cache_embeddings_path(
        cache_provider_slug,
        args.embedding_kind,
        "jsonl",
    )
    embeddings_path = _resolve_path_under_root(paths, embeddings_path)
    article_mapping_path = args.article_mapping_path or paths.article_embedding_cache_mapping_path(
        cache_provider_slug,
        args.embedding_kind,
    )
    article_mapping_path = _resolve_path_under_root(paths, article_mapping_path)
    manifest_path = args.manifest_path or paths.article_embedding_cache_manifest_path(
        cache_provider_slug,
        args.embedding_kind,
    )
    manifest_path = _resolve_path_under_root(paths, manifest_path)

    if args.provider != "hf-clip":
        raise ValueError(f"unsupported provider: {args.provider!r}")
    from hm_recsys.embeddings.providers.huggingface_clip import (
        HuggingFaceClipArticleEmbeddingProvider,
    )

    print(
        "Generating article embeddings: "
        f"provider={args.provider}, model_id={args.model_id}, "
        f"embedding_kind={args.embedding_kind}, batch_size={args.batch_size}, "
        f"max_articles={args.max_articles}",
        flush=True,
    )
    provider = HuggingFaceClipArticleEmbeddingProvider(
        model_id=args.model_id,
        revision=args.model_revision,
        embedding_kind=cast(EmbeddingCacheKind, args.embedding_kind),
        device=args.device,
    )
    config = ArticleEmbeddingCacheWriteConfig(
        provider_model_id=args.model_id,
        provider_model_revision=args.model_revision,
        embedding_kind=cast(EmbeddingCacheKind, args.embedding_kind),
        preprocessing=args.preprocessing,
        license=args.license_note,
        batch_size=args.batch_size,
        max_articles=args.max_articles,
        source_image_inventory_path=paths.article_image_inventory_manifest_path,
    )
    summary = write_article_embedding_cache_from_content_export(
        provider,
        raw_data_dir=paths.raw_data_dir,
        article_content_path=article_content_path,
        embeddings_path=embeddings_path,
        article_mapping_path=article_mapping_path,
        manifest_path=manifest_path,
        config=config,
    )

    print(f"Article content path: {summary.source_article_content_path}")
    print(f"Articles processed: {summary.article_count}")
    print(f"Embeddings written: {summary.embedding_count}")
    print(f"Missing embeddings: {summary.missing_embedding_count}")
    print(f"Missing-image rows skipped: {summary.skipped_missing_image_count}")
    print(f"Embeddings JSONL written to: {summary.embeddings_path}")
    print(f"Article mapping written to: {summary.article_mapping_path}")
    print(f"Manifest written to: {summary.manifest_path}")
    return 0


def _handle_content_similarity_diagnostics(args: argparse.Namespace) -> int:
    """Handle the ``content-similarity-diagnostics`` subcommand."""

    paths = ProjectPaths.from_root(args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    manifest_path = _resolve_path_under_root(paths, args.manifest_path)
    source_name = _effective_diagnostics_content_similarity_source_name(args)
    report_path = args.report_path or paths.content_similarity_diagnostics_report_path(
        cutoff=args.cutoff,
        source_name=source_name,
        manifest_path=manifest_path,
        popularity_prior_weight=args.popularity_prior_weight,
        popularity_lookback_days=args.popularity_lookback_days,
        candidate_pool_size=args.candidate_pool_size,
        max_target_customers=args.max_target_customers,
    )
    report_path = _resolve_path_under_root(paths, report_path)
    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)

    print(
        "Evaluating cached content-similarity source: "
        f"cutoff={args.cutoff}, manifest={manifest_path}, "
        f"source={source_name}, max_target_customers={args.max_target_customers}",
        flush=True,
    )
    report = evaluate_cached_content_similarity(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        submission_customer_ids=submission_customer_ids,
        manifest_path=manifest_path,
        source_name=source_name,
        evaluation_ks=tuple(args.evaluation_ks),
        max_history_items=args.max_history_items,
        exclude_history=not args.include_history,
        popularity_prior_weight=args.popularity_prior_weight,
        popularity_lookback_days=args.popularity_lookback_days,
        candidate_pool_size=args.candidate_pool_size,
        max_target_customers=args.max_target_customers,
    )
    written_report_path = write_content_similarity_diagnostics_report(report, report_path)

    print(f"Provider: {report.provider_name}")
    print(f"Embedding kind: {report.embedding_kind}")
    print(f"Embeddings loaded: {report.embedding_count}")
    print(f"Target customers: {report.target_customers}")
    print(f"Customers with embedding history: {report.rows_with_embedding_history}")
    print(f"Rows with candidates: {report.rows_with_candidates}")
    print(f"Candidate coverage: {report.candidate_coverage:.6f}")
    print(f"MAP@12: {report.map_at_12:.8f}")
    for k, value in report.recall_at_k.items():
        print(f"Recall@{k}: {value:.8f}")
    print(f"Diagnostics report written to: {written_report_path}")
    return 0


def _resolve_report_path(paths: ProjectPaths, report_path: Path | None) -> Path:
    """Resolve an optional report path against project paths.

    Args:
        paths: Canonical project paths.
        report_path: Optional CLI-provided report path.

    Returns:
        Absolute report path. Defaults to the data-contract report path when no
        override is provided.
    """

    if report_path is None:
        return paths.data_contract_report_path
    expanded = report_path.expanduser()
    if expanded.is_absolute():
        return expanded
    return paths.root / expanded


def _resolve_path_under_root(paths: ProjectPaths, path: Path) -> Path:
    """Resolve a CLI path against the project root when it is relative.

    Args:
        paths: Canonical project paths.
        path: Candidate CLI path.

    Returns:
        Absolute path for writing or reading.
    """

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return paths.root / expanded


def _resolve_optional_path_under_root(paths: ProjectPaths, path: Path | None) -> Path | None:
    """Resolve an optional CLI path against the project root."""

    if path is None:
        return None
    return _resolve_path_under_root(paths, path)


def _deterministic_ranker_weights_from_args(
    args: argparse.Namespace,
) -> DeterministicRankerWeights:
    """Return deterministic-ranker weights with optional CLI overrides applied."""

    updates: dict[str, float] = {}
    two_tower_presence_weight = getattr(args, "two_tower_ranker_presence_weight", None)
    if two_tower_presence_weight is not None:
        updates["two_tower_retrieval_presence_weight"] = two_tower_presence_weight
    two_tower_score_weight = getattr(args, "two_tower_ranker_score_weight", None)
    if two_tower_score_weight is not None:
        updates["two_tower_retrieval_score_weight"] = two_tower_score_weight
    if not updates:
        return DEFAULT_DETERMINISTIC_RANKER_WEIGHTS
    return replace(DEFAULT_DETERMINISTIC_RANKER_WEIGHTS, **updates)


def _lightgbm_behavioral_ranker_weights_from_args(
    args: argparse.Namespace,
) -> DeterministicRankerWeights:
    """Return the deterministic prior weights used by the behavioral GBDT blend.

    These defaults mirror the best full-validation diagnostic prior used before
    adding behavioral features. The LightGBM score is blended with this prior;
    the prior itself is not selected on the evaluation labels.
    """

    weights = LIGHTGBM_BEHAVIORAL_RANKER_PRIOR_WEIGHTS
    updates: dict[str, float] = {}
    two_tower_presence_weight = getattr(args, "two_tower_ranker_presence_weight", None)
    if two_tower_presence_weight is not None:
        updates["two_tower_retrieval_presence_weight"] = two_tower_presence_weight
    two_tower_score_weight = getattr(args, "two_tower_ranker_score_weight", None)
    if two_tower_score_weight is not None:
        updates["two_tower_retrieval_score_weight"] = two_tower_score_weight
    if updates:
        return replace(weights, **updates)
    return weights


def _lightgbm_behavioral_ranker_report_path(
    paths: ProjectPaths,
    args: argparse.Namespace,
) -> Path:
    train_cutoff = args.train_cutoff or "previous"
    name = (
        "lightgbm_behavioral_ranker_"
        f"train_{train_cutoff.replace('-', '_')}_"
        f"eval_{args.cutoff.replace('-', '_')}_"
        f"k_{args.k}_neg{args.negative_per_positive}_"
        f"lambda{_float_slug(args.blend_lambda)}"
    )
    if args.max_train_customers is not None:
        name = f"{name}_train_first_{args.max_train_customers}"
    if args.max_eval_customers is not None:
        name = f"{name}_eval_first_{args.max_eval_customers}"
    return paths.artifacts_dir / "ranker-baselines" / f"{name}.json"


def _lightgbm_behavioral_submission_slug(args: argparse.Namespace) -> str:
    """Build a compact slug for rich LightGBM behavioral submission artifacts."""

    name = (
        f"lightgbm_behavioral_rich_train_{_cli_safe_name(args.train_cutoff)}_"
        f"lookback_{args.popularity_lookback_days}_"
        f"candidate_k_{args.candidate_k}_rank_k_{args.k}_"
        f"negpp_{args.negative_per_positive}_lambda_{_float_slug(args.blend_lambda)}"
    )
    if not args.no_co_visitation:
        name = (
            f"{name}_covis_h{args.co_visitation_max_history_items}_"
            f"n{args.co_visitation_max_neighbors_per_item}"
        )
    if args.include_seasonal_popularity:
        name = (
            f"{name}_seasonal_shift{args.seasonal_shift_days}_" f"window{args.seasonal_window_days}"
        )
    if args.include_age_segment_popularity:
        name = (
            f"{name}_age_segment_b{args.age_segment_bucket_size}_"
            f"lookback{_age_segment_popularity_lookback_days(args)}"
        )
    if args.include_garment_group_popularity:
        name = (
            f"{name}_garment_group_lookback{_garment_group_popularity_lookback_days(args)}_"
            f"h{args.garment_group_max_history_items}"
        )
    if getattr(args, "include_product_code_popularity", False):
        name = (
            f"{name}_product_code_lookback{_product_code_popularity_lookback_days(args)}_"
            f"h{getattr(args, 'product_code_max_history_items', DEFAULT_MAX_HISTORY_ITEMS)}"
        )
    two_tower_slug = _two_tower_config_slug(args)
    if two_tower_slug is not None:
        name = f"{name}_two_tower_{two_tower_slug}"
    if args.extra_train_window:
        extra_cutoffs = "_".join(_cli_safe_name(window[0]) for window in args.extra_train_window)
        name = f"{name}_extra_train_{extra_cutoffs}"
    if args.max_train_customers is not None:
        name = f"{name}_train_first_{args.max_train_customers}_customers"
    if args.max_target_customers is not None:
        name = f"{name}_target_first_{args.max_target_customers}_customers"
    return _cli_safe_name(name)


def _cli_safe_name(value: str) -> str:
    """Return a conservative path-safe token for CLI-generated artifacts."""

    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_" for character in value
    ).strip("_")
    return safe or "artifact"


def _deterministic_tuning_grid_from_args(
    args: argparse.Namespace,
) -> DeterministicRankerTuningGrid:
    """Return the deterministic tuning grid for the configured source set.

    The default tuning grid stays compact for classic popularity/co-visitation
    experiments. When two-tower candidate rows are explicitly enabled, expand
    only the two-tower-specific ranker weights so selection still happens on the
    previous non-overlapping label window rather than on the evaluation window.
    """

    if getattr(args, "research_weight_grid", False):
        return _research_deterministic_tuning_grid(args)
    if not getattr(args, "include_two_tower_retrieval", False):
        return DEFAULT_DETERMINISTIC_RANKER_TUNING_GRID
    base_weights = _deterministic_ranker_weights_from_args(args)
    return DeterministicRankerTuningGrid(
        garment_group_popularity_presence_weights=(
            base_weights.garment_group_popularity_presence_weight,
        ),
        garment_group_popularity_score_weights=(
            base_weights.garment_group_popularity_score_weight,
        ),
        age_segment_popularity_presence_weights=(
            base_weights.age_segment_popularity_presence_weight,
        ),
        age_segment_popularity_score_weights=(base_weights.age_segment_popularity_score_weight,),
        source_count_weights=(base_weights.source_count_weight,),
        best_rank_score_weights=(base_weights.best_rank_score_weight,),
        two_tower_retrieval_presence_weights=(0.10, 0.30, 0.60, 1.00),
        two_tower_retrieval_score_weights=(0.05, 0.15, 0.30),
        two_tower_retrieval_rank_weights=(base_weights.two_tower_retrieval_rank_weight,),
        two_tower_retrieval_latest_customer_presence_weights=(0.0, 0.30, 0.60, 1.00),
        two_tower_retrieval_latest_customer_score_weights=(0.0, 0.05),
        two_tower_retrieval_latest_customer_rank_weights=(
            base_weights.two_tower_retrieval_latest_customer_rank_weight,
        ),
    )


def _research_deterministic_tuning_grid(args: argparse.Namespace) -> DeterministicRankerTuningGrid:
    """Return a compact high-signal grid for fast ranker research loops."""

    base_weights = _deterministic_ranker_weights_from_args(args)
    two_tower_presence_weights: tuple[float, ...] = (
        base_weights.two_tower_retrieval_presence_weight,
    )
    two_tower_rank_weights: tuple[float, ...] = (base_weights.two_tower_retrieval_rank_weight,)
    latest_customer_presence_weights: tuple[float, ...] = (
        base_weights.two_tower_retrieval_latest_customer_presence_weight,
    )
    latest_customer_rank_weights: tuple[float, ...] = (
        base_weights.two_tower_retrieval_latest_customer_rank_weight,
    )
    if getattr(args, "include_two_tower_retrieval", False):
        two_tower_presence_weights = (1.5,)
        two_tower_rank_weights = (0.0,)
        latest_customer_presence_weights = (1.0,)
        latest_customer_rank_weights = (0.0, 1.0)
    return DeterministicRankerTuningGrid(
        repeat_presence_weights=(2.5, 3.0, 3.5),
        repeat_score_weights=(0.5, 1.0),
        recent_popularity_presence_weights=(0.0, 0.25),
        recent_popularity_score_weights=(base_weights.recent_popularity_score_weight,),
        co_visitation_presence_weights=(1.0, 1.5),
        co_visitation_score_weights=(0.0,),
        garment_group_popularity_presence_weights=(0.8,),
        garment_group_popularity_score_weights=(0.55,),
        age_segment_popularity_presence_weights=(0.45,),
        age_segment_popularity_score_weights=(0.1,),
        source_count_weights=(0.15,),
        best_rank_score_weights=(0.0,),
        two_tower_retrieval_presence_weights=two_tower_presence_weights,
        two_tower_retrieval_score_weights=(0.0,),
        two_tower_retrieval_rank_weights=two_tower_rank_weights,
        two_tower_retrieval_latest_customer_presence_weights=latest_customer_presence_weights,
        two_tower_retrieval_latest_customer_score_weights=(0.0,),
        two_tower_retrieval_latest_customer_rank_weights=latest_customer_rank_weights,
    )


def _train_two_tower_candidate_model_if_enabled(
    paths: ProjectPaths,
    split: TemporalSplit,
    args: argparse.Namespace,
) -> TwoTowerSmokeModel | None:
    """Train the configured two-tower candidate model when requested."""

    if not getattr(args, "include_two_tower_retrieval", False):
        return None

    examples_path = getattr(args, "two_tower_examples_path", None) or paths.two_tower_examples_path(
        cutoff=split.cutoff.isoformat(),
        negatives_per_positive=args.two_tower_negatives_per_positive,
        seed=args.two_tower_seed,
        max_positive_examples=args.two_tower_max_positive_examples,
        positive_selection=args.two_tower_positive_selection,
        negative_sampling=args.two_tower_negative_sampling,
    )
    examples_path = _resolve_path_under_root(paths, examples_path)
    customer_mapping_path = getattr(
        args,
        "two_tower_customer_mapping_path",
        None,
    ) or paths.two_tower_customer_mapping_path(examples_path)
    article_mapping_path = getattr(
        args,
        "two_tower_article_mapping_path",
        None,
    ) or paths.two_tower_article_mapping_path(examples_path)
    customer_mapping_path = _resolve_path_under_root(paths, customer_mapping_path)
    article_mapping_path = _resolve_path_under_root(paths, article_mapping_path)
    config = TwoTowerSmokeTrainingConfig(
        embedding_dim=args.two_tower_embedding_dim,
        epochs=args.two_tower_epochs,
        learning_rate=args.two_tower_learning_rate,
        l2=args.two_tower_l2,
        seed=args.two_tower_seed,
        loss=args.two_tower_loss,
        max_training_examples=args.two_tower_max_training_examples,
        logq_correction_alpha=args.two_tower_logq_correction_alpha,
    )
    print(
        "Training two-tower candidate model from exported examples: " f"{examples_path}",
        flush=True,
    )
    model, training = train_two_tower_smoke_model_from_csv(
        examples_path=examples_path,
        customer_mapping_path=customer_mapping_path,
        article_mapping_path=article_mapping_path,
        config=config,
    )
    print(
        "Two-tower candidate model trained: "
        f"rows={training.rows_read}, positives={training.positive_examples}, "
        f"negatives={training.negative_examples}, loss={training.final_average_loss:.8f}",
        flush=True,
    )
    return model


def _two_tower_config_slug(args: argparse.Namespace) -> str | None:
    """Return a compact two-tower source config slug for artifact names."""

    if not getattr(args, "include_two_tower_retrieval", False):
        return None
    negative_sampling = getattr(args, "two_tower_negative_sampling", "random")
    negative_slug = f"_{negative_sampling}" if negative_sampling != "random" else ""
    logq_alpha = getattr(args, "two_tower_logq_correction_alpha", 0.0)
    logq_slug = f"_logq{logq_alpha:g}" if logq_alpha > 0.0 else ""
    return (
        f"{args.two_tower_positive_selection}_"
        f"pos{args.two_tower_max_positive_examples}_"
        f"neg{args.two_tower_negatives_per_positive}{negative_slug}_"
        f"dim{args.two_tower_embedding_dim}_"
        f"e{args.two_tower_epochs}_"
        f"{args.two_tower_loss}{logq_slug}"
    )


def _ranker_candidate_export_path(
    paths: ProjectPaths,
    split: TemporalSplit,
    args: argparse.Namespace,
) -> Path:
    """Return the default candidate export path for ranker CLI commands.

    Args:
        paths: Canonical project paths.
        split: Temporal split whose validation-label customers are exported.
        args: Parsed ranker command arguments.

    Returns:
        Absolute candidate export path under the project artifacts directory.
    """

    candidate_path = paths.candidate_export_path(
        cutoff=split.cutoff.isoformat(),
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        include_seasonal_popularity=getattr(args, "include_seasonal_popularity", False),
        seasonal_shift_days=(
            getattr(args, "seasonal_shift_days", DEFAULT_SEASONAL_SHIFT_DAYS)
            if getattr(args, "include_seasonal_popularity", False)
            else None
        ),
        seasonal_window_days=(
            getattr(args, "seasonal_window_days", DEFAULT_SEASONAL_WINDOW_DAYS)
            if getattr(args, "include_seasonal_popularity", False)
            else None
        ),
        include_age_segment_popularity=getattr(args, "include_age_segment_popularity", False),
        age_segment_bucket_size=getattr(args, "age_segment_bucket_size", None),
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if getattr(args, "include_age_segment_popularity", False)
            else None
        ),
        include_garment_group_popularity=getattr(
            args,
            "include_garment_group_popularity",
            False,
        ),
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if getattr(args, "include_garment_group_popularity", False)
            else None
        ),
        garment_group_max_history_items=(
            getattr(args, "garment_group_max_history_items", DEFAULT_MAX_HISTORY_ITEMS)
            if getattr(args, "include_garment_group_popularity", False)
            else None
        ),
        include_product_code_popularity=getattr(args, "include_product_code_popularity", False),
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if getattr(args, "include_product_code_popularity", False)
            else None
        ),
        product_code_max_history_items=(
            getattr(args, "product_code_max_history_items", DEFAULT_MAX_HISTORY_ITEMS)
            if getattr(args, "include_product_code_popularity", False)
            else None
        ),
        content_similarity_source_name=(
            _effective_content_similarity_source_name(args)
            if getattr(args, "content_similarity_manifest_path", None) is not None
            else None
        ),
        content_similarity_manifest_path=getattr(args, "content_similarity_manifest_path", None),
        content_similarity_popularity_prior_weight=(
            getattr(args, "content_similarity_popularity_prior_weight", 0.0)
            if getattr(args, "content_similarity_manifest_path", None) is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            getattr(args, "content_similarity_popularity_lookback_days", None)
            if getattr(args, "content_similarity_manifest_path", None) is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            getattr(args, "content_similarity_candidate_pool_size", None)
            if getattr(args, "content_similarity_manifest_path", None) is not None
            else None
        ),
        include_two_tower_retrieval=getattr(args, "include_two_tower_retrieval", False),
        two_tower_config_slug=_two_tower_config_slug(args),
        max_target_customers=getattr(args, "max_target_customers", None),
    )
    return _resolve_path_under_root(paths, candidate_path)


def _write_cached_ranker_candidate_export(
    cache: dict[Path, Path],
    paths: ProjectPaths,
    split: TemporalSplit,
    submission_customer_ids: set[str],
    output_path: Path,
    args: argparse.Namespace,
) -> Path:
    """Write or reuse a candidate export and its JSON summary.

    Args:
        cache: In-memory cache keyed by resolved candidate CSV path.
        paths: Canonical project paths.
        split: Temporal split to export.
        submission_customer_ids: Authoritative H&M submission customer universe.
        output_path: Candidate CSV path.
        args: Parsed ranker command arguments.

    Returns:
        Written JSON summary path.
    """

    cached = cache.get(output_path)
    if cached is not None:
        return cached

    candidate_report_path = _resolve_path_under_root(
        paths, paths.candidate_export_report_path(output_path)
    )
    if output_path.exists() and candidate_report_path.exists():
        cache[output_path] = candidate_report_path
        return candidate_report_path
    if output_path.exists():
        candidate_report_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_report_path.write_text(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "reused_existing_candidate_csv": True,
                    "summary": "Existing candidate CSV reused for fast ranker tuning.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        cache[output_path] = candidate_report_path
        return candidate_report_path

    two_tower_model = _train_two_tower_candidate_model_if_enabled(paths, split, args)
    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        submission_customer_ids=submission_customer_ids,
        output_path=output_path,
        k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        include_seasonal_popularity=getattr(args, "include_seasonal_popularity", False),
        seasonal_shift_days=getattr(args, "seasonal_shift_days", DEFAULT_SEASONAL_SHIFT_DAYS),
        seasonal_window_days=getattr(args, "seasonal_window_days", DEFAULT_SEASONAL_WINDOW_DAYS),
        include_age_segment_popularity=getattr(args, "include_age_segment_popularity", False),
        customer_segment_by_id=_load_customer_age_segments_if_enabled(paths, args),
        age_segment_bucket_size=getattr(
            args, "age_segment_bucket_size", DEFAULT_AGE_SEGMENT_BUCKET_SIZE
        ),
        age_segment_popularity_lookback_days=(
            _age_segment_popularity_lookback_days(args)
            if getattr(args, "include_age_segment_popularity", False)
            else None
        ),
        include_garment_group_popularity=getattr(
            args,
            "include_garment_group_popularity",
            False,
        ),
        article_garment_group_by_id=_load_article_garment_groups_if_enabled(paths, args),
        garment_group_popularity_lookback_days=(
            _garment_group_popularity_lookback_days(args)
            if getattr(args, "include_garment_group_popularity", False)
            else None
        ),
        garment_group_max_history_items=getattr(
            args,
            "garment_group_max_history_items",
            DEFAULT_MAX_HISTORY_ITEMS,
        ),
        include_product_code_popularity=getattr(args, "include_product_code_popularity", False),
        article_product_code_by_id=_load_article_product_codes_if_enabled(paths, args),
        product_code_popularity_lookback_days=(
            _product_code_popularity_lookback_days(args)
            if getattr(args, "include_product_code_popularity", False)
            else None
        ),
        product_code_max_history_items=getattr(
            args,
            "product_code_max_history_items",
            DEFAULT_MAX_HISTORY_ITEMS,
        ),
        content_similarity_manifest_path=_resolve_optional_path_under_root(
            paths, getattr(args, "content_similarity_manifest_path", None)
        ),
        content_similarity_source_name=_effective_content_similarity_source_name(args),
        content_similarity_max_history_items=getattr(
            args,
            "content_similarity_max_history_items",
            DEFAULT_MAX_HISTORY_ITEMS,
        ),
        content_similarity_exclude_history=not getattr(args, "include_content_history", False),
        content_similarity_popularity_prior_weight=getattr(
            args,
            "content_similarity_popularity_prior_weight",
            0.0,
        ),
        content_similarity_popularity_lookback_days=getattr(
            args,
            "content_similarity_popularity_lookback_days",
            None,
        ),
        content_similarity_candidate_pool_size=getattr(
            args,
            "content_similarity_candidate_pool_size",
            None,
        ),
        two_tower_model=two_tower_model,
        two_tower_source_name=getattr(args, "two_tower_source_name", TWO_TOWER_RETRIEVAL_SOURCE),
        two_tower_max_retrieval_articles=getattr(
            args,
            "two_tower_max_retrieval_articles",
            5000,
        ),
        max_target_customers=getattr(args, "max_target_customers", None),
    )
    written_report_path = write_candidate_export_summary(summary, candidate_report_path)
    cache[output_path] = written_report_path
    return written_report_path


def _cached_validation_labels_for_split(
    cache: dict[tuple[str, int], dict[str, tuple[str, ...]]],
    raw_data_dir: Path,
    split: TemporalSplit,
    submission_customer_ids: set[str],
    max_target_customers: int | None,
) -> dict[str, tuple[str, ...]]:
    """Return cached validation labels for a split and customer cap.

    Args:
        cache: In-memory labels cache keyed by cutoff and horizon.
        raw_data_dir: Directory containing H&M raw CSV files.
        split: Temporal split whose labels should be collected.
        submission_customer_ids: Authoritative H&M submission customer universe.
        max_target_customers: Optional deterministic cap.

    Returns:
        Validation labels keyed by selected target customer ID.
    """

    cache_key = (split.cutoff.isoformat(), split.horizon_days)
    labels = cache.get(cache_key)
    if labels is None:
        labels = _validation_labels_for_split(
            raw_data_dir=raw_data_dir,
            split=split,
            submission_customer_ids=submission_customer_ids,
            max_target_customers=max_target_customers,
        )
        cache[cache_key] = labels
    return labels


def _validation_labels_for_split(
    raw_data_dir: Path,
    split: TemporalSplit,
    submission_customer_ids: set[str],
    max_target_customers: int | None,
) -> dict[str, tuple[str, ...]]:
    """Collect validation labels filtered to submission customers and smoke cap.

    Args:
        raw_data_dir: Directory containing H&M raw CSV files.
        split: Temporal split whose labels should be collected.
        submission_customer_ids: Authoritative submission customer universe.
        max_target_customers: Optional deterministic cap.

    Returns:
        Validation labels keyed by selected target customer ID.
    """

    validation_data = summarize_temporal_split_with_labels(
        iter_transaction_events(raw_data_dir), split
    )
    target_customer_ids = select_validation_label_customer_ids(
        validation_labels=validation_data.validation_labels,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=max_target_customers,
    )
    return {
        customer_id: validation_data.validation_labels[customer_id]
        for customer_id in target_customer_ids
    }


def _effective_content_similarity_source_name(args: argparse.Namespace) -> str:
    """Return the source name implied by content-similarity CLI config."""

    source_name = getattr(args, "content_similarity_source_name", MULTIMODAL_SIMILARITY_SOURCE)
    prior_weight = getattr(args, "content_similarity_popularity_prior_weight", 0.0)
    if prior_weight > 0.0 and source_name == MULTIMODAL_SIMILARITY_SOURCE:
        return MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE
    return source_name


def _effective_diagnostics_content_similarity_source_name(args: argparse.Namespace) -> str:
    """Return the source name implied by content diagnostics CLI config."""

    source_name = getattr(args, "source_name", MULTIMODAL_SIMILARITY_SOURCE)
    prior_weight = getattr(args, "popularity_prior_weight", 0.0)
    if prior_weight > 0.0 and source_name == MULTIMODAL_SIMILARITY_SOURCE:
        return MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE
    return source_name


def _age_segment_popularity_lookback_days(args: argparse.Namespace) -> int:
    """Return the resolved age-segment popularity lookback for CLI args."""

    return int(
        args.age_segment_popularity_lookback_days
        if args.age_segment_popularity_lookback_days is not None
        else args.popularity_lookback_days
    )


def _garment_group_popularity_lookback_days(args: argparse.Namespace) -> int:
    """Return the resolved garment-group popularity lookback for CLI args."""

    return int(
        args.garment_group_popularity_lookback_days
        if args.garment_group_popularity_lookback_days is not None
        else args.popularity_lookback_days
    )


def _load_customer_age_segments_if_enabled(
    paths: ProjectPaths,
    args: argparse.Namespace,
) -> dict[str, str] | None:
    """Load customer age-segment mapping when the source is enabled."""

    if not getattr(args, "include_age_segment_popularity", False):
        return None
    return load_customer_age_segments(
        paths.raw_data_dir,
        bucket_size=args.age_segment_bucket_size,
    )


def _load_article_garment_groups_if_enabled(
    paths: ProjectPaths,
    args: argparse.Namespace,
) -> dict[str, str] | None:
    """Load article garment-group mapping when the source is enabled."""

    if not getattr(args, "include_garment_group_popularity", False):
        return None
    return load_article_attribute_values(paths.raw_data_dir)


def _load_article_product_codes_if_enabled(
    paths: ProjectPaths,
    args: argparse.Namespace,
) -> dict[str, str] | None:
    """Load article product-code mapping when the source is enabled."""

    if not getattr(args, "include_product_code_popularity", False):
        return None
    return load_article_attribute_values(paths.raw_data_dir, attribute_column="product_code")


def _product_code_popularity_lookback_days(args: argparse.Namespace) -> int:
    """Return the resolved product-code popularity lookback for CLI args."""

    return int(
        args.product_code_popularity_lookback_days
        if args.product_code_popularity_lookback_days is not None
        else args.popularity_lookback_days
    )


def _learned_ranker_config_slug(args: argparse.Namespace) -> str:
    """Build a compact filesystem-safe descriptor for learned-ranker config.

    Args:
        args: Parsed CLI arguments for the learned-ranker command.

    Returns:
        Compact config slug used in default artifact paths.
    """

    positive_weight = "auto" if args.positive_weight is None else _float_slug(args.positive_weight)
    return (
        f"e{args.epochs}_lr{_float_slug(args.learning_rate)}_"
        f"l2{_float_slug(args.l2)}_pw{positive_weight}_"
        f"maxpw{_float_slug(args.max_auto_positive_weight)}"
    )


def _float_slug(value: float) -> str:
    """Convert a float into a compact path-safe token."""

    return f"{value:g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
