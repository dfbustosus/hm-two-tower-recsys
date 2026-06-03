"""Command-line interface for H&M recommender validation utilities."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import cast

from hm_recsys.data.contracts import validate_hm_data_contract, write_data_contract_report
from hm_recsys.data.io import (
    iter_transaction_events,
    load_article_ids,
    load_submission_customer_ids,
    load_submission_customer_ids_in_order,
)
from hm_recsys.embeddings.article_content import write_article_content_export
from hm_recsys.embeddings.cache_manifest import EmbeddingCacheKind
from hm_recsys.embeddings.generation import (
    ArticleEmbeddingCacheWriteConfig,
    write_article_embedding_cache_from_content_export,
)
from hm_recsys.embeddings.image_inventory import write_article_image_inventory
from hm_recsys.evaluation.submission import (
    validate_submission_file,
    write_submission_file,
    write_submission_validation_report,
)
from hm_recsys.evaluation.temporal import (
    TemporalSplit,
    summarize_temporal_split,
    summarize_temporal_split_with_labels,
    write_temporal_split_summary,
)
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.ranking.deterministic import (
    evaluate_deterministic_ranker_from_csv,
    write_deterministic_ranker_report,
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
    build_learned_linear_ranker_submission_report,
    build_linear_ranker_submission_predictions,
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
from hm_recsys.retrieval.source_names import MULTIMODAL_SIMILARITY_SOURCE
from hm_recsys.training.two_tower_export import (
    TwoTowerExampleExportConfig,
    write_two_tower_example_export,
    write_two_tower_example_export_summary,
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
    ranker_parser.add_argument("--project-root", type=Path, default=None)
    ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    ranker_parser.add_argument("--candidate-output-path", type=Path, default=None)
    ranker_parser.add_argument("--candidate-report-path", type=Path, default=None)
    ranker_parser.add_argument("--report-path", type=Path, default=None)
    ranker_parser.set_defaults(handler=_handle_evaluate_ranker_baseline)

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
    learned_ranker_parser.add_argument("--project-root", type=Path, default=None)
    learned_ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    learned_ranker_parser.add_argument("--train-candidate-output-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--eval-candidate-output-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--train-candidate-report-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--eval-candidate-report-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--report-path", type=Path, default=None)
    learned_ranker_parser.set_defaults(handler=_handle_evaluate_learned_ranker_baseline)

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
        choices=("random",),
        help="Negative sampling strategy. Currently only random is implemented.",
    )
    two_tower_export_parser.add_argument("--project-root", type=Path, default=None)
    two_tower_export_parser.add_argument("--raw-data-dir", type=Path, default=None)
    two_tower_export_parser.add_argument("--examples-path", type=Path, default=None)
    two_tower_export_parser.add_argument("--customer-mapping-path", type=Path, default=None)
    two_tower_export_parser.add_argument("--article-mapping-path", type=Path, default=None)
    two_tower_export_parser.add_argument("--report-path", type=Path, default=None)
    two_tower_export_parser.set_defaults(handler=_handle_export_two_tower_examples)

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
        max_target_customers=args.max_target_customers,
    )
    output_path = _resolve_path_under_root(paths, output_path)
    report_path = args.report_path or paths.candidate_export_report_path(output_path)
    report_path = _resolve_path_under_root(paths, report_path)

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
        max_target_customers=args.max_target_customers,
    )
    candidate_output_path = _resolve_path_under_root(paths, candidate_output_path)
    candidate_report_path = args.candidate_report_path or paths.candidate_export_report_path(
        candidate_output_path
    )
    candidate_report_path = _resolve_path_under_root(paths, candidate_report_path)
    ranker_report_path = args.report_path or paths.ranker_baseline_report_path(
        cutoff=args.cutoff,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
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


def _handle_evaluate_learned_ranker_baseline(args: argparse.Namespace) -> int:
    """Handle the ``evaluate-learned-ranker-baseline`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after training/evaluating and writing artifacts.
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
    report_path = args.report_path or paths.learned_ranker_baseline_report_path(
        train_cutoff=train_split.cutoff.isoformat(),
        evaluation_cutoff=args.cutoff,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
        config_slug=_learned_ranker_config_slug(args),
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


def _handle_rolling_ranker_validation(args: argparse.Namespace) -> int:
    """Handle the ``rolling-ranker-validation`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the rolling validation report and candidate artifacts.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    cutoffs = tuple(args.cutoffs)
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


def _handle_export_two_tower_examples(args: argparse.Namespace) -> int:
    """Handle the ``export-two-tower-examples`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing examples, mappings, and summary metadata.
    """

    if args.negative_sampling != "random":
        raise ValueError(f"unsupported negative_sampling: {args.negative_sampling!r}")

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    config = TwoTowerExampleExportConfig(
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        negative_sampling="random",
        max_positive_examples=args.max_positive_examples,
    )
    examples_path = args.examples_path or paths.two_tower_examples_path(
        cutoff=args.cutoff,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        max_positive_examples=args.max_positive_examples,
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
        f"max_positive_examples={config.max_positive_examples}",
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
    output_path = args.output_path or paths.article_content_export_path
    output_path = _resolve_path_under_root(paths, output_path)
    report_path = args.report_path or paths.article_content_export_report_path
    report_path = _resolve_path_under_root(paths, report_path)

    print(
        "Exporting article content for embedding providers: "
        f"raw_data_dir={paths.raw_data_dir}, max_examples={args.max_examples}",
        flush=True,
    )
    summary = write_article_content_export(
        raw_data_dir=paths.raw_data_dir,
        output_path=output_path,
        report_path=report_path,
        max_examples=args.max_examples,
    )

    print(f"Articles: {summary.article_count}")
    print(f"Records written: {summary.records_written}")
    print(f"Image-available records: {summary.image_available_count}")
    print(f"Image-missing records: {summary.image_missing_count}")
    print(f"Empty combined-text records: {summary.empty_combined_text_count}")
    print(f"Content CSV written to: {summary.output_path}")
    print(f"Summary report written to: {summary.report_path}")
    return 0


def _handle_generate_article_embeddings(args: argparse.Namespace) -> int:
    """Handle the ``generate-article-embeddings`` subcommand."""

    paths = ProjectPaths.from_root(args.project_root, raw_data_dir=args.raw_data_dir)
    article_content_path = args.article_content_path or paths.article_content_export_path
    article_content_path = _resolve_path_under_root(paths, article_content_path)
    provider_slug = f"{args.provider}_{args.model_id}_{args.model_revision}"
    embeddings_path = args.embeddings_path or paths.article_embedding_cache_embeddings_path(
        provider_slug,
        args.embedding_kind,
        "jsonl",
    )
    embeddings_path = _resolve_path_under_root(paths, embeddings_path)
    article_mapping_path = args.article_mapping_path or paths.article_embedding_cache_mapping_path(
        provider_slug,
        args.embedding_kind,
    )
    article_mapping_path = _resolve_path_under_root(paths, article_mapping_path)
    manifest_path = args.manifest_path or paths.article_embedding_cache_manifest_path(
        provider_slug,
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
    report_path = args.report_path or paths.content_similarity_diagnostics_report_path(
        cutoff=args.cutoff,
        source_name=args.source_name,
        max_target_customers=args.max_target_customers,
    )
    report_path = _resolve_path_under_root(paths, report_path)
    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)

    print(
        "Evaluating cached content-similarity source: "
        f"cutoff={args.cutoff}, manifest={manifest_path}, "
        f"source={args.source_name}, max_target_customers={args.max_target_customers}",
        flush=True,
    )
    report = evaluate_cached_content_similarity(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        submission_customer_ids=submission_customer_ids,
        manifest_path=manifest_path,
        source_name=args.source_name,
        evaluation_ks=tuple(args.evaluation_ks),
        max_history_items=args.max_history_items,
        exclude_history=not args.include_history,
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
