"""Project path resolution helpers for local data and artifact locations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

RAW_COMPETITION_DIR_NAME = "h-and-m-personalized-fashion-recommendations"
PROJECT_MARKERS = ("pyproject.toml", "opencode.json", ".git")


def find_project_root(start: Path | str | None = None) -> Path:
    """Find the repository root by walking upward from ``start``.

    Args:
        start: Optional file or directory path to search from. Defaults to the
            current working directory.

    Returns:
        Resolved repository root path.

    Raises:
        FileNotFoundError: If no configured project marker is found.
    """

    current = Path.cwd() if start is None else Path(start).expanduser()
    current = current.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate

    raise FileNotFoundError(f"Could not find project root from {current}")


def resolve_under_root(root: Path, path: Path | str) -> Path:
    """Resolve ``path`` relative to ``root`` unless it is already absolute.

    Args:
        root: Project root used for relative paths.
        path: Candidate path to resolve.

    Returns:
        Absolute resolved path.
    """

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical local paths for this recommender project.

    Attributes:
        root: Repository root.
        raw_data_dir: Local-only raw H&M Kaggle data directory.
        artifacts_dir: Local-only metrics/report artifact directory.
        models_dir: Local-only checkpoints/embeddings/index directory.
        submissions_dir: Local-only generated submission directory.
    """

    root: Path
    raw_data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    submissions_dir: Path

    @classmethod
    def from_root(
        cls,
        root: Path | str | None = None,
        raw_data_dir: Path | str | None = None,
    ) -> ProjectPaths:
        """Construct canonical paths from an optional project root.

        Args:
            root: Optional project root or path within the project.
            raw_data_dir: Optional raw-data override, resolved under ``root`` when
                relative.

        Returns:
            Resolved ``ProjectPaths`` instance.
        """

        project_root = find_project_root(root)
        resolved_raw_data_dir = (
            resolve_under_root(project_root, raw_data_dir)
            if raw_data_dir is not None
            else project_root / "data" / "raw" / RAW_COMPETITION_DIR_NAME
        )
        return cls(
            root=project_root,
            raw_data_dir=resolved_raw_data_dir.resolve(),
            artifacts_dir=(project_root / "artifacts").resolve(),
            models_dir=(project_root / "models").resolve(),
            submissions_dir=(project_root / "submissions").resolve(),
        )

    @property
    def data_contract_report_path(self) -> Path:
        """Return the default data-contract JSON report path.

        Returns:
            Path under ``artifacts/data-contract/``.
        """

        return self.artifacts_dir / "data-contract" / "data_contract_report.json"

    def temporal_split_report_path(self, cutoff: str) -> Path:
        """Return the default temporal split report path for a cutoff.

        Args:
            cutoff: Cutoff string used in the report file name.

        Returns:
            Path under ``artifacts/validation/``.
        """

        return self.artifacts_dir / "validation" / f"temporal_split_{_safe_name(cutoff)}.json"

    def submission_validation_report_path(self, submission_path: Path | str) -> Path:
        """Return the default validation report path for a submission file.

        Args:
            submission_path: Submission CSV path whose stem is used in the report
                name.

        Returns:
            Path under ``artifacts/submission-validation/``.
        """

        stem = Path(submission_path).stem or "submission"
        return self.artifacts_dir / "submission-validation" / f"{_safe_name(stem)}.json"

    def baseline_report_path(self, cutoff: str, lookback_days: int, k: int) -> Path:
        """Return the default baseline evaluation report path.

        Args:
            cutoff: Validation cutoff date string.
            lookback_days: Recent popularity window length.
            k: Recommendation list length.

        Returns:
            Path under ``artifacts/baselines/``.
        """

        name = f"repeat_popularity_cutoff_{_safe_name(cutoff)}_lookback_{lookback_days}_k_{k}.json"
        return self.artifacts_dir / "baselines" / name

    def baseline_submission_path(self, lookback_days: int, k: int) -> Path:
        """Return the default repeat-plus-popularity submission CSV path.

        Args:
            lookback_days: Recent popularity window length.
            k: Recommendation list length.

        Returns:
            Path under ``submissions/``.
        """

        name = f"repeat_popularity_baseline_lookback_{lookback_days}_k_{k}.csv"
        return self.submissions_dir / name

    def candidate_diagnostics_report_path(
        self,
        cutoff: str,
        lookback_days: int,
        max_k: int,
        co_visitation_history_items: int | None = None,
        co_visitation_neighbors_per_item: int | None = None,
    ) -> Path:
        """Return the default candidate diagnostics report path.

        Args:
            cutoff: Validation cutoff date string.
            lookback_days: Recent popularity window length.
            max_k: Maximum candidate depth evaluated.
            co_visitation_history_items: Optional co-visitation history length.
            co_visitation_neighbors_per_item: Optional co-visitation neighbor count.

        Returns:
            Path under ``artifacts/candidate-diagnostics/``.
        """

        name = (
            f"candidate_diagnostics_cutoff_{_safe_name(cutoff)}_"
            f"lookback_{lookback_days}_max_k_{max_k}.json"
        )
        if co_visitation_history_items is not None and co_visitation_neighbors_per_item is not None:
            name = (
                f"candidate_diagnostics_cutoff_{_safe_name(cutoff)}_"
                f"lookback_{lookback_days}_max_k_{max_k}_"
                f"covis_h{co_visitation_history_items}_n{co_visitation_neighbors_per_item}.json"
            )
        return self.artifacts_dir / "candidate-diagnostics" / name

    def candidate_export_path(
        self,
        cutoff: str,
        k: int,
        lookback_days: int,
        co_visitation_history_items: int | None = None,
        co_visitation_neighbors_per_item: int | None = None,
        include_age_segment_popularity: bool = False,
        age_segment_bucket_size: int | None = None,
        age_segment_popularity_lookback_days: int | None = None,
        include_garment_group_popularity: bool = False,
        garment_group_popularity_lookback_days: int | None = None,
        garment_group_max_history_items: int | None = None,
        content_similarity_source_name: str | None = None,
        content_similarity_manifest_path: Path | str | None = None,
        content_similarity_popularity_prior_weight: float | None = None,
        content_similarity_popularity_lookback_days: int | None = None,
        content_similarity_candidate_pool_size: int | None = None,
        max_target_customers: int | None = None,
    ) -> Path:
        """Return the default ranker-ready candidate CSV path.

        Args:
            cutoff: Validation cutoff date string.
            k: Maximum candidates per source.
            lookback_days: Recent popularity window length.
            co_visitation_history_items: Optional co-visitation history length.
            co_visitation_neighbors_per_item: Optional co-visitation neighbor count.
            include_age_segment_popularity: Whether age-segment popularity is included.
            age_segment_bucket_size: Optional age-bucket width.
            age_segment_popularity_lookback_days: Optional segment-popularity lookback.
            include_garment_group_popularity: Whether garment-group affinity rows are included.
            garment_group_popularity_lookback_days: Optional garment-group popularity lookback.
            garment_group_max_history_items: Optional history length used for garment affinities.
            content_similarity_source_name: Optional cached content source name.
            content_similarity_manifest_path: Optional cached embedding manifest path.
            content_similarity_popularity_prior_weight: Optional popularity-prior weight.
            content_similarity_popularity_lookback_days: Optional popularity-prior lookback.
            content_similarity_candidate_pool_size: Optional content reranking pool size.
            max_target_customers: Optional deterministic smoke-run customer cap.

        Returns:
            Path under ``artifacts/candidate-exports/``.
        """

        name = f"candidates_cutoff_{_safe_name(cutoff)}_" f"lookback_{lookback_days}_k_{k}"
        if co_visitation_history_items is not None and co_visitation_neighbors_per_item is not None:
            name = (
                f"{name}_covis_h{co_visitation_history_items}_n{co_visitation_neighbors_per_item}"
            )
        name = _append_age_segment_slug(
            name,
            include_age_segment_popularity=include_age_segment_popularity,
            age_segment_bucket_size=age_segment_bucket_size,
            age_segment_popularity_lookback_days=age_segment_popularity_lookback_days,
        )
        name = _append_garment_group_slug(
            name,
            include_garment_group_popularity=include_garment_group_popularity,
            garment_group_popularity_lookback_days=garment_group_popularity_lookback_days,
            garment_group_max_history_items=garment_group_max_history_items,
        )
        content_slug = _content_similarity_slug(
            source_name=content_similarity_source_name,
            manifest_path=content_similarity_manifest_path,
            popularity_prior_weight=content_similarity_popularity_prior_weight,
            popularity_lookback_days=content_similarity_popularity_lookback_days,
            candidate_pool_size=content_similarity_candidate_pool_size,
        )
        if content_slug is not None:
            name = f"{name}_{content_slug}"
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "candidate-exports" / _artifact_filename(name, "csv")

    def candidate_export_report_path(self, export_path: Path | str) -> Path:
        """Return the default JSON summary path for a candidate CSV export.

        Args:
            export_path: Candidate CSV path whose stem is used for the report name.

        Returns:
            Path under ``artifacts/candidate-exports/``.
        """

        stem = Path(export_path).stem or "candidate_export"
        return self.artifacts_dir / "candidate-exports" / f"{_safe_name(stem)}.json"

    def ranker_baseline_report_path(
        self,
        cutoff: str,
        k: int,
        candidate_k: int,
        max_target_customers: int | None = None,
        lookback_days: int | None = None,
        co_visitation_history_items: int | None = None,
        co_visitation_neighbors_per_item: int | None = None,
        include_age_segment_popularity: bool = False,
        age_segment_bucket_size: int | None = None,
        age_segment_popularity_lookback_days: int | None = None,
        include_garment_group_popularity: bool = False,
        garment_group_popularity_lookback_days: int | None = None,
        garment_group_max_history_items: int | None = None,
        content_similarity_source_name: str | None = None,
        content_similarity_manifest_path: Path | str | None = None,
        content_similarity_popularity_prior_weight: float | None = None,
        content_similarity_popularity_lookback_days: int | None = None,
        content_similarity_candidate_pool_size: int | None = None,
    ) -> Path:
        """Return the default deterministic ranker baseline report path.

        Args:
            cutoff: Validation cutoff date string.
            k: Recommendation depth for MAP evaluation.
            candidate_k: Maximum candidates per source used to build features.
            max_target_customers: Optional deterministic smoke-run customer cap.
            lookback_days: Optional recent-popularity lookback length.
            co_visitation_history_items: Optional co-visitation history length.
            co_visitation_neighbors_per_item: Optional co-visitation neighbor count.
            include_age_segment_popularity: Whether age-segment popularity is included.
            age_segment_bucket_size: Optional age-bucket width.
            age_segment_popularity_lookback_days: Optional segment-popularity lookback.
            include_garment_group_popularity: Whether garment-group affinity rows are included.
            garment_group_popularity_lookback_days: Optional garment-group popularity lookback.
            garment_group_max_history_items: Optional history length used for garment affinities.
            content_similarity_source_name: Optional cached content source name.
            content_similarity_manifest_path: Optional cached embedding manifest path.
            content_similarity_popularity_prior_weight: Optional popularity-prior weight.
            content_similarity_popularity_lookback_days: Optional popularity-prior lookback.
            content_similarity_candidate_pool_size: Optional content reranking pool size.

        Returns:
            Path under ``artifacts/ranker-baselines/``.
        """

        name = (
            f"deterministic_ranker_cutoff_{_safe_name(cutoff)}_"
            f"candidate_k_{candidate_k}_rank_k_{k}"
        )
        name = _append_source_config_slug(
            name,
            lookback_days=lookback_days,
            co_visitation_history_items=co_visitation_history_items,
            co_visitation_neighbors_per_item=co_visitation_neighbors_per_item,
            include_age_segment_popularity=include_age_segment_popularity,
            age_segment_bucket_size=age_segment_bucket_size,
            age_segment_popularity_lookback_days=age_segment_popularity_lookback_days,
            include_garment_group_popularity=include_garment_group_popularity,
            garment_group_popularity_lookback_days=garment_group_popularity_lookback_days,
            garment_group_max_history_items=garment_group_max_history_items,
            content_similarity_source_name=content_similarity_source_name,
            content_similarity_manifest_path=content_similarity_manifest_path,
            content_similarity_popularity_prior_weight=content_similarity_popularity_prior_weight,
            content_similarity_popularity_lookback_days=content_similarity_popularity_lookback_days,
            content_similarity_candidate_pool_size=content_similarity_candidate_pool_size,
        )
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "ranker-baselines" / _artifact_filename(name, "json")

    def learned_ranker_baseline_report_path(
        self,
        train_cutoff: str,
        evaluation_cutoff: str,
        k: int,
        candidate_k: int,
        max_target_customers: int | None = None,
        config_slug: str | None = None,
        lookback_days: int | None = None,
        co_visitation_history_items: int | None = None,
        co_visitation_neighbors_per_item: int | None = None,
        include_age_segment_popularity: bool = False,
        age_segment_bucket_size: int | None = None,
        age_segment_popularity_lookback_days: int | None = None,
        include_garment_group_popularity: bool = False,
        garment_group_popularity_lookback_days: int | None = None,
        garment_group_max_history_items: int | None = None,
        content_similarity_source_name: str | None = None,
        content_similarity_manifest_path: Path | str | None = None,
        content_similarity_popularity_prior_weight: float | None = None,
        content_similarity_popularity_lookback_days: int | None = None,
        content_similarity_candidate_pool_size: int | None = None,
    ) -> Path:
        """Return the default learned linear ranker baseline report path.

        Args:
            train_cutoff: Training-label cutoff date string.
            evaluation_cutoff: Evaluation-label cutoff date string.
            k: Recommendation depth for MAP evaluation.
            candidate_k: Maximum candidates per source used to build features.
            max_target_customers: Optional deterministic smoke-run customer cap.
            config_slug: Optional filesystem-safe training config descriptor.
            lookback_days: Optional recent-popularity lookback length.
            co_visitation_history_items: Optional co-visitation history length.
            co_visitation_neighbors_per_item: Optional co-visitation neighbor count.
            include_age_segment_popularity: Whether age-segment popularity is included.
            age_segment_bucket_size: Optional age-bucket width.
            age_segment_popularity_lookback_days: Optional segment-popularity lookback.
            include_garment_group_popularity: Whether garment-group affinity rows are included.
            garment_group_popularity_lookback_days: Optional garment-group popularity lookback.
            garment_group_max_history_items: Optional history length used for garment affinities.
            content_similarity_source_name: Optional cached content source name.
            content_similarity_manifest_path: Optional cached embedding manifest path.
            content_similarity_popularity_prior_weight: Optional popularity-prior weight.
            content_similarity_popularity_lookback_days: Optional popularity-prior lookback.
            content_similarity_candidate_pool_size: Optional content reranking pool size.

        Returns:
            Path under ``artifacts/ranker-baselines/``.
        """

        name = (
            f"learned_linear_ranker_train_{_safe_name(train_cutoff)}_"
            f"eval_{_safe_name(evaluation_cutoff)}_candidate_k_{candidate_k}_rank_k_{k}"
        )
        name = _append_source_config_slug(
            name,
            lookback_days=lookback_days,
            co_visitation_history_items=co_visitation_history_items,
            co_visitation_neighbors_per_item=co_visitation_neighbors_per_item,
            include_age_segment_popularity=include_age_segment_popularity,
            age_segment_bucket_size=age_segment_bucket_size,
            age_segment_popularity_lookback_days=age_segment_popularity_lookback_days,
            include_garment_group_popularity=include_garment_group_popularity,
            garment_group_popularity_lookback_days=garment_group_popularity_lookback_days,
            garment_group_max_history_items=garment_group_max_history_items,
            content_similarity_source_name=content_similarity_source_name,
            content_similarity_manifest_path=content_similarity_manifest_path,
            content_similarity_popularity_prior_weight=content_similarity_popularity_prior_weight,
            content_similarity_popularity_lookback_days=content_similarity_popularity_lookback_days,
            content_similarity_candidate_pool_size=content_similarity_candidate_pool_size,
        )
        if config_slug is not None:
            name = f"{name}_{_safe_name(config_slug)}"
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "ranker-baselines" / _artifact_filename(name, "json")

    def deterministic_ranker_tuning_report_path(
        self,
        train_cutoff: str,
        evaluation_cutoff: str,
        k: int,
        candidate_k: int,
        max_target_customers: int | None = None,
        lookback_days: int | None = None,
        co_visitation_history_items: int | None = None,
        co_visitation_neighbors_per_item: int | None = None,
        include_age_segment_popularity: bool = False,
        age_segment_bucket_size: int | None = None,
        age_segment_popularity_lookback_days: int | None = None,
        include_garment_group_popularity: bool = False,
        garment_group_popularity_lookback_days: int | None = None,
        garment_group_max_history_items: int | None = None,
        content_similarity_source_name: str | None = None,
        content_similarity_manifest_path: Path | str | None = None,
        content_similarity_popularity_prior_weight: float | None = None,
        content_similarity_popularity_lookback_days: int | None = None,
        content_similarity_candidate_pool_size: int | None = None,
    ) -> Path:
        """Return the default deterministic-ranker tuning report path.

        Args:
            train_cutoff: Tuning-label cutoff date string.
            evaluation_cutoff: Evaluation-label cutoff date string.
            k: Recommendation depth for MAP evaluation.
            candidate_k: Maximum candidates per source used to build features.
            max_target_customers: Optional deterministic smoke-run customer cap.
            lookback_days: Optional recent-popularity lookback length.
            co_visitation_history_items: Optional co-visitation history length.
            co_visitation_neighbors_per_item: Optional co-visitation neighbor count.
            include_age_segment_popularity: Whether age-segment popularity is included.
            age_segment_bucket_size: Optional age-bucket width.
            age_segment_popularity_lookback_days: Optional segment-popularity lookback.
            include_garment_group_popularity: Whether garment-group affinity rows are included.
            garment_group_popularity_lookback_days: Optional garment-group popularity lookback.
            garment_group_max_history_items: Optional history length used for garment affinities.
            content_similarity_source_name: Optional cached content source name.
            content_similarity_manifest_path: Optional cached embedding manifest path.
            content_similarity_popularity_prior_weight: Optional popularity-prior weight.
            content_similarity_popularity_lookback_days: Optional popularity-prior lookback.
            content_similarity_candidate_pool_size: Optional content reranking pool size.

        Returns:
            Path under ``artifacts/ranker-baselines/``.
        """

        name = (
            f"deterministic_ranker_tuning_train_{_safe_name(train_cutoff)}_"
            f"eval_{_safe_name(evaluation_cutoff)}_candidate_k_{candidate_k}_rank_k_{k}"
        )
        name = _append_source_config_slug(
            name,
            lookback_days=lookback_days,
            co_visitation_history_items=co_visitation_history_items,
            co_visitation_neighbors_per_item=co_visitation_neighbors_per_item,
            include_age_segment_popularity=include_age_segment_popularity,
            age_segment_bucket_size=age_segment_bucket_size,
            age_segment_popularity_lookback_days=age_segment_popularity_lookback_days,
            include_garment_group_popularity=include_garment_group_popularity,
            garment_group_popularity_lookback_days=garment_group_popularity_lookback_days,
            garment_group_max_history_items=garment_group_max_history_items,
            content_similarity_source_name=content_similarity_source_name,
            content_similarity_manifest_path=content_similarity_manifest_path,
            content_similarity_popularity_prior_weight=content_similarity_popularity_prior_weight,
            content_similarity_popularity_lookback_days=content_similarity_popularity_lookback_days,
            content_similarity_candidate_pool_size=content_similarity_candidate_pool_size,
        )
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "ranker-baselines" / _artifact_filename(name, "json")

    def rolling_ranker_validation_report_path(
        self,
        cutoffs: Sequence[str],
        k: int,
        candidate_k: int,
        max_target_customers: int | None = None,
        config_slug: str | None = None,
        lookback_days: int | None = None,
        co_visitation_history_items: int | None = None,
        co_visitation_neighbors_per_item: int | None = None,
        include_age_segment_popularity: bool = False,
        age_segment_bucket_size: int | None = None,
        age_segment_popularity_lookback_days: int | None = None,
        include_garment_group_popularity: bool = False,
        garment_group_popularity_lookback_days: int | None = None,
        garment_group_max_history_items: int | None = None,
        content_similarity_source_name: str | None = None,
        content_similarity_manifest_path: Path | str | None = None,
        content_similarity_popularity_prior_weight: float | None = None,
        content_similarity_popularity_lookback_days: int | None = None,
        content_similarity_candidate_pool_size: int | None = None,
    ) -> Path:
        """Return the default rolling ranker validation report path.

        Args:
            cutoffs: Evaluation cutoff date strings included in the rolling report.
            k: Recommendation depth for MAP evaluation.
            candidate_k: Maximum candidates per source used to build features.
            max_target_customers: Optional deterministic smoke-run customer cap.
            config_slug: Optional filesystem-safe training config descriptor.
            lookback_days: Optional recent-popularity lookback length.
            co_visitation_history_items: Optional co-visitation history length.
            co_visitation_neighbors_per_item: Optional co-visitation neighbor count.
            include_age_segment_popularity: Whether age-segment popularity is included.
            age_segment_bucket_size: Optional age-bucket width.
            age_segment_popularity_lookback_days: Optional segment-popularity lookback.
            include_garment_group_popularity: Whether garment-group affinity rows are included.
            garment_group_popularity_lookback_days: Optional garment-group popularity lookback.
            garment_group_max_history_items: Optional history length used for garment affinities.
            content_similarity_source_name: Optional cached content source name.
            content_similarity_manifest_path: Optional cached embedding manifest path.
            content_similarity_popularity_prior_weight: Optional popularity-prior weight.
            content_similarity_popularity_lookback_days: Optional popularity-prior lookback.
            content_similarity_candidate_pool_size: Optional content reranking pool size.

        Returns:
            Path under ``artifacts/ranker-baselines/``.

        Raises:
            ValueError: If ``cutoffs`` is empty.
        """

        if not cutoffs:
            raise ValueError("cutoffs must contain at least one date")
        first_cutoff = _safe_name(cutoffs[0])
        last_cutoff = _safe_name(cutoffs[-1])
        name = (
            f"rolling_linear_ranker_eval_{first_cutoff}_to_{last_cutoff}_"
            f"windows_{len(cutoffs)}_candidate_k_{candidate_k}_rank_k_{k}"
        )
        name = _append_source_config_slug(
            name,
            lookback_days=lookback_days,
            co_visitation_history_items=co_visitation_history_items,
            co_visitation_neighbors_per_item=co_visitation_neighbors_per_item,
            include_age_segment_popularity=include_age_segment_popularity,
            age_segment_bucket_size=age_segment_bucket_size,
            age_segment_popularity_lookback_days=age_segment_popularity_lookback_days,
            include_garment_group_popularity=include_garment_group_popularity,
            garment_group_popularity_lookback_days=garment_group_popularity_lookback_days,
            garment_group_max_history_items=garment_group_max_history_items,
            content_similarity_source_name=content_similarity_source_name,
            content_similarity_manifest_path=content_similarity_manifest_path,
            content_similarity_popularity_prior_weight=content_similarity_popularity_prior_weight,
            content_similarity_popularity_lookback_days=content_similarity_popularity_lookback_days,
            content_similarity_candidate_pool_size=content_similarity_candidate_pool_size,
        )
        if config_slug is not None:
            name = f"{name}_{_safe_name(config_slug)}"
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "ranker-baselines" / _artifact_filename(name, "json")

    def learned_ranker_submission_path(
        self,
        k: int,
        candidate_k: int,
        lookback_days: int,
        co_visitation_history_items: int | None = None,
        co_visitation_neighbors_per_item: int | None = None,
        config_slug: str | None = None,
        content_similarity_source_name: str | None = None,
        content_similarity_manifest_path: Path | str | None = None,
        content_similarity_popularity_prior_weight: float | None = None,
        content_similarity_popularity_lookback_days: int | None = None,
        content_similarity_candidate_pool_size: int | None = None,
    ) -> Path:
        """Return the default learned linear ranker submission CSV path.

        Args:
            k: Recommendation depth for each submission row.
            candidate_k: Maximum candidates per source used for final ranking.
            lookback_days: Recent popularity window length.
            co_visitation_history_items: Optional co-visitation history length.
            co_visitation_neighbors_per_item: Optional co-visitation neighbor count.
            config_slug: Optional filesystem-safe training config descriptor.
            content_similarity_source_name: Optional cached content source name.
            content_similarity_manifest_path: Optional cached embedding manifest path.
            content_similarity_popularity_prior_weight: Optional popularity-prior weight.
            content_similarity_popularity_lookback_days: Optional popularity-prior lookback.
            content_similarity_candidate_pool_size: Optional content reranking pool size.

        Returns:
            Path under ``submissions/``.
        """

        name = (
            f"learned_linear_ranker_lookback_{lookback_days}_"
            f"candidate_k_{candidate_k}_rank_k_{k}"
        )
        if co_visitation_history_items is not None and co_visitation_neighbors_per_item is not None:
            name = (
                f"{name}_covis_h{co_visitation_history_items}_"
                f"n{co_visitation_neighbors_per_item}"
            )
        content_slug = _content_similarity_slug(
            source_name=content_similarity_source_name,
            manifest_path=content_similarity_manifest_path,
            popularity_prior_weight=content_similarity_popularity_prior_weight,
            popularity_lookback_days=content_similarity_popularity_lookback_days,
            candidate_pool_size=content_similarity_candidate_pool_size,
        )
        if content_slug is not None:
            name = f"{name}_{content_slug}"
        if config_slug is not None:
            name = f"{name}_{_safe_name(config_slug)}"
        return self.submissions_dir / _artifact_filename(name, "csv")

    def deterministic_ranker_submission_path(
        self,
        k: int,
        candidate_k: int,
        lookback_days: int,
        co_visitation_history_items: int | None = None,
        co_visitation_neighbors_per_item: int | None = None,
        include_age_segment_popularity: bool = False,
        age_segment_bucket_size: int | None = None,
        age_segment_popularity_lookback_days: int | None = None,
        include_garment_group_popularity: bool = False,
        garment_group_popularity_lookback_days: int | None = None,
        garment_group_max_history_items: int | None = None,
        tuning_slug: str | None = None,
    ) -> Path:
        """Return the default tuned deterministic-ranker submission CSV path."""

        name = (
            f"deterministic_ranker_tuned_lookback_{lookback_days}_"
            f"candidate_k_{candidate_k}_rank_k_{k}"
        )
        if co_visitation_history_items is not None and co_visitation_neighbors_per_item is not None:
            name = (
                f"{name}_covis_h{co_visitation_history_items}_"
                f"n{co_visitation_neighbors_per_item}"
            )
        name = _append_age_segment_slug(
            name,
            include_age_segment_popularity=include_age_segment_popularity,
            age_segment_bucket_size=age_segment_bucket_size,
            age_segment_popularity_lookback_days=age_segment_popularity_lookback_days,
        )
        name = _append_garment_group_slug(
            name,
            include_garment_group_popularity=include_garment_group_popularity,
            garment_group_popularity_lookback_days=garment_group_popularity_lookback_days,
            garment_group_max_history_items=garment_group_max_history_items,
        )
        if tuning_slug is not None:
            name = f"{name}_{_safe_name(tuning_slug)}"
        return self.submissions_dir / _artifact_filename(name, "csv")

    def learned_ranker_submission_report_path(self, submission_path: Path | str) -> Path:
        """Return the default JSON report path for a learned-ranker submission.

        Args:
            submission_path: Submission CSV path whose stem is used in the report name.

        Returns:
            Path under ``artifacts/ranker-submissions/``.
        """

        stem = Path(submission_path).stem or "learned_ranker_submission"
        return self.artifacts_dir / "ranker-submissions" / f"{_safe_name(stem)}.json"

    def deterministic_ranker_submission_report_path(self, submission_path: Path | str) -> Path:
        """Return the default JSON report path for deterministic-ranker submission."""

        stem = Path(submission_path).stem or "deterministic_ranker_submission"
        return self.artifacts_dir / "ranker-submissions" / f"{_safe_name(stem)}.json"

    def two_tower_examples_path(
        self,
        cutoff: str,
        negatives_per_positive: int,
        seed: int,
        max_positive_examples: int | None = None,
    ) -> Path:
        """Return the default two-tower examples CSV path.

        Args:
            cutoff: Exclusive training cutoff date string.
            negatives_per_positive: Requested negatives per positive pair.
            seed: Deterministic negative-sampling seed.
            max_positive_examples: Optional deterministic smoke-run cap.

        Returns:
            Path under ``artifacts/two-tower/``.
        """

        name = (
            f"two_tower_examples_cutoff_{_safe_name(cutoff)}_"
            f"neg{negatives_per_positive}_seed{seed}"
        )
        if max_positive_examples is not None:
            name = f"{name}_first_{max_positive_examples}_positives"
        return self.artifacts_dir / "two-tower" / f"{name}.csv"

    def two_tower_customer_mapping_path(self, examples_path: Path | str) -> Path:
        """Return the default customer mapping path for a two-tower export.

        Args:
            examples_path: Examples CSV path whose stem is used for the mapping name.

        Returns:
            Path under ``artifacts/two-tower/``.
        """

        stem = Path(examples_path).stem or "two_tower_examples"
        return self.artifacts_dir / "two-tower" / f"{_safe_name(stem)}_customers.csv"

    def two_tower_article_mapping_path(self, examples_path: Path | str) -> Path:
        """Return the default article mapping path for a two-tower export.

        Args:
            examples_path: Examples CSV path whose stem is used for the mapping name.

        Returns:
            Path under ``artifacts/two-tower/``.
        """

        stem = Path(examples_path).stem or "two_tower_examples"
        return self.artifacts_dir / "two-tower" / f"{_safe_name(stem)}_articles.csv"

    def two_tower_example_export_report_path(self, examples_path: Path | str) -> Path:
        """Return the default JSON report path for a two-tower examples export.

        Args:
            examples_path: Examples CSV path whose stem is used for the report name.

        Returns:
            Path under ``artifacts/two-tower/``.
        """

        stem = Path(examples_path).stem or "two_tower_examples"
        return self.artifacts_dir / "two-tower" / f"{_safe_name(stem)}.json"

    @property
    def article_image_inventory_manifest_path(self) -> Path:
        """Return the default article image inventory CSV path.

        Returns:
            Path under ``artifacts/multimodal/image-inventory/``.
        """

        inventory_dir = self.artifacts_dir / "multimodal" / "image-inventory"
        return inventory_dir / "article_image_inventory.csv"

    @property
    def article_image_inventory_report_path(self) -> Path:
        """Return the default article image inventory JSON report path.

        Returns:
            Path under ``artifacts/multimodal/image-inventory/``.
        """

        inventory_dir = self.artifacts_dir / "multimodal" / "image-inventory"
        return inventory_dir / "article_image_inventory.json"

    @property
    def article_content_export_path(self) -> Path:
        """Return the default article content export CSV path.

        Returns:
            Path under ``artifacts/multimodal/article-content/``.
        """

        content_dir = self.artifacts_dir / "multimodal" / "article-content"
        return content_dir / "article_content.csv"

    def article_content_export_path_for_config(
        self,
        max_articles: int | None = None,
        priority_cutoff: str | None = None,
        priority_lookback_days: int | None = None,
    ) -> Path:
        """Return an article-content CSV path for a bounded/prioritized export."""

        name = "article_content"
        if priority_cutoff is not None:
            name = f"{name}_priority_cutoff_{_safe_name(priority_cutoff)}"
        if priority_lookback_days is not None:
            name = f"{name}_lookback_{priority_lookback_days}"
        if max_articles is not None:
            name = f"{name}_first_{max_articles}_articles"
        return self.artifacts_dir / "multimodal" / "article-content" / f"{name}.csv"

    @property
    def article_content_export_report_path(self) -> Path:
        """Return the default article content export JSON report path.

        Returns:
            Path under ``artifacts/multimodal/article-content/``.
        """

        content_dir = self.artifacts_dir / "multimodal" / "article-content"
        return content_dir / "article_content.json"

    def article_content_export_report_path_for_path(self, export_path: Path | str) -> Path:
        """Return the JSON report path matching an article-content CSV path."""

        stem = Path(export_path).stem or "article_content"
        return self.artifacts_dir / "multimodal" / "article-content" / f"{_safe_name(stem)}.json"

    def article_embedding_cache_dir(self, provider_slug: str) -> Path:
        """Return the default article embedding cache directory for a provider.

        Args:
            provider_slug: Filesystem-safe provider/model descriptor.

        Returns:
            Path under ``models/embeddings/articles/``.
        """

        if not provider_slug:
            raise ValueError("provider_slug must not be empty")
        return self.models_dir / "embeddings" / "articles" / _safe_name(provider_slug)

    def article_embedding_cache_manifest_path(
        self,
        provider_slug: str,
        embedding_kind: str,
    ) -> Path:
        """Return the default article embedding cache manifest path.

        Args:
            provider_slug: Filesystem-safe provider/model descriptor.
            embedding_kind: Embedding family such as ``image`` or ``text``.

        Returns:
            JSON manifest path under ``models/embeddings/articles/``.
        """

        if not embedding_kind:
            raise ValueError("embedding_kind must not be empty")
        cache_dir = self.article_embedding_cache_dir(provider_slug)
        return cache_dir / f"{_safe_name(embedding_kind)}_manifest.json"

    def article_embedding_cache_embeddings_path(
        self,
        provider_slug: str,
        embedding_kind: str,
        vector_format: str = "jsonl",
    ) -> Path:
        """Return the default article embedding vector-cache path.

        Args:
            provider_slug: Filesystem-safe provider/model descriptor.
            embedding_kind: Embedding family such as ``image`` or ``text``.
            vector_format: File extension/format such as ``jsonl`` or ``npy``.

        Returns:
            Embedding cache path under ``models/embeddings/articles/``.
        """

        if not embedding_kind:
            raise ValueError("embedding_kind must not be empty")
        if not vector_format:
            raise ValueError("vector_format must not be empty")
        cache_dir = self.article_embedding_cache_dir(provider_slug)
        return cache_dir / f"{_safe_name(embedding_kind)}_embeddings.{_safe_name(vector_format)}"

    def article_embedding_cache_mapping_path(
        self,
        provider_slug: str,
        embedding_kind: str,
    ) -> Path:
        """Return the default article embedding row-mapping path."""

        if not embedding_kind:
            raise ValueError("embedding_kind must not be empty")
        cache_dir = self.article_embedding_cache_dir(provider_slug)
        return cache_dir / f"{_safe_name(embedding_kind)}_article_mapping.csv"

    def content_similarity_diagnostics_report_path(
        self,
        cutoff: str,
        source_name: str,
        manifest_path: Path | str | None = None,
        popularity_prior_weight: float | None = None,
        popularity_lookback_days: int | None = None,
        candidate_pool_size: int | None = None,
        max_target_customers: int | None = None,
    ) -> Path:
        """Return the default cached content-similarity diagnostics report path."""

        if not cutoff:
            raise ValueError("cutoff must not be empty")
        if not source_name:
            raise ValueError("source_name must not be empty")
        name = f"content_similarity_{_safe_name(source_name)}_cutoff_{_safe_name(cutoff)}"
        if manifest_path is not None:
            name = f"{name}_{_content_manifest_slug(manifest_path)}"
        name = _append_content_prior_slug(
            name,
            popularity_prior_weight=popularity_prior_weight,
            popularity_lookback_days=popularity_lookback_days,
            candidate_pool_size=candidate_pool_size,
        )
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return (
            self.artifacts_dir
            / "multimodal"
            / "content-similarity"
            / _artifact_filename(name, "json")
        )


def _safe_name(value: str) -> str:
    """Return a filesystem-safe name derived from an arbitrary string."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _append_source_config_slug(
    name: str,
    lookback_days: int | None = None,
    co_visitation_history_items: int | None = None,
    co_visitation_neighbors_per_item: int | None = None,
    include_age_segment_popularity: bool = False,
    age_segment_bucket_size: int | None = None,
    age_segment_popularity_lookback_days: int | None = None,
    include_garment_group_popularity: bool = False,
    garment_group_popularity_lookback_days: int | None = None,
    garment_group_max_history_items: int | None = None,
    content_similarity_source_name: str | None = None,
    content_similarity_manifest_path: Path | str | None = None,
    content_similarity_popularity_prior_weight: float | None = None,
    content_similarity_popularity_lookback_days: int | None = None,
    content_similarity_candidate_pool_size: int | None = None,
) -> str:
    """Append candidate-source configuration tokens to an artifact stem."""

    if lookback_days is not None:
        name = f"{name}_lookback_{lookback_days}"
    if co_visitation_history_items is not None and co_visitation_neighbors_per_item is not None:
        name = f"{name}_covis_h{co_visitation_history_items}_n{co_visitation_neighbors_per_item}"
    name = _append_age_segment_slug(
        name,
        include_age_segment_popularity=include_age_segment_popularity,
        age_segment_bucket_size=age_segment_bucket_size,
        age_segment_popularity_lookback_days=age_segment_popularity_lookback_days,
    )
    name = _append_garment_group_slug(
        name,
        include_garment_group_popularity=include_garment_group_popularity,
        garment_group_popularity_lookback_days=garment_group_popularity_lookback_days,
        garment_group_max_history_items=garment_group_max_history_items,
    )
    content_slug = _content_similarity_slug(
        source_name=content_similarity_source_name,
        manifest_path=content_similarity_manifest_path,
        popularity_prior_weight=content_similarity_popularity_prior_weight,
        popularity_lookback_days=content_similarity_popularity_lookback_days,
        candidate_pool_size=content_similarity_candidate_pool_size,
    )
    if content_slug is not None:
        name = f"{name}_{content_slug}"
    return name


def _content_similarity_slug(
    source_name: str | None,
    manifest_path: Path | str | None,
    popularity_prior_weight: float | None = None,
    popularity_lookback_days: int | None = None,
    candidate_pool_size: int | None = None,
) -> str | None:
    """Return a path-safe token for a cached content-similarity source."""

    if source_name is None and manifest_path is None:
        return None
    parts = ["content"]
    if source_name:
        parts.append(_safe_name(source_name))
    slug = _append_content_prior_slug(
        "_".join(parts),
        popularity_prior_weight=popularity_prior_weight,
        popularity_lookback_days=popularity_lookback_days,
        candidate_pool_size=candidate_pool_size,
    )
    if manifest_path is not None:
        slug = f"{slug}_{_content_manifest_slug(manifest_path)}"
    return slug


def _append_age_segment_slug(
    name: str,
    *,
    include_age_segment_popularity: bool,
    age_segment_bucket_size: int | None,
    age_segment_popularity_lookback_days: int | None,
) -> str:
    """Append age-segment candidate-source config tokens to a path stem."""

    if not include_age_segment_popularity:
        return name
    name = f"{name}_age_segment"
    if age_segment_bucket_size is not None:
        name = f"{name}_b{age_segment_bucket_size}"
    if age_segment_popularity_lookback_days is not None:
        name = f"{name}_lookback{age_segment_popularity_lookback_days}"
    return name


def _append_garment_group_slug(
    name: str,
    *,
    include_garment_group_popularity: bool,
    garment_group_popularity_lookback_days: int | None,
    garment_group_max_history_items: int | None,
) -> str:
    """Append garment-group candidate-source config tokens to a path stem."""

    if not include_garment_group_popularity:
        return name
    name = f"{name}_garment_group"
    if garment_group_popularity_lookback_days is not None:
        name = f"{name}_lookback{garment_group_popularity_lookback_days}"
    if garment_group_max_history_items is not None:
        name = f"{name}_h{garment_group_max_history_items}"
    return name


def _append_content_prior_slug(
    name: str,
    popularity_prior_weight: float | None = None,
    popularity_lookback_days: int | None = None,
    candidate_pool_size: int | None = None,
) -> str:
    """Append compact content-prior config tokens to a path stem."""

    if popularity_prior_weight is not None and popularity_prior_weight > 0.0:
        name = f"{name}_popw{_float_path_token(popularity_prior_weight)}"
    if popularity_lookback_days is not None:
        name = f"{name}_poplookback{popularity_lookback_days}"
    if candidate_pool_size is not None:
        name = f"{name}_pool{candidate_pool_size}"
    return name


def _content_manifest_slug(manifest_path: Path | str) -> str:
    """Return a compact, stable slug for an embedding-cache manifest path."""

    manifest = Path(manifest_path)
    label_source = manifest.parent.name or manifest.stem or "manifest"
    digest = sha256(str(manifest).encode("utf-8")).hexdigest()[:8]
    return f"{_compact_safe_name(label_source, max_length=24)}_{digest}"


def _compact_safe_name(value: str, max_length: int = 48) -> str:
    """Return a safe slug, shortened with a stable digest when needed."""

    safe_value = _safe_name(value)
    if len(safe_value) <= max_length:
        return safe_value
    digest = sha256(safe_value.encode("utf-8")).hexdigest()[:8]
    return f"{safe_value[:max_length]}_{digest}"


def _artifact_filename(stem: str, extension: str, max_stem_length: int = 180) -> str:
    """Return a filename short enough for common local filesystems."""

    return f"{_compact_safe_name(stem, max_length=max_stem_length)}.{extension}"


def _float_path_token(value: float) -> str:
    """Return a compact filesystem-safe token for a float."""

    return f"{value:g}".replace("-", "m").replace(".", "p")
