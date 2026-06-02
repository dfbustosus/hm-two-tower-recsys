"""Project path resolution helpers for local data and artifact locations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
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
        max_target_customers: int | None = None,
    ) -> Path:
        """Return the default ranker-ready candidate CSV path.

        Args:
            cutoff: Validation cutoff date string.
            k: Maximum candidates per source.
            lookback_days: Recent popularity window length.
            co_visitation_history_items: Optional co-visitation history length.
            co_visitation_neighbors_per_item: Optional co-visitation neighbor count.
            max_target_customers: Optional deterministic smoke-run customer cap.

        Returns:
            Path under ``artifacts/candidate-exports/``.
        """

        name = f"candidates_cutoff_{_safe_name(cutoff)}_" f"lookback_{lookback_days}_k_{k}"
        if co_visitation_history_items is not None and co_visitation_neighbors_per_item is not None:
            name = (
                f"{name}_covis_h{co_visitation_history_items}_n{co_visitation_neighbors_per_item}"
            )
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "candidate-exports" / f"{name}.csv"

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
    ) -> Path:
        """Return the default deterministic ranker baseline report path.

        Args:
            cutoff: Validation cutoff date string.
            k: Recommendation depth for MAP evaluation.
            candidate_k: Maximum candidates per source used to build features.
            max_target_customers: Optional deterministic smoke-run customer cap.

        Returns:
            Path under ``artifacts/ranker-baselines/``.
        """

        name = (
            f"deterministic_ranker_cutoff_{_safe_name(cutoff)}_"
            f"candidate_k_{candidate_k}_rank_k_{k}"
        )
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "ranker-baselines" / f"{name}.json"

    def learned_ranker_baseline_report_path(
        self,
        train_cutoff: str,
        evaluation_cutoff: str,
        k: int,
        candidate_k: int,
        max_target_customers: int | None = None,
        config_slug: str | None = None,
    ) -> Path:
        """Return the default learned linear ranker baseline report path.

        Args:
            train_cutoff: Training-label cutoff date string.
            evaluation_cutoff: Evaluation-label cutoff date string.
            k: Recommendation depth for MAP evaluation.
            candidate_k: Maximum candidates per source used to build features.
            max_target_customers: Optional deterministic smoke-run customer cap.
            config_slug: Optional filesystem-safe training config descriptor.

        Returns:
            Path under ``artifacts/ranker-baselines/``.
        """

        name = (
            f"learned_linear_ranker_train_{_safe_name(train_cutoff)}_"
            f"eval_{_safe_name(evaluation_cutoff)}_candidate_k_{candidate_k}_rank_k_{k}"
        )
        if config_slug is not None:
            name = f"{name}_{_safe_name(config_slug)}"
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "ranker-baselines" / f"{name}.json"

    def rolling_ranker_validation_report_path(
        self,
        cutoffs: Sequence[str],
        k: int,
        candidate_k: int,
        max_target_customers: int | None = None,
        config_slug: str | None = None,
    ) -> Path:
        """Return the default rolling ranker validation report path.

        Args:
            cutoffs: Evaluation cutoff date strings included in the rolling report.
            k: Recommendation depth for MAP evaluation.
            candidate_k: Maximum candidates per source used to build features.
            max_target_customers: Optional deterministic smoke-run customer cap.
            config_slug: Optional filesystem-safe training config descriptor.

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
        if config_slug is not None:
            name = f"{name}_{_safe_name(config_slug)}"
        if max_target_customers is not None:
            name = f"{name}_first_{max_target_customers}_customers"
        return self.artifacts_dir / "ranker-baselines" / f"{name}.json"


def _safe_name(value: str) -> str:
    """Return a filesystem-safe name derived from an arbitrary string."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
