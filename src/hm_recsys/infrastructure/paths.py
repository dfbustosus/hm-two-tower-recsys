"""Project path resolution helpers for local data and artifact locations."""

from __future__ import annotations

import re
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


def _safe_name(value: str) -> str:
    """Return a filesystem-safe name derived from an arbitrary string."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
