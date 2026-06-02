from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

RAW_COMPETITION_DIR_NAME = "h-and-m-personalized-fashion-recommendations"
PROJECT_MARKERS = ("pyproject.toml", "opencode.json", ".git")


def find_project_root(start: Path | str | None = None) -> Path:
    """Find the repository root by walking upward from ``start``."""
    current = Path.cwd() if start is None else Path(start).expanduser()
    current = current.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate

    raise FileNotFoundError(f"Could not find project root from {current}")


def resolve_under_root(root: Path, path: Path | str) -> Path:
    """Resolve ``path`` relative to ``root`` unless it is already absolute."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical local paths for this recommender project."""

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
        return self.artifacts_dir / "data-contract" / "data_contract_report.json"

    def temporal_split_report_path(self, cutoff: str) -> Path:
        return self.artifacts_dir / "validation" / f"temporal_split_{_safe_name(cutoff)}.json"

    def submission_validation_report_path(self, submission_path: Path | str) -> Path:
        stem = Path(submission_path).stem or "submission"
        return self.artifacts_dir / "submission-validation" / f"{_safe_name(stem)}.json"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
