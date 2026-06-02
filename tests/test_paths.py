from pathlib import Path

import pytest

from hm_recsys.infrastructure.paths import RAW_COMPETITION_DIR_NAME, ProjectPaths, find_project_root


def test_find_project_root_from_nested_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    nested = project_root / "src" / "package"
    nested.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")

    assert find_project_root(nested) == project_root


def test_find_project_root_raises_when_no_marker_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_project_root(tmp_path)


def test_project_paths_use_canonical_hm_raw_directory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")

    paths = ProjectPaths.from_root(tmp_path)

    assert paths.root == tmp_path
    assert paths.raw_data_dir == tmp_path / "data" / "raw" / RAW_COMPETITION_DIR_NAME
    assert paths.data_contract_report_path == (
        tmp_path / "artifacts" / "data-contract" / "data_contract_report.json"
    )


def test_project_paths_accept_relative_raw_data_override(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")

    paths = ProjectPaths.from_root(tmp_path, raw_data_dir="input/hm")

    assert paths.raw_data_dir == tmp_path / "input" / "hm"
