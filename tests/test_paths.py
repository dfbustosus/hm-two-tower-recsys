from pathlib import Path

import pytest

import hm_recsys.infrastructure.paths as path_module
from hm_recsys.infrastructure.paths import RAW_COMPETITION_DIR_NAME, ProjectPaths, find_project_root


def test_find_project_root_from_nested_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    nested = project_root / "src" / "package"
    nested.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")

    assert find_project_root(nested) == project_root


def test_find_project_root_raises_when_no_marker_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(path_module, "PROJECT_MARKERS", ("definitely-not-a-project-marker",))

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


def test_project_paths_include_learned_ranker_submission_locations(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)

    submission_path = paths.learned_ranker_submission_path(
        k=12,
        candidate_k=12,
        lookback_days=7,
        co_visitation_history_items=8,
        co_visitation_neighbors_per_item=100,
        config_slug="e3_lr0p01",
    )

    assert submission_path.parent == tmp_path / "submissions"
    assert submission_path.name.startswith("learned_linear_ranker_lookback_7")
    assert paths.learned_ranker_submission_report_path(submission_path) == (
        tmp_path / "artifacts" / "ranker-submissions" / f"{submission_path.stem}.json"
    )


def test_project_paths_encode_content_similarity_source_configs(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)
    manifest_path = (
        tmp_path
        / "models"
        / "embeddings"
        / "articles"
        / "hf-clip_fashion-clip_main"
        / "multimodal_manifest.json"
    )

    candidate_path = paths.candidate_export_path(
        cutoff="2020-09-16",
        k=100,
        lookback_days=7,
        co_visitation_history_items=8,
        co_visitation_neighbors_per_item=100,
        content_similarity_source_name="multimodal_similarity",
        content_similarity_manifest_path=manifest_path,
        max_target_customers=1000,
    )
    ranker_report_path = paths.ranker_baseline_report_path(
        cutoff="2020-09-16",
        k=12,
        candidate_k=100,
        max_target_customers=1000,
        lookback_days=7,
        co_visitation_history_items=8,
        co_visitation_neighbors_per_item=100,
        content_similarity_source_name="multimodal_similarity",
        content_similarity_manifest_path=manifest_path,
    )
    learned_report_path = paths.learned_ranker_baseline_report_path(
        train_cutoff="2020-09-09",
        evaluation_cutoff="2020-09-16",
        k=12,
        candidate_k=100,
        max_target_customers=1000,
        config_slug="e3_lr0p01",
        lookback_days=7,
        co_visitation_history_items=8,
        co_visitation_neighbors_per_item=100,
        content_similarity_source_name="multimodal_similarity",
        content_similarity_manifest_path=manifest_path,
    )

    assert "content_multimodal_similarity" in candidate_path.stem
    assert "content_multimodal_similarity" in ranker_report_path.stem
    assert "content_multimodal_similarity" in learned_report_path.stem
    assert "hf-clip_fashion-clip" in candidate_path.stem
    assert "lookback_7" in ranker_report_path.stem
    assert "covis_h8_n100" in learned_report_path.stem

    prior_candidate_path = paths.candidate_export_path(
        cutoff="2020-09-16",
        k=100,
        lookback_days=7,
        include_age_segment_popularity=True,
        age_segment_bucket_size=10,
        age_segment_popularity_lookback_days=7,
        include_garment_group_popularity=True,
        garment_group_popularity_lookback_days=7,
        garment_group_max_history_items=8,
        content_similarity_source_name="multimodal_similarity_popularity_prior",
        content_similarity_manifest_path=manifest_path,
        content_similarity_popularity_prior_weight=0.3,
        content_similarity_popularity_lookback_days=30,
        content_similarity_candidate_pool_size=200,
    )
    assert "age_segment_b10_lookback7" in prior_candidate_path.stem
    assert "garment_group_lookback7_h8" in prior_candidate_path.stem
    assert "popw0p3" in prior_candidate_path.stem
    assert len(prior_candidate_path.name) < 255


def test_content_similarity_artifact_names_stay_compact_for_long_manifests(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)
    long_manifest_path = (
        "models/embeddings/articles/"
        "hf-clip_patrickjohncyh_fashion-clip_main_article_content_priority_cutoff_"
        "2020-09-16_lookback_30_first_5000_articles_first_1000_articles/"
        "multimodal_manifest.json"
    )

    candidate_path = paths.candidate_export_path(
        cutoff="2020-09-09",
        k=100,
        lookback_days=7,
        co_visitation_history_items=8,
        co_visitation_neighbors_per_item=100,
        content_similarity_source_name="multimodal_similarity",
        content_similarity_manifest_path=long_manifest_path,
        max_target_customers=1000,
    )
    diagnostics_path = paths.content_similarity_diagnostics_report_path(
        cutoff="2020-09-16",
        source_name="multimodal_similarity",
        manifest_path=long_manifest_path,
        max_target_customers=1000,
    )

    assert len(candidate_path.name) < 255
    assert len(diagnostics_path.name) < 255
    assert "content_multimodal_similarity" in candidate_path.stem
    assert "hf-clip_patrickjohncyh" in candidate_path.stem


def test_project_paths_include_two_tower_export_locations(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)

    examples_path = paths.two_tower_examples_path(
        cutoff="2020-09-16",
        negatives_per_positive=2,
        seed=42,
        max_positive_examples=1000,
    )

    assert examples_path.parent == tmp_path / "artifacts" / "two-tower"
    assert examples_path.name == (
        "two_tower_examples_cutoff_2020-09-16_neg2_seed42_first_1000_positives.csv"
    )
    assert paths.two_tower_customer_mapping_path(examples_path).name.endswith("_customers.csv")
    assert paths.two_tower_article_mapping_path(examples_path).name.endswith("_articles.csv")
    assert paths.two_tower_example_export_report_path(examples_path) == (
        tmp_path / "artifacts" / "two-tower" / f"{examples_path.stem}.json"
    )


def test_project_paths_include_article_image_inventory_locations(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)

    assert paths.article_image_inventory_manifest_path == (
        tmp_path / "artifacts" / "multimodal" / "image-inventory" / "article_image_inventory.csv"
    )
    assert paths.article_image_inventory_report_path == (
        tmp_path / "artifacts" / "multimodal" / "image-inventory" / "article_image_inventory.json"
    )


def test_project_paths_include_article_content_and_embedding_cache_locations(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)

    assert paths.article_content_export_path == (
        tmp_path / "artifacts" / "multimodal" / "article-content" / "article_content.csv"
    )
    assert paths.article_content_export_report_path == (
        tmp_path / "artifacts" / "multimodal" / "article-content" / "article_content.json"
    )
    subset_content_path = paths.article_content_export_path_for_config(
        max_articles=1000,
        priority_cutoff="2020-09-16",
        priority_lookback_days=30,
    )
    assert subset_content_path == (
        tmp_path
        / "artifacts"
        / "multimodal"
        / "article-content"
        / "article_content_priority_cutoff_2020-09-16_lookback_30_first_1000_articles.csv"
    )
    assert paths.article_content_export_report_path_for_path(subset_content_path) == (
        tmp_path
        / "artifacts"
        / "multimodal"
        / "article-content"
        / "article_content_priority_cutoff_2020-09-16_lookback_30_first_1000_articles.json"
    )
    assert paths.article_embedding_cache_dir("FashionCLIP/v1") == (
        tmp_path / "models" / "embeddings" / "articles" / "FashionCLIP_v1"
    )
    assert paths.article_embedding_cache_manifest_path("FashionCLIP/v1", "image") == (
        tmp_path / "models" / "embeddings" / "articles" / "FashionCLIP_v1" / "image_manifest.json"
    )
    assert paths.article_embedding_cache_embeddings_path("FashionCLIP/v1", "image") == (
        tmp_path
        / "models"
        / "embeddings"
        / "articles"
        / "FashionCLIP_v1"
        / "image_embeddings.jsonl"
    )
    assert paths.article_embedding_cache_mapping_path("FashionCLIP/v1", "image") == (
        tmp_path
        / "models"
        / "embeddings"
        / "articles"
        / "FashionCLIP_v1"
        / "image_article_mapping.csv"
    )
    assert paths.content_similarity_diagnostics_report_path(
        "2020-09-16", "multimodal_similarity", max_target_customers=100
    ) == (
        tmp_path
        / "artifacts"
        / "multimodal"
        / "content-similarity"
        / "content_similarity_multimodal_similarity_cutoff_2020-09-16_first_100_customers.json"
    )
    assert paths.content_similarity_diagnostics_report_path(
        "2020-09-16",
        "multimodal_similarity",
        manifest_path="models/embeddings/articles/provider_subset/multimodal_manifest.json",
        max_target_customers=100,
    ) == (
        tmp_path
        / "artifacts"
        / "multimodal"
        / "content-similarity"
        / "content_similarity_multimodal_similarity_cutoff_2020-09-16_"
        "provider_subset_2f43e39e_first_100_customers.json"
    )
    with pytest.raises(ValueError, match="provider_slug"):
        paths.article_embedding_cache_dir("")
    with pytest.raises(ValueError, match="embedding_kind"):
        paths.article_embedding_cache_manifest_path("provider", "")
    with pytest.raises(ValueError, match="embedding_kind"):
        paths.article_embedding_cache_embeddings_path("provider", "")
    with pytest.raises(ValueError, match="vector_format"):
        paths.article_embedding_cache_embeddings_path("provider", "image", "")
    with pytest.raises(ValueError, match="embedding_kind"):
        paths.article_embedding_cache_mapping_path("provider", "")
    with pytest.raises(ValueError, match="cutoff"):
        paths.content_similarity_diagnostics_report_path("", "source")
    with pytest.raises(ValueError, match="source_name"):
        paths.content_similarity_diagnostics_report_path("2020-09-16", "")
