"""Checks that the pydantic models in weirdchat.types stay in sync with the
Transluce/WeirdChat dataset on the Hugging Face Hub.

For each parquet config this downloads one shard and (a) compares the parquet
column set against the model's fields, and (b) validates real rows through the
model — ``extra="forbid"`` makes added/renamed columns fail, and pydantic
validation catches type changes. It also checks the committed ``schema/*.json``
contract files against the models' JSON Schema exports.

Requires read access to the dataset repo: either a cached login or an
``HF_TOKEN`` env var (set as a secret in CI). Skips if the repo is unreachable
without credentials so fork PRs don't fail spuriously.
"""

# huggingface_hub's signatures are partially untyped, which strict pyright reports at the
# import site; the values we consume from it are fully annotated below.
# pyright: reportUnknownVariableType=false

import json
from typing import Any

import pytest
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from pydantic import BaseModel

from weirdchat.types import Behavior, PairwiseJudgment, Pattern, Prompt, SearchCompute, Transcript

REPO_ID = "Transluce/WeirdChat"
ROWS_PER_SHARD = 200  # validated per parquet shard; full-file validation is overkill for CI

# config name -> (row model, glob for its parquet shards)
CONFIGS: dict[str, tuple[type[BaseModel], str]] = {
    "rubrics": (Behavior, "data/rubrics.parquet"),
    "patterns": (Pattern, "data/patterns.parquet"),
    "prompts": (Prompt, "data/prompts/*/*.parquet"),
    "transcripts": (Transcript, "data/transcripts/*/*.parquet"),
    "pairwise_judgments": (PairwiseJudgment, "data/pairwise/*.parquet"),
    "search_compute": (SearchCompute, "data/search_compute.parquet"),
}


@pytest.fixture(scope="session")
def repo_files() -> list[str]:
    try:
        return HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as e:
        pytest.skip(f"cannot access {REPO_ID} (missing HF_TOKEN?): {e}")


def _shards(repo_files: list[str], pattern: str) -> list[str]:
    import fnmatch

    matches = sorted(f for f in repo_files if fnmatch.fnmatch(f, pattern))
    assert matches, f"no files in {REPO_ID} match {pattern!r}"
    return matches


@pytest.mark.parametrize("config", CONFIGS)
def test_parquet_matches_model(config: str, repo_files: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    model, pattern = CONFIGS[config]
    # One shard per config: all shards of a config are written with the same schema, and
    # validating one keeps CI downloads small (transcripts alone span ~90 files).
    shard = _shards(repo_files, pattern)[0]
    path = hf_hub_download(REPO_ID, shard, repo_type="dataset")
    table: pa.Table = pq.read_table(path)  # pyright: ignore[reportUnknownMemberType]

    assert set(table.column_names) == set(model.model_fields), (
        f"{config}: parquet columns diverge from {model.__name__} fields "
        f"(parquet-only: {set(table.column_names) - set(model.model_fields)}, "
        f"model-only: {set(model.model_fields) - set(table.column_names)})"
    )

    rows: list[dict[str, Any]] = table.slice(0, ROWS_PER_SHARD).to_pylist()
    assert rows, f"{config}: shard {shard} is empty"
    for row in rows:
        model.model_validate(row)


@pytest.mark.parametrize("config", CONFIGS)
def test_committed_json_schema_matches_model(config: str, repo_files: list[str]) -> None:
    model, _ = CONFIGS[config]
    schema_file = f"schema/{config}.schema.json"
    assert schema_file in repo_files
    path = hf_hub_download(REPO_ID, schema_file, repo_type="dataset")
    with open(path) as f:
        committed = json.load(f)
    assert committed == model.model_json_schema(), (
        f"{config}: {schema_file} in the dataset repo differs from "
        f"{model.__name__}.model_json_schema() — re-export the schema or update the model"
    )
