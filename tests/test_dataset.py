"""Smoke tests for the WeirdChat dataset client against the live HF repo.

Requires read access to the dataset repo (cached login or ``HF_TOKEN``), like
``test_hf_schema_sync.py``; skips if the repo is unreachable.
"""

import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from weirdchat import WeirdChat


@pytest.fixture(scope="session")
def client() -> WeirdChat:
    wc = WeirdChat()
    try:
        wc.behaviors()
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as e:
        pytest.skip(f"cannot access {wc.repo_id} (missing HF_TOKEN?): {e}")
    return wc


def test_behavior_lookup(client: WeirdChat) -> None:
    behaviors = client.behaviors()
    assert behaviors
    first = behaviors[0]
    assert client.behavior(first.behavior_id) == first
    with pytest.raises(KeyError):
        client.behavior("no-such-behavior")


def test_pattern_index_and_filters(client: WeirdChat) -> None:
    patterns = client.patterns()
    assert patterns
    top = patterns[0]
    assert client.pattern(top.pattern_id) == top
    filtered = client.patterns(behavior_id=top.behavior_id, subject_model=top.subject_model)
    assert top in filtered
    assert all(
        p.behavior_id == top.behavior_id and p.subject_model == top.subject_model
        for p in filtered
    )


def test_per_pattern_shards(client: WeirdChat) -> None:
    # A single-transcript-file pattern keeps the download small.
    pattern = min(client.patterns(), key=lambda p: (len(p.transcript_files), p.n_transcripts))
    prompts = client.prompts(pattern)
    transcripts = client.transcripts(pattern)
    assert len(prompts) == pattern.n_prompts
    assert len(transcripts) == pattern.n_transcripts
    assert all(p.pattern_id == pattern.pattern_id for p in prompts)
    assert all(t.pattern_id == pattern.pattern_id for t in transcripts)
    # Accessing by id resolves through the index to the same rows.
    assert client.prompts(pattern.pattern_id) == prompts
