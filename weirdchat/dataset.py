"""Typed client for the WeirdChat dataset on the Hugging Face Hub.

Usage::

    import weirdchat as wc

    patterns = wc.patterns(subject_model="thinkingmachines/inkling")
    transcripts = wc.transcripts(patterns[0])
    behavior = wc.behavior(patterns[0].behavior_id)

The module-level functions share one default :class:`WeirdChat` client; instantiate your own to
pin a dataset revision. Parquet shards are downloaded lazily on first use (and cached on disk by
``huggingface_hub``, so nothing is re-downloaded across sessions), then parsed into the pydantic
row models from :mod:`weirdchat.types`.
"""

# huggingface_hub's signatures are partially untyped, which strict pyright reports at the
# import site; the values we consume from it are fully annotated below.
# pyright: reportUnknownVariableType=false

from typing import Any

from huggingface_hub import hf_hub_download

from .types import Behavior, EloAxis, Pattern, PairwiseJudgment, Prompt, SearchCompute, Transcript

REPO_ID = "Transluce/WeirdChat"

PAIRWISE_AXES: tuple[EloAxis, ...] = ("prompt_naturalness", "unexpectedness", "harmfulness")


class WeirdChat:
    """Lazy, cached reader for one revision of the WeirdChat dataset repo.

    ``data/patterns.parquet`` is the index: each :class:`Pattern` row names the parquet shards
    holding its prompts and transcripts, so per-pattern access downloads only those shards rather
    than the whole dataset.
    """

    def __init__(self, repo_id: str = REPO_ID, revision: str | None = None) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self._shards: dict[str, list[dict[str, Any]]] = {}

    def _rows(self, path: str) -> list[dict[str, Any]]:
        if path not in self._shards:
            import pyarrow as pa
            import pyarrow.parquet as pq

            local = hf_hub_download(
                self.repo_id, path, repo_type="dataset", revision=self.revision
            )
            table: pa.Table = pq.read_table(local)  # pyright: ignore[reportUnknownMemberType]
            self._shards[path] = table.to_pylist()
        return self._shards[path]

    # -- rubrics ---------------------------------------------------------------------------

    def behaviors(self) -> list[Behavior]:
        """All behaviors with their judge rubrics, sorted by ``behavior_id``."""
        rows = [Behavior.model_validate(r) for r in self._rows("data/rubrics.parquet")]
        return sorted(rows, key=lambda b: b.behavior_id)

    def behavior(self, behavior_id: str) -> Behavior:
        for b in self.behaviors():
            if b.behavior_id == behavior_id:
                return b
        known = ", ".join(b.behavior_id for b in self.behaviors())
        raise KeyError(f"no behavior {behavior_id!r}; known behaviors: {known}")

    # -- patterns --------------------------------------------------------------------------

    def patterns(
        self,
        behavior_id: str | None = None,
        subject_model: str | None = None,
        method: str | None = None,
    ) -> list[Pattern]:
        """Patterns, optionally filtered, sorted by mean interestingness Elo (best first)."""
        rows = [Pattern.model_validate(r) for r in self._rows("data/patterns.parquet")]
        rows = [
            p
            for p in rows
            if (behavior_id is None or p.behavior_id == behavior_id)
            and (subject_model is None or p.subject_model == subject_model)
            and (method is None or p.method == method)
        ]
        return sorted(rows, key=lambda p: p.elo.mean or float("-inf"), reverse=True)

    def pattern(self, pattern_id: str) -> Pattern:
        for p in self.patterns():
            if p.pattern_id == pattern_id:
                return p
        raise KeyError(f"no pattern {pattern_id!r} (see patterns() for the index)")

    def _resolve(self, pattern: Pattern | str) -> Pattern:
        return pattern if isinstance(pattern, Pattern) else self.pattern(pattern)

    # -- per-pattern data ------------------------------------------------------------------

    def prompts(self, pattern: Pattern | str) -> list[Prompt]:
        """The measured candidate prompts of one pattern (accepts a ``Pattern`` or pattern id)."""
        pattern = self._resolve(pattern)
        rows = self._rows(pattern.prompt_file)
        return [
            Prompt.model_validate(r) for r in rows if r["pattern_id"] == pattern.pattern_id
        ]

    def transcripts(self, pattern: Pattern | str) -> list[Transcript]:
        """All judged rollouts of one pattern (accepts a ``Pattern`` or pattern id)."""
        pattern = self._resolve(pattern)
        return [
            Transcript.model_validate(r)
            for path in pattern.transcript_files
            for r in self._rows(path)
            if r["pattern_id"] == pattern.pattern_id
        ]

    def highlight_transcript(self, pattern: Pattern | str) -> Transcript | None:
        """The pattern's hand-picked showcase transcript, if it has one."""
        pattern = self._resolve(pattern)
        if pattern.highlight_transcript_id is None:
            return None
        for t in self.transcripts(pattern):
            if t.transcript_id == pattern.highlight_transcript_id:
                return t
        return None

    # -- dataset-wide extras ---------------------------------------------------------------

    def pairwise_judgments(self, axis: EloAxis | None = None) -> list[PairwiseJudgment]:
        """The interestingness-Elo tournament comparisons, for one axis or all three."""
        axes = PAIRWISE_AXES if axis is None else (axis,)
        return [
            PairwiseJudgment.model_validate(r)
            for a in axes
            for r in self._rows(f"data/pairwise/{a}.parquet")
        ]

    def search_compute(self) -> list[SearchCompute]:
        """Approximate subject-model sampling spent per (behavior, subject model, method)."""
        return [SearchCompute.model_validate(r) for r in self._rows("data/search_compute.parquet")]


_default_client: WeirdChat | None = None


def default_client() -> WeirdChat:
    """The shared client behind the module-level functions (created on first use)."""
    global _default_client
    if _default_client is None:
        _default_client = WeirdChat()
    return _default_client


def behaviors() -> list[Behavior]:
    return default_client().behaviors()


def behavior(behavior_id: str) -> Behavior:
    return default_client().behavior(behavior_id)


def patterns(
    behavior_id: str | None = None,
    subject_model: str | None = None,
    method: str | None = None,
) -> list[Pattern]:
    return default_client().patterns(behavior_id, subject_model, method)


def pattern(pattern_id: str) -> Pattern:
    return default_client().pattern(pattern_id)


def prompts(pattern: Pattern | str) -> list[Prompt]:
    return default_client().prompts(pattern)


def transcripts(pattern: Pattern | str) -> list[Transcript]:
    return default_client().transcripts(pattern)


def highlight_transcript(pattern: Pattern | str) -> Transcript | None:
    return default_client().highlight_transcript(pattern)


def pairwise_judgments(axis: EloAxis | None = None) -> list[PairwiseJudgment]:
    return default_client().pairwise_judgments(axis)


def search_compute() -> list[SearchCompute]:
    return default_client().search_compute()
