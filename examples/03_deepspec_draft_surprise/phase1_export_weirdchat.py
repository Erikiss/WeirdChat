"""Phase 1a — export the WeirdChat Qwen transcripts into DeepSpec's JSONL format.

Produces two line-aligned files:

- ``weird_transcripts.jsonl`` — {"id", "conversations"} rows in the format
  DeepSpec's ``JsonLineDataset``/``ConversationCollator`` consume.
- ``weird_meta.jsonl`` — per-line provenance and WeirdChat metrics (pattern id,
  behavior, Elo axes, match rates, judge verdict) for the phase-5 analysis.

The exporter pulls from the Hugging Face dataset via the ``weirdchat`` client,
so it needs the WeirdChat repo's environment (``uv sync``) and HF access.

Usage:
    python phase1_export_weirdchat.py --output-dir data/ \
        [--behavior fabricated-code-execution] [--include-unmatched] [--max-patterns N]
"""

from __future__ import annotations

import argparse
from typing import Any

from surprise_common import SUBJECT_MODEL_SLUG, messages_to_conversations, write_jsonl


def export_rows(
    patterns: list[Any],
    transcripts_of: Any,
    matched_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pure conversion from WeirdChat objects to (data, meta) row lists."""
    data_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for t in transcripts_of(pattern):
            if matched_only and not t.judgment.match:
                continue
            try:
                conversations = messages_to_conversations(t.messages)
            except ValueError as e:
                print(f"skip {t.transcript_id}: {e}")
                continue
            data_rows.append({"id": t.transcript_id, "conversations": conversations})
            meta_rows.append(
                {
                    "transcript_id": t.transcript_id,
                    "prompt_id": t.prompt_id,
                    "pattern_id": pattern.pattern_id,
                    "behavior_id": pattern.behavior_id,
                    "subject_model": pattern.subject_model,
                    "checkpoint": pattern.checkpoint,
                    "method": pattern.method,
                    "judge_match": bool(t.judgment.match),
                    "is_highlight": bool(t.is_highlight),
                    "match_rate": pattern.metrics.match_rate,
                    "openrouter_rate": (
                        pattern.openrouter_replication.rate
                        if pattern.openrouter_replication
                        else None
                    ),
                    "elo_mean": pattern.elo.mean,
                    "elo_unexpectedness": (
                        pattern.elo.unexpectedness.elo if pattern.elo.unexpectedness else None
                    ),
                    "elo_harmfulness": (
                        pattern.elo.harmfulness.elo if pattern.elo.harmfulness else None
                    ),
                    "elo_prompt_naturalness": (
                        pattern.elo.prompt_naturalness.elo
                        if pattern.elo.prompt_naturalness
                        else None
                    ),
                    "n_transcript_messages": len(t.messages),
                }
            )
    return data_rows, meta_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--behavior", default=None, help="restrict to one behavior_id")
    parser.add_argument("--max-patterns", type=int, default=None)
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="also export transcripts the judge did not mark as exhibiting the behavior",
    )
    args = parser.parse_args()

    import weirdchat as wc

    patterns = wc.patterns(behavior_id=args.behavior, subject_model=SUBJECT_MODEL_SLUG)
    if args.max_patterns is not None:
        patterns = patterns[: args.max_patterns]
    print(f"{len(patterns)} qwen patterns selected")

    data_rows, meta_rows = export_rows(
        patterns,
        transcripts_of=wc.transcripts,
        matched_only=not args.include_unmatched,
    )
    assert len(data_rows) == len(meta_rows)

    data_path = f"{args.output_dir}/weird_transcripts.jsonl"
    meta_path = f"{args.output_dir}/weird_meta.jsonl"
    write_jsonl(data_path, data_rows)
    write_jsonl(meta_path, meta_rows)
    checkpoints = {m["checkpoint"] for m in meta_rows}
    print(
        f"wrote {len(data_rows)} transcripts from {len(patterns)} patterns\n"
        f"  data: {data_path}\n  meta: {meta_path}\n"
        f"  dataset checkpoint(s): {sorted(checkpoints)}"
    )


if __name__ == "__main__":
    main()
