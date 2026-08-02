"""Phase 7a — temperature sweep of the tipping rate.

Resamples each selected pattern's prompts at several sampling temperatures via
OpenRouter and grades the completions with WeirdChat's reference rubric judge,
producing match counts per (pattern, temperature) for the Arrhenius analysis.

Everything except temperature follows the WeirdChat protocol (no system prompt,
reasoning disabled, max 1024 new tokens). The subject model and the judge both
run through OpenRouter; the judge is the dataset's uniform rubric judge
(`weirdchat.judge.RubricJudge`), so match rates are comparable to the dataset's.

Cost scales as n_temperatures x prompts_per_pattern x samples_per_prompt x
(1 generation + 1 judge) x n_patterns — keep defaults modest and scale up once
the curve shape looks right. Resumable: completed (pattern, temperature) cells
are skipped on restart.

Usage:
    export OPENROUTER_API_KEY=...
    python phase7_sweep.py --output data/temp_sweep.jsonl \
        --behaviors language-switching-english chemtrails-assertion \
        --patterns-per-behavior 3 --prompts-per-pattern 4 --samples-per-prompt 12 \
        --temperatures 0.5 0.7 0.85 1.0 1.15 1.3
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import anyio
from openai import AsyncOpenAI

from surprise_common import (
    SUBJECT_MODEL_SLUG,
    WEIRDCHAT_MAX_NEW_TOKENS,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_done(path: str) -> set[tuple[str, float]]:
    done: set[tuple[str, float]] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["pattern_id"], float(r["temperature"])))
    return done


async def sample_one(client, args, prompt_messages, temperature):
    """One completion; returns (assistant_text, completion_tokens) or None."""
    try:
        completion = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": m.role, "content": m.content} for m in prompt_messages],
            temperature=float(temperature),
            max_tokens=args.max_tokens,
            extra_body=json.loads(args.extra_body),
        )
    except Exception as e:  # noqa: BLE001 — a failed draw just drops out of the sample
        print(f"  gen error @T={temperature}: {type(e).__name__}")
        return None
    text = completion.choices[0].message.content or ""
    n_tok = getattr(getattr(completion, "usage", None), "completion_tokens", None)
    if not text.strip():
        return None
    return text, int(n_tok) if n_tok else max(len(text) // 4, 1)


async def run(args) -> None:
    import weirdchat as wc
    from weirdchat import Message
    from weirdchat.concurrency import run_bounded
    from weirdchat.judge import RubricJudge

    done = load_done(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=("EMPTY" if args.api_key_env == "NONE" else os.environ[args.api_key_env]),
        timeout=180.0,
        max_retries=3,
    )
    # The judge defaults to Gemma on OpenRouter, but can point at a self-hosted
    # OpenAI-compatible endpoint (--judge-base-url / --judge-api-key-env /
    # --judge-model). Self-hosting removes the OpenRouter rate limit *and* the
    # per-call judge bill, which is the real money sink at high sample counts.
    judge_client = AsyncOpenAI(
        base_url=args.judge_base_url,
        api_key=("EMPTY" if args.judge_api_key_env == "NONE" else os.environ[args.judge_api_key_env]),
        timeout=180.0,
        max_retries=3,
    )
    judge_kwargs: dict[str, Any] = {"client": judge_client}
    if args.judge_model:
        judge_kwargs["model"] = args.judge_model
    if args.judge_reasoning_effort:
        judge_kwargs["reasoning_effort"] = args.judge_reasoning_effort

    out = open(args.output, "a", encoding="utf-8")
    for behavior_id in args.behaviors:
        behavior = wc.behavior(behavior_id)
        judge = RubricJudge.for_behavior(behavior, **judge_kwargs)
        patterns = wc.patterns(behavior_id=behavior_id, subject_model=args.model)
        # Prefer the patterns that actually reproduce on OpenRouter.
        patterns.sort(
            key=lambda p: (p.openrouter_replication.rate if p.openrouter_replication else 0) or 0,
            reverse=True,
        )
        patterns = patterns[: args.patterns_per_behavior]
        print(f"{behavior_id}: {len(patterns)} patterns")

        for pattern in patterns:
            prompts = wc.prompts(pattern)[: args.prompts_per_pattern]
            for temperature in args.temperatures:
                if (pattern.pattern_id, float(temperature)) in done:
                    continue
                # Fan out generations across prompts x samples.
                gen_tasks = [
                    sample_one(client, args, p.messages, temperature)
                    for p in prompts
                    for _ in range(args.samples_per_prompt)
                ]
                gens = await run_bounded(gen_tasks, args.concurrency, "gen")
                prompt_of = [p for p in prompts for _ in range(args.samples_per_prompt)]

                rollouts, lengths = [], []
                for p, g in zip(prompt_of, gens):
                    if g is None:
                        continue
                    text, n_tok = g
                    rollouts.append(list(p.messages) + [Message(role="assistant", content=text)])
                    lengths.append(n_tok)
                if not rollouts:
                    print(f"  {pattern.pattern_id[:40]} T={temperature}: no completions")
                    continue

                verdicts = await run_bounded(
                    [judge.judge(r) for r in rollouts], args.judge_concurrency, "judge"
                )
                matched = sum(1 for v in verdicts if getattr(v, "match", False))
                mean_len = sum(lengths) / len(lengths)
                out.write(
                    json.dumps(
                        {
                            "pattern_id": pattern.pattern_id,
                            "behavior_id": behavior_id,
                            "subject_model": args.model,
                            "temperature": float(temperature),
                            "n_samples": len(rollouts),
                            "n_matched": matched,
                            "mean_completion_tokens": mean_len,
                        }
                    )
                    + "\n"
                )
                out.flush()
                print(
                    f"  {pattern.pattern_id[:40]} T={temperature}: "
                    f"{matched}/{len(rollouts)} matched (len~{mean_len:.0f})"
                )
    out.close()
    print(f"sweep written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/temp_sweep.jsonl")
    parser.add_argument("--behaviors", nargs="+", required=True)
    parser.add_argument("--model", default=SUBJECT_MODEL_SLUG)
    parser.add_argument("--patterns-per-behavior", type=int, default=3)
    parser.add_argument("--prompts-per-pattern", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=12)
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.5, 0.7, 0.85, 1.0, 1.15, 1.3]
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=WEIRDCHAT_MAX_NEW_TOKENS)
    parser.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--extra-body", default='{"reasoning": {"enabled": false}}')
    # Judge endpoint — defaults to Gemma on OpenRouter; override to self-host.
    parser.add_argument("--judge-base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--judge-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--judge-model", default=None)  # None -> RubricJudge default (google/gemma-4-31b-it)
    parser.add_argument("--judge-reasoning-effort", default=None)  # override/skip for non-reasoning local judges
    args = parser.parse_args()
    anyio.run(run, args)


if __name__ == "__main__":
    main()
