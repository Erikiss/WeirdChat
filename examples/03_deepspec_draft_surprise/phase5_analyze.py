"""Phase 5 — aggregate draft-surprise scores into an interpretation report.

Consumes the phase-4 outputs for the WeirdChat set and the baseline held-out
set (the null distribution) plus the phase-1 metadata sidecar, and writes a
markdown report answering:

- Which Qwen patterns are most atypical relative to the model's normal
  behavior (z-scored against the baseline null)?
- Does draft surprise track WeirdChat's judge-based metrics (unexpectedness /
  harmfulness Elo, match rates)? Spearman correlations across patterns.
- Where inside a transcript does the weirdness start? Token traces of the top
  patterns with the high-excess tokens marked.
- How does multi-token lookahead degrade on weird vs normal text (block-offset
  NLL curves)?

Pure-python (no torch needed), so it can run anywhere the score files are.

Usage:
    python phase5_analyze.py --weird-scores data/weird_scores.jsonl \
        --weird-meta data/weird_meta.jsonl \
        --baseline-scores data/baseline_scores.jsonl \
        --output report.md
"""

from __future__ import annotations

import argparse
from typing import Any

from surprise_common import read_jsonl, spearman

SIGMA_MARK = 3.0
SIGMA_STRONG = 5.0


def null_stats(baseline_scores: list[dict[str, Any]], key: str) -> tuple[float, float]:
    vals = [
        s["aggregates"][key]
        for s in baseline_scores
        if s.get("aggregates", {}).get(key) is not None
    ]
    assert len(vals) >= 2, f"baseline too small for null distribution of {key}"
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, var**0.5


def build_pattern_table(
    weird_scores: list[dict[str, Any]],
    meta_by_index: dict[int, dict[str, Any]],
    null_mean: float,
    null_std: float,
) -> list[dict[str, Any]]:
    """Aggregate transcript-level mean excess into per-pattern rows."""
    groups: dict[str, dict[str, Any]] = {}
    for s in weird_scores:
        meta = meta_by_index.get(s["index"])
        if meta is None or s["aggregates"].get("mean_excess") is None:
            continue
        g = groups.setdefault(
            meta["pattern_id"],
            {"meta": meta, "excesses": [], "best": None},
        )
        g["excesses"].append(s["aggregates"]["mean_excess"])
        if g["best"] is None or s["aggregates"]["mean_excess"] > g["best"]["aggregates"]["mean_excess"]:
            g["best"] = s
    rows = []
    for pattern_id, g in groups.items():
        n = len(g["excesses"])
        mean_excess = sum(g["excesses"]) / n
        rows.append(
            {
                "pattern_id": pattern_id,
                "behavior_id": g["meta"]["behavior_id"],
                "n_transcripts": n,
                "mean_excess": mean_excess,
                "z_vs_null": (mean_excess - null_mean) / null_std if null_std > 0 else None,
                "elo_mean": g["meta"].get("elo_mean"),
                "elo_unexpectedness": g["meta"].get("elo_unexpectedness"),
                "elo_harmfulness": g["meta"].get("elo_harmfulness"),
                "match_rate": g["meta"].get("match_rate"),
                "openrouter_rate": g["meta"].get("openrouter_rate"),
                "best_transcript": g["best"],
            }
        )
    rows.sort(key=lambda r: r["mean_excess"], reverse=True)
    return rows


def correlation_block(rows: list[dict[str, Any]]) -> list[tuple[str, float | None, int]]:
    out = []
    for metric in (
        "elo_unexpectedness",
        "elo_harmfulness",
        "elo_mean",
        "match_rate",
        "openrouter_rate",
    ):
        paired = [
            (r["mean_excess"], r[metric]) for r in rows if r.get(metric) is not None
        ]
        rho = spearman([p[0] for p in paired], [p[1] for p in paired]) if paired else None
        out.append((metric, rho, len(paired)))
    return out


def render_token_trace(
    score_row: dict[str, Any], null_mean: float, null_std: float, max_tokens: int = 400
) -> str:
    """Assistant tokens with high-excess ones marked: «+3σ», ⟪+5σ⟫."""
    pieces = []
    for tok, nd, nt in zip(
        score_row["token_strs"][:max_tokens],
        score_row["nll_draft"][:max_tokens],
        score_row["nll_target"][:max_tokens],
    ):
        excess = nd - nt
        z = (excess - null_mean) / null_std if null_std > 0 else 0.0
        text = tok.replace("\n", "\\n")
        if z >= SIGMA_STRONG:
            pieces.append(f"⟪{text}⟫")
        elif z >= SIGMA_MARK:
            pieces.append(f"«{text}»")
        else:
            pieces.append(text)
    return "".join(pieces)


def offset_curve(scores: list[dict[str, Any]]) -> list[float | None]:
    """Mean of per-sample block-offset NLL curves."""
    if not scores:
        return []
    width = max(len(s.get("offset_mean_nll_draft", [])) for s in scores)
    sums, counts = [0.0] * width, [0] * width
    for s in scores:
        for j, v in enumerate(s.get("offset_mean_nll_draft", [])):
            if v is not None:
                sums[j] += v
                counts[j] += 1
    return [round(sums[j] / counts[j], 4) if counts[j] else None for j in range(width)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weird-scores", required=True)
    parser.add_argument("--weird-meta", required=True)
    parser.add_argument("--baseline-scores", required=True)
    parser.add_argument("--output", default="report.md")
    parser.add_argument("--top-traces", type=int, default=10)
    args = parser.parse_args()

    weird_scores = read_jsonl(args.weird_scores)
    baseline_scores = read_jsonl(args.baseline_scores)
    meta_rows = read_jsonl(args.weird_meta)
    meta_by_index = dict(enumerate(meta_rows))

    # Per-token null for trace marking; per-sample null for pattern z-scores.
    tok_excess = [
        nd - nt
        for s in baseline_scores
        for nd, nt in zip(s["nll_draft"], s["nll_target"])
    ]
    assert tok_excess, "baseline scores contain no tokens"
    tok_mean = sum(tok_excess) / len(tok_excess)
    tok_std = (
        sum((v - tok_mean) ** 2 for v in tok_excess) / max(len(tok_excess) - 1, 1)
    ) ** 0.5
    null_mean, null_std = null_stats(baseline_scores, "mean_excess")

    rows = build_pattern_table(weird_scores, meta_by_index, null_mean, null_std)
    correlations = correlation_block(rows)

    lines = [
        "# Draft-surprise report: WeirdChat qwen/qwen3.6-35b-a3b",
        "",
        f"- weird transcripts scored: {len(weird_scores)} across {len(rows)} patterns",
        f"- baseline (null) samples: {len(baseline_scores)}",
        f"- null mean_excess: {null_mean:.4f} ± {null_std:.4f} (per transcript); "
        f"per-token excess: {tok_mean:.4f} ± {tok_std:.4f}",
        "",
        "## Patterns ranked by draft surprise (excess NLL vs baseline null)",
        "",
        "| # | pattern | behavior | n | mean_excess | z | Elo unexp. | Elo harm. | match_rate |",
        "|---|---------|----------|---|-------------|---|-----------|-----------|------------|",
    ]
    for i, r in enumerate(rows[:30], 1):
        z = f"{r['z_vs_null']:.1f}" if r["z_vs_null"] is not None else "–"
        unexp = f"{r['elo_unexpectedness']:.0f}" if r["elo_unexpectedness"] is not None else "–"
        harm = f"{r['elo_harmfulness']:.0f}" if r["elo_harmfulness"] is not None else "–"
        mr = f"{r['match_rate']:.0%}" if r["match_rate"] is not None else "–"
        lines.append(
            f"| {i} | `{r['pattern_id'][:60]}` | {r['behavior_id']} | {r['n_transcripts']} "
            f"| {r['mean_excess']:.3f} | {z} | {unexp} | {harm} | {mr} |"
        )

    lines += ["", "## Does draft surprise track WeirdChat's judged metrics?", ""]
    lines.append("| metric | Spearman ρ (patterns) | n |")
    lines.append("|--------|----------------------|---|")
    for metric, rho, n in correlations:
        lines.append(f"| {metric} | {f'{rho:.3f}' if rho is not None else '–'} | {n} |")

    lines += [
        "",
        "## Lookahead decay (mean NLL by block offset)",
        "",
        f"- weird:    {offset_curve(weird_scores)}",
        f"- baseline: {offset_curve(baseline_scores)}",
        "",
        "## Token traces of the most surprising patterns",
        "",
        f"Tokens marked «…» exceed +{SIGMA_MARK:.0f}σ and ⟪…⟫ +{SIGMA_STRONG:.0f}σ "
        "per-token excess vs the baseline null.",
        "",
    ]
    for r in rows[: args.top_traces]:
        best = r["best_transcript"]
        lines += [
            f"### {r['behavior_id']} — `{r['pattern_id'][:80]}`",
            "",
            f"transcript `{best.get('id')}`, mean_excess {best['aggregates']['mean_excess']:.3f}, "
            f"peak at scored-token #{int(best['aggregates']['peak_excess_token_idx'])}:",
            "",
            "```",
            render_token_trace(best, tok_mean, tok_std),
            "```",
            "",
        ]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report written to {args.output}")


if __name__ == "__main__":
    main()
