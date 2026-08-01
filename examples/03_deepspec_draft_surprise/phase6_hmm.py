"""Phase 6 (speculative) — latent-state "tipping" analysis via a Gaussian HMM.

Models each transcript's per-token excess sequence (draft NLL minus aligned
target NLL, from phase 4) as emissions of a K-state hidden Markov model:
state = the model's latent mode ("normal" vs "tipped"), transitions = the
tipping dynamics, emissions = the observed surprise. Fitted per behavior with
Baum-Welch (log-space, deterministic quantile init — no RNG), decoded with
Viterbi. The baseline held-out set is fitted too, as the control: a healthy
control shows no separated high-surprise state.

What this adds over phase 5's single peak index:
- a *probabilistic* switch point per transcript (first Viterbi entry into the
  high-surprise state) instead of an argmax heuristic,
- dwell statistics: does the model *stay* tipped (absorbing state, e.g.
  language switching) or flicker,
- per-behavior transition matrices — the "Markov chain of weirdness".

CPU-only, minutes on the full 7k-transcript file. Requires numpy.

Usage (Colab, Drive mounted):
    python phase6_hmm.py \
        --scores $DATA_DIR/weird_scores.jsonl --meta $DATA_DIR/weird_meta.jsonl \
        --baseline-scores $DATA_DIR/baseline_scores.jsonl \
        --output $DATA_DIR/hmm_report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def excess_sequence(row: dict[str, Any]):
    import numpy as np

    return np.asarray(row["nll_draft"], dtype=float) - np.asarray(
        row["nll_target"], dtype=float
    )


def _logsumexp(a, axis):
    import numpy as np

    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def fit_gaussian_hmm(sequences, k: int = 2, iters: int = 100, tol: float = 1e-5):
    """Baum-Welch for a 1-D Gaussian HMM, shared over all sequences.

    Deterministic quantile initialization; returns dict with pi, A, means,
    vars, loglik. States are sorted by mean ascending (state 0 = calmest).

    Vectorized across sequences: padded to the group's max length with a
    validity mask, so the (inherently sequential) time recurrences run once
    over T_max for all N sequences together instead of per sequence — the
    per-timestep Python overhead was the bottleneck at millions of tokens.
    """
    import numpy as np

    lengths = np.asarray([len(x) for x in sequences])
    n_seq, t_max = len(sequences), int(lengths.max())
    X = np.zeros((n_seq, t_max))
    for i, x in enumerate(sequences):
        X[i, : len(x)] = x
    valid = np.arange(t_max)[None, :] < lengths[:, None]  # (N, T)

    x_all = np.concatenate(sequences)
    means = np.quantile(x_all, [(i + 0.5) / k for i in range(k)]).astype(float)
    variances = np.full(k, max(float(x_all.var()), 1e-3))
    pi = np.full(k, 1.0 / k)
    A = np.full((k, k), 0.05 / max(k - 1, 1))
    np.fill_diagonal(A, 0.95)

    prev_ll = -np.inf
    for _ in range(iters):
        log_pi, log_A = np.log(pi), np.log(A)
        logB = -0.5 * (
            np.log(2 * np.pi * variances)[None, None, :]
            + (X[:, :, None] - means[None, None, :]) ** 2 / variances[None, None, :]
        )
        logB = np.where(valid[:, :, None], logB, 0.0)

        # Forward, carrying the last valid alpha through the padding so the
        # final column always holds each sequence's terminal alpha.
        la = np.empty((n_seq, t_max, k))
        la[:, 0] = log_pi[None, :] + logB[:, 0]
        for t in range(1, t_max):
            prev = la[:, t - 1]
            new = logB[:, t] + _logsumexp(prev[:, :, None] + log_A[None, :, :], axis=1)
            la[:, t] = np.where(valid[:, t, None], new, prev)
        ll_seq = _logsumexp(la[:, -1], axis=1)  # (N,)

        # Backward: beta is 0 at each sequence's last valid step and in padding.
        lb = np.zeros((n_seq, t_max, k))
        for t in range(t_max - 2, -1, -1):
            nxt = logB[:, t + 1] + lb[:, t + 1]
            val = _logsumexp(log_A[None, :, :] + nxt[:, None, :], axis=2)
            lb[:, t] = np.where(valid[:, t + 1, None], val, 0.0)

        gamma = np.exp(la + lb - ll_seq[:, None, None]) * valid[:, :, None]
        pair_valid = valid[:, 1:]  # transition t -> t+1 exists
        xi = (
            np.exp(
                la[:, :-1, :, None]
                + log_A[None, None, :, :]
                + (logB[:, 1:] + lb[:, 1:])[:, :, None, :]
                - ll_seq[:, None, None, None]
            )
            * pair_valid[:, :, None, None]
        )

        pi = gamma[:, 0].sum(axis=0) / n_seq
        A = xi.sum(axis=(0, 1))
        A /= np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        w_sum = gamma.sum(axis=(0, 1))
        means = (gamma * X[:, :, None]).sum(axis=(0, 1)) / np.maximum(w_sum, 1e-12)
        variances = np.maximum(
            (gamma * X[:, :, None] ** 2).sum(axis=(0, 1)) / np.maximum(w_sum, 1e-12)
            - means**2,
            1e-4,
        )
        ll_total = float(ll_seq.sum())
        if abs(ll_total - prev_ll) < tol * max(abs(prev_ll), 1.0):
            prev_ll = ll_total
            break
        prev_ll = ll_total

    order = np.argsort(means)
    return {
        "pi": pi[order],
        "A": A[np.ix_(order, order)],
        "means": means[order],
        "vars": variances[order],
        "loglik": float(prev_ll),
    }


def viterbi(x, model):
    """Most likely state path for one sequence under the fitted model."""
    import numpy as np

    means, variances = model["means"], model["vars"]
    log_pi, log_A = np.log(model["pi"] + 1e-12), np.log(model["A"] + 1e-12)
    logB = -0.5 * (
        np.log(2 * np.pi * variances)[None, :]
        + (x[:, None] - means[None, :]) ** 2 / variances[None, :]
    )
    T, k = logB.shape
    delta = np.empty((T, k))
    back = np.zeros((T, k), dtype=int)
    delta[0] = log_pi + logB[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        back[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + logB[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(delta[-1].argmax())
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1][path[t + 1]]
    return path


def segment_stats(path, top_state: int) -> dict[str, Any]:
    """First entry into the top state, and how much of the sequence it holds."""
    import numpy as np

    in_top = path == top_state
    first = int(np.argmax(in_top)) if in_top.any() else None
    return {
        "switches": int(np.count_nonzero(np.diff(path))),
        "first_top_idx": first,
        "top_frac": float(in_top.mean()),
    }


def main() -> None:
    import numpy as np
    import statistics as st

    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="full weird_scores.jsonl (per-token)")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--baseline-scores", required=True)
    parser.add_argument("--output", default="hmm_report.md")
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--min-transcripts", type=int, default=50)
    parser.add_argument("--min-tokens", type=int, default=20)
    args = parser.parse_args()

    meta = {i: m for i, m in enumerate(iter_jsonl(args.meta))}
    groups: dict[str, list[Any]] = defaultdict(list)
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(args.scores):
        if len(row.get("nll_draft", [])) < args.min_tokens:
            continue
        behavior = meta[row["index"]]["behavior_id"]
        groups[behavior].append(excess_sequence(row))
        rows_by_group[behavior].append(row)
    baseline = [
        excess_sequence(r)
        for r in iter_jsonl(args.baseline_scores)
        if len(r.get("nll_draft", [])) >= args.min_tokens
    ]

    lines = [
        "# Latent-state (HMM) tipping analysis",
        "",
        f"{args.states}-state Gaussian HMM over per-token excess; states sorted by mean",
        "(state 0 = normal mode, top state = tipped mode). `dwell` is the expected",
        "run length 1/(1-a_ii); `switched` is the share of transcripts that ever",
        "enter the tipped state; `first@` the median token index of first entry;",
        "`|hmm-peak|` the median distance between the HMM switch point and phase 5's",
        "argmax-excess token.",
        "",
        "| group | n_seq | mean_normal | mean_tipped | stay_norm | stay_tip | dwell_tip | tip% tokens | switched | first@ | \\|hmm-peak\\| |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fit_and_report(name: str, seqs, rows) -> None:
        model = fit_gaussian_hmm(seqs, k=args.states)
        top = args.states - 1
        stats = [segment_stats(viterbi(x, model), top) for x in seqs]
        switched = [s for s in stats if s["first_top_idx"] is not None]
        stay_n = float(model["A"][0, 0])
        stay_t = float(model["A"][top, top])
        dwell_t = 1.0 / max(1.0 - stay_t, 1e-9)
        tip_frac = float(np.mean([s["top_frac"] for s in stats]))
        first_at = (
            f"{st.median(s['first_top_idx'] for s in switched):.0f}" if switched else "–"
        )
        peak_dist = "–"
        if rows is not None and switched:
            dists = [
                abs(s["first_top_idx"] - r["aggregates"]["peak_excess_token_idx"])
                for s, r in zip(stats, rows)
                if s["first_top_idx"] is not None
            ]
            peak_dist = f"{st.median(dists):.0f}"
        lines.append(
            f"| {name} | {len(seqs)} | {model['means'][0]:.2f} | {model['means'][top]:.2f} "
            f"| {stay_n:.3f} | {stay_t:.3f} | {dwell_t:.0f} | {tip_frac:.0%} "
            f"| {len(switched) / len(stats):.0%} | {first_at} | {peak_dist} |"
        )

    if baseline:
        fit_and_report("BASELINE (control)", baseline, None)
    for behavior in sorted(groups, key=lambda b: -len(groups[b])):
        if len(groups[behavior]) < args.min_transcripts:
            continue
        fit_and_report(behavior, groups[behavior], rows_by_group[behavior])

    skipped = [b for b in groups if len(groups[b]) < args.min_transcripts]
    if skipped:
        lines += ["", f"skipped (fewer than {args.min_transcripts} transcripts): {', '.join(sorted(skipped))}"]
    lines += [
        "",
        "Reading guide: a real tipping behavior shows a well-separated tipped mean,",
        "a sticky tipped state (dwell >> 1) and a high switched-share, while the",
        "BASELINE control should show little state separation and near-zero tipped",
        "occupancy. If the control also 'tips', the states are modeling draft",
        "weakness rather than behavior.",
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report written to {args.output}")


if __name__ == "__main__":
    main()
