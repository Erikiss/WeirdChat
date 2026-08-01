"""Parameter-recovery tests for the phase-6 Gaussian HMM (needs numpy)."""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase6_hmm import fit_gaussian_hmm, segment_stats, viterbi  # noqa: E402


def _sample_hmm(rng, n_seq=40, length=200, means=(0.0, 6.0), stay=0.97):
    sequences, paths = [], []
    for _ in range(n_seq):
        state = 0
        xs, ss = [], []
        for _ in range(length):
            if rng.random() > stay:
                state = 1 - state
            ss.append(state)
            xs.append(rng.normal(means[state], 1.0))
        sequences.append(np.asarray(xs))
        paths.append(np.asarray(ss))
    return sequences, paths


def test_hmm_recovers_parameters_and_segmentation():
    rng = np.random.default_rng(42)
    sequences, true_paths = _sample_hmm(rng)
    model = fit_gaussian_hmm(sequences, k=2)

    # Means recovered (states sorted ascending by construction).
    assert abs(model["means"][0] - 0.0) < 0.5
    assert abs(model["means"][1] - 6.0) < 0.5
    # Sticky transitions recovered.
    assert model["A"][0, 0] > 0.9 and model["A"][1, 1] > 0.9

    # Viterbi segmentation matches the true hidden path almost everywhere.
    correct = total = 0
    for x, true in zip(sequences, true_paths):
        path = viterbi(x, model)
        correct += int((path == true).sum())
        total += len(true)
    assert correct / total > 0.95


def test_hmm_control_without_separation_stays_calm():
    rng = np.random.default_rng(7)
    # Single-mode data: the top state should capture few tokens even with k=2.
    sequences = [np.asarray(rng.normal(1.0, 1.0, size=300)) for _ in range(20)]
    model = fit_gaussian_hmm(sequences, k=2)
    top_frac = float(
        np.mean([segment_stats(viterbi(x, model), 1)["top_frac"] for x in sequences])
    )
    # Means nearly coincide and the separation is far below a real tipped mode.
    assert model["means"][1] - model["means"][0] < 3.0
    assert top_frac < 0.9  # no dominant "tipped" mode takeover


def _reference_fit(sequences, k=2, iters=100, tol=1e-5):
    """The original per-sequence Baum-Welch (pre-vectorization), kept as oracle."""
    from phase6_hmm import _logsumexp

    x_all = np.concatenate(sequences)
    means = np.quantile(x_all, [(i + 0.5) / k for i in range(k)]).astype(float)
    variances = np.full(k, max(float(x_all.var()), 1e-3))
    pi = np.full(k, 1.0 / k)
    A = np.full((k, k), 0.05 / max(k - 1, 1))
    np.fill_diagonal(A, 0.95)
    prev_ll = -np.inf
    for _ in range(iters):
        log_pi, log_A = np.log(pi), np.log(A)
        ll_total = 0.0
        gamma0 = np.zeros(k); xi_sum = np.zeros((k, k))
        w = np.zeros(k); wx = np.zeros(k); wx2 = np.zeros(k)
        for x in sequences:
            T = len(x)
            logB = -0.5 * (np.log(2 * np.pi * variances)[None, :]
                           + (x[:, None] - means[None, :]) ** 2 / variances[None, :])
            la = np.empty((T, k)); la[0] = log_pi + logB[0]
            for t in range(1, T):
                la[t] = logB[t] + _logsumexp(la[t - 1][:, None] + log_A, axis=0)
            lb = np.zeros((T, k))
            for t in range(T - 2, -1, -1):
                lb[t] = _logsumexp(log_A + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
            m = la[-1].max()
            ll = float(m + np.log(np.exp(la[-1] - m).sum()))
            ll_total += ll
            gamma = np.exp(la + lb - ll)
            gamma0 += gamma[0]
            for t in range(T - 1):
                xi_sum += np.exp(la[t][:, None] + log_A + (logB[t + 1] + lb[t + 1])[None, :] - ll)
            w += gamma.sum(axis=0)
            wx += (gamma * x[:, None]).sum(axis=0)
            wx2 += (gamma * (x[:, None] ** 2)).sum(axis=0)
        pi = gamma0 / len(sequences)
        A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
        means = wx / np.maximum(w, 1e-12)
        variances = np.maximum(wx2 / np.maximum(w, 1e-12) - means**2, 1e-4)
        if abs(ll_total - prev_ll) < tol * max(abs(prev_ll), 1.0):
            prev_ll = ll_total
            break
        prev_ll = ll_total
    order = np.argsort(means)
    return {"pi": pi[order], "A": A[np.ix_(order, order)],
            "means": means[order], "vars": variances[order], "loglik": float(prev_ll)}


def test_vectorized_fit_matches_reference_on_mixed_lengths():
    rng = np.random.default_rng(3)
    # Variable-length sequences exercise the padding/mask logic.
    sequences = []
    for length in (37, 80, 123, 200, 61, 145):
        state, xs = 0, []
        for _ in range(length):
            if rng.random() > 0.95:
                state = 1 - state
            xs.append(rng.normal((0.0, 5.0)[state], 1.0))
        sequences.append(np.asarray(xs))
    fast = fit_gaussian_hmm(sequences, k=2)
    ref = _reference_fit(sequences, k=2)
    assert np.allclose(fast["means"], ref["means"], atol=0.05)
    assert np.allclose(fast["A"], ref["A"], atol=0.02)
    assert abs(fast["loglik"] - ref["loglik"]) / abs(ref["loglik"]) < 0.01


def test_segment_stats_switch_indexing():
    path = np.asarray([0, 0, 1, 1, 0, 1])
    s = segment_stats(path, top_state=1)
    assert s["first_top_idx"] == 2
    assert s["switches"] == 3
    assert 0.49 < s["top_frac"] < 0.51
    none = segment_stats(np.zeros(5, dtype=int), top_state=1)
    assert none["first_top_idx"] is None and none["top_frac"] == 0.0
