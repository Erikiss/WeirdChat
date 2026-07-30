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


def test_segment_stats_switch_indexing():
    path = np.asarray([0, 0, 1, 1, 0, 1])
    s = segment_stats(path, top_state=1)
    assert s["first_top_idx"] == 2
    assert s["switches"] == 3
    assert 0.49 < s["top_frac"] < 0.51
    none = segment_stats(np.zeros(5, dtype=int), top_state=1)
    assert none["first_top_idx"] is None and none["top_frac"] == 0.0
