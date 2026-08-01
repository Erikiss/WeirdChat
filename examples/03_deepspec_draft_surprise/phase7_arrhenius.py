"""Phase 7b — Arrhenius / super-Arrhenius diagnosis of the tipping rate.

Given a temperature sweep (phase7_sweep.py output: match counts per behavior
at several sampling temperatures), asks whether the behavior's per-token
tipping hazard follows a simple Arrhenius law or is *super-Arrhenius*
(fragile, Vogel–Fulcher–Tammann-like) with a finite critical temperature.

Physics mapping (the softmax temperature IS the thermodynamic T):
- observable per (group, T): match probability p = matched / samples over a
  response of ~L tokens. The microscopic rate is the per-token nucleation
  hazard h = -ln(1 - p) / L (a survival model: each token may tip with hazard
  h, the response tips if any token does). Arrhenius is a law about h, not p,
  so the fits use h; the raw-p curvature is reported too for transparency.
- Arrhenius:      ln h = a - E * (1/T)            -> straight on an (1/T, ln h) plot
- super-Arrhenius: ln h = ln A - B / (T - T0), T0>0 -> concave; hazard -> 0 at T0
  (the finite critical temperature). Effective activation energy
  E_eff(T) = -d ln h / d(1/T) = B T^2 / (T - T0)^2 grows as T falls (fragile).

Verdict per group from three agreeing signals: sign of the quadratic curvature
c2 (bootstrap CI), AIC preference of VFT over Arrhenius, and a positive T0.

Pure numpy, CPU, seconds. Jeffreys smoothing (m+0.5)/(n+1) keeps zero-count
low-T points finite (they carry the strongest super-Arrhenius signal); such
points are flagged as detection-limited.

Usage:
    python phase7_arrhenius.py --sweep temp_sweep.jsonl --output arrhenius_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any


def jeffreys_rate(matched: int, samples: int) -> float:
    """Binomial point estimate that is never exactly 0 or 1."""
    return (matched + 0.5) / (samples + 1.0)


def per_token_hazard(p: float, mean_len: float) -> float:
    """Survival-model per-token hazard from a per-response match probability."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    length = max(float(mean_len), 1.0)
    return -math.log1p(-p) / length


def _polyfit(x, y, deg: int, w=None):
    import numpy as np

    return np.polyfit(np.asarray(x, float), np.asarray(y, float), deg, w=w)


def _sse(x, y, coeffs):
    import numpy as np

    resid = np.asarray(y, float) - np.polyval(coeffs, np.asarray(x, float))
    return float((resid**2).sum())


def _aic(sse: float, n: int, k: int) -> float:
    if sse <= 0 or n <= 0:
        return -math.inf
    return n * math.log(sse / n) + 2 * k


def fit_arrhenius(inv_t, ln_h):
    """Linear fit ln h = a - E/T. Returns (E, intercept, sse)."""
    slope, intercept = _polyfit(inv_t, ln_h, 1)
    return -float(slope), float(intercept), _sse(inv_t, ln_h, [slope, intercept])


def fit_quadratic(inv_t, ln_h):
    """Quadratic in 1/T. Returns (c2, c1, c0, sse); c2<0 => super-Arrhenius."""
    c2, c1, c0 = _polyfit(inv_t, ln_h, 2)
    return float(c2), float(c1), float(c0), _sse(inv_t, ln_h, [c2, c1, c0])


def fit_vft(temps, ln_h, n_grid: int = 400):
    """Fit ln h = lnA - B/(T - T0) by exact linear sub-fits over a T0 grid.

    For a fixed T0, ln h is linear in u = 1/(T - T0), so each T0 gets a
    closed-form least-squares fit; the grid picks the best T0 in
    [0, min(T)). Returns (T0, B, lnA, sse).
    """
    import numpy as np

    temps = np.asarray(temps, float)
    ln_h = np.asarray(ln_h, float)
    t_min = float(temps.min())
    best = (0.0, math.inf, 0.0, math.inf)
    # T0 must stay below the coldest sampled temperature.
    for t0 in np.linspace(0.0, t_min * 0.999, n_grid):
        u = 1.0 / (temps - t0)
        slope, intercept = np.polyfit(u, ln_h, 1)  # ln h = intercept + slope*u
        sse = float(((ln_h - (intercept + slope * u)) ** 2).sum())
        if sse < best[3]:
            best = (float(t0), float(-slope), float(intercept), sse)
    return best


def fragility_ratio(t0: float, temps) -> float:
    """E_eff(T_min)/E_eff(T_max) under VFT; >1 means fragile/super-Arrhenius."""
    lo, hi = min(temps), max(temps)
    if t0 <= 0:
        return 1.0
    return (lo**2 / (lo - t0) ** 2) / (hi**2 / (hi - t0) ** 2)


def diagnose(points: list[dict[str, Any]], n_boot: int = 500, seed: int = 0):
    """Classify one group's rate(T) curve. `points` are per-temperature dicts
    with keys temperature, matched, samples, mean_len (>=1 needed)."""
    import numpy as np

    pts = sorted(points, key=lambda d: d["temperature"])
    temps = [d["temperature"] for d in pts]
    inv_t = [1.0 / t for t in temps]
    p_hat = [jeffreys_rate(d["matched"], d["samples"]) for d in pts]
    ln_h = [math.log(per_token_hazard(p, d["mean_len"])) for p, d in zip(p_hat, pts)]
    zero_pts = [d["temperature"] for d in pts if d["matched"] == 0]
    # Rate at the detection ceiling: hazard is flat-topped there, biasing the
    # fit — the sub-saturation regime is where the curve shape lives.
    sat_pts = [d["temperature"] for d in pts if d["samples"] and d["matched"] / d["samples"] >= 0.98]

    result: dict[str, Any] = {
        "n_temps": len(pts),
        "temps": temps,
        "rate": [d["matched"] / max(d["samples"], 1) for d in pts],
        "ln_hazard": ln_h,
        "detection_limited_temps": zero_pts,
        "saturated_temps": sat_pts,
    }
    if len(pts) < 5:
        result["verdict"] = "INCONCLUSIVE (need >=5 temperatures)"
        return result
    n_informative = len(pts) - len(sat_pts)
    if n_informative < 4:
        result["verdict"] = "INCONCLUSIVE (rate saturates — add colder temperatures)"
        return result

    # Noise gate: a physical tipping rate is smooth in T, so a curve that
    # zig-zags by far more than binomial noise carries no fittable shape.
    rates = result["rate"]
    ses = [math.sqrt(max(p * (1 - p), 1e-6) / max(d["samples"], 1)) for p, d in zip(rates, pts)]
    diffs = [rates[i + 1] - rates[i] for i in range(len(rates) - 1)]
    direction_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
    median_se = sorted(ses)[len(ses) // 2] if ses else 0.0
    max_swing = max((abs(d) for d in diffs), default=0.0)
    result["direction_changes"] = direction_changes
    result["max_swing_over_se"] = max_swing / median_se if median_se > 0 else float("inf")
    if direction_changes >= 2 and max_swing > 4 * median_se:
        result["verdict"] = "INCONCLUSIVE (noise-dominated curve — more samples needed)"
        return result

    e_arr, _, sse_arr = fit_arrhenius(inv_t, ln_h)
    c2, _, _, sse_quad = fit_quadratic(inv_t, ln_h)
    t0, b_vft, _, sse_vft = fit_vft(temps, ln_h)
    aic_arr = _aic(sse_arr, len(pts), 2)
    aic_vft = _aic(sse_vft, len(pts), 3)
    d_aic = aic_arr - aic_vft  # >0 => VFT preferred

    # Parametric bootstrap over binomial noise for c2 and T0 CIs.
    rng = np.random.default_rng(seed)
    c2s, t0s = [], []
    for _ in range(n_boot):
        boot_ln_h = []
        for p, d in zip(p_hat, pts):
            m_star = int(rng.binomial(d["samples"], min(max(p, 0.0), 1.0)))
            boot_ln_h.append(
                math.log(per_token_hazard(jeffreys_rate(m_star, d["samples"]), d["mean_len"]))
            )
        c2s.append(fit_quadratic(inv_t, boot_ln_h)[0])
        t0s.append(fit_vft(temps, boot_ln_h)[0])
    c2_lo, c2_hi = float(np.percentile(c2s, 5)), float(np.percentile(c2s, 95))
    t0_lo, t0_hi = float(np.percentile(t0s, 5)), float(np.percentile(t0s, 95))

    if c2_hi < 0 and d_aic > 2 and t0 > 0:
        verdict = "SUPER-ARRHENIUS"
    elif c2_lo > 0:
        verdict = "SUB-ARRHENIUS"
    elif c2_lo <= 0 <= c2_hi and d_aic < 2:
        verdict = "ARRHENIUS"
    else:
        verdict = "INCONCLUSIVE"

    # A VFT fit that pushes T0 to the coldest sampled T is at the grid
    # boundary: E_eff and the fragility ratio diverge there and are meaningless.
    t0_at_boundary = t0 >= min(temps) * 0.98
    result.update(
        {
            "verdict": verdict,
            "E_arrhenius": e_arr,
            "c2": c2,
            "c2_ci": [c2_lo, c2_hi],
            "delta_aic_vft_minus_arr": d_aic,
            "T0": t0,
            "T0_ci": [t0_lo, t0_hi],
            "vft_B": b_vft,
            "vft_degenerate": t0_at_boundary,
            "fragility_ratio": (None if t0_at_boundary else fragility_ratio(t0, temps)),
        }
    )
    return result


def aggregate(sweep_rows: list[dict[str, Any]], key: str):
    """Pool sweep rows by `key` (e.g. behavior_id) and temperature."""
    groups: dict[str, dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"m": 0, "n": 0, "lsum": 0.0, "lcnt": 0}))
    for r in sweep_rows:
        cell = groups[r[key]][float(r["temperature"])]
        cell["m"] += int(r["n_matched"])
        cell["n"] += int(r["n_samples"])
        cell["lsum"] += float(r.get("mean_completion_tokens", 0)) * int(r["n_samples"])
        cell["lcnt"] += int(r["n_samples"])
    out: dict[str, list[dict[str, Any]]] = {}
    for name, by_t in groups.items():
        out[name] = [
            {
                "temperature": t,
                "matched": c["m"],
                "samples": c["n"],
                "mean_len": (c["lsum"] / c["lcnt"]) if c["lcnt"] else 1.0,
            }
            for t, c in sorted(by_t.items())
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="phase7_sweep.py output JSONL")
    parser.add_argument("--output", default="arrhenius_report.md")
    parser.add_argument("--key", default="behavior_id", choices=["behavior_id", "pattern_id"])
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.sweep, encoding="utf-8") if l.strip()]
    groups = aggregate(rows, args.key)
    results = {name: diagnose(pts, n_boot=args.n_boot) for name, pts in groups.items()}

    lines = [
        "# Arrhenius / super-Arrhenius diagnosis of the tipping rate",
        "",
        "Per-token hazard h(T) = -ln(1-p)/L fitted vs 1/T. SUPER-ARRHENIUS = the",
        "hazard falls faster than exponential as T drops (concave Arrhenius plot,",
        "c2<0), VFT preferred by AIC, and a finite critical temperature T0>0 where",
        "the extrapolated hazard vanishes. Fragility ratio = E_eff(T_min)/E_eff(T_max).",
        "",
        f"grouped by `{args.key}`; {len(rows)} sweep rows",
        "",
        "| group | verdict | #T | E_arr | c2 [90% CI] | ΔAIC(VFT) | T0 [90% CI] | fragility |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = {"SUPER-ARRHENIUS": 0, "SUB-ARRHENIUS": 1, "ARRHENIUS": 2}
    for name in sorted(results, key=lambda n: (order.get(results[n]["verdict"].split()[0], 3), n)):
        r = results[name]
        if "c2" not in r:
            lines.append(f"| {name} | {r['verdict']} | {r['n_temps']} | – | – | – | – | – |")
            continue
        frag = r.get("fragility_ratio")
        frag_str = "degenerate" if frag is None else f"{frag:.1f}x"
        lines.append(
            f"| {name} | {r['verdict']} | {r['n_temps']} | {r['E_arrhenius']:.2f} "
            f"| {r['c2']:.2f} [{r['c2_ci'][0]:.2f}, {r['c2_ci'][1]:.2f}] "
            f"| {r['delta_aic_vft_minus_arr']:+.1f} "
            f"| {r['T0']:.2f} [{r['T0_ci'][0]:.2f}, {r['T0_ci'][1]:.2f}] "
            f"| {frag_str} |"
        )
    lines += ["", "## Rate curves (match probability by temperature)", ""]
    for name in sorted(results):
        r = results[name]
        curve = "  ".join(f"T={t:.2f}:{p:.0%}" for t, p in zip(r["temps"], r["rate"]))
        flags = []
        if r.get("detection_limited_temps"):
            flags.append(f"zero-count at {r['detection_limited_temps']}")
        if r.get("saturated_temps"):
            flags.append(f"saturated at {r['saturated_temps']}")
        flag = f"  ({'; '.join(flags)})" if flags else ""
        lines.append(f"- **{name}** — {curve}{flag}")
    lines += [
        "",
        "Reading: a sharp super-Arrhenius verdict for the surface-form behaviors",
        "(language switching) with T0 well inside the sampled range, versus plain",
        "Arrhenius for fluent behaviors, would confirm the two-axes picture",
        "thermodynamically. INCONCLUSIVE usually means too few temperatures or a",
        "rate that never leaves the detection floor / ceiling in the sampled range.",
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report written to {args.output}")
    for name, r in results.items():
        print(f"  {name}: {r['verdict']}")


if __name__ == "__main__":
    main()
