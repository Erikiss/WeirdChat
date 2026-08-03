#!/usr/bin/env python3
"""Phase 10 probe batch — headless, parallel version of notebook Cells 14 + 14b.

Runs the 5-pattern perceptron (regimes + transitions) and the
precursor-or-prompt diagnostic on the raw routing captures, with the
embarrassingly-parallel parts fanned out over all cores via joblib:
  * the hidden-size comparison (sizes x CV folds)
  * the transcript-level permutation null (N_PERM full pipeline re-runs —
    the part that takes ~30 min sequentially on a 2-core Colab CPU)
On a 64-vCPU node the whole batch finishes in a few minutes, so N_PERM
defaults to 200 here (p-value resolution 0.005) instead of the notebook's 30.

Config via env: CAPTURE_DIR, DATA_DIR, OUT_DIR, N_PERM, N_JOBS.
Outputs into OUT_DIR: probe_results.json, five_pattern.png, precursor.png.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # joblib workers each get 1 BLAS thread
import json, glob, hashlib, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.stats import rankdata
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
warnings.filterwarnings("ignore")

CAPTURE_DIR = os.environ.get("CAPTURE_DIR", "/data/routing_caps")
DATA_DIR    = os.environ.get("DATA_DIR", "/data")
OUT_DIR     = os.environ.get("OUT_DIR", "probe_out")
N_PERM      = int(os.environ.get("N_PERM", "200"))
N_JOBS      = int(os.environ.get("N_JOBS", "-1"))
P_PRE = 2; SETTLE_HI = 19; SUS_MIN = 20; SUS_CAP = 40
HIDDEN = (25, 30, 64); UNITS = 30
NAMES = ["normal", "pre", "entry", "settling", "sustained"]
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS = {"params": dict(N_PERM=N_PERM, P_PRE=P_PRE, SETTLE_HI=SETTLE_HI,
                          SUS_MIN=SUS_MIN, SUS_CAP=SUS_CAP, HIDDEN=list(HIDDEN))}

def _rj(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def _ff(d, *names):
    for nm in names:
        p = os.path.join(d, nm)
        if os.path.isfile(p): return p
    for nm in names:
        h = sorted(glob.glob(os.path.join(d, "**", nm), recursive=True), key=len)
        if h: return h[0]
    return None

index = _rj(os.path.join(CAPTURE_DIR, "index.jsonl"))
_s = set(); index = [r for r in index if not (r["id"] in _s or _s.add(r["id"]))]
_cz = {}
def zl(rec):
    if rec["file"] not in _cz:
        z = np.load(os.path.join(CAPTURE_DIR, rec["file"]))
        _cz[rec["file"]] = (z["R"], z["A"])
        if len(_cz) > 300: _cz.pop(next(iter(_cz)))
    return _cz[rec["file"]]

tipped = [r for r in index if r["cls"] == "tipped" and r.get("tip_ai") is not None and r["tip_ai"] >= 1]
nsw = [r for r in index if r["cls"] == "tipped" and r.get("tip_ai") is None and not r.get("foreign_ai")]
ctl = [r for r in index if r["cls"] == "control"]
R0, _ = zl(index[0]); L, K = R0.shape[0], R0.shape[2]; E = max(256, int(R0.max()) + 1)
print("transcripts: switching=%d | same-behavior NON-switching=%d | other-behavior=%d"
      % (len(tipped), len(nsw), len(ctl)))

def token_cols(rec, i):
    R, A = zl(rec)
    if i < 0 or i >= len(A): return None
    return np.unique((np.arange(L)[:, None] * E + R[:, int(A[i]), :].astype(np.int64)).ravel())

def to_csr(row_lists):
    indptr = [0]; indices = []
    for cl in row_lists:
        indices.extend(cl.tolist()); indptr.append(len(indices))
    return sp.csr_matrix((np.ones(len(indices), np.float32),
                          np.array(indices), np.array(indptr)), shape=(len(row_lists), L * E))

def auc_ci(s, yy, boot=800, seed=0):
    a = roc_auc_score(yy, s); rng = np.random.default_rng(seed)
    p1 = np.where(yy == 1)[0]; p0 = np.where(yy == 0)[0]; bs = []
    for _ in range(boot):
        ii = np.concatenate([rng.choice(p1, len(p1)), rng.choice(p0, len(p0))])
        bs.append(roc_auc_score(yy[ii], s[ii]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# ========================= PART 1 — 5-pattern perceptron ====================
rows_i = []; y = []; grp = []; pos = []
rngd = np.random.default_rng(42)
def add_token(rec, i, label):
    c = token_cols(rec, i)
    if c is None: return
    rows_i.append(c); y.append(label); grp.append(rec["id"]); pos.append(int(i))
for r in tipped:
    t = r["tip_ai"]; fset = sorted(int(x) for x in r.get("foreign_ai", []))
    add_token(r, t, 2)
    for i in range(max(0, t - P_PRE), t): add_token(r, i, 1)
    for i in range(max(0, t - 10), max(0, t - P_PRE)): add_token(r, i, 0)
    for i in [x for x in fset if t < x <= t + SETTLE_HI]: add_token(r, i, 3)
    sus = [x for x in fset if x >= t + SUS_MIN]
    for i in (sus if len(sus) <= SUS_CAP else [int(v) for v in rngd.choice(sus, SUS_CAP, replace=False)]):
        add_token(r, i, 4)
for c in ctl:
    n = c["n_assistant"]
    pp = set(int(x) for x in rngd.integers(0, n, size=min(20, n))) if n > 0 else set()
    pp |= set(p for p in (0, 1, 2, 3) if p < n)
    for i in sorted(pp): add_token(c, i, 0)
X = to_csr(rows_i); y = np.array(y); grp = np.array(grp); pos = np.array(pos)
print("\nPART 1 dataset: %d tokens x %d expert features" % X.shape)
print("  " + " | ".join("%s=%d" % (NAMES[c], (y == c).sum()) for c in range(5)))

folds = list(GroupKFold(n_splits=5).split(X, y, groups=grp))
def fit_fold(h, tr_i, te_i):
    ytr = y[tr_i]; idx = np.arange(len(ytr))
    for c in (1, 2):
        e = np.arange(len(ytr))[ytr == c]
        if len(e): idx = np.concatenate([idx] + [e] * 49)
    clf = MLPClassifier(hidden_layer_sizes=(h,), activation="logistic",
                        alpha=1e-3, max_iter=400, random_state=0)
    clf.fit(X[tr_i][idx], ytr[idx])
    return h, te_i, clf.predict_proba(X[te_i]), list(clf.classes_)
fits = Parallel(n_jobs=N_JOBS)(delayed(fit_fold)(h, tr, te) for h in HIDDEN for tr, te in folds)
res = {}
for h in HIDDEN:
    P = np.zeros((len(y), 5))
    for hh, te_i, Pp, cls in fits:
        if hh != h: continue
        for j, c in enumerate(cls): P[te_i, int(c)] = Pp[:, j]
    pred = P.argmax(1)
    aucs = [roc_auc_score((y == c).astype(int), P[:, c]) for c in range(5)]
    res[h] = (P, pred, aucs)
print("\nhidden-size comparison (grouped 5-fold CV):")
print("  units | macro 1-vs-rest AUC | balanced accuracy")
for h in HIDDEN:
    tag = " <- factor-5/6 rule" if h in (25, 30) else " (reference)"
    print("   %3d  |        %.3f        |      %.3f%s" % (h, float(np.mean(res[h][2])),
          balanced_accuracy_score(y, res[h][1]), tag))
best = max((h for h in HIDDEN if h != 64), key=lambda h: np.mean(res[h][2]))
P, pred, aucs = res[best]
print("detail below uses the %d-unit perceptron" % best)
cis = [auc_ci(P[:, c], (y == c).astype(int), seed=c) for c in range(5)]
print("\nper-class one-vs-rest AUC (pooled test folds):")
for c in range(5):
    print("  %-10s AUC %.3f  [%.3f,%.3f]  (n=%d)" % (NAMES[c], *cis[c], int((y == c).sum())))
cm = confusion_matrix(y, pred, labels=list(range(5)), normalize="true")
print("\n5-class confusion (rows=true, cols=pred; row-normalized):")
print("  %-10s" % "" + "  ".join("%-9s" % n for n in NAMES))
for i in range(5):
    print("  %-10s" % NAMES[i] + "  ".join("%-9s" % ("%.2f" % cm[i, j]) for j in range(5)))
RESULTS["five_pattern"] = dict(
    best_units=int(best),
    size_table={int(h): float(np.mean(res[h][2])) for h in HIDDEN},
    per_class_auc={NAMES[c]: cis[c] for c in range(5)}, cm=cm.tolist())

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
im = ax[0].imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(NAMES, rotation=30, ha="right")
ax[0].set_yticks(range(5)); ax[0].set_yticklabels(NAMES)
for i in range(5):
    for j in range(5):
        ax[0].text(j, i, "%.2f" % cm[i, j], ha="center", va="center",
                   color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
ax[0].set_title("5-pattern confusion (%d units, row-norm.)" % best)
plt.colorbar(im, ax=ax[0], fraction=0.045)
aa = [c[0] for c in cis]; el = [c[0] - c[1] for c in cis]; eh = [c[2] - c[0] for c in cis]
ax[1].bar(range(5), aa, yerr=[el, eh], capsize=4, color="#2563EB", alpha=0.85)
ax[1].axhline(0.5, ls=":", color="black", lw=.8)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(NAMES, rotation=30, ha="right")
ax[1].set_ylim(0.4, 1.02); ax[1].set_ylabel("one-vs-rest ROC-AUC")
ax[1].set_title("per-pattern separability")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "five_pattern.png"), dpi=140)
plt.close(fig)

# ========================= PART 2 — precursor or prompt? ====================
if len(nsw) < 5:
    print("\nPART 2 skipped: only %d same-behavior non-switching transcripts (<5)" % len(nsw))
    RESULTS["precursor"] = None
else:
    rows2 = []; tag2 = []; grp2 = []
    def add2(rec, i, t):
        c = token_cols(rec, i)
        if c is None: return
        rows2.append(c); tag2.append(t); grp2.append(rec["id"])
    pre_sets = []
    for r in tipped:
        ps = [i for i in range(max(0, r["tip_ai"] - P_PRE), r["tip_ai"])]
        if ps: pre_sets.append(ps)
        for i in ps: add2(r, i, "pre")
    rngm = np.random.default_rng(7)
    def add_matched(rec, t):
        for p in pre_sets[rngm.integers(0, len(pre_sets))]:
            if p < rec["n_assistant"]: add2(rec, p, t)
    for r in nsw: add_matched(r, "nsw")
    for r in ctl: add_matched(r, "oth")
    X2 = to_csr(rows2); tag2 = np.array(tag2); grp2 = np.array(grp2)
    print("\nPART 2 tokens: pre=%d | nsw=%d | oth=%d (positions copied from pre multisets)"
          % ((tag2 == "pre").sum(), (tag2 == "nsw").sum(), (tag2 == "oth").sum()))

    def cv_auc(Xm, yb, gg):
        if yb.sum() < 5 or (1 - yb).sum() < 5 or len(set(gg)) < 5: return None
        Ps = np.zeros(len(yb))
        for tr_i, te_i in GroupKFold(n_splits=5).split(Xm, yb, groups=gg):
            if yb[tr_i].sum() == 0 or len(te_i) == 0: continue
            idx = np.arange(len(tr_i)); e = idx[yb[tr_i] == 1]
            idx = np.concatenate([idx] + [e] * 49) if len(e) else idx
            clf = MLPClassifier(hidden_layer_sizes=(UNITS,), activation="logistic",
                                alpha=1e-3, max_iter=400, random_state=0)
            clf.fit(Xm[tr_i][idx], yb[tr_i][idx])
            p = clf.predict_proba(Xm[te_i])[:, list(clf.classes_).index(1)]
            Ps[te_i] = rankdata(p) / len(p)
        return Ps
    def run_pair(neg_tag, groups=None, mask_extra=None, seed=0):
        m = (tag2 == "pre") | (tag2 == neg_tag)
        if mask_extra is not None: m &= mask_extra
        sel = np.where(m)[0]; Xm = X2[sel]; yb = (tag2[sel] == "pre").astype(int)
        gg = (grp2 if groups is None else groups)[sel]
        Ps = cv_auc(Xm, yb, gg)
        return (auc_ci(Ps, yb, seed=seed), Xm, yb, gg) if Ps is not None else (None,) * 4
    res2 = {}
    res2["A  pre vs other-behavior"] = run_pair("oth", seed=1)[0]
    resB, XB, ybB, ggB = run_pair("nsw", seed=2)
    res2["B  pre vs same-behavior non-switching"] = resB

    p_B = None; null = np.array([])
    if resB is not None:
        gids = sorted(set(ggB)); glab = {g: ybB[ggB == g][0] for g in gids}
        n_pos = sum(glab.values())
        rngp = np.random.default_rng(123)
        perms = [set(rngp.choice(gids, n_pos, replace=False)) for _ in range(N_PERM)]
        def one_perm(perm):
            yp = np.array([1 if g in perm else 0 for g in ggB])
            if yp.sum() < 5 or (1 - yp).sum() < 5: return None
            Pp = cv_auc(XB, yp, ggB)
            return None if Pp is None else roc_auc_score(yp, Pp)
        null = np.array([v for v in Parallel(n_jobs=N_JOBS)(
            delayed(one_perm)(pm) for pm in perms) if v is not None])
        p_B = float((1 + np.sum(null >= resB[0])) / (1 + len(null)))
        print("\npermutation null for B (%d transcript-level shuffles, parallel):" % len(null))
        print("  null AUC mean %.3f, 95%% range [%.3f,%.3f] | observed %.3f -> p=%.4f"
              % (null.mean(), np.percentile(null, 2.5), np.percentile(null, 97.5), resB[0], p_B))

    wp = _ff(DATA_DIR, "weird_transcripts.jsonl")
    pk = {}
    if wp:
        for tr in _rj(wp):
            u = next((t["content"] for t in tr["conversations"] if t["role"] == "user"), "")
            pk[tr["id"]] = hashlib.sha1(u.encode()).hexdigest()[:12]
    ksw = {pk[r["id"]] for r in tipped if r["id"] in pk}
    knsw = {pk[r["id"]] for r in nsw if r["id"] in pk}
    common = ksw & knsw
    print("\nprompt keys: switching=%d non-switching=%d OVERLAP=%d" % (len(ksw), len(knsw), len(common)))
    if len(common) >= 3:
        pgrp = np.array([pk.get(g, "?") for g in grp2])
        res2["C  prompt-matched, grouped by prompt"] = run_pair(
            "nsw", groups=pgrp, mask_extra=np.isin(pgrp, list(common)), seed=3)[0]
    else:
        print("  -> insufficient prompt overlap; test C skipped")

    print("\nRESULTS (perceptron %d units, grouped 5-fold CV, rank-pooled):" % UNITS)
    for k, v in res2.items():
        if v is None: print("  %-42s (not enough data)" % k); continue
        print("  %-42s AUC %.3f  [%.3f,%.3f]" % (k, *v))
    verdict = "NO_DATA"
    if resB is not None and p_B is not None:
        if p_B <= 0.05 and resB[0] > 0.75:
            verdict = "PRECURSOR"
            print("\n=> PRECURSOR: routing at the predecessor tokens already separates")
            print("   generations about to switch, within the same prompt family (p=%.4f)." % p_B)
        elif p_B > 0.20:
            verdict = "PROMPT_ARTIFACT"
            print("\n=> PROMPT ARTIFACT: in-family separation compatible with the null")
            print("   (p=%.4f); Cell 14's 1.000 was behavior/topic identity." % p_B)
        else:
            verdict = "INCONCLUSIVE"
            print("\n=> WEAK/INCONCLUSIVE (p=%.4f)." % p_B)
    RESULTS["precursor"] = dict(aucs={k: v for k, v in res2.items()}, p_B=p_B,
                                verdict=verdict,
                                null_mean=float(null.mean()) if len(null) else None,
                                null_95=[float(np.percentile(null, 2.5)),
                                         float(np.percentile(null, 97.5))] if len(null) else None)

    ok = [(k, v) for k, v in res2.items() if v]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    yy2 = np.arange(len(ok))
    ax.barh(yy2, [v[0] for _, v in ok],
            xerr=[[v[0] - v[1] for _, v in ok], [v[2] - v[0] for _, v in ok]],
            capsize=4, color=["#6B7280", "#2563EB", "#10B981"][:len(ok)], alpha=0.9)
    if len(null):
        ax.axvspan(np.percentile(null, 2.5), np.percentile(null, 97.5),
                   color="#DC2626", alpha=0.12, label="permutation null 95% (test B)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.axvline(0.5, ls=":", color="black", lw=.8)
    ax.set_yticks(yy2); ax.set_yticklabels([k for k, _ in ok], fontsize=9)
    ax.set_xlim(0.3, 1.02); ax.set_xlabel("ROC-AUC (pre vs control tokens)")
    ax.set_title("is the 'precursor' real or a prompt artifact?")
    ax.invert_yaxis(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "precursor.png"), dpi=140)
    plt.close(fig)

with open(os.path.join(OUT_DIR, "probe_results.json"), "w") as f:
    json.dump(RESULTS, f, indent=1)
print("\nresults ->", os.path.join(OUT_DIR, "probe_results.json"))
