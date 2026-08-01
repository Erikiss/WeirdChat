# Results: Colab A100 run (2026-07-29)

End-to-end run of the draft-surprise pipeline on one Colab A100-80GB against
`Qwen/Qwen3.6-35B-A3B-FP8` (the exact dataset checkpoint). Scale: baseline of
2,936 OpenRouter-regenerated conversations, DSpark draft trained ~230 steps
(10 epochs), 7,091 judge-matched weird transcripts from 326 patterns scored,
null distribution from 64 held-out baseline conversations (~48k tokens).

## Headline numbers

- Null (baseline held-out): draft NLL **5.01**, target-proxy NLL **0.21**,
  excess **4.80 ± 1.40** per transcript. The high draft NLL on its own
  training distribution marks the draft as heavily undertrained — a known
  trade-off of the mini-corpus run.
- Weird patterns range **6.7–10.6** mean excess (z up to **4.1** vs the null).
- Top 5 patterns by excess are all `language-switching-english`; token traces
  localize the tipping points exactly (e.g. Chinese onset directly after the
  `</think>` scaffold, peak at scored-token #31).

## Correlation with WeirdChat's judged metrics

Pooled over 326 patterns, excess **anti-correlates** with the judged axes
(Spearman): unexpectedness −0.248, harmfulness −0.369. This is not a pooling
artifact — within behaviors (8 behaviors with ≥10 patterns, n=290) the
sample-weighted mean is **−0.34**, and the between-behavior correlation of
behavior means is −0.31. Simpson's paradox was checked and ruled out.

Decomposing `excess = nll_draft − nll_target` against unexpectedness Elo:

| component | pooled ρ | within-behavior ρ (weighted) |
|---|---|---|
| `nll_draft` | −0.18 | −0.31 |
| `nll_target` | **+0.24** | **+0.12** (up to +0.49 in recommends-drunk-driving, +0.37 laser-at-aircraft) |
| `excess` | −0.25 | −0.34 |

## Interpretation

1. **Excess measures surface-form atypicality, not judged semantic
   weirdness.** Judges reward coherent, fluent, committed weirdness
   (chemtrails essays, confident harmful advice) — which is token-level
   *predictable*; garbled or code-switched text is token-level surprising but
   judged mundane. The two measures are complementary, and for semantic
   unexpectedness the excess even points the wrong way.
2. **The target's own NLL on its weird outputs is the judge-free correlate of
   unexpectedness**: semantically weirder patterns sit deeper in the model's
   own sampling tail (positive at pooled and within-behavior level). This
   quantity needs no draft at all — the draft's role would shift to a
   variance-reducing baseline once properly trained.
3. The undertrained draft (NLL 5.0 vs the ~1–2 of converged drafts) makes
   `nll_draft` behave like a generic-fluency meter, which amplifies (1).

## Caveats

Small null (64 transcripts) with a domain gap (Perfectblend instructions vs
natural chat prompts); baseline text from OpenRouter provider quantization vs
FP8 hidden states (the replication gap WeirdChat itself documents); the
`<think></think>` scaffold was scored (strip-think was off); FP8 dequantizes
to bf16 on Ampere, so "FP8 checkpoint" means FP8 weights, bf16 compute.

## If continuing

Train the draft to convergence on a larger baseline (multi-GPU or long Colab
runs), report `nll_target` and `excess` as two separate axes (semantic-tail
vs surface-form weirdness), enable `WEIRDSPEC_STRIP_THINK=1`, enlarge the
held-out null, and consider the weird-tuned contrast draft from the README's
optional extension for behavior fingerprints.

## Phase 6 addendum: HMM tipping analysis (2026-08-01)

A 2-state Gaussian HMM over per-token excess, fitted per behavior with the
baseline as control (16 groups, 7k sequences, CPU minutes after
vectorization). Key rows:

| group | mean_normal | mean_tipped | stay_tip | dwell_tip | tip% tokens |
|---|---|---|---|---|---|
| BASELINE (control) | 0.07 | 6.58 | 0.861 | 7 | 76% |
| typical behaviors (10 of 16) | ~0.1–0.3 | 6.6–7.6 | ~0.91 | 9–15 | 85–89% |
| **language-switching-english** | 5.66 | **12.25** | 0.752 | 4 | 56% |
| **unprompted-racial-slurs** | 3.12 | **10.01** | 0.490 | 2 | 45% |

Findings:

1. **The pre-registered control check fired**: the baseline also "tips" into a
   ~6.6-mean state covering 76% of tokens — the generic high state is the
   undertrained draft's default surprise level, not a behavior mode. For most
   behaviors the fitted states mirror the control and identify nothing
   behavior-specific.
2. **Two behaviors break the control pattern**, and they are exactly the
   surface-form-weird ones: language switching fits a well-separated
   ultra-hot state (mean 12.25, ~2x the control's; every transcript enters
   it, and its HMM switch points align best with the phase-5 peaks — median
   distance 97 vs 130–550 elsewhere), and racial slurs fit a hot *flickering*
   state (dwell 2 — short bursts at the slur tokens, not a sticky mode).
3. Reading: a latent "tipped mode" in the Markov sense is identifiable for
   token-level mode switches (language, slur bursts) even under a weak draft;
   for semantically weird but fluent behaviors it is not — consistent with the
   two-axes conclusion above. A converged draft (calm control) and/or k=3
   states, or emissions from `nll_target` instead of excess, are the natural
   refinements.
