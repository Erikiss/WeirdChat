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

## Phase 12 addendum: router-level expert ablation (2026-08-06)

Phases 7–12 asked a different question from the excess/NLL work above: not *how
surprising* is `language-switching-english`, but *where in the network does it
live*. Target: `Qwen/Qwen3.6-35B-A3B-FP8`, 40 MoE layers x 256 experts, top-8 —
10,240 (layer, expert) pairs. Method: a forward pre-hook on `mlp.experts` that
zeroes the router weight of chosen experts (`w.masked_fill(isin(idx, bad), 0)`).
No weight editing, no backup, acts at every position, removed by dropping the
hook. All runs are single self-provisioning Colab cells with an offline test
suite (245 tests) that executes each cell's body against a numpy miniature MoE.

### The set

Take the prompt that produces the behavior, swap the phrase `each service's
local name` for `each service's Japanese name`, and record which experts the
router selects at the decision position. Subtract the experts selected under
`each service's name`. That leaves **42 pairs of 10,240** — 0.4% of the model.

### What blocking them does

| arm | phrasing | without | with mask |
|---|---|---|---|
| JA | `each service's Japanese name` | 99–100% kana | **14–18%** |
| BR1 | `each service's Braille name` | 57–68% Braille | **14–16%** |
| MORSE | `... name written in Morse code` | 82–90% Morse | **25–29%** |
| SR | `each service's Serbian name` | 77–84% Cyrillic | 76–85% |
| RU | `each service's Russian name` | 95–96% Cyrillic | 96–98% |

Two independent runs, disjoint seeds (the second via `WIEDERHOLUNG=1`; the first
predates that parameter and used the older seed scheme, so it is not reproducible
by setting `WIEDERHOLUNG=0` — it is independent of the second either way). Pooled,
mask against dose-matched random control: Japanese 30/192 vs 188/192, Braille
28/192 vs 100/192, Morse 52/192 vs 169/192, all p < 1e-6; Serbian p = 0.22 and
Russian p = 0.50.

### Why this is neither script nor language

Both readings are refuted by the same table. A *script* set would drop Serbian
(Cyrillic) and spare Morse (pure ASCII). A *language* set would do the same. The
observed pattern is the opposite on both counts. What Japanese, Braille and Morse
share and Cyrillic does not: the name has to be **built character by character**.
A katakana rendering of "Google Drive" is a construction; `Гугл драјв` is a
lookup.

### The dose problem

The first control matched **pair count** (42 random vs 42 tested) and thereby
blocked roughly twice as many router slots, because ordinary experts fire more
often than the tested ones — 264 vs 557 slots on the Braille arm. The control was
the *harsher* intervention and still weaker, but a verdict rule that only asked
"are both significant?" recorded that as fragility. Fixed by measuring routing
over the whole text, counting slots per (layer, expert), and greedily assembling
a random set whose slot total lands within ±10% of the tested set's on that arm.
Effect is then reported per 100 blocked slots.

Across both runs, the tested set on the three construction arms scores **+15.8 to
+23.0** points per 100 slots; the random control on *all* arms scores **−2.4 to
+6.7**. The weakest construction effect is 2.4x the strongest random effect and
the two ranges do not overlap.

### It carries the quality, not just the attempt

The rate measure only asks whether Braille *appears*. Decoding it (Grade-1 dot
patterns, matched against the service names in the prompt) separates attempting
from succeeding. Pooled over both dose-matched runs, conditional on Braille being
present:

| condition | correct |
|---|---|
| no intervention | 82/120 = **68.3%** |
| tested set masked | 9/28 = **32.1%** (p = 0.00093 vs baseline) |
| dose-matched random | 69/100 = **69.0%** (p = 1.00 vs baseline) |

Mask against control directly: p = 0.00079. Not a length artifact — masked
answers carry the same amount of Braille (median 28 vs 29; a baseline answer
carries more in 55% of pairwise comparisons), and the gap survives within equal
character bands (79%→38%, p = 0.015; 71%→26%, p < 1e-4).

Morse is different: the model produces Morse *form* — dots, dashes, slashes,
correctly grouped — that decodes to noise. Baseline correctness is 0–2 of ~85.
That arm therefore measures the attempt only, and is the sharpest evidence for
it: pure ASCII, no script change, and the willingness still depends on the same
42 experts.

### Related findings from phases 7–11

- **Blocking redirects rather than removes.** Japanese gives way to Korean,
  Chinese and German; on the ambiguous prompt CJK falls 21→5 with French, German
  and Latvian appearing; the all-scripts count falls 73/96 → 30/96 spread across
  five scripts. The impulse is not deleted, it takes another route.
- **The interpretive disposition has no router seat.** Four searches for a
  consistent expert set behind the *decision* to reinterpret the instruction
  found none — maximum consistency 3 of 9 paired prefixes, even for a
  13-character insertion with a 47-point behavioral effect.
- **The header string is causal, not a marker.** Nine paired prefixes harvested
  from the model's own output, three conditions: `Service (Local Name)` 47.9%,
  no insertion 25.7%, `Service (Short Name)` 0.7%. A vs I p = 0.0078 (the exact
  floor at nine pairs), A vs B sign-flip permutation on the mean p = 0.031.

### Caveats

- **One prompt.** The set is defined on a single prompt and probed with variants
  of the same task. Whether the same 42 experts carry character-wise construction
  in an unrelated task is untested, and this is the largest limitation here —
  larger than any p-value above.
- **The random control is not perfectly inert.** Of ten random contrasts (5 arms
  x 2 runs), the smallest is p = 0.0069 (Serbian, run 1) with Braille at p =
  0.055. Under a global null the chance of a minimum that small is 0.066 —
  compatible with noise, but not comfortably so. A dose-matched block of ~260
  slots can occasionally do something.
- **The Braille quality result rests on 28 masked answers** pooled across two
  runs.
- **A seeding bug inflated apparent replication.** Draw seeds were derived from
  an arm's *position* in the arm list, so three consecutive runs produced
  bit-identical masked samples for one arm — three apparent confirmations from a
  single sample. Seeds now derive from the arm name, and `WIEDERHOLUNG` shifts
  every draw plus the random set. Consequence for what is pooled above: the two
  dose-matched runs are independent of each other and are pooled; the pilot and
  the first (non-dose-matched) control share the first run's masked samples
  exactly and are therefore *not* counted as further evidence anywhere here.
- The Cyrillic null may reflect lexical availability rather than the absence of
  construction as such — which is the hypothesis, not an independent check.
- Same FP8-dequantizes-to-bf16 caveat as above; on A100 this is bf16 compute.

### Notebooks

`phase12_experten_maske` (router masking), `phase12_sprachkarte` (four arms,
overlap null), `phase12_entscheidungsstelle` (harvest at the decision point),
`phase12_minimalpaar_v2` (paired prefixes, permutation test),
`phase12_schrift_gegen_sprache` (Braille/Morse/romaji pilot),
`phase12_schrift_kontrolle_dosis` (dose-matched control; set `WIEDERHOLUNG` for
an independent repeat), `phase12_nachlese_entziffern` (CPU-only: decodes Braille
and Morse, with a decoder positive control that halts the run if it fails).
