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

- **One prompt for the positive results.** The set is defined on a single prompt
  and every arm above is a variant of the same task. The transfer test is now
  run (see the phase 13 subsection) and it came back negative, which bounds the
  claim rather than extending it.
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

### Phase 13: the same set on multiplication (2026-08-06)

The obvious way to widen the claim is to put the same 42 experts on a task with
nothing to do with storage services. Multiplication is the principled choice, not
an arbitrary one: `47*83` has no lexical entry any more than `⠛⠕⠕⠛⠇⠑` does, so it
sits on the same retrieval-vs-construction axis that separated Cyrillic from
Braille. (Prompted by a LessWrong write-up of digit-wise multiplication in Qwen
2.5 7B — a different generation and a dense model, so only the task was borrowed,
not the finding. Our model does tokenize digits singly too: `110848` is six
tokens.)

Four tasks, 64 problems each, greedy decoding, three conditions, dose matched over
router slots as before. Pre-registered gate, inverted from phase 12: capital-city
retrieval must *survive* the tested mask, or nothing else counts.

| task | baseline | tested mask | dose-matched random |
|---|---|---|---|
| capitals (retrieval) | 100% | 100% | 100% |
| 1x1 digit (retrieval) | 100% | 100% | 100% |
| 2x2 digit | 100% | 100% | 100% |
| 3x3 digit | 93.8% | 92.2% | 93.8% |

All four `STILL`; per-digit accuracy unchanged (96.5% vs 96.0% on 3x3). This is a
tight null, not an underpowered one: at 64/64 under the mask the Wilson lower
bound is 94.3%, so any drop above **5.7 points** is excluded, against the **54–82
points** the same set produces on Japanese, Braille and Morse. The experts are not
idle during arithmetic either — they occupy 84–97 router slots per prompt, a
higher density than on the (much longer) storage prompt. They are active and
irrelevant.

**Consequence for the description above.** "No lexical template" was too wide:
`168` has none and does not fall. The defensible formulation is narrower —
**transcoding a known string into another symbol system, character by character.**
The content is fixed and only the symbol system changes. `Гугл драјв` is a
different word, not a transcoding; `47*83` is an unknown value, not a
transcoding. "Both are stepwise" does not create a shared mechanism.

Caveats specific to this run: 2x2 multiplication at 64/64 is plausibly retrieval
itself, so 3x3 is the only genuine construction arm and it carries the weakest
bound (drops above 10.8 points excluded). Greedy decoding pins three of four arms
to the ceiling; 4- or 5-digit operands with a 40–60% baseline would sharpen the
bound in both directions. And a null on a transfer test says "no effect found",
not "no effect".

### Phases 14–15: a routing screen finds a different set, and it does nothing (2026-08-06)

The set of 42 above comes from a **single-sample difference at one position**.
An obvious worry: is that how you *should* look for a circuit? The method paper
in `Erikiss/spectral-probe-circuits` proposes a different route — screen for
**persistent selectivity over the whole answer**, calibrate a null per frequency
stratum, then ablate the surviving set against a matched-random control. Phases
14 and 15 port that recipe to a MoE router and run it end to end.

**Phase 14 — the screen.** Selectivity is the firing rate in an arm divided by
the rate in the English reference arm, with the null calibrated by splitting the
reference arm against itself 200×, per log2 frequency stratum. A second floor is
calibrated from **Serbian vs Russian** — two arms with different text and the
same property — because surface-form differences alone selected a quarter of the
model in the first attempt. The stricter of the two floors is used.

| arm | pairs | max sel. | median |
|---|---|---|---|
| JA | 242 | 7727 | 142 |
| SR | 242 | 7741 | 144 |
| RU | 179 | 6965 | 128 |
| BR1 | 581 | 5466 | 117 |
| MORSE | 448 | 4889 | 109 |

`JA & BR1 & MORSE` = 17 pairs; minus `SR & RU` = **8 pairs**. Label-shuffle null:
0.0 expected, 17 observed, p = 0.0050 at the permutation floor. So the shared
signature is real *as routing*.

Two things are worth noting before the ablation. **Overlap with the 42 is zero**
— Jaccard 0.000, not one shared expert. And the normalised pairwise similarity
ranks Japanese with the Cyrillic arms, not with Braille and Morse:
SR/RU 39.7× > BR1/MORSE 11.6× > JA/SR 7.5× > JA/RU 7.3× > … > JA/BR1 2.9× >
JA/MORSE 1.7×. By persistent routing, Japanese is a *script* like Cyrillic. The
grouping that Phase 12 found causally does not appear in the routing statistics.

**Phase 15 — the ablation.** Four conditions, three matched on **expert count**:

| arm | baseline | SCREEN (8) | P12-subset (8) | RANDOM (8) | ALL 42 |
|---|---|---|---|---|---|
| JA (kana) | 100 % | 100 % | 95 % | 97 % | **9 %** |
| BR1 (braille) | 59 % | 47 % | 61 % | 66 % | **9 %** |
| MORSE | 83 % | 84 % | 89 % | 89 % | **23 %** |
| SR (cyrillic) | 86 % | 80 % | 81 % | 73 % | 84 % |
| RU (cyrillic) | 98 % | 98 % | 98 % | 100 % | 100 % |

The positive control is unambiguous: the full 42 collapse all three constructing
arms (p < 10⁻⁴ each) and leave both Cyrillic arms untouched — a clean replication
of Phase 12 inside this run. So the measurement chain works, and a null in the
other columns is a null about the sets, not about the apparatus.

**The 8 screen experts do nothing.** At Braille they occupy 2125 router slots —
45 % of what the full 42 block — and the score moves 59 → 47 %, p = 0.21. At
Morse, 1523 slots, 39 % of the control's dose, and nothing at all (p = 1.00).
This is §9.2 of the method paper confirmed in a MoE: the spectral/selectivity
signal alone is not a circuit finder. Persistent, highly selective, causally
inert.

**A random eighth of the 42 also does nothing** — at Japanese it blocks 4068
slots, 24 % of the full set's dose, and the score stays at 95 % (p = 0.24).
Verdict: `WIRKUNG-IST-VERTEILT`. The 42 are not 42 independent contributors;
whatever they do survives the removal of a fifth to a quarter of it.

**Caveats.** Count-matching trades one confound for another: eight experts do not
block as many router slots as forty-two, so the null columns sit at 8–45 % of the
positive control's dose depending on the arm. The screen null is reasonably
strong for Braille and Morse (39–45 % of dose) and weak for Japanese (7.6 %). Only
**one** random eighth was drawn, so "no eighth works" is not established — this
particular eighth did not. And the two sets live in different parts of the model:
the screen in layers [7, 14, 15, 17, 23, 24, 27], the 42 spread over 23 layers
with two thirds in layers 30–39.

**What this does not say.** It does not say the screen is worthless — it says a
selectivity screen and a causal set are different objects in this model, which is
exactly what the method paper warns about, and it says so with a working positive
control in the same run rather than by assertion. The obvious follow-up is a
dose–response curve inside the 42: block 8, 16, 24, 32, 42 and see whether the
effect appears gradually or at a cliff.

### Phase 19: what the clock can see — and a caveat for phases 14 and 17 (2026-08-07)

The proposal was a phase coordinate φ(t) ∈ [0,2π) inside the model's rhythm, then P(φ | e) per
expert, layer-conditioned as Δ_e(φ) = P(φ|e,L) − P(φ|L). Five independent reviews plus an
adversarial refutation pass concluded that this coordinate does not exist in this
architecture, for three independent reasons — and the run then measured all three rather than
asserting them.

**No carrier.** A phase needs something periodic. In autoregressive decoding exactly one event
repeats — the token — so the only carrier the mechanism admits has period 1, and at period 1
"phase within the period" *is* the position in the forward pass. Measured: 3 of 10 runs show
an autocorrelation peak above the within-run permutation null, all at lag 2 — the signature of
neighbour correlation, not a carrier. Verdict `KEIN-TRAEGER`.

**The finest separable time coordinate is the layer.** All eight experts of a layer are
processed at the same point of the pass. Δ_e(φ|L) has no residual variance that could belong
to *e*; what it would return is token selection — conditioning on "e was routed" selects
tokens with different content and KV length, i.e. exactly the confound the conditioning was
meant to remove.

**The arrow points the other way.** "Expert e prefers the high-load phase" needs load →
routing. Measured: **0 flipped slots out of 10 240, in 4 rounds under forced GEMM contention.**
Routing is a function of input and weights; clock and temperature do not enter the top-k. This
is a bound of 2.4 × 10⁻⁵, not a proof.

**What was measurable instead.** The MoE forward in transformers 5.x is a Python loop over the
hit experts (`for expert_idx in expert_hit:`), not a fused grouped GEMM. At B = 1 that is
exactly 8 iterations per layer, 320 per token. One expert is **6.00 MiB** (gate_up 4.00 + down
2.00), so a decode step addresses **2.01 GB** of expert weight — *independently of which eight*.
Between two touches of the same pair more than 3.5 GB streams through a 40 MB L2, so
cross-token cache reuse is arithmetically dead; that was the obvious rescue of the idea and it
falls. What survives is **diversity within a step**, which is causally forcible via the same
hook mechanism used for masking since Phase 12.

| forced diversity S | 320 | 640 | 1270 | 2245 | 3300 |
|---|---|---|---|---|---|
| median step time | 204.8 ms | 214.3 ms | 230.8 ms | 254.2 ms | 276.6 ms |

**κ = 0.0239 ms per additional distinct expert** (bootstrap CI [0.0232, 0.0253], block-label
permutation p = 0.0005). Pure bandwidth would predict 6 MiB / 1.55 TB/s ≈ 4.1 µs; the measured
24 µs is **6× that**, so the runtime pays for the *loop iteration* — kernel launches and syncs
— not for memory traffic. That also explains the 195 ms per token.

**Identity, at fixed diversity, is invisible.** Forcing the 42 versus a *freshly drawn*
rate-matched partner set each repeat (which also answers the Phase-15 caveat that only one
random eighth was ever drawn): paired median over 12 run pairs = **77.8 µs = 0.04 % of step
time = 3.3 experts' worth** of the diversity channel measured in the same run. The A/A pairs
calibrate the null inside the run and do not reject (p = 0.81).

**A correction to the run's own verdict logic.** The first run printed `UHR-BLIND` while the
dose curve above sat in the same protocol. The injection ladder feeds a delay into *individual
steps*, and at 195 ms per step with millisecond noise, 0.8 ms is not recoverable there — but
H1/H2 compare *run medians* over 90 steps, whose noise is smaller by √n. One number was being
used for two channels. The gate now carries both resolutions and declares blindness only if
both fail; the run-level resolution is measured from the A/A pairs rather than injected.

**The caveat for earlier phases.** Prefill and decode do not route identically: exact top-8
agreement is **80.6–82.2 %** across arms, mean overlap **7.80 of 8**. Roughly one in five
(layer, position) pairs flips a single expert — the signature of a near-tie in the eighth rank
under a different reduction order. Phases 12, 15, 16 and 18 are unaffected (`hole_routing`
reads through the cache path, i.e. as generation actually runs). **Phases 14 and 17 are
affected**: both derive routing from a single teacher-forced pass over prompt and answer. The
mean overlap is high, so this is a caveat and not a retraction — but the Phase-14 screen sets
in particular were already unstable, and this is one contributing cause.

**What this cannot say.** Nothing about Qwen3.6 — step time is a property of the *runtime*.
The same routing under a fused kernel, CUDA graphs, or vLLM would produce a different rhythm
entirely; every timing result here belongs in a section about transformers 5.x on sm80.
Nothing below the resolution. Nothing about a phase below the layer — there is none. And H1/H2
say nothing about behaviour: forced routing destroys the output, that is its purpose. H3 (load
counter by arm) is uninformative as run: at B = 1, S is exactly 320 for every arm by
construction, so the arm question needs a batch.

### Notebooks

`phase12_experten_maske` (router masking), `phase12_sprachkarte` (four arms,
overlap null), `phase12_entscheidungsstelle` (harvest at the decision point),
`phase12_minimalpaar_v2` (paired prefixes, permutation test),
`phase12_schrift_gegen_sprache` (Braille/Morse/romaji pilot),
`phase12_schrift_kontrolle_dosis` (dose-matched control; set `WIEDERHOLUNG` for
an independent repeat), `phase12_nachlese_entziffern` (CPU-only: decodes Braille
and Morse, with a decoder positive control that halts the run if it fails),
`phase13_rechnen` (the same set on multiplication; `WIEDERHOLUNG` for an
independent repeat), `phase14_screen` (routing-selectivity screen with two
independently calibrated floors), `phase15_ablation` (four conditions matched on
expert count, with the full 42 as a positive control that gates the verdict),
`phase16_kurve` (dose-response curve inside the 42, three nested chains),
`phase17_impuls` (when the experts fire: burstiness, character binding, co-firing,
depth vs time), `phase18_kern` (single-expert scan against a rate-matched empirical null),
`phase19_taktgeber` (what the clock can see: forced routing diversity, identity bound,
prefill-vs-decode audit).
