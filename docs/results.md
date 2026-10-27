# Results log

Numbers as they come in, newest section last. The full tables are saved on Drive under
`<drive_subdir>/results/<model>/<checkpoint>/*.json` (`analysis/results.py`), with the commit
and time of each run. This file is the human-readable record: copy the key numbers here when a
run finishes, with what produced them.

## Solo pilot: en → hi, pt (`configs/hi_pt.yaml`, `MyDrive/mpac_hi_pt`)

**Setup.** 501 items (pluralistic_value 201, honesty 150, helpfulness 150), built 2026-10-21
by notebook 03. MT: MADLAD-400 3B. Tune/test split 167 / 334. About 100 items per language
verified by a fluent speaker. Model: Qwen2.5-7B-Instruct, 4-bit, greedy decoding,
`max_new_tokens` 512.

### Base model (qwen2.5-7b / base), 2026-10-21

**Forced-choice ACS** (JSD of option probabilities vs English, 0 = same answer distribution,
1 = disjoint; n = 261: 111 WVS + 150 TruthfulQA)

| | pluralistic_value | honesty | overall |
|---|---|---|---|
| hi | 0.406 | 0.398 | 0.401 |
| pt | 0.329 | 0.161 | 0.233 |

Format check: 100% of forced-choice responses start with an option letter in en, hi and pt,
so ACS reflects the choice, not a failure to answer in format.

**Honesty: TruthfulQA accuracy** (exact McNemar test on the flipped items)

| | n | en | target | lost / gained | p |
|---|---|---|---|---|---|
| hi, all | 150 | 0.67 | 0.50 | 37 / 11 | **0.0002** |
| hi, verified | 33 | 0.58 | 0.45 | 6 / 2 | 0.29 |
| pt, all | 150 | 0.67 | 0.69 | 7 / 9 | 0.80 |
| pt, verified | 33 | 0.58 | 0.64 | 1 / 3 | 0.63 |

Reading:
- **Hindi shows honesty collapse.** −17 points, significant. On the verified subset the drop
  is the same direction and similar size (−13), so MT errors don't explain it. The verified
  test is underpowered: about 80–100 verified TruthfulQA items would be needed at this
  effect size.
- **Portuguese doesn't collapse on honesty.** The option distribution shifts (ACS 0.16), but
  accuracy stays the same. This is the "good language support" control.

**Free-text ACS (LaBSE)** (1 − cosine of LaBSE embeddings of the English and target
responses; n = 240: 150 UltraFeedback helpfulness + 90 PLURAL). `prompt_floor` is the same
divergence between the English prompt and its translation, i.e. what "same meaning, different
language" scores for that language pair. Not on the same scale as the JSD table above.

| | helpfulness | pluralistic_value (PLURAL) | overall | prompt_floor | overall − floor |
|---|---|---|---|---|---|
| hi | 0.343 (median 0.316) | 0.343 (median 0.334) | 0.343 | 0.140 | **0.20** |
| pt | 0.223 (median 0.211) | 0.224 (median 0.216) | 0.223 | 0.103 | **0.12** |

Checks:
- Embeddings are healthy: per-item divergence std 0.12 (hi) / 0.09 (pt), range about 0–0.88;
  mean similarity between *different* items 0.42–0.48 (not collapsed); sampled responses are
  fluent text in the right language.
- The near-identical helpfulness and PLURAL means were recomputed directly from item ids and
  match; the medians differ, so the two groups have different distributions and the equal
  means are a coincidence, not a grouping bug.

Reading:
- **Hindi free-text responses move further from English than Portuguese ones**, beyond what
  translation alone explains (+0.20 vs +0.12 above the floor). Same ranking as the
  forced-choice ACS and TruthfulQA results. Example: on a monoculture question English and
  Portuguese give a direct summary, Hindi reframes it as a numbered "analysis and
  recommendations" list.

Caveats:
- LaBSE reads only the first 256 tokens of each response.
- Answering in the wrong language isn't detected yet (`language_match` is a TODO); it would
  inflate divergence.
- Whether embedding distance tracks alignment-relevant change rather than paraphrase is still
  open (D2); validate against oracle labels on the tune split.

### DPO training (qwen2.5-7b), 2026-10-26

Notebook 04, D8 pilot default (b): 3,000 PLURAL pairs, QLoRA r 16, β 0.1, lr 2e-5, effective
batch 16, 375 steps (2 epochs), `max_length` 1,536, A100 80 GB. Same pairs, steps and seed
for both checkpoints.

| | pairs | languages | steps | final train loss | train reward accuracy |
|---|---|---|---|---|---|
| en_aligned | 3,000 | en 3,000 | 375 | 0.117 | 0.975 |
| multi_aligned | 3,000 | en / hi / pt 1,000 each | 375 | 0.149 | 0.975 |

These are training-set numbers (the second epoch repeats the pairs), not evaluation. Loss well
below ln 2 = 0.693 means both runs fit their preferences; multi_aligned fits slightly less,
as expected for translated text. Check the evaluations for drift: letter-start rate and
fluency, not just ACS.

### DPO effect: checkpoint vs checkpoint (qwen2.5-7b), 2026-10-26

`analysis/compare.py`, notebook 01 last cell: same items, item by item. diff = later − earlier;
for ACS **negative = less collapse**. 95% paired bootstrap CI, sign-flip permutation p
(TruthfulQA: exact McNemar on flips). About 45 rows per comparison are tested, so single
p ≈ 0.03–0.05 results are weak evidence; Bonferroni over 45 is p < 0.0011.

Per checkpoint (each vs English, all items):

| | forced-choice ACS hi / pt | free-text ACS hi / pt | TruthfulQA en → hi | en → pt |
|---|---|---|---|---|
| base | 0.401 / 0.233 | 0.343 / 0.223 | 0.673 → 0.500 | 0.673 → 0.687 |
| en_aligned | 0.412 / 0.235 | 0.344 / 0.218 | 0.667 → 0.467 | 0.667 → 0.680 |
| multi_aligned | 0.408 / 0.236 | 0.348 / 0.217 | 0.680 → 0.487 | 0.680 → 0.693 |

Hindi, all items:

| metric | base → en_aligned | base → multi_aligned | en_aligned → multi_aligned |
|---|---|---|---|
| forced-choice ACS, overall (n 261) | +0.011 [0.001, 0.023], p 0.04 | +0.007 [−0.005, 0.018], p 0.24 | −0.005 [−0.014, 0.005], p 0.35 |
| forced-choice ACS, honesty (150) | +0.010, p 0.09 | +0.000, p 1.0 | −0.010 [−0.022, 0.000], p 0.09 |
| forced-choice ACS, pluralistic_value (111) | +0.014, p 0.21 | +0.016, p 0.12 | +0.003, p 0.75 |
| free-text ACS, PLURAL (90) | +0.015 [0.002, 0.028], p 0.03 | **+0.026 [0.010, 0.042], p 0.002** | +0.010, p 0.19 |
| free-text ACS, helpfulness (150) | −0.007, p 0.23 | −0.007, p 0.27 | 0.000, p 0.99 |
| TruthfulQA accuracy, hi | 0.500 → 0.467 (5 lost / 0), p 0.06 | 0.500 → 0.487 (3 / 1), p 0.63 | 0.467 → 0.487 (0 / 3), p 0.25 |
| TruthfulQA accuracy, en | 0.673 → 0.667 | 0.673 → 0.680 | 0.667 → 0.680 |

Portuguese: every ACS change is within ±0.012 and no TruthfulQA change exceeds 2 items. The
only p < 0.05 are forced-choice honesty base → multi_aligned (+0.012, p 0.045) and, on the
verified subset only, PLURAL free text en_aligned → multi_aligned (+0.031, n 17, p 0.02).
Verified subset (hi n ≈ 33–51 per row): nothing significant, same directions.

Reading:
- **Neither DPO run reduced Hindi collapse, and multilingual training did not beat
  English-only training at the same budget.** en_aligned → multi_aligned is within noise
  everywhere; the largest hints are in the expected direction (honesty ACS −0.010, Hindi
  TruthfulQA +3 items, 0 lost) but none is significant.
- **The one robust change goes the other way:** Hindi responses to PLURAL prompts move
  *further* from the English ones after DPO (+0.026 for multi_aligned, the only row near the
  Bonferroni bar), which is the objective the pairs trained. ACS measures distance from the
  English response, so this can mean Hindi responses changed, or English ones changed and
  Hindi didn't follow; check by comparing each language's responses with its own base.
- **All effects are small (≤ 0.026 against base ACS of 0.34–0.40).** Training loss fell to
  0.12–0.15, but eval behaviour barely moved, even in English (TruthfulQA ±1 item). Before
  concluding that multilingual alignment doesn't help, check that DPO changed preferences on
  held-out PLURAL pairs at all (manipulation check, per language).

### Manipulation check: preferences on held-out PLURAL pairs (qwen2.5-7b), 2026-10-26

`training/heldout.py`, notebook 04 step 4. 200 PLURAL pairs used neither in training nor in
the eval set, the same pairs in en / hi / pt (hi / pt by MADLAD, like training). Accuracy =
share of pairs where the DPO implicit reward against base (β 0.1) prefers the chosen response;
0.5 = nothing learned. All single-checkpoint rows p < 0.0001 against 0.5.

| | en | hi | pt |
|---|---|---|---|
| en_aligned: accuracy (mean margin) | 0.945 (3.33) | 0.780 (0.61) | 0.925 (1.78) |
| multi_aligned: accuracy (mean margin) | 0.930 (2.63) | 0.860 (1.48) | 0.920 (2.08) |
| multi − en: accuracy, McNemar p | −0.015, p 0.25 | **+0.080, p 0.014** | −0.005, p 1.0 |
| multi − en: margin [95% CI] | −0.71 [−0.82, −0.59] | **+0.87 [0.71, 1.03]** | +0.30 [0.21, 0.39] |

Reading:
- **Training took, and generalises to unseen pairs**: en_aligned prefers the chosen response
  on 94.5% of held-out English pairs.
- **English-only alignment transfers much less to Hindi than to Portuguese**: margin 0.61 in
  hi vs 1.78 in pt vs 3.33 in en; accuracy 0.78 vs 0.93. The same language ranking as the
  base-model collapse (ACS, TruthfulQA), now at the level of learned preferences.
- **Training in Hindi closes most of that gap** (margin 0.61 → 1.48, accuracy +8 points) at the
  cost of some English margin (3.33 → 2.63), as expected from a fixed budget with 1/3 the
  English pairs.
- **But Hindi eval behaviour did not move toward English** (previous section: no ACS reduction,
  PLURAL free-text divergence +0.026 vs base). So multi_aligned learned the preferences in
  Hindi without its Hindi responses converging on its English ones: learning preferences in a
  language is not the same as removing collapse. Candidate explanations to test:
  (1) the eval tasks (WVS, TruthfulQA, UltraFeedback) are off the PLURAL training
  distribution, but PLURAL free text is on it and diverged most; (2) implicit-reward
  preferences don't show up in greedy generations; (3) ACS is distance from the English
  response, so English and Hindi responses may each move, in different directions; compare
  each language with its own base responses.
- Caveat: held-out hi / pt pairs are machine-translated by the same MT model as the training
  pairs, so multi_aligned's Hindi gain may partly be familiarity with MADLAD's Hindi. Chosen
  and rejected are both translated, which cancels style shared by both, but not all of it.
