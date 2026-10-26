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
