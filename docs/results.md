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
