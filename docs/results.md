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

**Free-text ACS (LaBSE)**: pending (notebook 01, "Free-text items" cell).
