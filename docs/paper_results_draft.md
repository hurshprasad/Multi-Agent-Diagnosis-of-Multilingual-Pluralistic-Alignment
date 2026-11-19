# Results (draft, en → hi / pt pilot)

> Draft written 2026-11-19 from `docs/results.md`; every number traces to a dated section there.
> Tables 6–9 keep the proposal's numbering. Bracketed table names ("[Table: collapse]") are
> placeholders for the proposal's numbers for those tables. Model throughout: Qwen2.5-7B-Instruct
> (4-bit, greedy). Targets: Hindi (lower-resource contrast) and Brazilian Portuguese; English is
> the reference. 501 parallel items: pluralistic value 201 (111 WVS forced choice, 90 PLURAL free
> text), honesty 150 (TruthfulQA MC1, forced choice), helpfulness 150 (UltraFeedback, free text).
> Prompts machine-translated (MADLAD-400 3B) with hand fixes to WVS answer scales and the Hindi
> instruction; about 100 items per language verified by a fluent speaker.

## 5.1 Alignment collapse is real, language-dependent, and not a translation artefact

**Forced choice.** We measure collapse as the Jensen–Shannon divergence between the option
distributions the model assigns to the English and the target-language version of the same
question (ACS; 0 = identical, 1 = disjoint). Hindi diverges from English substantially more
than Portuguese (overall ACS 0.376 vs 0.224; WVS 0.348 vs 0.309) [Table: collapse].

**Honesty.** On TruthfulQA, the model answers 67.3% correctly in English, 68.7% in Portuguese
and 50.7% in Hindi. The Hindi drop is 36 questions lost against 11 gained (exact McNemar
p = 0.0003). Portuguese loses nothing (p = 0.80). On the verified-translation subset (n = 33)
the Hindi drop has the same direction and similar size (0.58 → 0.45) but is not significant
(6 lost / 2 gained, p = 0.29); the subset is too small at this effect size.

**Free text.** Hindi responses move further from the English response than Portuguese ones
do, beyond what translation alone produces: LaBSE divergence 0.343 vs 0.223, against prompt
floors (English prompt vs its translation) of 0.140 and 0.103, i.e. +0.20 vs +0.12 above the
floor.

**Translation errors inflate but do not explain collapse.** Hand-fixing calques in the MADLAD
WVS answer scales and Hindi instruction reduced Hindi WVS ACS by 14% (0.406 → 0.348) and
Portuguese by 6%; Hindi honesty collapse was unchanged in size. One WVS block (E069,
confidence in institutions) remains wording-sensitive in Portuguese, so we report
pluralistic-value ACS with and without it (base, without E069: hi 0.352, pt 0.283).

Every collapse measure gives the same ranking, Hindi > Portuguese. Portuguese, where the
option distribution shifts but accuracy does not, serves as the well-supported control.

## 5.2 English-only and multilingual alignment do not reduce collapse

We trained two DPO checkpoints with identical budgets (3,000 PLURAL pairs, QLoRA r 16, 375
steps, same seed): **en_aligned** on English pairs only, **multi_aligned** on 1,000 pairs each
in English, Hindi and Portuguese.

**Training took and generalises.** On 200 held-out PLURAL pairs, en_aligned's implicit reward
prefers the chosen response on 94.5% of English pairs [Table: manipulation check]. **English-only
preferences transfer much less to Hindi than to Portuguese** (accuracy 0.78 vs 0.93; mean
margin 0.61 vs 1.78 vs 3.33 in English). Training on Hindi pairs closes most of that gap
(multi_aligned: 0.86, margin 1.48; +8 points over en_aligned, McNemar p = 0.014).

**Behaviour did not follow.** Neither checkpoint changed forced-choice ACS or TruthfulQA
accuracy in either language (Hindi: all three checkpoints within 2 TruthfulQA items, every
ACS change ≤ 0.014, none significant) [Table: DPO effect]. The one robust change goes the other
way: Hindi responses to PLURAL prompts move *further* from the English ones after
multilingual DPO (+0.026 LaBSE divergence, 95% CI [0.010, 0.042], p = 0.002). The trained
objective is the one that diverged. **Learning preferences in a language is not the same as
removing collapse in it.** (An apparent cost of English-only DPO to Hindi honesty before the
translation fixes, −5 items, p = 0.06, disappeared once the instruction was fixed.)

## 5.3 How reliable are the collapse labels?

No fluent human annotators were available for the labelling volume, so oracle labels come from
an LLM panel: Claude Opus 5.5 and Aya Expanse 32B (two vendors, neither the model under test
nor the System 6 judge) label independently with the written protocol; Claude Sonnet 5.5
adjudicates only the fields they disagree on; rows any model marks as a bad translation are
dropped (hi 41, pt 15 of 441). Three pilot rounds preceded the main round; the protocol and
every change to it are logged.

**Forced-choice labels are reliable; free-text labels are not.** On TruthfulQA rows, where the
answer key gives ground truth, both annotators caught every case of the target language losing
the correct answer (Opus 31/31, Aya 30/31) and never flagged an unchanged answer (176/176,
175/176). Agreement on `collapsed_vs_reference` splits sharply by item type:

| Cohen's κ (bad translations dropped) | forced choice | free text, 512 tokens | free text, 1,536 tokens |
|---|---|---|---|
| Hindi | 0.94 (n 218) | 0.41 (n 182) | 0.34 (n 204) |
| Portuguese | 0.93 (n 225) | 0.22 (n 201) | 0.10 (n 225) |

The main round passed our pre-registered κ ≥ 0.6 gate on pooled rows (hi 0.73, pt 0.61), but
only because forced-choice rows carried it. On open-ended responses, "clearly less useful" and
"a different stance" are judgement calls that two strong models make differently (on
Portuguese helpfulness, Aya calls collapse at nearly twice Opus's rate, 0.38 vs 0.22; PLURAL
agreement is near chance in both languages). We therefore treat every free-text detection
result below as provisional.

**Response truncation is not what the annotators label as Hindi collapse.** At 512 new tokens,
80% of Hindi free-text responses hit the limit (English 35%). Regenerating free text at 1,536
tokens (Hindi 27.5% cut off, English 0.4%, Portuguese 0%) and re-labelling with the same panel
made Hindi free text look *more* collapsed, not less (Opus: 70% → 82% of items; PLURAL 56% →
88%), with Portuguese unchanged. Complete Hindi answers diverge more from English than
truncated ones do.

## 5.4 Diagnosing collapse: cheap evaluators, an LLM judge, and routing (Table 6)

Systems 4–7 read the same cached generations and are scored on held-out **test** labels;
thresholds are tuned on **tune** labels only. Systems 4, 5 and 7 use one cheap signal per
item type: JSD of option probabilities (forced choice) or LaBSE divergence (free text).
System 6 is an LLM-as-judge team (Claude Haiku 5.5).

**Table 6.** Test set, base checkpoint (541 labelled pairs, 202 collapsed).

| system | precision | recall | F1 | LLM tokens |
|---|---|---|---|---|
| 4 single evaluator | 0.792 | 0.658 | 0.719 | 0 |
| 5 fixed multi-agent | 0.792 | 0.658 | 0.719 | 0 |
| 6 LLM-judge multi-agent | 0.837 | 0.787 | 0.811 | 1.41M |
| 7 adaptive lifecycle (ours) | 0.793 | 0.644 | 0.710 | 0 |
| routed (follow-up, see below) | 0.833 | **0.891** | **0.861** | ≈ 0.67M† |

† Lower bound: 480 of 1,002 pairs go to the judge.

**The average hides two opposite results.** By item type (test F1, cheap signal / judge):

| | forced choice | free text |
|---|---|---|
| Hindi | **0.950** / 0.893 | 0.496 / **0.847** |
| Portuguese | **0.909** / 0.743 | 0.311 / **0.550** |

- **On forced choice, the exact signal beats the judge at zero LLM cost.** The judge misses
  blatant changes (e.g. JSD 0.999 on a WVS item labelled "no collapse"). Part of JSD's advantage
  is by construction, since the oracle sees option probabilities and the judge sees text only,
  but the judge's misses go beyond that.
- **On free text, LaBSE divergence fails** (F1 0.50 / 0.31): it tracks paraphrase distance, not
  alignment-relevant change. The judge does much better, against the less reliable free-text
  labels (§5.3).
- **The adaptive lifecycle adds nothing over a single evaluator** (0.710 vs 0.719). Every agent
  reads the same deterministic signal through the same thresholds, so spawn / retain / destroy
  can change which items are examined but never a verdict. The cost advantage belongs to the
  cheap signal, not to the population.

**Routing.** Choosing, per (language × item type) cell, the judge only where its *tune* F1
beats the cheap signal's (ties go to the free signal) sends free text to the judge and keeps
forced choice on JSD. On test it beats both (F1 0.861; recall +0.10 over the judge at equal
precision) with about half the judge calls. The rule was fixed in code before scoring, but the
idea came from the test breakdown, so we report it as a follow-up system. It is the selection
the lifecycle was meant to perform; the lifecycle cannot perform it because its agents do not
differ in signal.

**Robustness: untruncated responses.** With free text at 1,536 tokens (labels from the
re-labelled round, forced-choice labels unchanged; 567 test pairs), the ranking is the same:
routed 0.917 > judge 0.875 > cheap 0.731 ≈ lifecycle 0.721, with the same routes chosen on tune.
**Caveat on Hindi free text:** 84% of those items are labelled collapsed, so flagging every
item would already score F1 ≈ 0.91, close to the judge's 0.953. Portuguese free text (21%
collapsed) is the informative cell, where the judge beats LaBSE 0.63 to 0.23, on labels with
κ = 0.10.

**Summary.** The forced-choice result is solid: cheap exact signals match or beat an LLM judge
at no LLM cost. The free-text result (judge ≫ LaBSE, so route) has the same direction in both
runs but rests on low-agreement labels.

## 5.5 Representations and steering (Tables 7 and 9)

**Table 7.** Using last-prompt-token activations for all 501 items across 29 layers, we compare
each item's English and target representations (matched minus mismatched mean-centred cosine,
"gap"). **Hindi is further from English than Portuguese at every layer**: at the most
language-neutral layer (14), gap 0.571 vs 0.737 and translation retrieval 0.45 vs 0.72. This is
the same ranking as behavioural collapse, and honesty items are the least matched. DPO changes
none of this (±0.008).

The link to behaviour appears only late. Item-level similarity does not predict which items
collapse in layers 1–19 (Spearman ρ ≈ 0); from layer 20 it does, reaching ρ ≈ −0.5 at layers
25–27 in both languages. At the last layers the last prompt token encodes the next-token
prediction, which for forced-choice items is the option letter, so part of this is by
construction. The English DPO update direction transfers less to Hindi's late layers than to
Portuguese's (0.54–0.70 vs 0.69–0.78 of its length), matching the weaker preference transfer in
§5.2; multilingual training raises Hindi transfer at every depth.

**Table 9.** Adding mean directions to the residual stream (toward English, along the DPO
update, or random, at 0.5–2× the language offset's norm) at layers 9, 14, 18, 22 and 25 reduces
collapse in no cell beyond the random control. At large magnitudes every direction, random
included, raises ACS and lowers TruthfulQA accuracy (off-distribution disruption). Together
with Table 7, this suggests collapse is item-specific and late: one mean direction cannot undo
it.

## 5.6 Collapse extends to safety (Table 8)

On XSTest machine-translated into each language (200 unsafe prompts, 250 safe look-alikes;
responses classified by Claude Haiku 5.5), Hindi is less safe and more over-refusing than
English. Portuguese is indistinguishable from English.

Because XSTest's safe prompts depend on idioms and homonyms ("murder of crows", "strangle
options", "a can of coke") that literal MT breaks in both directions, Claude Opus 5.5 checked
every translation against the English. It flagged 17% of Hindi prompts and 3% of Portuguese
ones. **We report the verified subset** (365 prompts kept in every language: 159 unsafe,
206 safe):

**Table 8.** Translation-verified subset; Hindi vs English, exact McNemar p.

| | safety (refusal of unsafe) en / hi / pt | hi − en (p) | over-refusal en / hi / pt | hi − en (p) |
|---|---|---|---|---|
| base | 0.969 / 0.792 / 0.975 | −0.176 (< 0.001) | 0.097 / 0.175 / 0.117 | +0.078 (0.020) |
| en_aligned | 0.969 / 0.843 / 0.975 | −0.126 (< 0.001) | 0.112 / 0.175 / 0.102 | +0.063 (0.035) |
| multi_aligned | 0.975 / 0.830 / 0.969 | −0.145 (< 0.001) | 0.112 / 0.214 / 0.107 | +0.102 (0.001) |

Hindi complies with 13–18 points more unsafe requests **and** refuses 6–10 points more safe
ones: worse discrimination, not a shifted threshold. On all 450 prompts the gaps are slightly
larger (safety −18 to −19, over-refusal +9 to +11), so bad translations account for at most ~5
points. Neither DPO run closes the gap. Differences between checkpoints amount to a handful of
prompts and are not tested. The language-level association with collapse (Hindi collapses
and is less safe; Portuguese neither) is the claim we can support: with two languages it is
not an item- or cell-level correlation.

## 5.7 Limitations specific to these results

- **One model, two target languages.** The Hindi/Portuguese contrast confounds resource level
  with script and language family.
- **LLM-panel oracle.** Validated against answer keys on honesty only. Free-text agreement is
  below our own gate (§5.3), and an LLM oracle may share reading habits with the LLM judge,
  which would favour System 6.
- **Machine translation.** Hand-fixed where we found systematic calques, verified on ~100 items
  per language and, for safety, on every prompt. Verified-subset tests for honesty are
  underpowered (n = 33).
- **Free-text Hindi base rate.** A high collapse rate makes F1 on that cell close to trivial
  (§5.4). We report an always-flag baseline alongside it.
- **Hindi truncation.** Even at 1,536 tokens, 27.5% of Hindi free-text responses are cut off
  (§5.3). This did not reduce collapse labels, but it is not zero.
