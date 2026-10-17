# Decisions and open questions

Record a decision here when the team settles it: what was chosen, why, and the date.

| ID | Question | Status |
|---|---|---|
| D0 | Language scope | **Decided:** en → pt, es, it, de, fr |
| D1 | Which dataset is "PLURAL"? | Open |
| D2 | Cheap signals and ACS definition | Proposed |
| D3 | Lifecycle utility and thresholds | Proposed |
| D4 | Machine translation model | Open |
| D5 | LLM judge model | Open |
| D6 | Safety / jailbreak benchmarks for the new languages | Open |
| D7 | Oracle labelling protocol | Open |

## D0 · Language scope (decided)

Moved from Hindi / Bengali / Tamil / Telugu / Marathi / Urdu to Portuguese / Spanish /
Italian / German / French because the target models support these languages well.

Consequences to keep in mind:
- The research question shifts from low-resource collapse to collapse *despite* good
  language support. Arguably cleaner, because pretraining weakness is a smaller confound, but
  the paper's framing and title need updating.
- IndicSafe / IndicJR no longer apply (see D6).
- Optional: keep one weaker-supported language as a contrast condition if compute allows.

## D1 · Dataset (open)

The proposal names "PLURAL". Confirm the exact dataset and Hugging Face id. Requirements:
prompts mappable to helpfulness / honesty / pluralistic value; preference pairs usable for
DPO; ideally a forced-choice subset (see D2).

## D2 · Cheap signals and ACS (proposed)

Implemented in `signals/acs.py`:
- **Forced-choice items:** Jensen-Shannon divergence between English and target-language
  option distributions read from logits. Exact, reproducible, no judge. Recommended: make a
  substantial share of the eval set forced-choice (opinion-style questions with fixed options).
- **Free-text items:** 1 − cosine similarity of LaBSE embeddings of the two responses.
- **Also planned:** language-ID check (catch silent switching to English), refusal detector.

Open: is embedding divergence a valid proxy for *alignment* divergence, or just
paraphrase distance? Validate against oracle labels before relying on it.

## D3 · Lifecycle utility (proposed)

Implemented in `agents/coordinator.py`:
`utility = 0.8 × oracle accuracy on labelled items in the batch + 0.2 × novelty`, where
novelty is the share of an agent's collapse flags that no other agent raised.
Destroy after 2 consecutive cycles below 0.6, or if redundant with a same-scope agent
(> 0.95 verdict agreement). Spawn a specialist for any (language, objective) cell whose mean
divergence exceeds 0.15. Population cap 24.

Open: the oracle signal depends on how many labelled items fall in each batch. If labels are
sparse, consider a label-free utility (e.g. stability of the agent's divergence estimate)
and keep oracle accuracy for evaluation only.

## D4 · Translation model (open)

Pick one public MT model and fix it for the whole study. Record the version.

## D5 · LLM judge (open)

Choose the judge model for the LLM-as-judge baseline. It should be strong in all six
languages and affordable for the full evaluation budget.

## D6 · Safety benchmarks (open)

Need public safety and jailbreak benchmarks covering pt / es / it / de / fr, plus an
over-refusal set. Check language coverage and licence before committing.

## D7 · Oracle labels (open)

Human labels for (generation, objective): `aligned` and `collapsed_vs_reference`
(`data/schema.py: OracleLabel`). Decide: number of items (config: 300), annotators per item,
agreement measure, and whether native or fluent speakers label each language.
