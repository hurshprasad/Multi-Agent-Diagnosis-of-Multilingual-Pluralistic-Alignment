# CLAUDE.md

Context for Claude Code (and humans) working in this repo.

## The study in one paragraph

Pluralistic alignment is evaluated as if a model had one behavioural policy, but it may
behave differently when the same request is asked in another language. We measure
**multilingual alignment collapse** (English reference vs pt / es / it / de / fr) for
helpfulness, honesty and pluralistic value, and test whether an **adaptive population of
lightweight evaluator agents** (spawn / select / retain / destroy, driven by cheap signals)
diagnoses where collapse happens as well as an LLM-as-judge population at much lower cost.
Then we check whether flagged cells show representation-level divergence (and whether
interventions change it) and whether collapse predicts downstream safety problems.

Proposal: `Experiment.docx` in the team's Claude Project. Open questions: `docs/decisions.md`.
Results so far: `docs/results.md`.

## The seven systems

1. base model  2. English-only aligned (DPO)  3. multilingual-aligned control (same budget)
4. single evaluator  5. fixed specialist multi-agent  6. LLM-as-judge multi-agent
7. adaptive lifecycle multi-agent (ours)

## Rules that keep the experiment valid

- **One generation cache.** Systems 4–7 read `generation/cache.py` outputs. Never generate
  inside an evaluator. Same outputs, same prompts, same item order, same evaluation budget.
- **en_aligned and multi_aligned get identical training budgets** (steps, LoRA config,
  seed, total number of pairs).
- **Evaluator decisions use cheap signals.** Any LLM call belongs in `baselines/llm_judge.py`
  and must record `cost_tokens`.
- **Oracle labels are never used to tune thresholds on the test split.** Hold out a tuning
  subset.
- **Collapse vs reference ≠ target-vs-target disagreement.** `Evaluation.compared_to`
  separates them; detection metrics only count comparisons against English.
- **Translation errors are a confound.** Report key results on the manually verified subset too.
- **Nothing on disk only in a Colab VM.** Use `paths.subdir(cfg, ...)`, which resolves to Drive.

## Code conventions

- Python ≥ 3.10, package code in `src/mpac/`, configs in YAML, records as dataclasses ↔ JSONL.
- Heavy imports (torch, transformers, sentence-transformers) go **inside** functions so
  `pip install -e .` plus `pytest` works on CPU without the `gpu` extra.
- Every new component needs a CPU test, using `data/synthetic.py` where a model would be needed.
- Run `pytest -q` before pushing.

## Colab

Notebooks in `notebooks/` are thin: setup cell, then calls into `src/mpac`. Put logic in the
package, not the notebook. T4 (16 GB): small variants, or 7-8B in 4-bit for generation only.
DPO on 7-8B needs L4/A100.

## Tracks and owners

Fill in names and keep this current.

| Track | Modules | Owner |
|---|---|---|
| Data + training | `data/`, `generation/`, `training/` | TBD |
| Signals + eval | `signals/`, `baselines/`, `analysis/` | TBD |
| Agents | `agents/` | TBD |
| Interp + safety | `interp/`, `safety/` | TBD |
