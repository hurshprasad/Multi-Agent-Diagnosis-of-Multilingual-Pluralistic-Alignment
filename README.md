# Multi-Agent Diagnosis of Multilingual Pluralistic Alignment Collapse

Does pluralistic alignment that holds in English weaken, become inconsistent, or contradict
itself in other languages? And can an adaptive population of cheap evaluator agents find
*where* it collapses (language × objective × layer) more efficiently than a single evaluator
or an LLM-as-judge?

**Languages:** English (reference) → Portuguese, Spanish, Italian, German, French.
The original proposal targeted South Asian languages; scope moved to these five because the
target models support them well (see `docs/decisions.md`, D0).

**Models:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (small variants for smoke tests).

## Run on Colab

| Notebook | Runtime | What it does |
|---|---|---|
| [`00_setup_and_smoke`](https://colab.research.google.com/github/hurshprasad/Multi-Agent-Diagnosis-of-Multilingual-Pluralistic-Alignment/blob/main/notebooks/00_setup_and_smoke.ipynb) | CPU | Install, mount Drive, run tests, run the agent lifecycle on synthetic data |
| [`01_generate`](https://colab.research.google.com/github/hurshprasad/Multi-Agent-Diagnosis-of-Multilingual-Pluralistic-Alignment/blob/main/notebooks/01_generate.ipynb) | GPU | Generate model outputs into the shared cache on Drive |
| [`02_phase0_decisions`](https://colab.research.google.com/github/hurshprasad/Multi-Agent-Diagnosis-of-Multilingual-Pluralistic-Alignment/blob/main/notebooks/02_phase0_decisions.ipynb) | T4 GPU | Evidence for D1 (datasets), D4 (MT model comparison), D7 (oracle sheet format) |
| [`03_build_dataset`](https://colab.research.google.com/github/hurshprasad/Multi-Agent-Diagnosis-of-Multilingual-Pluralistic-Alignment/blob/main/notebooks/03_build_dataset.ipynb) | T4 GPU | Build the parallel evaluation set: fetch sources, translate, assemble items, verification sheets, tune/test split |
| [`04_train_dpo`](https://colab.research.google.com/github/hurshprasad/Multi-Agent-Diagnosis-of-Multilingual-Pluralistic-Alignment/blob/main/notebooks/04_train_dpo.ipynb) | L4 / A100 GPU | DPO + LoRA for en_aligned and multi_aligned (same budget), plus the held-out preference check |
| [`05_oracle_and_evaluators`](https://colab.research.google.com/github/hurshprasad/Multi-Agent-Diagnosis-of-Multilingual-Pluralistic-Alignment/blob/main/notebooks/05_oracle_and_evaluators.ipynb) | CPU | Oracle labelling sheets and agreement, then Table 6: evaluator systems on the same outputs |

Every notebook starts with the same setup cell: clone or pull the repo, `pip install -e`,
mount Drive. All artifacts go to `MyDrive/mpac/`, so nothing is lost when a runtime resets,
and generation resumes where it stopped.

## Run locally

```bash
pip install -e ".[dev]"          # core + tests (CPU is fine)
pip install -e ".[gpu,signals]"  # generation, training, interpretability
pytest -q
```

## Pipeline

```
data ──► generation cache ──► evaluator systems ──► analysis
 │        (model × checkpoint      ├ single evaluator
 │         × language × item)      ├ fixed multi-agent
 │                                 ├ LLM-as-judge multi-agent (EvoAgent-style)
 └► DPO checkpoints                └ adaptive lifecycle (spawn / select / retain / destroy)
    (base, en_aligned,                         │
     multi_aligned)                            ▼
                               flagged cells ──► interp (activations, interventions)
                                             └─► safety (downstream consequences)
```

All evaluator systems read **the same cached outputs**; nothing downstream of generation
calls a model to produce responses. That is what makes the evaluator comparison controlled.

## Layout

```
configs/            default.yaml, smoke.yaml
notebooks/          Colab entry points
src/mpac/
  data/             schema, loaders, translation, synthetic test data
  generation/       shared cache + transformers backend (4-bit for T4)
  training/         DPO + LoRA for en_aligned / multi_aligned        (stub)
  signals/          Alignment Collapse Score, cheap signals
  agents/           AgentSpec/State, registry, lifecycle coordinator
  baselines/        single, fixed, LLM-as-judge                       (judge: stub)
  interp/           activation analysis and steering                  (stub)
  safety/           downstream safety evaluation                      (stub)
  analysis/         detection metrics vs oracle, coverage, cost
docs/decisions.md   open questions and decisions
tests/
```

**Working now:** config, Drive-aware paths, generation cache, generation backend, ACS,
agent registry, adaptive coordinator, single and fixed baselines, detection metrics,
synthetic data, tests. **Phase 0 (proposed, see `docs/decisions.md`):** tune/test splits, oracle labelling tools, MT candidates + round-trip comparison. Dataset build (`data/build.py`: PLURAL / TruthfulQA / UltraFeedback → translated parallel items) and translation verification. **Stubs with TODOs:** DPO, LLM judge,
interpretability, safety.

## Team

Hursh, Araz, Fernando, Apraajita. Track owners are listed in `CLAUDE.md`.

## License

MPL-2.0
