"""Table 6: the evaluator systems (4 single, 5 fixed, 7 adaptive) on the same cached outputs.

Owner: signals/eval track. System 6 (LLM-as-judge) is added once baselines/llm_judge.py is
wired up (D5).

Rules this follows (CLAUDE.md):
- Every system reads the same generations, embeddings, thresholds and language IDs.
- Same evaluation budget: the adaptive run goes first and its total number of evaluations
  becomes the budget of the single evaluator and the fixed team (split evenly over its agents).
- Oracle labels: the adaptive utility sees only `tune` labels; every reported score uses only
  `test` labels, counting labelled items a system never evaluated as missed.
"""

from __future__ import annotations

from mpac.data.schema import Generation, ParallelItem


def run_systems(cfg: dict, items: list[ParallelItem], gens: dict[str, dict[str, Generation]],
                oracle_tune: dict, oracle_test: dict, thresholds: dict | None = None,
                embeddings: dict | None = None, lang_ids: dict | None = None,
                registry_path=None) -> dict:
    """One model + checkpoint. Returns {"systems": name -> scores, "adaptive": summary}."""
    from mpac.agents.coordinator import LifecycleCoordinator
    from mpac.agents.registry import AgentRegistry
    from mpac.analysis.metrics import coverage, detection_scores
    from mpac.baselines.static import fixed_multi_agent, single_evaluator

    targets, objectives = cfg["languages"]["targets"], cfg["objectives"]
    ref, seed = cfg["languages"]["reference"], cfg["project"]["seed"]
    coord = LifecycleCoordinator(cfg, items, gens, oracle_tune, embeddings,
                                 registry=AgentRegistry(registry_path),
                                 thresholds=thresholds, lang_ids=lang_ids)
    summary = coord.run()
    budget = summary["total_evaluations"]
    runs = {
        "4 single evaluator": single_evaluator(items, gens, targets, ref, budget=budget, seed=seed,
                                               embeddings=embeddings, thresholds=thresholds,
                                               lang_ids=lang_ids),
        "5 fixed multi-agent": fixed_multi_agent(items, gens, targets, objectives, ref,
                                                 per_agent_budget=budget // (len(targets) * len(objectives)),
                                                 seed=seed, embeddings=embeddings,
                                                 thresholds=thresholds, lang_ids=lang_ids),
        "7 adaptive lifecycle": coord.all_evals,
    }
    systems = {}
    for name, evals in runs.items():
        cov = coverage(evals)
        systems[name] = {**detection_scores(evals, oracle_test, ref, count_unevaluated=True),
                         "n_evaluations": cov["n_evaluations"], "cells": cov["cells"],
                         "tokens": cov["tokens"]}
    summary = {k: v for k, v in summary.items() if k != "collapse_map"} | {
        "collapse_map": {f"{lang}/{obj}": v for (lang, obj), v in summary["collapse_map"].items()},
        "events": coord.registry.events}
    return {"systems": systems, "adaptive": summary, "budget": budget}


def per_cell_scores(evals, oracle_test: dict, reference: str = "en") -> dict[str, dict]:
    """Detection scores per (lang, objective) cell: where a system finds collapse and misses it."""
    from collections import defaultdict

    from mpac.analysis.metrics import detection_scores

    by_cell = defaultdict(dict)
    for (lang, iid, obj), y in oracle_test.items():
        by_cell[f"{lang}/{obj}"][(lang, iid, obj)] = y
    return {cell: detection_scores(evals, o, reference, count_unevaluated=True)
            for cell, o in sorted(by_cell.items())}


def load_signals(cfg: dict, model_id: str, checkpoint: str) -> dict:
    """Everything the lightweight evaluators read, from the Drive caches: items, generations,
    LaBSE embeddings (computed only if missing), language IDs and each language's prompt floor
    (from the saved acs_free_text table, notebook 01)."""
    from mpac.analysis.results import load_results
    from mpac.config import all_languages
    from mpac.data.loaders import load_items
    from mpac.generation.cache import GenerationCache
    from mpac.paths import subdir
    from mpac.signals.embeddings import embed_generations
    from mpac.signals.lightweight import LazyEmbedder, detect_languages

    items = load_items(cfg)
    gens = GenerationCache(subdir(cfg, "generations")).load_all(model_id, checkpoint,
                                                                all_languages(cfg))
    if not all(gens.values()):
        raise FileNotFoundError(f"no generations for {model_id}/{checkpoint}; run notebook 01")
    emb = embed_generations(gens, LazyEmbedder(), subdir(cfg, "embeddings"))
    saved = [r for r in load_results(cfg, "acs_free_text") if r["model_id"] == model_id]
    if not saved:
        raise FileNotFoundError("no acs_free_text results; run notebook 01's free-text cell")
    floor = {lang: v["prompt_floor"] for lang, v in saved[0]["data"].items()}
    return {"items": items, "gens": gens, "embeddings": emb, "floor": floor,
            "lang_ids": detect_languages(gens)}


def load_oracle(cfg: dict, model_id: str, checkpoint: str, source: str = "human",
                round_name: str = "main", signals: dict | None = None) -> tuple[dict, dict, dict]:
    """(tune labels, test labels, info). source="human": the resolved labels of
    oracle/<round_name>/; "proxy": data.oracle.proxy_oracle (pipeline check only)."""
    from mpac.data.oracle import load_round, proxy_oracle, resolve, to_oracle
    from mpac.data.splits import get_or_create_splits
    from mpac.paths import subdir

    if source == "proxy":
        s = signals or load_signals(cfg, model_id, checkpoint)
        splits = get_or_create_splits(cfg, s["items"])
        allo = proxy_oracle(s["items"], s["gens"], cfg["languages"]["reference"])
        info = {"source": "proxy (TruthfulQA answers, honesty only): NOT REPORTABLE"}
    elif source == "human":
        from mpac.data.loaders import load_items

        splits = get_or_create_splits(cfg, load_items(cfg))
        labels, adj, unresolved = [], [], 0
        for lab, ad in load_round(subdir(cfg, "oracle") / round_name).values():
            labels += lab
            adj += ad
        resolved, left = resolve(labels, adj)
        allo = to_oracle(resolved, model_id, checkpoint)
        unresolved = len(left)
        info = {"source": f"human, round {round_name}", "unresolved": unresolved}
    else:
        raise ValueError("source must be 'human' or 'proxy'")
    tune = {k: v for k, v in allo.items() if splits.get(k[1]) == "tune"}
    test = {k: v for k, v in allo.items() if splits.get(k[1]) == "test"}
    info |= {"n_tune": len(tune), "n_test": len(test), "n_collapsed_test": sum(test.values())}
    return tune, test, info


def table6(cfg: dict, model_id: str, checkpoint: str, source: str = "human",
           round_name: str = "main", embedding_margin: float = 0.15) -> dict:
    """Thresholds (floor start, tuned on tune labels), then every system on the same signals;
    saved to results/<model>/<checkpoint>/table6_<source>.json."""
    from mpac.analysis.results import save_results
    from mpac.signals.acs import item_divergences
    from mpac.signals.calibration import floor_thresholds, tune_thresholds

    s = load_signals(cfg, model_id, checkpoint)
    tune, test, info = load_oracle(cfg, model_id, checkpoint, source, round_name, s)
    if not test:
        raise ValueError(f"no test-split labels for {model_id}/{checkpoint} ({info})")
    ref, targets = cfg["languages"]["reference"], cfg["languages"]["targets"]
    start = floor_thresholds(targets, s["floor"], embedding_margin=embedding_margin)
    divs = {}
    for kind, signal in (("options", "options"), ("embedding", "embedding")):
        for lang, d in item_divergences(s["gens"], ref, s["embeddings"], signal).items():
            divs[(lang, kind)] = d
    thresholds, th_report = tune_thresholds(divs, tune, start)
    out = run_systems(cfg, s["items"], s["gens"], tune, test, thresholds, s["embeddings"],
                      s["lang_ids"])
    wrong_lang = {lang: sum(v != lang for v in ids.values())
                  for lang, ids in s["lang_ids"].items() if lang != ref}
    out |= {"oracle": info, "wrong_language_responses": wrong_lang,
            "thresholds": {f"{lang}/{kind}": v for (lang, kind), v in th_report.items()}}
    out["path"] = str(save_results(cfg, model_id, checkpoint, f"table6_{source}", out))
    return out
