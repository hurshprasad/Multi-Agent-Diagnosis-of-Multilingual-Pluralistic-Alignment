"""Checkpoint vs checkpoint: did DPO change collapse? (base, en_aligned, multi_aligned)

Owner: signals/eval track.

Rows are keyed by checkpoint names a and b; mean_a / mean_b are the two checkpoints' values.
The per-checkpoint tables (results.py) compare each language against English. This module
compares two checkpoints on the same items, item by item:
- paired_difference()   change in mean ACS (b − a) over items scored in both, with a paired
                        bootstrap 95% CI and a sign-flip permutation p-value.
- correctness_flips()   TruthfulQA items right in a but wrong in b (lost) and the reverse
                        (gained), with an exact McNemar test.
- compare_checkpoints() every metric, language and pair of checkpoints, read from the
                        generation and embedding caches (no GPU needed once embedded).
Negative ACS diff = less collapse in b. Positive accuracy diff = b is more truthful.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from mpac.analysis.results import mcnemar_exact


def paired_difference(a: dict[str, float], b: dict[str, float], n_boot: int = 10000,
                      seed: int = 0) -> dict[str, float]:
    """Mean of b − a over shared item ids, bootstrap CI and two-sided permutation p."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    d = np.array([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    flips = (rng.choice([-1.0, 1.0], (n_boot, len(d))) * d).mean(axis=1)
    p = (1 + np.sum(np.abs(flips) >= abs(d.mean()) - 1e-12)) / (n_boot + 1)
    return {"n": len(ids), "mean_a": float(np.mean([a[i] for i in ids])),
            "mean_b": float(np.mean([b[i] for i in ids])), "diff": float(d.mean()),
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
            "p": float(min(1.0, p))}


def _correct(gens_lang: dict, items_by_id: dict) -> dict[str, bool]:
    return {i: int(np.argmax(g.option_probs)) == items_by_id[i].meta["correct"]
            for i, g in gens_lang.items()
            if g.option_probs is not None and i in items_by_id and "correct" in items_by_id[i].meta}


def correctness_flips(gens_a: dict, gens_b: dict, items: list) -> dict[str, float]:
    """One language: accuracy of a and b on shared answerable items, flips and McNemar p."""
    by_id = {it.item_id: it for it in items}
    ra, rb = _correct(gens_a, by_id), _correct(gens_b, by_id)
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return {"n": 0}
    lost = sum(ra[i] and not rb[i] for i in ids)
    gained = sum(rb[i] and not ra[i] for i in ids)
    acc_a, acc_b = np.mean([ra[i] for i in ids]), np.mean([rb[i] for i in ids])
    return {"n": len(ids), "mean_a": float(acc_a), "mean_b": float(acc_b), "diff": float(acc_b - acc_a),
            "lost": lost, "gained": gained, "p": mcnemar_exact(lost, gained)}


def _restrict(divs: dict, objective: dict, obj: str) -> dict:
    return {i: v for i, v in divs.items() if obj == "overall" or objective.get(i) == obj}


def checkpoint_rows(gens: dict, emb: dict, items: list, reference: str = "en",
                    pairs: list[tuple[str, str]] | None = None, n_boot: int = 10000,
                    seed: int = 0) -> list[dict]:
    """gens: checkpoint -> lang -> item_id -> Generation; emb: checkpoint -> lang -> item_id ->
    vector (free-text responses). One row per (a, b, lang, metric)."""
    from mpac.signals.acs import item_divergences

    objective = {it.item_id: it.objective for it in items}
    pairs = pairs or list(combinations(gens, 2))
    divs = {ck: {"forced_choice": item_divergences(gens[ck], reference, signal="options"),
                 "free_text": item_divergences(gens[ck], reference, emb.get(ck), signal="embedding")}
            for ck in gens}
    rows = []
    for a, b in pairs:
        for lang in gens[a]:
            if lang != reference:
                for kind in ("forced_choice", "free_text"):
                    da, db = divs[a][kind].get(lang, {}), divs[b][kind].get(lang, {})
                    for obj in sorted({objective.get(i, "unknown") for i in da}) + ["overall"]:
                        sa, sb = (_restrict(d, objective, obj) for d in (da, db))
                        rows.append({"a": a, "b": b, "lang": lang, "metric": f"acs_{kind}",
                                     "objective": obj, **paired_difference(sa, sb, n_boot, seed)})
            if lang in gens[b]:
                rows.append({"a": a, "b": b, "lang": lang, "metric": "truthfulqa_accuracy",
                             "objective": "honesty",
                             **correctness_flips(gens[a][lang], gens[b][lang], items)})
    return rows


def compare_checkpoints(cfg: dict, model_id: str, checkpoints: list[str], embedder=None,
                        verified_only: bool = False, n_boot: int = 10000):
    """Every metric for each pair of checkpoints that has generations, as a DataFrame, saved to
    results/<model>/comparison/checkpoint_comparison[_verified].json. embedder is only needed
    for responses not yet embedded (notebook 01's free-text cell caches them)."""
    import pandas as pd

    from mpac.analysis.results import save_results
    from mpac.config import all_languages
    from mpac.data.loaders import load_items
    from mpac.generation.cache import GenerationCache
    from mpac.paths import subdir
    from mpac.signals.embeddings import embed_generations

    items = load_items(cfg, verified_only=verified_only)
    langs_of = {it.item_id: set(it.prompts) for it in items}   # verified: only accepted languages
    cache = GenerationCache(subdir(cfg, "generations"))
    gens = {}
    for ck in checkpoints:
        g = cache.load_all(model_id, ck, all_languages(cfg))
        if all(g.values()):
            gens[ck] = {lang: {i: x for i, x in by.items() if lang in langs_of.get(i, ())}
                        for lang, by in g.items()}
    emb = {ck: embed_generations(g, embedder, subdir(cfg, "embeddings")) for ck, g in gens.items()}
    rows = checkpoint_rows(gens, emb, items, cfg["languages"]["reference"],
                           [(a, b) for a, b in combinations(checkpoints, 2) if a in gens and b in gens],
                           n_boot, cfg["project"]["seed"])
    name = "checkpoint_comparison" + ("_verified" if verified_only else "")
    save_results(cfg, model_id, "comparison", name, {"rows": rows}, checkpoints=list(gens),
                 n_boot=n_boot)
    return pd.DataFrame(rows).set_index(["a", "b", "lang", "metric", "objective"])
