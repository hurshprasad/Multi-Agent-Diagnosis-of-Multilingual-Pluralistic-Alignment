"""Synthetic forced-choice data with planted collapse, for testing the agent system without a GPU.

Lets the agents track develop and debug the coordinator (and check it finds the planted cells)
while real generations are still being produced. Never use for reported results.
"""

from __future__ import annotations

import numpy as np

from mpac.data.schema import Generation, ParallelItem


def make_synthetic(
    targets: list[str],
    objectives: list[str],
    n_items: int = 120,
    collapsed: dict[tuple[str, str], float] | None = None,
    reference: str = "en",
    n_options: int = 3,
    seed: int = 0,
):
    """Return (items, gens, oracle).

    collapsed maps (lang, objective) -> probability an item in that cell collapses. Collapsed
    items get an option distribution far from English; others get small noise.
    oracle maps (lang, item_id, objective) -> True if the item was planted as collapsed.
    """
    rng = np.random.default_rng(seed)
    collapsed = collapsed or {}
    langs = [reference, *targets]
    items, oracle = [], {}
    gens: dict[str, dict[str, Generation]] = {lang: {} for lang in langs}
    for i in range(n_items):
        obj = objectives[i % len(objectives)]
        iid = f"syn-{i:04d}"
        labels = [chr(65 + k) for k in range(n_options)]
        items.append(ParallelItem(iid, obj, {lang: f"[{lang}] prompt {iid}" for lang in langs},
                                  options={lang: labels for lang in langs}))
        ref = rng.dirichlet(np.ones(n_options) * 0.5)
        gens[reference][iid] = Generation("synthetic", "base", reference, iid, "", ref.tolist())
        for lang in targets:
            is_collapsed = rng.random() < collapsed.get((lang, obj), 0.0)
            if is_collapsed:
                # Mass moves to the option English rated least likely.
                p = np.full(n_options, 0.05)
                p[int(np.argmin(ref))] = 1.0
            else:
                p = ref + rng.normal(0, 0.03, n_options)
            p = np.clip(p, 1e-6, None)
            p = p / p.sum()
            gens[lang][iid] = Generation("synthetic", "base", lang, iid, "", p.tolist())
            oracle[(lang, iid, obj)] = bool(is_collapsed)
    return items, gens, oracle
