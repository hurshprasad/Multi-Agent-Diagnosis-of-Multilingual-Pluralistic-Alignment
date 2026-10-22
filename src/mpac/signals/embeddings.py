"""Cross-lingual response embeddings for free-text ACS, cached next to the generations.

Owner: signals/eval track. Needs the `signals` extra (sentence-transformers).

- embed_generations()  LaBSE vectors for every free-text response (items without
                       option_probs), cached to <root>/embeddings/<model>/<checkpoint>/<lang>.npz
                       so a Colab reset doesn't re-encode. Forced-choice items are skipped:
                       they are scored from option probabilities.
- prompt_floor()       LaBSE divergence between the English prompt and its translation, per
                       language. Translations mean the same thing, so this is the divergence
                       LaBSE reports for "no change in meaning" in that language pair. Read
                       free-text ACS relative to it: LaBSE puts some language pairs further
                       apart than others even for perfect translations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mpac.data.schema import Generation, ParallelItem
from mpac.signals.acs import embedding_divergence


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return dict(zip(data["ids"].tolist(), data["vecs"]))


def embed_generations(gens: dict[str, dict[str, Generation]], embedder, cache_dir: str | Path,
                      batch_size: int = 64) -> dict[str, dict[str, np.ndarray]]:
    """lang -> item_id -> unit vector for free-text responses, encoding only what isn't cached.

    gens: lang -> item_id -> Generation (one model and checkpoint, as from cache.load_all).
    embedder: anything with encode(list[str]) -> array, e.g. lightweight.Embedder.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for lang, by_item in gens.items():
        free = {i: g for i, g in by_item.items() if g.option_probs is None and g.response.strip()}
        if not free:
            out[lang] = {}
            continue
        g0 = next(iter(free.values()))
        path = Path(cache_dir) / g0.model_id / g0.checkpoint / f"{lang}.npz"
        have = _load_npz(path)
        todo = [i for i in free if i not in have]
        for k in range(0, len(todo), batch_size):
            ids = todo[k : k + batch_size]
            for i, v in zip(ids, np.asarray(embedder.encode([free[i].response for i in ids]))):
                have[i] = v
        if todo:
            path.parent.mkdir(parents=True, exist_ok=True)
            ids = sorted(have)
            np.savez(path, ids=np.array(ids), vecs=np.stack([have[i] for i in ids]))
        out[lang] = {i: have[i] for i in free}
    return out


def prompt_floor(items: list[ParallelItem], embedder, targets: list[str],
                 reference: str = "en") -> dict[str, float]:
    """Mean LaBSE divergence between the reference prompt and each target-language prompt,
    over free-text items: the score a response would get if it only changed language."""
    free = [it for it in items if not it.is_forced_choice and reference in it.prompts]
    ref = np.asarray(embedder.encode([it.prompts[reference] for it in free]))
    out = {}
    for lang in targets:
        pairs = [(r, it) for r, it in zip(ref, free) if lang in it.prompts]
        tgt = np.asarray(embedder.encode([it.prompts[lang] for _, it in pairs]))
        out[lang] = float(np.mean([embedding_divergence(r, t) for (r, _), t in zip(pairs, tgt)]))
    return out
