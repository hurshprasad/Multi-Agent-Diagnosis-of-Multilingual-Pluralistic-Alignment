import numpy as np

from mpac.data.schema import Generation, ParallelItem
from mpac.signals.acs import alignment_collapse_scores
from mpac.signals.embeddings import embed_generations, prompt_floor


class FakeEmbedder:
    """Stands in for LaBSE: texts with the same first word get the same unit vector."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += len(texts)
        vecs = [np.random.default_rng(sum(map(ord, t.split()[0]))).normal(size=8) for t in texts]
        return np.stack([v / np.linalg.norm(v) for v in vecs])


def _gens():
    g = lambda lang, iid, resp, probs=None: Generation("m", "base", lang, iid, resp, probs)
    return {
        "en": {"uf-1": g("en", "uf-1", "plan the trip"), "tq-1": g("en", "tq-1", "A", [0.9, 0.1]),
               "pl-1": g("en", "pl-1", "both views matter")},
        "pt": {"uf-1": g("pt", "uf-1", "plan a viagem"), "tq-1": g("pt", "tq-1", "A", [0.8, 0.2]),
               "pl-1": g("pt", "pl-1", "both sides count")},
        "hi": {"uf-1": g("hi", "uf-1", "refuse"), "tq-1": g("hi", "tq-1", "B", [0.1, 0.9]),
               "pl-1": g("hi", "pl-1", "only one view")},
    }


def test_embed_generations_skips_forced_choice_and_caches(tmp_path):
    emb = FakeEmbedder()
    vecs = embed_generations(_gens(), emb, tmp_path)
    assert set(vecs["en"]) == {"uf-1", "pl-1"}          # tq-1 is scored from option_probs
    assert (tmp_path / "m" / "base" / "hi.npz").exists()
    calls = emb.calls
    again = embed_generations(_gens(), emb, tmp_path)
    assert emb.calls == calls                            # everything came from the cache
    assert np.allclose(again["pt"]["uf-1"], vecs["pt"]["uf-1"])


def test_free_text_and_forced_choice_scores_stay_separate(tmp_path):
    gens = _gens()
    vecs = embed_generations(gens, FakeEmbedder(), tmp_path)
    objectives = {"uf-1": "helpfulness", "tq-1": "honesty", "pl-1": "pluralistic_value"}
    free = alignment_collapse_scores(gens, objectives, embeddings=vecs, signal="embedding")
    fc = alignment_collapse_scores(gens, objectives, embeddings=vecs, signal="options")
    assert set(free["pt"]) == {"helpfulness", "pluralistic_value", "overall", "n"}
    assert set(fc["pt"]) == {"honesty", "overall", "n"}
    assert free["pt"]["helpfulness"] < 1e-9 and free["hi"]["helpfulness"] > 0
    assert free["pt"]["n"] == 2 and fc["pt"]["n"] == 1


def test_prompt_floor_uses_free_text_items_only():
    items = [
        ParallelItem("uf-1", "helpfulness", {"en": "plan x", "pt": "plan y", "hi": "yojana"}),
        ParallelItem("tq-1", "honesty", {"en": "q", "pt": "z", "hi": "w"},
                     options={"en": ["A", "B"], "pt": ["A", "B"], "hi": ["A", "B"]}),
    ]
    floor = prompt_floor(items, FakeEmbedder(), ["pt", "hi"])
    assert floor["pt"] < 1e-9 and floor["hi"] > 0
