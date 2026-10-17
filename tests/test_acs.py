import math

from mpac.data.schema import Generation
from mpac.signals.acs import (
    alignment_collapse_scores,
    embedding_divergence,
    js_divergence,
    worst_language,
)


def test_js_divergence_bounds():
    assert js_divergence([0.5, 0.5], [0.5, 0.5]) < 1e-9
    assert math.isclose(js_divergence([1, 0], [0, 1]), 1.0, abs_tol=1e-6)
    d = js_divergence([0.7, 0.2, 0.1], [0.1, 0.2, 0.7])
    assert 0 < d < 1


def test_embedding_divergence():
    assert embedding_divergence([1, 0], [1, 0]) < 1e-9
    assert math.isclose(embedding_divergence([1, 0], [0, 1]), 1.0)


def _g(lang, iid, probs):
    return Generation("m", "base", lang, iid, "", probs)


def test_acs_per_language_and_objective():
    gens = {
        "en": {"a": _g("en", "a", [0.9, 0.1]), "b": _g("en", "b", [0.5, 0.5])},
        "es": {"a": _g("es", "a", [0.9, 0.1]), "b": _g("es", "b", [0.5, 0.5])},
        "de": {"a": _g("de", "a", [0.1, 0.9]), "b": _g("de", "b", [0.5, 0.5])},
    }
    objectives = {"a": "honesty", "b": "helpfulness"}
    acs = alignment_collapse_scores(gens, objectives)
    assert set(acs) == {"es", "de"}
    assert acs["es"]["overall"] < 1e-6
    assert acs["de"]["honesty"] > 0.3 and acs["de"]["helpfulness"] < 1e-6
    assert worst_language(acs) == "de"
