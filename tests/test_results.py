import json
import math

import pytest

from mpac.analysis.results import (
    accuracy_vs_reference,
    load_results,
    mcnemar_exact,
    results_table,
    save_results,
)
from mpac.config import load_config
from mpac.data.schema import Generation, ParallelItem


def test_mcnemar_exact_matches_known_values():
    # Values from the pilot run, checked against scipy.stats.binomtest.
    assert math.isclose(mcnemar_exact(37, 11), 0.000222, rel_tol=0.01)
    assert math.isclose(mcnemar_exact(6, 2), 0.289, rel_tol=0.01)
    assert mcnemar_exact(3, 3) == 1.0 and math.isnan(mcnemar_exact(0, 0))


def _setup():
    items = [ParallelItem(f"tq-{i}", "honesty", {"en": "q"}, options={"en": ["A", "B"]},
                          meta={"correct": 0}) for i in range(4)]
    items.append(ParallelItem("wvs-1", "pluralistic_value", {"en": "q"}, options={"en": ["A", "B"]}))
    g = lambda lang, i, right: Generation("m", "base", lang, i, "", [0.9, 0.1] if right else [0.1, 0.9])
    gens = {"en": {f"tq-{i}": g("en", f"tq-{i}", True) for i in range(4)} | {"wvs-1": g("en", "wvs-1", True)},
            "hi": {f"tq-{i}": g("hi", f"tq-{i}", i < 2) for i in range(4)} | {"wvs-1": g("hi", "wvs-1", False)}}
    return items, gens


def test_accuracy_vs_reference_counts_flips_on_items_with_an_answer():
    items, gens = _setup()
    acc = accuracy_vs_reference(gens, items)
    assert acc["hi"] == {"n": 4, "acc_reference": 1.0, "acc": 0.5, "lost": 2, "gained": 0,
                         "mcnemar_p": 0.5}
    assert accuracy_vs_reference(gens, items, ids={"tq-0"})["hi"]["n"] == 1


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MPAC_ROOT", str(tmp_path))
    return load_config("configs/hi_pt.yaml")


def test_save_and_load_results_across_checkpoints(cfg, tmp_path):
    p = save_results(cfg, "m", "base", "acs", {"hi": {"overall": 0.4}, "pt": {"overall": 0.2}},
                     n_items=501)
    save_results(cfg, "m", "en_aligned", "acs", {"hi": {"overall": 0.3}})
    save_results(cfg, "m", "base", "other", {"hi": {"x": 1}})
    assert p == tmp_path / "results" / "m" / "base" / "acs.json"
    rec = json.loads(p.read_text())
    assert rec["meta"]["n_items"] == 501 and rec["meta"]["saved_at"]
    assert len(load_results(cfg, "acs")) == 2 and len(load_results(cfg)) == 3
    table = results_table(cfg, "acs")
    assert table.loc[("m", "en_aligned", "hi"), "overall"] == 0.3
    assert len(table) == 3
    # Saving again replaces the table rather than adding a second copy.
    save_results(cfg, "m", "base", "acs", {"hi": {"overall": 0.5}})
    assert results_table(cfg, "acs").loc[("m", "base", "hi"), "overall"] == 0.5
