"""End to end on CPU: Drive layout -> oracle round -> thresholds -> all systems -> saved table."""
import csv

import pytest

from mpac.analysis.evaluators import table6
from mpac.analysis.results import save_results
from mpac.config import load_config
from mpac.data.io import write_jsonl
from mpac.data.oracle import export_round, sample_for_labelling
from mpac.data.synthetic import make_synthetic
from mpac.generation.cache import GenerationCache

TARGETS = ["hi", "pt"]
OBJECTIVES = ["helpfulness", "honesty", "pluralistic_value"]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MPAC_ROOT", str(tmp_path))
    cfg = load_config("configs/hi_pt.yaml")
    cfg["agents"].update(lifecycle_cycles=4, batch_per_cycle=20)
    return cfg


def test_table6_from_filled_sheets(cfg, tmp_path):
    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=90,
                                         collapsed={("hi", "honesty"): 0.8})
    write_jsonl(tmp_path / "items" / "items.jsonl", [it.to_dict() for it in items])
    GenerationCache(tmp_path / "generations").add(g for by in gens.values() for g in by.values())
    save_results(cfg, "synthetic", "base", "acs_free_text",
                 {"hi": {"prompt_floor": 0.14}, "pt": {"prompt_floor": 0.10}})
    rows = sample_for_labelling(items, {("synthetic", "base"): gens}, n=90, targets=TARGETS)
    paths = export_round(rows, tmp_path / "oracle" / "main", {"hi": ["a", "b"], "pt": ["c", "d"]})
    for ps in paths.values():
        (ps[0].parent / "filled").mkdir()
        for p in ps:   # both annotators agree with the planted labels
            rs = list(csv.DictReader(p.open(encoding="utf-8")))
            for r in rs:
                r["aligned"] = "yes"
                r["collapsed_vs_reference"] = "yes" if oracle[(r["lang"], r["item_id"], r["objective"])] else "no"
            with open(p.parent / "filled" / p.name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rs[0]))
                w.writeheader()
                w.writerows(rs)

    out = table6(cfg, "synthetic", "base")
    assert out["oracle"]["n_test"] > 0 and out["oracle"]["unresolved"] == 0
    assert set(out["systems"]) == {"4 single evaluator", "5 fixed multi-agent", "7 adaptive lifecycle"}
    assert out["systems"]["7 adaptive lifecycle"]["f1"] > 0.5
    assert (tmp_path / "results" / "synthetic" / "base" / "table6_human.json").exists()
