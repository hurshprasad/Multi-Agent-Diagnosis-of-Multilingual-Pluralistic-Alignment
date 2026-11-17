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

    import pandas as pd
    review = pd.read_csv(out["review_path"])
    assert len(review) == 90 * len(TARGETS)          # one row per (language, item)
    assert {"oracle_collapsed", "divergence", "threshold", "systems_differ",
            "7 adaptive lifecycle | flagged", "4 single evaluator | n_evals"} <= set(review.columns)
    assert out["n_pairs_systems_differ"] == int(review["systems_differ"].sum())


def test_table6_with_llm_judge(cfg, tmp_path):
    from mpac.analysis.evaluators import make_judge
    from test_llm_judge import oracle_judge

    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=60,
                                         collapsed={("hi", "honesty"): 0.8})
    write_jsonl(tmp_path / "items" / "items.jsonl", [it.to_dict() for it in items])
    GenerationCache(tmp_path / "generations").add(g for by in gens.values() for g in by.values())
    save_results(cfg, "synthetic", "base", "acs_free_text",
                 {"hi": {"prompt_floor": 0.14}, "pt": {"prompt_floor": 0.10}})
    rows = sample_for_labelling(items, {("synthetic", "base"): gens}, n=60, targets=TARGETS)
    paths = export_round(rows, tmp_path / "oracle" / "main", {"hi": ["a", "b"], "pt": ["c", "d"]})
    for ps in paths.values():
        (ps[0].parent / "filled").mkdir()
        for p in ps:
            rs = list(csv.DictReader(p.open(encoding="utf-8")))
            for r in rs:
                r["aligned"] = "yes"
                r["collapsed_vs_reference"] = "yes" if oracle[(r["lang"], r["item_id"], r["objective"])] else "no"
            with open(p.parent / "filled" / p.name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rs[0]))
                w.writeheader()
                w.writerows(rs)

    out = table6(cfg, "synthetic", "base", judge=oracle_judge(oracle))
    six = out["systems"]["6 LLM-judge multi-agent"]
    assert six["tokens"] > 0 and out["systems"]["7 adaptive lifecycle"]["tokens"] == 0
    assert six["precision"] == 1.0              # the fake judge is the oracle
    assert list(out["systems"]) == sorted(out["systems"])

    cfg["baselines"]["llm_judge"]["model"] = cfg["oracle"]["llm_panel"]["annotators"]["opus"]
    with pytest.raises(ValueError, match="oracle panel"):
        make_judge(cfg)


def test_route_by_cell_chooses_on_tune_and_scores_on_test():
    from mpac.analysis.evaluators import CHEAP, JUDGE, route_by_cell

    def row(lang, sig, split, y, cheap, judge):
        return {"lang": lang, "signal": sig, "split": split, "oracle_collapsed": str(y),
                f"{CHEAP} | flagged": str(cheap), f"{JUDGE} | flagged": str(judge),
                f"{JUDGE} | n_evals": "1"}
    rows = []
    for split in ("tune", "test"):
        # options: the cheap signal is right, the judge misses half; embedding: the reverse
        rows += [row("hi", "options", split, True, True, i % 2 == 0) for i in range(4)]
        rows += [row("hi", "options", split, False, False, False) for _ in range(4)]
        rows += [row("hi", "embedding", split, True, i % 2 == 0, True) for i in range(4)]
        rows += [row("hi", "embedding", split, False, True, False) for _ in range(4)]
    rows.append(row("hi", "embedding", "", "", True, True))   # unlabelled: costs, isn't scored
    out = route_by_cell(rows, judge_tokens=1000.0)
    assert out["route"] == {"hi/embedding": JUDGE, "hi/options": CHEAP}
    assert out["test"]["routed"]["f1"] == 1.0
    assert out["test"]["routed"]["f1"] > max(out["test"][CHEAP]["f1"], out["test"][JUDGE]["f1"])
    assert out["pairs_to_judge"] == {"routed": 17, "all": 33}
    assert out["tokens_estimate"]["routed"] == 1000.0 / 33 * 17
    # a tie on tune keeps the zero-cost signal
    tie = [row("pt", "options", "tune", True, True, True), row("pt", "options", "test", True, True, False)]
    assert route_by_cell(tie)["route"] == {"pt/options": CHEAP}


def _fill(paths, label):
    """Both annotators of every sheet answer collapsed_vs_reference = label(row)."""
    for ps in paths.values():
        (ps[0].parent / "filled").mkdir()
        for p in ps:
            rs = list(csv.DictReader(p.open(encoding="utf-8")))
            for r in rs:
                r["aligned"] = "yes"
                r["collapsed_vs_reference"] = "yes" if label(r) else "no"
            with open(p.parent / "filled" / p.name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rs[0]))
                w.writeheader()
                w.writerows(rs)


def test_long_checkpoint_takes_free_text_labels_from_its_round_and_forced_choice_from_main(cfg, tmp_path):
    from dataclasses import replace

    from mpac.analysis.evaluators import load_oracle
    from mpac.generation.generate import copy_forced_choice, weights_checkpoint
    from mpac.training.dpo import adapter_dir

    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=60)
    # synthetic items are all forced choice: make the helpfulness ones free text
    items = [replace(it, options=None) if it.objective == "helpfulness" else it for it in items]
    free = {it.item_id for it in items if it.options is None}
    gens = {l: {i: replace(g, option_probs=None) if i in free else g for i, g in by.items()}
            for l, by in gens.items()}
    fc = {it.item_id for it in items if it.is_forced_choice}
    assert fc and len(fc) < len(items)
    write_jsonl(tmp_path / "items" / "items.jsonl", [it.to_dict() for it in items])
    cache = GenerationCache(tmp_path / "generations")
    cache.add(g for by in gens.values() for g in by.values())

    # _long: forced choice copied unchanged, free text re-generated (here: new text)
    langs = ["en", *TARGETS]
    n_fc = copy_forced_choice(cache, "synthetic", "base", "base_long", langs)
    assert n_fc == sum(g.option_probs is not None for by in gens.values() for g in by.values())
    assert copy_forced_choice(cache, "synthetic", "base", "base_long", langs) == 0   # idempotent
    cache.add(replace(g, checkpoint="base_long", response=g.response + " (longer)")
              for by in gens.values() for g in by.values() if g.option_probs is None)
    long = cache.load_all("synthetic", "base_long", langs)
    assert all(long[l][i].option_probs == gens[l][i].option_probs for l in langs for i in fc if i in gens[l])
    assert weights_checkpoint("base_long") == "base" and adapter_dir(cfg, "m", "base_long") is None
    assert adapter_dir(cfg, "m", "en_aligned_long") == adapter_dir(cfg, "m", "en_aligned")

    # llm_main labels base (planted labels); llm_long labels base_long's free text: all "yes"
    _fill(export_round(sample_for_labelling(items, {("synthetic", "base"): gens}, 10**9, TARGETS),
                       tmp_path / "oracle" / "llm_main", {"hi": ["a", "b"], "pt": ["c", "d"]}),
          lambda r: oracle[(r["lang"], r["item_id"], r["objective"])])
    free_long = {l: {i: g for i, g in by.items() if g.option_probs is None} for l, by in long.items()}
    _fill(export_round(sample_for_labelling(items, {("synthetic", "base_long"): free_long}, 10**9, TARGETS),
                       tmp_path / "oracle" / "llm_long", {"hi": ["a", "b"], "pt": ["c", "d"]}),
          lambda r: True)

    tune, test, info = load_oracle(cfg, "synthetic", "base_long", "llm", "llm_long",
                                   forced_choice_round="llm_main")
    allo = tune | test
    assert {k[1] for k in allo} == {it.item_id for it in items}
    assert all(v for k, v in allo.items() if k[1] not in fc)                       # from llm_long
    assert all(v == oracle[k] for k, v in allo.items() if k[1] in fc)             # from llm_main
    assert "llm_main (forced choice)" in info["source"]
    # without forced_choice_round, base_long has only its own (free-text) labels
    t2, s2, _ = load_oracle(cfg, "synthetic", "base_long", "llm", "llm_long")
    assert not {k[1] for k in t2 | s2} & fc
