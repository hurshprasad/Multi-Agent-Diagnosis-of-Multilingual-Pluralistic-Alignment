import math

import numpy as np

from mpac.analysis.compare import checkpoint_rows, correctness_flips, paired_difference
from mpac.data.schema import Generation, ParallelItem


def test_paired_difference_detects_a_consistent_shift_and_not_noise():
    rng = np.random.default_rng(0)
    a = {f"i{k}": float(v) for k, v in enumerate(rng.uniform(0, 1, 200))}
    shifted = paired_difference(a, {i: v - 0.05 for i, v in a.items()}, n_boot=2000)
    assert math.isclose(shifted["diff"], -0.05) and shifted["p"] < 0.01
    assert shifted["ci_low"] <= -0.05 <= shifted["ci_high"] + 1e-9
    noise = paired_difference(a, {i: v + rng.normal(0, 0.1) for i, v in a.items()}, n_boot=2000)
    assert noise["p"] > 0.01 and noise["ci_low"] < 0 < noise["ci_high"]
    assert paired_difference(a, {"other": 1.0}) == {"n": 0}


def _gen(ck, lang, i, probs=None, text="x"):
    return Generation("m", ck, lang, i, text, probs)


def test_correctness_flips_counts_items_right_in_a_wrong_in_b():
    items = [ParallelItem(f"tq-{k}", "honesty", {"en": "q"}, options={"en": ["A", "B"]},
                          meta={"correct": 0}) for k in range(4)]
    a = {f"tq-{k}": _gen("base", "hi", f"tq-{k}", [0.9, 0.1]) for k in range(4)}
    b = {f"tq-{k}": _gen("x", "hi", f"tq-{k}", [0.9, 0.1] if k else [0.1, 0.9]) for k in range(4)}
    r = correctness_flips(a, b, items)
    assert (r["n"], r["lost"], r["gained"], r["diff"]) == (4, 1, 0, -0.25)


def test_checkpoint_rows_cover_both_signals_objectives_and_truthfulqa():
    items = [ParallelItem("tq-0", "honesty", {"en": "q"}, options={"en": ["A", "B"]}, meta={"correct": 0}),
             ParallelItem("uf-0", "helpfulness", {"en": "q"})]
    def ck(name, hi_probs, hi_vec):
        g = {"en": {"tq-0": _gen(name, "en", "tq-0", [0.9, 0.1]), "uf-0": _gen(name, "en", "uf-0")},
             "hi": {"tq-0": _gen(name, "hi", "tq-0", hi_probs), "uf-0": _gen(name, "hi", "uf-0")}}
        return g, {"en": {"uf-0": np.array([1.0, 0.0])}, "hi": {"uf-0": np.array(hi_vec)}}
    (gb, eb), (ga, ea) = ck("base", [0.2, 0.8], [0.0, 1.0]), ck("en_aligned", [0.9, 0.1], [1.0, 0.0])
    rows = checkpoint_rows({"base": gb, "en_aligned": ga}, {"base": eb, "en_aligned": ea}, items,
                           n_boot=200)
    by = {(r["lang"], r["metric"], r["objective"]): r for r in rows}
    assert by[("hi", "acs_forced_choice", "honesty")]["diff"] < 0
    assert math.isclose(by[("hi", "acs_free_text", "helpfulness")]["diff"], -1.0)
    assert by[("hi", "truthfulqa_accuracy", "honesty")]["gained"] == 1
    assert by[("en", "truthfulqa_accuracy", "honesty")]["diff"] == 0
