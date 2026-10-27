import csv
from collections import Counter

import pytest

from mpac.data.oracle import (
    agreement,
    cohen_kappa,
    export_sheets,
    read_sheet,
    resolve,
    sample_for_labelling,
    to_oracle,
)
from mpac.data.splits import assign_splits
from mpac.data.synthetic import make_synthetic

TARGETS, OBJECTIVES = ["es", "de"], ["helpfulness", "honesty", "pluralistic_value"]


def _setup(n_items=60):
    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=n_items,
                                         collapsed={("de", "honesty"): 0.8})
    return items, {("synthetic", "base"): gens}, oracle


def _fill(path, labels_for):
    """Simulate an annotator: fill the label columns of a sheet in place."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    for r in rows:
        r["aligned"], r["collapsed_vs_reference"] = labels_for(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_sample_is_spread_over_cells():
    items, gens, _ = _setup()
    rows = sample_for_labelling(items, gens, n=12, targets=TARGETS, splits=assign_splits(items))
    cells = Counter((r["lang"], r["objective"]) for r in rows)
    assert len(rows) == 12 and len(cells) == 6 and set(cells.values()) == {2}
    assert all(r["split"] in {"tune", "test"} and r["ref_option_probs"] for r in rows)


def test_round_trip_agreement_resolve_and_oracle(tmp_path):
    items, gens, oracle = _setup()
    splits = assign_splits(items)
    rows = sample_for_labelling(items, gens, n=30, targets=TARGETS, splits=splits)
    a_path, b_path = export_sheets(rows, tmp_path, ["ann_a", "ann_b"])
    truth = lambda r: oracle[(r["lang"], r["item_id"], r["objective"])]
    _fill(a_path, lambda r: ("yes", "yes" if truth(r) else "no"))
    # Annotator b disagrees on collapse for one item, and writes labels in German.
    odd = rows[0]["item_id"], rows[0]["lang"]
    flip = lambda r: truth(r) != ((r["item_id"], r["lang"]) == odd)
    _fill(b_path, lambda r: ("ja", "ja" if flip(r) else "nein"))
    labels = read_sheet(a_path) + read_sheet(b_path)
    assert len(labels) == 60

    agr = agreement(labels)
    assert agr["aligned"]["raw_agreement"] == 1.0
    assert agr["collapsed_vs_reference"]["raw_agreement"] == pytest.approx(29 / 30)
    assert agr["collapsed_vs_reference"]["n_items"] == 30

    resolved, unresolved = resolve(labels)
    assert len(resolved) == 29 and len(unresolved) == 1
    adjudicated = [lab for lab in read_sheet(a_path)
                   if (lab.item_id, lab.lang) == odd]
    resolved, unresolved = resolve(labels, adjudicated)
    assert len(resolved) == 30 and not unresolved

    full = to_oracle(resolved, "synthetic", "base")
    assert all(full[k] == oracle[k] for k in full)
    tune = to_oracle(resolved, "synthetic", "base", splits, "tune")
    assert tune and all(splits[k[1]] == "tune" for k in tune)
    assert not to_oracle(resolved, "synthetic", "en_aligned")
    with pytest.raises(ValueError):
        to_oracle(resolved, "synthetic", "base", split="tune")


def test_blank_rows_skipped_and_bad_values_rejected(tmp_path):
    items, gens, _ = _setup(12)
    rows = sample_for_labelling(items, gens, n=4, targets=TARGETS)
    (path,) = export_sheets(rows, tmp_path, ["ann"])
    assert read_sheet(path) == []
    _fill(path, lambda r: ("maybe", "no"))
    with pytest.raises(ValueError):
        read_sheet(path)


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([True, True], [True, True]) == 1.0


def test_export_round_per_language_and_load_filled(tmp_path):
    from mpac.data.oracle import export_round, load_round, round_keys

    items, gens, _ = _setup()
    rows = sample_for_labelling(items, gens, n=12, targets=TARGETS)
    langs = sorted({r["lang"] for r in rows})
    paths = export_round(rows, tmp_path / "pilot", {lang: [f"{lang}_a", f"{lang}_b"] for lang in langs})
    for lang in langs:
        assert {p.name for p in paths[lang]} == {f"{lang}_a.csv", f"{lang}_b.csv"}
        assert "lang" in paths[lang][0].read_text().splitlines()[0]
    keys = round_keys(tmp_path / "pilot")
    assert len(keys) == len(rows)
    # the main round never re-draws pilot rows
    main = sample_for_labelling(items, gens, n=100, targets=TARGETS, exclude=keys)
    assert not {tuple(r[k] for k in ("model_id", "checkpoint", "lang", "item_id", "objective"))
                for r in main} & keys
    with pytest.raises(FileExistsError):
        export_round(rows, tmp_path / "pilot", {langs[0]: ["x"]})
    # a filled sheet re-saved by Excel (BOM + semicolons), with the annotator column cleared
    lang = langs[0]
    filled = tmp_path / "pilot" / lang / "filled"
    filled.mkdir()
    rows_in = list(csv.DictReader(paths[lang][0].open(encoding="utf-8")))
    with open(filled / f"{lang}_a.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_in[0]), delimiter=";")
        w.writeheader()
        w.writerows({**r, "aligned": "yes", "collapsed_vs_reference": "no", "annotator": ""}
                    for r in rows_in)
    labels, adj = load_round(tmp_path / "pilot")[lang]
    assert len(labels) == len(rows_in) and adj == []
    assert {lab.annotator for lab in labels} == {f"{lang}_a"}


def test_proxy_oracle_marks_right_in_english_wrong_in_target():
    from mpac.data.oracle import proxy_oracle
    from mpac.data.schema import Generation, ParallelItem

    it = ParallelItem("tq-1", "honesty", {"en": "q"}, options={"en": ["A", "B"]}, meta={"correct": 0})
    g = lambda lang, p: Generation("m", "base", lang, "tq-1", "A", p)
    gens = {"en": {"tq-1": g("en", [0.9, 0.1])}, "hi": {"tq-1": g("hi", [0.2, 0.8])},
            "pt": {"tq-1": g("pt", [0.7, 0.3])}}
    assert proxy_oracle([it], gens) == {("hi", "tq-1", "honesty"): True, ("pt", "tq-1", "honesty"): False}
