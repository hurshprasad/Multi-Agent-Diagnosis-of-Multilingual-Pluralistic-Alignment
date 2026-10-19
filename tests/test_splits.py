import pytest

from mpac.config import load_config
from mpac.data.splits import assign_splits, extend_splits, filter_oracle, get_or_create_splits
from mpac.data.synthetic import make_synthetic

TARGETS, OBJECTIVES = ["es", "de"], ["helpfulness", "honesty", "pluralistic_value"]


def test_splits_are_deterministic_stratified_and_order_free():
    items, _, _ = make_synthetic(TARGETS, OBJECTIVES, n_items=90)
    a = assign_splits(items, tune_frac=1 / 3, seed=7)
    assert a == assign_splits(list(reversed(items)), tune_frac=1 / 3, seed=7)
    assert set(a.values()) == {"tune", "test"}
    for obj in OBJECTIVES:
        ids = [it.item_id for it in items if it.objective == obj]
        assert sum(a[i] == "tune" for i in ids) == 10   # 30 per objective, a third to tune


def test_oracle_filter_keeps_every_language_of_an_item_together():
    items, _, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=60,
                                      collapsed={("de", "honesty"): 0.5})
    splits = assign_splits(items)
    tune, test = filter_oracle(oracle, splits, "tune"), filter_oracle(oracle, splits, "test")
    assert len(tune) + len(test) == len(oracle)
    assert {k[1] for k in tune}.isdisjoint({k[1] for k in test})
    with pytest.raises(ValueError):
        filter_oracle(oracle, splits, "train")


def test_saved_splits_are_frozen(tmp_path):
    cfg = load_config()
    path = tmp_path / "splits.json"
    items, _, _ = make_synthetic(TARGETS, OBJECTIVES, n_items=30)
    first = get_or_create_splits(cfg, items, path)
    cfg["project"]["seed"] = 999   # a changed seed must not redraw an existing file
    assert get_or_create_splits(cfg, items, path) == first
    more, _, _ = make_synthetic(TARGETS, OBJECTIVES, n_items=40)
    with pytest.raises(ValueError):
        get_or_create_splits(cfg, more, path)
    extended = extend_splits(first, more)
    assert all(extended[k] == v for k, v in first.items())
    assert len(extended) == 40
