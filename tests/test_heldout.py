import math

import pytest

from mpac.config import load_config
from mpac.data.io import write_jsonl
from mpac.training import heldout
from mpac.training.heldout import build_heldout_pairs, preference_report, score_heldout


def _row(i, country):
    return {"id": i, "country": country, "prompt": f"Question {i} about family values?",
            "pref": "good " * 20, "dispref": "bad " * 20}


ROWS = [_row(i, c) for i, c in enumerate(["BR", "DE", "US"] * 10)]


def fake_mt(texts, lang, source_lang="en"):
    return [f"<{lang}>{t}" for t in texts]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MPAC_ROOT", str(tmp_path))
    cfg = load_config("configs/hi_pt.yaml")
    cfg["training"]["n_heldout"] = 6
    return cfg


def test_build_heldout_pairs_excludes_training_pairs_and_translates_all(cfg, tmp_path):
    write_jsonl(tmp_path / "training" / "pairs" / "selected.jsonl",
                [{"pair_id": str(i), "prompt": ROWS[i]["prompt"]} for i in range(10)])
    out = build_heldout_pairs(cfg, items=[], rows=ROWS, translate=fake_mt)
    ids = [p["pair_id"] for p in out["en"]]
    assert len(ids) == 6 and not set(ids) & {str(i) for i in range(10)}
    for lang in ("hi", "pt"):
        assert [p["pair_id"] for p in out[lang]] == ids
        assert out[lang][0]["chosen"].startswith(f"<{lang}>")
    assert build_heldout_pairs(cfg, items=[], rows=[], translate=None) == out  # cached


def test_build_heldout_pairs_needs_training_pairs(cfg):
    with pytest.raises(FileNotFoundError, match="step 1"):
        build_heldout_pairs(cfg, items=[], rows=ROWS, translate=fake_mt)


def _lp(ck, lang, pid, c, r):
    return {"checkpoint": ck, "lang": lang, "pair_id": pid, "chosen": c, "rejected": r}


def test_preference_report_margins_relative_to_base():
    rows = []
    for k in range(4):
        pid = str(k)
        rows += [_lp("base", "hi", pid, -10.0, -10.0),
                 _lp("en_aligned", "hi", pid, -10.0, -10.5 if k == 0 else -9.0),   # 1 of 4 right
                 _lp("multi_aligned", "hi", pid, -9.0, -11.0)]                      # all right
    rep = {r["checkpoint"]: r for r in preference_report(rows, beta=0.1)}
    assert rep["en_aligned"]["accuracy"] == 0.25 and rep["multi_aligned"]["accuracy"] == 1.0
    assert math.isclose(rep["multi_aligned"]["mean_margin"], 0.1 * 2.0)
    diff = rep["multi_aligned - en_aligned"]
    assert diff["n"] == 4 and diff["accuracy"] == 0.75 and diff["mean_margin"] > 0


def test_score_heldout_resumes_from_cache(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(heldout, "response_logprob",
                        lambda m, t, prompt, resp, n: calls.append(resp) or -float(len(resp)))
    pairs = {"en": [{"pair_id": "1", "prompt": "p", "chosen": "ab", "rejected": "abc"}]}
    model = {"id": "m"}
    rows = score_heldout(cfg, model, pairs, tok=None, scorer=object())
    assert len(rows) == 3 and len(calls) == 6   # base + two adapters, chosen + rejected
    assert score_heldout(cfg, model, pairs, tok=None, scorer=object()) == rows and len(calls) == 6
