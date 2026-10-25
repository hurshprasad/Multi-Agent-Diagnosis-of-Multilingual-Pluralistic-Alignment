from collections import Counter

import pytest

from mpac.config import load_config
from mpac.data.schema import ParallelItem
from mpac.training.dpo import (
    adapter_dir,
    assign_languages,
    build_preference_pairs,
    eval_exclusions,
    length_report,
    select_pairs,
    usable_batch,
)


def _row(i, country, words=20):
    return {"id": i, "country": country, "prompt": f"Question {i} about family values?",
            "pref": " ".join(["good"] * words), "dispref": " ".join(["bad"] * words)}


ROWS = [_row(i, c) for i, c in enumerate(["BR", "DE", "US"] * 10)]


def fake_mt(texts, lang, source_lang="en"):
    return [f"<{lang}>{t}" for t in texts]


def test_select_pairs_round_robin_over_countries_and_deterministic():
    pairs = select_pairs(ROWS, 9, seed=1)
    assert Counter(p["country"] for p in pairs) == {"BR": 3, "DE": 3, "US": 3}
    assert [p["country"] for p in pairs[:3]] == ["BR", "DE", "US"]
    assert pairs == select_pairs(list(reversed(ROWS)), 9, seed=1)
    assert set(pairs[0]) == {"pair_id", "country", "prompt", "chosen", "rejected"}


def test_select_pairs_excludes_eval_prompts_long_and_duplicate_rows():
    rows = [_row(0, "BR"), _row(1, "BR", words=400), {**_row(2, "BR"), "dispref": "good " * 20},
            _row(3, "BR"), {**_row(4, "BR"), "prompt": "question 3   ABOUT family values?"},
            _row(5, "BR"), _row(6, "BR")]
    pairs = select_pairs(rows, 2, exclude_ids={"0"}, exclude_prompts={"question 5 about family values?"},
                         max_words=300)
    # 0 eval id, 1 too long, 2 chosen == rejected, 4 same prompt as 3, 5 eval prompt
    assert {p["pair_id"] for p in pairs} == {"3", "6"}
    with pytest.raises(ValueError, match="raise training.pool"):
        select_pairs(rows, 5, exclude_ids={"0"})


def test_eval_exclusions_reads_plural_items_only():
    items = [ParallelItem("plural-7", "pluralistic_value", {"en": "Is  family important?"},
                          meta={"source": "plural", "plural_id": 7}),
             ParallelItem("uf-1", "helpfulness", {"en": "Plan a trip"}, meta={"source": "ultrafeedback"})]
    assert eval_exclusions(items) == ({"7"}, {"is family important?"})


def test_assign_languages_balances_languages_within_each_country():
    mixed = assign_languages(select_pairs(ROWS, 9, seed=1), ["en", "hi", "pt"])
    assert Counter(p["lang"] for p in mixed) == {"en": 3, "hi": 3, "pt": 3}
    for c in ("BR", "DE", "US"):
        assert {p["lang"] for p in mixed if p["country"] == c} == {"en", "hi", "pt"}
    # Many countries with one or two pairs each (the quick run): totals still within 1.
    rows = [_row(i, f"C{i % 20:02d}") for i in range(48)]
    totals = Counter(p["lang"] for p in assign_languages(select_pairs(rows, 48, seed=1),
                                                         ["en", "hi", "pt"]))
    assert max(totals.values()) - min(totals.values()) <= 1


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MPAC_ROOT", str(tmp_path))
    cfg = load_config("configs/hi_pt.yaml")
    cfg["training"]["n_pairs"] = 9
    return cfg


def test_build_preference_pairs_same_pairs_and_count_for_both_checkpoints(cfg, tmp_path):
    items = [ParallelItem("plural-0", "pluralistic_value", {"en": ROWS[0]["prompt"]},
                          meta={"source": "plural", "plural_id": 0})]
    out = build_preference_pairs(cfg, items=items, rows=ROWS, translate=fake_mt)
    en, multi = out["en_aligned"], out["multi_aligned"]
    assert len(en) == len(multi) == 9
    assert "0" not in {p["pair_id"] for p in en}
    assert [p["pair_id"] for p in en] == [p["pair_id"] for p in multi]
    assert Counter(p["lang"] for p in multi) == {"en": 3, "hi": 3, "pt": 3}
    hi = next(p for p in multi if p["lang"] == "hi")
    assert hi["prompt"].startswith("<hi>") and hi["chosen"].startswith("<hi>")
    assert (tmp_path / "training" / "pairs" / "multi_aligned.jsonl").exists()

    def boom(texts, lang):
        raise AssertionError("re-translated")

    # A rerun reuses the selection and the translations, even with no rows supplied.
    again = build_preference_pairs(cfg, items=items, rows=[], translate=boom)
    assert again == out


def test_length_report_flags_truncation_per_language():
    pairs = [{"lang": "en", "prompt": "p", "chosen": "a b", "rejected": "a"},
             {"lang": "hi", "prompt": "p", "chosen": "a b c d", "rejected": "a b c"}]
    rep = length_report(pairs, lambda prompt, resp: len(resp.split()) + 1, max_length=4)
    assert rep["en"]["truncated"] == 0 and rep["hi"]["truncated"] == 0.5
    assert rep["hi"]["max"] == 5


def test_adapter_dir(cfg, tmp_path):
    assert adapter_dir(cfg, "m", "base") is None
    assert adapter_dir(cfg, "m", "en_aligned") == tmp_path / "checkpoints" / "m" / "en_aligned"


def test_usable_batch_matches_row_filter():
    rows = [_row(0, "BR"), _row(1, "BR", words=400), {**_row(2, "BR"), "dispref": "good " * 20}]
    batch = {k: [r[k] for r in rows] for k in rows[0]}
    assert usable_batch(batch, max_words=300, max_paragraph_words=150) == [True, False, False]
