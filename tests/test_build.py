import csv

import pytest

from mpac.config import load_config
from mpac.data import build
from mpac.data.build import (
    INSTRUCTION,
    assemble_items,
    format_prompt,
    texts_to_translate,
    truthfulqa_items,
    ultrafeedback_items,
    wvs_items,
    wvs_stem,
)
from mpac.data.loaders import load_items
from mpac.data.schema import SourceItem
from mpac.data.translate import TranslationCache
from mpac.data.verification import export_verification, read_sheet, verification_ids


def fake_mt(texts, lang, source_lang="en"):
    return [f"<{lang}>{t}" for t in texts]


def _wvs_row(code, main, text, options):
    return {"question_code": code, "main_question": main, "question_text": text,
            "answer_options": [{"code": c, "label": lab} for c, lab in options],
            "question_group_id": 0}


def test_wvs_stem_handles_titles_repeats_and_lead_ins():
    assert wvs_stem("Willingness to fight for country", "Would you fight?") == "Would you fight?"
    assert wvs_stem("Which do you believe in?", "Which do you believe in? - God") == \
        "Which do you believe in? - God"
    assert wvs_stem("Do you agree or disagree with the following statements?", "Work is a duty") \
        == "Do you agree or disagree with the following statements?\nWork is a duty"


def test_wvs_items_keep_code_zero_drop_missing_codes_and_sort():
    rows = [_wvs_row("E012", "Fight", "Would you fight?",
                     [("-2", "No answer"), ("1", "Yes"), ("0", "No")]),
            _wvs_row("X1", "Only one", "Q?", [("-1", "Don't know"), ("1", "Yes")])]
    (item,) = wvs_items(rows)
    assert item.item_id == "wvs-E012" and item.objective == "pluralistic_value"
    assert item.options == ["No", "Yes"] and item.meta["codes"] == [0, 1]


def test_wvs_mentioned_lists_become_yes_no_questions():
    rows = [_wvs_row("A042", "Which qualities are important? Choose up to five!", "Obedience",
                     [("0", "Not mentioned"), ("1", "Mentioned")])]
    (item,) = wvs_items(rows)
    assert item.stem.endswith("Would you choose this one: Obedience?")
    assert item.options == ["No", "Yes"] and item.meta["codes"] == [0, 1]


def test_truthfulqa_items_shuffle_and_track_the_correct_answer():
    rows = [{"question": f"Q{i}?", "mc1_targets": {
        "choices": ["right", "w1", "w2", "w3", "w4"], "labels": [1, 0, 0, 0, 0]}}
        for i in range(20)]
    items = truthfulqa_items(rows, max_options=4, seed=1)
    assert items == truthfulqa_items(rows, max_options=4, seed=1)
    assert all(len(it.options) == 4 and it.options[it.meta["correct"]] == "right" for it in items)
    assert len({it.meta["correct"] for it in items}) > 1   # correct answer isn't always A


def test_ultrafeedback_filter_drops_code_and_long_prompts():
    rows = [{"prompt": "How do I plan a trip?", "prompt_id": "a" * 20},
            {"prompt": "Fix this python code: def f(): pass", "prompt_id": "b" * 20},
            {"prompt": "word " * 200, "prompt_id": "c" * 20}]
    assert [it.item_id for it in ultrafeedback_items(rows)] == ["uf-aaaaaaaaaaaa"]


SOURCES = [
    SourceItem("wvs-A", "pluralistic_value", "wvs", "How important?", ["Not at all", "5", "Very"]),
    SourceItem("uf-1", "helpfulness", "ultrafeedback", "Plan a trip."),
]


def test_texts_and_assembly_keep_letters_and_numbers_out_of_mt():
    texts = texts_to_translate(SOURCES)
    assert "5" not in texts and INSTRUCTION in texts and len(texts) == 5
    table = dict(zip(texts, fake_mt(texts, "hi")))
    items = assemble_items(SOURCES, {"hi": table})
    fc = items[0]
    assert fc.prompts["en"] == format_prompt("How important?", ["Not at all", "5", "Very"],
                                             INSTRUCTION)
    assert fc.prompts["hi"].startswith("<hi>How important?\n\nA. <hi>Not at all\nB. 5\nC. ")
    assert fc.prompts["hi"].endswith(f"<hi>{INSTRUCTION}")
    assert fc.options == {"en": ["A", "B", "C"], "hi": ["A", "B", "C"]}
    assert items[1].prompts["hi"] == "<hi>Plan a trip." and items[1].options is None


def test_partially_translated_item_skips_that_language():
    texts = texts_to_translate(SOURCES)
    table = dict(zip(texts, fake_mt(texts, "pt")))
    del table["Very"]
    items = assemble_items(SOURCES, {"pt": table})
    assert "pt" not in items[0].prompts and "pt" in items[1].prompts


def test_translation_cache_resumes(tmp_path):
    cache = TranslationCache(tmp_path, "google/madlad400-3b-mt")
    assert cache.translate_missing(["a", "b", "a"], "hi", fake_mt, chunk=1, progress=False) == 2
    calls = []
    assert cache.translate_missing(["a", "b", "c"], "hi",
                                   lambda t, lang: calls.append(t) or fake_mt(t, lang),
                                   progress=False) == 1
    assert calls == [["c"]] and cache.load("hi")["c"] == "<hi>c"
    assert cache.path("hi").parent.name == "madlad400-3b-mt"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MPAC_ROOT", str(tmp_path))
    monkeypatch.setattr(build, "fetch_sources", lambda cfg: list(SOURCES))
    return load_config("configs/hi_pt.yaml")


def test_build_dataset_end_to_end_with_verification(cfg, tmp_path):
    items = build.build_dataset(cfg, translate=fake_mt)
    assert {lang for it in items for lang in it.prompts} == {"en", "hi", "pt"}
    assert (tmp_path / "translations" / "madlad400-3b-mt" / "hi.jsonl").exists()
    # A rerun reuses the cache: a translator that fails proves nothing is re-translated.
    def boom(texts, lang):
        raise AssertionError("re-translated")

    assert build.build_dataset(cfg, translate=boom)[0].prompts == items[0].prompts

    ids = verification_ids(items, n=2)
    path = tmp_path / "verification" / "hi.csv"
    assert export_verification(items, "hi", ids, path) == 2
    rows = read_sheet(path)
    for r in rows:
        if r["item_id"] == "uf-1":
            r["corrected"] = "यात्रा की योजना बनाएं।"
        else:
            r["ok"] = "हाँ"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(FileExistsError):
        export_verification(items, "hi", ids, path)

    loaded = {it.item_id: it for it in load_items(cfg)}
    assert loaded["uf-1"].prompts["hi"] == "यात्रा की योजना बनाएं।"
    assert loaded["wvs-A"].translation_verified == {"hi": True}
    verified = load_items(cfg, verified_only=True)
    assert all(set(it.prompts) == {"en", "hi"} for it in verified) and len(verified) == 2


def test_read_sheet_accepts_excel_bom_and_semicolons(tmp_path):
    path = tmp_path / "pt.csv"
    path.write_text("﻿Item_ID;objective;ok;corrected\nuf-1;helpfulness;sim;\n",
                    encoding="utf-8")
    assert read_sheet(path) == [{"item_id": "uf-1", "objective": "helpfulness",
                                 "ok": "sim", "corrected": ""}]
    path.write_text("id,ok\nuf-1,yes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no item_id column"):
        read_sheet(path)
