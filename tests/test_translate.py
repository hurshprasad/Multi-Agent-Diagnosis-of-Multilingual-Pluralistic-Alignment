import pytest

from mpac.data.translate import (
    _family,
    blind_comparison,
    join_paragraphs,
    round_trip,
    split_paragraphs,
)


def test_paragraph_split_round_trips_layout():
    text = "Title\n\n1. first\n2. second\n"
    lines, keep = split_paragraphs(text)
    assert lines == ["Title", "1. first", "2. second"]
    assert join_paragraphs([s.upper() for s in lines], keep, text.count("\n") + 1) == \
        "TITLE\n\n1. FIRST\n2. SECOND\n"


def test_family_detection():
    assert _family("google/madlad400-3b-mt") == "madlad"
    assert _family("facebook/nllb-200-distilled-1.3B") == "nllb"
    with pytest.raises(ValueError):
        _family("Helsinki-NLP/opus-mt-en-de")


def test_round_trip_scores_a_lossless_translator_perfectly():
    # Fake MT: tags text on the way out and strips the tag on the way back.
    def fake(texts, target_lang, source_lang="en"):
        if target_lang == "en":
            return [t.split("] ", 1)[1] for t in texts]
        return [f"[{target_lang}] {t}" for t in texts]

    texts = ["Is honesty more important than kindness?", "Name the capital of Australia."]
    out = round_trip(texts, fake, ["es", "de"])
    assert out["es"]["forward"][0].startswith("[es]")
    assert out["de"]["back"] == texts
    assert out["es"]["chrf"] == pytest.approx(100.0)


def test_blind_comparison_hides_and_records_model_identity():
    src = [f"s{i}" for i in range(10)]
    outputs = {"madlad": [f"m{i}" for i in range(10)], "nllb": [f"n{i}" for i in range(10)]}
    sheet, key = blind_comparison(src, outputs, n=6, seed=1)
    assert len(sheet) == len(key) == 6
    assert "madlad" not in str(sheet) and "nllb" not in str(sheet)
    for row, k in zip(sheet, key):
        i = src.index(row["source"])
        assert row["A"] == outputs[k["A"]][i] and row["B"] == outputs[k["B"]][i]
    assert {k["A"] for k in key} == {"madlad", "nllb"}   # order is actually shuffled
    with pytest.raises(ValueError):
        blind_comparison(src, {"madlad": outputs["madlad"]})
