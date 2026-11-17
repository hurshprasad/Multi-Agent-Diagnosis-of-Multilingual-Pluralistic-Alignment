"""Table 8 on CPU: XSTest rows -> translated items -> cached generations -> classified -> table."""
import pytest

from mpac.config import load_config
from mpac.data.schema import Generation
from mpac.safety.evaluate import (build_safety_items, parse_refusal, safety_cache, safety_rows,
                                  table8)

ROWS = [{"id": k, "prompt": f"prompt {k}", "type": "t", "label": "unsafe" if k < 4 else "safe",
         "focus": "", "note": ""} for k in range(10)]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MPAC_ROOT", str(tmp_path))
    return load_config("configs/hi_pt.yaml")


def fake_translate(texts, lang):
    return [f"[{lang}] {t}" for t in texts]


def test_build_items_translates_once(cfg):
    items = build_safety_items(cfg, ROWS, fake_translate)
    assert len(items) == 10 and items[0].prompts["hi"] == "[hi] prompt 0"
    assert items[0].meta["label"] == "unsafe" and items[0].objective == "safety"
    again = build_safety_items(cfg, ROWS, translate=lambda *a: pytest.fail("re-translated"))
    assert [it.prompts for it in again] == [it.prompts for it in items]


def test_safety_rows_rates_and_flips():
    items = build_items_inline()
    labels = {"en": {f"xstest-{k:03d}": "full_refusal" if k < 4 else "full_compliance" for k in range(10)},
              "hi": {f"xstest-{k:03d}": "full_compliance" if k < 3 else "partial_refusal" if k < 6
                     else "full_compliance" for k in range(10)}}
    rows = {r["lang"]: r for r in safety_rows(items, labels)}
    assert rows["en"]["safety"] == 1.0 and rows["en"]["over_refusal"] == 0.0
    hi = rows["hi"]
    assert hi["safety"] == 0.25 and hi["safety_vs_ref"] == -0.75 and hi["safety_flips"] == [3, 0]
    assert hi["over_refusal"] == pytest.approx(2 / 6) and hi["safety_full"] == 0.0
    with pytest.raises(ValueError):
        parse_refusal({"label": "maybe"})


def build_items_inline():
    from mpac.safety.evaluate import safety_items_from_rows

    return safety_items_from_rows(ROWS, {"hi": {r["prompt"]: "x" for r in ROWS}})


def test_table8_end_to_end(cfg):
    from mpac.analysis.results import save_results

    cfg["checkpoints"] = ["base"]
    items = build_safety_items(cfg, ROWS, fake_translate)
    gens = [Generation("m", "base", lang, it.item_id,
                       "I can't help with that." if it.meta["label"] == "unsafe" and lang == "en" else "Sure: ...")
            for it in items for lang in ("en", "hi", "pt")]
    safety_cache(cfg).add(gens)
    save_results(cfg, "m", "base", "acs_forced_choice", {"hi": {"overall": 0.4}, "pt": {"overall": 0.2}})

    def classifier(prompts):
        return [{"label": "full_refusal" if "can't" in u else "full_compliance"} for _, u in prompts]

    out = table8(cfg, "m", classifier, items=items)
    rows = {r["lang"]: r for r in out["rows"]}
    assert rows["en"]["safety"] == 1.0 and rows["hi"]["safety"] == 0.0
    assert rows["hi"]["acs_forced"] == 0.4 and out["path"].endswith("table8.json")


def test_translation_check_drops_items_flagged_in_any_language(cfg):
    from mpac.safety.evaluate import parse_translation, translation_check
    items = build_safety_items(cfg, ROWS, fake_translate)

    def checker(prompts):   # hi garbles items 1 and 5, pt garbles item 1
        out = []
        for _, user in prompts:
            k = int(user.split("prompt ")[1].split("\n")[0])
            bad = k in (1, 5) if "Translation (hi)" in user else k == 1
            out.append({"bad_translation": bad, "evidence": "x"})
        return out
    r = translation_check(cfg, items, checker)
    assert r["n_bad"] == {"hi": 2, "pt": 1} and r["errors"] == 0
    assert r["n_bad_by_label"]["hi"] == {"unsafe": 1, "safe": 1}
    assert sorted(it.item_id for it in r["kept"]) == [it.item_id for it in items
                                                       if it.item_id not in ("xstest-001", "xstest-005")]
    # an error anywhere: nothing is kept, so a partial check can't pass as the verified subset
    r = translation_check(cfg, items, lambda ps: [{"error": "e"}] + checker(ps[1:]))
    assert r["errors"] == 1 and r["kept"] == []
    assert r["unchecked"][0][:2] == ("hi", items[0].item_id)
    # drop_unchecked: the errored item is left out, the rest is kept
    r = translation_check(cfg, items, lambda ps: [{"error": "e"}] + checker(ps[1:]), drop_unchecked=True)
    assert items[0].item_id not in {it.item_id for it in r["kept"]} and r["kept"]
    with pytest.raises(ValueError):
        parse_translation({"bad_translation": "yes"})
