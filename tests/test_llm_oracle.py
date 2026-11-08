"""LLM-panel oracle on CPU: fake annotators stand in for Claude / Llama."""
import json

import pytest

from mpac.analysis.evaluators import table6
from mpac.analysis.results import save_results
from mpac.config import load_config
from mpac.data.io import write_jsonl
from mpac.data.llm_oracle import (
    REASON_FORMAT,
    REPLY_FORMAT,
    adjudicate,
    build_prompt,
    disagreements,
    known_answer_check,
    label_round,
    parse_reply,
)
from mpac.data.oracle import (
    agreement,
    drop_bad_translations,
    export_round,
    load_round,
    resolve,
    sample_for_labelling,
)
from mpac.data.schema import Generation, ParallelItem
from mpac.data.synthetic import make_synthetic
from mpac.generation.cache import GenerationCache

TARGETS = ["hi", "pt"]
OBJECTIVES = ["helpfulness", "honesty", "pluralistic_value"]


def _key_from_prompt(prompt):
    """The fake annotators read the planted label back out of the prompt's item id."""
    return prompt.split("prompt ")[1].split("\n")[0]


def _fake(oracle, lang_of, flip=frozenset(), fail=frozenset(), bad=frozenset()):
    def annotate(prompts):
        out = []
        for p in prompts:
            iid = _key_from_prompt(p)
            if iid in fail:
                out.append({"error": "stop_reason refusal"})
                continue
            truth = next(v for (lang, i, _), v in oracle.items() if i == iid and lang == lang_of(p))
            out.append({"aligned": True, "collapsed_vs_reference": truth != ((lang_of(p), iid) in flip),
                        "bad_translation": (lang_of(p), iid) in bad, "evidence": "test",
                        "cost_tokens": 10})
        return out
    return annotate


def _lang_of(prompt):
    return "hi" if "Hindi request" in prompt else "pt"


@pytest.fixture
def setup(tmp_path):
    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=60,
                                         collapsed={("hi", "honesty"): 0.8})
    rows = sample_for_labelling(items, {("synthetic", "base"): gens}, n=120, targets=TARGETS)
    round_dir = tmp_path / "oracle" / "llm_main"
    export_round(rows, round_dir, {lang: ["opus", "llama"] for lang in TARGETS})
    return items, gens, oracle, rows, round_dir


def test_parse_reply_accepts_wrapped_json_and_rejects_bad():
    d = parse_reply('Sure:\n{"aligned": true, "collapsed_vs_reference": false, '
                    '"bad_translation": false, "evidence": "same answer"}\n')
    assert d == {"aligned": True, "collapsed_vs_reference": False, "bad_translation": False,
                 "evidence": "same answer"}
    with pytest.raises(ValueError):
        parse_reply('{"aligned": "yes", "collapsed_vs_reference": false, "bad_translation": false}')
    with pytest.raises(ValueError):
        parse_reply("no json here")
    # reasoning first (open models), braces in it, labels in the last object
    d = parse_reply('The {English} answer is A, Hindi picks B.\n{"aligned": false, '
                    '"collapsed_vs_reference": true, "bad_translation": false, "evidence": "B"}')
    assert d["collapsed_vs_reference"] is True and d["evidence"] == "B"


def test_prompt_shows_both_responses_and_probs(setup):
    *_, rows, _ = setup
    p = build_prompt(rows[0])
    assert "English response:" in p and "option probabilities" in p and "Reply with JSON" in p
    assert p.endswith(REPLY_FORMAT) and "First reason" in p.replace(REPLY_FORMAT, REASON_FORMAT)


def test_panel_round_resume_adjudicate(setup):
    _, _, oracle, rows, round_dir = setup
    hi_ids = [r["item_id"] for r in rows if r["lang"] == "hi"]
    flip = frozenset(("hi", i) for i in hi_ids[:3])

    # opus fails two items (in both languages) the first time; a re-run retries only those.
    stats = label_round(round_dir, "opus", _fake(oracle, _lang_of, fail=frozenset(hi_ids[3:5])))
    assert stats["hi"]["errors"] == 2
    calls = []
    def counting(prompts):
        calls.append(len(prompts))
        return _fake(oracle, _lang_of)(prompts)
    stats = label_round(round_dir, "opus", counting)
    assert calls == [2, 2] and stats["hi"]["errors"] == 0 and stats["hi"]["labelled"] == len(hi_ids)

    label_round(round_dir, "llama", _fake(oracle, _lang_of, flip=flip))
    dis = disagreements(round_dir)
    assert {("hi", k[3]) for k in dis["hi"]} == flip and dis["pt"] == []

    labels, adj = load_round(round_dir)["hi"]
    assert agreement(labels)["collapsed_vs_reference"]["kappa"] < 1
    _, unresolved = resolve(labels, adj, min_annotators=2)
    assert len(unresolved) == 3

    adjudicate(round_dir, "sonnet", _fake(oracle, _lang_of))
    labels, adj = load_round(round_dir)["hi"]
    resolved, unresolved = resolve(labels, adj, min_annotators=2)
    assert unresolved == [] and len(adj) == 3
    assert all(lab.collapsed_vs_reference == oracle[(lab.lang, lab.item_id, lab.objective)]
               for lab in resolved.values())


def test_adjudicator_bad_translation_drops_item(setup):
    # Only the adjudicator sees the bad translation: the item must be dropped, not resolved
    # to the adjudication's forced "no / no".
    _, _, oracle, rows, round_dir = setup
    hi_ids = [r["item_id"] for r in rows if r["lang"] == "hi"]
    flip = frozenset(("hi", i) for i in hi_ids[:3])
    label_round(round_dir, "opus", _fake(oracle, _lang_of))
    label_round(round_dir, "llama", _fake(oracle, _lang_of, flip=flip))
    adjudicate(round_dir, "sonnet", _fake(oracle, _lang_of, bad=frozenset({("hi", hi_ids[0])})))
    labels, adj = load_round(round_dir)["hi"]
    kept, dropped = drop_bad_translations(labels, adj)
    assert dropped and all(hi_ids[0] in k for k in dropped)
    resolved, unresolved = resolve(kept, adj, min_annotators=2)
    assert unresolved == [] and len(resolved) == len(hi_ids) - 1


def test_single_annotator_is_not_unanimous(setup):
    _, _, oracle, _, round_dir = setup
    label_round(round_dir, "opus", _fake(oracle, _lang_of))
    labels, adj = load_round(round_dir)["hi"]
    assert resolve(labels, adj)[1] == []                   # old behaviour, min 1
    assert len(resolve(labels, adj, min_annotators=2)[1]) == len(labels)


def test_table6_reads_llm_round(setup, tmp_path, monkeypatch):
    items, gens, oracle, _, round_dir = setup
    monkeypatch.setenv("MPAC_ROOT", str(tmp_path))
    cfg = load_config("configs/hi_pt.yaml")
    cfg["agents"].update(lifecycle_cycles=4, batch_per_cycle=20)
    write_jsonl(tmp_path / "items" / "items.jsonl", [it.to_dict() for it in items])
    GenerationCache(tmp_path / "generations").add(g for by in gens.values() for g in by.values())
    save_results(cfg, "synthetic", "base", "acs_free_text",
                 {"hi": {"prompt_floor": 0.14}, "pt": {"prompt_floor": 0.10}})
    label_round(round_dir, "opus", _fake(oracle, _lang_of))
    label_round(round_dir, "llama", _fake(oracle, _lang_of))
    out = table6(cfg, "synthetic", "base", source="llm", round_name="llm_main")
    assert out["oracle"]["source"] == "llm, round llm_main" and out["oracle"]["unresolved"] == 0
    assert (tmp_path / "results" / "synthetic" / "base" / "table6_llm.json").exists()


def test_known_answer_check(tmp_path):
    # Three TruthfulQA-like items, correct option 0: lost (en right, hi wrong), same, gained.
    probs = {"lost": ([0.9, 0.1], [0.2, 0.8]), "same": ([0.9, 0.1], [0.8, 0.2]),
             "gain": ([0.3, 0.7], [0.7, 0.3])}
    items = [ParallelItem(iid, "honesty", {"en": f"prompt {iid}", "hi": f"prompt {iid}"},
                          options={"en": ["A", "B"], "hi": ["A", "B"]}, meta={"correct": 0})
             for iid in probs]
    gens = {"en": {i: Generation("m", "base", "en", i, "A", p[0]) for i, p in probs.items()},
            "hi": {i: Generation("m", "base", "hi", i, "A", p[1]) for i, p in probs.items()}}
    rows = sample_for_labelling(items, {("m", "base"): gens}, n=10, targets=["hi"])
    round_dir = tmp_path / "r"
    export_round(rows, round_dir, {"hi": ["good"]})
    truth = {"lost": (False, True), "same": (True, False), "gain": (True, True)}
    label_round(round_dir, "good", lambda ps: [
        {"aligned": truth[_key_from_prompt(p)][0],
         "collapsed_vs_reference": truth[_key_from_prompt(p)][1],
         "bad_translation": False, "evidence": ""} for p in ps])
    r = known_answer_check(round_dir, items)["good"]
    assert r["n"] == 3 and r["aligned_accuracy"] == 1.0
    assert r["n_lost"] == 1 and r["lost_recall"] == 1.0
    assert r["n_same"] == 1 and r["same_specificity"] == 1.0
    log = (round_dir / "hi" / "filled" / "good.jsonl").read_text().splitlines()
    assert len(log) == 3 and json.loads(log[0])["annotator"] == "good"

    # a second annotator marks "lost" as a bad translation: it leaves both annotators' counts
    round2 = tmp_path / "r2"
    export_round(rows, round2, {"hi": ["good", "picky"]})
    label_round(round2, "good", lambda ps: [
        {"aligned": truth[_key_from_prompt(p)][0],
         "collapsed_vs_reference": truth[_key_from_prompt(p)][1],
         "bad_translation": False, "evidence": ""} for p in ps])
    label_round(round2, "picky", lambda ps: [
        {"aligned": False, "collapsed_vs_reference": False,
         "bad_translation": _key_from_prompt(p) == "lost", "evidence": ""} for p in ps])
    r = known_answer_check(round2, items)
    assert r["good"]["n"] == 2 and r["good"]["n_lost"] == 0 and r["picky"]["n"] == 2
