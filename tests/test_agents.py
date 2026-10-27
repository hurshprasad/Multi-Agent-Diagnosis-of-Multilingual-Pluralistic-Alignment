from mpac.agents.base import AgentSpec, AgentState
from mpac.agents.coordinator import LifecycleCoordinator
from mpac.agents.registry import AgentRegistry
from mpac.analysis.metrics import detection_scores
from mpac.baselines.static import fixed_multi_agent, single_evaluator
from mpac.config import load_config
from mpac.data.synthetic import make_synthetic

TARGETS = ["pt", "es", "it", "de", "fr"]
OBJECTIVES = ["helpfulness", "honesty", "pluralistic_value"]
PLANTED = {("de", "honesty"): 0.9, ("it", "pluralistic_value"): 0.8}


def _cfg():
    cfg = load_config()
    cfg["agents"].update(population_cap=12, lifecycle_cycles=5, batch_per_cycle=30)
    return cfg


def test_agent_state_roundtrip(tmp_path):
    reg = AgentRegistry(tmp_path / "reg.json")
    reg.spawn(AgentSpec("specialist", "es", "honesty"), cycle=0, reason="test")
    reg.save()
    reg2 = AgentRegistry(tmp_path / "reg.json")
    assert list(reg2.states) == ["spec:es:honesty:divergence"]
    assert isinstance(reg2.states["spec:es:honesty:divergence"], AgentState)
    assert reg2.op_counts() == {"spawn": 1}


def test_coordinator_spawns_specialists_for_planted_collapse(tmp_path):
    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=150, collapsed=PLANTED)
    coord = LifecycleCoordinator(_cfg(), items, gens, oracle,
                                 registry=AgentRegistry(tmp_path / "reg.json"))
    summary = coord.run()
    alive = set(summary["final_alive"])
    assert "spec:de:honesty:divergence" in alive
    assert "spec:it:pluralistic_value:divergence" in alive
    assert len(alive) <= 12
    assert sum(a.startswith("cons:") for a in alive) <= 1   # one agent per unordered pair
    # the collapse map should rank the planted cells highest
    cmap = summary["collapse_map"]
    top2 = sorted(cmap, key=lambda k: cmap[k]["collapse_rate"], reverse=True)[:2]
    assert set(top2) == set(PLANTED)
    # consistency (target-vs-target) evidence must not be counted as collapse vs English
    s = detection_scores(coord.all_evals, oracle)
    assert s["precision"] > 0.9 and s["recall"] > 0.9


def test_baselines_and_detection():
    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=90, collapsed=PLANTED)
    single = single_evaluator(items, gens, TARGETS)
    fixed = fixed_multi_agent(items, gens, TARGETS, OBJECTIVES)
    for evals in (single, fixed):
        s = detection_scores(evals, oracle)
        assert s["recall"] > 0.8 and s["precision"] > 0.8


def test_thresholds_are_per_language_and_kind():
    from mpac.agents.base import LightweightAgent
    from mpac.data.schema import Generation, ParallelItem

    it = ParallelItem("uf-1", "helpfulness", {"en": "q", "hi": "q", "pt": "q"})
    gens = {lang: {"uf-1": Generation("m", "base", lang, "uf-1", "x")} for lang in ("en", "hi", "pt")}
    emb = {"en": {"uf-1": [1.0, 0.0]}, "hi": {"uf-1": [0.8, 0.6]}, "pt": {"uf-1": [0.8, 0.6]}}  # d = 0.2
    th = {("hi", "embedding"): 0.29, ("pt", "embedding"): 0.15}
    hi = LightweightAgent(AgentSpec("specialist", "hi", None), th).evaluate(it, gens, emb)
    pt = LightweightAgent(AgentSpec("specialist", "pt", None), th).evaluate(it, gens, emb)
    assert not hi.collapse and pt.collapse      # same divergence, different language floors
    # a response in the wrong language is collapse whatever its divergence
    wrong = LightweightAgent(AgentSpec("specialist", "hi", None), th,
                             {"en": {"uf-1": "en"}, "hi": {"uf-1": "en"}}).evaluate(it, gens, emb)
    assert wrong.collapse and "answered in en" in wrong.evidence


def test_run_systems_same_budget_and_test_labels_only():
    from mpac.analysis.evaluators import run_systems

    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=90, collapsed=PLANTED)
    ids = sorted({k[1] for k in oracle})
    tune_ids = set(ids[: len(ids) // 3])
    tune = {k: v for k, v in oracle.items() if k[1] in tune_ids}
    test = {k: v for k, v in oracle.items() if k[1] not in tune_ids}
    out = run_systems(_cfg(), items, gens, tune, test)
    sys = out["systems"]
    assert set(sys) == {"4 single evaluator", "5 fixed multi-agent", "7 adaptive lifecycle"}
    assert sys["7 adaptive lifecycle"]["n_evaluations"] == out["budget"]
    assert sys["4 single evaluator"]["n_evaluations"] <= out["budget"]
    for s in sys.values():
        assert s["n_labelled"] == len(test)     # unevaluated test items count as missed
    assert sys["7 adaptive lifecycle"]["recall"] > 0.5


def test_tune_thresholds_uses_labels_and_keeps_start_when_too_few():
    from mpac.signals.calibration import floor_thresholds, tune_thresholds

    start = floor_thresholds(["hi", "pt"], {"hi": 0.14, "pt": 0.10})
    assert abs(start[("hi", "embedding")] - 0.29) < 1e-9 and start[("pt", "options")] == 0.3
    divs = {("hi", "embedding"): {f"i{k}": 0.1 * k for k in range(10)},
            ("pt", "embedding"): {"j0": 0.5, "j1": 0.1}}
    oracle = {("hi", f"i{k}", "helpfulness"): k >= 6 for k in range(10)}
    oracle |= {("pt", "j0", "helpfulness"): True, ("pt", "j1", "helpfulness"): False}
    th, rep = tune_thresholds(divs, oracle, start)
    assert 0.5 < th[("hi", "embedding")] < 0.6 and rep[("hi", "embedding")]["source"] == "tuned"
    assert th[("pt", "embedding")] == start[("pt", "embedding")]   # 2 labels: too few


def test_language_match_and_detect_languages_skip_forced_choice():
    from mpac.data.schema import Generation
    from mpac.signals.lightweight import detect_languages, language_match

    fake = lambda text: ("hi", 0.9) if "नमस्ते" in text else ("en", 0.9)
    assert language_match("नमस्ते दुनिया", "hi", fake) == 1.0
    assert language_match("hello", "hi", fake) == 0.0 and language_match("  ", "hi", fake) == 0.0
    gens = {"hi": {"uf-1": Generation("m", "base", "hi", "uf-1", "hello"),
                   "tq-1": Generation("m", "base", "hi", "tq-1", "A", [1.0, 0.0])}}
    assert detect_languages(gens, fake) == {"hi": {"uf-1": "en"}}
