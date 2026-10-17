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
