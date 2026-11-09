"""System 6 (LLM-judge team) on CPU with a fake judge that reads the planted labels."""
import re

import pytest

from mpac.baselines.llm_judge import CachedJudge, LLMJudgeAgent, judge_prompt, llm_judge_team, parse_judge
from mpac.baselines.static import fixed_multi_agent
from mpac.agents.base import AgentSpec
from mpac.data.synthetic import make_synthetic

TARGETS = ["hi", "pt"]
OBJECTIVES = ["helpfulness", "honesty", "pluralistic_value"]


def oracle_judge(oracle, calls=None, fail=()):
    """Judge that answers from the planted labels; counts calls; errors on item ids in `fail`."""
    def judge(prompts):
        out = []
        for _, user in prompts:
            m = re.search(r"\[(\w+)\] prompt (syn-\d+)", user.rsplit("Request (", 1)[1])
            lang, iid = m.groups()
            if calls is not None:
                calls.append((lang, iid))
            if iid in fail:
                out.append({"error": "boom"})
                continue
            obj = next(o for (lg, i, o) in oracle if lg == lang and i == iid)
            y = oracle[(lang, iid, obj)]
            out.append({"divergence": 0.9 if y else 0.1, "collapse": y, "evidence": "x",
                        "cost_tokens": 100})
        return out
    return judge


def test_team_judges_the_same_items_as_system5_and_records_tokens():
    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=60,
                                         collapsed={("hi", "honesty"): 0.8})
    five = fixed_multi_agent(items, gens, TARGETS, OBJECTIVES, per_agent_budget=6, seed=3)
    six = llm_judge_team(items, gens, TARGETS, OBJECTIVES, oracle_judge(oracle),
                         per_agent_budget=6, seed=3)
    assert {(e.lang, e.item_id) for e in five} == {(e.lang, e.item_id) for e in six}
    assert all(e.cost_tokens == 100 and e.compared_to == "en" for e in six)
    assert all(e.collapse == oracle[(e.lang, e.item_id, e.objective)] for e in six)


def test_cached_judge_calls_once_and_retries_errors(tmp_path):
    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=6)
    prompts = [judge_prompt(it, "hi", gens) for it in items]
    calls = []
    j = CachedJudge(oracle_judge(oracle, calls, fail={items[0].item_id}), tmp_path / "j.jsonl", "m")
    first = j(prompts)
    assert "error" in first[0] and all("error" not in r for r in first[1:])
    assert len(calls) == 6 and j.n_errors == 1
    j2 = CachedJudge(oracle_judge(oracle, calls), tmp_path / "j.jsonl", "m")   # reloads from disk
    again = j2(prompts)
    assert len(calls) == 7                      # only the failed prompt is sent again
    assert all("error" not in r for r in again)


def test_agent_skips_failed_items_and_rejects_bad_replies():
    items, gens, oracle = make_synthetic(TARGETS, OBJECTIVES, n_items=6)
    agent = LLMJudgeAgent(AgentSpec("specialist", "pt", None, signal="llm_judge"),
                          oracle_judge(oracle, fail={items[0].item_id}))
    assert len(agent.evaluate_batch(items, gens)) == 5
    with pytest.raises(ValueError):
        LLMJudgeAgent(AgentSpec("consistency", "pt", None, other_lang="hi"), oracle_judge(oracle))
    with pytest.raises(ValueError):
        parse_judge({"divergence": "high", "collapse": True})
    assert parse_judge({"divergence": 1.7, "collapse": False})["divergence"] == 1.0
