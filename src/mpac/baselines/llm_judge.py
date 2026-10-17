"""LLM-as-judge evaluator agents (the EvoAgent-style baseline).

Owner: signals/eval track.

Two ways this baseline is used:
1. Set agents.evaluator_type=llm_judge: the same LifecycleCoordinator runs, but every agent
   scores items with an LLM instead of cheap signals. Isolates the evaluator's contribution.
2. evolve_population(): EvoAgent's own LLM-driven mutation + quality-check selection, for the
   "LLM-judge multi-agent system" row in Table 6.

Token usage is recorded in Evaluation.cost_tokens so cost and latency can be compared.
"""

from __future__ import annotations

from mpac.agents.base import Agent, Evaluation

LANG_NAMES = {"en": "English", "pt": "Portuguese", "es": "Spanish", "it": "Italian",
              "de": "German", "fr": "French"}


class LLMJudgeAgent(Agent):
    """TODO: implement `_call_judge` with the team's chosen judge model (docs/decisions.md, D5).

    Prompt templates are in agents/prompts/evo_templates.py (JUDGE_SYSTEM, JUDGE_USER).
    Parse the JSON reply into an Evaluation and set cost_tokens from the API/model usage.
    """

    def evaluate(self, item, gens, embeddings=None) -> Evaluation | None:
        raise NotImplementedError("LLM judge not wired up yet")


def evolve_population(*args, **kwargs):
    """TODO: EvoAgent loop — mutate evaluator descriptions (MUTATE), filter with QUALITY_CHECK,
    run for agents.lifecycle_cycles generations with population size capped at
    agents.population_cap. Mirrors the iteration / group size / selection-strategy knobs of
    EvoAgent's travelplanner/tools/planner/sole_planning.py.
    """
    raise NotImplementedError
