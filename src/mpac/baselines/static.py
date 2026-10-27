"""Non-adaptive evaluator baselines.

Owner: signals/eval track.

- single_evaluator: one evaluator, one threshold, applied to every language and objective.
- fixed_multi_agent: one specialist per (language, objective), all kept for the whole run,
  with the same per-agent evaluation budget as the adaptive system.

Both read the same cached generations as the adaptive coordinator, and take the same optional
per-language thresholds and language IDs (agents/base.LightweightAgent), so a difference in
Table 6 comes from how the population is organised, not from better-calibrated signals.
"""

from __future__ import annotations

import random

from mpac.agents.base import AgentSpec, Evaluation, LightweightAgent
from mpac.data.schema import Generation, ParallelItem


def single_evaluator(
    items: list[ParallelItem], gens: dict[str, dict[str, Generation]], targets: list[str],
    reference: str = "en", budget: int | None = None, threshold: float = 0.3, seed: int = 42,
    embeddings: dict | None = None, thresholds: dict | None = None, lang_ids: dict | None = None,
) -> list[Evaluation]:
    """One evaluator over all (item, language) pairs, optionally capped at `budget` evaluations."""
    pairs = [(it, lang) for it in items for lang in targets]
    if budget is not None:
        pairs = random.Random(seed).sample(pairs, min(budget, len(pairs)))
    out = []
    for it, lang in pairs:
        agent = LightweightAgent(AgentSpec("specialist", lang, None, reference=reference,
                                           collapse_threshold=threshold), thresholds, lang_ids)
        if e := agent.evaluate(it, gens, embeddings):
            out.append(Evaluation("single", e.item_id, e.lang, e.objective, e.divergence,
                                  e.collapse, evidence=e.evidence))
    return out


def fixed_multi_agent(
    items: list[ParallelItem], gens: dict[str, dict[str, Generation]], targets: list[str],
    objectives: list[str], reference: str = "en", per_agent_budget: int | None = None,
    seed: int = 42, embeddings: dict | None = None, thresholds: dict | None = None,
    lang_ids: dict | None = None,
) -> list[Evaluation]:
    rng = random.Random(seed)
    out = []
    for lang in targets:
        for obj in objectives:
            agent = LightweightAgent(AgentSpec("specialist", lang, obj, reference=reference),
                                     thresholds, lang_ids)
            scope = [it for it in items if agent.spec.covers(it)]
            if per_agent_budget is not None:
                scope = rng.sample(scope, min(per_agent_budget, len(scope)))
            out.extend(e for it in scope if (e := agent.evaluate(it, gens, embeddings)))
    return out
