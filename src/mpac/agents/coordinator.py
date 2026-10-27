"""Adaptive lifecycle coordinator: spawn, select, retain, destroy.

Owner: agents track.

One cycle:
  1. SELECT   every live agent evaluates a batch of in-scope items from the shared cache.
  2. SCORE    utility = accuracy on oracle-labelled items in the batch, plus a novelty bonus
              for collapse flags no other live agent raised (complementary evidence).
  3. RETAIN / DESTROY
              - utility below retain_threshold for `patience` consecutive cycles -> destroy
              - same scope (languages + objective) as a higher-utility agent and collapse
                verdicts agree > redundancy_threshold on shared items -> destroy (redundant)
              - destroyed agents can't be respawned for `patience` cycles (no churn)
  4. SPAWN    for every (lang, objective) cell where the share of collapse flags exceeds
              spawn_on_flag_rate and there is no live specialist, spawn one; when two
              target languages both diverge, spawn a consistency agent between them.
              Flags, not raw divergence: flags use per-language thresholds, raw divergence
              doesn't (LaBSE puts every Hindi response further from English than a
              Portuguese one), so a raw cut would call every Hindi cell hot.
              Population never exceeds population_cap.

All decisions use cheap numbers, never an LLM. The LLM-as-judge baseline swaps the agents'
signal, not this logic, so the comparison isolates the evaluation mechanism.

The utility formula and thresholds are provisional (docs/decisions.md, D3).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from mpac.agents.base import AgentSpec, Evaluation, build_agent
from mpac.agents.registry import AgentRegistry
from mpac.data.schema import Generation, ParallelItem

OracleKey = tuple[str, str, str]  # (lang, item_id, objective) -> collapsed_vs_reference


@dataclass
class CycleLog:
    cycle: int
    n_alive: int
    n_evaluations: int
    tokens: int
    utilities: dict[str, float] = field(default_factory=dict)


class LifecycleCoordinator:
    def __init__(
        self,
        cfg: dict,
        items: list[ParallelItem],
        gens: dict[str, dict[str, Generation]],
        oracle: dict[OracleKey, bool] | None = None,
        embeddings: dict | None = None,
        registry: AgentRegistry | None = None,
        patience: int = 2,
        novelty_weight: float = 0.2,
        thresholds: dict | None = None,
        lang_ids: dict | None = None,
    ):
        a = cfg["agents"]
        self.cap = a["population_cap"]
        self.cycles = a["lifecycle_cycles"]
        self.batch = a["batch_per_cycle"]
        self.retain_threshold = a["retain_threshold"]
        self.redundancy_threshold = a["redundancy_threshold"]
        self.spawn_threshold = a["spawn_on_flag_rate"]
        self.signal = "llm_judge" if a.get("evaluator_type") == "llm_judge" else "divergence"
        self.reference = cfg["languages"]["reference"]
        self.targets = cfg["languages"]["targets"]
        self.items = items
        self.gens = gens
        self.oracle = oracle or {}
        self.embeddings = embeddings
        self.thresholds = thresholds
        self.lang_ids = lang_ids
        self.registry = registry or AgentRegistry()
        self.patience = patience
        self.novelty_weight = novelty_weight
        self.rng = random.Random(cfg["project"]["seed"])
        self.logs: list[CycleLog] = []
        self.all_evals: list[Evaluation] = []

    # setup ---------------------------------------------------------------
    def seed_population(self) -> None:
        """Start with one all-objective specialist per target language."""
        for lang in self.targets:
            spec = AgentSpec("specialist", lang, None, reference=self.reference, signal=self.signal)
            if not self.registry.has_alive(spec):
                self.registry.spawn(spec, cycle=0, reason="seed")

    # one cycle -----------------------------------------------------------
    def _evaluate(self, cycle: int) -> dict[str, list[Evaluation]]:
        out: dict[str, list[Evaluation]] = {}
        for st in self.registry.alive():
            agent = build_agent(st.spec, self.thresholds, self.lang_ids)
            scope = [it for it in self.items if st.spec.covers(it)]
            batch = self.rng.sample(scope, min(self.batch, len(scope)))
            evals = [e for it in batch if (e := agent.evaluate(it, self.gens, self.embeddings))]
            st.n_evaluated += len(evals)
            st.tokens_used += sum(e.cost_tokens for e in evals)
            out[st.spec.agent_id] = evals
            self.all_evals.extend(evals)
        return out

    def _oracle_label(self, e: Evaluation) -> bool | None:
        """Ground truth for an evaluation. Against the reference it is the oracle label itself;
        for a target-vs-target comparison the two languages diverge iff exactly one collapsed."""
        key = (e.lang, e.item_id, e.objective)
        if key not in self.oracle:
            return None
        if e.compared_to == self.reference:
            return self.oracle[key]
        other = (e.compared_to, e.item_id, e.objective)
        return self.oracle[key] != self.oracle[other] if other in self.oracle else None

    def _utilities(self, evals: dict[str, list[Evaluation]]) -> dict[str, float]:
        flaggers: dict[tuple[str, str], set[str]] = defaultdict(set)
        for aid, es in evals.items():
            for e in es:
                if e.collapse:
                    flaggers[(e.lang, e.item_id)].add(aid)
        utils = {}
        for aid, es in evals.items():
            st = self.registry.states[aid]
            correct = seen = 0
            for e in es:
                label = self._oracle_label(e)
                if label is not None:
                    seen += 1
                    correct += int(e.collapse == label)
            st.n_oracle_seen += seen
            st.n_oracle_correct += correct
            acc = correct / seen if seen else (st.oracle_accuracy or 0.5)
            flags = [e for e in es if e.collapse]
            # Share of this agent's flags that no other agent raised; neutral 0.5 if it raised none.
            novelty = (sum(len(flaggers[(e.lang, e.item_id)]) == 1 for e in flags) / len(flags)
                       if flags else 0.5)
            utils[aid] = (1 - self.novelty_weight) * acc + self.novelty_weight * novelty
        return utils

    def _retain_or_destroy(self, cycle: int, evals: dict[str, list[Evaluation]], utils: dict[str, float]) -> None:
        for aid, u in utils.items():
            st = self.registry.states[aid]
            st.utility_history.append(u)
            st.low_utility_streak = st.low_utility_streak + 1 if u < self.retain_threshold else 0
        # redundancy: compare verdicts on shared items
        verdicts = {aid: {(e.lang, e.item_id, e.compared_to): e.collapse for e in es}
                    for aid, es in evals.items()}
        ids = sorted(utils, key=utils.get, reverse=True)
        for i, better in enumerate(ids):
            if not self.registry.states[better].alive:
                continue
            for worse in ids[i + 1:]:
                if not self.registry.states[worse].alive:
                    continue
                if self._scope(worse) != self._scope(better):
                    continue
                shared = verdicts[better].keys() & verdicts[worse].keys()
                if len(shared) < 5:
                    continue
                agree = np.mean([verdicts[better][k] == verdicts[worse][k] for k in shared])
                if agree > self.redundancy_threshold:
                    self.registry.destroy(worse, cycle, f"redundant with {better} ({agree:.2f})")
        for aid, u in utils.items():
            st = self.registry.states[aid]
            if st.alive and st.low_utility_streak >= self.patience:
                self.registry.destroy(aid, cycle, f"low utility x{st.low_utility_streak}")
            elif st.alive:
                self.registry.log("retain", aid, cycle, f"utility={u:.2f}")

    def _scope(self, agent_id: str) -> tuple:
        """Agents are only redundant if they judge exactly the same thing (same languages and
        objective) and differ just in signal or persona. A narrower specialist is never
        'redundant' with a generalist: its value is the extra sampling budget in its cell, and
        removing a generalist could leave a language's other objectives uncovered."""
        s = self.registry.states[agent_id].spec
        return (s.kind, s.lang, s.objective, s.other_lang, s.reference)

    def _cooling_down(self, spec: AgentSpec, cycle: int) -> bool:
        """Don't respawn an agent destroyed in the last `patience` cycles (prevents churn)."""
        st = self.registry.states.get(spec.agent_id)
        return bool(st and not st.alive and st.died_cycle is not None
                    and cycle - st.died_cycle < self.patience)

    def _spawn(self, cycle: int, evals: dict[str, list[Evaluation]]) -> None:
        cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        for es in evals.values():
            for e in es:
                if e.compared_to == self.reference:
                    cell[(e.lang, e.objective)].append(float(e.collapse))
        hot = sorted(((np.mean(v), k) for k, v in cell.items() if np.mean(v) > self.spawn_threshold),
                     reverse=True)
        for _, (lang, obj) in hot:
            if len(self.registry.alive()) >= self.cap:
                return
            spec = AgentSpec("specialist", lang, obj, reference=self.reference, signal=self.signal)
            if not self.registry.has_alive(spec) and not self._cooling_down(spec, cycle):
                self.registry.spawn(spec, cycle, f"unexplained divergence in {lang}/{obj}")
        hot_langs = list(dict.fromkeys(lang for _, (lang, _) in hot))
        for pair in pairwise(hot_langs):
            a, b = sorted(pair)  # one agent per unordered pair
            if len(self.registry.alive()) >= self.cap:
                return
            spec = AgentSpec("consistency", a, None, other_lang=b, reference=self.reference,
                             signal=self.signal)
            if not self.registry.has_alive(spec) and not self._cooling_down(spec, cycle):
                self.registry.spawn(spec, cycle, f"co-divergence {a}/{b}")

    # main loop -----------------------------------------------------------
    def run(self) -> dict:
        if not self.registry.alive():
            self.seed_population()
        for cycle in range(1, self.cycles + 1):
            evals = self._evaluate(cycle)
            utils = self._utilities(evals)
            self._retain_or_destroy(cycle, evals, utils)
            self._spawn(cycle, evals)
            self.logs.append(CycleLog(cycle, len(self.registry.alive()),
                                      sum(map(len, evals.values())),
                                      sum(e.cost_tokens for es in evals.values() for e in es), utils))
            self.registry.save()
        return self.summary()

    def collapse_map(self) -> dict[tuple[str, str], dict[str, float]]:
        """(lang, objective) -> mean divergence and collapse-flag rate over all evaluations."""
        cell: dict[tuple[str, str], list[Evaluation]] = defaultdict(list)
        for e in self.all_evals:
            if e.compared_to == self.reference:
                cell[(e.lang, e.objective)].append(e)
        return {k: {"mean_divergence": float(np.mean([e.divergence for e in v])),
                    "collapse_rate": float(np.mean([e.collapse for e in v])),
                    "n": len(v)} for k, v in cell.items()}

    def summary(self) -> dict:
        return {
            "final_alive": [s.spec.agent_id for s in self.registry.alive()],
            "lifecycle_ops": self.registry.op_counts(),
            "population_by_cycle": [log.n_alive for log in self.logs],
            "total_evaluations": len(self.all_evals),
            "total_tokens": sum(log.tokens for log in self.logs),
            "collapse_map": self.collapse_map(),
        }
