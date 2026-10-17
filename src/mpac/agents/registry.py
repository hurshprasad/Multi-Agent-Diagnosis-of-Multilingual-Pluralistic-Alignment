"""External agent-state store: one JSON file plus an append-only lifecycle event log.

JSON rather than a database so it can sit on Google Drive and be inspected by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from mpac.agents.base import AgentSpec, AgentState
from mpac.data.io import write_jsonl


class AgentRegistry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.states: dict[str, AgentState] = {}
        self.events: list[dict] = []   # spawn / retain / destroy log, used for lifecycle metrics
        if self.path and self.path.exists():
            self.load()

    # population ----------------------------------------------------------
    def alive(self) -> list[AgentState]:
        return [s for s in self.states.values() if s.alive]

    def has_alive(self, spec: AgentSpec) -> bool:
        st = self.states.get(spec.agent_id)
        return bool(st and st.alive)

    def spawn(self, spec: AgentSpec, cycle: int, reason: str) -> AgentState:
        st = AgentState(spec=spec, born_cycle=cycle)
        self.states[spec.agent_id] = st
        self.log("spawn", spec.agent_id, cycle, reason)
        return st

    def destroy(self, agent_id: str, cycle: int, reason: str) -> None:
        st = self.states[agent_id]
        st.alive, st.died_cycle, st.death_reason = False, cycle, reason
        self.log("destroy", agent_id, cycle, reason)

    def log(self, op: str, agent_id: str, cycle: int, reason: str = "") -> None:
        self.events.append({"op": op, "agent_id": agent_id, "cycle": cycle, "reason": reason})

    def op_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e["op"]] = counts.get(e["op"], 0) + 1
        return counts

    # persistence ---------------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"states": {k: v.to_dict() for k, v in self.states.items()}}, f, indent=2)
        write_jsonl(self.path.with_suffix(".events.jsonl"), self.events)

    def load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        self.states = {k: AgentState.from_dict(v) for k, v in data["states"].items()}
        ev = self.path.with_suffix(".events.jsonl")
        if ev.exists():
            from mpac.data.io import read_jsonl

            self.events = list(read_jsonl(ev))
