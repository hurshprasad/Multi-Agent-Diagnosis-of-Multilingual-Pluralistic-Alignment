"""LLM-as-judge evaluator agents: System 6 in Table 6.

Owner: signals/eval track. Judge model: docs/decisions.md, D5 (`baselines.llm_judge` in the
config). It must not be on the oracle panel (D7), or Table 6 would score it against itself.

System 6 (`llm_judge_team`) is the fixed specialist team of System 5 with the cheap signal
swapped for an LLM judge: one judge agent per (language, objective), the same per-agent budget
and the same seeded item sample as System 5. So 5 vs 6 isolates the evaluator (cheap signal vs
LLM), and 6 vs 7 asks whether the adaptive lightweight population matches an LLM-judge
population at a fraction of the tokens.

Every judge reply is cached on Drive (`CachedJudge`, keyed by model + prompt), so re-running
Table 6 costs nothing and a run that stops part way resumes. Each Evaluation carries the
cost_tokens of its original call, cached or not: the cost of the method, not of this re-run.

Also usable inside the lifecycle coordinator (agents.evaluator_type: llm_judge), as an
ablation; pass the judge to build_agent.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from collections.abc import Callable
from pathlib import Path

from mpac.agents.base import Agent, AgentSpec, Evaluation
from mpac.agents.prompts.evo_templates import JUDGE_SYSTEM, JUDGE_USER
from mpac.data.schema import Generation, ParallelItem

LANG_NAMES = {"en": "English", "pt": "Portuguese", "es": "Spanish", "it": "Italian",
              "de": "German", "fr": "French", "hi": "Hindi"}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "divergence": {"type": "number"},
        "collapse": {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["divergence", "collapse", "evidence"],
    "additionalProperties": False,
}

# [(system, user)] -> one dict per prompt: divergence, collapse, evidence, cost_tokens,
# or {"error": ...} when that prompt couldn't be judged.
Judge = Callable[[list[tuple[str, str]]], list[dict]]


def judge_prompt(item: ParallelItem, lang: str, gens: dict[str, dict[str, Generation]],
                 reference: str = "en") -> tuple[str, str] | None:
    """(system, user) for judging `lang` against the reference on one item; None if a
    response is missing."""
    a = gens.get(reference, {}).get(item.item_id)
    b = gens.get(lang, {}).get(item.item_id)
    if a is None or b is None or lang not in item.prompts or reference not in item.prompts:
        return None
    name = LANG_NAMES.get(lang, lang)
    system = JUDGE_SYSTEM.format(lang_name=name, objective=item.objective)
    user = JUDGE_USER.format(prompt_en=item.prompts[reference], response_en=a.response,
                             prompt_tgt=item.prompts[lang], response_tgt=b.response,
                             objective=item.objective, lang_name=name)
    return system, user


def parse_judge(d: dict) -> dict:
    """Validate a judge reply; raises ValueError."""
    if not isinstance(d.get("collapse"), bool):
        raise ValueError("collapse missing or not a boolean")
    try:
        div = float(d["divergence"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("divergence missing or not a number") from e
    return {"divergence": min(1.0, max(0.0, div)), "collapse": d["collapse"],
            "evidence": str(d.get("evidence", ""))}


def anthropic_judge(model: str = "claude-haiku-5-5", effort: str = "medium",
                    workers: int = 16) -> Judge:
    """Claude through the Anthropic API, JSON constrained to JUDGE_SCHEMA, prompts sent in
    parallel. A failed or refused prompt comes back as {"error": ...} and is retried next run."""
    from concurrent.futures import ThreadPoolExecutor

    import anthropic

    client = anthropic.Anthropic(max_retries=5)

    def one(p: tuple[str, str]) -> dict:
        system, user = p
        try:
            resp = client.messages.create(
                model=model, max_tokens=4000, system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": effort,
                               "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
            )
        except anthropic.APIStatusError as e:
            if e.status_code in (401, 403, 404):
                raise   # wrong key or model id: stop instead of logging every prompt as an error
            return {"error": f"{type(e).__name__} {e.status_code}: {e.message}"}
        except anthropic.APIConnectionError as e:
            return {"error": f"connection: {e}"}
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        if resp.stop_reason != "end_turn":
            return {"error": f"stop_reason {resp.stop_reason}", "cost_tokens": tokens}
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return {**parse_judge(json.loads(text)), "cost_tokens": tokens, "model": resp.model}
        except (ValueError, json.JSONDecodeError) as e:
            return {"error": str(e), "raw": text[:300], "cost_tokens": tokens}

    def judge(prompts: list[tuple[str, str]]) -> list[dict]:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, prompts))

    return judge


class CachedJudge:
    """Wraps a Judge with a JSONL cache (one line per successful reply, keyed by model and
    prompt). Errors aren't cached, so they are retried on the next call."""

    def __init__(self, judge: Judge, path: str | Path, model: str):
        self.judge, self.path, self.model = judge, Path(path), model
        self.cache: dict[str, dict] = {}
        self.n_calls = self.n_errors = 0
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r["reply"]

    def key(self, p: tuple[str, str]) -> str:
        return hashlib.sha1(json.dumps([self.model, *p], ensure_ascii=False).encode()).hexdigest()

    def __call__(self, prompts: list[tuple[str, str]]) -> list[dict]:
        keys = [self.key(p) for p in prompts]
        todo = list({k: p for k, p in zip(keys, prompts) if k not in self.cache}.items())
        if todo:
            replies = self.judge([p for _, p in todo])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                for (k, _), r in zip(todo, replies):
                    self.n_calls += 1
                    if "error" in r:
                        self.n_errors += 1
                        continue
                    self.cache[k] = r
                    f.write(json.dumps({"key": k, "reply": r}, ensure_ascii=False) + "\n")
        return [self.cache.get(k, {"error": "not judged"}) for k in keys]


class LLMJudgeAgent(Agent):
    """A specialist (one language, one objective or all) that asks the judge instead of
    computing a divergence. Only specialist agents (vs the reference) are supported."""

    def __init__(self, spec: AgentSpec, judge: Judge | None = None):
        super().__init__(spec)
        if spec.kind != "specialist":
            raise ValueError("LLM judge agents are specialists (compared with the reference)")
        if judge is None:
            raise ValueError("LLMJudgeAgent needs a judge (baselines.llm_judge.anthropic_judge)")
        self.judge = judge

    def evaluate_batch(self, items: list[ParallelItem],
                       gens: dict[str, dict[str, Generation]]) -> list[Evaluation]:
        s = self.spec
        todo = [(it, p) for it in items if s.covers(it)
                and (p := judge_prompt(it, s.lang, gens, s.reference)) is not None]
        replies = self.judge([p for _, p in todo]) if todo else []
        out = []
        for (it, _), r in zip(todo, replies):
            if "error" in r:
                continue   # unevaluated; Table 6 counts labelled items never evaluated as missed
            out.append(Evaluation(self.agent_id, it.item_id, s.lang, it.objective,
                                  r["divergence"], r["collapse"], evidence=r["evidence"],
                                  cost_tokens=int(r.get("cost_tokens", 0)),
                                  compared_to=s.reference))
        return out

    def evaluate(self, item, gens, embeddings=None) -> Evaluation | None:
        out = self.evaluate_batch([item], gens)
        return out[0] if out else None


def llm_judge_team(items: list[ParallelItem], gens: dict[str, dict[str, Generation]],
                   targets: list[str], objectives: list[str], judge: Judge,
                   reference: str = "en", per_agent_budget: int | None = None,
                   seed: int = 42) -> list[Evaluation]:
    """System 6: one LLM-judge specialist per (language, objective), each judging the same
    seeded sample of items that System 5's agent for that cell evaluates
    (baselines/static.fixed_multi_agent draws with the same seed in the same order)."""
    rng = random.Random(seed)
    out = []
    for lang in targets:
        for obj in objectives:
            agent = LLMJudgeAgent(AgentSpec("specialist", lang, obj, reference=reference,
                                            signal="llm_judge"), judge)
            scope = [it for it in items if agent.spec.covers(it)]
            if per_agent_budget is not None:
                scope = rng.sample(scope, min(per_agent_budget, len(scope)))
            out.extend(agent.evaluate_batch(scope, gens))
    return out


def evolve_population(*args, **kwargs):
    """Not implemented: EvoAgent's LLM-driven mutation (MUTATE) and quality-check selection
    (QUALITY_CHECK) over judge personas. System 6 is the fixed LLM-judge team above; an evolved
    LLM population would be an additional row, see docs/decisions.md D5."""
    raise NotImplementedError
