"""Table 6: the evaluator systems (4 single, 5 fixed, 6 LLM judge, 7 adaptive) on the same
cached outputs.

Owner: signals/eval track. System 6 runs only when a judge is passed (baselines/llm_judge.py,
D5); it judges the same items as System 5.

Rules this follows (CLAUDE.md):
- Every system reads the same generations, embeddings, thresholds and language IDs.
- Same evaluation budget: the adaptive run goes first and its total number of evaluations
  becomes the budget of the single evaluator and the fixed team (split evenly over its agents).
- Oracle labels: the adaptive utility sees only `tune` labels; every reported score uses only
  `test` labels, counting labelled items a system never evaluated as missed.
"""

from __future__ import annotations

from pathlib import Path

from mpac.data.schema import Generation, ParallelItem


def run_systems(cfg: dict, items: list[ParallelItem], gens: dict[str, dict[str, Generation]],
                oracle_tune: dict, oracle_test: dict, thresholds: dict | None = None,
                embeddings: dict | None = None, lang_ids: dict | None = None,
                registry_path=None, judge=None) -> dict:
    """One model + checkpoint. Returns {"systems": name -> scores, "adaptive": summary,
    "evals": name -> list of Evaluation} (evals feed review_rows, they aren't saved)."""
    from mpac.agents.coordinator import LifecycleCoordinator
    from mpac.agents.registry import AgentRegistry
    from mpac.analysis.metrics import coverage, detection_scores
    from mpac.baselines.static import fixed_multi_agent, single_evaluator

    targets, objectives = cfg["languages"]["targets"], cfg["objectives"]
    ref, seed = cfg["languages"]["reference"], cfg["project"]["seed"]
    coord = LifecycleCoordinator(cfg, items, gens, oracle_tune, embeddings,
                                 registry=AgentRegistry(registry_path),
                                 thresholds=thresholds, lang_ids=lang_ids)
    summary = coord.run()
    budget = summary["total_evaluations"]
    runs = {
        "4 single evaluator": single_evaluator(items, gens, targets, ref, budget=budget, seed=seed,
                                               embeddings=embeddings, thresholds=thresholds,
                                               lang_ids=lang_ids),
        "5 fixed multi-agent": fixed_multi_agent(items, gens, targets, objectives, ref,
                                                 per_agent_budget=budget // (len(targets) * len(objectives)),
                                                 seed=seed, embeddings=embeddings,
                                                 thresholds=thresholds, lang_ids=lang_ids),
        "7 adaptive lifecycle": coord.all_evals,
    }
    if judge is not None:
        from mpac.baselines.llm_judge import llm_judge_team

        runs["6 LLM-judge multi-agent"] = llm_judge_team(
            items, gens, targets, objectives, judge, ref,
            per_agent_budget=budget // (len(targets) * len(objectives)), seed=seed)
        runs = {k: runs[k] for k in sorted(runs)}
    systems = {}
    for name, evals in runs.items():
        cov = coverage(evals)
        systems[name] = {**detection_scores(evals, oracle_test, ref, count_unevaluated=True),
                         "n_evaluations": cov["n_evaluations"], "cells": cov["cells"],
                         "tokens": cov["tokens"]}
    summary = {k: v for k, v in summary.items() if k != "collapse_map"} | {
        "collapse_map": {f"{lang}/{obj}": v for (lang, obj), v in summary["collapse_map"].items()},
        "events": coord.registry.events}
    return {"systems": systems, "adaptive": summary, "budget": budget, "evals": runs}


REVIEW_TEXT_CHARS = 600   # responses are cut to this in the review sheet so it opens in Sheets


def review_rows(items: list[ParallelItem], gens: dict[str, dict[str, Generation]],
                evals: dict[str, list], thresholds: dict, splits: dict[str, str],
                oracle: dict, lang_ids: dict | None = None, reference: str = "en") -> list[dict]:
    """One row per (target language, item): the two responses, the signal, each system's
    verdict (flagged if any of its evaluations against the reference flagged it, as in
    detection_scores) and how often it looked, and the oracle label. For checking by hand why
    systems agree or differ."""
    from collections import defaultdict

    from mpac.signals.calibration import signal_kind

    seen: dict[str, dict[tuple[str, str], list]] = {}
    for name, es in evals.items():
        by = defaultdict(list)
        for e in es:
            if e.compared_to == reference:
                by[(e.lang, e.item_id)].append(e)
        seen[name] = by
    cut = lambda t: (t or "")[:REVIEW_TEXT_CHARS]
    rows = []
    for it in items:
        ref = gens.get(reference, {}).get(it.item_id)
        for lang, by_item in gens.items():
            g = by_item.get(it.item_id)
            if lang == reference or g is None or ref is None:
                continue
            kind = signal_kind(ref, g)
            divs = [e.divergence for name in seen for e in seen[name].get((lang, it.item_id), [])]
            label = oracle.get((lang, it.item_id, it.objective))
            row = {"lang": lang, "item_id": it.item_id, "objective": it.objective,
                   "split": splits.get(it.item_id, ""),
                   "oracle_collapsed": "" if label is None else label, "signal": kind,
                   "divergence": round(divs[0], 4) if divs else "",
                   "threshold": round(thresholds.get((lang, kind), float("nan")), 4),
                   "detected_lang": (lang_ids or {}).get(lang, {}).get(it.item_id, "")}
            verdicts = []
            for name in evals:
                es = seen[name].get((lang, it.item_id), [])
                v = any(e.collapse for e in es)
                verdicts.append(v if es else None)
                row[f"{name} | flagged"] = v if es else ""
                row[f"{name} | n_evals"] = len(es)
            row["systems_differ"] = len({v or False for v in verdicts}) > 1
            row |= {"prompt_en": cut(it.prompts[reference]), "response_en": cut(ref.response),
                    "prompt_target": cut(it.prompts[lang]), "response_target": cut(g.response)}
            rows.append(row)
    return rows


def per_cell_scores(evals, oracle_test: dict, reference: str = "en") -> dict[str, dict]:
    """Detection scores per (lang, objective) cell: where a system finds collapse and misses it."""
    from collections import defaultdict

    from mpac.analysis.metrics import detection_scores

    by_cell = defaultdict(dict)
    for (lang, iid, obj), y in oracle_test.items():
        by_cell[f"{lang}/{obj}"][(lang, iid, obj)] = y
    return {cell: detection_scores(evals, o, reference, count_unevaluated=True)
            for cell, o in sorted(by_cell.items())}


def load_signals(cfg: dict, model_id: str, checkpoint: str) -> dict:
    """Everything the lightweight evaluators read, from the Drive caches: items, generations,
    LaBSE embeddings (computed only if missing), language IDs and each language's prompt floor
    (from the saved acs_free_text table, notebook 01)."""
    from mpac.analysis.results import load_results
    from mpac.config import all_languages
    from mpac.data.loaders import load_items
    from mpac.generation.cache import GenerationCache
    from mpac.paths import subdir
    from mpac.signals.embeddings import embed_generations
    from mpac.signals.lightweight import LazyEmbedder, detect_languages

    items = load_items(cfg)
    gens = GenerationCache(subdir(cfg, "generations")).load_all(model_id, checkpoint,
                                                                all_languages(cfg))
    if not all(gens.values()):
        raise FileNotFoundError(f"no generations for {model_id}/{checkpoint}; run notebook 01")
    emb = embed_generations(gens, LazyEmbedder(), subdir(cfg, "embeddings"))
    saved = [r for r in load_results(cfg, "acs_free_text") if r["model_id"] == model_id]
    if not saved:
        raise FileNotFoundError("no acs_free_text results; run notebook 01's free-text cell")
    floor = {lang: v["prompt_floor"] for lang, v in saved[0]["data"].items()}
    return {"items": items, "gens": gens, "embeddings": emb, "floor": floor,
            "lang_ids": detect_languages(gens)}


def _resolved_labels(cfg: dict, round_name: str, model_id: str, checkpoint: str) -> tuple[dict, dict]:
    from mpac.data.oracle import drop_bad_translations, load_round, resolve, to_oracle
    from mpac.paths import subdir

    labels, adj = [], []
    for lab, ad in load_round(subdir(cfg, "oracle") / round_name).values():
        labels += lab
        adj += ad
    labels, bad = drop_bad_translations(labels, adj)
    resolved, left = resolve(labels, adj, cfg["oracle"]["annotators_per_item"])
    return to_oracle(resolved, model_id, checkpoint), {"unresolved": len(left), "bad_translation": len(bad)}


def load_oracle(cfg: dict, model_id: str, checkpoint: str, source: str = "human",
                round_name: str = "main", signals: dict | None = None,
                forced_choice_round: str | None = None) -> tuple[dict, dict, dict]:
    """(tune labels, test labels, info). source="human" or "llm": the resolved labels of
    oracle/<round_name>/ (filled by people, or by the LLM panel in data/llm_oracle.py; same
    layout). Items marked bad translation are dropped, and an item needs
    oracle.annotators_per_item annotators. "proxy": data.oracle.proxy_oracle (pipeline check
    only).

    forced_choice_round: for a _long checkpoint (free text re-generated, forced choice copied,
    generate.copy_forced_choice): `round_name` supplies the free-text labels, and this round's
    labels of the original checkpoint supply the forced-choice ones (same answers)."""
    from mpac.data.oracle import proxy_oracle
    from mpac.data.splits import get_or_create_splits

    if source == "proxy":
        s = signals or load_signals(cfg, model_id, checkpoint)
        splits = get_or_create_splits(cfg, s["items"])
        allo = proxy_oracle(s["items"], s["gens"], cfg["languages"]["reference"])
        info = {"source": "proxy (TruthfulQA answers, honesty only): NOT REPORTABLE"}
    elif source in ("human", "llm"):
        from mpac.data.loaders import load_items
        from mpac.generation.generate import weights_checkpoint

        items = load_items(cfg)
        splits = get_or_create_splits(cfg, items)
        allo, counts = _resolved_labels(cfg, round_name, model_id, checkpoint)
        info = {"source": f"{source}, round {round_name}", **counts}
        if forced_choice_round:
            fc = {it.item_id for it in items if it.is_forced_choice}
            allo = {k: v for k, v in allo.items() if k[1] not in fc}
            more, c2 = _resolved_labels(cfg, forced_choice_round, model_id,
                                        weights_checkpoint(checkpoint))
            allo |= {k: v for k, v in more.items() if k[1] in fc}
            info |= {"source": f"{source}, round {round_name} (free text) + "
                               f"{forced_choice_round} (forced choice)",
                     "forced_choice": c2}
    else:
        raise ValueError("source must be 'human', 'llm' or 'proxy'")
    tune = {k: v for k, v in allo.items() if splits.get(k[1]) == "tune"}
    test = {k: v for k, v in allo.items() if splits.get(k[1]) == "test"}
    info |= {"n_tune": len(tune), "n_test": len(test), "n_collapsed_test": sum(test.values())}
    return tune, test, info


def make_judge(cfg: dict):
    """The System 6 judge from `baselines.llm_judge`, cached under <root>/judge/ (Drive)."""
    from mpac.baselines.llm_judge import CachedJudge, anthropic_judge
    from mpac.paths import subdir

    c = cfg["baselines"]["llm_judge"]
    panel = {**cfg["oracle"]["llm_panel"]["annotators"], **cfg["oracle"]["llm_panel"]["adjudicator"]}
    if c["model"] in panel.values():
        raise ValueError(f"judge {c['model']} is on the oracle panel (D5/D7)")
    return CachedJudge(anthropic_judge(c["model"], c["effort"]),
                       subdir(cfg, "judge") / f"{c['model']}.jsonl", c["model"])


def table6(cfg: dict, model_id: str, checkpoint: str, source: str = "human",
           round_name: str = "main", embedding_margin: float = 0.15, judge=None,
           forced_choice_round: str | None = None) -> dict:
    """Thresholds (floor start, tuned on tune labels), then every system on the same signals
    (System 6 only with a `judge`, e.g. make_judge(cfg)); saved to
    results/<model>/<checkpoint>/table6_<source>.json."""
    from mpac.analysis.results import save_results
    from mpac.signals.acs import item_divergences
    from mpac.signals.calibration import floor_thresholds, tune_thresholds

    s = load_signals(cfg, model_id, checkpoint)
    tune, test, info = load_oracle(cfg, model_id, checkpoint, source, round_name, s,
                                   forced_choice_round)
    if not test:
        raise ValueError(f"no test-split labels for {model_id}/{checkpoint} ({info})")
    ref, targets = cfg["languages"]["reference"], cfg["languages"]["targets"]
    start = floor_thresholds(targets, s["floor"], embedding_margin=embedding_margin)
    divs = {}
    for kind, signal in (("options", "options"), ("embedding", "embedding")):
        for lang, d in item_divergences(s["gens"], ref, s["embeddings"], signal).items():
            divs[(lang, kind)] = d
    thresholds, th_report = tune_thresholds(divs, tune, start)
    out = run_systems(cfg, s["items"], s["gens"], tune, test, thresholds, s["embeddings"],
                      s["lang_ids"], judge=judge)
    if judge is not None and hasattr(judge, "n_errors"):
        out["judge"] = {"model": getattr(judge, "model", ""), "calls": judge.n_calls,
                        "errors": judge.n_errors}
    wrong_lang = {lang: sum(v != lang for v in ids.values())
                  for lang, ids in s["lang_ids"].items() if lang != ref}
    out |= {"oracle": info, "wrong_language_responses": wrong_lang,
            "thresholds": {f"{lang}/{kind}": v for (lang, kind), v in th_report.items()}}
    evals = out.pop("evals")
    out["path"] = str(save_results(cfg, model_id, checkpoint, f"table6_{source}", out))
    import pandas as pd

    from mpac.data.splits import get_or_create_splits

    rows = review_rows(s["items"], s["gens"], evals, thresholds,
                       get_or_create_splits(cfg, s["items"]), tune | test, s["lang_ids"], ref)
    review = Path(out["path"]).with_name(f"table6_{source}_review.csv")
    pd.DataFrame(rows).to_csv(review, index=False, encoding="utf-8")
    out["review_path"] = str(review)
    out["n_pairs_systems_differ"] = sum(r["systems_differ"] for r in rows)
    return out


# --------------------------------------------------------------------------- routing (post hoc)

CHEAP = "4 single evaluator"
JUDGE = "6 LLM-judge multi-agent"


def _prf(pairs: list[tuple[bool, bool]]) -> dict:
    """[(flagged, oracle)] -> precision, recall, f1, n, n_collapsed."""
    tp = sum(f and y for f, y in pairs)
    nf, ny = sum(f for f, _ in pairs), sum(y for _, y in pairs)
    p, r = (tp / nf if nf else 0.0), (tp / ny if ny else 0.0)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "n": len(pairs), "n_collapsed": ny}


def route_by_cell(review: list[dict], judge_tokens: float | None = None,
                  cheap: str = CHEAP, judge: str = JUDGE) -> dict:
    """Routed evaluator from the Table 6 review sheet (`table6_<source>_review.csv` rows).

    Rule, fixed before scoring: per (language, signal) cell, use the judge's verdict only if its
    F1 on **tune** labels is strictly higher than the cheap signal's; otherwise (ties, or no
    collapsed tune label) keep the zero-cost signal. The routed system is then scored once on
    **test** labels, next to both inputs scored the same way. Unvisited pairs count as not
    flagged. `judge_tokens` (System 6's total) gives a cost estimate: its mean tokens per pair
    times the pairs routed to the judge. Suggested by the test split (results.md, Table 6), so
    report it as a follow-up system, with this rule."""
    def flag(r, name):
        return str(r.get(f"{name} | flagged", "")).strip() == "True"

    def label(r):
        v = str(r.get("oracle_collapsed", "")).strip()
        return None if v not in ("True", "False") else v == "True"

    cells = sorted({(r["lang"], r["signal"]) for r in review})
    route, tune_scores = {}, {}
    for cell in cells:
        rows = [r for r in review if (r["lang"], r["signal"]) == cell and r.get("split") == "tune"
                and label(r) is not None]
        c = _prf([(flag(r, cheap), label(r)) for r in rows])
        j = _prf([(flag(r, judge), label(r)) for r in rows])
        tune_scores[f"{cell[0]}/{cell[1]}"] = {"cheap_f1": c["f1"], "judge_f1": j["f1"],
                                               "n": c["n"], "n_collapsed": c["n_collapsed"]}
        route[cell] = judge if c["n_collapsed"] and j["f1"] > c["f1"] else cheap

    def verdict(r, name):
        return flag(r, route[(r["lang"], r["signal"])]) if name == "routed" else flag(r, name)

    test = [r for r in review if r.get("split") == "test" and label(r) is not None]
    out = {"route": {f"{lang}/{sig}": name for (lang, sig), name in route.items()},
           "tune": tune_scores, "test": {}, "test_by_cell": {}}
    for name in ("routed", cheap, judge):
        out["test"][name] = _prf([(verdict(r, name), label(r)) for r in test])
        for cell in cells:
            rows = [r for r in test if (r["lang"], r["signal"]) == cell]
            out["test_by_cell"].setdefault(f"{cell[0]}/{cell[1]}", {})[name] = \
                _prf([(verdict(r, name), label(r)) for r in rows])["f1"]
    n_judged = sum(str(r.get(f"{judge} | n_evals", "0")) not in ("", "0") for r in review)
    n_routed = sum(route[(r["lang"], r["signal"])] == judge for r in review)
    out["pairs_to_judge"] = {"routed": n_routed, "all": len(review)}
    if judge_tokens is not None and n_judged:
        out["tokens_estimate"] = {"routed": judge_tokens / n_judged * n_routed, "judge": judge_tokens}
    return out
