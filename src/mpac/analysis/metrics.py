"""System-level metrics for comparing evaluators (Table 6) against the oracle labels.

Owner: signals/eval track.
"""

from __future__ import annotations

from collections.abc import Iterable

from mpac.agents.base import Evaluation


def detection_scores(evals: Iterable[Evaluation], oracle: dict[tuple[str, str, str], bool],
                     reference: str = "en", count_unevaluated: bool = False) -> dict:
    """Precision / recall / F1 of collapse flags against oracle labels, on labelled items only.

    When several agents evaluate the same (lang, item, objective), the system flags collapse
    if any agent does (the population acts as a union detector). Only evaluations made against
    the reference language count; target-vs-target consistency evidence is excluded.
    count_unevaluated=True scores every labelled item, counting ones the system never looked
    at as "not flagged", so a system can't raise recall by evaluating fewer items (Table 6).
    """
    verdict: dict[tuple[str, str, str], bool] = {}
    for e in evals:
        if e.compared_to != reference:
            continue
        key = (e.lang, e.item_id, e.objective)
        if key in oracle:
            verdict[key] = verdict.get(key, False) or e.collapse
    n_evaluated = len(verdict)
    if count_unevaluated:
        verdict = {k: verdict.get(k, False) for k in oracle}
    tp = sum(v and oracle[k] for k, v in verdict.items())
    fp = sum(v and not oracle[k] for k, v in verdict.items())
    fn = sum((not v) and oracle[k] for k, v in verdict.items())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_labelled": len(verdict),
            "n_labelled_evaluated": n_evaluated}


def coverage(evals: Iterable[Evaluation]) -> dict:
    """Languages and (language, objective) cells covered by at least one evaluation."""
    evals = list(evals)
    return {
        "languages": sorted({e.lang for e in evals}),
        "cells": len({(e.lang, e.objective) for e in evals}),
        "tokens": sum(e.cost_tokens for e in evals),
        "n_evaluations": len(evals),
    }
