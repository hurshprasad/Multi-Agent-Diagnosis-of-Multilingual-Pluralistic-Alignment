"""Saved result tables, so a score computed once in Colab never has to be recomputed.

Owner: signals/eval track.

Layout: <root>/results/<model_id>/<checkpoint>/<name>.json, each holding
{"name", "model_id", "checkpoint", "data", "meta": {saved_at, git_commit, drive_subdir, ...}}.
Saving again with the same name overwrites that table (the latest run wins).
load_results() gathers every table, e.g. to put base, en_aligned and multi_aligned side by side.

Also here: TruthfulQA accuracy against the English reference, with an exact McNemar test on
the items that flipped, which is how honesty collapse is reported (in addition to ACS).
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def mcnemar_exact(lost: int, gained: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of lost vs gained at p = 0.5."""
    n = lost + gained
    if n == 0:
        return float("nan")
    k = min(lost, gained)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def accuracy_vs_reference(gens: dict, items: list, reference: str = "en",
                          ids: set[str] | None = None) -> dict[str, dict[str, float]]:
    """Per target language: accuracy in the reference and target language on forced-choice
    items with a known answer (meta["correct"], i.e. TruthfulQA), how many items were lost or
    gained relative to the reference, and the McNemar p-value. ids restricts the items, e.g.
    to the verified subset."""
    by_id = {it.item_id: it for it in items}
    scored = [i for i, g in gens.get(reference, {}).items()
              if g.option_probs is not None and i in by_id and "correct" in by_id[i].meta
              and (ids is None or i in ids)]

    def right(lang: str, i: str) -> bool:
        return int(np.argmax(gens[lang][i].option_probs)) == by_id[i].meta["correct"]

    out = {}
    for lang, by_item in gens.items():
        if lang == reference:
            continue
        both = [i for i in scored if i in by_item and by_item[i].option_probs is not None]
        if not both:
            continue
        ref, tgt = [right(reference, i) for i in both], [right(lang, i) for i in both]
        lost = sum(r and not t for r, t in zip(ref, tgt))
        gained = sum(t and not r for r, t in zip(ref, tgt))
        out[lang] = {"n": len(both), "acc_reference": float(np.mean(ref)),
                     "acc": float(np.mean(tgt)), "lost": lost, "gained": gained,
                     "mcnemar_p": mcnemar_exact(lost, gained)}
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # no git, or not a checkout
        return None


def _results_dir(cfg: dict) -> Path:
    from mpac.paths import subdir

    return subdir(cfg, "results")


def save_results(cfg: dict, model_id: str, checkpoint: str, name: str, data: dict,
                 **meta) -> Path:
    """Write one result table to Drive, with when, from which commit and with which settings."""
    path = _results_dir(cfg) / model_id / checkpoint / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "model_id": model_id, "checkpoint": checkpoint, "data": data,
              "meta": {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "git_commit": _git_commit(),
                       "drive_subdir": cfg["project"].get("drive_subdir"),
                       "languages": cfg["languages"], **meta}}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=float))
    return path


def load_results(cfg: dict, name: str | None = None) -> list[dict]:
    """Every saved table (optionally only those called `name`), sorted by model and checkpoint."""
    root = _results_dir(cfg)
    paths = sorted(root.glob(f"*/*/{name or '*'}.json"))
    return [json.loads(p.read_text()) for p in paths]


def results_table(cfg: dict, name: str):
    """One saved table across every model and checkpoint, as a DataFrame indexed by
    (model_id, checkpoint, lang): the before / after comparison for DPO."""
    import pandas as pd

    rows = []
    for r in load_results(cfg, name):
        for lang, vals in r["data"].items():
            if isinstance(vals, dict):
                rows.append({"model_id": r["model_id"], "checkpoint": r["checkpoint"],
                             "lang": lang, **vals})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["model_id", "checkpoint", "lang"]).sort_index()
