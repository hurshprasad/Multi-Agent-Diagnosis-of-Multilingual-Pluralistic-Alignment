"""Tune / test split of evaluation items, assigned once and frozen.

Owner: data/training track.

Rule this enforces (CLAUDE.md): oracle labels are never used to tune thresholds or the
lifecycle utility on the test split. Thresholds, the D3 utility and any signal calibration
read only `tune` labels; every reported detection metric reads only `test` labels.

Splits are by item_id, not by (language, item): every language version of an item lands in
the same split, so an agent can't learn an item in Spanish and be scored on it in German.
Assignment is stratified by objective and deterministic given the seed, and is saved to
<root>/splits/splits.json the first time. After that it is only ever loaded, never redrawn.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from mpac.data.schema import ParallelItem

SPLITS = ("tune", "test")


def assign_splits(items: list[ParallelItem], tune_frac: float = 1 / 3,
                  seed: int = 42) -> dict[str, str]:
    """item_id -> "tune" | "test", stratified by objective."""
    by_obj: dict[str, list[str]] = defaultdict(list)
    for it in items:
        by_obj[it.objective].append(it.item_id)
    rng = random.Random(seed)
    out = {}
    for obj in sorted(by_obj):
        ids = sorted(set(by_obj[obj]))   # sort first so input order can't change the draw
        rng.shuffle(ids)
        n_tune = round(tune_frac * len(ids))
        out.update({iid: "tune" if i < n_tune else "test" for i, iid in enumerate(ids)})
    return out


def splits_path(cfg: dict) -> Path:
    from mpac.paths import subdir

    return subdir(cfg, "splits") / "splits.json"


def save_splits(splits: dict[str, str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=1, sort_keys=True))


def load_splits(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text())


def get_or_create_splits(cfg: dict, items: list[ParallelItem],
                         path: str | Path | None = None) -> dict[str, str]:
    """Load the frozen splits, or create them on first use.

    Raises if items appear that the saved splits don't cover: silently assigning them would
    let a re-run move items between tune and test. Extend deliberately with `extend_splits`.
    """
    path = Path(path) if path else splits_path(cfg)
    if path.exists():
        splits = load_splits(path)
        missing = sorted({it.item_id for it in items} - splits.keys())
        if missing:
            raise ValueError(f"{len(missing)} items have no split (e.g. {missing[:3]}). "
                             "Call extend_splits() on purpose instead of regenerating.")
        return splits
    splits = assign_splits(items, cfg["data"].get("tune_frac", 1 / 3), cfg["project"]["seed"])
    save_splits(splits, path)
    return splits


def extend_splits(splits: dict[str, str], new_items: list[ParallelItem],
                  tune_frac: float = 1 / 3, seed: int = 42) -> dict[str, str]:
    """Assign only items that have no split yet; existing assignments never change."""
    fresh = [it for it in new_items if it.item_id not in splits]
    return {**splits, **assign_splits(fresh, tune_frac, seed)}


def filter_oracle(oracle: dict[tuple[str, str, str], bool], splits: dict[str, str],
                  split: str) -> dict[tuple[str, str, str], bool]:
    """Keep oracle entries (lang, item_id, objective) whose item is in `split`."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return {k: v for k, v in oracle.items() if splits.get(k[1]) == split}
