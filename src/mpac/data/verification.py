"""Manual translation checks: the "verified subset" every key result is also reported on.

Owner: data/training track.

1. export_verification()  one CSV per language under <root>/verification/<lang>.csv, holding
                          the same n items in every language (stratified by objective), so
                          the verified subset is identical across languages.
2. A fluent speaker fills `ok` (yes / no) and, where the translation is wrong, `corrected`
   (the full corrected prompt). Open the CSV in Google Sheets, then download it as CSV
   back to the same path.
3. apply_verification()   sets ParallelItem.translation_verified[lang] and swaps in
                          corrected prompts. loaders.load_items() applies it automatically.

Do this BEFORE generating: a corrected prompt changes what the model should have been asked,
and the generation cache won't regenerate an item it already has.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from mpac.data.schema import ParallelItem

COLUMNS = ("item_id", "objective", "english", "translation", "ok", "corrected", "notes")

_YES = {"1", "y", "yes", "true", "ok", "sim", "s", "हाँ", "हां"}
_NO = {"0", "n", "no", "false", "não", "nao", "नहीं"}


def verification_ids(items: list[ParallelItem], n: int, seed: int = 42) -> list[str]:
    """n item ids, spread evenly over objectives (round-robin), same for every language."""
    by_obj: dict[str, list[str]] = defaultdict(list)
    for it in sorted(items, key=lambda it: it.item_id):
        by_obj[it.objective].append(it.item_id)
    rng = random.Random(seed)
    queues = [rng.sample(ids, len(ids)) for _, ids in sorted(by_obj.items())]
    out: list[str] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop())
    return out


def export_verification(items: list[ParallelItem], lang: str, ids: list[str], path: str | Path,
                        reference: str = "en") -> int:
    """Write the sheet for one language. Refuses to overwrite a sheet that has answers."""
    path = Path(path)
    if path.exists() and any(r.get("ok", "").strip() for r in read_sheet(path)):
        raise FileExistsError(f"{path} already has answers; not overwriting")
    by_id = {it.item_id: it for it in items}
    rows = [{"item_id": i, "objective": by_id[i].objective,
             "english": by_id[i].prompts[reference], "translation": by_id[i].prompts[lang],
             "ok": "", "corrected": "", "notes": ""}
            for i in ids if i in by_id and lang in by_id[i].prompts]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def read_sheet(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_verification(items: list[ParallelItem], lang: str, path: str | Path) -> dict[str, int]:
    """Apply one filled sheet in place. A row counts as verified when ok = yes, or when a
    corrected prompt is given (which replaces the MT prompt). Blank rows are left unverified.
    Returns counts of verified / corrected / rejected / unanswered rows."""
    by_id = {it.item_id: it for it in items}
    counts = {"verified": 0, "corrected": 0, "rejected": 0, "unanswered": 0}
    for row in read_sheet(path):
        it = by_id.get(row["item_id"])
        if it is None:
            continue
        ok, corrected = row.get("ok", "").strip().lower(), row.get("corrected", "").strip()
        if corrected:
            it.prompts[lang] = corrected
            it.translation_verified[lang] = True
            counts["corrected"] += 1
        elif ok in _YES:
            it.translation_verified[lang] = True
            counts["verified"] += 1
        elif ok in _NO:
            it.translation_verified[lang] = False
            counts["rejected"] += 1
        elif not ok:
            counts["unanswered"] += 1
        else:
            raise ValueError(f"{path}: can't read ok={row['ok']!r} for {row['item_id']}")
    return counts
