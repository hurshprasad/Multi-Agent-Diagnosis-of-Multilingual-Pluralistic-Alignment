"""The shared generation cache.

Every evaluator system (single, fixed, LLM-judge, adaptive) reads model outputs from here.
Nothing downstream of generation is allowed to call a model to produce responses, which is
what makes the evaluator comparison controlled.

Layout: <root>/generations/<model_id>/<checkpoint>/<lang>.jsonl
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mpac.data.io import read_jsonl, write_jsonl
from mpac.data.schema import Generation


class GenerationCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, model_id: str, checkpoint: str, lang: str) -> Path:
        return self.root / model_id / checkpoint / f"{lang}.jsonl"

    def done_ids(self, model_id: str, checkpoint: str, lang: str) -> set[str]:
        """Item ids already generated, so interrupted Colab runs resume instead of restarting."""
        return {d["item_id"] for d in read_jsonl(self._path(model_id, checkpoint, lang))}

    def add(self, gens: Iterable[Generation]) -> int:
        """Append generations, grouped by file. Skips items already present."""
        by_file: dict[Path, list[Generation]] = {}
        seen: dict[Path, set[str]] = {}
        for g in gens:
            p = self._path(g.model_id, g.checkpoint, g.lang)
            if p not in seen:
                seen[p] = self.done_ids(g.model_id, g.checkpoint, g.lang)
            if g.item_id in seen[p]:
                continue
            seen[p].add(g.item_id)
            by_file.setdefault(p, []).append(g)
        return sum(write_jsonl(p, rows, append=True) for p, rows in by_file.items())

    def load(self, model_id: str, checkpoint: str, lang: str) -> dict[str, Generation]:
        return {
            d["item_id"]: Generation.from_dict(d)
            for d in read_jsonl(self._path(model_id, checkpoint, lang))
        }

    def load_all(self, model_id: str, checkpoint: str, langs: list[str]) -> dict[str, dict[str, Generation]]:
        """lang -> item_id -> Generation."""
        return {lang: self.load(model_id, checkpoint, lang) for lang in langs}
