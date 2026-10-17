"""Translate English prompts into target languages with a public MT model.

Owner: data/training track.

Design notes
- Translate once, cache to <root>/translations/<lang>.jsonl, never re-translate silently.
- Keep forced-choice option labels (A/B/C) untranslated so option order is identical across languages.
- Flag n_verified_per_lang items per language for manual checking; store the result in
  ParallelItem.translation_verified so analyses can be re-run on verified items only.
"""

from __future__ import annotations


def translate_batch(texts: list[str], target_lang: str, model_name: str | None = None) -> list[str]:
    """TODO: implement with a public MT model (e.g. an NLLB or MADLAD checkpoint via transformers).

    Decide the MT model as a team (docs/decisions.md, D4) so it is fixed for the whole study.
    """
    raise NotImplementedError
