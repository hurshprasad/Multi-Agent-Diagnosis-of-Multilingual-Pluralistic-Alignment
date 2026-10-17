"""Downstream safety consequences of alignment collapse (Table 8).

Owner: interp/safety track.

The original proposal used IndicSafe / IndicJR. With the language scope moved to
pt / es / it / de / fr, the team needs to choose public multilingual safety and jailbreak
benchmarks that cover these languages (docs/decisions.md, D6).

Metrics per language: safety score, jailbreak attack success rate, over-refusal rate,
utility retention, and correlation of each with that language's ACS.
"""

from __future__ import annotations


def run_safety_eval(cfg: dict, model_id: str, checkpoint: str, lang: str) -> dict:
    """TODO: generate on the safety benchmark via generation/generate.py, then score."""
    raise NotImplementedError
