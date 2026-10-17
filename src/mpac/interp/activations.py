"""Representation-level analysis (Table 7) and causal interventions (Table 9).

Owner: interp/safety track. Requires the `gpu` extra.

Plan
1. Extract last-token residual-stream activations per layer for parallel prompts.
2. Representation similarity: mean cosine between English and target-language activations.
3. Alignment direction: mean(aligned checkpoint) - mean(base) activations on English prompts,
   per layer; measure how much of that direction appears in each target language.
4. Intervention: add or patch the English alignment direction into target-language forward
   passes at the most divergent layers and re-measure ACS. Correlation is not enough; only this
   step supports a causal claim.
Run only on (lang, objective) cells the agents flagged, to keep Colab compute bounded.
"""

from __future__ import annotations


def extract_activations(model, tok, prompts: list[str], layers: list[int] | None = None):
    """TODO: forward hooks on decoder layers; return {layer: tensor[n_prompts, hidden]}."""
    raise NotImplementedError


def alignment_direction(acts_aligned, acts_base):
    """TODO: per-layer mean difference vector, normalised."""
    raise NotImplementedError


def steer(model, direction, layer: int, alpha: float):
    """TODO: context manager adding alpha * direction at `layer` during generation."""
    raise NotImplementedError
