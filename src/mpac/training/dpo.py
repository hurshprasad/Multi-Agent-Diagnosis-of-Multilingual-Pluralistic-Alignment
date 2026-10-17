"""Build the en_aligned and multi_aligned checkpoints with DPO + LoRA (TRL).

Owner: data/training track. Requires the `gpu` extra.

Controls that must hold for Table 5 to be interpretable:
- Same base model, same LoRA config, same number of steps (training.budget_steps), same seed.
- en_aligned trains on English preference pairs only.
- multi_aligned trains on the same pairs mixed across reference + target languages, keeping
  the total number of pairs identical (so it sees fewer English pairs, not more data).
- Save LoRA adapters to <root>/checkpoints/<model_id>/<checkpoint>/ — generation/generate.py
  loads them via `adapter_path`.

Colab note: DPO on a 7-8B model with LoRA + 4-bit needs an L4 or A100 runtime; use the
model's small_variant on a T4. Save checkpoints to Drive every N steps.
"""

from __future__ import annotations


def build_preference_pairs(cfg: dict, languages: list[str]) -> list[dict]:
    """TODO: {prompt, chosen, rejected} pairs from the pluralistic dataset, translated as needed."""
    raise NotImplementedError


def train_dpo(cfg: dict, model_id: str, checkpoint: str, pairs: list[dict]) -> str:
    """TODO: trl.DPOTrainer with peft LoRA. Returns the adapter path."""
    raise NotImplementedError
