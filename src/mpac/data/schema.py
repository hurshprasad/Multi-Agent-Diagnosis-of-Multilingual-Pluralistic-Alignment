"""Core record types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceItem:
    """An English evaluation item before translation (data/build.py).

    The stem and each option are translated separately and assembled into a prompt per
    language afterwards, so option letters (A, B, C...) never pass through the MT model.
    """

    item_id: str
    objective: str                       # helpfulness | honesty | pluralistic_value
    source: str                          # wvs | plural | truthfulqa | ultrafeedback
    stem: str
    options: list[str] | None = None     # option texts in display order; None = free text
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SourceItem:
        return cls(**d)


@dataclass
class ParallelItem:
    """One evaluation prompt expressed in every language with the same semantic intent.

    `options` is set for forced-choice items (e.g. opinion questions with fixed answers).
    Forced-choice items let the Alignment Collapse Score be computed exactly, without a judge.
    """

    item_id: str
    objective: str                       # helpfulness | honesty | pluralistic_value
    prompts: dict[str, str]              # lang -> prompt text
    options: dict[str, list[str]] | None = None   # lang -> answer options (same order across langs)
    translation_verified: dict[str, bool] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_forced_choice(self) -> bool:
        return self.options is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ParallelItem:
        return cls(**d)


@dataclass
class Generation:
    """A single cached model output. Every evaluator system reads these; none regenerate."""

    model_id: str
    checkpoint: str        # base | en_aligned | multi_aligned
    lang: str
    item_id: str
    response: str
    # Probability over `options` for forced-choice items, read from logits (not sampled text).
    option_probs: list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.model_id, self.checkpoint, self.lang, self.item_id)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Generation:
        return cls(**d)


@dataclass
class OracleLabel:
    """Human-verified judgement for one (generation, objective): the ground truth for evaluators."""

    model_id: str
    checkpoint: str
    lang: str
    item_id: str
    objective: str
    aligned: bool                 # does the response satisfy the objective?
    collapsed_vs_reference: bool  # does it diverge from the English response's behaviour?
    annotator: str = ""
    notes: str = ""
