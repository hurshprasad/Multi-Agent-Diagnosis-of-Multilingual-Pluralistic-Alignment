"""Per-language collapse thresholds for the lightweight evaluators (D3).

Owner: signals/eval track.

One fixed threshold doesn't fit every language: LaBSE puts a Hindi translation further from
its English source than a Portuguese one (prompt_floor 0.14 vs 0.10 in the pilot), so a Hindi
agent with the same embedding threshold flags paraphrases as collapse. Thresholds are keyed by
(lang, kind), kind = "options" (JSD of forced-choice option probabilities, comparable across
languages) or "embedding" (LaBSE 1 − cosine, offset by each language's floor).

- floor_thresholds()  label-free starting point: options fixed, embedding = floor + margin.
- tune_thresholds()   per (lang, kind), the cut that best separates oracle labels. Pass only
                      **tune**-split labels (CLAUDE.md); cells with too few labels keep the start.
- signal_kind()       which kind a reference/target Generation pair is scored with.
"""

from __future__ import annotations

from itertools import pairwise

Thresholds = dict[tuple[str, str], float]


def signal_kind(ref, tgt) -> str:
    return "options" if ref.option_probs is not None and tgt.option_probs is not None \
        else "embedding"


def floor_thresholds(targets: list[str], floor: dict[str, float], options: float = 0.3,
                     embedding_margin: float = 0.15) -> Thresholds:
    out: Thresholds = {}
    for lang in targets:
        out[(lang, "options")] = options
        out[(lang, "embedding")] = floor.get(lang, 0.0) + embedding_margin
    return out


def _balanced_accuracy(ds: list[float], ys: list[bool], t: float) -> float:
    pos = [d > t for d, y in zip(ds, ys) if y]
    neg = [d <= t for d, y in zip(ds, ys) if not y]
    return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2


def tune_thresholds(divs: dict[tuple[str, str], dict[str, float]],
                    oracle: dict[tuple[str, str, str], bool], start: Thresholds,
                    min_labels: int = 6) -> tuple[Thresholds, dict]:
    """divs: (lang, kind) -> item_id -> divergence. oracle: (lang, item_id, objective) ->
    collapsed, tune split only. Picks the cut with the best balanced accuracy (ties: closest to
    the starting value). Returns thresholds and, per key, what was used and why."""
    labels: dict[tuple[str, str], bool] = {(lang, iid): y for (lang, iid, _), y in oracle.items()}
    out, report = dict(start), {}
    for key, by_item in divs.items():
        lang = key[0]
        pairs = [(d, labels[(lang, i)]) for i, d in by_item.items() if (lang, i) in labels]
        n_pos = sum(y for _, y in pairs)
        info = {"n_labels": len(pairs), "n_collapsed": n_pos, "start": start.get(key)}
        if len(pairs) < min_labels or n_pos == 0 or n_pos == len(pairs):
            report[key] = {**info, "threshold": start.get(key), "source": "start (too few labels)"}
            continue
        ds, ys = zip(*sorted(pairs))
        cuts = [(a + b) / 2 for a, b in pairwise(ds) if a != b] or [ds[0]]
        s0 = start.get(key, 0.3)
        best = max(cuts, key=lambda t: (_balanced_accuracy(ds, ys, t), -abs(t - s0)))
        out[key] = best
        report[key] = {**info, "threshold": best, "source": "tuned",
                       "balanced_accuracy": _balanced_accuracy(ds, ys, best)}
    return out, report
