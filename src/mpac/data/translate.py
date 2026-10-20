"""Translate English prompts into target languages with a public MT model.

Owner: data/training track. Requires the `gpu` extra (plus `mt` for round-trip scoring).

Design notes
- Translate once, cache to <root>/translations/<mt model>/<lang>.jsonl (TranslationCache),
  never re-translate silently.
- Keep forced-choice option labels (A/B/C) untranslated so option order is identical across languages.
- Flag n_verified_per_lang items per language for manual checking; store the result in
  ParallelItem.translation_verified so analyses can be re-run on verified items only.
- One MT model for the whole study (docs/decisions.md, D4). MT_CANDIDATES are the two being
  compared in notebooks/02_phase0_decisions.ipynb; delete the loser once D4 is decided.
- MADLAD and NLLB are sentence-level models. Long texts (e.g. DPO responses) are translated
  paragraph by paragraph and re-joined, which also keeps markdown line structure.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from mpac.data.io import read_jsonl, write_jsonl

MT_CANDIDATES = {
    "madlad": "google/madlad400-3b-mt",            # Apache-2.0, T5 architecture
    "nllb": "facebook/nllb-200-distilled-1.3B",    # CC-BY-NC-4.0 (research use only)
}

NLLB_CODES = {"en": "eng_Latn", "pt": "por_Latn", "es": "spa_Latn", "it": "ita_Latn",
              "de": "deu_Latn", "fr": "fra_Latn", "hi": "hin_Deva"}


def _family(model_name: str) -> str:
    name = model_name.lower()
    if "madlad" in name:
        return "madlad"
    if "nllb" in name:
        return "nllb"
    raise ValueError(f"unsupported MT model {model_name!r}; expected a MADLAD or NLLB checkpoint")


def split_paragraphs(text: str) -> tuple[list[str], list[int]]:
    """Non-empty lines to translate, plus each line's index in the original split, so
    `join_paragraphs` can put blank lines back where they were."""
    lines = text.split("\n")
    keep = [i for i, line in enumerate(lines) if line.strip()]
    return [lines[i] for i in keep], keep


def join_paragraphs(translated: list[str], keep: list[int], n_lines: int) -> str:
    out = [""] * n_lines
    for i, t in zip(keep, translated):
        out[i] = t
    return "\n".join(out)


class Translator:
    """Batched seq2seq translation on GPU (or CPU, slowly)."""

    def __init__(self, model_name: str = MT_CANDIDATES["madlad"], batch_size: int = 16,
                 max_new_tokens: int = 400, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.family = _family(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # T5-family models (MADLAD) overflow in fp16, and a T4 has no bf16, so use fp32 there
        # (3B params ~ 12 GB, fits a 16 GB T4). NLLB is fine in fp16.
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif self.device == "cuda" and self.family == "nllb":
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()

    def _translate_lines(self, lines: list[str], target_lang: str, source_lang: str) -> list[str]:
        import torch

        out: list[str] = []
        for i in range(0, len(lines), self.batch_size):
            batch = lines[i : i + self.batch_size]
            gen_kwargs = {"max_new_tokens": self.max_new_tokens, "num_beams": 4}
            if self.family == "madlad":
                batch = [f"<2{target_lang}> {t}" for t in batch]
            else:
                self.tok.src_lang = NLLB_CODES[source_lang]
                gen_kwargs["forced_bos_token_id"] = self.tok.convert_tokens_to_ids(
                    NLLB_CODES[target_lang])
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=512).to(self.device)
            with torch.no_grad():
                ids = self.model.generate(**enc, **gen_kwargs)
            out.extend(t.strip() for t in self.tok.batch_decode(ids, skip_special_tokens=True))
        return out

    def __call__(self, texts: list[str], target_lang: str, source_lang: str = "en") -> list[str]:
        """Translate whole texts; multi-line texts are split into lines and re-joined."""
        flat, spans = [], []
        for t in texts:
            lines, keep = split_paragraphs(t)
            spans.append((len(flat), len(lines), keep, t.count("\n") + 1))
            flat.extend(lines)
        done = self._translate_lines(flat, target_lang, source_lang) if flat else []
        return [join_paragraphs(done[s : s + n], keep, n_lines) for s, n, keep, n_lines in spans]


class TranslationCache:
    """English text -> translation, one JSONL per (MT model, language), append-only.

    Layout: <root>/<model short name>/<lang>.jsonl with rows {"src": ..., "tgt": ...}.
    Keyed by the exact English string, so a text shared by many items (an instruction, an
    answer option like "Agree") is translated once. Written chunk by chunk, so a Colab
    disconnect loses at most one chunk.
    """

    def __init__(self, root: str | Path, model_name: str):
        self.dir = Path(root) / model_name.rstrip("/").split("/")[-1]

    def path(self, lang: str) -> Path:
        return self.dir / f"{lang}.jsonl"

    def load(self, lang: str) -> dict[str, str]:
        return {r["src"]: r["tgt"] for r in read_jsonl(self.path(lang))}

    def missing(self, texts: list[str], lang: str) -> list[str]:
        have = self.load(lang)
        return [t for t in dict.fromkeys(texts) if t not in have]

    def translate_missing(self, texts: list[str], lang: str,
                          translate: Callable[[list[str], str], list[str]],
                          chunk: int = 64, progress: bool = True) -> int:
        """Translate texts not yet cached; returns how many were added."""
        todo = self.missing(texts, lang)
        steps = range(0, len(todo), chunk)
        if progress and todo:
            from tqdm.auto import tqdm

            steps = tqdm(steps, desc=f"translate en->{lang}")
        for i in steps:
            src = todo[i : i + chunk]
            write_jsonl(self.path(lang), [{"src": a, "tgt": b} for a, b in
                                          zip(src, translate(src, lang))], append=True)
        return len(todo)


_TRANSLATORS: dict[str, Translator] = {}


def translate_batch(texts: list[str], target_lang: str, model_name: str | None = None) -> list[str]:
    """Translate English `texts` into `target_lang`, reusing one loaded model per name."""
    name = model_name or MT_CANDIDATES["madlad"]
    if name not in _TRANSLATORS:
        _TRANSLATORS[name] = Translator(name)
    return _TRANSLATORS[name](texts, target_lang)


def round_trip(texts: list[str], translate: Callable[..., list[str]], langs: list[str],
               source_lang: str = "en") -> dict[str, dict]:
    """en -> lang -> en for each language; chrF of the back-translation against the source.

    A cheap, reference-free way to compare MT candidates (D4). It rewards literal
    translations, so it ranks models but does not replace native-speaker checks.
    `translate(texts, target_lang, source_lang=...)` is any callable, e.g. a Translator.
    Returns lang -> {"chrf": score, "forward": [...], "back": [...]}.
    """
    from sacrebleu.metrics import CHRF

    chrf = CHRF()
    out = {}
    for lang in langs:
        fwd = translate(texts, lang, source_lang=source_lang)
        back = translate(fwd, source_lang, source_lang=lang)
        out[lang] = {"chrf": chrf.corpus_score(back, [texts]).score, "forward": fwd, "back": back}
    return out


def blind_comparison(sources: list[str], outputs: dict[str, list[str]], n: int = 20,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Rows for a fluent speaker to compare two MT models without knowing which is which.

    outputs: model name -> translations aligned with `sources` (exactly two models).
    Returns (sheet rows with columns source / A / B / better, key rows mapping A and B back to
    model names). Share only the sheet; keep the key until the votes are in.
    """
    import random

    if len(outputs) != 2:
        raise ValueError("blind_comparison needs exactly two models")
    rng = random.Random(seed)
    idx = rng.sample(range(len(sources)), min(n, len(sources)))
    sheet, key = [], []
    for row, i in enumerate(idx):
        a, b = rng.sample(sorted(outputs), 2)
        sheet.append({"row": row, "source": sources[i], "A": outputs[a][i], "B": outputs[b][i],
                      "better": "", "notes": ""})
        key.append({"row": row, "A": a, "B": b})
    return sheet, key
