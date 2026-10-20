"""Build the parallel evaluation set: English sources -> translations -> ParallelItems.

Owner: data/training track. Notebook: notebooks/03_build_dataset.ipynb.

Steps and files (all under the artifact root, i.e. Drive on Colab):
  1. fetch_sources()     HF datasets -> SourceItems          build/source_items.jsonl
  2. translate           stems, options, instruction -> MT  translations/<mt>/<lang>.jsonl
  3. assemble_items()    SourceItems + translations          items/items.jsonl
                         -> one ParallelItem per item, with a prompt per language

Every step is cached: re-running skips work already on disk, so a disconnect costs little.
The source sample is drawn once; delete build/source_items.jsonl only to redraw it on purpose
(that invalidates splits and generations too).

Sources (docs/decisions.md, D1):
  wvs            pluralistic_value, forced choice   PLURAL `survey_questions` (WVS items)
  plural         pluralistic_value, free text       PLURAL preference prompts
  truthfulqa     honesty, forced choice             TruthfulQA mc1 (correct + 3 distractors)
  ultrafeedback  helpfulness, free text             UltraFeedback test_prefs, short non-code
"""

from __future__ import annotations

import hashlib
import random
import re

from mpac.data.io import read_jsonl, write_jsonl
from mpac.data.schema import ParallelItem, SourceItem

INSTRUCTION = "Answer with only the letter of your choice."

_CODE_HINT = re.compile(r"```|\bdef |\bfunction\b|SELECT |#include|\bcode\b|\bpython\b|\bjava\b",
                        re.IGNORECASE)


def option_labels(n: int) -> list[str]:
    if n > 26:
        raise ValueError("at most 26 options")
    return [chr(65 + i) for i in range(n)]


def format_prompt(stem: str, options: list[str] | None, instruction: str) -> str:
    """Free text: the stem. Forced choice: stem, lettered options, instruction."""
    if options is None:
        return stem
    lines = [f"{lab}. {opt}" for lab, opt in zip(option_labels(len(options)), options)]
    return f"{stem}\n\n" + "\n".join(lines) + f"\n\n{instruction}"


def _needs_translation(text: str) -> bool:
    """Numbers on a rating scale ("2", "7") stay as they are."""
    return not text.strip().isdigit()


# ---- per-source converters (pure: rows in, SourceItems out) --------------------------------

def wvs_stem(main_question: str, question_text: str) -> str:
    """WVS rows split a question into a shared lead-in and the item text. Sometimes the
    lead-in is just a title ("Willingness to fight for country") or repeats the item text;
    then the item text alone is the question."""
    main, text = main_question.strip(), question_text.strip()
    if main.lower().rstrip("?") in text.lower():
        return text
    if len(main) < 60 and not main.endswith(("?", ".", ":", "!")):
        return text
    return f"{main}\n{text}"


def wvs_items(rows: list[dict]) -> list[SourceItem]:
    """PLURAL `survey_questions` rows -> forced-choice items. Answer codes < 0 are WVS
    missing-data codes (don't know, not asked...) and are dropped; 0 is a real answer."""
    out = []
    for r in rows:
        opts = sorted(((int(o["code"]), o["label"]) for o in r["answer_options"]
                       if int(o["code"]) >= 0), key=lambda x: x[0])
        if len(opts) < 2:
            continue
        labels = [label for _, label in opts]
        stem = wvs_stem(r["main_question"], r["question_text"])
        if {lab.lower() for lab in labels} == {"mentioned", "not mentioned"}:
            # "Choose up to five qualities" lists: ask about this one quality, Yes / No.
            stem = f"{r['main_question'].strip()}\nWould you choose this one: {r['question_text'].strip()}?"
            labels = ["Yes" if lab.lower() == "mentioned" else "No" for lab in labels]
        out.append(SourceItem(
            item_id=f"wvs-{r['question_code']}", objective="pluralistic_value", source="wvs",
            stem=stem, options=labels,
            meta={"question_code": r["question_code"], "codes": [c for c, _ in opts],
                  "question_group_id": r.get("question_group_id")}))
    return out


def plural_items(rows: list[dict]) -> list[SourceItem]:
    """PLURAL preference rows -> free-text items. Row ids are kept so these prompts can be
    excluded from DPO training pairs later (D8)."""
    return [SourceItem(item_id=f"plural-{r['id']}", objective="pluralistic_value",
                       source="plural", stem=r["prompt"].strip(),
                       meta={"plural_id": r["id"], "country": r.get("country")})
            for r in rows]


def truthfulqa_items(rows: list[dict], max_options: int = 4, seed: int = 42) -> list[SourceItem]:
    """TruthfulQA multiple_choice rows -> forced-choice items. mc1 lists the correct answer
    first, so options are the correct answer plus the first (max_options - 1) distractors,
    shuffled with a per-question seed. meta["correct"] is the correct option's index."""
    out = []
    for i, r in enumerate(rows):
        choices, labels = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        correct = choices[labels.index(1)]
        wrong = [c for c, lab in zip(choices, labels) if lab == 0][: max_options - 1]
        opts = [correct, *wrong]
        random.Random(f"{seed}-{i}").shuffle(opts)
        out.append(SourceItem(item_id=f"tqa-{i:04d}", objective="honesty", source="truthfulqa",
                              stem=r["question"].strip(), options=opts,
                              meta={"correct": opts.index(correct)}))
    return out


def ultrafeedback_items(rows: list[dict], max_words: int = 120) -> list[SourceItem]:
    """UltraFeedback rows -> free-text helpfulness items, keeping short prompts with no sign
    of code (code and long documents translate badly and dominate generation time)."""
    out = []
    for r in rows:
        p = r["prompt"].strip()
        if len(p.split()) <= max_words and not _CODE_HINT.search(p):
            out.append(SourceItem(item_id=f"uf-{r['prompt_id'][:12]}", objective="helpfulness",
                                  source="ultrafeedback", stem=p))
    return out


def _sample(items: list[SourceItem], n: int | None, seed: int, tag: str) -> list[SourceItem]:
    if n is None or n >= len(items):
        return items
    return random.Random(f"{seed}-{tag}").sample(items, n)


# ---- fetching (needs `datasets`) ------------------------------------------------------------

def fetch_sources(cfg: dict) -> list[SourceItem]:
    """Download (or stream) each source and sample the configured number of items."""
    from datasets import load_dataset

    b, seed, src = cfg["build"], cfg["project"]["seed"], cfg["data"]["sources"]
    out: list[SourceItem] = []

    if b.get("wvs"):
        rows = load_dataset(src["pluralistic_value"], "survey_questions", split="train")
        out += _sample(wvs_items(list(rows)), b["wvs"], seed, "wvs")
    if b.get("plural"):
        # 2 GB of preference pairs: stream, shuffle in a buffer, take what we need.
        stream = load_dataset(src["pluralistic_value"], split="train", streaming=True)
        rows = list(stream.shuffle(seed=seed, buffer_size=10_000).take(b["plural"]))
        out += plural_items(rows)
    if b.get("truthfulqa"):
        rows = load_dataset(src["honesty"], "multiple_choice", split="validation")
        items = truthfulqa_items(list(rows), b.get("truthfulqa_max_options", 4), seed)
        out += _sample(items, b["truthfulqa"], seed, "truthfulqa")
    if b.get("ultrafeedback"):
        stream = load_dataset(src["helpfulness"], split="test_prefs", streaming=True)
        items = ultrafeedback_items(list(stream.take(max(4 * b["ultrafeedback"], 500))))
        out += _sample(items, b["ultrafeedback"], seed, "ultrafeedback")

    ids = [it.item_id for it in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item ids across sources")
    return out


# ---- translation + assembly -----------------------------------------------------------------

def texts_to_translate(sources: list[SourceItem], instruction: str = INSTRUCTION) -> list[str]:
    """Every distinct English string the MT model must see: stems, option texts, instruction."""
    texts = [s.stem for s in sources]
    texts += [o for s in sources for o in (s.options or []) if _needs_translation(o)]
    if any(s.options for s in sources):
        texts.append(instruction)
    return list(dict.fromkeys(texts))


def assemble_items(sources: list[SourceItem], tables: dict[str, dict[str, str]],
                   reference: str = "en", instruction: str = INSTRUCTION) -> list[ParallelItem]:
    """SourceItems + {lang: {english: translation}} -> ParallelItems.

    A language is left out of an item if any of its pieces is untranslated, so a partial
    translation run never produces a half-English prompt.
    """
    items = []
    for s in sources:
        prompts = {reference: format_prompt(s.stem, s.options, instruction)}
        options = {reference: option_labels(len(s.options))} if s.options else None
        for lang, table in tables.items():
            def tr(text: str, table=table) -> str | None:
                return table.get(text) if _needs_translation(text) else text

            stem, opts = tr(s.stem), [tr(o) for o in s.options or []]
            instr = tr(instruction) if s.options else ""
            if stem is None or instr is None or any(o is None for o in opts):
                continue
            prompts[lang] = format_prompt(stem, opts if s.options else None, instr)
            if options is not None:
                options[lang] = option_labels(len(opts))
        items.append(ParallelItem(item_id=s.item_id, objective=s.objective, prompts=prompts,
                                  options=options,
                                  meta={"source": s.source, **s.meta,
                                        "en_hash": hashlib.sha1(prompts[reference].encode())
                                        .hexdigest()[:10]}))
    return items


def build_dataset(cfg: dict, translate=None) -> list[ParallelItem]:
    """Run steps 1-3, reusing whatever is already on disk. `translate(texts, lang)` defaults
    to the configured MT model, which is only loaded if something still needs translating."""
    from mpac.data.translate import TranslationCache, Translator
    from mpac.paths import subdir

    reference, targets = cfg["languages"]["reference"], cfg["languages"]["targets"]
    src_path = subdir(cfg, "build") / "source_items.jsonl"
    if src_path.exists():
        sources = [SourceItem.from_dict(d) for d in read_jsonl(src_path)]
    else:
        sources = fetch_sources(cfg)
        write_jsonl(src_path, sources)

    cache = TranslationCache(subdir(cfg, "translations"), cfg["translation"]["model"])
    texts = texts_to_translate(sources)
    if translate is None and any(cache.missing(texts, lang) for lang in targets):
        translate = Translator(cfg["translation"]["model"])
    for lang in targets:
        cache.translate_missing(texts, lang, translate)

    items = assemble_items(sources, {lang: cache.load(lang) for lang in targets}, reference)
    write_jsonl(subdir(cfg, "items") / "items.jsonl", items)
    return items
