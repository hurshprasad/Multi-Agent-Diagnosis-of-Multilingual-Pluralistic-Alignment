"""Build the en_aligned and multi_aligned checkpoints with DPO + LoRA (TRL).

Owner: data/training track. Requires the `gpu` extra.

Controls that must hold for Table 5 to be interpretable:
- Same base model, same LoRA config, same number of steps (training.budget_steps), same seed.
- en_aligned trains on English preference pairs only.
- multi_aligned trains on the same pairs mixed across reference + target languages, keeping
  the total number of pairs identical (so it sees fewer English pairs, not more data).
- Save LoRA adapters to <root>/checkpoints/<model_id>/<checkpoint>/ — generation/generate.py
  loads them via `adapter_path`.

Pairs (D8, pilot default 2026-10-22: option b): a fixed sample of PLURAL pairs stratified
round-robin over countries, with no country conditioning. Every PLURAL prompt in the eval set
is excluded, by id and by prompt text. multi_aligned shows each pair in exactly one language,
balanced overall and within every country (assign_languages).
Responses are capped at training.max_words: Hindi needs about 2-3x the tokens of English, and
a pair truncated at max_length in one language only would be a confound.

Colab note: DPO on a 7-8B model with LoRA + 4-bit needs an L4 or A100 runtime; use the
model's small_variant on a T4. Checkpoints are saved to Drive every save_steps and training
resumes from the latest one.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

CHECKPOINTS = ("en_aligned", "multi_aligned")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def eval_exclusions(items) -> tuple[set[str], set[str]]:
    """PLURAL row ids and normalised English prompts used by the eval set (never train on them)."""
    plural = [it for it in items if it.meta.get("source") == "plural"]
    ids = {str(it.meta["plural_id"]) for it in plural if "plural_id" in it.meta}
    return ids, {_norm(it.prompts["en"]) for it in plural}


def _usable(r: dict, max_words: int, max_paragraph_words: int) -> bool:
    texts = [r.get("prompt") or "", r.get("pref") or "", r.get("dispref") or ""]
    if not all(t.strip() for t in texts) or _norm(texts[1]) == _norm(texts[2]):
        return False
    if any(len(t.split()) > max_words for t in texts[1:]):
        return False
    # MADLAD translates line by line with a 512-token input limit: very long lines would truncate.
    return all(len(line.split()) <= max_paragraph_words for t in texts for line in t.split("\n"))


def select_pairs(rows: Iterable[dict], n: int, seed: int = 42, exclude_ids: set[str] = frozenset(),
                 exclude_prompts: set[str] = frozenset(), max_words: int = 300,
                 max_paragraph_words: int = 150) -> list[dict]:
    """n PLURAL rows as {pair_id, country, prompt, chosen, rejected}, taken round-robin over
    countries (sorted) so no country dominates. Deterministic given rows and seed."""
    by_country: dict[str, list[dict]] = defaultdict(list)
    seen_prompts: set[str] = set()
    for r in rows:
        p = _norm(r.get("prompt") or "")
        if str(r.get("id")) in exclude_ids or p in exclude_prompts or p in seen_prompts:
            continue
        if _usable(r, max_words, max_paragraph_words):
            seen_prompts.add(p)  # one pair per prompt: different respondents can disagree
            by_country[str(r.get("country") or "unknown")].append(r)
    rng = random.Random(seed)
    queues = []
    for country in sorted(by_country):
        rs = sorted(by_country[country], key=lambda r: str(r["id"]))
        rng.shuffle(rs)
        queues.append(rs)
    out: list[dict] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                r = q.pop()
                out.append({"pair_id": str(r["id"]), "country": str(r.get("country") or "unknown"),
                            "prompt": r["prompt"].strip(), "chosen": r["pref"].strip(),
                            "rejected": r["dispref"].strip()})
    if len(out) < n:
        raise ValueError(f"only {len(out)} usable pairs for n_pairs={n}; raise training.pool "
                         "(null = all of PLURAL) or relax training.max_words")
    return out


def assign_languages(pairs: list[dict], langs: list[str]) -> list[dict]:
    """Copy of pairs (same order) with a `lang`. Countries are taken in order of first
    appearance and languages cycle through each country's pairs, carrying on from where the
    previous country stopped, so every country is balanced and the totals differ by at most 1."""
    by_country: dict[str, list[int]] = {}
    for k, p in enumerate(pairs):
        by_country.setdefault(p["country"], []).append(k)
    lang_of, n = {}, 0
    for idx in by_country.values():
        for k in idx:
            lang_of[k] = langs[n % len(langs)]
            n += 1
    return [{**p, "lang": lang_of[k]} for k, p in enumerate(pairs)]


def usable_batch(batch: dict, max_words: int, max_paragraph_words: int) -> list[bool]:
    """_usable over a column batch, for datasets.Dataset.filter(batched=True)."""
    rows = [dict(zip(batch, vals)) for vals in zip(*batch.values())]
    return [_usable(r, max_words, max_paragraph_words) for r in rows]


def fetch_plural_rows(cfg: dict) -> list[dict]:
    """PLURAL rows that fit the length limits, in a seeded random order, at most training.pool
    of them (null = all). Most PLURAL responses are ~500 words, so only a few percent pass:
    the whole split (~2 GB, memory mapped) is filtered before anything is loaded into memory."""
    import os

    from datasets import load_dataset

    t = cfg["training"]
    ds = load_dataset(cfg["data"]["dataset"], split="train")
    ds = ds.select_columns(["id", "country", "prompt", "pref", "dispref"])
    ds = ds.filter(usable_batch, batched=True, num_proc=os.cpu_count() or 1,
                   fn_kwargs={"max_words": t["max_words"],
                              "max_paragraph_words": t["max_paragraph_words"]})
    ds = ds.shuffle(seed=cfg["project"]["seed"])
    if t.get("pool"):
        ds = ds.select(range(min(t["pool"], len(ds))))
    return list(ds)


def build_preference_pairs(cfg: dict, items=None, rows: list[dict] | None = None,
                           translate: Callable | None = None) -> dict[str, list[dict]]:
    """checkpoint -> pairs for en_aligned (all English) and multi_aligned (same pairs, each in
    one language). Cached under <root>/training/pairs/; a rerun reuses the selection and only
    translates what is missing, so a Colab disconnect loses at most one chunk."""
    from mpac.data.io import read_jsonl, write_jsonl
    from mpac.data.loaders import load_items
    from mpac.data.translate import TranslationCache, Translator
    from mpac.paths import subdir

    t, ref = cfg["training"], cfg["languages"]["reference"]
    langs = [ref, *cfg["languages"]["targets"]]
    pair_dir = subdir(cfg, "training") / "pairs"
    sel_path = pair_dir / "selected.jsonl"
    if sel_path.exists():
        pairs = list(read_jsonl(sel_path))
    else:
        ex_ids, ex_prompts = eval_exclusions(items if items is not None else load_items(cfg))
        pairs = select_pairs(rows if rows is not None else fetch_plural_rows(cfg), t["n_pairs"],
                             cfg["project"]["seed"], ex_ids, ex_prompts, t["max_words"],
                             t["max_paragraph_words"])
        write_jsonl(sel_path, pairs)

    mixed = assign_languages(pairs, langs)
    cache = TranslationCache(subdir(cfg, "training") / "translations", cfg["translation"]["model"])
    for lang in langs[1:]:
        texts = [p[k] for p in mixed if p["lang"] == lang for k in ("prompt", "chosen", "rejected")]
        if translate is None and cache.missing(texts, lang):
            translate = Translator(cfg["translation"]["model"], batch_size=t["mt_batch_size"])
        cache.translate_missing(texts, lang, translate, chunk=16)
    tables = {lang: cache.load(lang) for lang in langs[1:]}
    multi = []
    for p in mixed:
        if p["lang"] != ref:
            tr = {k: tables[p["lang"]].get(p[k], "").strip() for k in ("prompt", "chosen", "rejected")}
            if not all(tr.values()):
                raise ValueError(f"pair {p['pair_id']}: empty {p['lang']} translation")
            p = {**p, **tr}
        multi.append(p)
    out = {"en_aligned": [{**p, "lang": ref} for p in pairs], "multi_aligned": multi}
    for name, ps in out.items():
        write_jsonl(pair_dir / f"{name}.jsonl", ps)
    return out


def length_report(pairs: list[dict], count_tokens: Callable[[str, str], int],
                  max_length: int) -> dict[str, dict[str, float]]:
    """Per language: median and max tokens of prompt + response, and the share of responses
    cut off at max_length. count_tokens(prompt, response) -> tokens with the chat template."""
    lens: dict[str, list[int]] = defaultdict(list)
    for p in pairs:
        for k in ("chosen", "rejected"):
            lens[p["lang"]].append(count_tokens(p["prompt"], p[k]))
    out = {}
    for lang, ls in lens.items():
        ls = sorted(ls)
        out[lang] = {"median": float(ls[len(ls) // 2]), "max": float(ls[-1]),
                     "truncated": sum(x > max_length for x in ls) / len(ls)}
    return out


def adapter_dir(cfg: dict, model_id: str, checkpoint: str) -> Path | None:
    """Where train_dpo saves (and generation loads) the LoRA adapter; None for base."""
    if checkpoint == "base":
        return None
    from mpac.paths import subdir

    return subdir(cfg, "checkpoints") / model_id / checkpoint


def _to_conversation(p: dict) -> dict:
    return {"prompt": [{"role": "user", "content": p["prompt"]}],
            "chosen": [{"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "assistant", "content": p["rejected"]}]}


def train_dpo(cfg: dict, model: dict, checkpoint: str, pairs: list[dict]) -> str:
    """QLoRA DPO with trl.DPOTrainer; the reference model is the base with adapters disabled.
    model: a `models:` entry from the config. Returns the adapter path. Resumes from the latest
    checkpoint under the adapter path if a previous session was cut off."""
    import dataclasses
    import platform

    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"checkpoint must be one of {CHECKPOINTS}")
    t = cfg["training"]
    if len(pairs) != t["n_pairs"]:  # the equal-budget rule
        raise ValueError(f"{checkpoint}: {len(pairs)} pairs, config says n_pairs={t['n_pairs']}")
    out = adapter_dir(cfg, model["id"], checkpoint)
    if (out / "adapter_model.safetensors").exists():
        print(f"{out} already has a trained adapter; delete it to retrain")
        return str(out)

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(model["hf_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model["hf_name"], device_map="auto", torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dtype))
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    lora = t["lora"]
    peft_config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                             target_modules="all-linear", task_type="CAUSAL_LM")

    wanted = {
        "output_dir": str(out), "max_steps": t["budget_steps"], "seed": cfg["project"]["seed"],
        "per_device_train_batch_size": t["batch_size"],
        "gradient_accumulation_steps": t["grad_accum"],
        "learning_rate": t["learning_rate"], "lr_scheduler_type": "cosine",
        "beta": t["beta"], "max_length": t["max_length"], "bf16": bf16, "fp16": not bf16,
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 5, "save_steps": t["save_steps"], "save_total_limit": 2,
        "report_to": "none",
    }
    fields = {f.name for f in dataclasses.fields(DPOConfig)}
    # 10% warmup: transformers 5 dropped warmup_ratio and takes a fraction in warmup_steps.
    if "warmup_ratio" in fields:
        wanted["warmup_ratio"] = 0.1
    else:
        wanted["warmup_steps"] = 0.1
    args = DPOConfig(**{k: v for k, v in wanted.items() if k in fields})
    dropped = set(wanted) - fields
    if dropped:
        print("not supported by this TRL version, ignored:", sorted(dropped))
    data = Dataset.from_list([_to_conversation(p) for p in pairs])
    kw = {"processing_class": tok} if "processing_class" in DPOTrainer.__init__.__code__.co_varnames \
        else {"tokenizer": tok}
    trainer = DPOTrainer(model=base, ref_model=None, args=args, train_dataset=data,
                         peft_config=peft_config, **kw)
    resume = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(out))
    meta = {"checkpoint": checkpoint, "model": model, "n_pairs": len(pairs),
            "steps": trainer.state.global_step,
            "langs": dict(Counter(p["lang"] for p in pairs)), "training": t,
            "seed": cfg["project"]["seed"], "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu", "bf16": bf16, "resumed": resume,
            "versions": {"trl": trl.__version__, "peft": peft.__version__,
                         "transformers": transformers.__version__, "torch": torch.__version__,
                         "python": platform.python_version()},
            "log": trainer.state.log_history}
    (out / "training_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return str(out)
