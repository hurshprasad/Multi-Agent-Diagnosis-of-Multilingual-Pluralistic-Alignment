"""Manipulation check: did DPO change preferences on held-out PLURAL pairs, in each language?

Owner: data/training track. Scoring needs the `gpu` extra.

The eval-set results (analysis/compare.py) barely moved after DPO. That is only informative if
training changed the model at all, and in which languages. This scores PLURAL pairs that
neither checkpoint trained on, each in every language (same pairs in en / hi / pt), with the
DPO implicit reward:
    margin = beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))]
where ref is the base model (adapter off). Preference accuracy = share of pairs with margin > 0;
0.5 = no learned preference. Expected if training worked: en_aligned well above 0.5 in en;
multi_aligned above 0.5 in hi and pt too; en_aligned in hi / pt shows how much transferred.

- build_heldout_pairs()  training.n_heldout pairs, excluding training pairs and eval prompts,
                         translated into every target language. Cached under training/heldout/.
- score_heldout()        summed response log-probs under base and each adapter, cached to
                         results/<model>/heldout/logprobs.jsonl so a disconnect resumes.
- preference_report()    accuracy, mean margin and tests per checkpoint and language, plus
                         multi_aligned vs en_aligned on the same pairs (CPU only).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mpac.analysis.compare import paired_difference
from mpac.analysis.results import mcnemar_exact
from mpac.training.dpo import CHECKPOINTS, _norm, eval_exclusions, fetch_plural_rows, select_pairs

FIELDS = ("prompt", "chosen", "rejected")


def build_heldout_pairs(cfg: dict, items=None, rows: list[dict] | None = None,
                        translate: Callable | None = None) -> dict[str, list[dict]]:
    """lang -> the same held-out pairs in that language. Needs training/pairs/selected.jsonl
    (notebook 04 step 1) so training pairs can be excluded."""
    from mpac.data.io import read_jsonl, write_jsonl
    from mpac.data.loaders import load_items
    from mpac.data.translate import TranslationCache, Translator
    from mpac.paths import subdir

    t, ref = cfg["training"], cfg["languages"]["reference"]
    root = subdir(cfg, "training")
    sel_path = root / "heldout" / "selected.jsonl"
    if sel_path.exists():
        pairs = list(read_jsonl(sel_path))
    else:
        train = list(read_jsonl(root / "pairs" / "selected.jsonl"))
        if not train:
            raise FileNotFoundError("no training pairs yet; run notebook 04 step 1 first")
        ex_ids, ex_prompts = eval_exclusions(items if items is not None else load_items(cfg))
        ex_ids |= {p["pair_id"] for p in train}
        ex_prompts |= {_norm(p["prompt"]) for p in train}
        pairs = select_pairs(rows if rows is not None else fetch_plural_rows(cfg), t["n_heldout"],
                             cfg["project"]["seed"] + 1, ex_ids, ex_prompts, t["max_words"],
                             t["max_paragraph_words"])
        write_jsonl(sel_path, pairs)

    cache = TranslationCache(root / "translations", cfg["translation"]["model"])
    out = {ref: [{**p, "lang": ref} for p in pairs]}
    for lang in cfg["languages"]["targets"]:
        texts = [p[k] for p in pairs for k in FIELDS]
        if translate is None and cache.missing(texts, lang):
            translate = Translator(cfg["translation"]["model"], batch_size=t["mt_batch_size"])
        cache.translate_missing(texts, lang, translate, chunk=16)
        table = cache.load(lang)
        out[lang] = []
        for p in pairs:
            tr = {k: table.get(p[k], "").strip() for k in FIELDS}
            if not all(tr.values()):
                raise ValueError(f"held-out pair {p['pair_id']}: empty {lang} translation")
            out[lang].append({**p, **tr, "lang": lang})
    for lang, ps in out.items():
        write_jsonl(root / "heldout" / f"{lang}.jsonl", ps)
    return out


def response_logprob(model, tok, prompt: str, response: str, max_length: int) -> float:
    """Summed log-prob of the assistant turn (with the chat template, as DPO trained on it),
    truncated at max_length tokens like training."""
    import torch

    user = [{"role": "user", "content": prompt}]
    p_text = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": response}],
                                   tokenize=False)
    n_prompt = len(tok(p_text, add_special_tokens=False)["input_ids"])
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_length]
    if len(ids) <= n_prompt:
        return float("nan")
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), dim=-1)
    tgt = x[0, 1:]
    return float(logp[torch.arange(len(tgt)), tgt][n_prompt - 1:].sum())


def load_scoring_model(cfg: dict, model: dict):
    """The base model in 4-bit (as trained) with both adapters loaded, switched per checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from mpac.training.dpo import adapter_dir

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(model["hf_name"])
    base = AutoModelForCausalLM.from_pretrained(
        model["hf_name"], device_map="auto", torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dtype))
    first, *rest = CHECKPOINTS
    m = PeftModel.from_pretrained(base, str(adapter_dir(cfg, model["id"], first)),
                                  adapter_name=first)
    for ck in rest:
        m.load_adapter(str(adapter_dir(cfg, model["id"], ck)), adapter_name=ck)
    m.eval()
    return tok, m


def score_heldout(cfg: dict, model: dict, pairs: dict[str, list[dict]], tok=None,
                  scorer=None) -> list[dict]:
    """{pair_id, lang, checkpoint, chosen, rejected} log-probs for base and each adapter.
    Appends to results/<model>/heldout/logprobs.jsonl and skips rows already there."""
    import contextlib

    from mpac.data.io import read_jsonl, write_jsonl
    from mpac.paths import subdir

    path = subdir(cfg, "results") / model["id"] / "heldout" / "logprobs.jsonl"
    rows = list(read_jsonl(path))
    done = {(r["pair_id"], r["lang"], r["checkpoint"]) for r in rows}
    if scorer is None:
        tok, scorer = load_scoring_model(cfg, model)
    max_length = cfg["training"]["max_length"]
    for ck in ("base", *CHECKPOINTS):
        if hasattr(scorer, "set_adapter") and ck != "base":
            scorer.set_adapter(ck)
        off = scorer.disable_adapter() if ck == "base" and hasattr(scorer, "disable_adapter") \
            else contextlib.nullcontext()
        with off:
            for lang, ps in pairs.items():
                new = []
                for p in ps:
                    if (p["pair_id"], lang, ck) in done:
                        continue
                    lp = {k: response_logprob(scorer, tok, p["prompt"], p[k], max_length)
                          for k in ("chosen", "rejected")}
                    new.append({"pair_id": p["pair_id"], "lang": lang, "checkpoint": ck, **lp})
                    if len(new) == 25:
                        write_jsonl(path, new, append=True)
                        rows += new
                        new = []
                write_jsonl(path, new, append=True)
                rows += new
    return rows


def preference_report(logprobs: list[dict], beta: float) -> list[dict]:
    """One row per (checkpoint, lang): n, accuracy (margin > 0), mean margin, exact binomial p
    against 0.5; then per lang multi_aligned vs en_aligned on the same pairs (diff in accuracy
    with McNemar p, diff in margin with bootstrap CI)."""
    lp = {(r["checkpoint"], r["lang"], r["pair_id"]): (r["chosen"], r["rejected"])
          for r in logprobs}
    langs = sorted({r["lang"] for r in logprobs})
    margins: dict[tuple[str, str], dict[str, float]] = {}
    for ck in CHECKPOINTS:
        for lang in langs:
            m = {}
            for (c, lg, pid), (ch, rj) in lp.items():
                if c == ck and lg == lang and ("base", lang, pid) in lp:
                    bch, brj = lp[("base", lang, pid)]
                    v = beta * ((ch - bch) - (rj - brj))
                    if np.isfinite(v):
                        m[pid] = v
            margins[(ck, lang)] = m
    rows = []
    for (ck, lang), m in margins.items():
        if not m:
            continue
        k = sum(v > 0 for v in m.values())
        rows.append({"checkpoint": ck, "lang": lang, "n": len(m), "accuracy": k / len(m),
                     "mean_margin": float(np.mean(list(m.values()))),
                     "p_vs_chance": mcnemar_exact(k, len(m) - k)})
    a, b = CHECKPOINTS
    for lang in langs:
        ma, mb = margins.get((a, lang), {}), margins.get((b, lang), {})
        ids = sorted(set(ma) & set(mb))
        if not ids:
            continue
        lost = sum(ma[i] > 0 >= mb[i] for i in ids)
        gained = sum(mb[i] > 0 >= ma[i] for i in ids)
        d = paired_difference(ma, mb)
        rows.append({"checkpoint": f"{b} - {a}", "lang": lang, "n": len(ids),
                     "accuracy": (gained - lost) / len(ids), "p_vs_chance": mcnemar_exact(lost, gained),
                     "mean_margin": d["diff"], "margin_ci_low": d["ci_low"],
                     "margin_ci_high": d["ci_high"], "margin_p": d["p"]})
    return rows
