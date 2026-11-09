"""Representation-level analysis (Table 7) and causal interventions (Table 9).

Owner: interp/safety track. Model code needs the `gpu` extra; the metrics are numpy only.

Layers are numbered like `output_hidden_states`: 0 = embeddings, l = output of decoder layer l.

Table 7 (`table7`), per checkpoint, target language and layer, on the eval prompts' last token:
- cross-lingual similarity: each language's activations are mean-centred (removes the language's
  own offset), then matched = mean cosine between an item's English and target activations,
  mismatched = mean cosine with other items' target activations, gap = matched - mismatched,
  retrieval = share of items whose nearest target activation is their own translation.
- link to behaviour: Spearman correlation, per language, between an item's matched cosine and
  its behavioural divergence from English (option JSD / LaBSE, the ACS signal). Negative =
  items whose representations diverge more also behave more differently.
- alignment-direction transfer: d_lang = mean(aligned) - mean(base) per layer.
  cos(d_en, d_tgt) and transfer = (d_tgt . d_en/|d_en|) / |d_en|, i.e. how much of the English
  DPO shift shows up in the target language (1 = all of it).

Table 9 (`table9`), the causal step, on the base model and forced-choice items of one language:
add alpha * |mean shift| * unit(v) to the residual stream after one decoder layer and re-measure
ACS (JSD to the unsteered English option probabilities) and TruthfulQA accuracy. Directions,
all computed on **tune** items and evaluated on **test** items:
- to_english: mean(h_en) - mean(h_tgt), the language offset at that layer
- alignment:  d_en from en_aligned (the English DPO direction)
- random:     a seeded Gaussian direction (control)
All are scaled to the norm of the to_english shift, so alpha = 1 is the same push for each.
Only a drop for to_english / alignment that random doesn't produce supports a causal claim.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from mpac.data.schema import ParallelItem

# --------------------------------------------------------------------------- metrics (numpy)


def _centre_normalise(h: np.ndarray) -> np.ndarray:
    """[n, d] -> mean-centred, unit-norm rows."""
    h = h.astype(np.float32) - h.astype(np.float32).mean(axis=0, keepdims=True)
    return h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)


def cross_lingual_similarity(h_ref: np.ndarray, h_tgt: np.ndarray) -> dict[str, np.ndarray]:
    """h_*: [layers, n, d], rows aligned by item. Per layer: matched, mismatched, gap, retrieval;
    plus item_cos [layers, n] (each item's matched cosine)."""
    n_layers, n, _ = h_ref.shape
    out = {k: np.zeros(n_layers) for k in ("matched", "mismatched", "gap", "retrieval")}
    item_cos = np.zeros((n_layers, n))
    for layer in range(n_layers):
        a, b = _centre_normalise(h_ref[layer]), _centre_normalise(h_tgt[layer])
        sim = a @ b.T
        diag = np.diag(sim)
        item_cos[layer] = diag
        out["matched"][layer] = diag.mean()
        out["mismatched"][layer] = (sim.sum() - diag.sum()) / (n * n - n) if n > 1 else np.nan
        out["retrieval"][layer] = float(np.mean(sim.argmax(axis=1) == np.arange(n)))
    out["gap"] = out["matched"] - out["mismatched"]
    out["item_cos"] = item_cos
    return out


def alignment_direction(acts_aligned: np.ndarray, acts_base: np.ndarray) -> np.ndarray:
    """[layers, n, d] each, same items -> [layers, d]: mean(aligned) - mean(base)."""
    return acts_aligned.astype(np.float32).mean(axis=1) - acts_base.astype(np.float32).mean(axis=1)


def direction_transfer(d_ref: np.ndarray, d_tgt: np.ndarray) -> dict[str, np.ndarray]:
    """[layers, d] each -> per layer cos(d_ref, d_tgt) and the share of d_ref's length that
    d_tgt has along d_ref."""
    nr = np.linalg.norm(d_ref, axis=1) + 1e-8
    nt = np.linalg.norm(d_tgt, axis=1) + 1e-8
    dot = np.sum(d_ref * d_tgt, axis=1)
    return {"cos": dot / (nr * nt), "transfer": dot / nr ** 2, "norm_ref": nr, "norm_tgt": nt}


def spearman(x, y) -> float:
    """Spearman rank correlation (average ranks for ties); nan with < 3 points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")

    def rank(v):
        order = v.argsort(kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):          # average ranks of ties
            m = v == val
            r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- model helpers


def decoder_layers(model):
    """The list of decoder blocks of a (possibly PEFT-wrapped) causal LM."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("can't find the decoder layers of this model")


def use_checkpoint(model, checkpoint: str):
    """Context for running `model` (PeftModel with adapters named like the checkpoints, as in
    training/heldout.load_scoring_model) as `checkpoint`; base disables the adapters."""
    if checkpoint == "base":
        return model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    model.set_adapter(checkpoint)
    return contextlib.nullcontext()


def _chat_batch(tok, prompts: list[str]):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    tok.padding_side = "left"   # the last position is then every prompt's last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)


def last_token_states(model, tok, prompts: list[str], batch_size: int = 8) -> np.ndarray:
    """[layers + 1, n, d] float16 residual stream at each prompt's last token (chat template
    with the generation prompt, i.e. the position that produces the answer)."""
    import torch

    chunks = []
    dev = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        enc = _chat_batch(tok, prompts[i:i + batch_size]).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        chunks.append(torch.stack([h[:, -1].float() for h in hs]).cpu().numpy().astype(np.float16))
    return np.concatenate(chunks, axis=1)


def extract_activations(model, tok, items: list[ParallelItem], lang: str, path: str | Path,
                        batch_size: int = 8) -> dict:
    """Last-token states for every item that has `lang`, saved to `path` (.npz with ids, h).
    Loads instead if the file already holds exactly these items."""
    path = Path(path)
    todo = [it for it in items if lang in it.prompts]
    ids = [it.item_id for it in todo]
    if path.exists():
        d = load_npz(path)
        if list(d["ids"]) == ids:
            return d
    h = last_token_states(model, tok, [it.prompts[lang] for it in todo], batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), h=h)
    return {"ids": np.array(ids), "h": h}


def load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"ids": z["ids"], "h": z["h"]}


@contextlib.contextmanager
def steer(model, vector, layer: int, alpha: float = 1.0):
    """Add alpha * vector to the residual stream after decoder layer `layer` (1-based, i.e.
    hidden_states[layer]) at every position, for the duration of the context."""
    import torch

    blocks = decoder_layers(model)
    if not 1 <= layer <= len(blocks):
        raise ValueError(f"layer must be in 1..{len(blocks)}")
    v = torch.as_tensor(np.asarray(vector), dtype=torch.float32)

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + (alpha * v).to(h.device, h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def batched_option_probs(model, tok, prompts: list[str], options: list[list[str]],
                         batch_size: int = 8) -> list[list[float]]:
    """Renormalised next-token probability over each prompt's option labels (as
    generation/generate.option_probs, batched)."""
    import torch

    out = []
    dev = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        enc = _chat_batch(tok, prompts[i:i + batch_size]).to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1].float()
        for row, opts in zip(logits, options[i:i + batch_size]):
            ids = [tok.encode(o, add_special_tokens=False)[0] for o in opts]
            out.append(torch.softmax(row[ids], dim=-1).tolist())
    return out


# --------------------------------------------------------------------------- tables


def _act_path(cfg: dict, model_id: str, checkpoint: str, lang: str) -> Path:
    from mpac.paths import subdir

    return subdir(cfg, "activations") / model_id / checkpoint / f"{lang}.npz"


def collect_activations(cfg: dict, model_entry: dict, checkpoints: list[str] | None = None,
                        items: list[ParallelItem] | None = None, tok=None, model=None,
                        batch_size: int = 8) -> dict:
    """Last-token states for every checkpoint and language, cached on Drive
    (activations/<model>/<checkpoint>/<lang>.npz, ~100 MB each for a 7B model)."""
    from mpac.config import all_languages
    from mpac.data.loaders import load_items

    items = items if items is not None else load_items(cfg)
    checkpoints = checkpoints or cfg["checkpoints"]
    if model is None:
        from mpac.training.heldout import load_scoring_model

        tok, model = load_scoring_model(cfg, model_entry)
    out = {}
    for ck in checkpoints:
        with use_checkpoint(model, ck):
            for lang in all_languages(cfg):
                out[(ck, lang)] = extract_activations(
                    model, tok, items, lang, _act_path(cfg, model_entry["id"], ck, lang), batch_size)
    return out


def _aligned(acts: dict, a: tuple, b: tuple, ids: list[str] | None = None):
    """Rows of acts[a] and acts[b] for the item ids both have (optionally only `ids`)."""
    ia = {i: k for k, i in enumerate(acts[a]["ids"].tolist())}
    ib = {i: k for k, i in enumerate(acts[b]["ids"].tolist())}
    common = [i for i in acts[a]["ids"].tolist() if i in ib and (ids is None or i in ids)]
    return (common, acts[a]["h"][:, [ia[i] for i in common]],
            acts[b]["h"][:, [ib[i] for i in common]])


def table7(cfg: dict, model_id: str, acts: dict, items: list[ParallelItem],
           divergences: dict[str, dict[str, dict[str, float]]] | None = None) -> dict:
    """acts: (checkpoint, lang) -> {"ids", "h"} (collect_activations). divergences:
    checkpoint -> lang -> item_id -> behavioural divergence from English (signals.acs
    .item_divergences), for the behaviour link. Returns per-layer curves and a summary."""
    ref, targets = cfg["languages"]["reference"], cfg["languages"]["targets"]
    objective = {it.item_id: it.objective for it in items}
    checkpoints = sorted({ck for ck, _ in acts}, key=lambda c: (c != "base", c))
    out = {"similarity": {}, "behaviour_link": {}, "transfer": {}, "summary": []}
    for ck in checkpoints:
        for lang in targets:
            if (ck, ref) not in acts or (ck, lang) not in acts:
                continue
            ids, hr, ht = _aligned(acts, (ck, ref), (ck, lang))
            sim = cross_lingual_similarity(hr, ht)
            mid = hr.shape[0] // 2
            key = f"{ck}/{lang}"
            out["similarity"][key] = {k: v.tolist() for k, v in sim.items() if k != "item_cos"}
            per_obj = {}
            for obj in sorted(set(objective.values())):
                sel = [k for k, i in enumerate(ids) if objective.get(i) == obj]
                if len(sel) > 2:
                    s = cross_lingual_similarity(hr[:, sel], ht[:, sel])
                    per_obj[obj] = {"gap_mid": float(s["gap"][mid]),
                                    "gap_mean": float(np.nanmean(s["gap"][1:]))}
            row = {"checkpoint": ck, "lang": lang, "n": len(ids), "mid_layer": mid,
                   "gap_mid": float(sim["gap"][mid]), "gap_mean": float(np.nanmean(sim["gap"][1:])),
                   "retrieval_mid": float(sim["retrieval"][mid]),
                   "min_gap_layer": int(np.nanargmin(sim["gap"][1:]) + 1),
                   "per_objective": per_obj}
            div = (divergences or {}).get(ck, {}).get(lang)
            if div:
                keep = [k for k, i in enumerate(ids) if i in div]
                rho = [spearman(sim["item_cos"][layer, keep], [div[ids[k]] for k in keep])
                       for layer in range(hr.shape[0])]
                out["behaviour_link"][key] = rho
                row["spearman_mid"] = rho[mid]
                row["n_behaviour"] = len(keep)
            out["summary"].append(row)
    for ck in checkpoints:
        if ck == "base" or ("base", ref) not in acts or (ck, ref) not in acts:
            continue
        _, a_ref, b_ref = _aligned(acts, (ck, ref), ("base", ref))
        d_ref = alignment_direction(a_ref, b_ref)
        for lang in targets:
            if (ck, lang) not in acts or ("base", lang) not in acts:
                continue
            _, a_t, b_t = _aligned(acts, (ck, lang), ("base", lang))
            tr = direction_transfer(d_ref, alignment_direction(a_t, b_t))
            out["transfer"][f"{ck}/{lang}"] = {k: v.tolist() for k, v in tr.items()}
            mid = d_ref.shape[0] // 2
            out["summary"].append({"checkpoint": ck, "lang": lang, "kind": "dpo_direction",
                                   "cos_mid": float(tr["cos"][mid]),
                                   "transfer_mid": float(tr["transfer"][mid]),
                                   "transfer_mean": float(np.mean(tr["transfer"][1:]))})
    return out


def table9(cfg: dict, model, tok, items: list[ParallelItem], acts: dict, splits: dict[str, str],
           lang: str, layers: list[int], alphas: list[float] = (0.5, 1.0, 2.0),
           directions: tuple[str, ...] = ("to_english", "alignment", "random"),
           objectives: list[str] | None = None, batch_size: int = 8, seed: int = 0) -> dict:
    """Steering on the base model (see module docstring). acts: (checkpoint, lang) -> saved
    activations (base for `ref` and `lang`; en_aligned for `ref` to get the alignment direction).
    Returns per (direction, layer, alpha) ACS and accuracy on test items with paired
    differences against alpha = 0."""
    from mpac.analysis.compare import paired_difference
    from mpac.signals.acs import js_divergence

    ref = cfg["languages"]["reference"]
    fc = [it for it in items if it.is_forced_choice and lang in it.prompts and ref in it.prompts
          and (objectives is None or it.objective in objectives)]
    tune_ids = {it.item_id for it in fc if splits.get(it.item_id) == "tune"}
    test = [it for it in fc if splits.get(it.item_id) == "test"]
    if not test or not tune_ids:
        raise ValueError("need forced-choice items in both the tune and test splits")

    def probs(lng, steer_ctx=contextlib.nullcontext()):
        with steer_ctx:
            return batched_option_probs(model, tok, [it.prompts[lng] for it in test],
                                        [it.options[lng] for it in test], batch_size)

    with use_checkpoint(model, "base"):
        p_ref = probs(ref)
        p0 = probs(lang)

    def scores(p):
        acs = {it.item_id: js_divergence(r, q) for it, r, q in zip(test, p_ref, p)}
        acc = {it.item_id: float(int(np.argmax(q)) == it.meta["correct"])
               for it, q in zip(test, p) if "correct" in it.meta}
        return acs, acc

    acs0, acc0 = scores(p0)
    rng = np.random.default_rng(seed)
    rows = [{"direction": "none", "layer": 0, "alpha": 0.0, "acs": float(np.mean(list(acs0.values()))),
             "acc": float(np.mean(list(acc0.values()))) if acc0 else float("nan"), "n": len(test)}]
    _, h_en, h_t = _aligned(acts, ("base", ref), ("base", lang), tune_ids)
    d_align = None
    if "alignment" in directions and ("en_aligned", ref) in acts:
        _, a, b = _aligned(acts, ("en_aligned", ref), ("base", ref), tune_ids)
        d_align = alignment_direction(a, b)
    for layer in layers:
        shift = h_en[layer].astype(np.float32).mean(0) - h_t[layer].astype(np.float32).mean(0)
        scale = float(np.linalg.norm(shift))
        vecs = {"to_english": shift, "random": rng.standard_normal(shift.shape)}
        if d_align is not None:
            vecs["alignment"] = d_align[layer]
        for name in directions:
            if name not in vecs:
                continue
            unit = vecs[name] / (np.linalg.norm(vecs[name]) + 1e-8)
            for alpha in alphas:
                with use_checkpoint(model, "base"):
                    p = probs(lang, steer(model, unit * scale, layer, alpha))
                acs, acc = scores(p)
                d_acs = paired_difference(acs0, acs)
                row = {"direction": name, "layer": layer, "alpha": alpha, "n": len(test),
                       "shift_norm": scale, "acs": d_acs["mean_b"], "acs_diff": d_acs["diff"],
                       "acs_ci": [d_acs["ci_low"], d_acs["ci_high"]], "acs_p": d_acs["p"]}
                if acc:
                    d_acc = paired_difference(acc0, acc)
                    row |= {"acc": d_acc["mean_b"], "acc_diff": d_acc["diff"], "acc_p": d_acc["p"]}
                rows.append(row)
    return {"lang": lang, "objectives": objectives, "n_test": len(test), "n_tune": len(tune_ids),
            "rows": rows}


# --------------------------------------------------------------------------- runners (notebook 06)


def run_table7(cfg: dict, model_entry: dict, tok=None, model=None, batch_size: int = 8) -> dict:
    """Activations for every checkpoint with an adapter (and base), the behaviour link against
    each checkpoint's cached generations, saved to results/<model>/interp/table7.json."""
    from mpac.analysis.results import save_results
    from mpac.config import all_languages
    from mpac.data.loaders import load_items
    from mpac.generation.cache import GenerationCache
    from mpac.paths import subdir
    from mpac.signals.acs import item_divergences
    from mpac.signals.embeddings import embed_generations
    from mpac.signals.lightweight import LazyEmbedder

    items = load_items(cfg)
    acts = collect_activations(cfg, model_entry, cfg["checkpoints"], items, tok, model, batch_size)
    cache, emb_dir = GenerationCache(subdir(cfg, "generations")), subdir(cfg, "embeddings")
    divs = {}
    for ck in cfg["checkpoints"]:
        g = cache.load_all(model_entry["id"], ck, all_languages(cfg))
        if all(g.values()):
            emb = embed_generations(g, LazyEmbedder(), emb_dir)
            divs[ck] = item_divergences(g, cfg["languages"]["reference"], emb)
    out = table7(cfg, model_entry["id"], acts, items, divs)
    out["path"] = str(save_results(cfg, model_entry["id"], "interp", "table7", out))
    return out


def default_layers(n_hidden_states: int) -> list[int]:
    """A third, half and two thirds of the way through the decoder."""
    n = n_hidden_states - 1
    return sorted({max(1, n // 3), n // 2, min(n, 2 * n // 3)})


def run_table9(cfg: dict, model_entry: dict, lang: str, objectives: list[str] | None = None,
               layers: list[int] | None = None, alphas=(0.5, 1.0, 2.0), tok=None, model=None,
               batch_size: int = 8) -> dict:
    """Steering on one language (run_table7 first: it caches the activations), saved to
    results/<model>/interp/table9_<lang>[_<objectives>].json."""
    from mpac.analysis.results import save_results
    from mpac.data.loaders import load_items
    from mpac.data.splits import get_or_create_splits

    items = load_items(cfg)
    splits = get_or_create_splits(cfg, items)
    ref = cfg["languages"]["reference"]
    if model is None:
        from mpac.training.heldout import load_scoring_model

        tok, model = load_scoring_model(cfg, model_entry)
    acts = {}
    for ck, lg in (("base", ref), ("base", lang), ("en_aligned", ref)):
        p = _act_path(cfg, model_entry["id"], ck, lg)
        if p.exists():
            acts[(ck, lg)] = load_npz(p)
    if ("base", ref) not in acts or ("base", lang) not in acts:
        raise FileNotFoundError("no base activations; run run_table7 (notebook 06, Table 7) first")
    layers = layers or default_layers(acts[("base", ref)]["h"].shape[0])
    out = table9(cfg, model, tok, items, acts, splits, lang, layers, list(alphas),
                 objectives=objectives, batch_size=batch_size, seed=cfg["project"]["seed"])
    name = f"table9_{lang}" + (f"_{'_'.join(objectives)}" if objectives else "")
    out["path"] = str(save_results(cfg, model_entry["id"], "interp", name, out))
    return out
