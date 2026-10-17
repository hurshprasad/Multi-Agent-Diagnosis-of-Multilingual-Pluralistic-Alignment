"""Generate responses for every (model, checkpoint, language, item) and write them to the cache.

Owner: data/training track. Requires the `gpu` extra.

Two outputs per item:
- `response`: greedy free-text answer (used by embedding / classifier / judge signals)
- `option_probs`: for forced-choice items, the probability mass on each option label read
  from next-token logits. This is the cheapest and most verifiable signal for ACS.
"""

from __future__ import annotations

from tqdm.auto import tqdm

from mpac.data.schema import Generation, ParallelItem
from mpac.generation.cache import GenerationCache


def load_model(hf_name: str, load_in_4bit: bool = True, adapter_path: str | None = None):
    """Load tokenizer + model. 4-bit lets 7-8B models fit on a 16 GB Colab T4."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(hf_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, device_map="auto", torch_dtype=torch.float16, quantization_config=quant
    )
    if adapter_path:  # en_aligned / multi_aligned LoRA adapters from training/dpo.py
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def _chat(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def option_probs(tok, model, prompt: str, options: list[str]) -> list[float]:
    """Renormalised next-token probability over single-token option labels (A, B, C...)."""
    import torch

    inputs = tok(_chat(tok, prompt), return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    ids = [tok.encode(o, add_special_tokens=False)[0] for o in options]
    p = torch.softmax(logits[ids].float(), dim=-1)
    return p.tolist()


def generate_for_language(
    tok, model, items: list[ParallelItem], *, model_id: str, checkpoint: str, lang: str,
    cache: GenerationCache, max_new_tokens: int = 512, batch_size: int = 8,
) -> int:
    """Generate for one language, skipping items already cached. Returns number written."""
    import torch

    done = cache.done_ids(model_id, checkpoint, lang)
    todo = [it for it in items if lang in it.prompts and it.item_id not in done]
    written = 0
    for i in tqdm(range(0, len(todo), batch_size), desc=f"{model_id}/{checkpoint}/{lang}"):
        batch = todo[i : i + batch_size]
        texts = [_chat(tok, it.prompts[lang]) for it in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        responses = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        gens = []
        for it, resp in zip(batch, responses):
            probs = option_probs(tok, model, it.prompts[lang], it.options[lang]) if it.is_forced_choice else None
            gens.append(Generation(model_id, checkpoint, lang, it.item_id, resp.strip(), probs))
        written += cache.add(gens)  # write per batch so a Colab disconnect loses at most one batch
    return written
