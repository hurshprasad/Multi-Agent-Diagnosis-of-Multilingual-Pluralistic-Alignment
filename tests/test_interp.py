"""Tables 7 and 9 on CPU: numpy metrics always; model code on a tiny random Qwen2 (needs torch)."""
import numpy as np
import pytest

from mpac.data.schema import ParallelItem
from mpac.interp.activations import (alignment_direction, cross_lingual_similarity,
                                     direction_transfer, spearman, table7)


def test_similarity_identical_vs_shuffled():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, 20, 8))
    same = cross_lingual_similarity(h, h + 5.0)          # a constant language offset is removed
    assert np.allclose(same["matched"], 1, atol=1e-5) and np.allclose(same["retrieval"], 1)
    shuf = cross_lingual_similarity(h, h[:, rng.permutation(20)])
    assert (same["gap"] > shuf["gap"]).all()


def test_direction_transfer_and_spearman():
    d = np.array([[1.0, 0.0], [0.0, 2.0]])
    t = direction_transfer(d, d * 0.5)
    assert np.allclose(t["cos"], 1) and np.allclose(t["transfer"], 0.5)
    base = np.zeros((2, 4, 2))
    assert np.allclose(alignment_direction(base + 1, base), 1)
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert np.isnan(spearman([1, 2], [1, 2]))


def test_table7_summary_and_behaviour_link():
    rng = np.random.default_rng(1)
    cfg = {"languages": {"reference": "en", "targets": ["hi"]}}
    n = 12
    ids = np.array([f"i{k}" for k in range(n)])
    h_en = rng.standard_normal((4, n, 6))
    noise = np.linspace(0, 3, n)[None, :, None] * rng.standard_normal((4, n, 6))
    acts = {("base", "en"): {"ids": ids, "h": h_en}, ("base", "hi"): {"ids": ids, "h": h_en + noise},
            ("en_aligned", "en"): {"ids": ids, "h": h_en + 1.0},
            ("en_aligned", "hi"): {"ids": ids, "h": h_en + noise + 0.5}}
    items = [ParallelItem(i, "honesty", {"en": "x", "hi": "y"}) for i in ids]
    div = {"base": {"hi": {i: float(k) for k, i in enumerate(ids)}}}   # more noise = more divergence
    out = table7(cfg, "m", acts, items, div)
    base = next(r for r in out["summary"] if r["checkpoint"] == "base")
    assert base["spearman_mid"] < -0.3
    tr = next(r for r in out["summary"] if r.get("kind") == "dpo_direction")
    assert tr["transfer_mid"] == pytest.approx(0.5, abs=1e-4)


# ------------------------------------------------------------------ tiny model

def tiny_model_and_tokenizer():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["[PAD]", "[UNK]", "A", "B", "C", "ASSISTANT", "en", "hi", "q0", "q1", "q2", "q3",
             "q4", "q5", "q6", "q7", "q8", "q9", "pick", "one", "long", "words", "here"]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]")
    tok.chat_template = ("{% for m in messages %}{{ m['content'] }} {% endfor %}"
                         "{% if add_generation_prompt %}ASSISTANT{% endif %}")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(words), hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=3, num_attention_heads=2,
                                   num_key_value_heads=1, max_position_embeddings=64)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    return tok, model


def test_last_token_states_batching_and_steer():
    torch = pytest.importorskip("torch")
    tok, model = tiny_model_and_tokenizer()
    from mpac.interp.activations import batched_option_probs, last_token_states, steer

    prompts = ["en q1 pick one", "hi q2", "en q3 long words here pick one"]
    h = last_token_states(model, tok, prompts, batch_size=3)
    assert h.shape == (4, 3, 16)
    alone = last_token_states(model, tok, prompts[1:2], batch_size=1)
    assert np.allclose(h[:, 1], alone[:, 0], atol=2e-2)   # left padding doesn't change the state

    opts = [["A", "B", "C"]] * 3
    p0 = batched_option_probs(model, tok, prompts, opts)
    assert np.allclose([sum(p) for p in p0], 1)
    with steer(model, np.ones(16), layer=2, alpha=0.0):
        assert np.allclose(batched_option_probs(model, tok, prompts, opts), p0, atol=1e-5)
    with steer(model, np.ones(16) * 5, layer=2, alpha=1.0):
        assert not np.allclose(batched_option_probs(model, tok, prompts, opts), p0, atol=1e-3)
    assert np.allclose(batched_option_probs(model, tok, prompts, opts), p0, atol=1e-5)  # hook removed
    with pytest.raises(ValueError):
        with steer(model, np.ones(16), layer=9):
            pass


def test_table9_and_collect_on_tiny_model(tmp_path):
    pytest.importorskip("torch")
    tok, model = tiny_model_and_tokenizer()
    from mpac.interp.activations import extract_activations, table9

    items = [ParallelItem(f"t{k}", "honesty", {"en": f"en q{k} pick one", "hi": f"hi q{k}"},
                          options={"en": ["A", "B", "C"], "hi": ["A", "B", "C"]},
                          meta={"correct": k % 3}) for k in range(10)]
    splits = {f"t{k}": "tune" if k < 4 else "test" for k in range(10)}
    acts = {}
    for lang in ("en", "hi"):
        acts[("base", lang)] = extract_activations(model, tok, items, lang, tmp_path / f"{lang}.npz")
        acts[("en_aligned", lang)] = {"ids": acts[("base", lang)]["ids"],
                                      "h": acts[("base", lang)]["h"] + 0.1}
    again = extract_activations(model, tok, items, "en", tmp_path / "en.npz")   # from the cache
    assert np.array_equal(again["h"], acts[("base", "en")]["h"])
    cfg = {"languages": {"reference": "en", "targets": ["hi"]}}
    out = table9(cfg, model, tok, items, acts, splits, "hi", layers=[1, 2], alphas=[1.0])
    assert out["n_test"] == 6 and out["n_tune"] == 4
    names = {(r["direction"], r["layer"]) for r in out["rows"]}
    assert {("none", 0), ("to_english", 1), ("alignment", 2), ("random", 1)} <= names
    assert all(0 <= r["acs"] <= 1 for r in out["rows"])


def test_runners_on_drive_layout(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from mpac.config import load_config
    from mpac.data.io import write_jsonl
    from mpac.data.synthetic import make_synthetic
    from mpac.generation.cache import GenerationCache
    from mpac.interp.activations import run_table7, run_table9

    monkeypatch.setenv("MPAC_ROOT", str(tmp_path))
    cfg = load_config("configs/hi_pt.yaml")
    cfg["checkpoints"] = ["base"]
    items, gens, _ = make_synthetic(["hi", "pt"], ["honesty", "pluralistic_value"], n_items=24,
                                    collapsed={("hi", "honesty"): 0.7})
    write_jsonl(tmp_path / "items" / "items.jsonl", [it.to_dict() for it in items])
    GenerationCache(tmp_path / "generations").add(g for by in gens.values() for g in by.values())
    tok, model = tiny_model_and_tokenizer()
    entry = {"id": "synthetic", "hf_name": "none"}
    t7 = run_table7(cfg, entry, tok, model)
    assert {r["lang"] for r in t7["summary"]} == {"hi", "pt"} and "base/hi" in t7["behaviour_link"]
    assert (tmp_path / "activations" / "synthetic" / "base" / "hi.npz").exists()
    t9 = run_table9(cfg, entry, "hi", ["honesty"], alphas=[1.0], tok=tok, model=model)
    assert t9["n_test"] > 0 and t9["path"].endswith("table9_hi_honesty.json")
