from mpac.config import load_config
from mpac.data.schema import Generation
from mpac.generation.cache import GenerationCache


def test_cache_dedupes_and_resumes(tmp_path):
    cache = GenerationCache(tmp_path)
    g = Generation("m", "base", "es", "x1", "hola", [0.2, 0.8])
    assert cache.add([g, g]) == 1
    assert cache.add([g]) == 0
    assert cache.done_ids("m", "base", "es") == {"x1"}
    assert cache.load("m", "base", "es")["x1"].option_probs == [0.2, 0.8]


def test_config_override_replaces_lists():
    cfg = load_config("configs/smoke.yaml")
    assert cfg["languages"]["targets"] == ["es", "de"]
    assert cfg["agents"]["retain_threshold"] == 0.6   # untouched default survives the merge


def test_get_model_by_id():
    import pytest

    from mpac.config import get_model

    cfg = load_config("configs/hi_pt.yaml")
    assert get_model(cfg)["id"] == "qwen2.5-7b"
    assert get_model(cfg, "llama3.1-8b")["hf_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert get_model(load_config("configs/hi_pt.yaml", "configs/hi_pt_quick.yaml"))["id"] == \
        "qwen2.5-1.5b"
    with pytest.raises(KeyError):
        get_model(cfg, "gpt-5")


def test_token_limit_report_counts_cut_free_text_only():
    from mpac.data.schema import Generation, ParallelItem
    from mpac.generation.generate import token_limit_report

    tok = lambda text, add_special_tokens=False: {"input_ids": text.split()}   # one token per word
    items = [ParallelItem(i, obj, {"en": "q"}) for i, obj in
             (("a", "helpfulness"), ("b", "helpfulness"), ("c", "honesty"), ("d", "honesty"))]
    gens = {"hi": {"a": Generation("m", "base", "hi", "a", "w " * 10),           # cut (re-encoded)
                   "b": Generation("m", "base", "hi", "b", "w w"),
                   "c": Generation("m", "base", "hi", "c", "x", meta={"hit_limit": True}),
                   "d": Generation("m", "base", "hi", "d", "w " * 10, [1.0, 0.0])}}   # forced choice
    r = token_limit_report(gens, items, tok, max_new_tokens=10)["hi"]
    assert r["n_free_text"] == 3 and r["n_hit_limit"] == 2
    assert r["rate_helpfulness"] == 0.5 and r["rate_honesty"] == 1.0
