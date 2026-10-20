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
