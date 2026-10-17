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
