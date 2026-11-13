from types import SimpleNamespace

import numpy as np

from .config import PROJECT_ROOT, ROOT, Settings
from .evidence_adapter import BExtractionAdapter
from .models import bind_extraction_cache_scope, make_rag, provider


def test_b_is_an_application_in_the_shared_project():
    assert ROOT == PROJECT_ROOT / "applications/insurance_b"
    assert Settings.__dataclass_fields__["root"].default == ROOT


async def test_provider_uses_copied_b_credentials_and_explicit_temperature(
    settings, monkeypatch
):
    from lightrag.llm import openai

    calls = []

    async def complete(model, prompt, **kwargs):
        calls.append((model, prompt, kwargs))
        return "synthetic response"

    monkeypatch.setattr(openai, "openai_complete_if_cache", complete)
    assert (
        await provider(settings, "query")("synthetic request") == "synthetic response"
    )
    model, prompt, arguments = calls[0]
    assert model == settings.get("QUERY_MODEL")
    assert prompt == "synthetic request"
    assert arguments["api_key"] == "offline-test"
    assert arguments["temperature"] == 0
    assert arguments["extra_body"] == {"enable_thinking": False}


async def test_embedding_sends_the_configured_dimension_and_uses_its_own_endpoint(
    settings, monkeypatch
):
    import lightrag
    from lightrag.llm.openai import openai_embed

    settings = Settings(
        {**settings.values, "B_EMBEDDING_BASE_URL": "https://embedding.example/v1"},
        settings.root,
    )
    calls = []

    async def embed(texts, **kwargs):
        calls.append(kwargs)
        return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr(openai_embed, "func", embed)
    monkeypatch.setattr(
        lightrag, "LightRAG", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    rag = make_rag(settings)
    result = await rag.embedding_func(["synthetic embedding input"])
    assert result.shape == (1, 1024)
    assert calls[0]["embedding_dim"] == 1024
    assert calls[0]["max_token_size"] == 8192
    assert calls[0]["base_url"] == "https://embedding.example/v1"
    assert calls[0]["api_key"] == "offline-test"


async def test_identical_native_prompts_cannot_reuse_another_products_selection(
    settings,
):
    from lightrag.utils import use_llm_func_with_cache

    class Rag:
        def _build_global_config(self):
            return {
                "llm_cache_identities": {
                    "extract": {
                        "role": "extract",
                        "model": "same-model",
                        "binding": "openai",
                        "host": None,
                    }
                }
            }

    class Cache:
        def __init__(self):
            self.rows = {}
            self.global_config = {"enable_llm_cache_for_entity_extract": True}

        async def get_by_id(self, key):
            return self.rows.get(key)

        async def upsert(self, values):
            self.rows.update(values)

    adapter = BExtractionAdapter(settings, None)
    rag, cache, calls = Rag(), Cache(), []
    bind_extraction_cache_scope(rag, adapter)

    async def complete(prompt, **kwargs):
        calls.append(adapter.entry["product"])
        return adapter.entry["product"]

    answers = []
    for product in ["产品甲", "产品乙", "产品乙"]:
        adapter.begin_document(
            "track",
            {"product": product, "sha256": product, "relative_path": product + ".pdf"},
        )
        value, _ = await use_llm_func_with_cache(
            "共用公司标题",
            complete,
            llm_response_cache=cache,
            llm_cache_identity=rag._build_global_config()["llm_cache_identities"][
                "extract"
            ],
        )
        answers.append(value)
    assert answers == ["产品甲", "产品乙", "产品乙"]
    assert calls == ["产品甲", "产品乙"]
