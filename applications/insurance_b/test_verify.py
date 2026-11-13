import pytest

from .verify import foreign_product_entities, verified_chunk_strategy


@pytest.mark.parametrize("foreign", ["产品乙", "产品乙｜计划一", "产品乙|计划一"])
def test_readback_rejects_another_products_scoped_entities(foreign):
    assert foreign_product_entities(
        ["产品甲", "产品甲｜计划一", "通用概念", foreign],
        "产品甲",
        {"产品甲", "产品乙"},
    ) == [foreign]


def test_a_legacy_fixed_path_requires_native_default_and_exact_f_parameters(settings):
    from dataclasses import replace
    from types import SimpleNamespace

    from lightrag.chunker import chunking_by_token_size
    from lightrag.pipeline import _format_chunking_params

    a = replace(settings, values={**settings.values, "B_PREPARATION": "business"})
    prepared = {"chunking": {"params": {"split_by_character": "original-boundary"}}}
    metadata = {
        "chunk_method": "legacy_chunking_func",
        "chunk_opts": _format_chunking_params(
            8192,
            {
                "split_by_character": "original-boundary",
                "split_by_character_only": True,
                "overlap": 0,
            },
        ),
    }
    rag = SimpleNamespace(chunking_func=chunking_by_token_size)
    assert verified_chunk_strategy(a, rag, metadata, prepared)
    assert not verified_chunk_strategy(
        a, SimpleNamespace(chunking_func=lambda *args: []), metadata, prepared
    )
    assert not verified_chunk_strategy(
        a, rag, {**metadata, "chunk_opts": "size=100"}, prepared
    )
    assert not verified_chunk_strategy(settings, rag, metadata, prepared)
