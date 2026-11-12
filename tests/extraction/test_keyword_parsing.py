from unittest.mock import patch

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _parse_keywords_payload, extract_keywords_only


class _FakeKeywordModel:
    def model_dump(self):
        return {
            "high_level_keywords": ["AI"],
            "low_level_keywords": ["RAG", "Graph"],
        }


class _FakeTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


def _keyword_global_config(
    model: str, binding: str = "openai", keyword_func=None
) -> dict:
    return {
        "addon_params": {"language": "en"},
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": {"keyword": keyword_func} if keyword_func else {},
        "llm_cache_identities": {
            "keyword": {
                "role": "keyword",
                "binding": binding,
                "model": model,
                "host": "https://api.example.com/v1",
            }
        },
    }


@pytest.mark.offline
def test_parse_keywords_payload_accepts_model_like_objects():
    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(_FakeKeywordModel())

    assert is_valid is True
    assert hl_keywords == ["AI"]
    assert ll_keywords == ["RAG", "Graph"]


@pytest.mark.offline
def test_parse_keywords_payload_extracts_json_from_wrapped_text():
    result = """
    analysis first
    {"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]}
    trailing note
    """

    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(result)

    assert is_valid is True
    assert hl_keywords == ["AI", "Agents"]
    assert ll_keywords == ["RAG", "LightRAG"]


@pytest.mark.offline
@pytest.mark.parametrize(
    "result",
    [
        '```json\n{"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]}\n```',
        '```json {"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]}```',
    ],
    ids=["multiline-fence", "single-line-fence"],
)
def test_parse_keywords_payload_strips_markdown_fence(result):
    """Keyword parsing shares utils.strip_markdown_code_fence; both multi-line
    and single-line (no interior newline) fences must be stripped."""
    is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(result)

    assert is_valid is True
    assert hl_keywords == ["AI", "Agents"]
    assert ll_keywords == ["RAG", "LightRAG"]


@pytest.mark.offline
def test_parse_keywords_payload_warns_when_json_repair_is_used():
    broken_result = (
        '{"high_level_keywords":"AI, Agents","low_level_keywords":["RAG","LightRAG"]'
    )

    with patch("lightrag.operate.logger.warning") as mocked_warning:
        is_valid, hl_keywords, ll_keywords = _parse_keywords_payload(broken_result)

    assert is_valid is True
    assert hl_keywords == ["AI", "Agents"]
    assert ll_keywords == ["RAG", "LightRAG"]
    mocked_warning.assert_called_once()
    assert (
        "Keyword extraction response required JSON repair"
        in mocked_warning.call_args[0][0]
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_keywords_only_accepts_empty_keyword_cache_without_requery():
    async def should_not_run(*_args, **_kwargs):
        raise AssertionError(
            "keyword LLM should not be called on a valid empty cache hit"
        )

    param = QueryParam()
    global_config = _keyword_global_config("model-a", keyword_func=should_not_run)

    with patch(
        "lightrag.operate.handle_cache",
        return_value=('{"high_level_keywords":[],"low_level_keywords":[]}', None),
    ):
        hl_keywords, ll_keywords = await extract_keywords_only(
            "hello",
            param,
            global_config,
            hashing_kv=None,
        )

    assert hl_keywords == []
    assert ll_keywords == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_keywords_only_partitions_cache_by_keyword_llm_identity():
    cache = _FakeKVStorage()
    calls = 0

    async def keyword_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (
            '{"high_level_keywords":["model-'
            + str(calls)
            + '"],"low_level_keywords":["rag"]}'
        )

    param = QueryParam()

    first_hl, first_ll = await extract_keywords_only(
        "same query",
        param,
        _keyword_global_config("model-a", keyword_func=keyword_model),
        hashing_kv=cache,
    )
    second_hl, second_ll = await extract_keywords_only(
        "same query",
        param,
        _keyword_global_config("model-b", keyword_func=keyword_model),
        hashing_kv=cache,
    )

    assert first_hl == ["model-1"]
    assert first_ll == ["rag"]
    assert second_hl == ["model-2"]
    assert second_ll == ["rag"]
    assert calls == 2
    assert len(cache._store) == 2


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix"])
@pytest.mark.parametrize("length", [49, 50, 53, 200])
async def test_empty_keywords_still_build_context_for_every_query_length(mode, length):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from lightrag.operate import kg_query

    query = "问" * length
    build = AsyncMock(return_value=SimpleNamespace(context="retrieved", raw_data={}))
    config = {"role_llm_funcs": {"query": AsyncMock()}}
    with (
        patch("lightrag.operate.get_keywords_from_query", return_value=([], [])),
        patch("lightrag.operate._build_query_context", build),
    ):
        result = await kg_query(
            query,
            None,
            None,
            None,
            None,
            QueryParam(mode=mode, only_need_context=True),
            config,
        )
    assert result.content == "retrieved"
    assert build.call_args.args[1] == ("" if mode == "global" else query)
    assert build.call_args.args[2] == ("" if mode == "local" else query)
    assert build.call_args.args[7].mode == mode


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,valid",
    [
        ('{"high_level_keywords":[],"low_level_keywords":[]}', True),
        ('{"wrong_field":[]}', False),
        ('{"high_level_keywords":null,"low_level_keywords":{}}', False),
        ("[]", False),
    ],
)
async def test_failed_keyword_trace_preserves_raw_response_and_schema_state(raw, valid):
    from unittest.mock import AsyncMock

    with patch("lightrag.operate.logger.warning") as warning:
        assert await extract_keywords_only(
            "长问题" * 30,
            QueryParam(),
            _keyword_global_config("model", keyword_func=AsyncMock(return_value=raw)),
        ) == ([], [])
    trace = [
        call for call in warning.call_args_list if "raw_response=%r" in call.args[0]
    ]
    assert len(trace) == 1
    assert trace[0].args[2] is valid
    assert trace[0].args[3] == raw
