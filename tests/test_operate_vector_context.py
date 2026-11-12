"""``_get_vector_context`` must not turn a backend query failure into a
confident "zero relevant chunks" answer.

Regression for a swallow-vs-raise fix: the function used to wrap
``chunks_vdb.query()`` in a broad ``except Exception: return []``, so any
vector-store error (timeout, connection reset, embedding failure) looked
identical to "nothing relevant found" -- both to ``naive_query`` and to the
mix-mode branch in ``_perform_kg_search``, where it silently dropped the
vector-search leg while KG results kept flowing. This mirrors the contract
``_get_node_data``/``_get_edge_data`` already follow: no try/except around
the vdb query, so a real failure propagates instead of being swallowed.
"""

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _get_vector_context

pytestmark = pytest.mark.offline


class _FailingVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        raise RuntimeError("vector backend unavailable")


class _EmptyVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        return []


class _HitVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        return [{"content": "hello", "id": "chunk-1", "file_path": "a.txt"}]


async def test_get_vector_context_propagates_backend_error():
    """A transport/backend failure must raise, not be reported as no-results."""
    with pytest.raises(RuntimeError, match="vector backend unavailable"):
        await _get_vector_context(
            "query", _FailingVDB(), QueryParam(), query_embedding=[0.1]
        )


async def test_get_vector_context_confirmed_empty_returns_empty_list():
    """Genuinely zero hits stay a plain empty list, not an error."""
    result = await _get_vector_context(
        "query", _EmptyVDB(), QueryParam(), query_embedding=[0.1]
    )
    assert result == []


async def test_get_vector_context_happy_path_unaffected():
    result = await _get_vector_context(
        "query", _HitVDB(), QueryParam(), query_embedding=[0.1]
    )
    assert len(result) == 1
    assert result[0]["content"] == "hello"
    assert result[0]["chunk_id"] == "chunk-1"


async def test_vector_overfetch_separates_recall_from_final_chunk_limit(monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setenv("VECTOR_CHUNK_CANDIDATE_MULTIPLIER", "4")
    vdb = _HitVDB()
    vdb.query = AsyncMock(return_value=[])
    param = QueryParam(chunk_top_k=12)
    await _get_vector_context("京安心期满延续", vdb, param)
    assert vdb.query.call_args.kwargs["top_k"] == 48
    assert param.chunk_top_k == 12


@pytest.mark.parametrize("value", ["0", "-1", "17"])
async def test_invalid_overfetch_refused_before_vector_call(monkeypatch, value):
    from unittest.mock import AsyncMock

    monkeypatch.setenv("VECTOR_CHUNK_CANDIDATE_MULTIPLIER", value)
    vdb = _HitVDB()
    vdb.query = AsyncMock()
    with pytest.raises(ValueError, match="MULTIPLIER"):
        await _get_vector_context("query", vdb, QueryParam())
    vdb.query.assert_not_awaited()
