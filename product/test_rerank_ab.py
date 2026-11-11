"""Network-free regression tests; the optional local-copy smoke is also mocked."""

import asyncio
import copy
import json
import os
from pathlib import Path
import shutil
import socket

import httpx
import pytest

from product import rerank_ab as h


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("External network forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def chat(model=h.MODELS["answer"], content="Synthetic answer"):
    return {
        "id": "synthetic-id",
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def rerank(count=14):
    return {
        "request_id": "synthetic-rerank",
        "output": {
            "results": [
                {"index": i, "relevance_score": 1 - n / 100}
                for n, i in enumerate(range(count - 1, max(-1, count - 13), -1))
            ]
        },
        "usage": {"total_tokens": 30},
    }


def cases():
    return [
        {"id": f"case_{i}", "question": f"Synthetic question {i}?"} for i in range(3)
    ]


def provider(tmp_path, handler, rows=None, verifier=lambda: None):
    (tmp_path / "attempts").mkdir(mode=0o700)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = h.Provider(
        tmp_path,
        {"schedule": rows or h.schedule(cases())},
        {kind: "synthetic-private-key-123456" for kind in h.MODELS},
        client,
        verifier,
    )
    p.case_id = "case_0"
    return p


def test_full_schedule_is_exact_per_case_stage():
    rows = h.schedule(cases())
    assert len(rows) == 15
    assert [r["stage"] for r in rows[:3]] == ["keyword", "embedding", "rerank"]
    for case in cases():
        assert sorted(r["stage"] for r in rows if r["case_id"] == case["id"]) == [
            "answer_off",
            "answer_on",
            "embedding",
            "keyword",
            "rerank",
        ]


def test_attempt_is_durable_before_http_and_raw_bytes_are_exact(tmp_path):
    raw = json.dumps(
        chat(
            h.MODELS["keyword"],
            json.dumps(
                {"high_level_keywords": ["high"], "low_level_keywords": ["low"]}
            ),
        ),
        indent=3,
    ).encode()

    def handler(request):
        folder = tmp_path / "attempts/0001"
        assert (
            h.load(folder / "attempt.json")["phase"]
            == "reserved_sent_or_unknown_no_retry"
        )
        assert h.read(folder / "request.json") == request.content
        return httpx.Response(200, content=raw)

    p = provider(tmp_path, handler)
    asyncio.run(p.keyword("Synthetic prompt", response_format={"type": "json_object"}))
    assert h.read(tmp_path / "attempts/0001/response.json") == raw
    assert p.completed[0]["response_sha256"] == h.digest(raw)
    assert (tmp_path / "attempts/0001/response.json").stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("status", [301, 400, 429, 500])
def test_no_http_retry_no_error_body_and_no_fallback(tmp_path, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=b"DO NOT SAVE THIS ERROR")

    p = provider(tmp_path, handler)
    with pytest.raises(h.Stop, match="Non200"):
        asyncio.run(p.keyword("prompt", response_format={"type": "json_object"}))
    with pytest.raises(h.Stop, match="PriorFailure"):
        asyncio.run(p.embedding(["query"]))
    with pytest.raises(h.Stop, match="Non200Response"):
        p.raise_if_failed()
    assert len(calls) == p.attempted == 1
    failure = h.load(tmp_path / "attempts/0001/error.json")
    assert (
        failure["error_type"] == "Non200Response" and failure["http_status"] == status
    )
    assert p.first_failure == {"error_type": "Non200Response", "http_status": status}
    assert p.failure_record_saved is True
    assert not (tmp_path / "attempts/0001/response.json").exists()
    assert not any(
        b"DO NOT SAVE" in f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()
    )


def test_timeout_counts_once_and_blocks_retry(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private provider diagnostic")

    p = provider(tmp_path, handler)
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(p.keyword("prompt", response_format={"type": "json_object"}))
    with pytest.raises(h.Stop, match="PriorFailure"):
        asyncio.run(
            p.keyword("changed prompt", response_format={"type": "json_object"})
        )
    assert len(calls) == 1
    record = h.load(tmp_path / "attempts/0001/error.json")
    assert record["error_type"] == "Timeout" and record["http_status"] is None
    assert not any(
        b"private provider diagnostic" in f.read_bytes()
        for f in tmp_path.rglob("*")
        if f.is_file()
    )


def test_failure_category_never_serializes_untrusted_exception_text():
    assert (
        h.failure_category(RuntimeError("secret diagnostic"))
        == "InvalidResponseOrLocalFailure"
    )
    assert h.failure_category(h.Stop("secret diagnostic")) == "ProtocolGuardRejected"


def test_error_record_write_failure_preserves_first_error_and_poison(
    tmp_path, monkeypatch
):
    p = provider(tmp_path, lambda _: httpx.Response(429, content=b"private body"))
    original = h.write_json

    def fail_error_write(path, value):
        if path.name == "error.json":
            raise OSError("private filesystem diagnostic")
        original(path, value)

    monkeypatch.setattr(h, "write_json", fail_error_write)
    with pytest.raises(h.Stop, match="Non200Response"):
        asyncio.run(p.keyword("prompt", response_format={"type": "json_object"}))
    assert p.first_failure == {"error_type": "Non200Response", "http_status": 429}
    assert p.failure_record_saved is False and p.attempted == 1
    with pytest.raises(h.Stop, match="PriorFailure"):
        asyncio.run(p.embedding(["query"]))
    assert not any(
        b"private" in f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()
    )


def test_changed_fingerprint_prevents_send(tmp_path):
    calls = []

    def verifier():
        raise h.Stop("SourceChanged")

    p = provider(tmp_path, lambda request: calls.append(request), verifier=verifier)
    with pytest.raises(h.Stop, match="SourceChanged"):
        asyncio.run(p.keyword("prompt", response_format={"type": "json_object"}))
    assert calls == [] and p.attempted == 0


def test_stage_and_cap_cannot_be_bypassed(tmp_path):
    calls = []
    p = provider(tmp_path, lambda request: calls.append(request))
    with pytest.raises(h.Stop, match="CaseStage"):
        asyncio.run(p.embedding(["query"]))
    assert not calls
    p.failed = False
    p.attempted = 15
    with pytest.raises(h.Stop, match="AttemptCap"):
        asyncio.run(p.embedding(["query"]))
    assert not calls


@pytest.mark.parametrize(
    "mutate,category",
    [
        (lambda d: d["choices"][0].update(finish_reason="length"), "NonStop"),
        (lambda d: d["choices"][0]["message"].update(refusal="no"), "ExplicitRefusal"),
        (
            lambda d: d["choices"][0]["message"].update(reasoning_content="thinking"),
            "UnexpectedReasoning",
        ),
        (
            lambda d: d["usage"].update(
                completion_tokens_details={"reasoning_tokens": 1}
            ),
            "UnexpectedReasoning",
        ),
        (lambda d: d.update(model="unexpected"), "ResponseModel"),
        (lambda d: d["choices"][0]["message"].update(content=""), "EmptyAnswer"),
        (lambda d: d.pop("usage"), "MissingUsage"),
        (
            lambda d: d["choices"][0]["message"].update(tool_calls=[{}]),
            "UnexpectedTool",
        ),
    ],
)
def test_incomplete_and_unapproved_chat_halts(mutate, category):
    data = chat()
    mutate(data)
    with pytest.raises(h.Stop, match=category):
        h.response_data(h.canonical(data), "answer_off", {}, [])


def test_secret_echo_not_persisted(tmp_path):
    secret = "synthetic-private-key-123456"
    raw = h.canonical(chat(h.MODELS["keyword"], secret))
    p = provider(tmp_path, lambda request: httpx.Response(200, content=raw))
    with pytest.raises(h.Stop, match="CredentialEcho"):
        asyncio.run(p.keyword("prompt", response_format={"type": "json_object"}))
    assert not any(
        secret.encode() in f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()
    )
    with pytest.raises(h.Stop, match="CredentialEcho"):
        h.no_secrets(json.dumps({"echo": "密钥"}, ensure_ascii=True).encode(), ["密钥"])


@pytest.mark.parametrize(
    "mutation,category",
    [
        (lambda r: r[1].update(index=r[0]["index"]), "RerankIndex"),
        (lambda r: r[0].update(index=True), "RerankIndex"),
        (lambda r: r[0].update(index=14), "RerankIndex"),
        (lambda r: r[1].update(relevance_score=1.1), "RerankScore"),
        (lambda r: r[2].update(relevance_score=1), "RerankNotSorted"),
        (lambda r: r.pop(), "RerankCount"),
        (lambda r: r[0].update(document={"text": "unexpected"}), "UnexpectedReturned"),
    ],
)
def test_invalid_ranking_is_not_a_fallback(mutation, category):
    data = rerank()
    mutation(data["output"]["results"])
    with pytest.raises(h.Stop, match=category):
        h.validate_ranking(data, 14, 12)


def test_rerank_missing_model_is_recorded_not_fabricated():
    data = rerank()
    assert (
        h.response_data(
            h.canonical(data),
            "rerank",
            {"input": {"documents": ["x"] * 14}, "parameters": {"top_n": 12}},
            [],
        )
        == data
    )
    assert "model" not in data


def test_length_gate_is_conservative_and_never_splits():
    assert h.rerank_size("query", ["短文"])["document_utf8_bytes"] == [6]
    for query, docs in [
        ("q" * 4001, ["x"]),
        ("q", ["x" * 30001]),
        ("q", ["x"] * 501),
        ("q" * 4000, ["x" * 30000] * 4),
    ]:
        with pytest.raises(h.Stop):
            h.rerank_size(query, docs)


def test_embedding_requires_all_original_vectors():
    raw = {
        "id": "embedding-id",
        "model": h.MODELS["embedding"],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "data": [{"index": 0, "embedding": [0.5] * 1024}],
    }
    assert (
        len(
            h.response_data(h.canonical(raw), "embedding", {"input": ["query"]}, [])[
                "data"
            ][0]["embedding"]
        )
        == 1024
    )
    raw.pop("data")
    with pytest.raises(h.Stop, match="EmbeddingCount"):
        h.response_data(h.canonical(raw), "embedding", {"input": ["query"]}, [])


def test_capture_uses_full_candidates_and_a_equivalence_then_b_order():
    _, QueryParam, operate, utils, prompts = h.native_modules()
    tokenizer = utils.TiktokenTokenizer("gpt-4o-mini")
    entities = [
        {"id": 0, "entity": "Synthetic policy", "description": "A synthetic policy."}
    ]
    relations = [
        {
            "id": 0,
            "entity1": "Policy",
            "entity2": "Event",
            "description": "A synthetic relation.",
        }
    ]
    chunks = [
        {
            "chunk_id": f"c{i}",
            "content": f"Synthetic document text number {i}.",
            "file_path": f"source-{i % 2}.txt",
        }
        for i in range(14)
    ]
    cfg = {
        "tokenizer": tokenizer,
        "min_rerank_score": 0.0,
        "user_prompt_prefix": "Synthetic policy.\n",
    }
    query = "What does this synthetic policy say?"
    qp = QueryParam(**h.QUERY)
    original = copy.deepcopy(chunks)

    class Ranking:
        async def rerank(self, query, documents, top_n):
            assert documents == [c["content"] for c in original]
            return rerank()["output"]["results"]

    async def run():
        with h.Capture(operate, utils) as capture:
            native, _ = await operate._build_context_str(
                entities,
                relations,
                chunks,
                query,
                qp,
                cfg,
                relation_id_to_original={("Policy", "Event"): {"source_id": "c1"}},
                system_prompt=prompts["rag_response"],
            )
            assert len(capture.args["merged_chunks"]) == 14
            a, _, fa = await capture.build(False, Ranking())
            b, _, fb = await capture.build(True, Ranking())
            assert a == native and len(fa) == len(fb) == 12
            assert [c["chunk_id"] for c in fa] == [f"c{i}" for i in range(12)]
            assert [c["chunk_id"] for c in fb] == [f"c{i}" for i in range(13, 1, -1)]
            assert chunks == original and capture.args["merged_chunks"] == original
            assert h.prompt_text(
                a, query, qp, cfg, prompts["rag_response"], utils
            ) == h.prompt_text(native, query, qp, cfg, prompts["rag_response"], utils)
            for context in (a, b):
                assert json.dumps(entities[0], ensure_ascii=False) in context
                assert json.dumps(relations[0], ensure_ascii=False) in context
            assert fa[0]["reference_id"] == "1" and fb[0]["reference_id"] == "1"
            assert fa[0]["chunk_id"] != fb[0]["chunk_id"]

    asyncio.run(run())


def test_token_truncation_remains_native_after_rerank():
    _, QueryParam, operate, utils, prompts = h.native_modules()
    cfg = {"tokenizer": utils.TiktokenTokenizer("gpt-4o-mini"), "min_rerank_score": 0.0}
    chunks = [
        {
            "chunk_id": str(i),
            "content": "long token content " * 500,
            "file_path": "synthetic.txt",
        }
        for i in range(14)
    ]
    qp = QueryParam(**{**h.QUERY, "max_total_tokens": 3600})

    class Ranking:
        async def rerank(self, query, documents, top_n):
            return rerank()["output"]["results"]

    async def run():
        with h.Capture(operate, utils) as capture:
            await operate._build_context_str(
                [],
                [],
                chunks,
                "Synthetic question",
                qp,
                cfg,
                system_prompt=prompts["rag_response"],
            )
            _, _, final = await capture.build(True, Ranking())
            assert 0 < len(final) < 12
            assert [c["chunk_id"] for c in final] == [
                str(i) for i in range(13, 13 - len(final), -1)
            ]

    asyncio.run(run())


def test_process_lock_excludes_second_runner(tmp_path):
    with h.run_lock(tmp_path):
        with pytest.raises(h.Stop, match="RunLocked"):
            with h.run_lock(tmp_path):
                pass


def test_network_guard_blocks_incidental_dns_and_connections():
    with h.network_guard():
        with pytest.raises(h.Stop, match="UnguardedNetworkBlocked"):
            socket.getaddrinfo("example.invalid", 443)
        with socket.socket() as sock:
            with pytest.raises(h.Stop, match="UnguardedNetworkBlocked"):
                sock.connect(("127.0.0.1", 9))


def test_complete_mock_run_fifteen_attempts_blind_bundle_and_no_restart(
    tmp_path, monkeypatch
):
    m = {
        "cases": cases(),
        "schedule": h.schedule(cases()),
        "blind_assignment": h.blind_order(cases()),
    }
    keys = {kind: "synthetic-private-key-123456" for kind in h.MODELS}
    monkeypatch.setattr(h, "verify", lambda *_: (m, keys))
    review = tmp_path / "review.json"
    h.write_json(review, {"approved": True, "preflight_sha256": "a" * 64})
    review_sha = h.digest(h.read(review))
    observed = []

    def handler(request):
        observed.append(request)
        payload = h.parse(request.content)
        kind = payload["model"]
        if kind == h.MODELS["keyword"]:
            data = chat(
                kind,
                json.dumps(
                    {"high_level_keywords": ["high"], "low_level_keywords": ["low"]}
                ),
            )
        elif kind == h.MODELS["embedding"]:
            data = {
                "id": "embedding-id",
                "model": kind,
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
                "data": [{"index": 0, "embedding": [0.5] * 1024}],
            }
        elif kind == h.MODELS["rerank"]:
            data = rerank(2)
        else:
            data = chat()
        return httpx.Response(200, content=json.dumps(data, indent=2).encode())

    async def fake_pairs(run, manifest, p):
        pairs = []
        for case in manifest["cases"]:
            p.case_id = case["id"]
            await p.keyword(
                "Synthetic keyword prompt", response_format={"type": "json_object"}
            )
            await p.embedding([case["question"]])
            await p.rerank(
                case["question"], ["Synthetic first", "Synthetic second"], 12
            )
            folder = run / "cases" / case["id"]
            h.mkdir(folder)
            prompt = (
                "Synthetic system prompt\n\n---User Query---\n\n" + case["question"]
            )
            for arm in ("off", "on"):
                h.write_new(folder / f"{arm}-prompt.txt", prompt.encode())
                h.write_json(
                    folder / f"{arm}-context.json",
                    {
                        "lineage": [
                            {
                                "chunk_id": "synthetic",
                                "reference_id": "1",
                                "source_sha256": "b" * 64,
                                "rerank_score": 0.5,
                            }
                        ]
                    },
                )
            pairs.append({**case, "off": prompt, "on": prompt})
        return pairs

    monkeypatch.setattr(h, "build_pairs", fake_pairs)

    def factory():
        return httpx.Client(transport=httpx.MockTransport(handler))

    result = h.execute(tmp_path, "a" * 64, review, review_sha, factory)
    assert result == {"phase": "complete", "attempted": 15, "completed": 15}
    assert len(observed) == 15
    comparisons = h.load(tmp_path / "run-summary.json")["pair_comparisons"]
    assert all(
        c["prompt_equal"] and c["off_prompt_sha256"] == c["on_prompt_sha256"]
        for c in comparisons
    )
    blind = h.load(tmp_path / "blind-review-input.json")
    assert len(blind["entries"]) == 6
    for item in blind["entries"]:
        assert set(item) == {
            "blind_id",
            "case_id",
            "question",
            "answer",
            "actual_prompt",
            "reference_sources",
        }
        assert "rerank_score" not in item["reference_sources"][0]
    with pytest.raises(h.Stop, match="PreviousRunNoResume"):
        h.execute(tmp_path, "a" * 64, review, review_sha, factory)
    assert len(observed) == 15


def test_failed_run_has_no_blind_bundle_and_cannot_resume(tmp_path, monkeypatch):
    m = {"cases": cases(), "schedule": h.schedule(cases())}
    keys = {kind: "synthetic-private-key-123456" for kind in h.MODELS}
    monkeypatch.setattr(h, "verify", lambda *_: (m, keys))
    review = tmp_path / "review.json"
    h.write_json(review, {"approved": True, "preflight_sha256": "a" * 64})
    review_sha = h.digest(h.read(review))

    async def failed_pairs(run, manifest, p):
        p.case_id = "case_0"
        try:
            await p.keyword("prompt", response_format={"type": "json_object"})
        except h.Stop:
            pass  # Simulate core swallowing the provider error and returning failure.
        raise h.Stop("RetrievalCallCount")

    monkeypatch.setattr(h, "build_pairs", failed_pairs)

    def factory():
        return httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(429, content=b"private error")
            )
        )

    with pytest.raises(h.Stop, match="Non200"):
        h.execute(tmp_path, "a" * 64, review, review_sha, factory)
    assert h.load(tmp_path / "halt.json")["attempted"] == 1
    assert h.load(tmp_path / "halt.json")["first_provider_failure"] == {
        "error_type": "Non200Response",
        "http_status": 429,
    }
    assert h.load(tmp_path / "halt.json")["failure_record_saved"] is True
    assert not (tmp_path / "blind-review-input.json").exists()
    with pytest.raises(h.Stop, match="PreviousRunNoResume"):
        h.execute(tmp_path, "a" * 64, review, review_sha, factory)


@pytest.mark.skipif(
    not os.environ.get("RERANK_AB_TEST_STORAGE"),
    reason="Optional private-copy integration fixture not supplied",
)
def test_actual_storage_offline_retrieval_and_capture(tmp_path):
    """Existing public product content never leaves the machine: all HTTP is mocked."""
    source = Path(os.environ["RERANK_AB_TEST_STORAGE"])
    for target in (tmp_path / "source-storage", tmp_path / "runtime-storage"):
        target.mkdir(mode=0o700)
        for name in h.STORAGE_FILES:
            shutil.copyfile(source / name, target / name)
    (tmp_path / "cases").mkdir(mode=0o700)
    chunks = h.load(source / "kv_store_text_chunks.json")
    _, _, _, utils, prompts = h.native_modules()
    m = {
        "cases": cases(),
        "schedule": h.schedule(cases()),
        "tokenizer_probe": utils.TiktokenTokenizer("gpt-4o-mini").encode(
            "Offline tokenizer verification. 通用保险条款。\n12345"
        ),
        "templates": {
            "rag_response": prompts["rag_response"],
            "kg_query_context": prompts["kg_query_context"],
            "user_prompt_prefix": "Synthetic policy.\n",
        },
        "sources": {
            v["file_path"]: {"source_sha256": h.digest(v["file_path"].encode())}
            for v in chunks.values()
        },
    }

    def handler(request):
        payload = h.parse(request.content)
        if payload["model"] == h.MODELS["keyword"]:
            return httpx.Response(
                200,
                json=chat(
                    h.MODELS["keyword"],
                    json.dumps(
                        {
                            "high_level_keywords": ["insurance"],
                            "low_level_keywords": ["policy"],
                        }
                    ),
                ),
            )
        if payload["model"] == h.MODELS["embedding"]:
            import numpy as np

            vdb = h.load(source / "vdb_chunks.json")
            vector = (
                np.frombuffer(
                    __import__("base64").b64decode(vdb["matrix"]), dtype=np.float32
                )
                .reshape(-1, 1024)[0]
                .tolist()
            )
            return httpx.Response(
                200,
                json={
                    "id": "embedding-id",
                    "model": h.MODELS["embedding"],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    "data": [
                        {"index": i, "embedding": vector}
                        for i in range(len(payload["input"]))
                    ],
                },
            )
        return httpx.Response(200, json=rerank(len(payload["input"]["documents"])))

    p = provider(tmp_path, handler)
    p.manifest = m
    hashes = h.inventory(tmp_path / "runtime-storage")
    pairs = asyncio.run(h.build_pairs(tmp_path, m, p))
    assert len(pairs) == 3 and p.attempted == len(p.completed) == 9
    assert h.inventory(tmp_path / "runtime-storage") == hashes
    assert len(list((tmp_path / "cases").glob("*/candidate-snapshot.json"))) == 3
