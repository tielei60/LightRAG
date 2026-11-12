"""Offline checks for exact same-call prompt capture and writer guards."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from product.query_trace import (
    QueryInputCapture,
    build_native_reranker,
    require_closed_writer,
    traced_query_role,
)


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_arguments_are_captured_before_same_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            forwarded = []

            async def provider(prompt, **kwargs):
                self.assertTrue(path.exists())
                forwarded.append((prompt, kwargs))
                return "原答"

            capture = QueryInputCapture(
                provider, "configured-model", {"enable_thinking": False}
            )
            capture.begin(path)
            answer = await capture(
                "原始问题",
                system_prompt="完整图和片段原文",
                history_messages=[],
                stream=True,
                hashing_kv=object(),
            )
            saved = json.loads(path.read_text())["calls"][0]
            self.assertEqual(answer, "原答")
            self.assertEqual(saved["system_prompt"], "完整图和片段原文")
            self.assertEqual(saved["prompt"], forwarded[0][0])
            self.assertEqual(saved["history_messages"], [])
            self.assertFalse(saved["provider_options"]["enable_thinking"])
            self.assertNotIn("hashing_kv", saved)

    async def test_failed_provider_retains_input_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"

            async def provider(*args, **kwargs):
                raise RuntimeError("provider failed")

            capture = QueryInputCapture(provider, "model", {})
            capture.begin(path)
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                await capture("问题", system_prompt="原文")
            saved = json.loads(path.read_text())["calls"][0]
            self.assertEqual(saved["state"], "failed")
            self.assertEqual(saved["system_prompt"], "原文")

    async def test_natural_answer_trace_records_consumed_provider_stream_and_native_envelope(
        self,
    ):
        from product.test_grounded_query import native_prompt, wrapped

        calls = []

        async def provider(prompt, **kwargs):
            calls.append((prompt, kwargs))
            if not kwargs.get("stream"):
                return json.dumps(
                    {
                        "supported": True,
                        "scope_consistent": True,
                        "formula_consistent": True,
                        "issues": [],
                        "corrected_answer": None,
                    }
                )

            async def generated():
                yield "这种情形不能退款，"
                yield "因为条款明确不退还现金价值。[1]"

            return generated()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            capture = QueryInputCapture(
                provider, "configured-model", {"enable_thinking": False}
            )
            query = traced_query_role(
                capture,
                {
                    "PRODUCT_GROUNDED_QUERY": "true",
                    "QUERY_LLM_MODEL": "configured-model",
                },
            )
            system = native_prompt(
                [
                    {
                        "reference_id": "1",
                        "content": wrapped(
                            "原文明确不退还现金价值。共同前提及例外保留。"
                        ),
                    }
                ]
            )
            for _ in range(2):
                capture.begin(path)
                result = await query(
                    "能退款吗？", system_prompt=system, history_messages=[], stream=True
                )
                answer = "".join([part async for part in result])
                saved = json.loads(path.read_text())
                self.assertEqual(saved["native_inputs"][0]["system_prompt"], system)
                self.assertEqual(saved["calls"][0]["prompt"], "能退款吗？")
                model_documents = json.loads(
                    saved["calls"][0]["system_prompt"]
                    .split("<source_documents>\n", 1)[1]
                    .split("\n</source_documents>", 1)[0]
                )
                self.assertIn(
                    "原文明确不退还现金价值",
                    json.dumps(model_documents, ensure_ascii=False),
                )
                self.assertEqual(saved["calls"][0]["prompt"], calls[-2][0])
                self.assertTrue(saved["calls"][0]["stream"])
                self.assertEqual(saved["calls"][0]["state"], "completed")
                self.assertEqual(
                    saved["calls"][0]["response_text"],
                    "这种情形不能退款，因为条款明确不退还现金价值。[1]",
                )
                self.assertTrue(answer.startswith(saved["calls"][0]["response_text"]))
                self.assertEqual(
                    [row["text"] for row in saved["calls"][0]["response_chunks"]],
                    ["这种情形不能退款，", "因为条款明确不退还现金价值。[1]"],
                )
                self.assertNotIn("source_units", model_documents)
                self.assertNotIn("U0001", saved["calls"][0]["prompt"])
                self.assertNotIn("unit_ids", saved["calls"][0]["response_text"])
                self.assertNotIn("response_format", calls[-2][1])
            self.assertEqual(len(calls), 4)

    async def test_capture_does_not_claim_iterator_is_completed_before_consumption(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"

            async def provider(*_args, **_kwargs):
                async def stream():
                    yield "第一段"
                    yield "第二段"

                return stream()

            capture = QueryInputCapture(provider, "model", {})
            capture.begin(path)
            result = await capture("问题", system_prompt="完整来源", stream=True)
            self.assertEqual(capture.calls[0]["state"], "streaming")
            self.assertEqual(capture.calls[0]["response_text"], "")
            self.assertEqual(await anext(result), "第一段")
            self.assertEqual(capture.calls[0]["response_text"], "第一段")
            await result.aclose()
            self.assertEqual(capture.calls[0]["state"], "closed")
            self.assertNotEqual(
                json.loads(path.read_text())["calls"][0]["state"], "completed"
            )

    async def test_capture_keeps_partial_provider_stream_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"

            async def provider(*_args, **_kwargs):
                async def stream():
                    yield "已收到的原始模型输出"
                    raise RuntimeError("stream failed")

                return stream()

            capture = QueryInputCapture(provider, "model", {})
            capture.begin(path)
            result = await capture("问题", system_prompt="完整来源", stream=True)
            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                _ = [part async for part in result]
            saved = json.loads(path.read_text())["calls"][0]
            self.assertEqual(saved["state"], "failed")
            self.assertEqual(saved["response_text"], "已收到的原始模型输出")
            self.assertEqual(saved["error_type"], "RuntimeError")

    async def test_plain_query_trace_keeps_original_callback_when_not_enabled(self):
        capture = QueryInputCapture(AsyncMock(), "model", {})
        self.assertIs(traced_query_role(capture, {}), capture)

    async def test_answer_trace_passes_batch_query_limits_without_ambient_env(self):
        capture = QueryInputCapture(AsyncMock(), "model", {})
        with (
            patch("product.answer_query.AnswerQuery") as adapter,
            patch(
                "lightrag.prompt.load_user_prompt_prefix_source",
                return_value="batch trusted prefix",
            ),
            patch.dict(
                "os.environ",
                {"QUERY_MAX_ASYNC_LLM": "99", "QUERY_LLM_TIMEOUT": "999"},
            ),
        ):
            traced_query_role(
                capture,
                {
                    "PRODUCT_GROUNDED_QUERY": "true",
                    "QUERY_LLM_MODEL": "batch-model",
                    "QUERY_MAX_ASYNC_LLM": "3",
                    "QUERY_LLM_TIMEOUT": "19.5",
                    "MAX_ASYNC_LLM": "7",
                    "LLM_TIMEOUT": "120",
                },
            )
        adapter.assert_called_once_with(
            capture,
            record=capture.record_grounding,
            answer_instructions="batch trusted prefix",
            provider_max_async=3,
            provider_timeout=19.5,
        )


class WriterGuardTests(unittest.TestCase):
    def test_live_recorded_service_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            (batch / "server.pid").write_text("123")
            with (
                patch("product.query_trace.os.kill"),
                self.assertRaisesRegex(RuntimeError, "recorded HTTP"),
            ):
                require_closed_writer(batch, {"HOST": "127.0.0.1", "PORT": "9621"})

    def test_incomplete_import_is_refused(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("product.query_trace.socket.socket") as sock,
        ):
            sock.return_value.__enter__.return_value.connect_ex.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "not finished"):
                require_closed_writer(
                    Path(directory), {"HOST": "127.0.0.1", "PORT": "9621"}
                )


class RerankTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_aliyun_hook_forwards_config_and_query_parameters(self):
        provider = AsyncMock(return_value=[{"index": 1, "relevance_score": 0.9}])
        config = {
            "RERANK_BINDING": "aliyun",
            "RERANK_MODEL": "configured-rerank",
            "RERANK_BINDING_HOST": "https://rerank.example/endpoint",
            "RERANK_BINDING_API_KEY": "fixture-key",
        }
        with patch("lightrag.rerank.ali_rerank", provider):
            rerank = build_native_reranker(config)
            result = await rerank(
                "query", ["first", "second"], top_n=1, extra_body={"flag": True}
            )
        provider.assert_awaited_once_with(
            query="query",
            documents=["first", "second"],
            top_n=1,
            api_key="fixture-key",
            model="configured-rerank",
            base_url="https://rerank.example/endpoint",
            extra_body={"flag": True},
        )
        self.assertEqual(result[0]["index"], 1)

    async def test_cohere_hook_matches_server_chunking_parameters(self):
        provider = AsyncMock(return_value=[])
        config = {
            "RERANK_BINDING": "cohere",
            "RERANK_MODEL": "model",
            "RERANK_BINDING_HOST": "https://rerank.example",
            "RERANK_ENABLE_CHUNKING": "true",
            "RERANK_MAX_TOKENS_PER_DOC": "512",
        }
        with patch("lightrag.rerank.cohere_rerank", provider):
            await build_native_reranker(config)("q", ["d"])
        self.assertTrue(provider.call_args.kwargs["enable_chunking"])
        self.assertEqual(provider.call_args.kwargs["max_tokens_per_doc"], 512)
        self.assertIsNone(build_native_reranker({"RERANK_BINDING": "null"}))


if __name__ == "__main__":
    unittest.main()
