"""Offline verification of persistence checks and same-request query evidence."""

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import networkx as nx
import numpy as np

from product.verify_batch import (
    QueryCase,
    checked_child,
    digest,
    fixed_query_settings,
    percentile,
    public_rerank_settings,
    run_checks,
    run_query,
    summarize_timings,
    verification_source,
    verification_storage,
    verify_snapshot,
)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_incremental_readback_uses_registered_workspace_instead_of_last_batch(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    batch, accumulated = tmp_path / "batch", tmp_path / "accumulated"
    with (
        patch("product.workspace_registry.registry_path", return_value=registry),
        patch(
            "product.workspace_registry.workspace_snapshot", return_value=accumulated
        ) as snapshot,
    ):
        assert verification_source(tmp_path, batch, "products") == accumulated
    snapshot.assert_called_once_with(tmp_path, "products")


def test_unregistered_readback_preserves_single_batch_behavior(tmp_path):
    with (
        patch(
            "product.workspace_registry.registry_path",
            return_value=tmp_path / "absent.json",
        ),
        patch("product.workspace_registry.workspace_snapshot") as snapshot,
    ):
        assert (
            verification_source(tmp_path, tmp_path / "batch", "products")
            == tmp_path / "batch"
        )
    snapshot.assert_not_called()


def test_database_verification_reads_database_without_using_a_stale_local_target(
    tmp_path,
):
    stale = tmp_path / "storage/master"
    stale.mkdir(parents=True)
    export = tmp_path / "database-export"

    async def snapshot(config, root, directory):
        assert config["WORKSPACE"] == "master"
        assert root == tmp_path
        assert directory.parent == tmp_path / "batch/verification"
        return export

    config = {"WORKSPACE": "master", "WORKING_DIR": str(tmp_path / "storage")}
    with (
        patch("product.database_storage.is_database_backend", return_value=True),
        patch(
            "product.database_storage.snapshot_database_config", side_effect=snapshot
        ),
    ):
        assert verification_storage(tmp_path, tmp_path / "batch", config) == export


class ReadbackTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.batch, self.storage = self.root / "batch", self.root / "storage"
        self.batch.mkdir()
        self.storage.mkdir()
        self.product = "测试产品"
        self.content = "【原文】测试产品责任甲：等待30日。"
        self.doc_id = "doc-fixture"
        self.chunk_id = self.doc_id + "-chunk-000"
        self.label = "条款.md"
        payload = {
            "text": self.content,
            "file_source": self.label,
            "chunking": {"params": {"split_by_character": "<BOUNDARY>"}},
        }
        prepared = self.batch / "prepared.json"
        write(prepared, payload)
        write(
            self.batch / "manifest.json",
            {
                "workspace": "fixture",
                "documents": [
                    {
                        "product": self.product,
                        "prepared_path": prepared.name,
                        "prepared_sha256": hashlib.sha256(
                            prepared.read_bytes()
                        ).hexdigest(),
                        "ingestion_label": "docs/insurance_docs/测试产品/" + self.label,
                    }
                ],
            },
        )
        self.statuses = {
            self.doc_id: {
                "status": "processed",
                "file_path": self.label,
                "chunks_list": [self.chunk_id],
            }
        }
        self.chunks = {
            self.chunk_id: {"content": self.content, "full_doc_id": self.doc_id}
        }
        write(self.storage / "kv_store_doc_status.json", self.statuses)
        write(
            self.storage / "kv_store_full_docs.json",
            {self.doc_id: {"content": self.content}},
        )
        write(self.storage / "kv_store_text_chunks.json", self.chunks)
        self.graph = nx.Graph()
        for name in [self.product, self.product + "｜责任甲"]:
            self.graph.add_node(name, source_id=self.chunk_id, description=self.content)
        self.graph.add_edge(
            self.product,
            self.product + "｜责任甲",
            source_id=self.chunk_id,
            description=self.content,
        )
        nx.write_graphml(
            self.graph, self.storage / "graph_chunk_entity_relation.graphml"
        )
        self.write_vectors(
            "chunks", [{"__id__": self.chunk_id, "content": self.content}]
        )
        self.write_vectors(
            "entities",
            [
                {"__id__": digest(name, "ent-"), "content": self.content}
                for name in self.graph.nodes
            ],
        )
        self.write_vectors(
            "relationships",
            [
                {
                    "__id__": digest("".join(sorted((a, b))), "rel-"),
                    "content": self.content,
                }
                for a, b in self.graph.edges
            ],
        )

    def write_vectors(self, namespace, rows):
        matrix = np.ones((len(rows), 2), dtype=np.float32)
        write(
            self.storage / f"vdb_{namespace}.json",
            {
                "embedding_dim": 2,
                "data": rows,
                "matrix": base64.b64encode(matrix.tobytes()).decode(),
            },
        )

    def report(self):
        return verify_snapshot(self.batch, self.storage, 2)

    def test_complete_snapshot_passes_without_writing_storages(self):
        before = {p.name: p.read_bytes() for p in self.storage.iterdir()}
        report = self.report()
        self.assertTrue(report["passed"], report["issues"])
        self.assertEqual(report["documents"], 1)
        self.assertEqual(
            report["vectors"], {"entities": 2, "relationships": 1, "chunks": 1}
        )
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.storage.iterdir()}
        )

    def test_failed_document_is_not_reported_complete(self):
        self.statuses[self.doc_id]["status"] = "failed"
        write(self.storage / "kv_store_doc_status.json", self.statuses)
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                issue.startswith("document_not_processed") for issue in report["issues"]
            )
        )

    def test_chunk_corruption_fails_exact_content_check(self):
        self.chunks[self.chunk_id]["content"] = "删去等待期"
        write(self.storage / "kv_store_text_chunks.json", self.chunks)
        self.assertIn(f"chunk_text_mismatch:{self.chunk_id}", self.report()["issues"])

    def test_source_label_mismatch_is_rejected_after_basename_normalization(self):
        self.statuses[self.doc_id]["file_path"] = "其他条款.md"
        write(self.storage / "kv_store_doc_status.json", self.statuses)
        self.assertIn("document_source_labels_mismatch", self.report()["issues"])

    def test_repeated_prepared_text_requires_matching_multiplicity(self):
        prepared = self.batch / "prepared.json"
        payload = json.loads(prepared.read_text())
        payload["text"] += "<BOUNDARY>" + self.content
        write(prepared, payload)
        manifest = json.loads((self.batch / "manifest.json").read_text())
        manifest["documents"][0]["prepared_sha256"] = hashlib.sha256(
            prepared.read_bytes()
        ).hexdigest()
        write(self.batch / "manifest.json", manifest)
        self.assertTrue(
            any(
                issue.startswith("chunk_content_multiset_mismatch")
                for issue in self.report()["issues"]
            )
        )

    def test_untracked_chunk_is_rejected_even_if_content_matches(self):
        self.statuses[self.doc_id]["chunks_list"] = []
        write(self.storage / "kv_store_doc_status.json", self.statuses)
        self.assertIn("document_chunk_membership_mismatch", self.report()["issues"])

    def test_actual_ordinal_chunk_ids_preserve_cross_product_source_checks(self):
        product, label = "另一测试产品", "其他条款.md"
        content, doc_id, chunk_id = (
            "另一测试产品责任：无等待期。",
            "doc-other",
            "doc-other-chunk-000",
        )
        prepared = self.batch / "other.json"
        write(
            prepared,
            {
                "text": content,
                "file_source": label,
                "chunking": {"params": {"split_by_character": "<BOUNDARY>"}},
            },
        )
        manifest = json.loads((self.batch / "manifest.json").read_text())
        manifest["documents"].append(
            {
                "product": product,
                "prepared_path": prepared.name,
                "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
                "ingestion_label": "docs/" + label,
            }
        )
        write(self.batch / "manifest.json", manifest)
        self.statuses[doc_id] = {
            "status": "processed",
            "file_path": label,
            "chunks_list": [chunk_id],
        }
        self.chunks[chunk_id] = {"content": content, "full_doc_id": doc_id}
        write(self.storage / "kv_store_doc_status.json", self.statuses)
        write(self.storage / "kv_store_text_chunks.json", self.chunks)
        write(
            self.storage / "kv_store_full_docs.json",
            {self.doc_id: {"content": self.content}, doc_id: {"content": content}},
        )
        self.write_vectors(
            "chunks",
            [
                {"__id__": key, "content": row["content"]}
                for key, row in self.chunks.items()
            ],
        )
        self.graph.nodes[self.product]["source_id"] = chunk_id
        nx.write_graphml(
            self.graph, self.storage / "graph_chunk_entity_relation.graphml"
        )
        report = self.report()
        self.assertEqual(
            report["issues"], [f"entity_cross_product_source:{self.product}:{chunk_id}"]
        )

    def test_unknown_graph_source_fails_attribution_check(self):
        self.graph.nodes[self.product]["source_id"] = "chunk-absent"
        nx.write_graphml(
            self.graph, self.storage / "graph_chunk_entity_relation.graphml"
        )
        self.assertTrue(
            any(
                issue.startswith("entity_source_missing")
                for issue in self.report()["issues"]
            )
        )

    def test_missing_vector_fails_set_check(self):
        self.write_vectors("chunks", [])
        self.assertIn("vector_index_set_mismatch:chunks", self.report()["issues"])

    def test_zero_vector_fails_embedding_check(self):
        path = self.storage / "vdb_chunks.json"
        data = json.loads(path.read_text())
        data["matrix"] = base64.b64encode(
            np.zeros((1, 2), dtype=np.float32).tobytes()
        ).decode()
        write(path, data)
        self.assertIn("vector_invalid:chunks", self.report()["issues"])

    def test_prepared_artifact_tampering_is_rejected(self):
        (self.batch / "prepared.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Prepared input changed"):
            self.report()

    def test_path_escape_is_rejected(self):
        external = self.root / "external.json"
        external.write_text("{}")
        with self.assertRaisesRegex(ValueError, "escapes"):
            checked_child(self.batch, "../external.json")


class QueryTests(unittest.TestCase):
    def test_stream_captures_references_and_answer_from_same_request(self):
        requests = []

        def handle(request):
            requests.append((request.url.path, json.loads(request.content)))
            return httpx.Response(
                200,
                text='{"progress":"retrieving"}\n{"references":[{"file_path":"原件.md","reference_id":"1","content":["约定比例的60%"]}]}\n{"response":"保留约定比例"}\n{"response":"和结算前提[1]"}\n{"response_time":1.2}\n',
            )

        with httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(handle)
        ) as client:
            result = run_query(
                client,
                QueryCase("ratio", "比例是多少？", ("约定比例的60%",)),
                fixed_query_settings({}),
            )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][0], "/query/stream")
        self.assertTrue(requests[0][1]["include_chunk_content"])
        self.assertEqual(result["answer"], "保留约定比例和结算前提[1]")
        self.assertTrue(result["evidence_term_checks"]["passed"])
        self.assertIn("not captured", result["same_request_context_kind"])

    def test_missing_evidence_is_reported_without_grading_answer(self):
        def handle(request):
            return httpx.Response(
                200, json={"response": "固定60%", "references": [{"content": ["60%"]}]}
            )

        with httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(handle)
        ) as client:
            result = run_query(
                client,
                QueryCase("ratio", "比例", ("约定比例的60%",)),
                fixed_query_settings({}),
                stream=False,
            )
        self.assertEqual(result["evidence_term_checks"]["missing"], ["约定比例的60%"])
        self.assertEqual(result["business_review"], "pending")

    def test_stream_error_is_not_recorded_as_success(self):
        with (
            httpx.Client(
                base_url="http://test",
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, text='{"error":"provider failed"}\n')
                ),
            ) as client,
            self.assertRaisesRegex(RuntimeError, "provider failed"),
        ):
            run_query(client, QueryCase("fixture", "问题"), fixed_query_settings({}))

    def test_fixed_settings_apply_to_all_questions(self):
        settings = fixed_query_settings({"TOP_K": "40", "MAX_TOTAL_TOKENS": "20000"})
        self.assertEqual(settings["top_k"], 40)
        self.assertEqual(settings["max_total_tokens"], 20000)
        self.assertEqual(settings["conversation_history"], [])
        self.assertFalse(settings["enable_rerank"])

    def test_configured_reranker_is_enabled_and_credentials_are_not_reported(self):
        config = {
            "RERANK_BINDING": "aliyun",
            "RERANK_MODEL": "configured-rerank",
            "RERANK_BINDING_API_KEY": "private-fixture-key",
        }
        self.assertTrue(fixed_query_settings(config)["enable_rerank"])
        self.assertEqual(
            public_rerank_settings(config)["RERANK_MODEL"], "configured-rerank"
        )
        self.assertNotIn(
            "private-fixture-key", json.dumps(public_rerank_settings(config))
        )
        self.assertFalse(
            fixed_query_settings({"RERANK_BINDING": "null"})["enable_rerank"]
        )

    def test_timings_keep_first_and_repeated_requests_separate(self):
        rows = [
            {
                "cache_condition": "first_request",
                "total_seconds": 10,
                "first_output_seconds": 4,
            },
            {
                "cache_condition": "repeated_request",
                "total_seconds": 2,
                "first_output_seconds": 1,
            },
            {"cache_condition": "repeated_request", "error": "failed"},
        ]
        result = summarize_timings(rows)
        self.assertEqual(result["first_request"]["samples"], 1)
        self.assertEqual(result["repeated_request"]["samples"], 1)
        self.assertEqual(result["repeated_request"]["total_p50_seconds"], 2)
        self.assertIsNone(percentile([], 0.95))

    def test_wrong_server_workspace_is_rejected_before_query(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"WORKSPACE": "expected"}
            client = httpx.Client(
                base_url="http://test",
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200, json={"configuration": {"workspace": "other"}}
                    )
                ),
            )
            with (
                patch(
                    "product.verify_batch.httpx.Client", return_value=client
                ) as factory,
                self.assertRaisesRegex(ValueError, "workspace"),
            ):
                run_checks(
                    Path(directory),
                    "http://test",
                    config,
                    (QueryCase("fixture", "问题"),),
                    1,
                    True,
                )
            self.assertFalse(factory.call_args.kwargs["trust_env"])


if __name__ == "__main__":
    unittest.main()
