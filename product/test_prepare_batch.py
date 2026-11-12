"""Offline regressions for source-faithful, complete native preparation units."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from product import prepare_batch
from product.prepare_batch import clause_spans, prepare_document
from product.prepare_context import DELIMITER


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "条款.txt"
        self.product = "测试医疗保险"
        self.source = (
            "测试医疗保险条款\n目录\n1.1 合同\n2.1 保险责任\n2.2 其他约定\n7.1 住院医疗费用\n"
            "正文\n1.1 合同 本合同有效。\n"
            "2.1 保险责任 六十周岁以下且投保基本责任方可选择可选责任。\n"
            "2.1.1 基本责任 因疾病住院（见释义）后给付。\n"
            "2.1.2 可选责任 津贴每天十元，但每次最多三十日。\n"
            "2.2 其他约定 除上述情况外仍须通知。\n"
            "7． 释义\n7.1 住院 指正式办理住院并实际入住病房，但不含挂床。\n"
            "7.2 住院医疗费用 指符合当地基本医疗支付范围的费用，不包括自费项目。\n"
            "（1）床位费 不包括陪人床。\n（2）手术费 不包括器官本身。\n"
        )
        self.path.write_text(self.source)

    def prepare(self, **kwargs):
        return prepare_document(
            self.path,
            self.product,
            "policy",
            len,
            target_tokens=1000,
            token_limit=4000,
            **kwargs,
        )

    def test_exact_document_title_has_its_own_narrow_type_support(self):
        _, catalogs, report = self.prepare()
        titles = [
            unit
            for catalog in catalogs
            for unit in catalog["units"]
            if unit.get("evidence_purpose") == "verified-original-document-title"
        ]
        self.assertEqual(len(titles), 1)
        unit = titles[0]
        self.assertEqual(self.source[unit["start"] : unit["end"]], "测试医疗保险条款")
        self.assertEqual(unit["entity_types"], ["产品文档"])
        self.assertEqual(unit["role"], "primary")
        front = next(
            unit
            for unit in catalogs[0]["units"]
            if unit["start"] == 0 and unit["end"] > 20
        )
        self.assertEqual(front["entity_types"], ["Product", "Organization"])
        self.assertEqual(report["document_role"], "policy")

    def test_filename_alone_cannot_create_document_title_evidence(self):
        self.path.write_text(
            self.source.replace("测试医疗保险条款", "测试医疗保险资料")
        )
        _, catalogs, report = self.prepare()
        self.assertEqual(
            sum(
                unit.get("evidence_purpose") == "verified-original-document-title"
                for catalog in catalogs
                for unit in catalog["units"]
            ),
            0,
        )
        self.assertIsNone(report["verified_document_title"])

    def test_table_of_contents_is_not_used_as_real_clause_boundary(self):
        spans = clause_spans(self.source)
        first = next(span for span in spans if span.label == "1.1")
        self.assertEqual(first.start, self.source.index("1.1 合同 本合同"))
        self.assertEqual(
            "".join(self.source[s.start : s.end] for s in spans), self.source
        )

    def test_responsibility_keeps_its_complete_parent_premise(self):
        payload, _, report = self.prepare()
        chunks = payload["text"].split(DELIMITER)
        optional_index = next(
            i
            for i, row in enumerate(report["chunks"])
            if row["clause_ids"] == ["2.1.2"]
        )
        chunk = chunks[optional_index]
        self.assertIn("六十周岁以下且投保基本责任方可选择可选责任", chunk)
        self.assertIn("津贴每天十元，但每次最多三十日", chunk)
        self.assertNotIn("因疾病住院", chunk)
        self.assertEqual(
            report["chunks"][optional_index]["copies"][1]["reason"], "structural-parent"
        )

    def test_explicit_post_definition_is_copied_into_the_native_chunk(self):
        payload, catalogs, report = self.prepare()
        index = next(
            i
            for i, row in enumerate(report["chunks"])
            if row["clause_ids"] == ["2.1.1"]
        )
        chunk = payload["text"].split(DELIMITER)[index]
        self.assertIn("住院 指正式办理住院并实际入住病房，但不含挂床", chunk)
        self.assertTrue(
            any(
                unit["role"] == "explicit-definition-reference"
                for unit in catalogs[index]["units"]
            )
        )

    def test_primary_coverage_counts_source_once_despite_dependency_copies(self):
        _, catalogs, report = self.prepare()
        self.assertEqual(
            "".join(self.source[row["start"] : row["end"]] for row in report["chunks"]),
            self.source,
        )
        for catalog in catalogs:
            self.assertEqual(
                len({u["id"] for u in catalog["units"]}), len(catalog["units"])
            )

    def test_definition_subitems_keep_the_full_definition_and_exceptions(self):
        payload, catalogs, _ = self.prepare()
        selected = [
            u
            for catalog in catalogs
            for u in catalog["units"]
            if self.source[u["start"] : u["end"]].startswith("7.2")
        ]
        self.assertTrue(selected)
        text = self.source[selected[0]["start"] : selected[0]["end"]]
        self.assertIn("不包括自费项目", text)
        self.assertIn("床位费 不包括陪人床", text)
        self.assertIn("手术费 不包括器官本身", text)
        self.assertIn(text, payload["text"])

    def test_stable_ids_bind_source_hash_and_original_position(self):
        first = self.prepare()[1]
        self.assertEqual(first, self.prepare()[1])
        self.path.write_text(self.source.replace("十元", "二十元"))
        changed = self.prepare()[1]
        self.assertNotEqual(first[0]["units"][0]["id"], changed[0]["units"][0]["id"])

    def test_plan_table_remains_whole_with_its_footnote(self):
        table = "# 测试医疗保险\n\n## 一、计划\n|责任|计划一|计划二|\n|---|---|---|\n|津贴|—|100|\n\n仅在先选择基本责任时可选。\n\n## 二、年龄\n六十周岁以下。"
        self.path.write_text(table)
        payload, catalogs, _ = prepare_document(
            self.path, self.product, "underwriting", len
        )
        self.assertEqual(len(payload["text"].split(DELIMITER)), 3)
        unit = next(
            u
            for catalog in catalogs
            for u in catalog["units"]
            if "|责任|" in table[u["start"] : u["end"]]
        )
        self.assertIn("计划二", table[unit["start"] : unit["end"]])
        self.assertIn("仅在先选择基本责任时可选", table[unit["start"] : unit["end"]])

    def test_responsibility_subheadings_prevent_sibling_rule_leakage(self):
        text = (
            "测试医疗保险条款\n2.1 保险责任 先投保基本责任才可选择可选责任。\n"
            "2.1.1 基本责任\n甲医疗保险\n金\n"
            "被保险人发生甲情形时给付。期满后仅本项责任延续二十日。\n"
            "乙医疗保险金\n被保险人发生乙情形时给付。\n"
            "本公司累计给付的甲医疗保险金与乙医疗保险金之和不得超过约定金额。\n"
            "2.2 责任免除 故意行为不承担责任。"
        )
        self.path.write_text(text)
        payload, catalogs, report = self.prepare()
        indexed = list(
            zip(
                payload["text"].split(DELIMITER),
                catalogs,
                report["chunks"],
                strict=True,
            )
        )
        first = next(
            row
            for row in indexed
            if row[1]["scope"]["responsibility_heading"] == "甲医疗保险\n金"
        )
        second = next(
            row
            for row in indexed
            if row[1]["scope"]["responsibility_heading"] == "乙医疗保险金"
        )
        self.assertIn("延续二十日", first[0])
        self.assertNotIn("延续二十日", second[0])
        for chunk, catalog, _ in (first, second):
            self.assertIn("先投保基本责任才可选择可选责任", chunk)
            self.assertIn("之和不得超过约定金额", chunk)
            self.assertEqual(catalog["scope"]["document_role"], "policy")
            self.assertEqual(catalog["scope"]["unit_kind"], "responsibility")
        self.assertTrue(
            any(
                unit.get("evidence_purpose") == "shared-sibling-rule"
                for unit in first[1]["units"]
            )
        )
        self.assertEqual(
            "".join(text[row["start"] : row["end"]] for row in report["chunks"]), text
        )

    def test_disclosure_responsibilities_keep_common_premise_without_sibling_body(self):
        text = (
            "测试医疗保险产品说明\n一、产品基本特征\n（一）保险责任\n"
            "只有先投保基本责任方可约定可选责任，且以保险单记载为准。\n"
            "1.基本责任\n【甲医疗保险金】\n被保险人发生甲情形给付，期满延续二十日。\n"
            "【乙医疗保险金】\n被保险人发生乙情形给付。\n"
            "2.可选责任\n【丙津贴保险金】\n被保险人住院超过三日后给付，每次最多十日。\n"
            "（二）责任免除\n故意行为除外。\n二、退保\n（一）退保\n已发生给付或已发生约定事故但尚未支付时，不退现金价值。\n"
        )
        self.path.write_text(text)
        payload, catalogs, _ = prepare_document(
            self.path, self.product, "disclosure", len
        )
        rows = list(zip(payload["text"].split(DELIMITER), catalogs, strict=True))
        second = next(
            row
            for row in rows
            if row[1]["scope"]["responsibility_heading"] == "【乙医疗保险金】"
        )
        self.assertNotIn("延续二十日", second[0])
        self.assertIn("只有先投保基本责任方可约定可选责任", second[0])
        optional = next(
            row
            for row in rows
            if row[1]["scope"]["responsibility_heading"] == "【丙津贴保险金】"
        )
        self.assertIn("只有先投保基本责任方可约定可选责任", optional[0])
        self.assertIn("住院超过三日后给付，每次最多十日", optional[0])
        refund = [chunk for chunk, _ in rows if "尚未支付时，不退现金价值" in chunk]
        self.assertEqual(len(refund), 1)

    def test_disclosure_plan_heading_limits_parent_and_shared_rule_scope(self):
        text = (
            "测试医疗保险产品说明书\n一、产品基本特征\n（一）保险责任\n"
            "本合同含两个保障计划，按保险单所载计划承担相应保险责任。\n"
            "保障计划一：\n在本合同保险期间内，按下列约定承担责任。\n"
            "【生存保险金】\n每个保单周年日生存，给付生存保险金。\n"
            "【身故保险金】\n若身故，本合同终止，向身故保险金受益人给付"
            "累计所交保险费扣除累计已给付生存保险金后的余额。\n"
            "保障计划二：\n在本合同保险期间内，按下列约定承担责任。\n"
            "【生存保险金】\n每个保单周年日生存，给付生存保险金。\n"
            "（二）责任免除\n仅按合同约定处理。\n"
        )
        self.path.write_text(text)
        payload, catalogs, report = prepare_document(
            self.path, self.product, "disclosure", len
        )
        rows = [
            (chunk, catalog)
            for chunk, catalog in zip(
                payload["text"].split(DELIMITER), catalogs, strict=True
            )
            if catalog["scope"]["unit_kind"] == "responsibility"
        ]
        self.assertEqual(len(rows), 3)
        plan_two = rows[-1]
        self.assertIn("保障计划二：", plan_two[1]["scope"]["heading_path"])
        self.assertIn("保障计划二：", plan_two[0])
        self.assertNotIn("保障计划一", plan_two[0])
        self.assertNotIn("身故保险金", plan_two[0])
        self.assertNotIn("扣除累计已给付", plan_two[0])
        for chunk, catalog in rows[:2]:
            self.assertIn("保障计划一：", catalog["scope"]["heading_path"])
            self.assertNotIn("保障计划二", chunk)
        self.assertNotIn("身故保险金", rows[0][0])
        self.assertEqual(
            "".join(text[row["start"] : row["end"]] for row in report["chunks"]), text
        )

    def test_repeated_responsibility_name_is_not_two_distinct_shared_targets(self):
        text = (
            "测试医疗保险产品说明书\n一、产品基本特征\n（一）保险责任\n"
            "【生存保险金】\n按第一种安排给付。\n"
            "【生存保险金】\n本段生存保险金仅适用于第二种安排。\n"
            "（二）责任免除\n按合同约定处理。\n"
        )
        self.path.write_text(text)
        payload, catalogs, _ = prepare_document(
            self.path, self.product, "disclosure", len
        )
        first = next(
            chunk
            for chunk, catalog in zip(
                payload["text"].split(DELIMITER), catalogs, strict=True
            )
            if catalog["scope"]["unit_kind"] == "responsibility"
        )
        self.assertNotIn("仅适用于第二种安排", first)

    def test_plan_parent_can_contain_basic_and_optional_responsibility_groups(self):
        text = (
            "测试医疗保险产品说明书\n一、产品基本特征\n（一）保险责任\n"
            "按所选保障计划承担责任。\n保障计划1：\n仅本计划适用。\n"
            "1.基本责任\n【甲保险金】\n符合甲条件给付。\n"
            "2.可选责任\n【乙保险金】\n符合乙条件给付。\n"
            "保障计划2：\n仅另一计划适用。\n"
            "【丙保险金】\n符合丙条件给付。\n"
        )
        self.path.write_text(text)
        payload, catalogs, _ = prepare_document(
            self.path, self.product, "disclosure", len
        )
        optional = next(
            (chunk, catalog)
            for chunk, catalog in zip(
                payload["text"].split(DELIMITER), catalogs, strict=True
            )
            if catalog["scope"]["responsibility_heading"] == "【乙保险金】"
        )
        self.assertIn("保障计划1：", optional[1]["scope"]["heading_path"])
        self.assertIn("2.可选责任", optional[1]["scope"]["heading_path"])
        self.assertIn("仅本计划适用", optional[0])
        self.assertNotIn("仅另一计划适用", optional[0])

    def test_definitions_are_separate_complete_units_not_token_packed(self):
        _, catalogs, report = self.prepare()
        frames = [
            row for row in report["chunks"] if row["scope"]["unit_kind"] == "definition"
        ]
        self.assertEqual([row["clause_ids"] for row in frames], [["7.1"], ["7.2"]])
        expense = next(
            unit
            for catalog in catalogs
            for unit in catalog["units"]
            if unit["role"] == "primary"
            and unit.get("scope", {}).get("unit_kind") == "definition"
            and self.source[unit["start"] : unit["end"]].startswith("7.2")
        )
        self.assertIn("不包括自费项目", self.source[expense["start"] : expense["end"]])
        self.assertIn(
            "（2）手术费 不包括器官本身", self.source[expense["start"] : expense["end"]]
        )

    def test_table_facts_bind_zero_cell_to_exact_plan_responsibility_scope(self):
        text = (
            "# 测试医疗保险\n\n## 计划与保障\n"
            "| 责任 | 少儿计划 | 成人计划 |\n| --- | --- | --- |\n"
            "| 甲责任 | 0元 | 5万元 |\n| 乙责任 | — | 0元／天 |\n"
            "注：须先选择基本责任。\n\n## 职业要求\n仅限定于原文明确的人群。"
        )
        self.path.write_text(text)
        payload, catalogs, _ = prepare_document(
            self.path, self.product, "underwriting", len
        )
        facts = [
            (catalog, fact) for catalog in catalogs for fact in catalog["table_facts"]
        ]
        self.assertEqual(len(facts), 4)
        zero = next(
            fact
            for _, fact in facts
            if fact["row_header"] == "甲责任" and fact["column_header"] == "少儿计划"
        )
        self.assertEqual(zero["cell_kind"], "zero")
        self.assertEqual(zero["raw_value"], "0元")
        self.assertEqual(text[zero["cell_start"] : zero["cell_end"]], "0元")
        chunk = next(
            chunk for chunk in payload["text"].split(DELIMITER) if "| 甲责任 |" in chunk
        )
        self.assertIn("须先选择基本责任", chunk)
        self.assertTrue(
            all(
                catalog["scope"]["document_role"] == "underwriting"
                for catalog, _ in facts
            )
        )

    def test_oversized_complete_unit_fails_instead_of_silently_splitting(self):
        with self.assertRaisesRegex(ValueError, "exceeds token limit"):
            prepare_document(self.path, self.product, "policy", len, token_limit=10)

    def test_unproven_product_or_reserved_delimiter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            prepare_document(self.path, "另一个产品", "policy", len)
        self.path.write_text(self.source + DELIMITER)
        with self.assertRaisesRegex(ValueError, "delimiter"):
            self.prepare()

    def test_catalog_hash_matches_exact_native_chunk_text(self):
        payload, catalogs, _ = self.prepare()
        for chunk, catalog in zip(
            payload["text"].split(DELIMITER), catalogs, strict=True
        ):
            self.assertEqual(chunk, chunk.strip())
            self.assertEqual(prepare_batch.digest(chunk), catalog["content_sha256"])
            for unit in catalog["units"]:
                self.assertIn(self.source[unit["start"] : unit["end"]], chunk)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.config = self.root / ".env"
        self.config.write_text(
            "EMBEDDING_TOKEN_LIMIT=8192\nQUERY_LLM_MODEL=qwen3.8-max\nQUERY_OPENAI_LLM_EXTRA_BODY='{\"enable_thinking\": false}'\n"
        )
        for name in ("甲", "乙"):
            directory = self.sources / name
            directory.mkdir()
            for filename in (
                f"测试{name}保险条款.pdf",
                f"测试{name}保险产品说明书.pdf",
                f"测试{name}保险投保及核保规则.md",
                f"测试{name}保险费率表.pdf",
            ):
                (directory / filename).write_text(
                    f"测试{name}保险\n1.1 条件 不得忽略例外。"
                )
        self.root_patch = patch.object(prepare_batch, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.extract_patch = patch.object(
            prepare_batch,
            "extract_source",
            side_effect=lambda path: (path.read_text(), [(0, len(path.read_text()))]),
        )
        self.extract_patch.start()
        self.addCleanup(self.extract_patch.stop)

    def build(self):
        return prepare_batch.build_batch(
            "two",
            ["甲", "乙"],
            "products_two",
            source_root=self.sources,
            config_path=self.config,
        )

    def test_two_product_manifest_inventories_exclusions_and_source_hashes(self):
        manifest = self.build()
        self.assertEqual(manifest["expected_product_count"], 2)
        self.assertEqual(len(manifest["documents"]), 6)
        self.assertEqual(len(manifest["excluded_documents"]), 2)
        for entry in manifest["documents"]:
            target = self.root / "output/two" / entry["prepared_path"]
            self.assertEqual(
                prepare_batch.digest(target.read_bytes()), entry["prepared_sha256"]
            )
        runtime = self.root / "output/two/runtime.env"
        self.assertEqual(runtime.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            prepare_batch.dotenv_values(runtime)["QUERY_LLM_MODEL"], "qwen3.8-max"
        )
        self.assertEqual(
            json.loads(
                prepare_batch.dotenv_values(runtime)["QUERY_OPENAI_LLM_EXTRA_BODY"]
            ),
            {"enable_thinking": False},
        )

    def test_existing_batch_is_not_overwritten(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()

    def test_existing_workspace_requires_explicit_incremental_mode(self):
        (self.root / "rag_storage/products_two").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "--append"):
            self.build()
        self.assertFalse((self.root / "output/two").exists())

    def test_incremental_preparation_preserves_target_query_configuration(self):
        (self.root / "rag_storage/products_two").mkdir(parents=True)
        self.config.write_text(
            self.config.read_text()
            + "WORKSPACE=products_two\nPORT=9623\nMAX_ENTITY_TOKENS=2000\n"
            + "MAX_RELATION_TOKENS=2500\nPROMPT_DIR=/frozen/profile\n"
        )
        manifest = prepare_batch.build_batch(
            "append-one",
            ["甲"],
            "products_two",
            source_root=self.sources,
            config_path=self.config,
            append=True,
        )
        config = prepare_batch.dotenv_values(
            self.root / "output/append-one/runtime.env"
        )
        self.assertEqual(manifest["ingestion_mode"], "append")
        self.assertEqual(len(manifest["documents"]), 3)
        self.assertEqual(config["WORKSPACE"], "products_two")
        self.assertEqual(config["PORT"], "9623")
        self.assertEqual(config["MAX_ENTITY_TOKENS"], "2000")
        self.assertEqual(config["MAX_RELATION_TOKENS"], "2500")
        self.assertEqual(config["PROMPT_DIR"], "/frozen/profile")

    def test_incremental_preparation_refuses_missing_or_different_target(self):
        options = {
            "source_root": self.sources,
            "config_path": self.config,
            "append": True,
        }
        with self.assertRaisesRegex(ValueError, "existing workspace"):
            prepare_batch.build_batch("append-one", ["甲"], "products_two", **options)
        (self.root / "rag_storage/products_two").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "target workspace"):
            prepare_batch.build_batch("append-one", ["甲"], "products_two", **options)
        self.assertFalse((self.root / "output/append-one").exists())

    def test_database_append_preparation_does_not_require_local_storage_files(self):
        self.config.write_text(
            self.config.read_text() + "WORKSPACE=master\nPORT=9623\n"
            "LIGHTRAG_KV_STORAGE=PGKVStorage\nLIGHTRAG_VECTOR_STORAGE=PGVectorStorage\n"
            "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage\nLIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage\n"
        )
        manifest = prepare_batch.build_batch(
            "db-append",
            ["甲"],
            "master",
            source_root=self.sources,
            config_path=self.config,
            append=True,
        )
        self.assertEqual(manifest["workspace"], "master")
        self.assertEqual(manifest["ingestion_mode"], "append")
        self.assertFalse((self.root / "rag_storage/master").exists())

    def test_missing_required_source_prevents_partial_batch_output(self):
        (self.sources / "乙" / "测试乙保险投保及核保规则.md").unlink()
        with self.assertRaisesRegex(ValueError, "required document role"):
            self.build()
        self.assertFalse((self.root / "output/two").exists())

    def test_source_directory_escape_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.sources / "escape").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            prepare_batch.discover(self.sources, ["escape"])


if __name__ == "__main__":
    unittest.main()
