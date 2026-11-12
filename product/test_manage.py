"""Offline source-boundary checks; no model calls or business data writes."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from product import manage


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.source = root / "source"
        self.source.mkdir()
        self.manifest = root / "manifest.json"
        self.entries = []
        for index in range(3):
            path = self.source / f"{index}.txt"
            path.write_text(f"fixture-{index}")
            self.entries.append(
                {
                    "product": f"fixture-{index}",
                    "relative_path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

    def write_manifest(self):
        self.manifest.write_text(
            json.dumps({"source_root": str(self.source), "documents": self.entries})
        )

    def test_selected_sources_are_verified(self):
        self.write_manifest()
        with patch.object(manage, "MANIFEST", self.manifest):
            self.assertEqual(len(manage.checked_files()), 3)

    def test_changed_source_is_rejected(self):
        self.write_manifest()
        (self.source / "0.txt").write_text("changed fixture")
        with (
            patch.object(manage, "MANIFEST", self.manifest),
            self.assertRaises(ValueError),
        ):
            manage.checked_files()

    def test_source_symlink_escape_is_rejected(self):
        external = self.source.parent / "outside.txt"
        external.write_text("outside fixture")
        path = self.source / "0.txt"
        path.unlink()
        path.symlink_to(external)
        self.entries[0]["sha256"] = hashlib.sha256(external.read_bytes()).hexdigest()
        self.write_manifest()
        with (
            patch.object(manage, "MANIFEST", self.manifest),
            self.assertRaises(ValueError),
        ):
            manage.checked_files()

    def test_product_scope_cannot_expand(self):
        self.entries.append(
            {
                "product": "fourth-fixture",
                "relative_path": "0.txt",
                "sha256": self.entries[0]["sha256"],
            }
        )
        self.write_manifest()
        with (
            patch.object(manage, "MANIFEST", self.manifest),
            self.assertRaises(ValueError),
        ):
            manage.checked_files()

    def test_single_product_batch_uses_explicit_manifest_scope(self):
        self.entries = self.entries[:1]
        self.manifest.write_text(
            json.dumps(
                {
                    "source_root": str(self.source),
                    "expected_product_count": 1,
                    "documents": self.entries,
                }
            )
        )
        with patch.object(manage, "MANIFEST", self.manifest):
            self.assertEqual(len(manage.checked_files()), 1)

    def test_two_product_batch_uses_explicit_manifest_scope(self):
        self.entries = self.entries[:2]
        self.manifest.write_text(
            json.dumps(
                {
                    "source_root": str(self.source),
                    "expected_product_count": 2,
                    "documents": self.entries,
                }
            )
        )
        with patch.object(manage, "MANIFEST", self.manifest):
            self.assertEqual(len(manage.checked_files()), 2)

    def test_five_product_migration_uses_explicit_manifest_scope(self):
        for index in (3, 4):
            path = self.source / f"{index}.txt"
            path.write_text(f"fixture-{index}")
            self.entries.append(
                {
                    "product": f"fixture-{index}",
                    "relative_path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        self.manifest.write_text(
            json.dumps(
                {
                    "source_root": str(self.source),
                    "expected_product_count": 5,
                    "documents": self.entries,
                }
            )
        )
        with patch.object(manage, "MANIFEST", self.manifest):
            self.assertEqual(len(manage.checked_files()), 5)

    def test_prepared_input_tampering_is_rejected(self):
        payload = self.source.parent / "prepared.json"
        payload.write_text(json.dumps({"file_source": "0.txt", "text": "fixture"}))
        self.entries[0].update(
            prepared_path=payload.name,
            prepared_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
        )
        self.write_manifest()
        payload.write_text("changed")
        with (
            patch.object(manage, "MANIFEST", self.manifest),
            patch.object(manage, "OUTPUT", self.source.parent),
            self.assertRaises(ValueError),
        ):
            manage.checked_files()

    def test_prepared_source_identity_mismatch_is_rejected(self):
        payload = self.source.parent / "prepared.json"
        payload.write_text(json.dumps({"file_source": "other.txt", "text": "fixture"}))
        self.entries[0].update(
            prepared_path=payload.name,
            prepared_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
        )
        self.write_manifest()
        with (
            patch.object(manage, "MANIFEST", self.manifest),
            patch.object(manage, "OUTPUT", self.source.parent),
            self.assertRaises(ValueError),
        ):
            manage.checked_files()


class LocalConnectionTests(unittest.TestCase):
    def test_status_bypasses_system_proxy_for_loopback(self):
        with (
            patch.object(manage, "base_url", return_value="http://127.0.0.1:9621"),
            patch.object(manage.httpx, "Client") as client,
            patch("builtins.print"),
        ):
            client.return_value.__enter__.return_value.get.return_value.json.return_value = {
                "configuration": {}
            }
            manage.status()
        self.assertFalse(client.call_args.kwargs["trust_env"])


class ServerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.output = self.root / "output" / "candidate"
        self.output.mkdir(parents=True)
        self.pid_file = self.output / "server.pid"
        patcher = patch.multiple(
            manage, ROOT=self.root, OUTPUT=self.output, PID_FILE=self.pid_file
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def launch(self, grounded=None):
        config = {"HOST": "127.0.0.1", "PORT": "9621", "WORKSPACE": "fixture"}
        if grounded is not None:
            config["PRODUCT_GROUNDED_QUERY"] = grounded
        with (
            patch.object(manage, "settings", return_value=config),
            patch.object(manage.subprocess, "Popen") as process,
            patch("builtins.print"),
        ):
            process.return_value.pid = 1234
            manage.start()
        return process.call_args

    def test_old_batch_uses_native_launcher(self):
        with patch.dict(manage.os.environ, {"PRODUCT_GROUNDED_QUERY": "true"}):
            call = self.launch()
        self.assertEqual(call.args[0], [str(self.root / ".venv/bin/lightrag-server")])
        self.assertTrue(call.kwargs["start_new_session"])
        self.assertEqual(self.pid_file.read_text(), "1234")

    def test_opt_in_batch_uses_product_bootstrap(self):
        call = self.launch("true")
        self.assertEqual(
            call.args[0], [str(self.root / ".venv/bin/python"), "-m", "product.server"]
        )
        self.assertEqual(call.kwargs["env"]["PRODUCT_GROUNDED_QUERY"], "true")
        self.assertEqual(call.kwargs["cwd"], self.root)

    def test_start_refuses_workspace_with_active_incremental_writer(self):
        lock = self.root / "output/workspaces/fixture/writer.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text('{"pid": 123}')
        with self.assertRaisesRegex(ValueError, "writer reservation"):
            self.launch("true")
        self.assertFalse(self.pid_file.exists())
        self.assertTrue(lock.exists())

    def test_stop_recognizes_product_bootstrap_and_keeps_pid_guard(self):
        self.pid_file.write_text("1234")
        with (
            patch("psutil.Process") as process,
            patch.object(manage.os, "kill") as kill,
            patch("builtins.print"),
        ):
            process.return_value.cwd.return_value = str(self.root)
            process.return_value.cmdline.return_value = [
                "python",
                "-m",
                "product.server",
            ]
            manage.stop()
            kill.assert_called_once_with(1234, manage.signal.SIGTERM)
            kill.reset_mock()
            process.return_value.cwd.return_value = str(self.root / "other-project")
            with self.assertRaisesRegex(RuntimeError, "not this repository"):
                manage.stop()
            kill.assert_not_called()

    def test_stop_rejects_unrelated_python_process(self):
        self.pid_file.write_text("1234")
        with (
            patch("psutil.Process") as process,
            patch.object(manage.os, "kill") as kill,
        ):
            process.return_value.cwd.return_value = str(self.root)
            process.return_value.cmdline.return_value = [
                "python",
                "some-product.server-file.py",
            ]
            with self.assertRaises(RuntimeError):
                manage.stop()
            kill.assert_not_called()


class ProductBootstrapTests(unittest.TestCase):
    def test_wrapper_only_constrains_query_provider(self):
        from product.answer_query import AnswerQuery
        from product.server import wrap_query_role

        async def provider(*_args, **_kwargs):
            return "unused"

        self.assertIs(wrap_query_role("extract", provider), provider)
        self.assertIs(wrap_query_role("keyword", provider), provider)
        query = wrap_query_role("query", provider)
        self.assertIsInstance(query, AnswerQuery)
        self.assertIs(query.__wrapped__, provider)

    def test_query_wrapper_uses_query_provider_limits_from_process_configuration(self):
        from product import server

        async def provider(*_args, **_kwargs):
            return "unused"

        with (
            patch.dict(
                server.os.environ,
                {
                    "QUERY_MAX_ASYNC_LLM": "3",
                    "MAX_ASYNC_LLM": "7",
                    "QUERY_LLM_TIMEOUT": "19.5",
                    "LLM_TIMEOUT": "120",
                },
                clear=True,
            ),
            patch.object(server, "AnswerQuery") as grounded,
            patch.object(
                server,
                "load_user_prompt_prefix_source",
                return_value="configured trusted instructions",
            ),
        ):
            server.wrap_query_role("query", provider)
        grounded.assert_called_once_with(
            provider,
            record=server.record_grounding_event,
            answer_instructions="configured trusted instructions",
            provider_max_async=3,
            provider_timeout=19.5,
        )

    def test_query_wrapper_uses_general_provider_limits_and_validates_them(self):
        from product import server

        async def provider(*_args, **_kwargs):
            return "unused"

        with (
            patch.dict(
                server.os.environ,
                {"MAX_ASYNC_LLM": "4", "LLM_TIMEOUT": "37"},
                clear=True,
            ),
            patch.object(server, "AnswerQuery") as grounded,
            patch.object(
                server,
                "load_user_prompt_prefix_source",
                return_value="configured trusted instructions",
            ),
        ):
            server.wrap_query_role("query", provider)
        self.assertEqual(grounded.call_args.kwargs["provider_max_async"], 4)
        self.assertEqual(grounded.call_args.kwargs["provider_timeout"], 37.0)
        with patch.dict(server.os.environ, {"QUERY_MAX_ASYNC_LLM": "0"}, clear=True):
            with self.assertRaises(ValueError):
                server.wrap_query_role("query", provider)
            self.assertIs(server.wrap_query_role("extract", provider), provider)

    def test_query_wrapper_loads_the_configured_prompt_prefix_file_verbatim(self):
        from product import server

        async def provider(*_args, **_kwargs):
            return "unused"

        with tempfile.TemporaryDirectory() as directory:
            prompt_dir = Path(directory)
            (prompt_dir / "user_prompt").mkdir()
            source = "优先使用用户指定的简短问答风格。\n不省略原文条件。\n"
            (prompt_dir / "user_prompt" / "style.md").write_text(source)
            with (
                patch.dict(
                    server.os.environ,
                    {
                        "PROMPT_DIR": str(prompt_dir),
                        "USER_PROMPT_PREFIX_FILE": "style.md",
                    },
                    clear=True,
                ),
                patch.object(server, "AnswerQuery") as answer,
            ):
                server.wrap_query_role("query", provider)
            self.assertEqual(answer.call_args.kwargs["answer_instructions"], source)

    def test_bootstrap_disables_native_answer_cache_and_uses_official_main(self):
        from product import server

        with (
            patch("sys.argv", ["lightrag-server"]),
            patch.dict(server.os.environ, {"ENABLE_LLM_CACHE": "true"}),
            patch("lightrag.api.lightrag_server.main") as native_main,
        ):
            server.main()
            self.assertEqual(server.os.environ["ENABLE_LLM_CACHE"], "false")
            native_main.assert_called_once_with(
                role_llm_func_wrapper=server.wrap_query_role
            )


if __name__ == "__main__":
    unittest.main()
