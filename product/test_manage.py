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


if __name__ == "__main__":
    unittest.main()
