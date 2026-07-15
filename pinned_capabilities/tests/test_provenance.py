import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinned_capabilities.manifest import content_hash
from pinned_capabilities.provenance import (
    bind_file,
    bind_manifest,
    bind_nearest_manifest,
    bind_reference,
    bind_snapshot,
    inspect_snapshot,
    sha256_file,
    validate_snapshot,
)


class ProvenanceTests(unittest.TestCase):
    def _save_snapshot(self, path: Path, *, metadata=None, step=120) -> None:
        torch.save(
            {
                "schema_version": 1,
                "step": step,
                "model": {"weight": torch.ones(2)},
                "optimizer": {},
                "scheduler": None,
                "scaler": None,
                "stream": {"position": 4},
                "rng": {"numpy": np.arange(4, dtype=np.uint32)},
                "metadata": metadata or {},
            },
            path,
        )

    def test_hash_and_file_binding_use_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"immutable input\n")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(sha256_file(path, chunk_size=3), expected)
            self.assertEqual(
                bind_file(path),
                {"path": str(path), "size_bytes": 16, "sha256": expected},
            )

    def test_reference_binding_requires_a_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            reference.write_text(json.dumps({"bands": {"solved_c_int": 4.0}}))
            binding = bind_reference(reference)
            self.assertEqual(binding["sha256"], sha256_file(reference))
            reference.write_text("[]")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                bind_reference(reference)

    def test_manifest_binding_verifies_internal_digest_and_nearest_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"kind": "test", "experiment": {"seed": 7}}
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "config": config,
                        "config_sha256": content_hash(config),
                    }
                )
            )
            artifact = root / "seed_7" / "snapshot.pt"
            artifact.parent.mkdir()
            artifact.write_bytes(b"artifact")
            binding = bind_nearest_manifest(artifact)
            self.assertEqual(binding["path"], str(manifest))
            self.assertEqual(binding["config_sha256"], content_hash(config))

            payload = json.loads(manifest.read_text())
            payload["config"]["kind"] = "mutated"
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                bind_manifest(manifest)

    def test_snapshot_inspection_is_restricted_and_manifest_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot.pt"
            config = {"seed": 7, "learning_rate": 0.001}
            self._save_snapshot(
                snapshot,
                metadata={
                    "kind": "expressed_start",
                    "seed": 7,
                    "config": config,
                    "config_sha256": content_hash(config),
                },
            )
            info = inspect_snapshot(snapshot)
            self.assertEqual(info["schema_version"], 1)
            self.assertEqual(info["step"], 120)
            self.assertEqual(info["metadata"]["seed"], 7)
            json.dumps(info, allow_nan=False)

            binding = bind_snapshot(
                snapshot, expected_seed=7, expected_step=120, expected_config=config
            )
            self.assertEqual(binding["sha256"], sha256_file(snapshot))
            self.assertEqual(
                binding["checks"], {"seed": "matched", "step": "matched", "config": "matched"}
            )

    def test_absent_optional_metadata_is_reported_not_assumed(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot.pt"
            self._save_snapshot(snapshot)
            binding = bind_snapshot(
                snapshot,
                expected_seed=3,
                expected_step=120,
                expected_config={"seed": 3},
            )
            self.assertEqual(
                binding["checks"],
                {"seed": "unavailable", "step": "matched", "config": "unavailable"},
            )

    def test_snapshot_identity_mismatches_fail(self):
        info = {
            "schema_version": 1,
            "step": 120,
            "metadata": {"seed": 7, "config_sha256": content_hash({"seed": 7})},
        }
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            validate_snapshot(info, expected_seed=8)
        with self.assertRaisesRegex(ValueError, "step mismatch"):
            validate_snapshot(info, expected_step=121)
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            validate_snapshot(info, expected_config={"seed": 8})

    def test_restricted_loader_rejects_unallowlisted_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "unsafe.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "step": 0,
                    "metadata": {},
                    "untrusted": Path("should-not-be-loaded"),
                },
                snapshot,
            )
            with self.assertRaisesRegex(ValueError, "restricted loading"):
                inspect_snapshot(snapshot)

    def test_snapshot_structure_and_metadata_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot.pt"
            self._save_snapshot(snapshot, step=-1)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                inspect_snapshot(snapshot)
            self._save_snapshot(snapshot, metadata={"bad": torch.ones(1)})
            with self.assertRaisesRegex(ValueError, "unsupported value Tensor"):
                inspect_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
