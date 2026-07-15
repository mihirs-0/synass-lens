import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from pinned_capabilities.config import (
    Gate0Config,
    MBCExperimentConfig,
    MetricConfig,
    StateConfig,
)
from pinned_capabilities.gate0 import deep_linear_control
from pinned_capabilities.gate0_calibration import (
    adjudicate_gate0_calibration,
    adjudicate_gate0_erasure_precheck,
    bind_passed_gate0_calibration,
)
from pinned_capabilities.manifest import content_hash, freeze_manifest
from pinned_capabilities.provenance import bind_file, bind_nearest_manifest, bind_snapshot


class Gate0CalibrationTests(unittest.TestCase):
    protocol_version = "test-1.4.0"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gate = Gate0Config()
        self.metric = MetricConfig()
        self.state = StateConfig()
        self.experiment = replace(
            MBCExperimentConfig(),
            seed=self.gate.calibration_seed,
            batch_size=self.gate.calibration_batch_size,
            device="cpu",
        )

        source = self.root / "source"
        freeze_manifest({"kind": "calibration_snapshot_source"}, source / "manifest.json")
        self.snapshot = source / "seed_100" / "solved_snapshot.pt"
        self.snapshot.parent.mkdir()
        torch.save(
            {
                "schema_version": 1,
                "step": 6_400,
                "metadata": {
                    "kind": "solved_reference",
                    "seed": self.gate.calibration_seed,
                    "config": asdict(self.experiment),
                    "config_sha256": content_hash(self.experiment),
                },
            },
            self.snapshot,
        )

        self.positive = self.root / "positive" / "positive_control.json"
        self.positive.parent.mkdir()
        self.positive.write_text(
            json.dumps(deep_linear_control((0.5, 0.9, 1.0, 1.1, 1.5)))
        )
        freeze_manifest(
            {
                "kind": "gate0_positive_control",
                "protocol_version": self.protocol_version,
                "learning_rates": [0.5, 0.9, 1.0, 1.1, 1.5],
                "target": 1.0,
            },
            self.positive.parent / "manifest.json",
        )
        self.erasure = self.root / "erasure" / "scan.json"
        self.local = self.root / "local" / "scan.json"
        self.erasure.parent.mkdir()
        self.local.parent.mkdir()
        self._write_erasure(("retained", "erased", "retained", "erased", "retained"))
        self._write_local()
        self._write_scan_manifest(self.erasure, "gate0_erasure_scan")
        self._write_scan_manifest(self.local, "gate0_local_stability_scan")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _snapshot_input(self):
        return {
            "artifact": bind_snapshot(
                self.snapshot,
                expected_seed=self.gate.calibration_seed,
                expected_config=self.experiment,
            ),
            "source_manifest": bind_nearest_manifest(self.snapshot),
        }

    def _manifest_config(self, kind: str):
        config = {
            "kind": kind,
            "protocol_version": self.protocol_version,
            "experiment": asdict(self.experiment),
            "metric": asdict(self.metric),
            "gate0": asdict(self.gate),
            "snapshot": str(self.snapshot),
            "snapshot_input": self._snapshot_input(),
            "learning_rates": list(self.gate.learning_rates),
        }
        if kind == "gate0_erasure_scan":
            config["state"] = asdict(self.state)
            config["reference_input"] = {
                "artifact": {"sha256": "a" * 64},
                "source_manifest": {"sha256": "b" * 64},
            }
        return config

    def _write_scan_manifest(self, scan: Path, kind: str, *, config=None) -> None:
        freeze_manifest(config or self._manifest_config(kind), scan.parent / "manifest.json")

    def _write_erasure(self, outcomes) -> None:
        branches = []
        for rate, outcome in zip(self.gate.learning_rates, outcomes):
            provenance = {
                "schema_version": 1,
                "kind": "gate0_erasure",
                "controls": {
                    "experiment": asdict(self.experiment),
                    "metric": asdict(self.metric),
                    "gate": asdict(self.gate),
                    "thresholds": asdict(self.state),
                    "learning_rate": rate,
                },
                "sources": {
                    "snapshot_sha256": self._snapshot_input()["artifact"]["sha256"],
                    "reference_sha256": "a" * 64,
                },
            }
            provenance["cell_sha256"] = content_hash(provenance)
            branch = {
                    "learning_rate": rate,
                    "outcome": outcome,
                    "erased": outcome == "erased",
                    "sustained_entry_step": 1_000 if outcome == "erased" else None,
                    "provenance": provenance,
                }
            branch["result_sha256"] = content_hash(branch)
            branches.append(branch)
            cell = self.erasure.parent / f"eta_{rate:.12g}".replace(".", "p") / "result.json"
            cell.parent.mkdir(parents=True, exist_ok=True)
            cell.write_text(json.dumps(branch))
        brackets = [
            {
                "lower_learning_rate": lower["learning_rate"],
                "upper_learning_rate": upper["learning_rate"],
            }
            for lower, upper in zip(branches, branches[1:])
            if lower["outcome"] == "retained" and upper["outcome"] == "erased"
        ]
        self.erasure.write_text(
            json.dumps(
                {
                    "complete": True,
                    "requested_learning_rates": list(self.gate.learning_rates),
                    "strict_adjacent_brackets": brackets,
                    "branches": branches,
                }
            )
        )

    def _measurement(self, rate: float, *, certified: bool = True):
        residual = 0.001 if certified else 0.02
        provenance = {
            "schema_version": 1,
            "kind": "gate0_local_stability",
            "controls": {
                "experiment": asdict(self.experiment),
                "metric": asdict(self.metric),
                "gate": asdict(self.gate),
                "learning_rate": rate,
                "augmented_tolerance": 1e-3,
                "augmented_max_iterations": 100,
            },
            "sources": {
                "snapshot_sha256": self._snapshot_input()["artifact"]["sha256"],
            },
        }
        provenance["cell_sha256"] = content_hash(provenance)
        result = {
            "learning_rate": rate,
            "augmented_spectral_radius": 0.9 + rate,
            "augmented_max_relative_residual": residual,
            "augmented_certified": certified,
            "augmented_eigenvalues": [
                {
                    "real": 0.9,
                    "imag": 0.0,
                    "magnitude": 0.9 + rate,
                    "relative_residual": residual,
                }
                for _ in range(self.gate.augmented_eigenvalue_count)
            ],
            "stream_unchanged": True,
            "provenance": provenance,
        }
        result["result_sha256"] = content_hash(result)
        return result

    def _write_local(self, *, uncertified_rates=()) -> None:
        uncertified_rates = tuple(uncertified_rates)
        measurements = [
            self._measurement(rate, certified=rate not in uncertified_rates)
            for rate in self.gate.learning_rates
        ]
        for measurement in measurements:
            rate = measurement["learning_rate"]
            cell = self.local.parent / f"eta_{rate:.12g}".replace(".", "p") / "local_stability.json"
            cell.parent.mkdir(parents=True, exist_ok=True)
            cell.write_text(json.dumps(measurement))
        self.local.write_text(
            json.dumps(
                {
                    "complete": True,
                    "requested_learning_rates": list(self.gate.learning_rates),
                    "all_augmented_eigenpairs_certified": not uncertified_rates,
                    "uncertified_learning_rates": list(uncertified_rates),
                    "measurements": measurements,
                }
            )
        )

    def _adjudicate(self, *, output=None):
        if output is not None:
            output = Path(output)
            freeze_manifest(
                {
                    "kind": "gate0_calibration_adjudication",
                    "protocol_version": self.protocol_version,
                    "experiment": asdict(self.experiment),
                    "gate0": asdict(self.gate),
                    "positive_control_input": {
                        "artifact": bind_file(self.positive),
                        "source_manifest": bind_nearest_manifest(self.positive),
                    },
                    "erasure_scan_input": {
                        "artifact": bind_file(self.erasure),
                        "source_manifest": bind_nearest_manifest(self.erasure),
                    },
                    "local_scan_input": {
                        "artifact": bind_file(self.local),
                        "source_manifest": bind_nearest_manifest(self.local),
                    },
                    "snapshot": str(self.snapshot),
                    "snapshot_input": self._snapshot_input(),
                },
                output.parent / "manifest.json",
            )
        return adjudicate_gate0_calibration(
            positive_control_path=self.positive,
            erasure_scan_path=self.erasure,
            local_scan_path=self.local,
            snapshot_path=self.snapshot,
            experiment=self.experiment,
            gate=self.gate,
            protocol_version=self.protocol_version,
            output_path=output,
        )

    def _replace_manifest_config(self, scan: Path, config) -> None:
        path = scan.parent / "manifest.json"
        payload = json.loads(path.read_text())
        payload["config"] = config
        payload["config_sha256"] = content_hash(config)
        path.write_text(json.dumps(payload))

    def test_pass_binds_inputs_and_selects_smallest_adjacent_bracket(self) -> None:
        output = self.root / "calibration.json"
        report = self._adjudicate(output=output)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["action"], "open_gate0")
        self.assertEqual(
            report["selected_empirical_bracket"],
            {"lower_learning_rate": 0.003, "upper_learning_rate": 0.006},
        )
        self.assertEqual(report["frozen_erasure_bracket"], report["selected_empirical_bracket"])
        self.assertEqual(report["frozen_learning_rates"], list(self.gate.learning_rates))
        self.assertEqual(report, json.loads(output.read_text()))
        self.assertEqual(
            report["inputs"]["calibration_snapshot"]["artifact"]["checks"]["config"],
            "matched",
        )

        bound = bind_passed_gate0_calibration(
            output,
            expected_gate=self.gate,
            expected_protocol_version=self.protocol_version,
        )
        self.assertEqual(bound["status"], "pass")
        self.assertEqual(bound["protocol_version"], self.protocol_version)
        self.assertEqual(bound["frozen_erasure_bracket"], report["frozen_erasure_bracket"])
        self.assertEqual(
            bound["calibration_snapshot_sha256"],
            report["inputs"]["calibration_snapshot"]["artifact"]["sha256"],
        )

    def test_valid_negative_evidence_emits_stop(self) -> None:
        self._write_erasure(("retained", "retained", "retained", "retained", "retained"))
        report = self._adjudicate()
        self.assertEqual(report["status"], "stop")
        self.assertEqual(report["action"], "stop_before_gate0")
        self.assertIsNone(report["selected_empirical_bracket"])
        self.assertIsNone(report["frozen_erasure_bracket"])
        self.assertFalse(report["checks"]["strict_empirical_bracket_found"])

    def test_erasure_precheck_stops_without_requiring_local_scan(self) -> None:
        self._write_erasure(("retained", "unresolved", "unresolved", "erased", "unresolved"))
        report = adjudicate_gate0_erasure_precheck(
            positive_control_path=self.positive,
            erasure_scan_path=self.erasure,
            snapshot_path=self.snapshot,
            experiment=self.experiment,
            gate=self.gate,
            protocol_version=self.protocol_version,
        )
        self.assertEqual(report["status"], "stop")
        self.assertEqual(report["action"], "stop_before_gate0")
        self.assertFalse(report["checks"]["strict_empirical_bracket_found"])
        self.assertNotIn("local_stability_scan", report["inputs"])

    def test_uncertified_local_cell_stops_without_discarding_empirical_summary(self) -> None:
        self._write_local(uncertified_rates=(0.025,))
        report = self._adjudicate()
        self.assertEqual(report["status"], "stop")
        self.assertTrue(report["checks"]["strict_empirical_bracket_found"])
        self.assertFalse(report["checks"]["all_local_cells_certified"])
        self.assertIsNotNone(report["selected_empirical_bracket"])
        self.assertIsNone(report["frozen_erasure_bracket"])

    def test_failed_positive_control_stops(self) -> None:
        payload = json.loads(self.positive.read_text())
        payload["rows"][0]["nontrivial_multiplier"] = 0.123
        self.positive.write_text(json.dumps(payload))
        report = self._adjudicate()
        self.assertEqual(report["status"], "stop")
        self.assertFalse(report["checks"]["positive_control_passed"])
        self.assertIn("positive control", report["reasons"][0])

    def test_exact_manifest_controls_and_snapshot_binding_are_required(self) -> None:
        cases = []
        wrong_batch = self._manifest_config("gate0_erasure_scan")
        wrong_batch["experiment"]["batch_size"] = 512
        cases.append((wrong_batch, "batch size"))

        wrong_grid = self._manifest_config("gate0_erasure_scan")
        wrong_grid["learning_rates"][-1] = 0.1
        cases.append((wrong_grid, "exact Gate 0 grid"))

        wrong_snapshot = self._manifest_config("gate0_erasure_scan")
        wrong_snapshot["snapshot_input"]["artifact"]["sha256"] = "0" * 64
        cases.append((wrong_snapshot, "manifest snapshot"))

        original = json.loads((self.erasure.parent / "manifest.json").read_text())
        for config, message in cases:
            with self.subTest(message=message):
                (self.erasure.parent / "manifest.json").write_text(json.dumps(original))
                self._replace_manifest_config(self.erasure, config)
                with self.assertRaisesRegex(ValueError, message):
                    self._adjudicate()

    def test_positive_control_requires_its_frozen_sibling_manifest(self) -> None:
        manifest = self.positive.parent / "manifest.json"
        original = json.loads(manifest.read_text())
        cases = [
            ({**original["config"], "kind": "unregistered_control"}, "positive_control"),
            ({**original["config"], "protocol_version": "wrong"}, "protocol version"),
            ({**original["config"], "learning_rates": [0.5, 1.0, 1.5]}, "learning-rate grid"),
        ]
        for config, message in cases:
            with self.subTest(message=message):
                payload = dict(original)
                payload["config"] = config
                payload["config_sha256"] = content_hash(config)
                manifest.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, message):
                    self._adjudicate()

    def test_scan_claims_are_rederived_and_incomplete_inputs_are_rejected(self) -> None:
        payload = json.loads(self.erasure.read_text())
        payload["strict_adjacent_brackets"] = []
        self.erasure.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self._adjudicate()

        self._write_erasure(("retained", "erased", "retained", "erased", "retained"))
        payload = json.loads(self.local.read_text())
        payload["complete"] = False
        self.local.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self._adjudicate()

    def test_passed_binding_rejects_stop_and_tampered_controls(self) -> None:
        output = self.root / "calibration.json"
        self._adjudicate(output=output)
        payload = json.loads(output.read_text())
        payload["frozen_erasure_bracket"] = {
            "lower_learning_rate": 0.003,
            "upper_learning_rate": 0.012,
        }
        output.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "not adjacent"):
            bind_passed_gate0_calibration(output)

        payload["status"] = "stop"
        payload["action"] = "stop_before_gate0"
        output.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "passed calibration"):
            bind_passed_gate0_calibration(output)

    def test_passed_binding_can_enforce_expected_protocol_and_gate(self) -> None:
        output = self.root / "calibration.json"
        self._adjudicate(output=output)
        with self.assertRaisesRegex(ValueError, "protocol version"):
            bind_passed_gate0_calibration(
                output,
                expected_protocol_version="wrong",
            )
        with self.assertRaisesRegex(ValueError, "calibration seed"):
            bind_passed_gate0_calibration(
                output,
                expected_gate=replace(self.gate, calibration_seed=101),
            )


if __name__ == "__main__":
    unittest.main()
