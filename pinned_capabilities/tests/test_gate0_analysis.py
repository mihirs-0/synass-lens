import math
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from pinned_capabilities.gate0_analysis import (
    BatchContrast,
    CellConclusion,
    CellEvidence,
    CellKey,
    ContrastConclusion,
    Direction,
    Gate0Action,
    Gate0Decision,
    ResidualObservation,
    ThresholdInterval,
    analyze_gate0,
    analyze_gate0_payload,
    bind_gate0_analysis_report,
    assess_batch_contrast,
    assess_cell,
    empirical_acquisition_boundary,
    empirical_erasure_boundary,
    gate0_analysis_to_dict,
    gate0_payload_source_paths,
    predicted_local_boundary,
    registered_exact_16x_contrasts,
)
from pinned_capabilities.config import Gate0Config
from pinned_capabilities.provenance import bind_file
from pinned_capabilities.manifest import content_hash, freeze_manifest


SEEDS = (0, 1, 2, 3, 4)
RATES = (0.003, 0.006, 0.012, 0.025, 0.05)


def residuals(
    contrast: BatchContrast,
    paired_log_effects,
):
    rows = []
    for seed, effect in zip(SEEDS, paired_log_effects):
        baseline = 0.07 * seed
        rows.extend(
            (
                ResidualObservation(
                    seed,
                    contrast.lower_batch_size,
                    contrast.direction,
                    baseline,
                    contrast.configuration,
                ),
                ResidualObservation(
                    seed,
                    contrast.upper_batch_size,
                    contrast.direction,
                    baseline + effect,
                    contrast.configuration,
                ),
            )
        )
    return rows


class IntervalAssessmentTests(unittest.TestCase):
    def test_intermediate_factor_is_explicitly_ambiguous(self) -> None:
        key = CellKey(0, 128, Direction.ERASURE)
        result = assess_cell(
            CellEvidence(
                key,
                ThresholdInterval.point(1.75),
                ThresholdInterval.point(1.0),
                True,
            )
        )
        self.assertEqual(result.conclusion, CellConclusion.AMBIGUOUS)
        self.assertEqual(result.minimum_mismatch_factor, 1.75)
        self.assertEqual(result.maximum_mismatch_factor, 1.75)

    def test_censored_interval_never_enters_the_registered_center_estimator(self) -> None:
        key = CellKey(0, 128, Direction.ERASURE)
        result = assess_cell(
            CellEvidence(
                key,
                ThresholdInterval.right_censored(2.1),
                ThresholdInterval.point(1.0),
                True,
            )
        )
        self.assertEqual(result.conclusion, CellConclusion.AMBIGUOUS)
        self.assertGreater(result.minimum_mismatch_factor, 2.0)
        self.assertTrue(math.isinf(result.maximum_mismatch_factor))

        unresolved = assess_cell(
            CellEvidence(
                key,
                ThresholdInterval.left_censored(1.2),
                ThresholdInterval.point(1.0),
                True,
            )
        )
        self.assertEqual(unresolved.conclusion, CellConclusion.AMBIGUOUS)

    def test_identical_coarse_intervals_support_reduction_by_registered_centers(self) -> None:
        key = CellKey(0, 128, Direction.ERASURE)
        result = assess_cell(
            CellEvidence(
                key,
                ThresholdInterval.observed(0.006, 0.012),
                ThresholdInterval.observed(0.006, 0.012),
                True,
            )
        )
        self.assertEqual(result.conclusion, CellConclusion.REDUCTION)
        self.assertEqual(result.minimum_mismatch_factor, 1.0)
        self.assertEqual(result.maximum_mismatch_factor, 2.0)

    def test_missing_and_uncertified_predictors_are_explicit(self) -> None:
        key = CellKey(0, 128, Direction.ACQUISITION)
        missing = assess_cell(
            CellEvidence(
                key,
                ThresholdInterval.point(1.0),
                ThresholdInterval.missing(),
                True,
            )
        )
        uncertified = assess_cell(
            CellEvidence(
                key,
                ThresholdInterval.point(1.0),
                ThresholdInterval.point(1.0),
                False,
            )
        )
        self.assertEqual(missing.conclusion, CellConclusion.MISSING_PREDICTOR)
        self.assertEqual(uncertified.conclusion, CellConclusion.UNCERTIFIED_PREDICTOR)


class PairedBatchEffectTests(unittest.TestCase):
    def test_registered_ladder_yields_only_the_two_exact_16x_pairs(self) -> None:
        contrasts = registered_exact_16x_contrasts(directions=(Direction.ERASURE,))
        self.assertEqual(
            [(item.lower_batch_size, item.upper_batch_size) for item in contrasts],
            [(32, 512), (128, 2_048)],
        )
        with self.assertRaisesRegex(ValueError, "exactly 16x"):
            BatchContrast(32, 2_048, Direction.ERASURE)

    def test_cluster_bootstrap_is_paired_within_seed(self) -> None:
        contrast = BatchContrast(32, 512, Direction.ERASURE)
        effect = math.log(1.10)
        assessment = assess_batch_contrast(
            contrast,
            residuals(contrast, [effect] * 5),
        )
        self.assertEqual(assessment.paired_seed_count, 5)
        self.assertAlmostEqual(assessment.mean_log_effect, effect)
        self.assertAlmostEqual(assessment.confidence_interval[0], effect)
        self.assertAlmostEqual(assessment.confidence_interval[1], effect)
        self.assertEqual(assessment.conclusion, ContrastConclusion.REDUCTION)

    def test_over_25_percent_with_interval_excluding_zero_is_non_reduction(self) -> None:
        contrast = BatchContrast(128, 2_048, Direction.ERASURE)
        effects = [math.log(value) for value in (1.28, 1.31, 1.34, 1.37, 1.40)]
        assessment = assess_batch_contrast(contrast, residuals(contrast, effects))
        self.assertEqual(assessment.conclusion, ContrastConclusion.NON_REDUCTION)
        self.assertGreater(assessment.mean_log_effect, math.log(1.25))
        self.assertGreater(assessment.confidence_interval[0], 0.0)

    def test_missing_seed_makes_contrast_incomplete(self) -> None:
        contrast = BatchContrast(32, 512, Direction.ACQUISITION)
        assessment = assess_batch_contrast(
            contrast,
            residuals(contrast, [0.0] * 4),
        )
        self.assertEqual(assessment.conclusion, ContrastConclusion.INCOMPLETE)
        self.assertEqual(assessment.missing_seeds, (4,))
        self.assertIsNone(assessment.confidence_interval)


class GateDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.erasure = CellKey(0, 128, Direction.ERASURE)
        self.acquisition = CellKey(0, 128, Direction.ACQUISITION)
        self.registered_cells = tuple(
            CellKey(seed, batch, direction)
            for direction in (Direction.ERASURE, Direction.ACQUISITION)
            for batch in (32, 128, 512, 2_048)
            for seed in SEEDS
        )
        self.contrasts = registered_exact_16x_contrasts()

    @staticmethod
    def matching(key: CellKey) -> CellEvidence:
        return CellEvidence(
            key,
            ThresholdInterval.observed(0.95, 1.05),
            ThresholdInterval.observed(0.95, 1.05),
            True,
        )

    def reduction_residuals(self):
        rows = []
        for contrast in self.contrasts:
            rows.extend(residuals(contrast, [math.log(1.05)] * 5))
        return rows

    def test_early_continue_survives_missing_other_evidence(self) -> None:
        decision = analyze_gate0(
            registered_cells=(self.erasure, self.acquisition),
            cell_evidence=(
                CellEvidence(
                    self.erasure,
                    ThresholdInterval.point(2.2),
                    ThresholdInterval.point(1.0),
                    True,
                ),
            ),
            registered_contrasts=self.contrasts,
            residual_observations=(),
        )
        self.assertEqual(decision.decision, Gate0Decision.CONTINUE)
        self.assertEqual(decision.action, Gate0Action.CONTINUE)

    def test_full_kill_requires_both_directions_and_equivalence_evidence(self) -> None:
        complete = analyze_gate0(
            registered_cells=self.registered_cells,
            cell_evidence=tuple(self.matching(key) for key in self.registered_cells),
            registered_contrasts=self.contrasts,
            residual_observations=self.reduction_residuals(),
        )
        self.assertEqual(complete.decision, Gate0Decision.KILL)
        self.assertEqual(complete.action, Gate0Action.STOP)

        self.assertTrue(
            all(item.conclusion is CellConclusion.REDUCTION for item in complete.cell_assessments)
        )
        self.assertTrue(
            all(
                item.conclusion is ContrastConclusion.REDUCTION
                for item in complete.contrast_assessments
            )
        )

    def test_ambiguous_1p5_to_2x_region_has_total_inconclusive_decision(self) -> None:
        ambiguous = CellEvidence(
            self.erasure,
            ThresholdInterval.point(1.75),
            ThresholdInterval.point(1.0),
            True,
        )
        result = analyze_gate0(
            registered_cells=self.registered_cells,
            cell_evidence=tuple(
                ambiguous if key == self.erasure else self.matching(key)
                for key in self.registered_cells
            ),
            registered_contrasts=self.contrasts,
            residual_observations=self.reduction_residuals(),
        )
        self.assertEqual(result.decision, Gate0Decision.INCONCLUSIVE)
        self.assertEqual(result.action, Gate0Action.STOP)
        self.assertIn("ambiguous", " ".join(result.reasons))

    def test_partial_reduction_collects_more_evidence_instead_of_stopping(self) -> None:
        result = analyze_gate0(
            registered_cells=(self.erasure, self.acquisition),
            cell_evidence=(self.matching(self.erasure),),
            registered_contrasts=(),
            residual_observations=(),
        )
        self.assertEqual(result.decision, Gate0Decision.INCONCLUSIVE)
        self.assertEqual(result.action, Gate0Action.COLLECT)

    def test_missing_direction_or_contrast_cannot_kill(self) -> None:
        erasure_cells = tuple(
            key for key in self.registered_cells if key.direction is Direction.ERASURE
        )
        erasure_contrasts = tuple(
            item for item in self.contrasts if item.direction is Direction.ERASURE
        )
        erasure_residuals = []
        for contrast in erasure_contrasts:
            erasure_residuals.extend(residuals(contrast, [math.log(1.05)] * 5))
        missing_direction = analyze_gate0(
            registered_cells=erasure_cells,
            cell_evidence=tuple(self.matching(key) for key in erasure_cells),
            registered_contrasts=erasure_contrasts,
            residual_observations=erasure_residuals,
        )
        missing_contrast_seed = analyze_gate0(
            registered_cells=self.registered_cells,
            cell_evidence=tuple(self.matching(key) for key in self.registered_cells),
            registered_contrasts=self.contrasts,
            residual_observations=self.reduction_residuals()[:-1],
        )
        self.assertEqual(missing_direction.decision, Gate0Decision.INCONCLUSIVE)
        self.assertEqual(missing_contrast_seed.decision, Gate0Decision.INCONCLUSIVE)

    def test_residual_non_reduction_triggers_continue_when_cells_match(self) -> None:
        rows = residuals(
            self.contrasts[0],
            [math.log(value) for value in (1.30, 1.32, 1.34, 1.36, 1.38)],
        )
        rows.extend(residuals(self.contrasts[1], [0.0] * 5))
        for contrast in self.contrasts[2:]:
            rows.extend(residuals(contrast, [0.0] * 5))
        result = analyze_gate0(
            registered_cells=self.registered_cells,
            cell_evidence=tuple(self.matching(key) for key in self.registered_cells),
            registered_contrasts=self.contrasts,
            residual_observations=rows,
        )
        self.assertEqual(result.decision, Gate0Decision.CONTINUE)


class ScanConversionTests(unittest.TestCase):
    @staticmethod
    def _write_scan_manifest(
        directory: Path,
        *,
        kind: str,
        seed: int = 0,
        batch_size: int = 128,
        snapshot_sha: str = "1" * 64,
        reference_sha: str = "2" * 64,
        learning_rates=RATES,
    ) -> None:
        config = {
            "kind": kind,
            "experiment": {"seed": seed, "batch_size": batch_size},
            "metric": {},
            "gate0": {
                "augmented_eigenvalue_count": 3,
                "augmented_max_relative_residual": 0.01,
                "erase_horizon": 2_000,
            },
            "learning_rates": list(learning_rates),
            "snapshot_input": {
                "artifact": {"sha256": snapshot_sha},
                "source_manifest": {"sha256": "3" * 64},
            },
        }
        if kind != "gate0_local_stability_scan":
            config["state"] = {}
            config["reference_input"] = {
                "artifact": {"sha256": reference_sha},
                "source_manifest": {"sha256": "4" * 64},
            }
        freeze_manifest(config, directory / "manifest.json")

    def _write_scan_pair(
        self,
        root: Path,
        *,
        direction: str = "erasure",
        seed: int = 0,
        batch_size: int = 128,
        empirical_rates=RATES,
        local_rates=RATES,
        empirical_snapshot_sha: str = "1" * 64,
        local_snapshot_sha: str = "1" * 64,
        reference_sha: str = "2" * 64,
    ):
        empirical_dir = root / "empirical"
        local_dir = root / "local"
        empirical_dir.mkdir(parents=True)
        local_dir.mkdir(parents=True)
        empirical_path = empirical_dir / "scan.json"
        local_path = local_dir / "scan.json"
        if direction == "erasure":
            outcomes = ("retained", "retained", "erased", "erased", "erased")
            empirical_kind = "gate0_erasure_scan"
        else:
            outcomes = ("transitioned", "transitioned", "censored", "censored", "censored")
            empirical_kind = "gate0_acquisition_scan"
        empirical_branches = []
        for rate, outcome in zip(empirical_rates, outcomes):
            provenance = {
                "schema_version": 1,
                "kind": "gate0_erasure" if direction == "erasure" else "gate0_acquisition",
                "controls": {
                    "experiment": {"seed": seed, "batch_size": batch_size},
                    "metric": {},
                    "gate": {
                        "augmented_eigenvalue_count": 3,
                        "augmented_max_relative_residual": 0.01,
                        "erase_horizon": 2_000,
                    },
                    "thresholds": {},
                    "learning_rate": rate,
                },
                "sources": {
                    "snapshot_sha256": empirical_snapshot_sha,
                    "reference_sha256": reference_sha,
                },
            }
            provenance["cell_sha256"] = content_hash(provenance)
            if direction == "erasure":
                branch = {
                    "learning_rate": rate,
                    "outcome": outcome,
                    "erased": outcome == "erased",
                    "sustained_entry_step": 1_000 if outcome == "erased" else None,
                    "provenance": provenance,
                }
            else:
                branch = {
                    "learning_rate": rate,
                    "outcome": outcome,
                    "transition_step": 1_000 if outcome == "transitioned" else None,
                    "provenance": provenance,
                }
            branch["result_sha256"] = content_hash(branch)
            empirical_branches.append(branch)
            child = empirical_dir / f"eta_{rate:.12g}".replace(".", "p") / "result.json"
            child.parent.mkdir(parents=True, exist_ok=True)
            child.write_text(json.dumps(branch))
        empirical_payload = {
            "complete": True,
            "requested_learning_rates": list(empirical_rates),
            "branches": empirical_branches,
        }
        if direction == "erasure":
            empirical_payload["strict_adjacent_brackets"] = [
                {
                    "lower_learning_rate": lower["learning_rate"],
                    "upper_learning_rate": upper["learning_rate"],
                }
                for lower, upper in zip(empirical_branches, empirical_branches[1:])
                if lower["outcome"] == "retained" and upper["outcome"] == "erased"
            ]
        empirical_path.write_text(json.dumps(empirical_payload))

        local_measurements = []
        for rate, radius in zip(local_rates, (0.8, 0.9, 1.1, 1.2, 1.3)):
            provenance = {
                "schema_version": 1,
                "kind": "gate0_local_stability",
                "controls": {
                    "experiment": {"seed": seed, "batch_size": batch_size},
                    "metric": {},
                    "gate": {
                        "augmented_eigenvalue_count": 3,
                        "augmented_max_relative_residual": 0.01,
                        "erase_horizon": 2_000,
                    },
                    "learning_rate": rate,
                    "augmented_tolerance": 1e-3,
                    "augmented_max_iterations": 100,
                },
                "sources": {"snapshot_sha256": local_snapshot_sha},
            }
            provenance["cell_sha256"] = content_hash(provenance)
            measurement = {
                "learning_rate": rate,
                "augmented_spectral_radius": radius,
                "augmented_max_relative_residual": 0.001,
                "augmented_certified": True,
                "augmented_eigenvalues": [
                    {
                        "real": radius,
                        "imag": 0.0,
                        "magnitude": radius,
                        "relative_residual": 0.001,
                    }
                    for _ in range(3)
                ],
                "stream_unchanged": True,
                "provenance": provenance,
            }
            measurement["result_sha256"] = content_hash(measurement)
            local_measurements.append(measurement)
            child = local_dir / f"eta_{rate:.12g}".replace(".", "p") / "local_stability.json"
            child.parent.mkdir(parents=True, exist_ok=True)
            child.write_text(json.dumps(measurement))
        local_path.write_text(
            json.dumps(
                {
                    "complete": True,
                    "requested_learning_rates": list(local_rates),
                    "all_augmented_eigenpairs_certified": True,
                    "uncertified_learning_rates": [],
                    "measurements": local_measurements,
                }
            )
        )
        self._write_scan_manifest(
            empirical_dir,
            kind=empirical_kind,
            seed=seed,
            batch_size=batch_size,
            snapshot_sha=empirical_snapshot_sha,
            reference_sha=reference_sha,
        )
        self._write_scan_manifest(
            local_dir,
            kind="gate0_local_stability_scan",
            seed=seed,
            batch_size=batch_size,
            snapshot_sha=local_snapshot_sha,
        )
        return empirical_path, local_path

    def test_empirical_scan_conversions_are_strict_and_censored(self) -> None:
        erasure = {
            "complete": True,
            "branches": [
                {"learning_rate": 0.01, "outcome": "retained"},
                {"learning_rate": 0.02, "outcome": "erased"},
            ],
        }
        self.assertEqual(
            empirical_erasure_boundary(erasure),
            ThresholdInterval.observed(0.01, 0.02),
        )
        erasure["branches"][1]["outcome"] = "diverged"
        self.assertTrue(empirical_erasure_boundary(erasure).is_missing)

        acquisition = {
            "complete": True,
            "branches": [
                {"learning_rate": 0.01, "outcome": "transitioned"},
                {"learning_rate": 0.02, "outcome": "censored"},
            ],
        }
        self.assertEqual(
            empirical_acquisition_boundary(acquisition),
            ThresholdInterval.observed(0.01, 0.02),
        )

    def test_local_unit_circle_conversion_is_directional_and_certified(self) -> None:
        scan = {
            "complete": True,
            "measurements": [
                {
                    "learning_rate": 0.01,
                    "augmented_spectral_radius": 0.9,
                    "augmented_certified": True,
                },
                {
                    "learning_rate": 0.02,
                    "augmented_spectral_radius": 1.1,
                    "augmented_certified": True,
                },
            ],
        }
        interval, certified = predicted_local_boundary(scan, Direction.ERASURE)
        self.assertTrue(certified)
        self.assertEqual(interval, ThresholdInterval.observed(0.01, 0.02))
        acquisition_interval, _ = predicted_local_boundary(scan, Direction.ACQUISITION)
        self.assertTrue(acquisition_interval.is_missing)
        scan["measurements"][1]["augmented_certified"] = False
        _, certified = predicted_local_boundary(scan, Direction.ERASURE)
        self.assertFalse(certified)

    def test_json_evidence_schema_executes_and_serializes_total_action(self) -> None:
        payload = {
            "cells": [
                {
                    "seed": 0,
                    "batch_size": 128,
                    "direction": "erasure",
                    "empirical_boundary": {
                        "kind": "observed",
                        "lower": 2.2,
                        "upper": 2.2,
                    },
                    "predicted_boundary": {
                        "kind": "observed",
                        "lower": 1.0,
                        "upper": 1.0,
                    },
                    "predictor_certified": True,
                }
            ],
            "contrasts": [],
            "residual_observations": [],
        }
        report = gate0_analysis_to_dict(analyze_gate0_payload(payload))
        self.assertEqual(report["decision"], "continue")
        self.assertEqual(report["action"], "continue")

    def test_official_payload_derives_boundaries_and_residual_from_raw_scans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empirical_path, local_path = self._write_scan_pair(root)
            payload = {
                "cells": [
                    {
                        "seed": 0,
                        "batch_size": 128,
                        "direction": "erasure",
                        "empirical_scan": str(empirical_path),
                        "local_scan": str(local_path),
                    }
                ]
            }
            self.assertEqual(
                gate0_payload_source_paths(payload), (empirical_path, local_path)
            )
            report = analyze_gate0_payload(payload, require_scan_sources=True)
            cell = next(
                item
                for item in report.cell_assessments
                if item.key
                == CellKey(seed=0, batch_size=128, direction=Direction.ERASURE)
            )
            self.assertEqual(cell.conclusion, CellConclusion.REDUCTION)
            self.assertEqual(report.decision, Gate0Decision.INCONCLUSIVE)
            self.assertEqual(report.action, Gate0Action.COLLECT)

    def test_official_payload_rejects_the_old_mismatched_grids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empirical_path, local_path = self._write_scan_pair(
                Path(directory),
                empirical_rates=(0.01, 0.02),
                local_rates=(0.001, 0.002),
            )
            payload = {
                "cells": [{
                    "seed": 0,
                    "batch_size": 128,
                    "direction": "erasure",
                    "empirical_scan": str(empirical_path),
                    "local_scan": str(local_path),
                }]
            }
            with self.assertRaisesRegex(ValueError, "exact frozen Gate 0 grid"):
                analyze_gate0_payload(payload, require_scan_sources=True)

    def test_official_payload_rejects_direction_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empirical_path, local_path = self._write_scan_pair(Path(directory))
            payload = {
                "cells": [{
                    "seed": 0,
                    "batch_size": 128,
                    "direction": "acquisition",
                    "empirical_scan": str(empirical_path),
                    "local_scan": str(local_path),
                }]
            }
            with self.assertRaisesRegex(ValueError, "manifest kind"):
                analyze_gate0_payload(payload, require_scan_sources=True)

    def test_official_payload_rejects_reused_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empirical_path, local_path = self._write_scan_pair(Path(directory))
            row = {
                "seed": 0,
                "batch_size": 128,
                "direction": "erasure",
                "empirical_scan": str(empirical_path),
                "local_scan": str(local_path),
            }
            payload = {"cells": [row, dict(row)]}
            with self.assertRaisesRegex(ValueError, "assigned more than once"):
                gate0_payload_source_paths(payload)
            with self.assertRaisesRegex(ValueError, "assigned more than once"):
                analyze_gate0_payload(payload, require_scan_sources=True)

    def test_official_payload_rejects_manifest_and_snapshot_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empirical_path, local_path = self._write_scan_pair(
                root,
                local_snapshot_sha="5" * 64,
            )
            payload = {
                "cells": [{
                    "seed": 0,
                    "batch_size": 128,
                    "direction": "erasure",
                    "empirical_scan": str(empirical_path),
                    "local_scan": str(local_path),
                }]
            }
            with self.assertRaisesRegex(ValueError, "same snapshot artifact"):
                analyze_gate0_payload(payload, require_scan_sources=True)

            local_manifest = local_path.parent / "manifest.json"
            manifest = json.loads(local_manifest.read_text())
            manifest["config"]["experiment"]["seed"] = 4
            local_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "configuration digest mismatch"):
                analyze_gate0_payload(payload, require_scan_sources=True)

    def test_official_payload_rejects_unique_observation_and_manifest_identity_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empirical_path, local_path = self._write_scan_pair(root)
            payload = {
                "cells": [{
                    "seed": 0,
                    "batch_size": 128,
                    "direction": "erasure",
                    "empirical_scan": str(empirical_path),
                    "local_scan": str(local_path),
                }]
            }
            empirical = json.loads(empirical_path.read_text())
            empirical["branches"][-1]["learning_rate"] = 0.025
            empirical_path.write_text(json.dumps(empirical))
            with self.assertRaisesRegex(ValueError, "unique learning rates"):
                analyze_gate0_payload(payload, require_scan_sources=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empirical_path, local_path = self._write_scan_pair(root, seed=1)
            payload = {
                "cells": [{
                    "seed": 0,
                    "batch_size": 128,
                    "direction": "erasure",
                    "empirical_scan": str(empirical_path),
                    "local_scan": str(local_path),
                }]
            }
            with self.assertRaisesRegex(ValueError, "manifest seed"):
                analyze_gate0_payload(payload, require_scan_sources=True)

    def test_official_payload_rejects_aggregate_child_and_seal_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empirical_path, local_path = self._write_scan_pair(Path(directory))
            payload = {
                "cells": [{
                    "seed": 0,
                    "batch_size": 128,
                    "direction": "erasure",
                    "empirical_scan": str(empirical_path),
                    "local_scan": str(local_path),
                }]
            }
            empirical = json.loads(empirical_path.read_text())
            empirical["branches"][0]["outcome"] = "erased"
            empirical_path.write_text(json.dumps(empirical))
            with self.assertRaisesRegex(ValueError, "aggregate disagrees with child"):
                analyze_gate0_payload(payload, require_scan_sources=True)

        with tempfile.TemporaryDirectory() as directory:
            empirical_path, local_path = self._write_scan_pair(Path(directory))
            payload = {
                "cells": [{
                    "seed": 0,
                    "batch_size": 128,
                    "direction": "erasure",
                    "empirical_scan": str(empirical_path),
                    "local_scan": str(local_path),
                }]
            }
            local = json.loads(local_path.read_text())
            row = local["measurements"][0]
            row["augmented_spectral_radius"] = 9.0
            child = local_path.parent / "eta_0p003" / "local_stability.json"
            child.write_text(json.dumps(row))
            local_path.write_text(json.dumps(local))
            with self.assertRaisesRegex(ValueError, "tampered result_sha256"):
                analyze_gate0_payload(payload, require_scan_sources=True)

    def test_official_payload_requires_one_common_reference_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_empirical, first_local = self._write_scan_pair(root / "first")
            second_empirical, second_local = self._write_scan_pair(
                root / "second", seed=1, reference_sha="6" * 64
            )
            payload = {
                "cells": [
                    {
                        "seed": 0,
                        "batch_size": 128,
                        "direction": "erasure",
                        "empirical_scan": str(first_empirical),
                        "local_scan": str(first_local),
                    },
                    {
                        "seed": 1,
                        "batch_size": 128,
                        "direction": "erasure",
                        "empirical_scan": str(second_empirical),
                        "local_scan": str(second_local),
                    },
                ]
            }
            with self.assertRaisesRegex(ValueError, "same reference"):
                analyze_gate0_payload(payload, require_scan_sources=True)

    def test_payload_rejects_any_finalize_control(self) -> None:
        for value in (False, True, "false"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "completion is derived"):
                    analyze_gate0_payload({"cells": [], "finalize": value})

    def test_report_binding_rederives_and_rejects_forged_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.json"
            evidence_path.write_text(json.dumps({"cells": []}))
            gate = json.loads(json.dumps(asdict(Gate0Config())))
            freeze_manifest(
                {
                    "kind": "gate0_decision_analysis",
                    "protocol_version": "test-1.4.0",
                    "gate0": gate,
                    "evidence_input": bind_file(evidence_path),
                    "scan_inputs": [],
                    "registry_complete": False,
                },
                root / "manifest.json",
            )
            analysis = analyze_gate0_payload(
                {"cells": []},
                registered_seeds=SEEDS,
                registered_batch_sizes=(32, 128, 512, 2_048),
                expected_learning_rates=RATES,
            )
            valid = {
                "schema_version": 1,
                "kind": "gate0_decision_analysis",
                "protocol_version": "test-1.4.0",
                "registry_complete": False,
                **gate0_analysis_to_dict(analysis),
            }
            report_path = root / "gate0_analysis.json"
            report_path.write_text(json.dumps(valid))
            bound = bind_gate0_analysis_report(
                report_path,
                expected_protocol_version="test-1.4.0",
                expected_gate0_config=gate,
            )
            self.assertEqual(bound["action"], "collect")

            forged = dict(valid)
            forged["decision"] = "continue"
            forged["action"] = "continue"
            report_path.write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, "fresh analysis"):
                bind_gate0_analysis_report(
                    report_path,
                    expected_protocol_version="test-1.4.0",
                    expected_gate0_config=gate,
                )


if __name__ == "__main__":
    unittest.main()
    bind_gate0_analysis_report,
