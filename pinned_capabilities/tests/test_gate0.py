import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pinned_capabilities.gate0 import deep_linear_control
from pinned_capabilities.local_stability import (
    AugmentedAdamWLinearization,
    capability_preconditioned_curvature,
    largest_preconditioned_curvature,
)
from pinned_capabilities.local_measurement import measure_checkpoint_local_stability_scan


class _ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _RecordingExecutor:
    instances = []

    def __init__(self, *, max_workers, mp_context):
        self.max_workers = max_workers
        self.start_method = mp_context.get_start_method()
        self.submissions = []
        type(self).instances.append(self)

    def submit(self, function, *args):
        self.submissions.append((function, args))
        return _ImmediateFuture(function(*args))

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_controls = (wait, cancel_futures)


class DeepLinearControlTests(unittest.TestCase):
    def test_known_stability_boundary(self) -> None:
        result = deep_linear_control((0.9, 1.0, 1.1), target=1.0)
        self.assertEqual(result["analytic_critical_learning_rate"], 1.0)
        rows = result["rows"]
        self.assertTrue(rows[0]["locally_stable"])
        self.assertFalse(rows[1]["locally_stable"])
        self.assertFalse(rows[2]["locally_stable"])
        self.assertAlmostEqual(rows[0]["nontrivial_multiplier"], -0.8, places=10)

    def test_augmented_adam_control_exercises_shared_numerical_pipeline(self) -> None:
        result = deep_linear_control((0.9, 1.0, 1.1), target=1.0)
        control = result["augmented_adam_control"]

        self.assertEqual(
            control["system"], "two_parameter_diagonal_quadratic_adamw"
        )
        self.assertEqual(control["curvature_diagonal"], [2.0, 5.0])
        self.assertEqual(control["augmented_dimension"], 6)
        self.assertEqual(control["dominant_eigenvalue_count"], 3)
        self.assertEqual(len(control["dense_hvp_jacobian"]), 6)
        self.assertTrue(
            all(len(row) == 6 for row in control["dense_hvp_jacobian"])
        )
        self.assertEqual(len(control["centered_finite_difference_jacobian"]), 6)
        self.assertLess(control["jacobian_max_absolute_error"], 2e-7)
        self.assertLess(control["jacobian_relative_frobenius_error"], 2e-7)
        self.assertEqual(len(control["dominant_eigenvalues"]), 3)
        self.assertEqual(len(control["eigenpair_relative_residuals"]), 3)
        self.assertLess(control["maximum_eigenpair_relative_residual"], 1e-10)

        # The complete result is safe to write into the positive-control JSON.
        json.dumps(result, allow_nan=False)


class CurvatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.y = torch.nn.Parameter(torch.tensor(-0.3, dtype=torch.float64))
        self.optimizer = torch.optim.AdamW([self.x, self.y], lr=0.1, eps=1e-8)
        beta2 = self.optimizer.param_groups[0]["betas"][1]
        for parameter in (self.x, self.y):
            self.optimizer.state[parameter] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.full_like(parameter, 1.0 - beta2),
            }

    def loss(self) -> torch.Tensor:
        return 0.5 * (2.0 * self.x.square() + 5.0 * self.y.square())

    def test_capability_direction_and_largest_mode(self) -> None:
        score = self.x
        observed = capability_preconditioned_curvature(
            self.loss(), score, (self.x, self.y), self.optimizer
        )
        expected_scale = 1.0 / (1.0 + 1e-8)
        self.assertTrue(math.isclose(observed.item(), 2.0 * expected_scale, rel_tol=1e-7))
        largest = largest_preconditioned_curvature(
            self.loss, (self.x, self.y), self.optimizer, iterations=40, seed=3
        )
        self.assertTrue(math.isclose(largest.item(), 5.0 * expected_scale, rel_tol=1e-6))

    def test_augmented_adam_jacobian_matches_finite_difference(self) -> None:
        linearization = AugmentedAdamWLinearization(
            self.loss(), (self.x, self.y), self.optimizer
        )
        observed = linearization.dense_jacobian()
        theta = torch.tensor([self.x.item(), self.y.item()], dtype=torch.float64)
        moment = torch.stack(
            [self.optimizer.state[p]["exp_avg"] for p in (self.x, self.y)]
        )
        second = torch.stack(
            [self.optimizer.state[p]["exp_avg_sq"] for p in (self.x, self.y)]
        )
        base = torch.cat((theta, moment, second))
        beta1, beta2 = self.optimizer.param_groups[0]["betas"]
        learning_rate = self.optimizer.param_groups[0]["lr"]
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        eps = self.optimizer.param_groups[0]["eps"]

        def state_map(value):
            current_theta, current_m, current_v = value[:2], value[2:4], value[4:]
            gradient = torch.tensor([2.0, 5.0], dtype=torch.float64) * current_theta
            next_m = beta1 * current_m + (1 - beta1) * gradient
            next_v = beta2 * current_v + (1 - beta2) * gradient.square()
            m_hat = next_m / (1 - beta1**2)
            v_hat = next_v / (1 - beta2**2)
            next_theta = (1 - learning_rate * weight_decay) * current_theta
            next_theta = next_theta - learning_rate * m_hat / (v_hat.sqrt() + eps)
            return torch.cat((next_theta, next_m, next_v))

        epsilon = 1e-6
        numerical = []
        for index in range(6):
            direction = torch.zeros(6, dtype=torch.float64)
            direction[index] = epsilon
            numerical.append(((state_map(base + direction) - state_map(base - direction)) / (2 * epsilon)).numpy())
        numerical = torch.from_numpy(np.column_stack(numerical))
        torch.testing.assert_close(torch.from_numpy(observed), numerical, rtol=2e-5, atol=2e-7)
        values, vectors = linearization.dominant_eigenpairs(count=2)
        residuals = linearization.eigenpair_residuals(values, vectors)
        self.assertTrue(np.all(residuals < 1e-10))


class LocalStabilityScanTests(unittest.TestCase):
    @staticmethod
    def fake_measure(*args, **kwargs):
        rate = kwargs["learning_rate"]
        result = {
            "learning_rate": rate,
            "augmented_certified": rate == 0.01,
            "augmented_spectral_radius": 1.0 + rate,
        }
        output = Path(args[4])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result))
        return result

    def test_scan_resumes_completed_rate_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "pinned_capabilities.local_measurement.measure_checkpoint_local_stability",
            side_effect=self.fake_measure,
        ) as measure, patch(
            "pinned_capabilities.local_measurement.sha256_file", return_value="a" * 64
        ):
            summary = measure_checkpoint_local_stability_scan(
                object(), object(), object(), Path("snapshot"), [0.01, 0.02], Path(directory)
            )
            self.assertEqual(measure.call_count, 2)
            self.assertFalse(summary["all_augmented_eigenpairs_certified"])
            self.assertEqual(summary["uncertified_learning_rates"], [0.02])
            with patch(
                "pinned_capabilities.local_measurement.measure_checkpoint_local_stability",
                side_effect=AssertionError("completed cells should not rerun"),
            ):
                resumed = measure_checkpoint_local_stability_scan(
                    object(), object(), object(), Path("snapshot"), [0.01, 0.02], Path(directory)
                )
            self.assertEqual(resumed, summary)

    def test_scan_cache_rejects_copied_tampered_and_stale_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "pinned_capabilities.local_measurement.measure_checkpoint_local_stability",
            side_effect=self.fake_measure,
        ):
            root = Path(directory)
            snapshot = root / "snapshot.pt"
            snapshot.write_bytes(b"source-v1")
            output = root / "scan"
            measure_checkpoint_local_stability_scan(
                object(), object(), object(), snapshot, [0.01, 0.02], output
            )

            first = json.loads(
                (output / "eta_0p01" / "local_stability.json").read_text()
            )
            first["learning_rate"] = 0.02
            second_path = output / "eta_0p02" / "local_stability.json"
            original_second = second_path.read_text()
            second_path.write_text(json.dumps(first))
            with self.assertRaisesRegex(ValueError, "mismatched provenance"):
                measure_checkpoint_local_stability_scan(
                    object(), object(), object(), snapshot, [0.01, 0.02], output
                )

            second = json.loads(original_second)
            second["provenance"]["controls"]["learning_rate"] = 9.0
            second_path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "tampered provenance"):
                measure_checkpoint_local_stability_scan(
                    object(), object(), object(), snapshot, [0.01, 0.02], output
                )

            second = json.loads(original_second)
            second["augmented_spectral_radius"] = 123.0
            second_path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "tampered result payload"):
                measure_checkpoint_local_stability_scan(
                    object(), object(), object(), snapshot, [0.01, 0.02], output
                )

            second_path.write_text(original_second)
            snapshot.write_bytes(b"source-v2")
            with self.assertRaisesRegex(ValueError, "mismatched provenance"):
                measure_checkpoint_local_stability_scan(
                    object(), object(), object(), snapshot, [0.01, 0.02], output
                )

    def test_worker_rehashes_snapshot_before_load(self) -> None:
        from pinned_capabilities import local_measurement

        with tempfile.TemporaryDirectory() as directory, patch(
            "pinned_capabilities.local_measurement.measure_checkpoint_local_stability",
            side_effect=self.fake_measure,
        ):
            root = Path(directory)
            snapshot = root / "snapshot.pt"
            snapshot.write_bytes(b"scheduled-source")
            output = root / "scan"
            measure_checkpoint_local_stability_scan(
                object(), object(), object(), snapshot, [0.01], output
            )
            provenance = json.loads(
                (output / "eta_0p01" / "local_stability.json").read_text()
            )["provenance"]
            snapshot.write_bytes(b"replaced-before-worker")
            with self.assertRaisesRegex(ValueError, "changed before worker load"):
                local_measurement._measure_local_stability_scan_cell(
                    object(), object(), object(), snapshot, 0.01,
                    root / "worker.json", provenance,
                )

    def test_parallel_scan_matches_serial_order_and_uses_spawn(self) -> None:
        _RecordingExecutor.instances.clear()
        with tempfile.TemporaryDirectory() as directory, patch(
            "pinned_capabilities.local_measurement.measure_checkpoint_local_stability",
            side_effect=self.fake_measure,
        ), patch(
            "pinned_capabilities.local_measurement.sha256_file", return_value="a" * 64
        ):
            root = Path(directory)
            serial = measure_checkpoint_local_stability_scan(
                object(), object(), object(), Path("snapshot"),
                [0.03, 0.01, 0.02], root / "serial",
            )
            with patch(
                "pinned_capabilities.local_measurement.ProcessPoolExecutor",
                _RecordingExecutor,
            ):
                parallel = measure_checkpoint_local_stability_scan(
                    object(), object(), object(), Path("snapshot"),
                    [0.03, 0.01, 0.02], root / "parallel", workers=2,
                )

            self.assertEqual(parallel, serial)
            self.assertEqual(
                (root / "parallel" / "scan.json").read_text(),
                (root / "serial" / "scan.json").read_text(),
            )
            executor = _RecordingExecutor.instances[-1]
            self.assertEqual(executor.max_workers, 2)
            self.assertEqual(executor.start_method, "spawn")
            self.assertEqual(len(executor.submissions), 3)
            self.assertEqual(executor.shutdown_controls, (True, False))

    def test_parallel_scan_schedules_only_incomplete_rate_cells(self) -> None:
        _RecordingExecutor.instances.clear()
        with tempfile.TemporaryDirectory() as directory, patch(
            "pinned_capabilities.local_measurement.measure_checkpoint_local_stability",
            side_effect=self.fake_measure,
        ), patch(
            "pinned_capabilities.local_measurement.ProcessPoolExecutor",
            _RecordingExecutor,
        ), patch(
            "pinned_capabilities.local_measurement.sha256_file", return_value="a" * 64
        ):
            output = Path(directory)
            measure_checkpoint_local_stability_scan(
                object(), object(), object(), Path("snapshot"), [0.01], output
            )
            _RecordingExecutor.instances.clear()
            summary = measure_checkpoint_local_stability_scan(
                object(), object(), object(), Path("snapshot"),
                [0.01, 0.02, 0.03], output, workers=2,
            )

            self.assertTrue(summary["complete"])
            self.assertEqual(len(_RecordingExecutor.instances[-1].submissions), 2)

    def test_scan_rejects_nonpositive_worker_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers must be at least one"):
            measure_checkpoint_local_stability_scan(
                object(), object(), object(), Path("snapshot"), [0.01],
                Path("unused"), workers=0,
            )


if __name__ == "__main__":
    unittest.main()
