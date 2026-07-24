import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pinned_capabilities.config import Gate0Config, MBCExperimentConfig, MetricConfig
from pinned_capabilities.experiment import MBCExperiment
from pinned_capabilities.gate0_boundary import (
    AcquisitionResult,
    ErasureResult,
    geometric_erasure_bisection,
    run_acquisition_branch,
    run_acquisition_scan,
    run_erasure_branch,
    run_erasure_scan,
    sustained_band_entry,
)
from pinned_capabilities.snapshot import save_snapshot
from pinned_capabilities.state import ReferenceBands


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


class ErasureClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = ReferenceBands(plateau_mean=0.0, plateau_sd=0.1, solved_c_int=8.0,
                                        chance_em_mean=0.0, chance_em_sd=0.0,
                                        q_star_loss_mean=2.0, q_star_loss_sd=0.1)

    def test_requires_sustained_entry_and_complete_hold(self) -> None:
        rows = [
            {"branch_step": 50.0, "c_int": 2.0, "full_vocab_ce": 2.0},
            {"branch_step": 100.0, "c_int": 0.1, "full_vocab_ce": 2.0},
            {"branch_step": 150.0, "c_int": 0.5, "full_vocab_ce": 2.0},
            {"branch_step": 200.0, "c_int": 0.2, "full_vocab_ce": 2.0},
            {"branch_step": 250.0, "c_int": 0.1, "full_vocab_ce": 2.0},
        ]
        self.assertEqual(sustained_band_entry(rows, self.reference, branch_end_step=250), 200)
        self.assertIsNone(sustained_band_entry(rows[:-1], self.reference, branch_end_step=250))

    def test_rejects_loss_divergence_inside_interaction_band(self) -> None:
        rows = [
            {"branch_step": 50.0, "c_int": 0.1, "full_vocab_ce": 50.0},
            {"branch_step": 100.0, "c_int": 0.1, "full_vocab_ce": 50.0},
        ]
        self.assertIsNone(sustained_band_entry(rows, self.reference, branch_end_step=100))

    @staticmethod
    def result(rate: float, outcome: str) -> ErasureResult:
        return ErasureResult(
            learning_rate=rate,
            erased=outcome == "erased",
            outcome=outcome,
            sustained_entry_step=100 if outcome == "erased" else None,
            final_c_int=8.0 if outcome == "retained" else 0.0,
            final_exact_match=1.0 if outcome == "retained" else 0.0,
            final_delta_z=0.0,
            final_full_vocab_ce=0.01 if outcome == "retained" else 2.0,
        )

    def test_erasure_scan_resumes_completed_branches(self) -> None:
        def fake_branch(*args, **kwargs):
            rate = args[5]
            return self.result(rate, "retained" if rate == 0.01 else "erased")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ):
            output = Path(temporary)
            with patch(
                "pinned_capabilities.gate0_boundary.run_erasure_branch",
                side_effect=fake_branch,
            ) as runner:
                summary = run_erasure_scan(
                    object(), object(), object(), self.reference, Path("snapshot"),
                    [0.01, 0.02], output,
                )
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(
                summary["strict_adjacent_brackets"],
                [{"lower_learning_rate": 0.01, "upper_learning_rate": 0.02}],
            )
            with patch(
                "pinned_capabilities.gate0_boundary.run_erasure_branch",
                side_effect=AssertionError("completed branches should not rerun"),
            ):
                resumed = run_erasure_scan(
                    object(), object(), object(), self.reference, Path("snapshot"),
                    [0.01, 0.02], output,
                )
            self.assertEqual(resumed, summary)

    def test_erasure_cache_rejects_copied_tampered_and_stale_cells(self) -> None:
        def fake_branch(*args, **kwargs):
            return self.result(args[5], "retained")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.run_erasure_branch",
            side_effect=fake_branch,
        ):
            root = Path(temporary)
            snapshot = root / "snapshot.pt"
            snapshot.write_bytes(b"source-v1")
            output = root / "scan"
            run_erasure_scan(
                object(), object(), object(), self.reference, snapshot,
                [0.01, 0.02], output,
            )

            first = __import__("json").loads(
                (output / "eta_0p01" / "result.json").read_text()
            )
            first["learning_rate"] = 0.02
            second_path = output / "eta_0p02" / "result.json"
            original_second = second_path.read_text()
            second_path.write_text(__import__("json").dumps(first))
            with self.assertRaisesRegex(ValueError, "mismatched provenance"):
                run_erasure_scan(
                    object(), object(), object(), self.reference, snapshot,
                    [0.01, 0.02], output,
                )

            second = __import__("json").loads(original_second)
            second["provenance"]["controls"]["learning_rate"] = 9.0
            second_path.write_text(__import__("json").dumps(second))
            with self.assertRaisesRegex(ValueError, "tampered provenance"):
                run_erasure_scan(
                    object(), object(), object(), self.reference, snapshot,
                    [0.01, 0.02], output,
                )

            second = __import__("json").loads(original_second)
            second["final_c_int"] = 123.0
            second_path.write_text(__import__("json").dumps(second))
            with self.assertRaisesRegex(ValueError, "tampered result payload"):
                run_erasure_scan(
                    object(), object(), object(), self.reference, snapshot,
                    [0.01, 0.02], output,
                )

            second_path.write_text(original_second)
            snapshot.write_bytes(b"source-v2")
            with self.assertRaisesRegex(ValueError, "mismatched provenance"):
                run_erasure_scan(
                    object(), object(), object(), self.reference, snapshot,
                    [0.01, 0.02], output,
                )

    def test_erasure_cache_rejects_inconsistent_outcome_flag(self) -> None:
        def fake_branch(*args, **kwargs):
            return self.result(args[5], "retained")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.run_erasure_branch",
            side_effect=fake_branch,
        ):
            root = Path(temporary)
            snapshot = root / "snapshot.pt"
            snapshot.write_bytes(b"source")
            output = root / "scan"
            run_erasure_scan(
                MBCExperimentConfig(), MetricConfig(), Gate0Config(),
                self.reference, snapshot, [0.01], output,
            )
            result_path = output / "eta_0p01" / "result.json"
            payload = __import__("json").loads(result_path.read_text())
            payload["erased"] = True
            result_path.write_text(__import__("json").dumps(payload))
            with self.assertRaisesRegex(ValueError, "inconsistent outcome/erased"):
                run_erasure_scan(
                    MBCExperimentConfig(), MetricConfig(), Gate0Config(),
                    self.reference, snapshot, [0.01], output,
                )

    def test_erasure_worker_rehashes_source_before_load(self) -> None:
        from pinned_capabilities import gate0_boundary

        def fake_branch(*args, **kwargs):
            return self.result(args[5], "retained")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.run_erasure_branch",
            side_effect=fake_branch,
        ):
            root = Path(temporary)
            snapshot = root / "snapshot.pt"
            snapshot.write_bytes(b"scheduled-source")
            output = root / "scan"
            run_erasure_scan(
                object(), object(), object(), self.reference, snapshot, [0.01], output
            )
            provenance = __import__("json").loads(
                (output / "eta_0p01" / "result.json").read_text()
            )["provenance"]
            snapshot.write_bytes(b"replaced-before-worker")
            with self.assertRaisesRegex(ValueError, "changed before worker load"):
                gate0_boundary._run_erasure_scan_cell(
                    object(), object(), object(), self.reference, snapshot, 0.01,
                    root / "worker", None, provenance,
                )

    def test_parallel_scan_matches_serial_order_and_uses_spawn(self) -> None:
        def fake_branch(*args, **kwargs):
            rate = args[5]
            return self.result(rate, "retained" if rate < 0.02 else "erased")

        _RecordingExecutor.instances.clear()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.run_erasure_branch",
            side_effect=fake_branch,
        ), patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ):
            root = Path(temporary)
            serial = run_erasure_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.03, 0.01, 0.02], root / "serial",
            )
            with patch(
                "pinned_capabilities.gate0_boundary.ProcessPoolExecutor",
                _RecordingExecutor,
            ):
                parallel = run_erasure_scan(
                    object(), object(), object(), self.reference, Path("snapshot"),
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
        def fake_branch(*args, **kwargs):
            rate = args[5]
            return self.result(rate, "retained")

        _RecordingExecutor.instances.clear()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.run_erasure_branch",
            side_effect=fake_branch,
        ), patch(
            "pinned_capabilities.gate0_boundary.ProcessPoolExecutor",
            _RecordingExecutor,
        ), patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ):
            output = Path(temporary)
            run_erasure_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.01], output,
            )
            _RecordingExecutor.instances.clear()
            summary = run_erasure_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.01, 0.02, 0.03], output, workers=2,
            )

            self.assertTrue(summary["complete"])
            self.assertEqual(len(_RecordingExecutor.instances[-1].submissions), 2)

    def test_erasure_scan_rejects_nonpositive_worker_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers must be at least one"):
            run_erasure_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.01], Path("unused"), workers=0,
            )

    def test_spawned_processes_execute_real_disjoint_rate_cells(self) -> None:
        if not hasattr(os, "sysconf"):
            self.skipTest("platform does not expose the process-pool semaphore query")
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except PermissionError:
            self.skipTest("sandbox forbids the semaphore query required by process pools")
        experiment_config = MBCExperimentConfig(
            seed=83,
            n_unique_b=12,
            k=3,
            b_length=3,
            a_length=2,
            z_length=1,
            n_layers=1,
            n_heads=1,
            d_model=16,
            d_head=16,
            d_mlp=32,
            batch_size=6,
            probe_b_count=5,
            device="cpu",
        )
        metric = MetricConfig(eval_every=1, quartets_per_probe=4)
        gate = Gate0Config(erase_horizon=1, erase_hold_steps=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = MBCExperiment(experiment_config, metric)
            snapshot = save_snapshot(
                root / "source.pt",
                model=source.model,
                optimizer=source.optimizer,
                stream=source.stream,
                step=source.step,
            )
            serial = run_erasure_scan(
                experiment_config,
                metric,
                gate,
                self.reference,
                snapshot,
                [0.001, 0.002],
                root / "serial",
            )
            summary = run_erasure_scan(
                experiment_config,
                metric,
                gate,
                self.reference,
                snapshot,
                [0.001, 0.002],
                root / "scan",
                workers=2,
            )

            self.assertTrue(summary["complete"])
            self.assertEqual(summary, serial)
            self.assertEqual(
                (root / "scan" / "scan.json").read_text(),
                (root / "serial" / "scan.json").read_text(),
            )
            self.assertEqual(
                [branch["learning_rate"] for branch in summary["branches"]],
                [0.001, 0.002],
            )
            self.assertTrue((root / "scan" / "eta_0p001" / "result.json").exists())
            self.assertTrue((root / "scan" / "eta_0p002" / "result.json").exists())

    def test_bisection_rejects_divergent_midpoint(self) -> None:
        def fake_branch(*args, **kwargs):
            rate = args[5]
            if rate == 0.01:
                return self.result(rate, "retained")
            if rate == 0.04:
                return self.result(rate, "erased")
            return self.result(rate, "diverged")

        gate = SimpleNamespace(boundary_bisection_steps=1)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.run_erasure_branch",
            side_effect=fake_branch,
        ), patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ):
            with self.assertRaisesRegex(ValueError, "invalid midpoint outcome: diverged"):
                geometric_erasure_bisection(
                    object(), object(), gate, self.reference, Path("snapshot"),
                    lower_learning_rate=0.01,
                    upper_learning_rate=0.04,
                    output_dir=Path(temporary),
                )

    def test_erasure_branch_resumes_exact_trajectory(self) -> None:
        class DummyExperiment:
            advances = 0

            def __init__(self, config, metric):
                self.step = 0
                self.model = object()
                self.optimizer = object()
                self.stream = object()
                self.device = "cpu"

            def advance(self, steps):
                type(self).advances += 1
                if type(self).advances == 2:
                    raise RuntimeError("simulated interruption")
                self.step += steps
                return {"train_loss": 0.0, "step": float(self.step)}

            def evaluate(self):
                return {
                    "c_int": 8.0,
                    "exact_match": 1.0,
                    "delta_z": 1.0,
                    "full_vocab_ce": 0.01,
                }

        def fake_load(path, **kwargs):
            return {"step": 0 if Path(path).name == "source.pt" else 1_000}

        def fake_save(path, **kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
            return path

        gate = SimpleNamespace(erase_hold_steps=2_000, erase_horizon=1_000)
        metric = SimpleNamespace(eval_every=1_000)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.MBCExperiment", DummyExperiment
        ), patch(
            "pinned_capabilities.gate0_boundary.load_snapshot", side_effect=fake_load
        ), patch(
            "pinned_capabilities.gate0_boundary.save_snapshot", side_effect=fake_save
        ), patch("pinned_capabilities.gate0_boundary.set_learning_rates"):
            source_path = Path(temporary) / "source.pt"
            source_path.write_bytes(b"source")
            kwargs = dict(
                experiment_config=MBCExperimentConfig(),
                metric=metric,
                gate=gate,
                reference=self.reference,
                snapshot_path=source_path,
                learning_rate=0.01,
                output_dir=Path(temporary),
            )
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_erasure_branch(**kwargs)
            progress_path = Path(temporary) / "progress.json"
            original_progress = progress_path.read_text()
            progress = __import__("json").loads(original_progress)
            progress["provenance"]["controls"]["learning_rate"] = 9.0
            progress_path.write_text(__import__("json").dumps(progress))
            with self.assertRaisesRegex(ValueError, "tampered provenance"):
                run_erasure_branch(**kwargs)
            progress_path.write_text(original_progress)
            progress = __import__("json").loads(original_progress)
            legacy_progress = dict(progress)
            legacy_progress.pop("metrics_prefix")
            progress_path.write_text(__import__("json").dumps(legacy_progress))
            with self.assertRaisesRegex(ValueError, "no metrics-prefix provenance"):
                run_erasure_branch(**kwargs)
            progress_path.write_text(original_progress)
            checkpoint = Path(temporary) / progress["checkpoint_path"]
            checkpoint.write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint digest mismatch"):
                run_erasure_branch(**kwargs)
            checkpoint.write_bytes(b"checkpoint")

            metrics_path = Path(temporary) / "metrics.jsonl"
            original_metrics = metrics_path.read_bytes()
            altered_metrics = original_metrics.replace(b'"c_int": 8.0', b'"c_int": 0.0')
            self.assertNotEqual(altered_metrics, original_metrics)
            self.assertEqual(len(altered_metrics), len(original_metrics))
            metrics_path.write_bytes(altered_metrics)
            with self.assertRaisesRegex(ValueError, "metrics-prefix digest mismatch"):
                run_erasure_branch(**kwargs)

            metrics_path.write_bytes(original_metrics[:-1])
            with self.assertRaisesRegex(ValueError, "shorter than its bound prefix"):
                run_erasure_branch(**kwargs)

            metrics_path.write_bytes(original_metrics + b'{"kind":')
            with self.assertRaisesRegex(ValueError, "truncated or lacks a final newline"):
                run_erasure_branch(**kwargs)

            metrics_path.write_bytes(original_metrics + b"not-json\n")
            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                run_erasure_branch(**kwargs)

            speculative = __import__("json").loads(original_metrics.splitlines()[-1])
            speculative["branch_step"] = 1_500.0
            speculative["step"] = 1_500.0
            metrics_path.write_bytes(
                original_metrics
                + (__import__("json").dumps(speculative, sort_keys=True) + "\n").encode()
            )
            result = run_erasure_branch(**kwargs)
            self.assertEqual(result.outcome, "retained")
            logged = [
                __import__("json").loads(row)
                for row in metrics_path.read_text().splitlines()
            ]
            self.assertEqual([row["branch_step"] for row in logged], [1_000.0, 2_000.0])


class AcquisitionScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = ReferenceBands(
            plateau_mean=0.0,
            plateau_sd=0.1,
            solved_c_int=8.0,
            chance_em_mean=0.0,
            chance_em_sd=0.0,
            q_star_loss_mean=2.0,
            q_star_loss_sd=0.1,
        )

    @staticmethod
    def result(rate: float, outcome: str) -> AcquisitionResult:
        return AcquisitionResult(
            learning_rate=rate,
            outcome=outcome,
            transition_step=100 if outcome == "transitioned" else None,
            final_c_int=8.0 if outcome == "transitioned" else 0.0,
            final_exact_match=1.0 if outcome == "transitioned" else 0.0,
            final_delta_z=1.0 if outcome == "transitioned" else 0.0,
            final_full_vocab_ce=20.0 if outcome == "diverged" else 2.0,
        )

    def test_scan_preserves_strict_outcomes_and_resumes_completed_cells(self) -> None:
        outcomes = {0.01: "transitioned", 0.02: "censored", 0.03: "diverged"}

        def fake_branch(*args, **kwargs):
            rate = args[5]
            return self.result(rate, outcomes[rate])

        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ):
            output = Path(temporary)
            with patch(
                "pinned_capabilities.gate0_boundary.run_acquisition_branch",
                side_effect=fake_branch,
            ) as runner:
                summary = run_acquisition_scan(
                    object(), object(), object(), self.reference, Path("snapshot"),
                    [0.03, 0.01, 0.02], output,
                )
            self.assertEqual(runner.call_count, 3)
            self.assertEqual(summary["largest_transitioning_learning_rate"], 0.01)
            self.assertEqual(
                summary["outcome_counts"],
                {"transitioned": 1, "censored": 1, "diverged": 1},
            )
            self.assertEqual(
                [branch["outcome"] for branch in summary["branches"]],
                ["transitioned", "censored", "diverged"],
            )
            with patch(
                "pinned_capabilities.gate0_boundary.run_acquisition_branch",
                side_effect=AssertionError("completed acquisition cells should not rerun"),
            ):
                resumed = run_acquisition_scan(
                    object(), object(), object(), self.reference, Path("snapshot"),
                    [0.01, 0.02, 0.03], output,
                )
            self.assertEqual(resumed, summary)

    def test_acquisition_cache_is_bound_to_reference_and_controls(self) -> None:
        def fake_branch(*args, **kwargs):
            return self.result(args[5], "censored")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.run_acquisition_branch",
            side_effect=fake_branch,
        ):
            root = Path(temporary)
            snapshot = root / "snapshot.pt"
            snapshot.write_bytes(b"source")
            output = root / "scan"
            run_acquisition_scan(
                object(), object(), object(), self.reference, snapshot, [0.01], output
            )
            changed_reference = ReferenceBands(
                plateau_mean=0.2,
                plateau_sd=self.reference.plateau_sd,
                solved_c_int=self.reference.solved_c_int,
                chance_em_mean=self.reference.chance_em_mean,
                chance_em_sd=self.reference.chance_em_sd,
                q_star_loss_mean=self.reference.q_star_loss_mean,
                q_star_loss_sd=self.reference.q_star_loss_sd,
            )
            with self.assertRaisesRegex(ValueError, "mismatched provenance"):
                run_acquisition_scan(
                    object(), object(), object(), changed_reference,
                    snapshot, [0.01], output,
                )

    def test_parallel_scan_matches_serial_and_schedules_only_missing_cells(self) -> None:
        def fake_branch(*args, **kwargs):
            rate = args[5]
            return self.result(rate, "transitioned" if rate < 0.03 else "censored")

        _RecordingExecutor.instances.clear()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.run_acquisition_branch",
            side_effect=fake_branch,
        ), patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ):
            root = Path(temporary)
            serial = run_acquisition_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.03, 0.01, 0.02], root / "serial",
            )
            parallel_root = root / "parallel"
            run_acquisition_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.01], parallel_root,
            )
            _RecordingExecutor.instances.clear()
            with patch(
                "pinned_capabilities.gate0_boundary.ProcessPoolExecutor",
                _RecordingExecutor,
            ):
                parallel = run_acquisition_scan(
                    object(), object(), object(), self.reference, Path("snapshot"),
                    [0.03, 0.01, 0.02], parallel_root, workers=2,
                )

            self.assertEqual(parallel, serial)
            self.assertEqual(
                (parallel_root / "scan.json").read_text(),
                (root / "serial" / "scan.json").read_text(),
            )
            executor = _RecordingExecutor.instances[-1]
            self.assertEqual(executor.max_workers, 2)
            self.assertEqual(executor.start_method, "spawn")
            self.assertEqual(len(executor.submissions), 2)
            self.assertEqual(executor.shutdown_controls, (True, False))

    def test_scan_rejects_invalid_rates_workers_and_cached_outcomes(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers must be at least one"):
            run_acquisition_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.01], Path("unused"), workers=0,
            )
        with self.assertRaisesRegex(ValueError, "learning rates must be positive"):
            run_acquisition_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.0], Path("unused"),
            )
        with self.assertRaisesRegex(ValueError, "learning rates must be unique"):
            run_acquisition_scan(
                object(), object(), object(), self.reference, Path("snapshot"),
                [0.01, 0.01], Path("unused"),
            )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ):
            result_path = Path(temporary) / "eta_0p01" / "result.json"
            result_path.parent.mkdir()
            payload = asdict(self.result(0.01, "censored"))
            payload["outcome"] = "stable"
            result_path.write_text(__import__("json").dumps(payload))
            with self.assertRaisesRegex(ValueError, "invalid outcome"):
                run_acquisition_scan(
                    object(), object(), object(), self.reference, Path("snapshot"),
                    [0.01], Path(temporary),
                )

    def test_spawned_processes_execute_real_acquisition_cells(self) -> None:
        if not hasattr(os, "sysconf"):
            self.skipTest("platform does not expose the process-pool semaphore query")
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except PermissionError:
            self.skipTest("sandbox forbids the semaphore query required by process pools")
        experiment_config = MBCExperimentConfig(
            seed=89,
            n_unique_b=12,
            k=3,
            b_length=3,
            a_length=2,
            z_length=1,
            n_layers=1,
            n_heads=1,
            d_model=16,
            d_head=16,
            d_mlp=32,
            batch_size=6,
            probe_b_count=5,
            device="cpu",
        )
        metric = MetricConfig(eval_every=1, quartets_per_probe=4)
        gate = Gate0Config(acquire_horizon=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = MBCExperiment(experiment_config, metric)
            initial = source.evaluate()
            reference = ReferenceBands(
                plateau_mean=initial["c_int"],
                plateau_sd=1.0,
                solved_c_int=8.0,
                chance_em_mean=0.0,
                chance_em_sd=0.0,
                q_star_loss_mean=initial["full_vocab_ce"],
                q_star_loss_sd=0.1,
            )
            snapshot = save_snapshot(
                root / "source.pt",
                model=source.model,
                optimizer=source.optimizer,
                stream=source.stream,
                step=source.step,
            )
            serial = run_acquisition_scan(
                experiment_config,
                metric,
                gate,
                reference,
                snapshot,
                [0.001, 0.002],
                root / "serial",
            )
            parallel = run_acquisition_scan(
                experiment_config,
                metric,
                gate,
                reference,
                snapshot,
                [0.001, 0.002],
                root / "parallel",
                workers=2,
            )

            self.assertEqual(parallel, serial)
            self.assertEqual(
                (root / "parallel" / "scan.json").read_text(),
                (root / "serial" / "scan.json").read_text(),
            )
            self.assertTrue((root / "parallel" / "eta_0p001" / "result.json").exists())
            self.assertTrue((root / "parallel" / "eta_0p002" / "result.json").exists())

    def test_acquisition_branch_resumes_exact_trajectory(self) -> None:
        class DummyExperiment:
            advances = 0

            def __init__(self, config, metric):
                self.step = 0
                self.model = object()
                self.optimizer = object()
                self.stream = object()
                self.device = "cpu"

            def advance(self, steps):
                type(self).advances += 1
                if type(self).advances == 2:
                    raise RuntimeError("simulated interruption")
                self.step += steps
                return {"train_loss": 0.0, "step": float(self.step)}

            def evaluate(self):
                return {
                    "c_int": 0.0,
                    "exact_match": 0.0,
                    "delta_z": 0.0,
                    "full_vocab_ce": 2.0,
                }

        def fake_load(path, **kwargs):
            return {"step": 0 if Path(path).name == "source.pt" else 1_000}

        def fake_save(path, **kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
            return path

        gate = SimpleNamespace(acquire_horizon=2_000)
        metric = SimpleNamespace(eval_every=1_000)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.MBCExperiment", DummyExperiment
        ), patch(
            "pinned_capabilities.gate0_boundary.load_snapshot", side_effect=fake_load
        ), patch(
            "pinned_capabilities.gate0_boundary.save_snapshot", side_effect=fake_save
        ), patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ), patch("pinned_capabilities.gate0_boundary.set_learning_rates"):
            kwargs = dict(
                experiment_config=MBCExperimentConfig(),
                metric=metric,
                gate=gate,
                reference=self.reference,
                snapshot_path=Path("source.pt"),
                learning_rate=0.01,
                output_dir=Path(temporary),
            )
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_acquisition_branch(**kwargs)
            progress = __import__("json").loads(
                (Path(temporary) / "progress.json").read_text()
            )
            metrics_path = Path(temporary) / "metrics.jsonl"
            original_metrics = metrics_path.read_bytes()
            self.assertEqual(
                progress["metrics_prefix"]["size_bytes"], len(original_metrics)
            )
            self.assertEqual(len(progress["metrics_prefix"]["sha256"]), 64)
            altered_metrics = original_metrics.replace(
                b'"delta_z": 0.0', b'"delta_z": 9.0'
            )
            self.assertNotEqual(altered_metrics, original_metrics)
            metrics_path.write_bytes(altered_metrics)
            with self.assertRaisesRegex(ValueError, "metrics-prefix digest mismatch"):
                run_acquisition_branch(**kwargs)
            metrics_path.write_bytes(original_metrics)
            result = run_acquisition_branch(**kwargs)
            self.assertEqual(result.outcome, "censored")
            logged = [
                __import__("json").loads(row)
                for row in metrics_path.read_text().splitlines()
            ]
            timed = [row for row in logged if row["kind"] == "gate0_acquisition"]
            self.assertEqual([row["branch_step"] for row in timed], [1_000.0, 2_000.0])

    def test_divergence_invalidates_an_observed_acquisition_transition(self) -> None:
        class DummyExperiment:
            def __init__(self, config, metric):
                self.step = 0
                self.model = object()
                self.optimizer = object()
                self.stream = object()
                self.device = "cpu"

            def advance(self, steps):
                self.step += steps
                return {"train_loss": 0.0, "step": float(self.step)}

            def evaluate(self):
                return {
                    "c_int": 0.0 if self.step == 0 else 8.0,
                    "exact_match": 0.0,
                    "delta_z": 0.0,
                    "full_vocab_ce": 2.0 if self.step == 0 else 20.0,
                }

        def fake_save(path, **kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
            return path

        with tempfile.TemporaryDirectory() as temporary, patch(
            "pinned_capabilities.gate0_boundary.MBCExperiment", DummyExperiment
        ), patch(
            "pinned_capabilities.gate0_boundary.load_snapshot", return_value={"step": 0}
        ), patch(
            "pinned_capabilities.gate0_boundary.save_snapshot", side_effect=fake_save
        ), patch(
            "pinned_capabilities.gate0_boundary.first_transition_step", return_value=1
        ), patch(
            "pinned_capabilities.gate0_boundary.sha256_file", return_value="a" * 64
        ), patch("pinned_capabilities.gate0_boundary.set_learning_rates"):
            result = run_acquisition_branch(
                MBCExperimentConfig(),
                SimpleNamespace(eval_every=1),
                SimpleNamespace(acquire_horizon=1),
                self.reference,
                Path("source.pt"),
                0.01,
                Path(temporary),
            )

        self.assertEqual(result.outcome, "diverged")
        self.assertEqual(result.transition_step, 1)

    def test_cli_manifest_freezes_science_but_not_worker_count(self) -> None:
        from pinned_capabilities import __main__ as cli

        summary = {
            "complete": True,
            "requested_learning_rates": [0.003, 0.006, 0.012, 0.025, 0.05],
            "largest_transitioning_learning_rate": None,
            "outcome_counts": {"transitioned": 0, "censored": 5, "diverged": 0},
            "branches": [],
        }
        argv = [
            "pinned_capabilities",
            "acquisition-scan",
            "--reference", "reference.json",
            "--snapshot", "suppressed.pt",
            "--seed", "3",
            "--batch-size", "512",
            "--learning-rates", "0.003", "0.006", "0.012", "0.025", "0.05",
            "--workers", "3",
            "--calibration", "calibration.json",
            "--local-scan", "local-scan.json",
            "--first-cell-analysis", "first-cell.json",
            "--output", "scan-output",
            "--device", "cpu",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            cli, "load_reference_bands", return_value=self.reference
        ), patch.object(
            cli, "_reference_input", return_value={"artifact": "reference-binding"}
        ), patch.object(
            cli, "_snapshot_input", return_value={"artifact": "snapshot-binding"}
        ), patch.object(
            cli, "_calibration_input", return_value={"artifact": "calibration-binding"}
        ), patch.object(
            cli, "_local_prerequisite_input", return_value={"artifact": "local-binding"}
        ), patch.object(
            cli, "_first_cell_analysis_input", return_value={"artifact": "first-binding"}
        ), patch.object(
            cli, "freeze_manifest"
        ) as freeze, patch.object(
            cli, "run_acquisition_scan", return_value=summary
        ) as runner, redirect_stdout(io.StringIO()):
            cli.main()

        frozen = freeze.call_args.args[0]
        self.assertEqual(frozen["kind"], "gate0_acquisition_scan")
        self.assertEqual(frozen["learning_rates"], (0.003, 0.006, 0.012, 0.025, 0.05))
        self.assertEqual(frozen["experiment"].seed, 3)
        self.assertEqual(frozen["experiment"].batch_size, 512)
        self.assertEqual(frozen["reference_input"], {"artifact": "reference-binding"})
        self.assertEqual(frozen["snapshot_input"], {"artifact": "snapshot-binding"})
        self.assertEqual(frozen["calibration_input"], {"artifact": "calibration-binding"})
        self.assertEqual(frozen["local_prerequisite"], {"artifact": "local-binding"})
        self.assertEqual(frozen["first_cell_analysis"], {"artifact": "first-binding"})
        self.assertNotIn("workers", frozen)
        self.assertEqual(runner.call_args.kwargs["workers"], 3)


if __name__ == "__main__":
    unittest.main()
