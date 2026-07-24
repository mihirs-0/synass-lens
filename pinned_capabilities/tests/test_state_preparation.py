import json
import tempfile
import unittest
from dataclasses import replace
from json import loads
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pinned_capabilities.config import MBCExperimentConfig
from pinned_capabilities.state import ReferenceBands, StateThresholds
from pinned_capabilities.state_preparation import prepare_expressed_checkpoint


class DummyExperiment:
    advances = 0
    fail_on_advance = None

    def __init__(self, config, metric):
        self.step = 0
        self.model = object()
        self.optimizer = object()
        self.stream = object()
        self.device = "cpu"

    def advance(self, steps):
        type(self).advances += 1
        if type(self).advances == type(self).fail_on_advance:
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


def fake_save(path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"checkpoint-step-{kwargs['step']}".encode())
    return path


class StatePreparationResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        DummyExperiment.advances = 0
        DummyExperiment.fail_on_advance = 2
        self.reference = ReferenceBands(
            plateau_mean=0.0,
            plateau_sd=0.1,
            solved_c_int=8.0,
            chance_em_mean=0.0,
            chance_em_sd=0.0,
            q_star_loss_mean=2.0,
            q_star_loss_sd=0.1,
        )
        self.metric = SimpleNamespace(eval_every=1_000, solved_hold_steps=1_000)
        self.thresholds = StateThresholds()

    def kwargs(self, directory: str, **updates):
        values = dict(
            experiment_config=MBCExperimentConfig(),
            metric=self.metric,
            reference=self.reference,
            thresholds=self.thresholds,
            learning_rate=0.001,
            save_at_step=2_000,
            output_dir=Path(directory),
        )
        values.update(updates)
        return values

    def interrupt(self, kwargs) -> None:
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            prepare_expressed_checkpoint(**kwargs)

    def patches(self):
        return (
            patch("pinned_capabilities.state_preparation.MBCExperiment", DummyExperiment),
            patch(
                "pinned_capabilities.state_preparation.load_snapshot",
                return_value={"step": 1_000},
            ),
            patch("pinned_capabilities.state_preparation.save_snapshot", side_effect=fake_save),
        )

    def test_expressed_preparation_resumes_exact_hashed_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment_patch, load_patch, save_patch = self.patches()
            with experiment_patch, load_patch as resumed, save_patch:
                kwargs = self.kwargs(directory)
                self.interrupt(kwargs)
                progress = loads((Path(directory) / "progress.json").read_text())
                self.assertEqual(progress["completed_step"], 1_000)
                self.assertIn("provenance", progress)
                self.assertIn("checkpoint_sha256", progress)
                self.assertIn("metrics_prefix_sha256", progress)

                # A crash may append an uncheckpointed suffix. Resume must verify the
                # exact committed prefix and discard only that suffix.
                metrics = Path(directory) / "metrics.jsonl"
                committed_size = progress["metrics_prefix_size_bytes"]
                with metrics.open("ab") as stream:
                    stream.write(b"partial uncommitted JSON")
                result = prepare_expressed_checkpoint(**kwargs)

                self.assertEqual(result.outcome, "expressed")
                self.assertEqual(result.step, 2_000)
                self.assertTrue(Path(result.snapshot_path).exists())
                self.assertIsNotNone(result.snapshot_sha256)
                self.assertIsNotNone(result.result_sha256)
                rows = [loads(row) for row in metrics.read_text().splitlines()]
                steps = [
                    row["step"]
                    for row in rows
                    if row["kind"] == "expressed_preparation"
                ]
                self.assertEqual(steps, [1_000.0, 2_000.0])
                self.assertGreater(metrics.stat().st_size, committed_size)
                self.assertEqual(resumed.call_count, 1)

    def test_progress_rejects_every_changed_bound_input_before_checkpoint_load(self) -> None:
        variants = {
            "experiment": {
                "experiment_config": replace(MBCExperimentConfig(), seed=4),
            },
            "metric": {
                "metric": SimpleNamespace(eval_every=1_000, solved_hold_steps=2_000),
            },
            "reference": {
                "reference": replace(self.reference, solved_c_int=9.0),
            },
            "thresholds": {
                "thresholds": replace(self.thresholds, expressed_fraction=0.8),
            },
            "controls": {"learning_rate": 0.002},
        }
        for label, updates in variants.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                DummyExperiment.advances = 0
                DummyExperiment.fail_on_advance = 2
                experiment_patch, load_patch, save_patch = self.patches()
                with experiment_patch, load_patch as resumed, save_patch:
                    kwargs = self.kwargs(directory)
                    self.interrupt(kwargs)
                    with self.assertRaisesRegex(ValueError, "mismatched provenance"):
                        prepare_expressed_checkpoint(**{**kwargs, **updates})
                    self.assertEqual(resumed.call_count, 0)

    def test_progress_rejects_tampered_checkpoint_and_metrics_prefix(self) -> None:
        for artifact in ("checkpoint", "metrics"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                DummyExperiment.advances = 0
                DummyExperiment.fail_on_advance = 2
                experiment_patch, load_patch, save_patch = self.patches()
                with experiment_patch, load_patch as resumed, save_patch:
                    kwargs = self.kwargs(directory)
                    self.interrupt(kwargs)
                    progress = loads((Path(directory) / "progress.json").read_text())
                    if artifact == "checkpoint":
                        checkpoint = Path(directory) / progress["checkpoint_path"]
                        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
                        message = "checkpoint digest mismatch"
                    else:
                        metrics = Path(directory) / "metrics.jsonl"
                        payload = bytearray(metrics.read_bytes())
                        payload[0] ^= 1
                        metrics.write_bytes(payload)
                        message = "metrics prefix digest mismatch"
                    with self.assertRaisesRegex(ValueError, message):
                        prepare_expressed_checkpoint(**kwargs)
                    self.assertEqual(resumed.call_count, 0)

    def test_completed_result_rejects_changed_inputs_payload_and_final_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment_patch, load_patch, save_patch = self.patches()
            with experiment_patch, load_patch, save_patch:
                kwargs = self.kwargs(directory)
                self.interrupt(kwargs)
                result = prepare_expressed_checkpoint(**kwargs)
                result_path = Path(directory) / "result.json"
                original_result = result_path.read_bytes()
                snapshot_path = Path(result.snapshot_path)
                original_snapshot = snapshot_path.read_bytes()

                with self.assertRaisesRegex(ValueError, "mismatched provenance"):
                    prepare_expressed_checkpoint(
                        **{
                            **kwargs,
                            "thresholds": replace(
                                self.thresholds,
                                expressed_fraction=0.8,
                            ),
                        }
                    )

                payload = json.loads(original_result)
                payload["c_int"] = 7.0
                result_path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "tampered result payload"):
                    prepare_expressed_checkpoint(**kwargs)

                result_path.write_bytes(original_result)
                snapshot_path.write_bytes(original_snapshot + b"tampered")
                with self.assertRaisesRegex(ValueError, "final snapshot digest mismatch"):
                    prepare_expressed_checkpoint(**kwargs)


if __name__ == "__main__":
    unittest.main()
