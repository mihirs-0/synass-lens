import tempfile
import unittest
from json import loads
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pinned_capabilities.batch_stream import DeterministicBatchStream
from pinned_capabilities.memory_surgery_run import _run_arm, _weight_distance
from pinned_capabilities.snapshot import save_snapshot
from pinned_capabilities.state import ReferenceBands, StateThresholds


class MemoryFactorialValidationTests(unittest.TestCase):
    def test_weight_distance_matches_global_parameter_delta(self) -> None:
        left = torch.nn.Linear(2, 1)
        right = torch.nn.Linear(2, 1)
        right.load_state_dict(left.state_dict())
        with torch.no_grad():
            right.weight.add_(3.0)
            right.bias.add_(4.0)
        left_opt = torch.optim.AdamW(left.parameters())
        right_opt = torch.optim.AdamW(right.parameters())
        left_stream = DeterministicBatchStream(4, 2, 1)
        right_stream = DeterministicBatchStream(4, 2, 1)
        with tempfile.TemporaryDirectory() as directory:
            left_path = Path(directory) / "left.pt"
            right_path = Path(directory) / "right.pt"
            save_snapshot(left_path, model=left, optimizer=left_opt, stream=left_stream, step=2)
            save_snapshot(right_path, model=right, optimizer=right_opt, stream=right_stream, step=2)
            left_payload = torch.load(left_path, weights_only=False)
            right_payload = torch.load(right_path, weights_only=False)
        self.assertAlmostEqual(_weight_distance(left_payload, right_payload), (3**2 * 2 + 4**2) ** 0.5)

    def test_arm_resumes_from_last_transactional_checkpoint(self) -> None:
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

        def fake_save(path, **kwargs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
            return path

        reference = ReferenceBands(
            plateau_mean=0.0,
            plateau_sd=0.1,
            solved_c_int=8.0,
            chance_em_mean=0.0,
            chance_em_sd=0.0,
            q_star_loss_mean=2.0,
            q_star_loss_sd=0.1,
        )
        metric = SimpleNamespace(eval_every=1_000)
        with tempfile.TemporaryDirectory() as directory, patch(
            "pinned_capabilities.memory_surgery_run.MBCExperiment", DummyExperiment
        ), patch(
            "pinned_capabilities.memory_surgery_run.load_crossed_snapshot",
            return_value={"step": 0},
        ) as crossed, patch(
            "pinned_capabilities.memory_surgery_run.load_snapshot",
            return_value={"step": 1_000},
        ) as resumed, patch(
            "pinned_capabilities.memory_surgery_run.save_snapshot", side_effect=fake_save
        ), patch("pinned_capabilities.memory_surgery_run.set_learning_rates"):
            output = Path(directory)
            kwargs = dict(
                experiment_config=SimpleNamespace(),
                metric=metric,
                reference=reference,
                thresholds=StateThresholds(),
                weights_path=Path("weights.pt"),
                optimizer_path=Path("optimizer.pt"),
                weights_source="expressed",
                optimizer_source="suppressed",
                learning_rate=0.001,
                challenge_steps=2_000,
                output_dir=output,
            )
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                _run_arm(**kwargs)
            progress = loads((output / "progress.json").read_text())
            self.assertEqual(progress["completed_branch_steps"], 1_000)
            result = _run_arm(**kwargs)
            self.assertEqual(result.outcome, "expressed")
            rows = (output / "metrics.jsonl").read_text().splitlines()
            self.assertEqual([loads(row)["branch_step"] for row in rows], [1_000.0, 2_000.0])
            self.assertEqual(crossed.call_count, 1)
            self.assertEqual(resumed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
