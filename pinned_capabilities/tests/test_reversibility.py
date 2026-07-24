import tempfile
import unittest
from json import loads
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pinned_capabilities.config import MBCExperimentConfig
from pinned_capabilities.reversibility_run import run_transition_timing
from pinned_capabilities.state import ReferenceBands, StateThresholds


class TransitionTimingResumeTests(unittest.TestCase):
    def test_resumes_sustained_expression_clock(self) -> None:
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
            "pinned_capabilities.reversibility_run.MBCExperiment", DummyExperiment
        ), patch(
            "pinned_capabilities.reversibility_run.load_snapshot",
            return_value={"step": 1_000},
        ) as resumed, patch(
            "pinned_capabilities.reversibility_run.save_snapshot", side_effect=fake_save
        ), patch("pinned_capabilities.reversibility_run.set_learning_rates"):
            kwargs = dict(
                experiment_config=MBCExperimentConfig(),
                metric=metric,
                reference=reference,
                thresholds=StateThresholds(),
                arm="recovery",
                learning_rate=0.001,
                maximum_steps=3_000,
                output_dir=Path(directory),
            )
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_transition_timing(**kwargs)
            progress = loads((Path(directory) / "progress.json").read_text())
            self.assertEqual(progress["completed_branch_steps"], 1_000)
            self.assertEqual(progress["expressed_step"], 1_000)
            result = run_transition_timing(**kwargs)
            self.assertEqual(result.outcome, "expressed")
            self.assertEqual(result.expressed_step, 1_000)
            self.assertEqual(result.final_step, 2_000)
            rows = [loads(row) for row in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            timed_steps = [row["branch_step"] for row in rows if row["kind"] == "transition_timing"]
            self.assertEqual(timed_steps, [1_000.0, 2_000.0])
            self.assertEqual(resumed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
