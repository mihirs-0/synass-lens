import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pinned_capabilities.config import MBCExperimentConfig
from pinned_capabilities.gate0_boundary import (
    ErasureResult,
    geometric_erasure_bisection,
    run_erasure_branch,
    run_erasure_scan,
    sustained_band_entry,
)
from pinned_capabilities.state import ReferenceBands


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

        with tempfile.TemporaryDirectory() as temporary:
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
                run_erasure_branch(**kwargs)
            result = run_erasure_branch(**kwargs)
            self.assertEqual(result.outcome, "retained")
            logged = [
                __import__("json").loads(row)
                for row in (Path(temporary) / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["branch_step"] for row in logged], [1_000.0, 2_000.0])


if __name__ == "__main__":
    unittest.main()
