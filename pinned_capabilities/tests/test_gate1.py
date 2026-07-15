import tempfile
import unittest
from pathlib import Path

import torch

from pinned_capabilities.batch_stream import DeterministicBatchStream
from pinned_capabilities.gate1 import (
    add_norm_matched_gaussian_noise,
    equilibration_slopes,
    geometric_levels,
    is_equilibrated,
)
from pinned_capabilities.snapshot import load_crossed_snapshot, save_snapshot


class DwellTests(unittest.TestCase):
    def test_geometric_grid_and_two_probe_equilibration(self) -> None:
        levels = geometric_levels(0.1, 0.01, 12)
        self.assertEqual(len(levels), 13)
        self.assertEqual(levels[0], 0.1)
        self.assertEqual(levels[-1], 0.01)
        rows = [
            {
                "step": float(step),
                "probe_0_c_int": 2.0 + 1e-5 * step,
                "probe_1_c_int": 3.0 - 1e-5 * step,
            }
            for step in range(0, 2_001, 50)
        ]
        slopes = equilibration_slopes(rows)
        self.assertAlmostEqual(slopes[0], 0.01)
        self.assertAlmostEqual(slopes[1], -0.01)
        self.assertTrue(is_equilibrated(rows, plateau_sd=0.02))
        rows[-1]["probe_1_c_int"] += 10
        self.assertFalse(is_equilibrated(rows, plateau_sd=0.02))


class MemorySurgeryTests(unittest.TestCase):
    @staticmethod
    def _make(seed):
        torch.manual_seed(seed)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        stream = DeterministicBatchStream(9, 3, seed=seed)
        x = torch.ones(3, 2)
        y = torch.zeros(3, 1)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
        stream.next_indices()
        return model, optimizer, stream

    def test_crossed_snapshot_uses_declared_sources(self) -> None:
        weights_model, weights_opt, weights_stream = self._make(3)
        state_model, state_opt, state_stream = self._make(5)
        expected_weights = [value.detach().clone() for value in weights_model.parameters()]
        expected_moments = [
            state_opt.state[parameter]["exp_avg"].detach().clone()
            for parameter in state_model.parameters()
        ]
        expected_cursor = state_stream.cursor
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "weights.pt"
            state_path = Path(directory) / "state.pt"
            save_snapshot(
                weights_path,
                model=weights_model,
                optimizer=weights_opt,
                stream=weights_stream,
                step=1,
                metadata={"state": "expressed"},
            )
            save_snapshot(
                state_path,
                model=state_model,
                optimizer=state_opt,
                stream=state_stream,
                step=2,
                metadata={"state": "suppressed"},
            )
            crossed_model, crossed_opt, crossed_stream = self._make(11)
            result = load_crossed_snapshot(
                weights_path=weights_path,
                optimizer_path=state_path,
                model=crossed_model,
                optimizer=crossed_opt,
                stream=crossed_stream,
            )
        for expected, actual in zip(expected_weights, crossed_model.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for expected, parameter in zip(expected_moments, crossed_model.parameters()):
            torch.testing.assert_close(expected, crossed_opt.state[parameter]["exp_avg"], rtol=0, atol=0)
        self.assertEqual(crossed_stream.cursor, expected_cursor)
        self.assertEqual(result["weights_metadata"]["state"], "expressed")
        self.assertEqual(result["optimizer_metadata"]["state"], "suppressed")

    def test_gaussian_control_has_requested_global_norm(self) -> None:
        model = torch.nn.Linear(3, 2)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        add_norm_matched_gaussian_noise(model, target_norm=0.25, seed=19)
        norm = sum(
            (parameter - old).square().sum()
            for parameter, old in zip(model.parameters(), before)
        ).sqrt()
        self.assertAlmostEqual(float(norm.item()), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
