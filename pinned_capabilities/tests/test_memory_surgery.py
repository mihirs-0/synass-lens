import tempfile
import unittest
from pathlib import Path

import torch

from pinned_capabilities.batch_stream import DeterministicBatchStream
from pinned_capabilities.memory_surgery_run import _weight_distance
from pinned_capabilities.snapshot import save_snapshot


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


if __name__ == "__main__":
    unittest.main()
