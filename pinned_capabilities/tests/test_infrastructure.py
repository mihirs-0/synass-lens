import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinned_capabilities.batch_stream import DeterministicBatchStream
from pinned_capabilities.parameter_groups import (
    GROUP_NAMES,
    grouped_named_parameters,
    optimizer_groups,
    set_learning_rates,
)
from pinned_capabilities.snapshot import load_snapshot, save_snapshot


class ToyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(3)
        self.attn = torch.nn.Linear(3, 3)
        self.mlp = torch.nn.Linear(3, 3)


class ToyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 3)
        self.pos_embed = torch.nn.Embedding(8, 3)
        self.blocks = torch.nn.ModuleList([ToyBlock()])
        self.ln_final = torch.nn.LayerNorm(3)
        self.unembed = torch.nn.Linear(3, 8)


class BatchStreamTests(unittest.TestCase):
    def test_resume_across_epoch_boundary(self) -> None:
        stream = DeterministicBatchStream(7, 3, seed=91)
        stream.next_indices()
        stream.next_indices()
        saved = copy.deepcopy(stream.state_dict())
        expected = [stream.next_indices(), stream.next_indices(), stream.next_indices()]

        resumed = DeterministicBatchStream(7, 3, seed=0)
        resumed.load_state_dict(saved)
        actual = [resumed.next_indices(), resumed.next_indices(), resumed.next_indices()]
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(stream.epoch, resumed.epoch)
        self.assertEqual(stream.cursor, resumed.cursor)


class ParameterGroupTests(unittest.TestCase):
    def test_groups_are_exhaustive_disjoint_and_mutable(self) -> None:
        model = ToyTransformer()
        named = grouped_named_parameters(model)
        self.assertEqual(tuple(named), GROUP_NAMES)
        assigned = [name for values in named.values() for name, _ in values]
        self.assertEqual(sorted(assigned), sorted(name for name, _ in model.named_parameters()))
        self.assertEqual(len(assigned), len(set(assigned)))

        groups = optimizer_groups(model, 1e-3, {"mlp": 0.1, "attention": 2.0})
        optimizer = torch.optim.AdamW(groups)
        rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(rates["mlp"], 1e-4)
        self.assertEqual(rates["attention"], 2e-3)
        set_learning_rates(optimizer, 3e-3, {"mlp": 0.0})
        rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(rates["mlp"], 0.0)
        self.assertEqual(rates["attention"], 3e-3)


class SnapshotTests(unittest.TestCase):
    @staticmethod
    def _update(model, optimizer, stream):
        indices = stream.next_indices()
        x = indices.float().unsqueeze(1) / 10
        y = 2 * x - 0.25
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        return indices, loss.detach()

    def test_snapshot_reproduces_next_update_and_rng(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        stream = DeterministicBatchStream(11, 4, seed=13)
        self._update(model, optimizer, stream)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.pt"
            save_snapshot(path, model=model, optimizer=optimizer, stream=stream, step=1)
            expected_random = (random.random(), np.random.rand(), torch.rand(()))
            expected_indices, expected_loss = self._update(model, optimizer, stream)
            expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]

            restored_model = torch.nn.Linear(1, 1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=99)
            restored_stream = DeterministicBatchStream(11, 4, seed=999)
            result = load_snapshot(
                path,
                model=restored_model,
                optimizer=restored_optimizer,
                stream=restored_stream,
            )
            actual_random = (random.random(), np.random.rand(), torch.rand(()))
            actual_indices, actual_loss = self._update(
                restored_model, restored_optimizer, restored_stream
            )

        self.assertEqual(result["step"], 1)
        self.assertEqual(expected_random[:2], actual_random[:2])
        torch.testing.assert_close(expected_random[2], actual_random[2], rtol=0, atol=0)
        torch.testing.assert_close(expected_indices, actual_indices, rtol=0, atol=0)
        torch.testing.assert_close(expected_loss, actual_loss, rtol=0, atol=0)
        for expected, actual in zip(expected_parameters, restored_model.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
