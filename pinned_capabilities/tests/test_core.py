from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pinned_capabilities.config import ProtocolConfig
from pinned_capabilities.manifest import content_hash, freeze_manifest
from pinned_capabilities.metrics import evaluate_counterfactual_metrics, sample_quartets
from pinned_capabilities.state import (
    ReferenceBands,
    StateThresholds,
    first_transition_step,
    is_expressed,
    is_jointly_suppressed,
    is_suppressed,
)


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.n_b, self.k, self.vocab = 4, 3, 32
        self.answers = torch.arange(self.n_b * self.k).reshape(self.n_b, self.k)
        self.quartets = sample_quartets(self.n_b, self.k, 512, seed=7)

    def test_additive_shortcuts_cancel(self) -> None:
        logits = torch.zeros(self.n_b, self.k, self.vocab)
        b_term = torch.randn(self.n_b, self.vocab, generator=torch.Generator().manual_seed(1))
        z_term = torch.randn(self.k, self.vocab, generator=torch.Generator().manual_seed(2))
        logits = b_term[:, None, :] + z_term[None, :, :]
        metrics = evaluate_counterfactual_metrics(logits, self.answers, self.quartets)
        self.assertLess(abs(float(metrics.c_int.mean())), 1e-6)

    def test_joint_lookup_has_positive_interaction(self) -> None:
        logits = torch.zeros(self.n_b, self.k, self.vocab)
        for b in range(self.n_b):
            for z in range(self.k):
                logits[b, z, self.answers[b, z]] = 8.0
        metrics = evaluate_counterfactual_metrics(logits, self.answers, self.quartets)
        self.assertGreater(float(metrics.c_int.mean()), 5.0)
        self.assertGreater(float(metrics.delta_z.mean()), 5.0)
        self.assertEqual(float(metrics.candidate_exact_match.mean()), 1.0)

    def test_constant_machine_is_zero(self) -> None:
        logits = torch.zeros(self.n_b, self.k, self.vocab)
        metrics = evaluate_counterfactual_metrics(logits, self.answers, self.quartets)
        self.assertAlmostEqual(float(metrics.c_int.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(metrics.delta_z.mean()), 0.0, places=6)


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = ReferenceBands(0.0, 0.01, 10.0, 0.1, 0.01)
        self.thresholds = StateThresholds()

    def test_state_labels_and_transition(self) -> None:
        rows = [
            {"step": step, "c_int": 0.0, "exact_match": 0.1}
            for step in range(0, 2_001, 50)
        ]
        self.assertTrue(is_suppressed(rows, self.reference, self.thresholds))
        flat_rows = [{**row, "full_vocab_ce": 2.0} for row in rows]
        flat_reference = ReferenceBands(0.0, 1.0, 10.0, 0.1, 0.02, 2.0, 0.1)
        self.assertTrue(is_jointly_suppressed(flat_rows, flat_reference, self.thresholds))
        flat_rows[-1]["full_vocab_ce"] = 20.0
        self.assertFalse(is_jointly_suppressed(flat_rows, flat_reference, self.thresholds))
        self.assertTrue(is_expressed(6.0, 0.9, self.reference, self.thresholds))
        self.assertFalse(is_expressed(6.0, 0.89, self.reference, self.thresholds))
        transition_rows = rows + [
            {"step": step, "c_int": 3.0, "exact_match": 0.5}
            for step in range(2_050, 3_101, 50)
        ]
        self.assertEqual(first_transition_step(transition_rows, self.reference, self.thresholds), 2_050)


class ManifestTests(unittest.TestCase):
    def test_hash_is_order_invariant_and_manifest_is_immutable(self) -> None:
        self.assertEqual(content_hash({"a": 1, "b": 2}), content_hash({"b": 2, "a": 1}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            first = freeze_manifest(ProtocolConfig(), path)
            second = freeze_manifest(ProtocolConfig(), path)
            self.assertEqual(first, second)
            with self.assertRaises(FileExistsError):
                freeze_manifest({"different": True}, path)
            self.assertEqual(json.loads(path.read_text())["config_sha256"], first["config_sha256"])


if __name__ == "__main__":
    unittest.main()
