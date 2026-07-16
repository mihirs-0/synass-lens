import math
import tempfile
import unittest
from pathlib import Path

from pinned_capabilities.config import MBCExperimentConfig, MetricConfig
from pinned_capabilities.experiment import MBCExperiment
from pinned_capabilities.gate0e_null import (
    crossing_rate,
    radius_at,
    radius_table,
    refresh_data_seed,
    refresh_moments,
)
from pinned_capabilities.snapshot import save_snapshot


def table(entries):
    return [
        {"learning_rate": rate, "radius": radius, "certified": certified}
        for rate, radius, certified in entries
    ]


class RadiusTableTests(unittest.TestCase):
    def test_radius_table_sorts_and_extracts(self) -> None:
        scan = {
            "measurements": [
                {
                    "learning_rate": 0.01,
                    "augmented_spectral_radius": 1.2,
                    "augmented_certified": True,
                },
                {
                    "learning_rate": 0.003,
                    "augmented_spectral_radius": 1.05,
                    "augmented_certified": True,
                },
            ]
        }
        rows = radius_table(scan)
        self.assertEqual([row["learning_rate"] for row in rows], [0.003, 0.01])

    def test_interpolation_is_exact_on_loglog_line(self) -> None:
        rows = table([(0.001, 1.0, True), (0.01, 2.0, True)])
        mid = radius_at(rows, math.sqrt(0.001 * 0.01))
        self.assertAlmostEqual(mid, math.sqrt(2.0), places=12)

    def test_interpolation_rejects_outside_grid(self) -> None:
        rows = table([(0.001, 1.0, True), (0.01, 2.0, True)])
        with self.assertRaises(ValueError):
            radius_at(rows, 0.02)

    def test_crossing_first_upward(self) -> None:
        rows = table(
            [(0.003, 0.9, True), (0.01, 1.1, True), (0.03, 1.5, True)]
        )
        result = crossing_rate(rows, 1.0)
        self.assertEqual(result["kind"], "crossing")
        self.assertTrue(0.003 < result["predicted_eta50"] < 0.01)

    def test_crossing_skips_uncertified_segment(self) -> None:
        rows = table(
            [(0.003, 0.9, True), (0.01, 1.1, False), (0.03, 1.5, True)]
        )
        result = crossing_rate(rows, 1.0)
        self.assertEqual(result["kind"], "crossing")
        self.assertTrue(0.003 < result["predicted_eta50"] < 0.03)

    def test_left_and_right_censoring(self) -> None:
        high = table([(0.003, 1.2, True), (0.01, 1.5, True)])
        self.assertEqual(crossing_rate(high, 1.0)["kind"], "left_censored")
        low = table([(0.003, 0.8, True), (0.01, 0.9, True)])
        self.assertEqual(crossing_rate(low, 1.0)["kind"], "right_censored")

    def test_multiple_crossings_reported(self) -> None:
        rows = table(
            [
                (0.003, 0.9, True),
                (0.01, 1.1, True),
                (0.02, 0.95, True),
                (0.05, 1.3, True),
            ]
        )
        result = crossing_rate(rows, 1.0)
        self.assertEqual(len(result["all_crossings"]), 2)
        self.assertTrue(result["predicted_eta50"] < 0.01)


class RefreshTests(unittest.TestCase):
    def test_refresh_produces_bound_reusable_snapshot(self) -> None:
        config = MBCExperimentConfig(
            seed=6,
            n_unique_b=24,
            k=3,
            n_layers=1,
            n_heads=2,
            d_model=32,
            d_head=16,
            d_mlp=64,
            batch_size=16,
            probe_b_count=8,
            device="cpu",
        )
        metric = MetricConfig(eval_every=50, quartets_per_probe=32)
        with tempfile.TemporaryDirectory() as workspace:
            workspace = Path(workspace)
            experiment = MBCExperiment(config, metric)
            experiment.advance(5)
            base = workspace / "base.pt"
            save_snapshot(
                base,
                model=experiment.model,
                optimizer=experiment.optimizer,
                stream=experiment.stream,
                step=experiment.step,
                metadata={"kind": "test"},
            )
            record = refresh_moments(
                config, metric, base, workspace / "refresh", refresh_steps=20
            )
            self.assertEqual(record["controls"]["refresh_steps"], 20)
            self.assertEqual(
                record["controls"]["refresh_data_seed"],
                refresh_data_seed(6, 16),
            )
            self.assertGreater(record["theta_relative_drift"], 0.0)
            self.assertTrue((workspace / "refresh/refreshed_snapshot.pt").exists())
            again = refresh_moments(
                config, metric, base, workspace / "refresh", refresh_steps=20
            )
            self.assertEqual(again, record)

    def test_refresh_seed_depends_on_batch(self) -> None:
        self.assertNotEqual(refresh_data_seed(0, 32), refresh_data_seed(0, 512))


if __name__ == "__main__":
    unittest.main()
