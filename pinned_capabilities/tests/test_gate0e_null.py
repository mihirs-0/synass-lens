import math
import tempfile
import unittest
from pathlib import Path

from pinned_capabilities.config import MBCExperimentConfig, MetricConfig
from pinned_capabilities.experiment import MBCExperiment
from pinned_capabilities.gate0e_null import (
    batch_discriminator,
    crossing_rate,
    decision_block_diffusion,
    diffusion_slope,
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


def cell(ln_d, se=0.02):
    return {"ln_d": ln_d, "jackknife_se": se}


def diffusion_cells(slope, se=0.02):
    cells = {}
    for seed in (0, 1):
        for batch in (32, 128, 512):
            cells[f"{seed}:{batch}"] = cell(
                slope * (math.log(batch) - math.log(128)), se
            )
    return cells


def down_predictions():
    return {
        "0:128": 0.012, "1:128": 0.012,
        "0:32": 0.03, "1:32": 0.03,
        "0:512": 0.005, "1:512": 0.005,
    }


def one_d_decision_block(values, length=64):
    count = len(values)
    mean = sum(values) / count
    return {
        "length": length,
        "count": count,
        "block_powers": [value * value for value in values],
        "block_mean_dots": [mean * value for value in values],
        "mean_power": mean * mean,
    }


class DecisionBlockTests(unittest.TestCase):
    def test_point_and_jackknife_from_scalars(self) -> None:
        values = [1.0, 2.0] * 4
        estimate = decision_block_diffusion(one_d_decision_block(values))
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate["ln_d"], math.log(0.25 / 64), places=10)
        self.assertGreater(estimate["jackknife_se"], 0.0)
        self.assertEqual(estimate["blocks"], 8)

    def test_too_few_blocks_returns_none(self) -> None:
        self.assertIsNone(decision_block_diffusion(one_d_decision_block([1.0, 2.0])))
        self.assertIsNone(decision_block_diffusion(None))

    def test_degenerate_variance_returns_none(self) -> None:
        self.assertIsNone(decision_block_diffusion(one_d_decision_block([1.0] * 8)))


class DiscriminatorTests(unittest.TestCase):
    def test_diffusion_slope_recovers_exponent(self) -> None:
        slope = diffusion_slope({32: 4.0, 128: 1.0, 512: 0.25})
        self.assertAlmostEqual(slope, -1.0, places=10)

    def test_sgd_like_diffusion_opposing_null_is_decisive(self) -> None:
        block = batch_discriminator(down_predictions(), diffusion_cells(-1.0))
        self.assertEqual(block["noise_predicted_direction"], "up")
        self.assertEqual(block["v_conditioned_direction"], "down")
        self.assertEqual(block["status"], "decisive")
        low, high = block["slope_interval_95"]
        self.assertLess(high, -0.15)

    def test_wide_uncertainty_destroys_decisiveness(self) -> None:
        block = batch_discriminator(down_predictions(), diffusion_cells(-1.0, se=2.0))
        self.assertEqual(block["noise_predicted_direction"], "flat_or_uncertain")
        self.assertEqual(block["status"], "non_discriminating")

    def test_flat_diffusion_is_non_discriminating(self) -> None:
        block = batch_discriminator(down_predictions(), diffusion_cells(0.0))
        self.assertEqual(block["noise_predicted_direction"], "flat_or_uncertain")
        self.assertEqual(block["status"], "non_discriminating")

    def test_agreeing_directions_are_non_discriminating(self) -> None:
        block = batch_discriminator(down_predictions(), diffusion_cells(1.0))
        self.assertEqual(block["noise_predicted_direction"], "down")
        self.assertEqual(block["status"], "non_discriminating")

    def test_censored_prediction_is_non_discriminating(self) -> None:
        predictions = down_predictions()
        predictions["1:512"] = None
        block = batch_discriminator(predictions, diffusion_cells(-1.0))
        self.assertIsNone(block["v_conditioned_direction"])
        self.assertEqual(block["status"], "non_discriminating")

    def test_small_predicted_shift_fails_magnitude_floor(self) -> None:
        predictions = {
            "0:128": 0.012, "1:128": 0.012,
            "0:32": 0.0125, "1:32": 0.0125,
            "0:512": 0.0115, "1:512": 0.0115,
        }
        block = batch_discriminator(predictions, diffusion_cells(-1.0))
        self.assertIsNone(block["v_conditioned_direction"])
        self.assertEqual(block["status"], "non_discriminating")

    def test_missing_diffusion_cell_is_non_discriminating(self) -> None:
        cells = diffusion_cells(-1.0)
        cells["1:512"] = None
        block = batch_discriminator(down_predictions(), cells)
        self.assertFalse(block["cells_complete"])
        self.assertEqual(block["status"], "non_discriminating")


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
            diffusion = record["update_diffusion"]
            self.assertEqual(diffusion["steps"], 20)
            self.assertEqual(diffusion["batch_size"], 16)
            self.assertGreater(diffusion["mean_step_power"], 0.0)
            self.assertGreaterEqual(
                diffusion["mean_step_power"], diffusion["drift_power"]
            )
            self.assertGreaterEqual(diffusion["update_variance_power"], 0.0)
            self.assertAlmostEqual(
                diffusion["block_diffusion"]["1"],
                diffusion["update_variance_power"],
                places=10,
            )
            self.assertIsNotNone(diffusion["block_diffusion"]["4"])
            self.assertIsNone(diffusion["block_diffusion"]["64"])
            self.assertIsNone(diffusion["decision_block"])
            self.assertTrue((workspace / "refresh/refreshed_snapshot.pt").exists())
            again = refresh_moments(
                config, metric, base, workspace / "refresh", refresh_steps=20
            )
            self.assertEqual(again, record)

    def test_refresh_seed_depends_on_batch(self) -> None:
        self.assertNotEqual(refresh_data_seed(0, 32), refresh_data_seed(0, 512))


if __name__ == "__main__":
    unittest.main()
