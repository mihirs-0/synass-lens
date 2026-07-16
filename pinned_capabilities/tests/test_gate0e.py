import math
import unittest

from pinned_capabilities.gate0e_control import (
    ControlConfig,
    critical_points,
    run_positive_control,
)
from pinned_capabilities.gate0e_escape import (
    aggregate_cell,
    classify_stream,
    curve_statistics,
    stream_data_seed,
    StreamResult,
)
from pinned_capabilities.gate0e_stats import (
    RateOutcomes,
    arrhenius_fit,
    bootstrap_eta50_ci,
    curve_valid,
    logistic_fit,
    monotonicity_flags,
    shifted_grid,
)
from pinned_capabilities.state import ReferenceBands, StateThresholds

BANDS = ReferenceBands(
    plateau_mean=0.0,
    plateau_sd=0.01,
    solved_c_int=16.0,
    chance_em_mean=0.0,
    chance_em_sd=0.0,
    q_star_loss_mean=3.5,
    q_star_loss_sd=0.001,
)
THRESHOLDS = StateThresholds()
HOLD = 16_000


def build_rows(c_int_fn, ce_fn, hold=HOLD, expressed_fn=None):
    rows = []
    for step in range(50, hold + 1, 50):
        c = float(c_int_fn(step))
        ce = float(ce_fn(step))
        rows.append(
            {
                "branch_step": float(step),
                "c_int": c,
                "full_vocab_ce": ce,
                "exact_match": 1.0 if (expressed_fn and expressed_fn(step)) else 0.0,
                "delta_z": 18.0 if (expressed_fn and expressed_fn(step)) else 0.0,
                "probe_0_c_int": c,
                "probe_1_c_int": c,
                "probe_0_full_vocab_ce": ce,
                "probe_1_full_vocab_ce": ce,
            }
        )
    return rows


class StatsTests(unittest.TestCase):
    def cells(self):
        return [
            RateOutcomes(0.003, 0, 8, 0, 0),
            RateOutcomes(0.008, 2, 5, 1, 0),
            RateOutcomes(0.02, 6, 1, 1, 0),
            RateOutcomes(0.05, 8, 0, 0, 0),
        ]

    def test_logistic_fit_recovers_midpoint(self) -> None:
        fit = logistic_fit(self.cells())
        self.assertTrue(fit.slope > 0)
        self.assertIsNotNone(fit.eta50)
        self.assertTrue(0.008 < fit.eta50 < 0.05)

    def test_bootstrap_ci_brackets_eta50(self) -> None:
        cells = self.cells()
        fit = logistic_fit(cells)
        labels = {
            0.003: [0] * 8,
            0.008: [1, 1, 0, 0, 0, 0, 0, 0],
            0.02: [1, 1, 1, 1, 1, 1, 0, 0],
            0.05: [1] * 8,
        }
        ci = bootstrap_eta50_ci(cells, labels, replicates=200, seed=7)
        self.assertIsNotNone(ci.lower)
        self.assertLess(ci.lower, fit.eta50)
        self.assertGreater(ci.upper, fit.eta50)

    def test_unstable_rate_definition(self) -> None:
        self.assertTrue(RateOutcomes(0.05, 1, 1, 2, 4).unstable)
        self.assertFalse(RateOutcomes(0.05, 3, 2, 0, 3).unstable)

    def test_primary_and_sensitivity_conventions(self) -> None:
        cell = RateOutcomes(0.02, 3, 2, 3, 0)
        self.assertAlmostEqual(cell.primary_fraction(), 3 / 8)
        self.assertAlmostEqual(cell.sensitivity_fraction(), 6 / 8)

    def test_curve_validity_rule(self) -> None:
        good = [RateOutcomes(rate, 4, 4, 0, 0) for rate in (0.003, 0.008, 0.02, 0.05)]
        self.assertTrue(curve_valid(good))
        few = good[:3]
        self.assertFalse(curve_valid(few))
        thin = good[:3] + [RateOutcomes(0.1, 3, 2, 0, 3)]
        self.assertFalse(curve_valid(thin))

    def test_monotonicity_flags_fire_on_real_decrease(self) -> None:
        cells = [
            RateOutcomes(0.003, 8, 0, 0, 0),
            RateOutcomes(0.008, 0, 8, 0, 0),
        ]
        self.assertEqual(len(monotonicity_flags(cells)), 1)
        increasing = [
            RateOutcomes(0.003, 0, 8, 0, 0),
            RateOutcomes(0.008, 8, 0, 0, 0),
        ]
        self.assertEqual(monotonicity_flags(increasing), [])

    def test_arrhenius_fit_on_exact_line(self) -> None:
        cells = [
            RateOutcomes(0.005, 4, 4, 0, 0),
            RateOutcomes(0.01, 4, 4, 0, 0),
            RateOutcomes(0.02, 4, 4, 0, 0),
        ]
        taus = {rate: math.exp(0.03 / rate + 1.0) for rate in (0.005, 0.01, 0.02)}
        fit = arrhenius_fit(cells, taus)
        self.assertAlmostEqual(fit.slope, 0.03, places=6)
        self.assertGreater(fit.r_squared, 0.999)

    def test_grid_shift_halves_once(self) -> None:
        self.assertEqual(shifted_grid((0.01, 0.02)), [0.005, 0.01])


class ClassifyStreamTests(unittest.TestCase):
    def test_erased_tau_is_window_start_and_irreversible(self) -> None:
        def c_int(step):
            if step < 4_000:
                return 5.0
            if step <= 9_000:
                return 0.0
            return 6.0  # re-expression must not unmark the label

        rows = build_rows(c_int, lambda step: 3.5)
        verdict = classify_stream(rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(verdict["outcome"], "erased")
        self.assertEqual(verdict["tau"], 4_000)

    def test_divergence_before_suppression_wins(self) -> None:
        def ce(step):
            return 30.0 if step == 1_000 else 3.5

        rows = build_rows(lambda step: 0.0 if step >= 4_000 else 5.0, ce)
        verdict = classify_stream(rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(verdict["outcome"], "diverged")

    def test_divergence_after_erasure_stays_erased(self) -> None:
        def c_int(step):
            return 0.0 if step <= 9_000 else 8.0

        def ce(step):
            if step <= 9_000:
                return 3.5
            return 30.0 if step == 12_000 else 1.0

        rows = build_rows(c_int, ce)
        verdict = classify_stream(rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(verdict["outcome"], "erased")
        self.assertEqual(verdict["tau"], 50)

    def test_retained_requires_expressed_end(self) -> None:
        rows = build_rows(
            lambda step: 9.0,
            lambda step: 0.05,
            expressed_fn=lambda step: True,
        )
        verdict = classify_stream(rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(verdict["outcome"], "retained")

    def test_wandering_is_unresolved(self) -> None:
        rows = build_rows(lambda step: 5.0, lambda step: 2.0)
        verdict = classify_stream(rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(verdict["outcome"], "unresolved")

    def test_interrupted_window_never_sustains(self) -> None:
        def c_int(step):
            if 4_000 <= step < 5_500 or 5_600 <= step < 7_000:
                return 0.0
            return 5.0

        rows = build_rows(c_int, lambda step: 3.5)
        verdict = classify_stream(rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(verdict["outcome"], "unresolved")


class AggregationTests(unittest.TestCase):
    def build_stream(self, rate, index, outcome, tau=None):
        return StreamResult(
            learning_rate=rate,
            stream_index=index,
            batch_size=128,
            weight_decay=0.01,
            outcome=outcome,
            tau=tau,
            first_divergence_step=None,
            final_c_int=0.0,
            final_exact_match=0.0,
            final_delta_z=0.0,
            final_full_vocab_ce=3.58,
            hold_steps=HOLD,
            stream_seed=stream_data_seed(100, rate, index, 128),
        )

    def test_cell_aggregation_counts_and_medians(self) -> None:
        streams = [
            self.build_stream(0.02, 0, "erased", 900),
            self.build_stream(0.02, 1, "erased", 2_000),
            self.build_stream(0.02, 2, "erased", 5_000),
            self.build_stream(0.02, 3, "retained"),
            self.build_stream(0.02, 4, "unresolved"),
            self.build_stream(0.02, 5, "diverged"),
        ]
        cell = aggregate_cell(streams)
        self.assertEqual(cell["counts"], {"erased": 3, "retained": 1, "unresolved": 1, "diverged": 1})
        self.assertAlmostEqual(cell["primary_fraction"], 3 / 5)
        self.assertAlmostEqual(cell["sensitivity_fraction"], 4 / 5)
        self.assertEqual(cell["median_observed_tau"], 2_000.0)
        self.assertEqual(len(cell["stream_labels"]), 5)
        self.assertFalse(cell["unstable"])

    def test_curve_statistics_pipeline(self) -> None:
        cells = []
        design = [
            (0.003, ["retained"] * 8),
            (0.008, ["erased"] * 2 + ["retained"] * 6),
            (0.02, ["erased"] * 6 + ["unresolved"] * 2),
            (0.05, ["erased"] * 8),
        ]
        for rate, outcomes in design:
            streams = [
                self.build_stream(rate, index, outcome, tau=1_000 + 100 * index if outcome == "erased" else None)
                for index, outcome in enumerate(outcomes)
            ]
            cells.append(aggregate_cell(streams))
        statistics = curve_statistics(cells)
        self.assertTrue(statistics["curve_valid"])
        self.assertIsNotNone(statistics["eta50"])
        self.assertTrue(0.005 < statistics["eta50"] < 0.05)
        self.assertEqual(statistics["monotonicity_flags"], [])

    def test_stream_seed_derivation_is_stable_and_distinct(self) -> None:
        seeds = {
            stream_data_seed(100, rate, index, 128)
            for rate in (0.003, 0.005)
            for index in range(8)
        }
        self.assertEqual(len(seeds), 16)
        self.assertEqual(
            stream_data_seed(100, 0.003, 0, 128), stream_data_seed(100, 0.003, 0, 128)
        )


class PositiveControlTests(unittest.TestCase):
    def test_barrier_geometry_matches_quartic_roots(self) -> None:
        geometry = critical_points(1.0, 0.3)
        x = geometry["barrier_x"]
        self.assertAlmostEqual(x**4 - x + 0.3, 0.0, places=8)
        self.assertGreater(geometry["well_x"], x)
        self.assertGreater(geometry["barrier_height"], 0.0)

    def test_no_barrier_rejected(self) -> None:
        with self.assertRaises(ValueError):
            critical_points(1.0, 0.6)

    def test_small_control_runs_and_reports(self) -> None:
        config = ControlConfig(
            horizon_steps=4_000,
            streams=8,
            learning_rates=(0.008, 0.012, 0.02),
        )
        report = run_positive_control(config)
        self.assertEqual(report["kind"], "gate0e_positive_control")
        self.assertEqual(len(report["cells"]), 3)
        for cell in report["cells"].values():
            counts = cell["outcomes"]
            total = sum(counts[key] for key in ("erased", "retained", "unresolved", "diverged"))
            self.assertEqual(total, 8)


if __name__ == "__main__":
    unittest.main()
