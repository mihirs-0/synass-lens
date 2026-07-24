import math
import unittest

from pinned_capabilities.gate0_autopsy import (
    BASE_HOLD_STEPS,
    ESCALATED_HOLD_STEPS,
    classify_cell,
    escalated_hold_steps,
    q_star_bounds,
    select_branches,
    slow_erasure_crossing_step,
)
from pinned_capabilities.state import ReferenceBands, StateThresholds

HOLD = 8_000

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


def build_rows(c_int_fn, ce_fn, c_int_fn_1=None, ce_fn_1=None):
    c_int_fn_1 = c_int_fn_1 or c_int_fn
    ce_fn_1 = ce_fn_1 or ce_fn
    rows = []
    for step in range(50, HOLD + 1, 50):
        rows.append(
            {
                "branch_step": float(step),
                "probe_0_c_int": float(c_int_fn(step)),
                "probe_1_c_int": float(c_int_fn_1(step)),
                "probe_0_full_vocab_ce": float(ce_fn(step)),
                "probe_1_full_vocab_ce": float(ce_fn_1(step)),
            }
        )
    return rows


class BandTests(unittest.TestCase):
    def test_q_star_band_uses_relative_floor(self) -> None:
        low, high = q_star_bounds(BANDS, THRESHOLDS)
        self.assertAlmostEqual(high - low, 2 * 0.02 * 3.5)


class ClassifierTests(unittest.TestCase):
    def test_deadline_artifact_settles_late_but_settles(self) -> None:
        rows = build_rows(
            lambda step: 5.0 if step < 5_500 else 0.0, lambda step: 3.5
        )
        cell = classify_cell(0.012, rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(cell.label, "deadline_artifact")

    def test_near_divergence_precedes_deadline_artifact(self) -> None:
        rows = build_rows(
            lambda step: 5.0 if step < 5_500 else 0.0,
            lambda step: 100.0 if step == 300 else 3.5,
        )
        cell = classify_cell(0.012, rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(cell.label, "near_divergent")
        self.assertEqual(
            cell.diagnostics["near_divergent"]["probe_0_first_divergent_row"][
                "branch_step"
            ],
            300.0,
        )

    def test_single_probe_divergence_does_not_fire(self) -> None:
        rows = build_rows(
            lambda step: 5.0 if step < 5_500 else 0.0,
            lambda step: 100.0 if step == 300 else 3.5,
            ce_fn_1=lambda step: 3.5,
        )
        cell = classify_cell(0.012, rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(cell.label, "deadline_artifact")

    def test_slow_erasure_descending_above_band(self) -> None:
        rows = build_rows(lambda step: 10.0 - step * (9.5 / HOLD), lambda step: 3.5)
        cell = classify_cell(0.05, rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(cell.label, "slow_erasure")

    def test_intermediate_flat_level_between_bands(self) -> None:
        rows = build_rows(lambda step: 3.0, lambda step: 3.5)
        cell = classify_cell(0.006, rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(cell.label, "intermediate")

    def test_rising_final_quarter_is_stochastic(self) -> None:
        def c_int(step):
            if step <= 6_000:
                return 5.0
            return 2.0 + (step - 6_000) * (12.0 / 2_000)

        rows = build_rows(c_int, lambda step: 3.5)
        cell = classify_cell(0.006, rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(cell.label, "stochastic")

    def test_flat_level_with_ce_outside_band_is_stochastic(self) -> None:
        rows = build_rows(lambda step: 3.0, lambda step: 4.0)
        cell = classify_cell(0.006, rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(cell.label, "stochastic")

    def test_both_probes_required_for_every_rule(self) -> None:
        rows = build_rows(
            lambda step: 5.0 if step < 5_500 else 0.0,
            lambda step: 3.5,
            c_int_fn_1=lambda step: 5.0 if step < 6_500 else 0.0,
        )
        cell = classify_cell(0.012, rows, BANDS, THRESHOLDS, HOLD)
        self.assertEqual(cell.label, "stochastic")


class EscalationTests(unittest.TestCase):
    def test_shallow_slope_extrapolates_past_escalation_step(self) -> None:
        rows = build_rows(lambda step: 3.0 - (step - 7_000) * 1e-4, lambda step: 3.5)
        crossing = slow_erasure_crossing_step(rows, BANDS, THRESHOLDS, HOLD)
        self.assertGreater(crossing, 14_000)
        self.assertEqual(escalated_hold_steps([crossing]), ESCALATED_HOLD_STEPS)

    def test_steep_slope_stays_at_base_hold(self) -> None:
        rows = build_rows(lambda step: 10.0 - step * 1.2e-3, lambda step: 3.5)
        crossing = slow_erasure_crossing_step(rows, BANDS, THRESHOLDS, HOLD)
        self.assertLess(crossing, 14_000)
        self.assertEqual(escalated_hold_steps([crossing]), BASE_HOLD_STEPS)

    def test_non_negative_slope_never_reaches_band(self) -> None:
        rows = build_rows(lambda step: 3.0, lambda step: 3.5)
        crossing = slow_erasure_crossing_step(rows, BANDS, THRESHOLDS, HOLD)
        self.assertTrue(math.isinf(crossing))
        self.assertEqual(escalated_hold_steps([crossing]), ESCALATED_HOLD_STEPS)

    def test_no_slow_erasure_cells_stays_at_base_hold(self) -> None:
        self.assertEqual(escalated_hold_steps([]), BASE_HOLD_STEPS)


class BranchSelectionTests(unittest.TestCase):
    def test_any_stochastic_selects_a(self) -> None:
        activation = select_branches(
            {0.006: "stochastic", 0.012: "deadline_artifact", 0.05: "slow_erasure"}
        )
        self.assertEqual(activation.primary_branch, "A")
        self.assertFalse(activation.include_branch_c)
        self.assertIsNone(activation.grid_cap)

    def test_all_deterministic_selects_b(self) -> None:
        activation = select_branches(
            {0.006: "deadline_artifact", 0.012: "slow_erasure", 0.05: "slow_erasure"}
        )
        self.assertEqual(activation.primary_branch, "B")

    def test_all_intermediate_runs_c_then_a(self) -> None:
        activation = select_branches(
            {0.006: "intermediate", 0.012: "intermediate", 0.05: "intermediate"}
        )
        self.assertEqual(activation.primary_branch, "A")
        self.assertTrue(activation.include_branch_c)

    def test_mixed_intermediate_completion_selects_a_plus_c(self) -> None:
        activation = select_branches(
            {0.006: "intermediate", 0.012: "slow_erasure", 0.05: "deadline_artifact"}
        )
        self.assertEqual(activation.primary_branch, "A")
        self.assertTrue(activation.include_branch_c)

    def test_near_divergent_top_caps_grid_and_selects_no_branch(self) -> None:
        activation = select_branches(
            {0.006: "deadline_artifact", 0.012: "slow_erasure", 0.05: "near_divergent"}
        )
        self.assertEqual(activation.primary_branch, "B")
        self.assertEqual(activation.grid_cap, 0.032)

    def test_unknown_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_branches({0.006: "mystery"})


if __name__ == "__main__":
    unittest.main()
