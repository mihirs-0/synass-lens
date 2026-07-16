import math
import unittest

from pinned_capabilities.gate0e_verdict import (
    arrhenius_secondary,
    batch_shift_analysis,
    gate0e_verdict,
    seed_ratio_check,
    wd_mediation_flag,
)


def build_cell(rate, labels, taus=None, diverged=0):
    erased = sum(labels)
    retained = len(labels) - erased
    taus = taus or []
    classified = len(labels)
    return {
        "learning_rate": rate,
        "counts": {
            "erased": erased,
            "retained": retained,
            "unresolved": 0,
            "diverged": diverged,
        },
        "unstable": diverged * 2 >= (classified + diverged) and diverged > 0,
        "primary_fraction": erased / classified if classified else None,
        "sensitivity_fraction": erased / classified if classified else None,
        "median_observed_tau": sorted(taus)[len(taus) // 2] if taus else None,
        "stream_labels": list(labels),
    }


def build_curve(cells, eta50=None, batch=128):
    return {
        "kind": "gate0e_escape_curve",
        "batch_size": batch,
        "cells": cells,
        "statistics": {"eta50": eta50},
    }


def crossing(value):
    return {"prediction": {"kind": "crossing", "predicted_eta50": value}}


RATES = (0.008, 0.02)


def base_batch_curve(eta50=None):
    return build_curve(
        [
            build_cell(0.008, [1, 1, 0, 0, 0, 0, 0, 0]),
            build_cell(0.02, [1, 1, 1, 1, 1, 1, 0, 0]),
        ],
        eta50=eta50,
    )


def shifted_up_curve():
    # Boundary moved up: erasure fractions collapse at the fixed rates, to the
    # point of saturation at the lower rate (eta50 unbracketed is tolerated).
    return build_curve(
        [
            build_cell(0.008, [0, 0, 0, 0, 0, 0]),
            build_cell(0.02, [1, 0, 0, 0, 0, 0]),
        ],
        batch=32,
    )


def shifted_down_curve():
    return build_curve(
        [
            build_cell(0.008, [1, 1, 1, 1, 1, 0]),
            build_cell(0.02, [1, 1, 1, 1, 1, 1]),
        ],
        batch=512,
    )


def matched_predictions():
    return {
        "kind": "gate0e_null_predictions",
        "c_star": 1.07,
        "dev_eta50": 0.012,
        "predictions": {
            "0:128": crossing(0.012),
            "1:128": crossing(0.012),
            "2:128": crossing(0.012),
            "0:32": crossing(0.03),
            "1:32": crossing(0.03),
            "0:512": crossing(0.005),
            "1:512": crossing(0.005),
        },
    }


def matching_batch_curves():
    return {
        "0:128": base_batch_curve(),
        "1:128": base_batch_curve(),
        "0:32": shifted_up_curve(),
        "1:32": shifted_up_curve(),
        "0:512": shifted_down_curve(),
        "1:512": shifted_down_curve(),
    }


def opposite_batch_curves():
    up, down = shifted_up_curve(), shifted_down_curve()
    up["batch_size"], down["batch_size"] = 512, 32
    return {
        "0:128": base_batch_curve(),
        "1:128": base_batch_curve(),
        "0:32": down,
        "1:32": down,
        "0:512": up,
        "1:512": up,
    }


class SeedRatioTests(unittest.TestCase):
    def test_within_and_miss_bands(self) -> None:
        pred = {"kind": "crossing", "predicted_eta50": 0.012}
        self.assertTrue(seed_ratio_check(0.013, pred)["within_match"])
        middle = seed_ratio_check(0.02, pred)
        self.assertFalse(middle["within_match"])
        self.assertFalse(middle["off_miss"])
        self.assertTrue(seed_ratio_check(0.03, pred)["off_miss"])

    def test_censored_predictions(self) -> None:
        right = {"kind": "right_censored", "bound": 0.05}
        self.assertTrue(seed_ratio_check(0.02, right)["off_miss"])
        self.assertFalse(seed_ratio_check(0.04, right)["off_miss"])
        left = {"kind": "left_censored", "bound": 0.003}
        self.assertTrue(seed_ratio_check(0.007, left)["off_miss"])

    def test_undefined_observation_supports_nothing(self) -> None:
        check = seed_ratio_check(None, {"kind": "crossing", "predicted_eta50": 0.01})
        self.assertFalse(check["within_match"])
        self.assertFalse(check["off_miss"])


class BatchShiftTests(unittest.TestCase):
    def test_matching_shifts_are_detected(self) -> None:
        analysis = batch_shift_analysis(
            matching_batch_curves(), RATES, matched_predictions()["predictions"],
            replicates=400,
        )
        self.assertEqual(analysis["predicted_eta50_sign"], {"32": 1, "512": -1})
        self.assertTrue(analysis["matches_prediction"])
        self.assertFalse(analysis["opposite_of_prediction"])
        self.assertLess(analysis["observed"]["32"]["mean_logodds_shift"], 0)
        self.assertGreater(analysis["observed"]["512"]["mean_logodds_shift"], 0)

    def test_opposite_shifts_are_detected(self) -> None:
        analysis = batch_shift_analysis(
            opposite_batch_curves(), RATES, matched_predictions()["predictions"],
            replicates=400,
        )
        self.assertFalse(analysis["matches_prediction"])
        self.assertTrue(analysis["opposite_of_prediction"])

    def test_censored_prediction_disables_discriminator(self) -> None:
        predictions = matched_predictions()["predictions"]
        predictions["0:512"] = {
            "prediction": {"kind": "right_censored", "bound": 0.05}
        }
        analysis = batch_shift_analysis(
            matching_batch_curves(), RATES, predictions, replicates=100
        )
        self.assertIsNone(analysis["matches_prediction"])
        self.assertIsNone(analysis["opposite_of_prediction"])


class VerdictTests(unittest.TestCase):
    def gate_curves(self, eta50s):
        curves = {}
        for seed, eta50 in zip((0, 1, 2), eta50s):
            curves[seed] = build_curve(
                [
                    build_cell(0.008, [1, 0, 0, 0, 0, 0, 0, 0], taus=[3000]),
                    build_cell(0.0125, [1, 1, 1, 0, 0, 0, 0, 0], taus=[2000, 4000, 9000]),
                    build_cell(0.02, [1, 1, 1, 1, 1, 1, 0, 0], taus=[900, 1200, 2000, 3000, 5000, 8000]),
                    build_cell(0.032, [1, 1, 1, 1, 1, 1, 1, 0], taus=[500, 700, 900, 1200, 1500, 2200, 4000]),
                ],
                eta50=eta50,
            )
        return curves

    def test_null_wins_requires_everything(self) -> None:
        report = gate0e_verdict(
            matched_predictions(),
            self.gate_curves([0.011, 0.013, 0.0115]),
            matching_batch_curves(),
            RATES,
            replicates=300,
        )
        self.assertEqual(report["outcome"], "null_wins")
        self.assertEqual(report["action"], "stop_program")

    def test_two_misses_lose_the_null(self) -> None:
        report = gate0e_verdict(
            matched_predictions(),
            self.gate_curves([0.028, 0.03, 0.013]),
            matching_batch_curves(),
            RATES,
            replicates=300,
        )
        self.assertEqual(report["outcome"], "null_loses")
        self.assertEqual(report["claim_status"], "earned")
        self.assertEqual(report["miss_count"], 2)

    def test_opposite_batch_sign_loses_the_null(self) -> None:
        report = gate0e_verdict(
            matched_predictions(),
            self.gate_curves([0.011, 0.013, 0.0115]),
            opposite_batch_curves(),
            RATES,
            replicates=300,
        )
        self.assertEqual(report["outcome"], "null_loses")

    def test_middle_band_is_ambiguous(self) -> None:
        report = gate0e_verdict(
            matched_predictions(),
            self.gate_curves([0.011, 0.02, 0.0115]),
            matching_batch_curves(),
            RATES,
            replicates=300,
        )
        self.assertEqual(report["outcome"], "ambiguous")
        self.assertEqual(report["claim_status"], "demoted")

    def test_arrhenius_secondary_reports(self) -> None:
        curves = self.gate_curves([0.011, 0.013, 0.0115])
        report = arrhenius_secondary(curves)
        self.assertGreaterEqual(report["cells"], 5)
        self.assertIn("noise_activated_regime", report)


class WdFlagTests(unittest.TestCase):
    def primary(self):
        return build_curve(
            [
                build_cell(0.005, [0] * 8),
                build_cell(0.008, [1, 1, 1, 1, 0, 0, 0, 0]),
                build_cell(0.0125, [1] * 6 + [0, 0]),
                build_cell(0.02, [1] * 8),
            ]
        )

    def test_flag_fires_when_lambda0_retains_everywhere(self) -> None:
        wd = build_curve(
            [
                build_cell(0.005, [0, 0, 0, 0]),
                build_cell(0.008, [0, 0, 0, 0]),
                build_cell(0.0125, [0, 0, 0, 0]),
                build_cell(0.02, [0, 0, 0, 0]),
            ]
        )
        flag = wd_mediation_flag(wd, self.primary())
        self.assertTrue(flag["erasure_is_wd_mediated"])

    def test_flag_stays_off_when_lambda0_erases(self) -> None:
        wd = build_curve(
            [
                build_cell(0.005, [0, 0, 0, 0]),
                build_cell(0.008, [1, 0, 0, 0]),
                build_cell(0.0125, [1, 1, 0, 0]),
                build_cell(0.02, [1, 1, 1, 0]),
            ]
        )
        flag = wd_mediation_flag(wd, self.primary())
        self.assertFalse(flag["erasure_is_wd_mediated"])


if __name__ == "__main__":
    unittest.main()
