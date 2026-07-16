import unittest

from pinned_capabilities.gate0e_dwell_audit import audit_cells, audit_trajectory, episodes
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


def build_rows(profile, hold=10_000):
    rows = []
    for step in range(50, hold + 1, 50):
        c, expressed = profile(step)
        rows.append(
            {
                "branch_step": float(step),
                "c_int": c,
                "full_vocab_ce": 3.5 if not expressed else 0.02,
                "exact_match": 1.0 if expressed else 0.0,
                "delta_z": 18.0 if expressed else 0.0,
                "probe_0_c_int": c,
                "probe_1_c_int": c,
                "probe_0_full_vocab_ce": 3.5 if not expressed else 0.02,
                "probe_1_full_vocab_ce": 3.5 if not expressed else 0.02,
            }
        )
    return rows


def basin_like(step):
    # expressed 0-4000, transition 4000-4500, suppressed 4500-10000
    if step <= 4_000:
        return 9.0, True
    if step <= 4_500:
        return 4.0, False
    return 0.0, False


def round_trip(step):
    # expressed, transition, suppressed dwell, transition, expressed again
    if step <= 3_000:
        return 9.0, True
    if step <= 3_500:
        return 4.0, False
    if step <= 6_000:
        return 0.0, False
    if step <= 6_500:
        return 4.0, False
    return 9.0, True


def grazing(step):
    # suppressed but pops just outside the band briefly every 1000 steps
    if step % 1_000 == 0:
        return 5.0, False
    return 0.0, False


class EpisodeTests(unittest.TestCase):
    def test_basin_like_segmentation(self) -> None:
        report = audit_trajectory(build_rows(basin_like), BANDS, THRESHOLDS)
        labels = [run["label"] for run in report["episodes"]]
        self.assertEqual(labels, ["E", "B", "S"])
        self.assertEqual(report["recrossings"], 1)
        self.assertEqual(len(report["transition_spans"]), 1)
        self.assertEqual(report["graze_spans"], [])
        self.assertEqual(
            report["sustained_entry_by_window"]["2000"], 4_550
        )

    def test_grazing_counts_excursions_not_transitions(self) -> None:
        report = audit_trajectory(build_rows(grazing), BANDS, THRESHOLDS)
        self.assertEqual(report["transition_spans"], [])
        self.assertGreaterEqual(len(report["graze_spans"]), 8)
        self.assertEqual(report["recrossings"], 0)

    def test_censored_monotone_trajectory_uses_lower_bound(self) -> None:
        report = audit_cells(
            {"a": build_rows(basin_like), "b": build_rows(basin_like)},
            BANDS,
            THRESHOLDS,
        )
        self.assertIsNone(report["dwell_to_transition_ratio"])
        self.assertIsNotNone(report["censored_dwell_ratio_lower_bound"])
        self.assertGreater(report["censored_dwell_ratio_lower_bound"], 5.0)

    def test_round_trip_yields_completed_dwell_ratio(self) -> None:
        report = audit_cells({"a": build_rows(round_trip)}, BANDS, THRESHOLDS)
        self.assertFalse(report["decision_bearing"])
        self.assertIsNotNone(report["dwell_to_transition_ratio"])
        self.assertGreater(report["dwell_to_transition_ratio"], 4.0)
        self.assertEqual(report["per_cell"]["a"]["recrossings"], 2)


if __name__ == "__main__":
    unittest.main()
