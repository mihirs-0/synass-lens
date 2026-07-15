import unittest

from pinned_capabilities.gate0_boundary import sustained_band_entry
from pinned_capabilities.state import ReferenceBands


class ErasureClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = ReferenceBands(plateau_mean=0.0, plateau_sd=0.1, solved_c_int=8.0,
                                        chance_em_mean=0.0, chance_em_sd=0.0,
                                        q_star_loss_mean=2.0, q_star_loss_sd=0.1)

    def test_requires_sustained_entry_and_complete_hold(self) -> None:
        rows = [
            {"branch_step": 50.0, "c_int": 2.0, "full_vocab_ce": 2.0},
            {"branch_step": 100.0, "c_int": 0.1, "full_vocab_ce": 2.0},
            {"branch_step": 150.0, "c_int": 0.5, "full_vocab_ce": 2.0},
            {"branch_step": 200.0, "c_int": 0.2, "full_vocab_ce": 2.0},
            {"branch_step": 250.0, "c_int": 0.1, "full_vocab_ce": 2.0},
        ]
        self.assertEqual(sustained_band_entry(rows, self.reference, branch_end_step=250), 200)
        self.assertIsNone(sustained_band_entry(rows[:-1], self.reference, branch_end_step=250))

    def test_rejects_loss_divergence_inside_interaction_band(self) -> None:
        rows = [
            {"branch_step": 50.0, "c_int": 0.1, "full_vocab_ce": 50.0},
            {"branch_step": 100.0, "c_int": 0.1, "full_vocab_ce": 50.0},
        ]
        self.assertIsNone(sustained_band_entry(rows, self.reference, branch_end_step=100))


if __name__ == "__main__":
    unittest.main()
