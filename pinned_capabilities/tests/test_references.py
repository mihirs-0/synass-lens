import math
import unittest

from pinned_capabilities.references import empirical_constant_machine
from src.data import CharTokenizer, generate_mappings


class ConstantMachineTests(unittest.TestCase):
    def test_reference_is_empirical_not_log_vocab(self) -> None:
        tokenizer = CharTokenizer(vocab_chars="abcdef")
        mapping = generate_mappings(
            n_unique_b=4,
            k=2,
            b_length=2,
            a_length=2,
            z_length=1,
            vocab_chars="abcdef",
            seed=9,
            task="bz_to_a",
            enforce_unique_a_first_char_per_b=True,
        )
        result = empirical_constant_machine(mapping, tokenizer)
        self.assertEqual(result["candidate_first_token_loss"], math.log(2))
        self.assertGreater(result["training_loss"], 0.0)
        self.assertLess(result["training_loss"], math.log(tokenizer.vocab_size))
        self.assertEqual(len(result["position_entropies"]), 3)  # two target positions plus EOS


if __name__ == "__main__":
    unittest.main()
