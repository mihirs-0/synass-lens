import unittest

import torch

from pinned_capabilities.mbc import (
    build_mbc_probes,
    differentiable_mbc_c_int,
    evaluate_mbc_probe,
)
from pinned_capabilities.metrics import sample_quartets
from pinned_capabilities.parameter_groups import grouped_named_parameters
from src.data import CharTokenizer, generate_mappings
from src.model.hooked_transformer import create_hooked_transformer


class MBCIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(5)
        cls.tokenizer = CharTokenizer()
        cls.mapping = generate_mappings(
            n_unique_b=12,
            k=3,
            b_length=3,
            a_length=2,
            z_length=1,
            vocab_chars=cls.tokenizer.vocab_chars,
            seed=17,
            task="bz_to_a",
            enforce_unique_a_first_char_per_b=True,
        )
        cls.model = create_hooked_transformer(
            tokenizer=cls.tokenizer,
            n_layers=1,
            n_heads=1,
            d_model=16,
            d_head=16,
            d_mlp=32,
            device="cpu",
        )

    def test_probe_build_and_forward(self) -> None:
        left, right = build_mbc_probes(
            self.mapping, self.tokenizer, n_b=5, seeds=(101, 103)
        )
        self.assertTrue(set(left.b_strings).isdisjoint(right.b_strings))
        self.assertEqual(left.input_ids.shape[0], 15)
        self.assertEqual(left.answer_token_ids.shape, (5, 3, 2))
        quartets = sample_quartets(5, 3, 24, seed=107)
        result = evaluate_mbc_probe(self.model, left, quartets, batch_size=4)
        self.assertEqual(set(result), {"c_int", "delta_z", "exact_match", "full_vocab_ce"})
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result.values()))
        self.assertGreaterEqual(result["exact_match"], 0.0)
        self.assertLessEqual(result["exact_match"], 1.0)

    def test_real_model_parameter_partition_is_exhaustive(self) -> None:
        groups = grouped_named_parameters(self.model)
        assigned = {name for values in groups.values() for name, _ in values}
        trainable = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        self.assertEqual(assigned, trainable)

    def test_interaction_assay_is_differentiable(self) -> None:
        probe, _ = build_mbc_probes(
            self.mapping, self.tokenizer, n_b=5, seeds=(101, 103)
        )
        quartets = sample_quartets(5, 3, 12, seed=109)
        self.model.zero_grad(set_to_none=True)
        score = differentiable_mbc_c_int(self.model, probe, quartets)
        score.backward()
        gradients = [
            parameter.grad for parameter in self.model.parameters() if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
