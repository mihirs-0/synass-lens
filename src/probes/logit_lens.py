"""
Logit Lens probe.

Projects intermediate residual stream states to vocabulary space
to see "when" (at which layer) the model "knows" the correct answer.

The logit lens technique:
1. Take residual stream at layer L
2. Apply final LayerNorm and unembedding
3. Get probability distribution over vocabulary
4. Check if correct token is top-1 or measure its probability

Expected finding:
- Early in training: correct answer probability low at all layers
- Mid training (before z-usage): correct probability increases but
  model might be "guessing" based on marginal P(A|B) (uniform over K options)
- Late training: correct probability high, especially at later layers
"""

from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .base import BaseProbe


class LogitLensProbe(BaseProbe):
    """
    Probe that measures when each layer "knows" the correct answer.
    """
    
    name = "logit_lens"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.first_token_only = config.get("first_token_only", False) if config else False
        
    def run(
        self,
        model: HookedTransformer,
        dataloader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Run logit lens probe.
        
        Returns:
            {
                "correct_prob_by_layer": (n_layers + 1,) array
                    Probability assigned to correct next token at each layer
                    Index 0 = embedding layer, indices 1-n_layers = after each layer
                "top1_accuracy_by_layer": (n_layers + 1,) array
                    Whether correct token is argmax at each layer
                "entropy_by_layer": (n_layers + 1,) array
                    Entropy of predicted distribution at each layer
            }
        """
        model.eval()
        n_layers = model.cfg.n_layers
        
        # Accumulators: +1 for embedding layer (before any transformer blocks)
        correct_prob_sum = torch.zeros(n_layers + 1, device=device)
        top1_correct_sum = torch.zeros(n_layers + 1, device=device)
        entropy_sum = torch.zeros(n_layers + 1, device=device)
        n_tokens = 0
        tokens_per_layer_count = torch.zeros(n_layers + 1, device=device)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Logit lens probe", leave=False):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                target_start_positions = batch["target_start_positions"].to(device)
                target_end_positions = batch["target_end_positions"].to(device)
                
                # Run with cache
                _, cache = model.run_with_cache(input_ids)
                
                # Get the unembedding matrix and final LayerNorm
                W_U = model.W_U  # (d_model, vocab_size)
                
                # For each layer (including embedding), project to vocab
                for layer_idx in range(n_layers + 1):
                    if layer_idx == 0:
                        # After embedding, before any blocks
                        resid = cache["hook_embed"] + cache["hook_pos_embed"]
                    else:
                        # After transformer block (layer_idx - 1)
                        resid = cache["resid_post", layer_idx - 1]
                    
                    # Apply final LayerNorm
                    resid_normed = model.ln_final(resid)
                    
                    # Project to vocabulary
                    logits = resid_normed @ W_U  # (batch, seq, vocab)
                    
                    # Compute probabilities
                    probs = torch.softmax(logits, dim=-1)
                    
                    # For each example, look at target positions only
                    batch_size = input_ids.shape[0]
                    for b in range(batch_size):
                        target_start = target_start_positions[b].item()
                        target_end = target_end_positions[b].item()
                        
                        # We predict position i+1 from position i
                        # So for target tokens, we look at logits at positions
                        # (target_start-1) to (target_end-1) predicting target_start..target_end
                        
                        if self.first_token_only:
                            pos_iter = range(target_start, target_start + 1)
                        else:
                            pos_iter = range(target_start, target_end)
                        
                        for pos in pos_iter:
                            pred_pos = pos - 1  # Position where prediction is made
                            target = labels[b, pos].item()
                            
                            if target == -100:
                                continue
                                
                            pred_probs = probs[b, pred_pos]  # (vocab,)
                            
                            # Probability of correct token
                            correct_prob = pred_probs[target].item()
                            correct_prob_sum[layer_idx] += correct_prob
                            
                            # Top-1 accuracy
                            top1 = pred_probs.argmax().item()
                            top1_correct_sum[layer_idx] += float(top1 == target)
                            
                            # Entropy
                            entropy = -(pred_probs * (pred_probs + 1e-10).log()).sum().item()
                            entropy_sum[layer_idx] += entropy
                            
                            n_tokens += 1
                            tokens_per_layer_count[layer_idx] += 1
        
        # Average (divide by n_tokens, accounting for n_layers+1 factor)
        tokens_per_layer = torch.clamp(tokens_per_layer_count, min=1.0)
        
        return {
            "correct_prob_by_layer": (correct_prob_sum / tokens_per_layer).cpu(),
            "top1_accuracy_by_layer": (top1_correct_sum / tokens_per_layer).cpu(),
            "entropy_by_layer": (entropy_sum / tokens_per_layer).cpu(),
            "n_tokens_total": n_tokens,
        }
