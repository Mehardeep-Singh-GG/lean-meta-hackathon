#!/usr/bin/env python3
"""
WikiText-2 Sequence Perplexity Evaluator.

Standard WikiText-2 test set evaluation protocol:
- Computes exact sequence cross-entropy loss and exponentiated token perplexity:
  PPL = exp( 1/N * sum_{i=1}^N L_i )
- Supports both direct logits-vs-target evaluation and sliding-window model evaluation.
- Operates on bundled canonical WikiText-2 test split (zero external download required).
- Zero mock scores, zero approximations: exact PyTorch tensor cross-entropy mathematics.
"""

import os
import math
import json
from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "wikitext2_test.json"
)


def compute_cross_entropy_and_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    shift: bool = True,
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Computes exact cross-entropy loss, bits-per-character/token, and perplexity.

    Args:
        logits: Model output logits tensor of shape (batch, seq_len, vocab_size) or (N, vocab_size).
        targets: Target token IDs tensor of shape (batch, seq_len) or (N,).
        shift: If True, shifts logits[:-1] and targets[1:] for standard autoregressive causal modeling.
               If False, assumes logits and targets are already aligned.
        ignore_index: Target index to ignore in cross entropy (e.g. -100).

    Returns:
        Dict with mean_loss_nats, mean_loss_bits, perplexity, and valid_tokens.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("logits and targets must both be torch.Tensor instances.")

    logits = logits.float()

    if shift:
        if logits.dim() < 2 or targets.dim() < 2:
            raise ValueError("When shift=True, logits and targets must have at least 2 dimensions [batch, seq_len, ...].")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
    else:
        shift_logits = logits.contiguous()
        shift_labels = targets.contiguous()

    vocab_size = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    # Compute unreduced loss to accurately count non-ignored tokens
    loss_unreduced = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="none")
    mask = (flat_labels != ignore_index)
    valid_tokens = int(mask.sum().item())

    if valid_tokens == 0:
        raise ValueError("No valid tokens found for perplexity evaluation (all tokens ignored).")

    total_nll = float(loss_unreduced.sum().item())
    mean_nll_nats = total_nll / valid_tokens
    mean_nll_bits = mean_nll_nats / math.log(2.0)
    ppl = math.exp(mean_nll_nats)

    return {
        "mean_loss_nats": float(mean_nll_nats),
        "mean_loss_bits": float(mean_nll_bits),
        "perplexity": float(ppl),
        "total_tokens": valid_tokens
    }


def evaluate_sliding_window_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_length: int = 1024,
    stride: int = 512,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Computes exact sliding-window perplexity on a long sequence of token IDs.
    Follows standard HuggingFace / PyTorch sliding-window evaluation protocol:
    tokens outside the current stride are conditioned upon but their loss is masked out.

    Args:
        model: Autoregressive causal language model.
        input_ids: 1D or 2D tensor of token IDs.
        max_length: Maximum context window of the model.
        stride: Stride step size between windows.
        device: Device to run evaluation on.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    model.eval()
    model.to(device)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    seq_len = input_ids.size(1)
    nlls = []
    total_tokens = 0

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - max(begin_loc, end_loc - stride)

            sub_ids = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = sub_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(sub_ids, labels=target_ids)
            
            # Loss is average across non -100 targets in the window
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)
            total_tokens += trg_len

            if end_loc == seq_len:
                break

    total_nll = torch.stack(nlls).sum().item()
    mean_nats = total_nll / total_tokens
    mean_bits = mean_nats / math.log(2.0)
    ppl = math.exp(mean_nats)

    return {
        "benchmark": "WikiText-2",
        "perplexity": float(ppl),
        "mean_loss_nats": float(mean_nats),
        "mean_loss_bits": float(mean_bits),
        "total_tokens": int(total_tokens),
        "stride": int(stride),
        "max_length": int(max_length)
    }


class WikiTextPPL:
    """Canonical evaluator for WikiText-2 language modeling perplexity."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or DEFAULT_DATA_PATH
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"WikiText benchmark data file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data_blob = json.load(f)

        self.corpus = self.data_blob.get("corpus", "")
        self.articles = self.data_blob.get("articles", [])
        self.token_sequences = self.data_blob.get("token_sequences", {})

    def get_corpus(self) -> str:
        """Returns the raw test text corpus."""
        return self.corpus

    def get_pretokenized_ids(self, key: str = "gpt2") -> Optional[torch.Tensor]:
        """Returns pre-tokenized token ID tensor if available."""
        seq = self.token_sequences.get(key)
        if seq is not None:
            return torch.tensor(seq, dtype=torch.long)
        return None

    def evaluate_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        shift: bool = True
    ) -> Dict[str, float]:
        """Direct exact cross-entropy and perplexity calculation on tensor logits."""
        res = compute_cross_entropy_and_perplexity(logits, targets, shift=shift)
        return {
            "benchmark": "WikiText-2",
            **res
        }

    def evaluate_model(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        token_key: Optional[str] = None,
        max_length: int = 1024,
        stride: int = 512,
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a causal language model on the canonical WikiText-2 test set.
        Can use pre-tokenized IDs or tokenize with a provided tokenizer.
        """
        if token_key is not None:
            if token_key not in self.token_sequences:
                raise ValueError(f"Requested token_key '{token_key}' not found in pre-tokenized sequences: {list(self.token_sequences.keys())}")
            input_ids = self.get_pretokenized_ids(token_key)
        elif tokenizer is not None:
            enc = tokenizer(self.corpus, return_tensors="pt")
            input_ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        elif "gpt2" in self.token_sequences:
            input_ids = self.get_pretokenized_ids("gpt2")
        else:
            raise ValueError("Either a valid tokenizer or pre-tokenized token_key must be supplied.")

        return evaluate_sliding_window_perplexity(
            model=model,
            input_ids=input_ids,
            max_length=max_length,
            stride=stride,
            device=device
        )


def main():
    evaluator = WikiTextPPL()
    print(f"Loaded WikiText-2 test corpus: {len(evaluator.corpus)} characters, {len(evaluator.articles)} articles.")
    for k, tokens in evaluator.token_sequences.items():
        print(f"  Pre-tokenized '{k}': {len(tokens)} tokens.")

    # Demonstration with exact theoretical cross-entropy
    dummy_vocab = 100
    dummy_seq = 50
    uniform_logits = torch.zeros(1, dummy_seq, dummy_vocab)
    dummy_targets = torch.randint(0, dummy_vocab, (1, dummy_seq))
    res = evaluator.evaluate_logits(uniform_logits, dummy_targets)
    expected_ppl = dummy_vocab
    print(f"Uniform Logits Test -> PPL: {res['perplexity']:.2f} (Expected: {expected_ppl:.2f})")
    return res


if __name__ == "__main__":
    main()
