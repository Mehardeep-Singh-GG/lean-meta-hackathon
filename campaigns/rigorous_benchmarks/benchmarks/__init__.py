"""
Canonical Benchmark Suite for Project LEAN.

5 Independent Canonical Evaluators:
- GSM8KEvaluator (gsm8k_eval.py)
- ARCEvaluator (arc_eval.py)
- MMLUEvaluator (mmlu_eval.py)
- HumanEvalEvaluator (humaneval_eval.py)
- WikiTextPPL (wikitext_ppl.py)
"""

from .gsm8k_eval import GSM8KEvaluator, extract_gsm8k_answer, gsm8k_exact_match
from .arc_eval import ARCEvaluator, extract_arc_choice, format_arc_prompt
from .mmlu_eval import MMLUEvaluator, extract_mmlu_choice, format_mmlu_5shot_prompt
from .humaneval_eval import HumanEvalEvaluator, estimate_pass_at_k, run_code_with_timeout
from .wikitext_ppl import WikiTextPPL, compute_cross_entropy_and_perplexity, evaluate_sliding_window_perplexity

__all__ = [
    "GSM8KEvaluator",
    "extract_gsm8k_answer",
    "gsm8k_exact_match",
    "ARCEvaluator",
    "extract_arc_choice",
    "format_arc_prompt",
    "MMLUEvaluator",
    "extract_mmlu_choice",
    "format_mmlu_5shot_prompt",
    "HumanEvalEvaluator",
    "estimate_pass_at_k",
    "run_code_with_timeout",
    "WikiTextPPL",
    "compute_cross_entropy_and_perplexity",
    "evaluate_sliding_window_perplexity",
]
