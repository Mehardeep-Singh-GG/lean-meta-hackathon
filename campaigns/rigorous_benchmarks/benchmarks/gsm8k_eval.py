#!/usr/bin/env python3
"""
GSM8K Benchmark Evaluator.

Standard GSM8K (Grade School Math 8K) evaluation protocol:
- Extracts final numerical answer from chain-of-thought model generations.
- Handles standard patterns including '#### <number>', '\\boxed{<number>}',
  'the answer is <number>', comma separators, decimals, and negative numbers.
- Computes exact match accuracy across verified GSM8K test problems.
- Standalone evaluator operating with zero network calls on cached benchmark data.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Union


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "gsm8k_test.json"
)


def normalize_numeric_string(s: str) -> Optional[float]:
    """
    Cleans and converts a numerical string to a float for comparison.
    Handles currency symbols, comma grouping, percentages, and whitespace.
    """
    if s is None:
        return None
    # Strip common enclosing characters / prefixes
    s = s.strip()
    s = re.sub(r"[\$€£%]", "", s)
    s = s.replace(",", "").strip()
    
    # Remove trailing periods if not part of a decimal
    if s.endswith("."):
        s = s[:-1]
        
    try:
        val = float(s)
        if val.is_integer():
            return int(val)
        return val
    except (ValueError, TypeError):
        return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extracts the final numeric answer from a model reasoning completion.

    Precedence:
    1. '#### <number>' (canonical GSM8K format)
    2. '\\boxed{<number>}' (canonical MATH format)
    3. 'the answer is <number>' / 'answer: <number>' / '= <number>'
    4. Fallback: last numeric token in the generation.
    """
    if not text or not isinstance(text, str):
        return None
        
    cleaned_text = text.strip()
    
    # 1. Standard GSM8K delimiter: '#### <number>'
    m_delim = re.search(r"####\s*([+\-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?|[+\-]?[0-9]+(?:\.[0-9]+)?)", cleaned_text)
    if m_delim:
        return m_delim.group(1).replace(",", "").strip()
        
    # 2. LaTeX boxed format: \boxed{<content>}
    m_boxed = re.search(r"\\boxed\{\s*([+\-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*\}", cleaned_text)
    if m_boxed:
        return m_boxed.group(1).replace(",", "").strip()

    # 3. Phrased answers: 'The answer is <number>', 'Answer: <number>'
    m_phrase = re.findall(
        r"(?:the answer is|answer is|final answer is|answer:)\s*([+\-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        cleaned_text,
        re.IGNORECASE
    )
    if m_phrase:
        return m_phrase[-1].replace(",", "").strip()

    # 4. Fallback: scan for all isolated numbers and return the final one
    numbers = re.findall(r"([+\-]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?|[+\-]?[0-9]+(?:\.[0-9]+)?)", cleaned_text)
    if numbers:
        # Filter out obvious false positives like lone '+' or '-'
        valid_nums = [n.replace(",", "").strip() for n in numbers if any(c.isdigit() for c in n)]
        if valid_nums:
            return valid_nums[-1]

    return None


def gsm8k_exact_match(prediction_str: Optional[str], target_str: str) -> bool:
    """
    Checks exact match equivalence between extracted prediction and target number.
    Supports both integer and floating point equivalence.
    """
    if prediction_str is None or target_str is None:
        return False
        
    pred_val = normalize_numeric_string(prediction_str)
    targ_val = normalize_numeric_string(target_str)
    
    if pred_val is not None and targ_val is not None:
        # Check numerical equality
        if isinstance(pred_val, int) and isinstance(targ_val, int):
            return pred_val == targ_val
        return abs(float(pred_val) - float(targ_val)) < 1e-5
        
    # Fallback to stripped string comparison
    return str(prediction_str).strip() == str(target_str).strip()


class GSM8KEvaluator:
    """Canonical evaluator for GSM8K reasoning benchmark."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or DEFAULT_DATA_PATH
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"GSM8K benchmark data file not found at: {self.data_path}")
            
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_dataset(self) -> List[Dict[str, Any]]:
        """Returns the loaded GSM8K dataset problems."""
        return self.dataset

    def evaluate_sample(
        self,
        prediction: str,
        target: str,
        question: Optional[str] = None,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluates a single problem generation against target answer."""
        extracted = extract_gsm8k_answer(prediction)
        is_correct = gsm8k_exact_match(extracted, target)
        return {
            "id": item_id,
            "question": question,
            "prediction_text": prediction,
            "extracted_answer": extracted,
            "target_answer": target,
            "is_correct": is_correct
        }

    def evaluate_predictions(
        self,
        predictions: Union[List[str], List[Dict[str, Any]]],
        dataset: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a sequence of model completions against the benchmark dataset.

        Args:
            predictions: List of generation strings or list of dicts with 'id' / 'prediction'.
            dataset: Optional override dataset; defaults to self.dataset.

        Returns:
            Dict containing:
                - accuracy: float in [0.0, 1.0]
                - accuracy_percent: float in [0.0, 100.0]
                - num_correct: int
                - total: int
                - details: per-problem evaluation breakdown
        """
        data = dataset or self.dataset
        if len(predictions) != len(data):
            raise ValueError(
                f"Prediction count ({len(predictions)}) does not match dataset size ({len(data)})."
            )

        details = []
        num_correct = 0

        for i, item in enumerate(data):
            pred_entry = predictions[i]
            if isinstance(pred_entry, dict):
                pred_text = pred_entry.get("prediction", pred_entry.get("text", ""))
            else:
                pred_text = str(pred_entry)

            sample_res = self.evaluate_sample(
                prediction=pred_text,
                target=item["target"],
                question=item.get("question"),
                item_id=item.get("id")
            )
            if sample_res["is_correct"]:
                num_correct += 1
            details.append(sample_res)

        total = len(data)
        acc = num_correct / total if total > 0 else 0.0

        return {
            "benchmark": "GSM8K",
            "accuracy": float(acc),
            "accuracy_percent": float(acc * 100.0),
            "num_correct": int(num_correct),
            "total": int(total),
            "details": details
        }


def main():
    evaluator = GSM8KEvaluator()
    print(f"Loaded {len(evaluator)} GSM8K benchmark problems.")
    canonical_preds = [item["solution"] for item in evaluator.get_dataset()]
    results = evaluator.evaluate_predictions(canonical_preds)
    print(f"Ground Truth Solution Accuracy: {results['accuracy_percent']:.2f}% ({results['num_correct']}/{results['total']})")
    return results


if __name__ == "__main__":
    main()
