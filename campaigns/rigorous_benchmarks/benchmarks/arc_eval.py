#!/usr/bin/env python3
"""
ARC-Challenge Benchmark Evaluator.

Standard AI2 ARC (AI2 Reasoning Challenge) Challenge-Set evaluation protocol:
- Evaluates 4-choice multiple choice science questions (A, B, C, D).
- Supports text generation answers (explicit option letters or option text matching).
- Supports choice logit / log-likelihood scoring (argmax of normalized token/sequence scores).
- Bundles verified ARC-Challenge test subset in self-contained JSON.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Union


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "arc_challenge_test.json"
)

VALID_LABELS = ["A", "B", "C", "D"]


def format_arc_prompt(item: Dict[str, Any]) -> str:
    """
    Constructs a standard zero-shot multiple-choice prompt for an ARC item.
    """
    question = item["question"].strip()
    choices = item["choices"]
    
    choice_lines = []
    for c in choices:
        choice_lines.append(f"{c['label']}. {c['text']}")
        
    choices_str = "\n".join(choice_lines)
    return f"Question: {question}\n{choices_str}\nAnswer:"


def extract_arc_choice(
    text: str,
    choices: Optional[List[Dict[str, str]]] = None,
    valid_labels: Optional[List[str]] = None
) -> Optional[str]:
    """
    Extracts the multiple choice letter selection (A, B, C, D) from model output.

    Precedence:
    1. Direct patterns like 'Answer: (A)', 'Answer: A', 'Option C', 'Choice B', '(D)'
    2. Leading answer token: 'A.', 'A)', 'A:'
    3. Trailing answer token: '... is A.'
    4. Exact or near-exact match of choice text within prediction.
    5. Lone letter in text.
    """
    if not text or not isinstance(text, str):
        return None

    labels = valid_labels or VALID_LABELS
    cleaned = text.strip()

    # If the text is just a single letter
    if cleaned.upper() in labels:
        return cleaned.upper()

    # 1. Direct answer indicator
    m_direct = re.search(
        r"(?:the correct answer is|the answer is|answer is|answer:|choice:|option:)\s*\(?([A-D])\)?",
        cleaned,
        re.IGNORECASE
    )
    if m_direct and m_direct.group(1).upper() in labels:
        return m_direct.group(1).upper()

    # 2. Leading answer token like "A." or "(A)" or "A:"
    m_leading = re.match(r"^\(?([A-D])\)?[.:\s]", cleaned, re.IGNORECASE)
    if m_leading and m_leading.group(1).upper() in labels:
        return m_leading.group(1).upper()

    # 3. Boxed answer \boxed{A}
    m_boxed = re.search(r"\\boxed\{\s*([A-D])\s*\}", cleaned, re.IGNORECASE)
    if m_boxed and m_boxed.group(1).upper() in labels:
        return m_boxed.group(1).upper()

    # 4. Content match if choices are provided
    if choices:
        for c in choices:
            c_text = c.get("text", "").strip().lower()
            if len(c_text) > 4 and c_text in cleaned.lower():
                return c.get("label", "").upper()

    # 5. Fallback: search for first or last standalone letter [A-D]
    letters = re.findall(r"\b([A-D])\b", cleaned, re.IGNORECASE)
    if letters:
        # Return first standalone letter
        return letters[0].upper()

    return None


class ARCEvaluator:
    """Canonical evaluator for AI2 ARC-Challenge science reasoning benchmark."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or DEFAULT_DATA_PATH
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ARC benchmark data file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_dataset(self) -> List[Dict[str, Any]]:
        return self.dataset

    def format_prompt(self, item: Dict[str, Any]) -> str:
        return format_arc_prompt(item)

    def evaluate_sample(
        self,
        prediction: str,
        ground_truth: str,
        choices: Optional[List[Dict[str, str]]] = None,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluates a single prediction against ground truth letter."""
        extracted = extract_arc_choice(prediction, choices=choices)
        gt = ground_truth.strip().upper()
        is_correct = (extracted == gt) if extracted is not None else False
        return {
            "id": item_id,
            "prediction_raw": prediction,
            "extracted_choice": extracted,
            "ground_truth": gt,
            "is_correct": is_correct
        }

    def evaluate_predictions(
        self,
        predictions: Union[List[str], List[Dict[str, Any]]],
        dataset: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a sequence of text predictions against the dataset.
        """
        data = dataset or self.dataset
        if len(predictions) != len(data):
            raise ValueError(
                f"Prediction count ({len(predictions)}) does not match dataset size ({len(data)})."
            )

        num_correct = 0
        details = []

        for i, item in enumerate(data):
            pred_entry = predictions[i]
            if isinstance(pred_entry, dict):
                pred_text = pred_entry.get("prediction", pred_entry.get("text", ""))
            else:
                pred_text = str(pred_entry)

            sample_res = self.evaluate_sample(
                prediction=pred_text,
                ground_truth=item["answerKey"],
                choices=item.get("choices"),
                item_id=item.get("id")
            )
            if sample_res["is_correct"]:
                num_correct += 1
            details.append(sample_res)

        total = len(data)
        acc = num_correct / total if total > 0 else 0.0

        return {
            "benchmark": "ARC-Challenge",
            "mode": "generative_text",
            "accuracy": float(acc),
            "accuracy_percent": float(acc * 100.0),
            "num_correct": int(num_correct),
            "total": int(total),
            "details": details
        }

    def evaluate_logits(
        self,
        choice_scores_list: List[Dict[str, float]],
        dataset: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates predictions based on choice logit scores or log-likelihoods.
        Args:
            choice_scores_list: List of dicts mapping choice label ('A', 'B', 'C', 'D') to score.
        """
        data = dataset or self.dataset
        if len(choice_scores_list) != len(data):
            raise ValueError(
                f"Score count ({len(choice_scores_list)}) does not match dataset size ({len(data)})."
            )

        num_correct = 0
        details = []

        for i, item in enumerate(data):
            scores = choice_scores_list[i]
            best_choice = max(scores.items(), key=lambda x: x[1])[0].upper()
            gt = item["answerKey"].strip().upper()
            is_correct = (best_choice == gt)
            if is_correct:
                num_correct += 1

            details.append({
                "id": item.get("id"),
                "choice_scores": scores,
                "selected_choice": best_choice,
                "ground_truth": gt,
                "is_correct": is_correct
            })

        total = len(data)
        acc = num_correct / total if total > 0 else 0.0

        return {
            "benchmark": "ARC-Challenge",
            "mode": "choice_logits",
            "accuracy": float(acc),
            "accuracy_percent": float(acc * 100.0),
            "num_correct": int(num_correct),
            "total": int(total),
            "details": details
        }


def main():
    evaluator = ARCEvaluator()
    print(f"Loaded {len(evaluator)} ARC-Challenge questions.")
    gt_preds = [item["answerKey"] for item in evaluator.get_dataset()]
    res = evaluator.evaluate_predictions(gt_preds)
    print(f"Ground Truth Key Accuracy: {res['accuracy_percent']:.2f}% ({res['num_correct']}/{res['total']})")
    return res


if __name__ == "__main__":
    main()
