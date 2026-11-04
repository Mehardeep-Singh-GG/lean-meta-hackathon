#!/usr/bin/env python3
"""
MMLU (Massive Multitask Language Understanding) Benchmark Evaluator.

Standard MMLU evaluation protocol covering 5 canonical disciplines:
1. Elementary Mathematics (STEM)
2. Formal Logic (STEM / Symbolic Reasoning)
3. High School Physics (STEM)
4. College Computer Science (STEM)
5. High School World History (Factual Recall / Humanities)

Features:
- Standard 5-shot prompt formatting from authentic dev demonstrations.
- Evaluates generative model completions and/or choice logit scores.
- Computes per-subject accuracy, macro-average across the 5 domains, and micro-accuracy.
- Self-contained, using embedded verified benchmark items.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Union


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "mmlu_test.json"
)

SUBJECT_DESCRIPTIONS = {
    "elementary_mathematics": "elementary mathematics",
    "formal_logic": "formal logic",
    "high_school_physics": "high school physics",
    "college_computer_science": "computer science",
    "high_school_world_history": "world history"
}

VALID_LABELS = ["A", "B", "C", "D"]


def format_mmlu_question_block(item: Dict[str, Any], include_answer: bool = False) -> str:
    """Formats a single question and its choices."""
    q_text = item["question"].strip()
    choices = item["choices"]
    choice_lines = [f"{c['label']}. {c['text']}" for c in choices]
    block = f"{q_text}\n" + "\n".join(choice_lines) + "\nAnswer:"
    if include_answer:
        block += f" {item['answer']}"
    return block


def format_mmlu_5shot_prompt(
    item: Dict[str, Any],
    dev_shots: Optional[List[Dict[str, Any]]] = None,
    n_shot: int = 5
) -> str:
    """
    Constructs the canonical Hendrycks 5-shot MMLU prompt:
    'The following are multiple choice questions (with answers) about <subject>.\n\n<shots>\n\n<test_item>'
    """
    subject_key = item.get("subject", "")
    sub_desc = SUBJECT_DESCRIPTIONS.get(subject_key, subject_key.replace("_", " "))
    header = f"The following are multiple choice questions (with answers) about {sub_desc}.\n\n"

    shots_text = ""
    if dev_shots:
        selected_shots = dev_shots[:n_shot]
        shot_blocks = [format_mmlu_question_block(shot, include_answer=True) for shot in selected_shots]
        if shot_blocks:
            shots_text = "\n\n".join(shot_blocks) + "\n\n"

    test_block = format_mmlu_question_block(item, include_answer=False)
    return header + shots_text + test_block


def extract_mmlu_choice(
    text: str,
    choices: Optional[List[Dict[str, str]]] = None,
    valid_labels: Optional[List[str]] = None
) -> Optional[str]:
    """
    Extracts the multiple choice letter (A, B, C, D) from MMLU model completions.
    """
    if not text or not isinstance(text, str):
        return None

    labels = valid_labels or VALID_LABELS
    cleaned = text.strip()

    # Exact single letter
    if cleaned.upper() in labels:
        return cleaned.upper()

    # Direct answer declaration
    m_direct = re.search(
        r"(?:the correct answer is|the answer is|answer is|answer:)\s*\(?([A-D])\)?",
        cleaned,
        re.IGNORECASE
    )
    if m_direct and m_direct.group(1).upper() in labels:
        return m_direct.group(1).upper()

    # Leading answer format: "(B)", "B.", "B)"
    m_lead = re.match(r"^\(?([A-D])\)?[.:\s]", cleaned, re.IGNORECASE)
    if m_lead and m_lead.group(1).upper() in labels:
        return m_lead.group(1).upper()

    # LaTeX boxed
    m_boxed = re.search(r"\\boxed\{\s*([A-D])\s*\}", cleaned, re.IGNORECASE)
    if m_boxed and m_boxed.group(1).upper() in labels:
        return m_boxed.group(1).upper()

    # Match choice text if provided
    if choices:
        for c in choices:
            c_text = c.get("text", "").strip().lower()
            if len(c_text) > 4 and c_text in cleaned.lower():
                return c.get("label", "").upper()

    # First standalone letter
    letters = re.findall(r"\b([A-D])\b", cleaned, re.IGNORECASE)
    if letters:
        return letters[0].upper()

    return None


class MMLUEvaluator:
    """Canonical evaluator for the 5-subject MMLU benchmark."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or DEFAULT_DATA_PATH
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"MMLU benchmark data file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data_blob = json.load(f)

        self.subjects = data_blob["subjects"]
        self.dataset = data_blob["test"]
        self.dev_few_shot = data_blob.get("dev_few_shot", {})

    def __len__(self) -> int:
        return len(self.dataset)

    def get_dataset(self) -> List[Dict[str, Any]]:
        return self.dataset

    def get_subject_items(self, subject: str) -> List[Dict[str, Any]]:
        """Returns items corresponding to a specific subject."""
        return [item for item in self.dataset if item.get("subject") == subject]

    def format_prompt(self, item: Dict[str, Any], n_shot: int = 5) -> str:
        """Returns standard 5-shot prompt for a test item."""
        dev_shots = self.dev_few_shot.get(item.get("subject"), [])
        return format_mmlu_5shot_prompt(item, dev_shots=dev_shots, n_shot=n_shot)

    def evaluate_sample(
        self,
        prediction: str,
        ground_truth: str,
        choices: Optional[List[Dict[str, str]]] = None,
        subject: Optional[str] = None,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluates single sample."""
        extracted = extract_mmlu_choice(prediction, choices=choices)
        gt = ground_truth.strip().upper()
        is_correct = (extracted == gt) if extracted is not None else False
        return {
            "id": item_id,
            "subject": subject,
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
        Evaluates predictions across all subjects.
        Returns per-subject accuracy, macro-average, and detailed breakdown.
        """
        data = dataset or self.dataset
        if len(predictions) != len(data):
            raise ValueError(
                f"Prediction count ({len(predictions)}) does not match dataset size ({len(data)})."
            )

        subject_stats: Dict[str, Dict[str, int]] = {
            s: {"correct": 0, "total": 0} for s in self.subjects
        }
        details = []

        for i, item in enumerate(data):
            pred_entry = predictions[i]
            if isinstance(pred_entry, dict):
                pred_text = pred_entry.get("prediction", pred_entry.get("text", ""))
            else:
                pred_text = str(pred_entry)

            sample_res = self.evaluate_sample(
                prediction=pred_text,
                ground_truth=item["answer"],
                choices=item.get("choices"),
                subject=item.get("subject"),
                item_id=item.get("id")
            )
            sub = item.get("subject", "unknown")
            if sub not in subject_stats:
                subject_stats[sub] = {"correct": 0, "total": 0}

            subject_stats[sub]["total"] += 1
            if sample_res["is_correct"]:
                subject_stats[sub]["correct"] += 1

            details.append(sample_res)

        # Compute per-subject accuracy and macro average
        per_subject_acc = {}
        for sub, counts in subject_stats.items():
            tot = counts["total"]
            acc = counts["correct"] / tot if tot > 0 else 0.0
            per_subject_acc[sub] = {
                "accuracy": float(acc),
                "accuracy_percent": float(acc * 100.0),
                "correct": counts["correct"],
                "total": tot
            }

        macro_avg = (
            sum(per_subject_acc[s]["accuracy"] for s in self.subjects if s in per_subject_acc)
            / len(self.subjects)
            if self.subjects else 0.0
        )

        total_correct = sum(s["correct"] for s in subject_stats.values())
        total_items = sum(s["total"] for s in subject_stats.values())
        micro_acc = total_correct / total_items if total_items > 0 else 0.0

        return {
            "benchmark": "MMLU_5Subject",
            "macro_average": float(macro_avg),
            "macro_average_percent": float(macro_avg * 100.0),
            "micro_average": float(micro_acc),
            "micro_average_percent": float(micro_acc * 100.0),
            "total_correct": int(total_correct),
            "total_items": int(total_items),
            "per_subject": per_subject_acc,
            "details": details
        }

    def evaluate_logits(
        self,
        choice_scores_list: List[Dict[str, float]],
        dataset: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates predictions based on choice logits.
        """
        data = dataset or self.dataset
        if len(choice_scores_list) != len(data):
            raise ValueError(
                f"Score count ({len(choice_scores_list)}) does not match dataset size ({len(data)})."
            )

        subject_stats: Dict[str, Dict[str, int]] = {
            s: {"correct": 0, "total": 0} for s in self.subjects
        }
        details = []

        for i, item in enumerate(data):
            scores = choice_scores_list[i]
            best_choice = max(scores.items(), key=lambda x: x[1])[0].upper()
            gt = item["answer"].strip().upper()
            is_correct = (best_choice == gt)

            sub = item.get("subject", "unknown")
            if sub not in subject_stats:
                subject_stats[sub] = {"correct": 0, "total": 0}

            subject_stats[sub]["total"] += 1
            if is_correct:
                subject_stats[sub]["correct"] += 1

            details.append({
                "id": item.get("id"),
                "subject": sub,
                "scores": scores,
                "selected_choice": best_choice,
                "ground_truth": gt,
                "is_correct": is_correct
            })

        per_subject_acc = {}
        for sub, counts in subject_stats.items():
            tot = counts["total"]
            acc = counts["correct"] / tot if tot > 0 else 0.0
            per_subject_acc[sub] = {
                "accuracy": float(acc),
                "accuracy_percent": float(acc * 100.0),
                "correct": counts["correct"],
                "total": tot
            }

        macro_avg = (
            sum(per_subject_acc[s]["accuracy"] for s in self.subjects if s in per_subject_acc)
            / len(self.subjects)
            if self.subjects else 0.0
        )
        total_correct = sum(s["correct"] for s in subject_stats.values())
        total_items = sum(s["total"] for s in subject_stats.values())
        micro_acc = total_correct / total_items if total_items > 0 else 0.0

        return {
            "benchmark": "MMLU_5Subject",
            "mode": "choice_logits",
            "macro_average": float(macro_avg),
            "macro_average_percent": float(macro_avg * 100.0),
            "micro_average": float(micro_acc),
            "micro_average_percent": float(micro_acc * 100.0),
            "total_correct": int(total_correct),
            "total_items": int(total_items),
            "per_subject": per_subject_acc,
            "details": details
        }


def main():
    evaluator = MMLUEvaluator()
    print(f"Loaded {len(evaluator)} MMLU test questions across {len(evaluator.subjects)} subjects.")
    gt_preds = [item["answer"] for item in evaluator.get_dataset()]
    res = evaluator.evaluate_predictions(gt_preds)
    print(f"Ground Truth Macro Accuracy: {res['macro_average_percent']:.2f}%")
    for sub, stats in res["per_subject"].items():
        print(f"  {sub}: {stats['accuracy_percent']:.1f}% ({stats['correct']}/{stats['total']})")
    return res


if __name__ == "__main__":
    main()
