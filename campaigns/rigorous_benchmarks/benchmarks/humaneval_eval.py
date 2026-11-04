#!/usr/bin/env python3
"""
HumanEval Benchmark Evaluator.

Standard HumanEval (OpenAI Code Generation) evaluation protocol:
- Tests generated Python functions against rigorous unit tests and assertions.
- Executes code in an isolated subprocess with strict execution timeouts to prevent hangs.
- Uses OS pipe communication for zero-overhead, deadlock-free process synchronization.
- Computes unbiased Pass@k estimates (specifically Pass@1 and Pass@3) following
  the canonical combinatorial formulation from Chen et al. (2021).
- Self-contained, embedding verified canonical HumanEval programming tasks.
"""

import os
import json
import time
import multiprocessing
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "humaneval_test.json"
)


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[int, List[int], np.ndarray],
    k: int
) -> float:
    """
    Computes the unbiased estimator for pass@k from Chen et al. (2021):
    pass@k = E [ 1 - C(n - c, k) / C(n, k) ]
           = 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)

    Args:
        num_samples: Total number of samples per problem (n).
        num_correct: Number of passing samples per problem (c).
        k: The k parameter (e.g. 1 or 3).

    Returns:
        float: The pass@k score in [0.0, 1.0].
    """
    if isinstance(num_samples, int):
        num_samples = [num_samples]
    if isinstance(num_correct, int):
        num_correct = [num_correct]

    n_arr = np.asarray(num_samples, dtype=int)
    c_arr = np.asarray(num_correct, dtype=int)

    if len(n_arr) != len(c_arr):
        raise ValueError("num_samples and num_correct arrays must have equal length.")

    scores = []
    for n, c in zip(n_arr, c_arr):
        if n < k:
            scores.append(c / n if n > 0 else 0.0)
            continue

        if n - c < k:
            scores.append(1.0)
            continue

        prod = 1.0
        for i in range(k):
            prod *= (n - c - i) / (n - i)
        scores.append(1.0 - prod)

    return float(np.mean(scores)) if len(scores) > 0 else 0.0


def _worker_execute_pipe(program: str, conn):
    """Subprocess target that executes the program and sends result via pipe."""
    try:
        exec_globals = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "math": __import__("math"),
            "typing": __import__("typing"),
            "re": __import__("re"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "copy": __import__("copy"),
            "heapq": __import__("heapq"),
            "bisect": __import__("bisect"),
            "random": __import__("random"),
            "string": __import__("string"),
        }
        exec(program, exec_globals)
        conn.send({"status": "passed", "error": None})
    except AssertionError as ae:
        conn.send({"status": "failed", "error": f"AssertionError: {ae}"})
    except Exception as e:
        conn.send({"status": "error", "error": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


def run_code_with_timeout(program: str, timeout: float = 3.0) -> Dict[str, Any]:
    """
    Executes Python program text inside an isolated process with hard timeout.
    Uses Pipe for communication to avoid multiprocessing.Queue feeder thread deadlocks.
    """
    ctx = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else multiprocessing
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker_execute_pipe, args=(program, child_conn))
    proc.daemon = True
    start_time = time.time()
    proc.start()
    child_conn.close()  # Close unused write end in parent

    # Poll pipe with timeout
    has_data = parent_conn.poll(timeout)
    elapsed = time.time() - start_time

    if has_data:
        try:
            res = parent_conn.recv()
            proc.join(timeout=0.2)
            if proc.is_alive():
                proc.terminate()
            parent_conn.close()
            return {
                "passed": res["status"] == "passed",
                "status": res["status"],
                "error": res["error"],
                "runtime_sec": elapsed
            }
        except EOFError:
            pass

    # Timed out or aborted
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.2)
    parent_conn.close()

    return {
        "passed": False,
        "status": "timeout" if not has_data else "aborted",
        "error": f"Execution timed out after {timeout:.1f}s" if not has_data else "Process terminated abruptly",
        "runtime_sec": elapsed
    }


def assemble_program(prompt: str, completion: str, test_code: str, entry_point: str) -> str:
    """Combines prompt, model completion, and test assertions into executable script."""
    if f"def {entry_point}" in completion:
        code_body = completion
    else:
        code_body = prompt + ("\n" if not prompt.endswith("\n") else "") + completion

    full_program = (
        code_body
        + "\n\n"
        + test_code
        + f"\n\ncheck({entry_point})\n"
    )
    return full_program


class HumanEvalEvaluator:
    """Canonical evaluator for HumanEval code generation benchmark."""

    def __init__(self, data_path: Optional[str] = None, default_timeout: float = 3.0):
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.default_timeout = default_timeout
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"HumanEval benchmark data file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_dataset(self) -> List[Dict[str, Any]]:
        return self.dataset

    def evaluate_sample(
        self,
        item: Dict[str, Any],
        completion: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single completion for a specific HumanEval problem.
        """
        to = timeout or self.default_timeout
        program = assemble_program(
            prompt=item["prompt"],
            completion=completion,
            test_code=item["test"],
            entry_point=item["entry_point"]
        )
        res = run_code_with_timeout(program, timeout=to)
        return {
            "task_id": item["task_id"],
            "entry_point": item["entry_point"],
            "passed": res["passed"],
            "status": res["status"],
            "error": res["error"],
            "runtime_sec": res["runtime_sec"]
        }

    def evaluate_completions(
        self,
        completions_by_task: Dict[str, List[str]],
        dataset: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates completions where each task may have 1 or more candidates.
        Computes Pass@1 and Pass@3.

        Args:
            completions_by_task: Mapping from task_id (e.g. 'HumanEval/0') to list of completion strings.
            dataset: Optional subset of tasks to evaluate against. If None, evaluates against
                     tasks provided in completions_by_task (or all tasks in self.dataset).
            timeout: Subprocess execution timeout in seconds.
        """
        if dataset is not None:
            target_items = dataset
        else:
            # If callers provide a subset of tasks in completions_by_task, evaluate only those tasks
            target_tasks = set(completions_by_task.keys())
            if target_tasks.issubset({it["task_id"] for it in self.dataset}) and len(target_tasks) < len(self.dataset):
                target_items = [it for it in self.dataset if it["task_id"] in target_tasks]
            else:
                target_items = self.dataset

        sample_counts = []
        correct_counts = []
        details = {}

        for item in target_items:
            task_id = item["task_id"]
            task_completions = completions_by_task.get(task_id, [])

            task_correct = 0
            task_details = []

            for comp in task_completions:
                res = self.evaluate_sample(item, comp, timeout=timeout)
                if res["passed"]:
                    task_correct += 1
                task_details.append(res)

            n_samples = len(task_completions)
            sample_counts.append(n_samples)
            correct_counts.append(task_correct)

            details[task_id] = {
                "n_samples": n_samples,
                "n_correct": task_correct,
                "evaluations": task_details
            }

        pass_1 = estimate_pass_at_k(sample_counts, correct_counts, k=1)
        
        has_multi = any(n >= 3 for n in sample_counts)
        pass_3 = estimate_pass_at_k(sample_counts, correct_counts, k=3) if has_multi else None

        total_correct = sum(correct_counts)
        total_samples = sum(sample_counts)

        return {
            "benchmark": "HumanEval",
            "total_tasks": len(target_items),
            "total_samples": int(total_samples),
            "total_correct": int(total_correct),
            "pass@1": float(pass_1),
            "pass@1_percent": float(pass_1 * 100.0),
            "pass@3": float(pass_3) if pass_3 is not None else None,
            "pass@3_percent": float(pass_3 * 100.0) if pass_3 is not None else None,
            "details": details
        }


def main(subset_size: Optional[int] = None):
    evaluator = HumanEvalEvaluator()
    dataset = evaluator.get_dataset()
    if subset_size is not None:
        dataset = dataset[:subset_size]
    print(f"Loaded {len(dataset)} HumanEval benchmark tasks for evaluation.")
    
    canonical_map = {
        item["task_id"]: [item["canonical_solution"]]
        for item in dataset
    }
    results = evaluator.evaluate_completions(canonical_map, dataset=dataset)
    print(f"Canonical Solutions Pass@1: {results['pass@1_percent']:.2f}% ({results['total_correct']}/{results['total_samples']})")
    return results


if __name__ == "__main__":
    main()
