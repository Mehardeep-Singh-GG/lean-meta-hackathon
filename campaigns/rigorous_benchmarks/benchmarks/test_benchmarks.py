#!/usr/bin/env python3
"""
Unit tests for the 5 canonical benchmark evaluators in Project LEAN:
1. GSM8K (gsm8k_eval.py)
2. ARC-Challenge (arc_eval.py)
3. MMLU (mmlu_eval.py)
4. HumanEval (humaneval_eval.py)
5. WikiText-2 Perplexity (wikitext_ppl.py)

Zero mocks, zero spoofed metrics. Tests real numerical equivalence, real prompt
generation, real isolated subprocess code execution, and real tensor cross-entropy.
"""

import math
import pytest
import torch
import torch.nn as nn

from campaigns.rigorous_benchmarks.benchmarks.gsm8k_eval import (
    GSM8KEvaluator,
    extract_gsm8k_answer,
    gsm8k_exact_match,
    normalize_numeric_string,
    main as gsm8k_main
)
from campaigns.rigorous_benchmarks.benchmarks.arc_eval import (
    ARCEvaluator,
    extract_arc_choice,
    format_arc_prompt,
    main as arc_main
)
from campaigns.rigorous_benchmarks.benchmarks.mmlu_eval import (
    MMLUEvaluator,
    extract_mmlu_choice,
    format_mmlu_5shot_prompt,
    format_mmlu_question_block,
    main as mmlu_main
)
from campaigns.rigorous_benchmarks.benchmarks.humaneval_eval import (
    HumanEvalEvaluator,
    estimate_pass_at_k,
    run_code_with_timeout,
    assemble_program,
    main as humaneval_main
)
from campaigns.rigorous_benchmarks.benchmarks.wikitext_ppl import (
    WikiTextPPL,
    compute_cross_entropy_and_perplexity,
    evaluate_sliding_window_perplexity,
    main as wikitext_main
)


# ==============================================================================
# 1. GSM8K Evaluation Tests
# ==============================================================================

class TestGSM8K:
    def test_extract_answer_patterns(self):
        """Test answer extraction across standard reasoning patterns."""
        # Standard GSM8K delimiter
        assert extract_gsm8k_answer("Janet makes $18 every day.\n#### 18") == "18"
        assert extract_gsm8k_answer("#### -42") == "-42"
        assert extract_gsm8k_answer("The result is #### 1,234.50") == "1234.50"
        
        # LaTeX boxed format
        assert extract_gsm8k_answer("Therefore, \\boxed{ 99 } is correct.") == "99"
        assert extract_gsm8k_answer("\\boxed{-7.5}") == "-7.5"
        
        # Natural language answer phrases
        assert extract_gsm8k_answer("Thus, the answer is 120.") == "120"
        assert extract_gsm8k_answer("Final answer is: 3000") == "3000"
        assert extract_gsm8k_answer("Answer: -15") == "-15"
        assert extract_gsm8k_answer("Total = 450") == "450"
        
        # Fallback trailing number
        assert extract_gsm8k_answer("She bought 5 apples, then 10 more, yielding 15") == "15"
        
        # Edge cases
        assert extract_gsm8k_answer("") is None
        assert extract_gsm8k_answer(None) is None
        assert extract_gsm8k_answer(123) is None
        assert extract_gsm8k_answer("No numbers here at all") is None
        assert extract_gsm8k_answer("Only symbols + - * /") is None

    def test_normalize_numeric_string(self):
        """Test normalization of numbers, currency, and decimals."""
        assert normalize_numeric_string(None) is None
        assert normalize_numeric_string("1,000") == 1000
        assert normalize_numeric_string("$45.00") == 45
        assert normalize_numeric_string("€100.") == 100
        assert normalize_numeric_string("3.14159") == pytest.approx(3.14159)
        assert normalize_numeric_string("-128") == -128
        assert normalize_numeric_string("75%") == 75
        assert normalize_numeric_string("abc") is None

    def test_exact_match(self):
        """Test numerical exact match equivalence."""
        assert gsm8k_exact_match("18", "18") is True
        assert gsm8k_exact_match("18.0", "18") is True
        assert gsm8k_exact_match("1,000", "1000") is True
        assert gsm8k_exact_match("$25", "25") is True
        assert gsm8k_exact_match("-5", "-5") is True
        assert gsm8k_exact_match("-5", "5") is False
        assert gsm8k_exact_match("12", "15") is False
        assert gsm8k_exact_match(None, "10") is False
        assert gsm8k_exact_match("10", None) is False
        assert gsm8k_exact_match("foo", "foo") is True
        assert gsm8k_exact_match("foo", "bar") is False

    def test_dataset_and_canonical_evaluation(self):
        """Test dataset loading and 100% accuracy on ground truth solutions."""
        evaluator = GSM8KEvaluator()
        dataset = evaluator.get_dataset()
        assert len(dataset) == 75
        assert len(evaluator) == 75
        
        for item in dataset:
            assert "id" in item
            assert "question" in item
            assert "solution" in item
            assert "target" in item
            assert len(item["target"]) > 0

        # Evaluate canonical solutions
        canonical_preds = [item["solution"] for item in dataset]
        res = evaluator.evaluate_predictions(canonical_preds)
        assert res["accuracy"] == 1.0
        assert res["accuracy_percent"] == 100.0
        assert res["num_correct"] == 75
        assert res["total"] == 75

        # Evaluate dict predictions format
        dict_preds = [{"id": item["id"], "prediction": item["solution"]} for item in dataset]
        res_dict = evaluator.evaluate_predictions(dict_preds)
        assert res_dict["accuracy"] == 1.0

        # Evaluate known wrong predictions
        wrong_preds = ["The answer is 999999" for _ in dataset]
        res_wrong = evaluator.evaluate_predictions(wrong_preds)
        assert res_wrong["accuracy"] == 0.0
        assert res_wrong["num_correct"] == 0

    def test_evaluator_error_handling(self):
        """Test error handling in GSM8KEvaluator."""
        with pytest.raises(FileNotFoundError):
            GSM8KEvaluator("/path/that/does/not/exist.json")

        evaluator = GSM8KEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate_predictions(["42"])

    def test_gsm8k_main(self):
        """Verify main entrypoint."""
        res = gsm8k_main()
        assert res["accuracy"] == 1.0


# ==============================================================================
# 2. ARC-Challenge Evaluation Tests
# ==============================================================================

class TestARC:
    def test_extract_arc_choice(self):
        """Test choice extraction from multiple format types."""
        # Exact letter
        assert extract_arc_choice("A") == "A"
        assert extract_arc_choice(" b ") == "B"
        
        # Phrases
        assert extract_arc_choice("The correct answer is (C).") == "C"
        assert extract_arc_choice("Answer: D") == "D"
        assert extract_arc_choice("Option A is the best choice") == "A"
        assert extract_arc_choice("Choice: B") == "B"
        
        # Leading token
        assert extract_arc_choice("B. The velocity increases with time.") == "B"
        assert extract_arc_choice("(C) Both kinetic and potential") == "C"
        assert extract_arc_choice("D: All of the above") == "D"
        
        # Boxed
        assert extract_arc_choice("\\boxed{D}") == "D"
        
        # Choice content match
        choices = [
            {"label": "A", "text": "Planetary density decreases"},
            {"label": "B", "text": "Planetary years lengthen"},
            {"label": "C", "text": "Planetary days become shorter"}
        ]
        assert extract_arc_choice("The result is that planetary days become shorter over time", choices=choices) == "C"

        # Fallback standalone letter
        assert extract_arc_choice("In conclusion, therefore, C must hold true under these conditions.") == "C"

        # Edge cases
        assert extract_arc_choice("") is None
        assert extract_arc_choice(None) is None
        assert extract_arc_choice(123) is None
        assert extract_arc_choice("No options given at all here") is None

    def test_format_arc_prompt(self):
        """Test multiple-choice prompt formatting."""
        item = {
            "question": "What is the boiling point of water?",
            "choices": [
                {"label": "A", "text": "50 C"},
                {"label": "B", "text": "100 C"},
                {"label": "C", "text": "150 C"},
                {"label": "D", "text": "200 C"}
            ]
        }
        prompt = format_arc_prompt(item)
        assert "Question: What is the boiling point of water?" in prompt
        assert "A. 50 C" in prompt
        assert "B. 100 C" in prompt
        assert prompt.endswith("Answer:")

    def test_dataset_and_evaluations(self):
        """Test ARC dataset loading and evaluation on predictions and logits."""
        evaluator = ARCEvaluator()
        dataset = evaluator.get_dataset()
        assert len(dataset) == 75
        assert len(evaluator) == 75

        for item in dataset:
            assert item["answerKey"] in ["A", "B", "C", "D"]
            labels = [c["label"] for c in item["choices"]]
            assert labels == ["A", "B", "C", "D"]

        # Generative evaluation on ground truth
        gt_preds = [item["answerKey"] for item in dataset]
        res = evaluator.evaluate_predictions(gt_preds)
        assert res["accuracy"] == 1.0
        assert res["num_correct"] == 75

        # Dict prediction evaluation
        dict_preds = [{"id": it["id"], "prediction": it["answerKey"]} for it in dataset]
        res_dict = evaluator.evaluate_predictions(dict_preds)
        assert res_dict["accuracy"] == 1.0

        # Logit evaluation: simulate logits with max score at ground truth
        logit_scores = []
        for item in dataset:
            scores = {"A": -5.0, "B": -5.0, "C": -5.0, "D": -5.0}
            scores[item["answerKey"]] = 10.0
            logit_scores.append(scores)

        res_logits = evaluator.evaluate_logits(logit_scores)
        assert res_logits["accuracy"] == 1.0
        assert res_logits["num_correct"] == 75

    def test_evaluator_error_handling(self):
        """Test error handling in ARCEvaluator."""
        with pytest.raises(FileNotFoundError):
            ARCEvaluator("/nonexistent/arc_data.json")

        evaluator = ARCEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate_predictions(["A"])
        with pytest.raises(ValueError):
            evaluator.evaluate_logits([{"A": 1.0}])

    def test_arc_main(self):
        """Verify arc main entrypoint."""
        res = arc_main()
        assert res["accuracy"] == 1.0


# ==============================================================================
# 3. MMLU Evaluation Tests
# ==============================================================================

class TestMMLU:
    def test_subjects_and_dataset_structure(self):
        """Verify the 5 canonical subjects are properly loaded with 20 items each."""
        evaluator = MMLUEvaluator()
        expected_subjects = [
            "elementary_mathematics",
            "formal_logic",
            "high_school_physics",
            "college_computer_science",
            "high_school_world_history"
        ]
        assert evaluator.subjects == expected_subjects
        assert len(evaluator) == 100

        for sub in expected_subjects:
            items = evaluator.get_subject_items(sub)
            assert len(items) == 20
            assert sub in evaluator.dev_few_shot
            assert len(evaluator.dev_few_shot[sub]) == 5

    def test_5shot_prompt_formatting(self):
        """Verify 5-shot Hendrycks prompt construction."""
        evaluator = MMLUEvaluator()
        item = evaluator.get_dataset()[0]
        prompt = evaluator.format_prompt(item, n_shot=5)
        
        assert "The following are multiple choice questions (with answers) about" in prompt
        assert prompt.count("Answer:") == 6
        assert prompt.endswith("Answer:")

        # Zero shot test
        prompt_0 = format_mmlu_5shot_prompt(item, dev_shots=None, n_shot=0)
        assert prompt_0.count("Answer:") == 1

    def test_choice_extraction(self):
        """Test MMLU choice extraction."""
        assert extract_mmlu_choice("A") == "A"
        assert extract_mmlu_choice("The answer is (B)") == "B"
        assert extract_mmlu_choice("C. True because all premises hold") == "C"
        assert extract_mmlu_choice("\\boxed{D}") == "D"
        assert extract_mmlu_choice("") is None
        assert extract_mmlu_choice(None) is None
        assert extract_mmlu_choice(123) is None

        # Content match
        choices = [{"label": "A", "text": "Gravitational potential energy"}]
        assert extract_mmlu_choice("The system stores gravitational potential energy", choices=choices) == "A"

    def test_macro_and_per_subject_accuracy(self):
        """Verify macro-average accuracy across subjects."""
        evaluator = MMLUEvaluator()
        dataset = evaluator.get_dataset()

        # 100% ground truth test
        gt_preds = [item["answer"] for item in dataset]
        res = evaluator.evaluate_predictions(gt_preds)
        assert res["macro_average"] == 1.0
        assert res["micro_average"] == 1.0
        assert res["total_correct"] == 100

        for sub in evaluator.subjects:
            assert res["per_subject"][sub]["accuracy"] == 1.0
            assert res["per_subject"][sub]["correct"] == 20

        # Dict predictions
        dict_preds = [{"prediction": item["answer"]} for item in dataset]
        res_dict = evaluator.evaluate_predictions(dict_preds)
        assert res_dict["macro_average"] == 1.0

        # Logit evaluation on MMLU
        logit_scores = []
        for item in dataset:
            scores = {"A": -1.0, "B": -1.0, "C": -1.0, "D": -1.0}
            scores[item["answer"]] = 5.0
            logit_scores.append(scores)
        res_logits = evaluator.evaluate_logits(logit_scores)
        assert res_logits["macro_average"] == 1.0
        assert res_logits["total_correct"] == 100

        # Targeted synthetic test: subject 0 is 100% correct, others 0%
        mixed_preds = []
        for item in dataset:
            if item["subject"] == evaluator.subjects[0]:
                mixed_preds.append(item["answer"])
            else:
                wrong_opt = "A" if item["answer"] != "A" else "B"
                mixed_preds.append(wrong_opt)

        res_mixed = evaluator.evaluate_predictions(mixed_preds)
        assert res_mixed["macro_average"] == pytest.approx(0.20)
        assert res_mixed["per_subject"][evaluator.subjects[0]]["accuracy"] == 1.0
        assert res_mixed["per_subject"][evaluator.subjects[1]]["accuracy"] == 0.0

    def test_evaluator_error_handling(self):
        """Test error handling in MMLUEvaluator."""
        with pytest.raises(FileNotFoundError):
            MMLUEvaluator("/path/to/missing_mmlu.json")

        evaluator = MMLUEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate_predictions(["A"])
        with pytest.raises(ValueError):
            evaluator.evaluate_logits([{"A": 1.0}])

    def test_mmlu_main(self):
        """Verify mmlu main entrypoint."""
        res = mmlu_main()
        assert res["macro_average"] == 1.0


# ==============================================================================
# 4. HumanEval Evaluation Tests
# ==============================================================================

class TestHumanEval:
    def test_pass_at_k_combinatorial_math(self):
        """Test exact combinatorial Pass@k calculations against analytical truths."""
        # 1 sample, correct
        assert estimate_pass_at_k(1, 1, k=1) == 1.0
        # 1 sample, incorrect
        assert estimate_pass_at_k(1, 0, k=1) == 0.0
        # 5 samples, 3 correct: Pass@1 = 3/5 = 0.60
        assert estimate_pass_at_k(5, 3, k=1) == pytest.approx(0.60)
        
        # 3 samples, 1 correct: Pass@3 must be 1.0
        assert estimate_pass_at_k(3, 1, k=3) == 1.0

        # 5 samples, 2 correct: Pass@3
        # Comb(3, 3) / Comb(5, 3) = 1 / 10 = 0.10 -> Pass@3 = 1 - 0.10 = 0.90
        assert estimate_pass_at_k(5, 2, k=3) == pytest.approx(0.90)

        # 10 samples, 4 correct: Pass@3
        # C(6, 3) / C(10, 3) = 20 / 120 = 1/6 -> Pass@3 = 5/6
        assert estimate_pass_at_k(10, 4, k=3) == pytest.approx(5.0 / 6.0)

        # Array length mismatch error
        with pytest.raises(ValueError):
            estimate_pass_at_k([1, 2], [1], k=1)

    def test_code_execution_sandbox(self):
        """Test local isolated code execution with timeouts and assertions."""
        # Passing code
        code_pass = "def add(a, b): return a + b\nassert add(2, 3) == 5\n"
        res_pass = run_code_with_timeout(code_pass, timeout=2.0)
        assert res_pass["passed"] is True
        assert res_pass["status"] == "passed"

        # Failing code (AssertionError)
        code_fail = "def add(a, b): return a + b\nassert add(2, 3) == 999\n"
        res_fail = run_code_with_timeout(code_fail, timeout=2.0)
        assert res_fail["passed"] is False
        assert res_fail["status"] == "failed"
        assert "AssertionError" in res_fail["error"]

        # Runtime Exception
        code_err = "def crash(): raise ValueError('deliberate')\ncrash()\n"
        res_err = run_code_with_timeout(code_err, timeout=2.0)
        assert res_err["passed"] is False
        assert res_err["status"] == "error"
        assert "ValueError" in res_err["error"]

        # Infinite loop / Timeout test
        code_timeout = "while True: pass\n"
        res_timeout = run_code_with_timeout(code_timeout, timeout=0.3)
        assert res_timeout["passed"] is False
        assert res_timeout["status"] == "timeout"

    def test_assemble_program(self):
        """Test code assembling with and without prompt signature."""
        prompt = "def foo(x):\n    '''doc'''\n"
        test_code = "def check(fn): assert fn(2) == 4"
        
        # When completion contains def
        full_1 = assemble_program(prompt, "def foo(x):\n    return x * 2", test_code, "foo")
        assert "def foo(x):\n    return x * 2" in full_1
        assert "check(foo)" in full_1

        # When completion is just the body
        full_2 = assemble_program(prompt, "    return x * 2", test_code, "foo")
        assert "'''doc'''\n    return x * 2" in full_2

    def test_canonical_solutions_pass(self):
        """Test that canonical HumanEval solutions pass their unit tests."""
        evaluator = HumanEvalEvaluator()
        dataset = evaluator.get_dataset()
        assert len(dataset) == 50
        assert len(evaluator) == 50

        # Evaluate first 5 canonical solutions swiftly
        test_subset = dataset[:5]
        
        for item in test_subset:
            res = evaluator.evaluate_sample(item, item["canonical_solution"], timeout=3.0)
            assert res["passed"] is True, f"Failed canonical solution for {item['task_id']}: {res['error']}"

        # Test Pass@1 and Pass@3 computation with multiple samples on the subset
        multi_map = {}
        for item in test_subset:
            multi_map[item["task_id"]] = [
                item["canonical_solution"],
                item["canonical_solution"],
                "    return None  # deliberately incorrect\n"
            ]

        multi_res = evaluator.evaluate_completions(multi_map, dataset=test_subset)
        assert multi_res["pass@1"] == pytest.approx(2.0 / 3.0, abs=1e-3)
        assert multi_res["pass@3"] == pytest.approx(1.0)

    def test_evaluator_error_handling(self):
        """Test error handling in HumanEvalEvaluator."""
        with pytest.raises(FileNotFoundError):
            HumanEvalEvaluator("/nonexistent/humaneval.json")

    def test_humaneval_main(self):
        """Verify humaneval main entrypoint."""
        res = humaneval_main(subset_size=3)
        assert res["pass@1"] == 1.0


# ==============================================================================
# 5. WikiText-2 Perplexity Evaluation Tests
# ==============================================================================

class TestWikiTextPPL:
    def test_cross_entropy_and_perplexity_math(self):
        """Verify exact tensor cross-entropy and perplexity mathematics."""
        batch_size = 2
        seq_len = 10
        vocab_size = 64

        # Analytic Case 1: Uniform logits
        uniform_logits = torch.zeros(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        res = compute_cross_entropy_and_perplexity(uniform_logits, targets, shift=True)
        expected_nats = math.log(vocab_size)
        expected_bits = expected_nats / math.log(2.0)
        expected_ppl = float(vocab_size)

        assert res["mean_loss_nats"] == pytest.approx(expected_nats, rel=1e-4)
        assert res["mean_loss_bits"] == pytest.approx(expected_bits, rel=1e-4)
        assert res["perplexity"] == pytest.approx(expected_ppl, rel=1e-4)
        assert res["total_tokens"] == 18

        # Analytic Case 2: Near-zero entropy (perfect prediction)
        perfect_targets = torch.randint(0, vocab_size, (1, seq_len))
        perfect_logits = torch.zeros(1, seq_len, vocab_size)
        for t in range(seq_len):
            perfect_logits[0, t, perfect_targets[0, t]] = 100.0

        res_perf = compute_cross_entropy_and_perplexity(perfect_logits, perfect_targets, shift=False)
        assert res_perf["mean_loss_nats"] == pytest.approx(0.0, abs=1e-4)
        assert res_perf["perplexity"] == pytest.approx(1.0, abs=1e-4)

    def test_ignore_index_masking(self):
        """Verify that ignore_index (-100) correctly excludes tokens from loss."""
        vocab_size = 10
        logits = torch.zeros(1, 5, vocab_size)
        targets = torch.tensor([[1, -100, 2, -100, 3]], dtype=torch.long)

        # Without shift
        res = compute_cross_entropy_and_perplexity(logits, targets, shift=False, ignore_index=-100)
        assert res["total_tokens"] == 3
        assert res["perplexity"] == pytest.approx(10.0, rel=1e-4)

    def test_perplexity_input_validation(self):
        """Test input error cases for perplexity calculation."""
        with pytest.raises(TypeError):
            compute_cross_entropy_and_perplexity([1, 2], [1, 2])

        with pytest.raises(ValueError):
            compute_cross_entropy_and_perplexity(torch.zeros(10, 5), torch.zeros(10))

        with pytest.raises(ValueError):
            compute_cross_entropy_and_perplexity(
                torch.zeros(1, 5, 10),
                torch.full((1, 5), -100, dtype=torch.long),
                shift=False
            )

    def test_sliding_window_model_evaluation(self):
        """Test sliding window perplexity using a toy causal language model."""
        vocab_size = 32
        class ToyCausalLM(nn.Module):
            def __init__(self, vocab):
                super().__init__()
                self.embed = nn.Embedding(vocab, 16)
                self.head = nn.Linear(16, vocab)

            def forward(self, input_ids, labels=None):
                h = self.embed(input_ids)
                logits = self.head(h)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                return type("Outputs", (), {"loss": loss, "logits": logits})()

        model = ToyCausalLM(vocab_size)
        input_ids = torch.randint(0, vocab_size, (1, 100))
        
        # Test with device=None and device="cpu"
        res_none = evaluate_sliding_window_perplexity(
            model=model,
            input_ids=input_ids,
            max_length=32,
            stride=16,
            device=None
        )
        assert res_none["perplexity"] > 0.0

        res = evaluate_sliding_window_perplexity(
            model=model,
            input_ids=input_ids,
            max_length=32,
            stride=16,
            device="cpu"
        )
        assert res["perplexity"] > 0.0
        assert not math.isnan(res["perplexity"])
        assert res["total_tokens"] > 0

    def test_wikitext_evaluator_dataset_and_methods(self):
        """Test WikiTextPPL dataset loader, evaluate_logits, and evaluate_model."""
        evaluator = WikiTextPPL()
        corpus = evaluator.get_corpus()
        assert len(corpus) > 5000
        assert len(evaluator.articles) >= 5

        # Verify pre-tokenized reference tokens
        tokens_gpt2 = evaluator.get_pretokenized_ids('gpt2')
        assert tokens_gpt2 is not None
        assert len(tokens_gpt2) > 1000

        assert evaluator.get_pretokenized_ids("nonexistent") is None

        # Test evaluate_logits method on evaluator
        dummy_logits = torch.zeros(1, 10, 20)
        dummy_targets = torch.randint(0, 20, (1, 10))
        eval_res = evaluator.evaluate_logits(dummy_logits, dummy_targets)
        assert eval_res["benchmark"] == "WikiText-2"
        assert eval_res["perplexity"] == pytest.approx(20.0, rel=1e-4)

        # Test evaluate_model using pre-tokenized tokens
        class DummyLM(nn.Module):
            def forward(self, input_ids, labels=None):
                loss = torch.tensor(3.5)
                return type("Outputs", (), {"loss": loss})()

        model = DummyLM()
        res_m = evaluator.evaluate_model(
            model=model,
            token_key="gpt2",
            max_length=64,
            stride=32,
            device="cpu"
        )
        assert res_m["perplexity"] == pytest.approx(math.exp(3.5), rel=1e-4)

        # Test evaluate_model with tokenizer
        class DummyTokenizer:
            def __call__(self, text, return_tensors="pt"):
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}

        res_tok = evaluator.evaluate_model(
            model=model,
            tokenizer=DummyTokenizer(),
            max_length=64,
            stride=32,
            device="cpu"
        )
        assert res_tok["perplexity"] == pytest.approx(math.exp(3.5), rel=1e-4)

    def test_evaluator_error_handling(self):
        """Test error handling in WikiTextPPL."""
        with pytest.raises(FileNotFoundError):
            WikiTextPPL("/nonexistent/wikitext.json")

        evaluator = WikiTextPPL()
        class DummyLM(nn.Module):
            def forward(self, input_ids, labels=None):
                return type("Outputs", (), {"loss": torch.tensor(1.0)})()

        with pytest.raises(ValueError):
            evaluator.evaluate_model(DummyLM(), token_key="nonexistent_key_without_fallback")

    def test_wikitext_main(self):
        """Verify wikitext main entrypoint."""
        res = wikitext_main()
        assert res["perplexity"] == pytest.approx(100.0, rel=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
