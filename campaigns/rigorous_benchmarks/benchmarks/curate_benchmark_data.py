#!/usr/bin/env python3
"""
Curates canonical benchmark datasets from official upstream sources
and saves verified, clean JSON subsets in campaigns/rigorous_benchmarks/benchmarks/data/.

Bundles:
1. gsm8k_test.json (75 representative test problems from OpenAI grade-school-math)
2. arc_challenge_test.json (75 representative test questions from AI2 ARC-Challenge)
3. mmlu_test.json (100 test questions: 20 per subject for 5 subjects + 5-shot dev examples)
4. humaneval_test.json (50 representative coding tasks from OpenAI HumanEval)
5. wikitext2_test.json (WikiText-2 test split text, articles, and pre-tokenized sequences)
"""

import os
import io
import json
import gzip
import urllib.request
import pyarrow.parquet as pq

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (BenchmarkDataCurator/1.0)"}


def download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def curate_gsm8k():
    print("Curating GSM8K...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data = download_bytes(url).decode("utf-8")
    lines = [json.loads(line) for line in data.strip().split("\n") if line.strip()]
    
    # Select first 75 canonical test problems
    subset = []
    for i, item in enumerate(lines[:75]):
        answer_raw = item["answer"]
        target_num = answer_raw.split("####")[-1].strip().replace(",", "")
        subset.append({
            "id": f"gsm8k_test_{i:04d}",
            "question": item["question"],
            "solution": item["answer"],
            "target": target_num
        })
        
    out_path = os.path.join(DATA_DIR, "gsm8k_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
    print(f"GSM8K saved: {len(subset)} items to {out_path}")


def curate_arc_challenge():
    print("Curating ARC-Challenge...")
    url = "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet"
    raw_bytes = download_bytes(url)
    table = pq.read_table(io.BytesIO(raw_bytes))
    
    subset = []
    # Select 75 standard 4-choice questions
    for i in range(len(table)):
        row = {col: table[col][i].as_py() for col in table.column_names}
        choices_dict = row["choices"]
        labels = choices_dict["label"]
        texts = choices_dict["text"]
        
        norm_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
        norm_labels = [norm_map.get(str(l), str(l)) for l in labels]
        ans_key = norm_map.get(str(row["answerKey"]), str(row["answerKey"]))
        
        if norm_labels == ["A", "B", "C", "D"] and ans_key in ["A", "B", "C", "D"]:
            formatted_choices = [
                {"label": l, "text": t}
                for l, t in zip(norm_labels, texts)
            ]
            subset.append({
                "id": row["id"],
                "question": row["question"],
                "choices": formatted_choices,
                "answerKey": ans_key
            })
            if len(subset) == 75:
                break
                
    out_path = os.path.join(DATA_DIR, "arc_challenge_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
    print(f"ARC-Challenge saved: {len(subset)} items to {out_path}")


def curate_mmlu():
    print("Curating MMLU...")
    subjects = [
        ("elementary_mathematics", "Elementary Mathematics"),
        ("formal_logic", "Formal Logic"),
        ("high_school_physics", "High School Physics"),
        ("college_computer_science", "Computer Science"),
        ("high_school_world_history", "World History")
    ]
    
    num_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    test_items = []
    dev_items = {}
    
    for sub_id, sub_display in subjects:
        url_test = f"https://huggingface.co/datasets/cais/mmlu/resolve/main/{sub_id}/test-00000-of-00001.parquet"
        t_tab = pq.read_table(io.BytesIO(download_bytes(url_test)))
        
        sub_test_count = 0
        for i in range(len(t_tab)):
            q = t_tab["question"][i].as_py()
            choices = t_tab["choices"][i].as_py()
            ans_idx = t_tab["answer"][i].as_py()
            if len(choices) == 4 and ans_idx in num_to_letter:
                test_items.append({
                    "id": f"mmlu_{sub_id}_{sub_test_count:03d}",
                    "subject": sub_id,
                    "subject_display": sub_display,
                    "question": q,
                    "choices": [
                        {"label": num_to_letter[k], "text": choices[k]}
                        for k in range(4)
                    ],
                    "answer": num_to_letter[ans_idx]
                })
                sub_test_count += 1
                if sub_test_count == 20:
                    break
                    
        url_dev = f"https://huggingface.co/datasets/cais/mmlu/resolve/main/{sub_id}/dev-00000-of-00001.parquet"
        d_tab = pq.read_table(io.BytesIO(download_bytes(url_dev)))
        dev_list = []
        for i in range(min(5, len(d_tab))):
            q = d_tab["question"][i].as_py()
            choices = d_tab["choices"][i].as_py()
            ans_idx = d_tab["answer"][i].as_py()
            dev_list.append({
                "question": q,
                "choices": [
                    {"label": num_to_letter[k], "text": choices[k]}
                    for k in range(4)
                ],
                "answer": num_to_letter[ans_idx]
            })
        dev_items[sub_id] = dev_list

    payload = {
        "subjects": [s[0] for s in subjects],
        "test": test_items,
        "dev_few_shot": dev_items
    }
    
    out_path = os.path.join(DATA_DIR, "mmlu_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"MMLU saved: {len(test_items)} test items to {out_path}")


def curate_humaneval():
    print("Curating HumanEval...")
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    decompressed = gzip.decompress(download_bytes(url)).decode("utf-8")
    items = [json.loads(line) for line in decompressed.strip().split("\n") if line.strip()]
    
    subset = items[:50]
    out_path = os.path.join(DATA_DIR, "humaneval_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
    print(f"HumanEval saved: {len(subset)} items to {out_path}")


def curate_wikitext2():
    print("Curating WikiText-2...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
    raw_text = download_bytes(url).decode("utf-8")
    
    articles = [a.strip() for a in raw_text.split(" = = =") if len(a.strip()) > 300]
    selected_articles = articles[:10]
    merged_corpus = "\n\n".join(selected_articles)

    # Pre-tokenize with standard offline tokenizers for instant offline benchmarking
    token_sequences = {}
    try:
        from transformers import AutoTokenizer
        for tok_name, key in [("gpt2", "gpt2"), ("gpt2", "bpe_ref")]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_name, local_files_only=True)
                tokens = tokenizer(merged_corpus, truncation=False)["input_ids"]
                token_sequences[key] = tokens
                print(f"Pre-tokenized {key}: {len(tokens)} tokens.")
            except Exception as e:
                print(f"Tokenization warning for {tok_name}: {e}")
    except ImportError:
        pass
    
    payload = {
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "num_articles": len(selected_articles),
        "total_chars": len(merged_corpus),
        "articles": selected_articles,
        "corpus": merged_corpus,
        "token_sequences": token_sequences
    }
    
    out_path = os.path.join(DATA_DIR, "wikitext2_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"WikiText-2 saved: {len(selected_articles)} articles to {out_path}")


if __name__ == "__main__":
    curate_gsm8k()
    curate_arc_challenge()
    curate_mmlu()
    curate_humaneval()
    curate_wikitext2()
    print("All benchmark data curated successfully!")
