"""
Thinking Core Retrieval Store.

Implements:
1. High-Speed In-Memory BM25 Passage Index over Canonical Entity-Relation Corpus.
2. Context Injection Pipeline for Augmented Thinking Core Inference:
   - Retrieves top-k evidence passages for given factual queries.
   - Formulates conditioned prompt: "Context: [Passage]\nQuestion: [Query]\nAnswer:"
   - Evaluates factual extraction accuracy on compressed models.
"""

import math
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from src.lean.eval.probes import RELATIONS_DB


class BM25Index:
    """Fast, deterministic in-memory BM25 index."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_freqs: Counter = Counter()
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def fit(self, documents: List[Dict[str, Any]]):
        self.corpus = documents
        self.doc_lens = []
        self.doc_freqs = Counter()
        
        for doc in documents:
            toks = set(self.tokenize(doc["text"]))
            for t in toks:
                self.doc_freqs[t] += 1
            self.doc_lens.append(len(self.tokenize(doc["text"])))
            
        self.avg_doc_len = float(np_mean := sum(self.doc_lens) / max(1, len(self.doc_lens)))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        q_toks = self.tokenize(query)
        scores = []
        n_docs = len(self.corpus)
        
        for i, doc in enumerate(self.corpus):
            doc_toks = self.tokenize(doc["text"])
            doc_len = self.doc_lens[i]
            score = 0.0
            doc_counts = Counter(doc_toks)
            
            for t in q_toks:
                if t in self.doc_freqs:
                    df = self.doc_freqs[t]
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf = doc_counts.get(t, 0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                    score += idf * (tf * (self.k1 + 1.0)) / max(1e-4, denom)
                    
            scores.append((self.corpus[i], score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def build_canonical_knowledge_corpus() -> List[Dict[str, Any]]:
    """Builds a verified corpus of entity-relation passages from RELATIONS_DB."""
    documents = []
    doc_id = 0
    for rel_name, tmpl, facts in RELATIONS_DB:
        for sub, obj, freq in facts:
            # Construct a rich informative Wikipedia-style sentence
            text = f"{sub} is associated with {obj}. Specifically, {tmpl.format(sub=sub)} {obj}."
            documents.append({
                "doc_id": doc_id,
                "relation": rel_name,
                "subject": sub,
                "target": obj,
                "text": text,
                "frequency": freq
            })
            doc_id += 1
    return documents


class ThinkingCoreRetrievalPipeline:
    """Pairs a compressed Thinking Core model with an external BM25 retrieval store."""
    def __init__(self, index: Optional[BM25Index] = None):
        if index is not None:
            self.index = index
            self.corpus = getattr(index, "corpus", [])
        else:
            self.index = BM25Index()
            self.corpus = build_canonical_knowledge_corpus()
            self.index.fit(self.corpus)

    def augment_prompt(self, query: str, top_k: int = 1) -> str:
        results = self.index.search(query, top_k=top_k)
        if not results:
            return query
        evidence = results[0][0]["text"]
        return f"Passage: {evidence}\nQuestion: {query}"

    def evaluate_retrieval_augmented_accuracy(
        self,
        model: Any,
        tokenizer: Any,
        probes: List[Dict[str, Any]],
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Evaluates factual cloze accuracy when evidence passages are retrieved and injected.
        """
        import torch
        from collections import defaultdict
        
        model.eval()
        rel_candidates = defaultdict(set)
        for p in probes:
            rel_candidates[p["relation"]].add(p["target"])
            
        correct = 0
        total = len(probes)
        
        with torch.no_grad():
            for p in probes:
                raw_prompt = p["prompt"]
                target = p["target"]
                rel = p["relation"]
                cands = rel_candidates[rel]
                
                # Retrieve evidence and formulate prompt
                aug_prompt = self.augment_prompt(raw_prompt, top_k=1)
                
                inp = tokenizer(aug_prompt, return_tensors="pt").to(device)
                logits = model(**inp).logits[0, -1, :]
                
                scores = {}
                for c in cands:
                    c_id = tokenizer.encode(" " + c, add_special_tokens=False)
                    if not c_id:
                        c_id = tokenizer.encode(c, add_special_tokens=False)
                    scores[c] = float(logits[c_id[0]].item())
                    
                pred = max(scores.items(), key=lambda x: x[1])[0]
                if pred == target:
                    correct += 1
                    
        return {
            "accuracy": float(correct / max(1, total)),
            "correct_count": correct,
            "total_evaluated": total
        }
