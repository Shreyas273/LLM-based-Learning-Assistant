"""
RAG evaluation framework for this project.

Inputs:
  - question: str
  - retrieved_context: str | list[str]
  - generated_answer: str

Metrics (0..1):
  - context_recall: proxy for "how much of the answer is grounded in retrieved context"
  - answer_faithfulness: proxy for "answer claims supported by context"
  - answer_relevance: "answer relevance to question"

Notes:
  - True context recall/faithfulness typically need a ground-truth reference or an LLM judge.
  - This script provides strong offline proxies with no required external services.
  - If `ragas` is installed and you configure it separately, you can enable it via --engine ragas.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


def _normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip().lower()
    # keep basic punctuation for sentence splitting, but normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    # Simple sentence splitter; good enough for evaluation heuristics
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> List[str]:
    t = _normalize_text(text)
    # words + numbers; drop 1-char noise
    toks = re.findall(r"[a-z0-9]{2,}", t)
    return toks


def _ngram_set(tokens: List[str], n: int) -> set[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cosine_sparse(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    # dot
    dot = 0.0
    # iterate smaller map
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    for k, v in vec_a.items():
        dot += v * vec_b.get(k, 0.0)
    # norms
    na = math.sqrt(sum(v * v for v in vec_a.values()))
    nb = math.sqrt(sum(v * v for v in vec_b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _tfidf_vector(text: str, *, idf: Dict[str, float]) -> Dict[str, float]:
    toks = _tokenize(text)
    if not toks:
        return {}
    tf: Dict[str, float] = {}
    for t in toks:
        tf[t] = tf.get(t, 0.0) + 1.0
    # log tf
    vec = {t: (1.0 + math.log(v)) * idf.get(t, 0.0) for t, v in tf.items()}
    return vec


def _build_idf(docs: List[str]) -> Dict[str, float]:
    """
    Builds IDF over a small local corpus (question, answer, context).
    Smooth IDF: log((N+1)/(df+1)) + 1
    """
    N = max(1, len(docs))
    df: Dict[str, int] = {}
    for d in docs:
        seen = set(_tokenize(d))
        for t in seen:
            df[t] = df.get(t, 0) + 1
    idf = {t: (math.log((N + 1) / (c + 1)) + 1.0) for t, c in df.items()}
    return idf


def _context_to_text(ctx: Union[str, List[str], None]) -> str:
    if ctx is None:
        return ""
    if isinstance(ctx, list):
        return "\n\n".join([str(x) for x in ctx if str(x).strip()])
    return str(ctx)


@dataclass
class RagEvalScores:
    context_recall: float
    answer_faithfulness: float
    answer_relevance: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "context_recall": float(self.context_recall),
            "answer_faithfulness": float(self.answer_faithfulness),
            "answer_relevance": float(self.answer_relevance),
        }


def evaluate_custom(question: str, retrieved_context: Union[str, List[str], None], generated_answer: str) -> RagEvalScores:
    """
    Offline, no-LLM evaluation proxies.

    - answer_relevance: cosine(question, answer) under local TF-IDF
    - answer_faithfulness: average per-sentence max similarity to context (n-gram Jaccard + tfidf cosine)
    - context_recall: fraction of answer sentences that have "enough" similarity to context (thresholded)
    """
    q = question or ""
    a = generated_answer or ""
    c = _context_to_text(retrieved_context)

    # Build local IDF from tiny corpus to avoid external deps
    idf = _build_idf([q, a, c])
    qv = _tfidf_vector(q, idf=idf)
    av = _tfidf_vector(a, idf=idf)
    answer_relevance = _cosine_sparse(qv, av)

    # If no context, faithfulness/recall should be 0
    if not c.strip():
        return RagEvalScores(context_recall=0.0, answer_faithfulness=0.0, answer_relevance=answer_relevance)

    cv = _tfidf_vector(c, idf=idf)

    # Sentence-level grounding checks
    sentences = _split_sentences(a)
    if not sentences:
        # empty answer
        return RagEvalScores(context_recall=0.0, answer_faithfulness=0.0, answer_relevance=answer_relevance)

    # Pre-tokenize context for n-gram overlap
    ctx_tokens = _tokenize(c)
    ctx_bigrams = _ngram_set(ctx_tokens, 2)
    ctx_trigrams = _ngram_set(ctx_tokens, 3)

    per_sent_support: List[float] = []
    supported_count = 0

    for s in sentences:
        sv = _tfidf_vector(s, idf=idf)
        sim_tfidf = _cosine_sparse(sv, cv)

        stoks = _tokenize(s)
        sb = _ngram_set(stoks, 2)
        st = _ngram_set(stoks, 3)
        sim_ng = 0.0
        if sb or st:
            # Weight trigrams higher than bigrams to reduce trivial matches
            sim_ng = 0.4 * _jaccard(sb, ctx_bigrams) + 0.6 * _jaccard(st, ctx_trigrams)

        # Combine (cap within 0..1)
        support = max(0.0, min(1.0, 0.65 * sim_tfidf + 0.35 * sim_ng))
        per_sent_support.append(support)

        if support >= 0.22:
            supported_count += 1

    answer_faithfulness = sum(per_sent_support) / len(per_sent_support)
    context_recall = supported_count / len(per_sent_support)

    return RagEvalScores(
        context_recall=float(max(0.0, min(1.0, context_recall))),
        answer_faithfulness=float(max(0.0, min(1.0, answer_faithfulness))),
        answer_relevance=float(max(0.0, min(1.0, answer_relevance))),
    )


def evaluate_ragas(question: str, retrieved_context: Union[str, List[str], None], generated_answer: str) -> RagEvalScores:
    """
    Optional RAGAS integration.

    RAGAS typically requires an embeddings model + LLM to judge faithfulness/relevance robustly.
    This function will try to run RAGAS if installed; otherwise it raises ImportError.
    """
    # Keep import local so the script works without ragas installed.
    from ragas import evaluate  # type: ignore
    from ragas.metrics import context_recall, faithfulness, answer_relevancy  # type: ignore
    from datasets import Dataset  # type: ignore

    ctx = retrieved_context
    if isinstance(ctx, str):
        contexts = [ctx]
    elif isinstance(ctx, list):
        contexts = [str(x) for x in ctx]
    else:
        contexts = [""]

    ds = Dataset.from_list(
        [
            {
                "question": question,
                "answer": generated_answer,
                "contexts": contexts,
            }
        ]
    )

    # NOTE: This relies on ragas default configuration (may need env/config in your setup).
    result = evaluate(ds, metrics=[context_recall, faithfulness, answer_relevancy])

    # Result is a Dataset-like object; convert carefully
    # Common access patterns: result.to_pandas() / dict(result)
    try:
        row = result.to_pandas().iloc[0].to_dict()
    except Exception:
        row = dict(result)[0] if isinstance(result, list) else dict(result)

    return RagEvalScores(
        context_recall=float(row.get("context_recall", 0.0)),
        answer_faithfulness=float(row.get("faithfulness", 0.0)),
        answer_relevance=float(row.get("answer_relevancy", 0.0)),
    )


def evaluate_single(
    question: str,
    retrieved_context: Union[str, List[str], None],
    generated_answer: str,
    *,
    engine: str = "custom",
) -> Dict[str, Any]:
    engine = (engine or "custom").lower().strip()
    if engine == "ragas":
        scores = evaluate_ragas(question, retrieved_context, generated_answer)
    else:
        scores = evaluate_custom(question, retrieved_context, generated_answer)
    return {
        "engine": engine,
        "input": {
            "question": question,
            "retrieved_context": retrieved_context,
            "generated_answer": generated_answer,
        },
        "scores": scores.to_dict(),
    }


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    # JSONL if multiple lines and first non-space isn't '['
    if "\n" in text and not text.lstrip().startswith("["):
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    return [obj]


def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG outputs (custom or RAGAS).")
    ap.add_argument("--engine", choices=["custom", "ragas"], default="custom", help="Evaluation engine to use.")
    ap.add_argument("--input", help="Path to JSON/JSONL with {question, retrieved_context, generated_answer}.")
    ap.add_argument("--question", help="Question text (single example).")
    ap.add_argument("--retrieved_context", help="Context text (single example).")
    ap.add_argument("--answer", help="Generated answer text (single example).")
    ap.add_argument("--out", help="Write results JSON to this file (optional).")
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []

    if args.input:
        rows = _read_json_or_jsonl(args.input)
        for row in rows:
            results.append(
                evaluate_single(
                    question=str(row.get("question", "")),
                    retrieved_context=row.get("retrieved_context"),
                    generated_answer=str(row.get("generated_answer", "")),
                    engine=args.engine,
                )
            )
    else:
        results.append(
            evaluate_single(
                question=args.question or "",
                retrieved_context=args.retrieved_context or "",
                generated_answer=args.answer or "",
                engine=args.engine,
            )
        )

    payload: Dict[str, Any] = {
        "count": len(results),
        "results": results,
        "aggregate": {
            "context_recall": round(sum(r["scores"]["context_recall"] for r in results) / max(1, len(results)), 4),
            "answer_faithfulness": round(sum(r["scores"]["answer_faithfulness"] for r in results) / max(1, len(results)), 4),
            "answer_relevance": round(sum(r["scores"]["answer_relevance"] for r in results) / max(1, len(results)), 4),
        },
    }

    out_text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        print(out_text)


if __name__ == "__main__":
    main()

