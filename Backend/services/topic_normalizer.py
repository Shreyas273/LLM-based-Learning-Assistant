from __future__ import annotations

import re
from typing import Dict, Iterable, List


_ACRONYMS = {
    "os": "OS",
    "dbms": "DBMS",
    "cn": "CN",
    "tcp": "TCP",
    "ip": "IP",
    "api": "API",
    "oop": "OOP",
    "sql": "SQL",
    "dns": "DNS",
    "http": "HTTP",
    "ai": "AI",
    "ml": "ML",
    "nlp": "NLP",
    "vlsi": "VLSI",
    "dsp": "DSP",
    "iot": "IoT",
}


_SYNONYMS: Dict[str, str] = {
    "binary trees": "Binary Tree",
    "binary tree": "Binary Tree",
    "bst": "Binary Search Tree",
    "binary search trees": "Binary Search Tree",
    "binary search tree": "Binary Search Tree",
    "graphs": "Graph",
    "graph traversal": "Graph Traversal",
    "dfs": "Depth First Search",
    "depth-first search": "Depth First Search",
    "bfs": "Breadth First Search",
    "breadth-first search": "Breadth First Search",
    "dp": "Dynamic Programming",
    "dynamic programming": "Dynamic Programming",
    "sorting algorithms": "Sorting",
    "searching algorithms": "Searching",
}


def _title_case_with_acronyms(s: str) -> str:
    words = re.split(r"\s+", s.strip())
    out = []
    for w in words:
        lw = w.lower()
        if lw in _ACRONYMS:
            out.append(_ACRONYMS[lw])
        elif re.fullmatch(r"[a-z]+", lw):
            out.append(lw.capitalize())
        else:
            # keep mixed tokens like "B-Tree", "O(n)"
            out.append(w)
    return " ".join(out).strip()


def _is_sentence_like(s: str) -> bool:
    # Reject if it's too long or contains common sentence punctuation patterns.
    if len(s) > 60:
        return True
    if any(p in s for p in [".", ";", ":", "?", "!", "\n"]):
        return True
    # Reject if it looks like a clause ("is a", "that", etc.)
    if re.search(r"\b(is|are|was|were|that|which|works|based)\b", s.lower()):
        return True
    return False


def normalize_topic(topic: str) -> str:
    t = (topic or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" -•\t")
    if not t:
        return ""

    # Strip surrounding quotes/backticks
    t = t.strip("\"'`")

    # Apply synonym mapping (case-insensitive)
    key = t.lower()
    if key in _SYNONYMS:
        t = _SYNONYMS[key]

    # Normalize capitalization
    t = _title_case_with_acronyms(t)

    # Clean trailing punctuation
    t = t.rstrip(".,;:!?)").lstrip("(").strip()

    return t


def normalize_topics(topics: Iterable[str], *, max_topics: int = 10) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in topics or []:
        t = normalize_topic(str(raw))
        if not t:
            continue
        if _is_sentence_like(t):
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= max_topics:
            break
    return out

