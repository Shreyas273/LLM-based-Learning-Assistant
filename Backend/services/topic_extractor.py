from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from services.llm_service import ask_llm_for_json
from services.topic_normalizer import normalize_topics


def extract_topics_from_text(text: str, *, max_topics: int = 10) -> List[str]:
    """
    Use Gemini (via ask_llm_for_json) to extract clean academic topics.

    Rules enforced:
    - return only topic names (no sentences)
    - max 10 topics (configurable)
    - normalized naming + dedupe
    """
    snippet = (text or "").strip()
    if len(snippet) > 8000:
        snippet = snippet[:8000]

    prompt = f"""
You are an academic topic extractor.

Task:
Extract up to {max_topics} REAL academic topics and concepts from the text.

STRICT RULES:
- Output ONLY a JSON array of strings.
- Each item must be a SHORT topic name (2-6 words typical), not a sentence.
- Do NOT include definitions, examples, or chunk text.
- Do NOT include time complexities or formulas.
- Do NOT include filler like "introduction", "conclusion", "chapter", "section".
- Avoid duplicates and near-duplicates.
- Use standard academic naming (Title Case).

Bad examples (DO NOT do this):
- "binary search tree is a node based..."
- "algorithm that works in O(n log n)"

Good examples:
["Binary Search Tree", "Graph Traversal", "Dynamic Programming"]

TEXT:
{snippet}
"""

    data = ask_llm_for_json(prompt)
    topics: List[str] = []
    if isinstance(data, list):
        topics = [str(x) for x in data]
    elif isinstance(data, dict) and isinstance(data.get("topics"), list):
        topics = [str(x) for x in data.get("topics", [])]

    return normalize_topics(topics, max_topics=max_topics)


def extract_topics_from_chunks(chunks: Union[List[str], str], *, max_topics: int = 10) -> List[str]:
    if isinstance(chunks, str):
        return extract_topics_from_text(chunks, max_topics=max_topics)
    text = "\n\n".join([str(c) for c in (chunks or []) if str(c).strip()])
    return extract_topics_from_text(text, max_topics=max_topics)

