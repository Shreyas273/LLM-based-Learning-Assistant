from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import os
import re

from services.llm_service import ask_llm_for_json


BRANCHES = ["Computer Engineering (CE)", "Information Technology (IT)", "Electronics & Communication (EC)"]
SCHOOL_SUBJECTS = ["Physics", "Chemistry", "Mathematics", "Biology", "Computer Science"]


@dataclass
class SubjectPrediction:
    branch: Optional[str]
    subject: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"branch": self.branch, "subject": self.subject, "confidence": float(self.confidence)}


_RULE_KEYWORDS: Dict[str, List[str]] = {
    # B.Tech branches (broad)
    "Computer Engineering (CE)": [
        "data structure", "algorithm", "operating system", "os", "dbms", "database",
        "computer network", "cn", "compiler", "software engineering", "oop", "java", "c++", "python",
        "cpu scheduling", "deadlock", "normalization", "indexing", "sql", "tcp", "ip",
    ],
    "Information Technology (IT)": [
        "web", "http", "api", "rest", "frontend", "backend", "cloud", "devops", "docker", "kubernetes",
        "microservice", "distributed", "system design", "security", "authentication", "oauth",
        "react", "node", "express", "django", "flask",
    ],
    "Electronics & Communication (EC)": [
        "signal", "signals", "dsp", "digital signal processing", "communication", "modulation",
        "antenna", "rf", "microwave", "vlsi", "verilog", "vhdl", "embedded", "microcontroller",
        "semiconductor", "analog", "digital electronics", "network analysis", "control system",
    ],
    # School subjects
    "Physics": ["force", "motion", "kinematics", "newton", "work", "energy", "power", "gravitation", "thermodynamics", "optics", "electricity", "magnetism", "current", "voltage", "resistance"],
    "Chemistry": ["mole", "molality", "molarity", "atomic", "bond", "ionic", "covalent", "organic", "inorganic", "reaction", "equilibrium", "acid", "base", "ph", "periodic table"],
    "Mathematics": ["calculus", "derivative", "integration", "matrix", "vector", "probability", "statistics", "trigonometry", "algebra", "geometry", "differential equation", "limit"],
    "Biology": ["cell", "genetics", "dna", "rna", "enzyme", "protein", "photosynthesis", "respiration", "ecosystem", "evolution", "taxonomy", "human physiology"],
    "Computer Science": ["data structures", "algorithms", "programming", "computer networks", "operating systems", "dbms", "machine learning", "ai", "compiler", "complexity", "graph", "tree"],
}


def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _rule_score(text: str) -> Dict[str, float]:
    t = _normalize(text)
    scores: Dict[str, float] = {}
    for label, kws in _RULE_KEYWORDS.items():
        s = 0.0
        for kw in kws:
            if kw in t:
                s += 1.0
        if s > 0:
            scores[label] = s
    return scores


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    mx = max(scores.values())
    exps = {k: math.exp(v - mx) for k, v in scores.items()}
    denom = sum(exps.values()) or 1.0
    return {k: (v / denom) for k, v in exps.items()}


def _try_embedding_similarity(text: str) -> Optional[SubjectPrediction]:
    """
    Optional embeddings similarity against label descriptions.
    Uses sentence_transformers if available; otherwise returns None.
    """
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
    except Exception:
        return None

    model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        return None

    labels = BRANCHES + SCHOOL_SUBJECTS
    desc = {
        "Computer Engineering (CE)": "core computer engineering subjects: data structures, algorithms, OS, DBMS, networks, architecture, compilers",
        "Information Technology (IT)": "information technology subjects: web development, cloud, devops, distributed systems, security, software systems",
        "Electronics & Communication (EC)": "electronics and communication subjects: signals, DSP, communication systems, VLSI, embedded, analog and digital electronics",
        "Physics": "physics topics: mechanics, electricity, magnetism, optics, thermodynamics",
        "Chemistry": "chemistry topics: organic chemistry, inorganic chemistry, physical chemistry, reactions, bonding",
        "Mathematics": "mathematics topics: calculus, algebra, probability, statistics, matrices, differential equations",
        "Biology": "biology topics: cells, genetics, physiology, ecology, evolution",
        "Computer Science": "computer science topics: programming, algorithms, data structures, AI, databases, networks",
    }

    t = text or ""
    q_emb = model.encode(t, convert_to_tensor=True, normalize_embeddings=True)
    label_embs = model.encode([desc[l] for l in labels], convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, label_embs)[0]
    best_idx = int(sims.argmax().item())
    best_label = labels[best_idx]
    best_score = float(sims[best_idx].item())

    # Convert cosine similarity (~0..1) into a conservative confidence
    conf = max(0.0, min(1.0, (best_score - 0.25) / 0.5))
    branch = best_label if best_label in BRANCHES else None
    subject = best_label
    return SubjectPrediction(branch=branch, subject=subject, confidence=conf)


def _llm_fallback(text: str) -> SubjectPrediction:
    labels = BRANCHES + SCHOOL_SUBJECTS
    prompt = f"""
You classify academic content into ONE label from an allowed list.

Allowed labels:
{labels}

Rules:
- Return STRICT JSON only.
- Pick the single best matching label.
- Provide a confidence 0..1.

Text:
{(text or "")[:5000]}

Return:
{{
  "label": "one of allowed labels",
  "confidence": 0.0
}}
"""
    data = ask_llm_for_json(prompt)
    label = None
    conf = 0.2
    if isinstance(data, dict):
        label = str(data.get("label") or "").strip()
        try:
            conf = float(data.get("confidence", conf))
        except Exception:
            conf = conf
    if label not in labels:
        # default conservative fallback
        label = "Computer Science"
        conf = 0.2
    branch = label if label in BRANCHES else None
    return SubjectPrediction(branch=branch, subject=label, confidence=max(0.0, min(1.0, conf)))


def classify_text(text: str) -> Dict[str, Any]:
    """
    Classify a PDF (or question) into one of:
      - B.Tech branches: CE / IT / EC
      - School subjects: Physics / Chemistry / Mathematics / Biology / Computer Science

    Strategy:
      1) keyword rules
      2) embeddings similarity (if available)
      3) LLM fallback
    """
    rule_scores = _rule_score(text)
    if rule_scores:
        probs = _softmax(rule_scores)
        best_label = max(probs.items(), key=lambda x: x[1])[0]
        # confidence boosted if winner margin is high
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else sorted_probs[0]
        conf = max(0.0, min(1.0, 0.55 + margin))
        branch = best_label if best_label in BRANCHES else None
        return SubjectPrediction(branch=branch, subject=best_label, confidence=conf).to_dict()

    emb = _try_embedding_similarity(text)
    if emb and emb.confidence >= 0.35:
        return emb.to_dict()

    return _llm_fallback(text).to_dict()

