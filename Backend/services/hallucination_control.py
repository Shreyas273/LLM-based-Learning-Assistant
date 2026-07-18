import json
import os
import re
from typing import Dict, List, Tuple
from services.vector_db_service import query_documents
from services.llm.llm_router import LLMRouter

_llm = LLMRouter()
_embedding_model = None
_util = None


def _get_embedding_model():
    """Lazy-load embedding model; skipped when SKIP_LOCAL_EMBEDDINGS=1."""
    global _embedding_model, _util
    if os.getenv("SKIP_LOCAL_EMBEDDINGS", "0").lower() in ("1", "true", "yes"):
        return None, None
    if _embedding_model is not None:
        return _embedding_model, _util
    try:
        from sentence_transformers import SentenceTransformer, util

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        _util = util
    except Exception as e:
        print(f"Embedding model unavailable: {e}")
        _embedding_model = None
        _util = None
    return _embedding_model, _util


def verify_response_claims(response: str, context: str, query: str = "") -> Dict:
    """
    Advanced claim-level verification using LLM.
    Breaks response into claims and verifies each against context.
    """
    try:
        prompt = f"""
        You are a strict fact-checker for an AI assistant.
        
        Context (Source Material):
        {context}
        
        AI Response:
        {response}
        
        Task:
        1. Break the AI Response into individual factual claims.
        2. For each claim, verify if it is supported by the Context.
        3. If supported, quote the exact sentence from Context.
        4. If not supported, mark as UNSUPPORTED.
        
        Return JSON format:
        {{
            "claims": [
                {{
                    "claim": "The exact claim text",
                    "status": "SUPPORTED" or "UNSUPPORTED",
                    "evidence": "Quote from context or empty string"
                }}
            ],
            "supported_count": <number>,
            "unsupported_count": <number>,
            "overall_assessment": "Verified" or "Warning"
        }}
        """
        
        resp = _llm.generate(prompt + "\n\nOutput STRICT JSON only. No markdown.", temperature=0.0, max_tokens=900, top_p=0.9, top_k=40, seed=42)
        analysis = _llm.safe_json_extract(resp.text)
        return analysis if isinstance(analysis, dict) else {
            "claims": [],
            "supported_count": 0,
            "unsupported_count": 0,
            "overall_assessment": "Unverified",
            "error": "Invalid JSON from verifier",
        }
        
    except Exception as e:
        print(f"Fact check error: {e}")
        return {
            "claims": [],
            "supported_count": 0,
            "unsupported_count": 0,
            "overall_assessment": "Unverified",
            "error": str(e)
        }

def check_retrieval_quality(context_data: List[Dict]) -> str:
    """
    Check if retrieved context is relevant enough.
    Returns: "High", "Medium", "Low"
    """
    if not context_data:
        return "Low"
        
    # Check top score
    top_score = context_data[0].get('score', 0)
    
    # Thresholds (calibrated for Chroma l2 distance conversion)
    if top_score > 70:
        return "High"
    elif top_score > 40:
        return "Medium"
    else:
        return "Low"

def semantic_similarity_check(answer: str, context: str) -> float:
    """
    Calculate semantic similarity between answer and context using embeddings.
    Returns cosine similarity score (0.0 to 1.0).
    """
    try:
        embedding_model, util = _get_embedding_model()
        if embedding_model is None or util is None:
            return 0.5
        answer_emb = embedding_model.encode(answer, convert_to_tensor=True)
        context_emb = embedding_model.encode(context, convert_to_tensor=True)
        
        # Compute cosine similarity
        score = util.pytorch_cos_sim(answer_emb, context_emb).item()
        return max(0.0, min(1.0, score)) # Clamp
    except Exception as e:
        print(f"Similarity check error: {e}")
        return 0.5 # Neutral fallback
