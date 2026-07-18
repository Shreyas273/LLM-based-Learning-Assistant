from datetime import datetime
from typing import Dict, Any, List, Optional

from database.mongo_connection import db


# Collections (None when Mongo is unavailable)
users_collection = db["users"] if db is not None else None
interactions_collection = db["interactions"] if db is not None else None


def insert_interaction(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Insert a single interaction document.
    Returns None when Mongo is unavailable.
    """
    if interactions_collection is None:
        return None

    doc = {
        "userId": payload["userId"],
        "question": payload["question"],
        "answer": payload["answer"],
        "topic": payload.get("topic"),
        "confidence": float(payload.get("confidence", 0.0)),
        "masteryLevel": payload.get("masteryLevel", "unknown"),
        "difficulty": payload.get("difficulty", "medium"),
        "responseTime": payload.get("responseTime"),
        "correct": bool(payload.get("correct", False)),
        "sessionId": payload.get("sessionId"),
        "timestamp": payload.get("timestamp") or datetime.utcnow(),
    }
    interactions_collection.insert_one(doc)
    return doc


def find_user_interactions(user_id: str, session_id: str | None = None) -> List[Dict[str, Any]]:
    """
    Return all interactions for a given user (and optional session), sorted by timestamp.
    Returns [] when Mongo is unavailable.
    """
    if interactions_collection is None:
        return []

    query: Dict[str, Any] = {"userId": user_id}
    if session_id and session_id != "current":
        query["sessionId"] = session_id

    sort_dir = -1 if session_id == "current" else 1
    cursor = interactions_collection.find(query).sort("timestamp", sort_dir)
    interactions = list(cursor)

    if session_id == "current":
        interactions = interactions[:50]

    return interactions
