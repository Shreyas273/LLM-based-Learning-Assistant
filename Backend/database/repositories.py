from datetime import datetime
from typing import Dict, Any, List

from database.mongo_connection import db


# Collections
users_collection = db["users"]
interactions_collection = db["interactions"]


def insert_interaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert a single interaction document.
    This is the only place that writes to the interactions collection.
    """
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
    """
    query: Dict[str, Any] = {"userId": user_id}
    if session_id and session_id != "current":
        query["sessionId"] = session_id

    sort_dir = -1 if session_id == "current" else 1
    cursor = interactions_collection.find(query).sort("timestamp", sort_dir)
    interactions = list(cursor)

    if session_id == "current":
        # Limit size for the "current" session view
        interactions = interactions[:50]

    return interactions

