from datetime import datetime
from typing import Dict, Any

from database.repositories import insert_interaction


def save_interaction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public entrypoint for persisting an interaction.
    Delegates to the MongoDB repository layer.
    """
    payload = {
        **data,
        "timestamp": data.get("timestamp") or datetime.utcnow(),
    }
    return insert_interaction(payload)