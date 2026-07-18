from pymongo import MongoClient
import os
import logging

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "").strip()

client = None
db = None

if MONGO_URI:
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        db = client["ai_learning_studio"]
        # Force a quick ping so bad SRV/DNS fails here (and is caught) instead of later.
        client.admin.command("ping")
        logger.info("Connected to MongoDB")
    except Exception as exc:
        logger.warning("MongoDB unavailable; continuing without it: %s", exc)
        client = None
        db = None
else:
    logger.warning("MONGO_URI is not set; Mongo-backed features are disabled.")
