from pymongo import MongoClient
import os


MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI is not set. Configure it in your environment or .env file."
    )

client = MongoClient(MONGO_URI)

# Single logical database for the app
db = client["ai_learning_studio"]
