from pymongo import MongoClient
import os


MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://ai_user:strongpassword123@ai-learning-cluster.0xtenbe.mongodb.net/?appName=ai-learning-cluster",
)

client = MongoClient(MONGO_URI)

# Single logical database for the app
db = client["ai_learning_studio"]

