import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# LLM Provider routing (default: ollama for offline operation)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

UPLOAD_DIR = "data/uploads"
EXTRACTED_TEXT_DIR = "data/extracted_text"
HISTORY_FILE = "data/history.json"
PROGRESS_FILE = "data/progress.json"

CHUNK_SIZE = 500
MAX_CHUNKS_PER_QUERY = 3

TOPICS = ["DSA", "OS", "DBMS", "CN", "OOP"]

MASTERY_THRESHOLDS = {
    "weak": 0.4,
    "average": 0.7,
    "strong": 1.0
}

FORGETTING_CURVE_STRENGTH_FACTOR = 1.0