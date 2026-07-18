import json
import os
import time
import hashlib
from typing import Any, Optional

CACHE_FILE = "data/llm_cache.json"

_redis_client = None
_redis_available = False

def _init_redis():
    """
    Lazy Redis init.
    - Uses REDIS_URL if provided, else host/port defaults.
    - Gracefully falls back to file cache if redis isn't installed/running.
    """
    global _redis_client, _redis_available
    if _redis_available or _redis_client is not None:
        return
    try:
        import redis  # type: ignore

        url = os.getenv("REDIS_URL")
        if url:
            _redis_client = redis.from_url(url, decode_responses=True)
        else:
            host = os.getenv("REDIS_HOST", "127.0.0.1")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))
            _redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

        # quick ping to validate
        _redis_client.ping()
        _redis_available = True
    except Exception:
        _redis_client = None
        _redis_available = False

def _cache_key(question: str) -> str:
    # Stable key, avoids huge raw strings in Redis
    digest = hashlib.sha256((question or "").encode("utf-8")).hexdigest()
    return f"llm:answer:{digest}"

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def get_cached_answer(question):
    """
    Primary: Redis (if available), fallback: JSON file.
    Returns cached object (often tuple payload) or None.
    """
    _init_redis()
    if _redis_available and _redis_client:
        try:
            raw = _redis_client.get(_cache_key(question))
            if not raw:
                return None
            return json.loads(raw)
        except Exception:
            # fall through to file cache
            pass

    cache = load_cache()
    return cache.get(question)

def store_answer(question, answer):
    """
    Primary: Redis (if available), fallback: JSON file.
    TTL is configurable via LLM_CACHE_TTL_SECONDS (default: 6 hours).
    """
    ttl = int(os.getenv("LLM_CACHE_TTL_SECONDS", str(6 * 60 * 60)))
    _init_redis()
    if _redis_available and _redis_client:
        try:
            payload = json.dumps(answer)
            if ttl > 0:
                _redis_client.setex(_cache_key(question), ttl, payload)
            else:
                _redis_client.set(_cache_key(question), payload)
            return
        except Exception:
            # fall through to file cache
            pass

    cache = load_cache()
    cache[question] = answer
    save_cache(cache)