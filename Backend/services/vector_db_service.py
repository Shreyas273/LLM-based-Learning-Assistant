import chromadb
from chromadb.config import Settings
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import threading
import queue
import asyncio

# Initialize ChromaDB client
# stored in Backend/data/chroma_db
CHROMA_DB_DIR = os.path.join(os.getcwd(), "data", "chroma_db")
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

try:
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    client = None

# --- Optional embedding model (batch) ---
_embedder = None
_embedder_lock = threading.Lock()

def _get_embedder():
    """
    Lazily initialize a sentence-transformers embedder if available.
    If not available, returns None and Chroma will use its default embedding behavior.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"))
        except Exception:
            _embedder = None
    return _embedder

def _batch_embed(texts: List[str], batch_size: int = 64) -> Optional[List[List[float]]]:
    model = _get_embedder()
    if model is None:
        return None
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        vecs = model.encode(chunk, show_progress_bar=False, normalize_embeddings=True)
        # vecs can be numpy array; convert to nested lists
        embeddings.extend([v.tolist() for v in vecs])
    return embeddings

def get_collection(name: str):
    """Get or create a collection"""
    if not client:
        return None
    return client.get_or_create_collection(name=name)

def add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
    *,
    batch_size: int = 256,
):
    """
    Add documents to a specific collection.
    If ids are not provided, UUIDs will be generated.
    """
    if not client:
        print("ChromaDB client is not initialized.")
        return False
        
    try:
        collection = get_collection(collection_name)
        if not ids:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Batch embedding generation (optional) + chunked add to reduce memory spikes.
        for i in range(0, len(documents), batch_size):
            docs_batch = documents[i : i + batch_size]
            metas_batch = metadatas[i : i + batch_size]
            ids_batch = ids[i : i + batch_size]

            embeds = _batch_embed(docs_batch, batch_size=min(64, max(8, len(docs_batch))))
            if embeds is not None:
                collection.add(
                    documents=docs_batch,
                    metadatas=metas_batch,
                    ids=ids_batch,
                    embeddings=embeds,
                )
            else:
                collection.add(
                    documents=docs_batch,
                    metadatas=metas_batch,
                    ids=ids_batch,
                )
        return True
    except Exception as e:
        print(f"Error adding documents to {collection_name}: {e}")
        return False

def query_documents(collection_name: str, query_text: str, n_results: int = 3, where: Optional[Dict] = None):
    """
    Query the collection for similar documents.
    """
    if not client:
        return []
        
    try:
        collection = get_collection(collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        return results
    except Exception as e:
        print(f"Error querying {collection_name}: {e}")
        return []


# --- Async wrappers (FastAPI-friendly) ---
async def query_documents_async(collection_name: str, query_text: str, n_results: int = 3, where: Optional[Dict] = None):
    """
    Async-friendly wrapper around the (blocking) Chroma query.
    Offloads to a thread so FastAPI event loop isn't blocked.
    """
    return await asyncio.to_thread(query_documents, collection_name, query_text, n_results, where)

async def add_documents_async(
    collection_name: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
    *,
    batch_size: int = 256,
):
    """
    Async-friendly wrapper around add_documents (blocking).
    """
    return await asyncio.to_thread(add_documents, collection_name, documents, metadatas, ids, batch_size=batch_size)


# --- Background indexing queue ---
_index_queue: "queue.Queue[Tuple[str, List[str], List[Dict[str, Any]], Optional[List[str]], int]]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()

def _index_worker():
    while True:
        try:
            collection_name, documents, metadatas, ids, batch_size = _index_queue.get()
            try:
                add_documents(collection_name, documents, metadatas, ids, batch_size=batch_size)
            finally:
                _index_queue.task_done()
        except Exception:
            # Keep worker alive
            continue

def ensure_index_worker_started():
    """
    Safe to call multiple times.
    Starts a daemon thread that drains the indexing queue.
    """
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_index_worker, name="chroma-index-worker", daemon=True)
        t.start()
        _worker_started = True

def enqueue_indexing(
    collection_name: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
    *,
    batch_size: int = 256,
) -> bool:
    """
    Queue indexing work to run in background worker thread.
    Returns True if enqueued.
    """
    ensure_index_worker_started()
    _index_queue.put((collection_name, documents, metadatas, ids, batch_size))
    return True

def delete_collection(collection_name: str):
    """Delete a collection"""
    if not client:
        return False
    try:
        client.delete_collection(collection_name)
        return True
    except Exception as e:
        print(f"Error deleting collection {collection_name}: {e}")
        return False

def list_collections():
    if not client:
        return []
    return client.list_collections()
