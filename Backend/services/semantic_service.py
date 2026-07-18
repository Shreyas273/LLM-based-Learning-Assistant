from services.vector_db_service import query_documents, query_documents_async
import logging
from typing import Any, Dict, List, Optional, Tuple
import asyncio

try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
except Exception as e:
    logging.warning(f"Could not load CrossEncoder: {e}")
    reranker = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logging.warning("rank_bm25 not installed, BM25 scoring will be skipped")
    BM25Okapi = None

def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default

def _extract_pdf_and_page(metadata: Optional[Dict[str, Any]]) -> Tuple[str, Optional[int]]:
    """
    Normalize metadata into (pdf_name, page_number) without assuming schema.
    - pdf_name: prefers 'pdf_name', then 'source'
    - page_number: prefers 'page_number', then 'page'
    """
    meta = metadata or {}
    pdf_name = str(meta.get("pdf_name") or meta.get("source") or "").strip()
    page_raw = meta.get("page_number", meta.get("page", None))
    page_number = None if page_raw is None else _coerce_int(page_raw, default=0)
    return pdf_name, page_number

def _group_adjacent_chunks(
    items: List[Dict[str, Any]],
    *,
    max_gap: int = 1,
    max_group_chars: int = 2200,
) -> List[Dict[str, Any]]:
    """
    Groups adjacent chunks from the same pdf/page to improve context coherence.
    Adjacency is based on chunk_id distance <= max_gap when available.
    Returns a list of grouped items: {text, metadata, score, chunk_ids}.
    """
    if not items:
        return []

    # Sort by (pdf, page, chunk_id) so we can sweep and group.
    def sort_key(it: Dict[str, Any]) -> Tuple[str, int, int]:
        meta = it.get("metadata") or {}
        pdf, page = _extract_pdf_and_page(meta)
        page_i = -1 if page is None else int(page)
        chunk_id = _coerce_int(meta.get("chunk_id", meta.get("chunk", -1)), default=-1)
        return (pdf, page_i, chunk_id)

    sorted_items = sorted(items, key=sort_key)

    grouped: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for it in sorted_items:
        meta = it.get("metadata") or {}
        pdf, page = _extract_pdf_and_page(meta)
        chunk_id = _coerce_int(meta.get("chunk_id", meta.get("chunk", -1)), default=-1)
        score = float(it.get("score") or 0.0)
        text = (it.get("text") or "").strip()

        if not text:
            continue

        if cur is None:
            cur = {
                "text": text,
                "metadata": meta,
                "score": score,
                "chunk_ids": [chunk_id] if chunk_id != -1 else [],
            }
            continue

        cur_meta = cur.get("metadata") or {}
        cur_pdf, cur_page = _extract_pdf_and_page(cur_meta)
        cur_chunk_ids = cur.get("chunk_ids") or []
        last_chunk_id = cur_chunk_ids[-1] if cur_chunk_ids else None

        same_doc = (pdf and cur_pdf and pdf == cur_pdf) and (page is not None and cur_page is not None and int(page) == int(cur_page))
        adjacent = False
        if same_doc and last_chunk_id is not None and chunk_id != -1:
            adjacent = (chunk_id - last_chunk_id) <= max_gap and (chunk_id - last_chunk_id) >= 0

        if same_doc and (adjacent or (last_chunk_id is None and chunk_id == -1)):
            # Append, but keep groups bounded so prompts don't explode.
            if len(cur["text"]) + 2 + len(text) <= max_group_chars:
                cur["text"] = cur["text"] + "\n\n" + text
            else:
                # If group too long, start a new group (keeps ordering).
                grouped.append(cur)
                cur = {
                    "text": text,
                    "metadata": meta,
                    "score": score,
                    "chunk_ids": [chunk_id] if chunk_id != -1 else [],
                }
                continue

            cur["score"] = max(float(cur.get("score") or 0.0), score)
            if chunk_id != -1:
                cur["chunk_ids"] = cur_chunk_ids + [chunk_id]
        else:
            grouped.append(cur)
            cur = {
                "text": text,
                "metadata": meta,
                "score": score,
                "chunk_ids": [chunk_id] if chunk_id != -1 else [],
            }

    if cur is not None:
        grouped.append(cur)

    # Prefer higher scoring groups.
    grouped.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return grouped

def get_relevant_chunks(question, file_filter=None, top_k=3, previous_topic=None):
    """
    Deprecated: Returns only text chunks. Use retrieve_context for full data.
    Kept for backward compatibility if needed, but updated to use Vector DB.
    """
    context = retrieve_context(question, file_filter, top_k, previous_topic)
    return [item["text"] for item in context]

def get_confidence_score(chunks, question):
    """
    Deprecated: Calculates confidence based on passed chunks.
    Since we don't pass chunks anymore, this is tricky. 
    If chunks are passed (legacy), we return 0 or re-implement logic?
    Let's assumes this is strictly for the new flow.
    """
    # Logic moved to retrieve_context
    return 0.0

def retrieve_context(question, file_filter=None, top_k=6, previous_topic=None):
    """
    Advanced Hybrid Retrieval Pipeline:
    1. Vector Retrieval (ChromaDB) -> Top 60
    2. Sparse Retrieval (BM25) over candidates
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-Encoder Reranking
    5. Strict Threshold Filtering
    """
    where = None
    if file_filter:
        if isinstance(file_filter, list):
            where = {"source": {"$in": file_filter}}
        else:
            where = {"source": file_filter}
    
    # 1. Query Analysis (We use the pre-reformulated standalone query)
    search_query = question
    print(f"[RETRIEVAL] Executing search for: '{search_query}'")
    
    # Stage 1: Dense Retrieval (Fetch wide pool)
    fetch_k = max(top_k * 10, 40)
    results = query_documents(
        "pdf_knowledge",
        search_query,
        n_results=fetch_k,
        where=where
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Rank Dense Results
    # dist is L2 distance, lower is better. Sorting by dist ascending.
    dense_ranked = sorted(zip(docs, metas, dists), key=lambda x: x[2])
    
    # Stage 2: Sparse (BM25) Scoring
    # Since we lack a global Elasticsearch index, we apply BM25 over the wide dense pool.
    bm25_ranked = []
    if BM25Okapi:
        tokenized_docs = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = search_query.lower().split()
        bm25_raw_scores = bm25.get_scores(tokenized_query)
        
        # Sort by BM25 score descending
        bm25_ranked = sorted(zip(docs, metas, bm25_raw_scores), key=lambda x: x[2], reverse=True)

    # Stage 3: Reciprocal Rank Fusion (RRF)
    # RRF Score = 1 / (k + rank)
    k_rrf = 60
    rrf_scores = {}
    
    for rank, (doc, meta, _) in enumerate(dense_ranked):
        doc_id = meta.get("chunk_id", hash(doc)) # fallback to hash if no chunk_id
        rrf_scores[doc_id] = {"text": doc, "metadata": meta, "score": 1.0 / (k_rrf + rank + 1)}
        
    for rank, (doc, meta, score) in enumerate(bm25_ranked):
        # Only count if BM25 found actual overlaps
        if score > 0:
             doc_id = meta.get("chunk_id", hash(doc))
             if doc_id in rrf_scores:
                 rrf_scores[doc_id]["score"] += 1.0 / (k_rrf + rank + 1)
             else:
                 rrf_scores[doc_id] = {"text": doc, "metadata": meta, "score": 1.0 / (k_rrf + rank + 1)}

    # Sort by RRF and take top candidates for reranking
    fused_candidates = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:max(top_k * 3, 15)]

    # Stage 4: Cross-Encoder Reranking
    if reranker:
        rerank_pairs = [[search_query, item["text"]] for item in fused_candidates]
        try:
            cross_scores = reranker.predict(rerank_pairs)
            for item, score in zip(fused_candidates, cross_scores):
                # Normalize cross_scores roughly to 0-100 range
                norm_score = max(0, min(100, (score + 5) * 10))
                item["final_score"] = norm_score
        except Exception as e:
            logging.warning(f"Reranking failed: {e}")
            for item in fused_candidates:
                 # fallback mapping
                item["final_score"] = min(100, item["score"] * 3000) 
    else:
        for item in fused_candidates:
            item["final_score"] = min(100, item["score"] * 3000)

    # Final Sort
    final_results = sorted(fused_candidates, key=lambda x: x["final_score"], reverse=True)

    # Stage 5: Strict Threshold Filtering (Placement-Proofing against Hallucination)
    context_data = []
    THRESHOLD = 2.0

    # Pull a few extra so grouping adjacent chunks has room.
    preselect_k = max(top_k * 3, top_k)
    for item in final_results[:preselect_k]:
        if item["final_score"] < THRESHOLD:
            print(f"[RETRIEVAL] Low score: {item['final_score']:.1f}")
            continue
        
        context_data.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "score": item["final_score"]
        })

    # Safety fallback
    if not context_data and final_results:
        best = final_results[0]
        print("[RETRIEVAL] Fallback using best chunk")
        context_data.append({
            "text": best["text"],
            "metadata": best["metadata"],
            "score": best["final_score"]
        })

    # Stage 6: Group adjacent chunks to improve coherence (backward compatible structure).
    grouped = _group_adjacent_chunks(context_data)
    return grouped[:top_k]


async def retrieve_context_async(question, file_filter=None, top_k=6, previous_topic=None):
    """
    Async-friendly version of retrieve_context().
    Offloads blocking Chroma query and (optionally) reranker to avoid blocking FastAPI event loop.
    """
    where = None
    if file_filter:
        if isinstance(file_filter, list):
            where = {"source": {"$in": file_filter}}
        else:
            where = {"source": file_filter}

    search_query = question

    fetch_k = max(top_k * 10, 40)
    results = await query_documents_async(
        "pdf_knowledge",
        search_query,
        n_results=fetch_k,
        where=where,
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    dense_ranked = sorted(zip(docs, metas, dists), key=lambda x: x[2])

    bm25_ranked = []
    if BM25Okapi:
        tokenized_docs = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = search_query.lower().split()
        bm25_raw_scores = bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(zip(docs, metas, bm25_raw_scores), key=lambda x: x[2], reverse=True)

    k_rrf = 60
    rrf_scores = {}

    for rank, (doc, meta, _) in enumerate(dense_ranked):
        doc_id = meta.get("chunk_id", hash(doc))
        rrf_scores[doc_id] = {"text": doc, "metadata": meta, "score": 1.0 / (k_rrf + rank + 1)}

    for rank, (doc, meta, score) in enumerate(bm25_ranked):
        if score > 0:
            doc_id = meta.get("chunk_id", hash(doc))
            if doc_id in rrf_scores:
                rrf_scores[doc_id]["score"] += 1.0 / (k_rrf + rank + 1)
            else:
                rrf_scores[doc_id] = {"text": doc, "metadata": meta, "score": 1.0 / (k_rrf + rank + 1)}

    fused_candidates = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:max(top_k * 3, 15)]

    # Cross-encoder reranking can be expensive; offload if present.
    if reranker:
        rerank_pairs = [[search_query, item["text"]] for item in fused_candidates]
        try:
            cross_scores = await asyncio.to_thread(reranker.predict, rerank_pairs)
            for item, score in zip(fused_candidates, cross_scores):
                norm_score = max(0, min(100, (score + 5) * 10))
                item["final_score"] = norm_score
        except Exception as e:
            logging.warning(f"Reranking failed: {e}")
            for item in fused_candidates:
                item["final_score"] = min(100, item["score"] * 3000)
    else:
        for item in fused_candidates:
            item["final_score"] = min(100, item["score"] * 3000)

    final_results = sorted(fused_candidates, key=lambda x: x["final_score"], reverse=True)

    context_data = []
    THRESHOLD = 2.0
    preselect_k = max(top_k * 3, top_k)
    for item in final_results[:preselect_k]:
        if item["final_score"] < THRESHOLD:
            continue
        context_data.append({"text": item["text"], "metadata": item["metadata"], "score": item["final_score"]})

    if not context_data and final_results:
        best = final_results[0]
        context_data.append({"text": best["text"], "metadata": best["metadata"], "score": best["final_score"]})

    grouped = _group_adjacent_chunks(context_data)
    return grouped[:top_k]

def get_all_document_chunks(file_filter):
    """
    Retrieve all chunks for a specific document without semantic search.
    Used for 'explain whole pdf' queries.
    """
    if not file_filter:
        return []
        
    try:
        from services.vector_db_service import get_collection
        collection = get_collection("pdf_knowledge")
        if not collection:
            return []
            
        # Clean the file_filter just in case it has an extension
        clean_filter = file_filter.replace(".pdf", "").replace(".json", "")
            
        # Try both the exact and cleaned version
        results = collection.get(where={"source": clean_filter})
        
        # If no results, let's try a softer match by fetching a large amount and filtering in Python
        if not results or not results.get("documents"):
             print(f"Exact match for '{clean_filter}' failed. Trying partial match...")
             all_docs = collection.get()
             if not all_docs or not all_docs.get("documents"):
                 return []
                 
             combined = []
             for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
                 if meta and "source" in meta and clean_filter.lower() in str(meta["source"]).lower():
                     combined.append({
                         "text": doc,
                         "metadata": meta,
                         "sort_key": meta.get("page", 0) * 1000 + meta.get("chunk_id", 0)
                     })
             
             if not combined:
                 return []
                 
             combined.sort(key=lambda x: x["sort_key"])
             return [item["text"] for item in combined]
            
        # Optional: Sort chunks by chunk_id or page if that metadata exists
        combined = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            combined.append({
                "text": doc,
                "metadata": meta,
                # Simple sort key using page and chunk_id if available
                "sort_key": meta.get("page", 0) * 1000 + meta.get("chunk_id", 0)
            })
            
        combined.sort(key=lambda x: x["sort_key"])
        
        return [item["text"] for item in combined]
    except Exception as e:
        import logging
        logging.error(f"Error fetching all document chunks: {e}")
        return []