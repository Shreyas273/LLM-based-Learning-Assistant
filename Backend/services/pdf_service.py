import shutil
import pdfplumber
import hashlib
from pathlib import Path


UPLOAD_DIR = Path("data/uploads")

def save_pdf(file):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(file_path)

def extract_and_chunk(pdf_path):
    import os
    import json
    from services.vector_db_service import enqueue_indexing
    from services.text_processing import clean_text, semantic_chunking
    from services.subject_classifier import classify_text
    from services.topic_extractor import extract_topics_from_chunks
    
    filename = os.path.basename(pdf_path)
    # Remove extension for source name if desired, or keep it. User sample used "biology.pdf" as source.
    # Let's keep filename as source.
    
    all_chunks = []
    all_metadatas = []
    final_text_chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text()
            if raw_text:
                # 1. Clean Text
                cleaned_text = clean_text(raw_text)
                
                # 2. Semantic Chunking
                page_chunks = semantic_chunking(cleaned_text)
                
                for chunk in page_chunks:
                    final_text_chunks.append(chunk)
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "pdf_name": filename.replace(".pdf", ""),
                        "source": filename.replace(".pdf", ""),  # backward compatible
                        "page_number": i + 1,
                        "page": i + 1,  # backward compatible
                        "chunk_id": len(all_chunks)
                    })
    
    # --- Subject + Topic pipeline (modular) ---
    # Use a short sample of chunks to classify/extract topics.
    sample_text = "\n\n".join(final_text_chunks[:12])
    subj = classify_text(sample_text)
    topics = extract_topics_from_chunks(final_text_chunks[:40], max_topics=10)

    # Persist PDF-level metadata (used by graph + UI; keeps old structure intact)
    try:
        os.makedirs("data/metadata", exist_ok=True)
        meta_path = os.path.join("data", "metadata", f"{filename.replace('.pdf','')}_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pdf_name": filename.replace(".pdf", ""),
                    "subject": subj.get("subject"),
                    "branch": subj.get("branch"),
                    "confidence": subj.get("confidence", 0.0),
                    "topics": topics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass

    # Attach subject/topics metadata to each chunk for retrieval/answers.
    for m in all_metadatas:
        m["subject"] = subj.get("subject")
        m["branch"] = subj.get("branch")
        m["topics"] = topics

    # Index into Vector DB (background queue; non-blocking)
    if all_chunks:
        enqueue_indexing("pdf_knowledge", all_chunks, all_metadatas)
        
    return final_text_chunks

def get_file_hash(file_path):
    with open(file_path,"rb") as f:
        return hashlib.md5(f.read()).hexdigest()