import re

def clean_text(text):
    """
    Cleans extracted PDF text.
    - Fixes broken lines (e.g. "sen-\ntence" -> "sentence").
    - Removes excessive whitespace.
    - Removes common noise like page numbers (basic heuristic).
    """
    if not text:
        return ""
    
    # fix hyphenated words at line end: "exam-\nple" -> "example"
    text = re.sub(r'-\n', '', text)
    
    # fix broken lines within sentences: "word\nword" -> "word word"
    # Heuristic: If line ends with lowercase or comma, likely continues.
    # But simple approach: replace single newline with space, double newline with newline.
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def semantic_chunking(text, chunk_size=800, overlap=150):
    """
    Chunks text semantically with overlap.
    1. Split by paragraphs (double newline).
    2. If paragraph > chunk_size, split by sentences.
    3. If sentence > chunk_size, split by hard limit.
    4. Recombine with overlap.
    """
    if not text:
        return []
        
    # 1. Split by paragraphs (approximate) - input text might be flat from clean_text, 
    # so we rely on sentence splitters mostly if paragraphs are lost.
    # But if clean_text preserved double newlines, use them.
    # For now, let's assume sophisticated splitting.
    
    # Simple recursive splitting strategy:
    # Split by ". " -> sentences.
    
    sentences = re.split(r'(?<=[.?!])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If single sentence is massive (unlikely but possible), force split it
        if sentence_len > chunk_size:
            # If current chunk has content, save it first
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
            
            # Split long sentence by character limit
            for i in range(0, sentence_len, chunk_size - overlap):
                chunks.append(sentence[i : i + chunk_size])
            continue
            
        # Check if adding sentence exceeds size
        if current_length + sentence_len + 1 > chunk_size:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= 50: # Validation: skip tiny chunks
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            # Overlap: keep last N chars or last few sentences?
            # Easier: keep last sentence[s] that fit in overlap size
            overlap_buffer = []
            overlap_len = 0
            for prev_s in reversed(current_chunk):
                if overlap_len + len(prev_s) < overlap:
                    overlap_buffer.insert(0, prev_s)
                    overlap_len += len(prev_s)
                else:
                    break
            
            current_chunk = overlap_buffer + [sentence]
            current_length = overlap_len + len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + 1
            
    # Add last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= 50:
            chunks.append(chunk_text)
            
    return chunks
