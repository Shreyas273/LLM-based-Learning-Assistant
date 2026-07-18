import asyncio
import time
from services.llm.llm_router import LLMRouter

_llm = LLMRouter()

async def summarize_content(content: str, type: str = "general") -> dict:
    """
    Summarizes content with hierarchical chunking for long texts.
    Returns: { "summary": str, "meta": { "processing_time": float, "chunks": int } }
    """
    start_time = time.time()
    
    if not content or len(content.strip()) < 50:
        return {
            "summary": "Content is too short or empty to summarize.",
            "meta": {"processing_time": 0.0, "chunks": 0}
        }

    # Token/Char Limit for single pass
    MAX_CHUNK_SIZE = 12000 
    
    summary_text = ""
    chunk_count = 1
    
    if len(content) > MAX_CHUNK_SIZE:
        summary_text, chunk_count = await _recursive_summarize(content, MAX_CHUNK_SIZE)
    else:
        summary_text = await _generate_summary(content, type)
        
    processing_time = round(time.time() - start_time, 2)
    
    return {
        "summary": summary_text,
        "meta": {
            "processing_time": processing_time,
            "chunks": chunk_count,
            "tool": "summarizer"
        }
    }

async def _recursive_summarize(content, chunk_size):
    """Splits content, summarizes chunks, then summarizes the summaries."""
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    chunk_summaries = []
    
    for chunk in chunks:
        summary = await _generate_summary(chunk, is_partial=True)
        chunk_summaries.append(summary)
    
    combined_summary_text = "\n\n".join(chunk_summaries)
    final_summary = await _generate_summary(combined_summary_text, final_pass=True)
    
    return final_summary, len(chunks)

async def _generate_summary(text, type="general", is_partial=False, final_pass=False):
    style_instruction = ""
    if is_partial:
        style_instruction = "Focus on extracting key facts to start building a larger summary. Be concise."
    elif final_pass:
        style_instruction = """
        Create a Final Executive Summary from the provided partial summaries.
        Output strict markdown:
        # Executive Summary
        (1-2 sentences)
        ## Key Findings
        - Bullet points
        ## Conclusion
        """
    else:
         style_instruction = """
        Output strict markdown:
        # Summary
        (Concise overview)
        ## Main Points
        - Bullet points
        """
        
    prompt = f"""
    Summarize the following text.
    Context: {type}
    Instruction: {style_instruction}
    
    Text:
    {text[:30000]}
    """
    
    try:
        response = await asyncio.to_thread(_llm.generate, prompt, temperature=0.2, max_tokens=900, top_p=0.9, top_k=40, seed=42)
        return response.text.strip()
    except Exception as e:
        return f"Summary generation failed: {str(e)}"
