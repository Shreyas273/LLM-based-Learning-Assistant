from fastapi import UploadFile
from PIL import Image
import io
import asyncio
import time
import os
from services.llm.llm_router import LLMRouter

_llm = LLMRouter()

async def analyze_image(file: UploadFile, task: str = "summarize") -> dict:
    """Analyze image using vision model with validation"""
    start_time = time.time()
    try:
        # Ollama llama3.1:8b is text-only. Keep API stable by returning a clear error.
        if os.getenv("LLM_PROVIDER", "ollama").lower().strip() == "ollama":
            return {"error": "Image analysis requires a vision-capable model. Configure a vision provider/model.", "type": "unsupported"}

        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        prompts = {
            "summarize": "Describe this image in detail and summarize any text or visual information present.",
            "math": """
            Solve the math problem shown in this image step-by-step.
            Format:
            **Analysis**: Identify the problem type.
            **Step 1**: ...
            **Step 2**: ...
            **Solution**: Final answer.
            """,
            "chemistry": "Solve the chemistry problem or explain the reaction shown. Show balancing/steps.",
            "physics": "Solve the physics problem. List knowns, unknowns, formulas, and steps.",
            "homework": "Act as a tutor. Explain how to solve this homework problem clearly."
        }
        
        prompt = prompts.get(task, prompts["summarize"])
        
        # If Gemini provider is enabled (optional), users can swap LLM_PROVIDER=gemini.
        # With the router in gemini mode, treat the image as unsupported for now.
        return {"error": "Image analysis is not enabled for the current provider.", "type": "unsupported"}
        
        processing_time = round(time.time() - start_time, 2)
        
        return {
            "result": response.text,
            "meta": {
                "processing_time": processing_time,
                "tool": "image_solver",
                "task": task
            }
        }
        
    except asyncio.TimeoutError:
         return {"error": "Image analysis timed out.", "type": "timeout"}
    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}", "type": "process_error"}

async def solve_problem(query: str, subject: str) -> dict:
    """Solve text problems with reasoning template"""
    start_time = time.time()
    
    if not query or len(query.strip()) < 3:
        return {"error": "Please provide a valid problem statement.", "type": "validation_error"}

    prompts = {
        "math": "Solve this math problem step-by-step.",
        "chemistry": "Solve this chemistry problem step-by-step.",
        "physics": "Solve this physics problem step-by-step.",
        "graphing": "Provide equation and key graph points (intercepts, vertex) for:",
        "homework": "Help solve this homework question:"
    }
    
    core_prompt = prompts.get(subject, "Solve:")
    
    full_prompt = f"""
    {core_prompt}
    
    Problem:
    {query}
    
    Format:
    **Problem Analysis**:
    **Step-by-Step Solution**:
    **Final Answer**:
    """
    
    try:
        response = await asyncio.to_thread(_llm.generate, full_prompt, temperature=0.2, max_tokens=900, top_p=0.9, top_k=40, seed=42)
        processing_time = round(time.time() - start_time, 2)
        
        return {
            "result": response.text.strip(),
            "meta": {
                "processing_time": processing_time,
                "tool": "text_solver",
                "subject": subject
            }
        }
    except Exception as e:
        return {"error": f"Solver error: {str(e)}", "type": "api_error"}
