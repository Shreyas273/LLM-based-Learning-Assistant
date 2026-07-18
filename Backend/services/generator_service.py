import asyncio
import json
import re
import time
from services.llm.llm_router import LLMRouter

_llm = LLMRouter()

async def generate_content(topic: str, task: str) -> dict:
    """Generate content with Schema Validation and Metadata"""
    start_time = time.time()
    
    result = None
    
    if task == "quiz":
        quiz_json = await _generate_quiz(topic)
        result = quiz_json
    else:
        # Standard generation for other tasks
        prompts = {
             "humanizer": f"Rewrite to sound natural:\n\n{topic}",
             "paper_writer": f"Write an academic paper on: '{topic}' with Abstract, Intro, Body, Conclusion.",
             "essay_writer": f"Write an essay on: '{topic}'.",
             "mind_map": f"Create a text-based mind map for: '{topic}'. Use indentation.",
             "answer": f"Answer concisely: '{topic}'"
        }
        
        prompt = prompts.get(task, f"Generate content for: {topic}")
        
        try:
            response = await asyncio.to_thread(_llm.generate, prompt, temperature=0.2, max_tokens=1200, top_p=0.9, top_k=40, seed=42)
            result = response.text
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}", "type": "api_error"}

    processing_time = round(time.time() - start_time, 2)
    return {
        "result": result,
        "meta": {
            "processing_time": processing_time,
            "tool": "content_generator",
            "task": task
        }
    }

async def _generate_quiz(topic):
    prompt = f"""
    Generate a Medium difficulty quiz with 5 multiple-choice questions on: '{topic}'.
    
    Output STRICT JSON format:
    {{
      "title": "Quiz Title",
      "questions": [
        {{
          "id": 1,
          "question": "Question text?",
          "options": ["A", "B", "C", "D"],
          "correct_answer": "Correct Option Text",
          "explanation": "Why it is correct"
        }}
      ]
    }}
    """
    
    for attempt in range(2):
        try:
            response = await asyncio.to_thread(_llm.generate, prompt + "\n\nOutput STRICT JSON only. No markdown.", temperature=0.0, max_tokens=1200, top_p=0.9, top_k=40, seed=42)
            text = response.text.strip()
            text = re.sub(r"```json\s*|\s*```", "", text)
            
            # Validate JSON
            data = json.loads(text)
            if "questions" in data:
                return data # Return object directly
        except Exception:
            continue
            
    return {"error": "Failed to generate valid quiz JSON.", "type": "schema_error"}
