import json
from services.llm.llm_router import LLMRouter

_llm = LLMRouter()

def classify_intent(question, selected_file=None):
    """
    Classify the user's question into an intent.
    Intents:
    - DOCUMENT_QUERY: Question about the specific uploaded document.
    - GENERAL_KNOWLEDGE: General question not needing the document.
    - CROSS_REFERENCE: Question asking to compare multiple documents.
    - TOOL_REQUEST: Request to use a tool (summarize, explain, solve).
    """
    
    file_context = f"User has selected file: {selected_file}" if selected_file else "No specific file selected."
    
    prompt = f"""
    You are an intent classifier for an educational AI assistant.
    
    Context:
    {file_context}
    
    User Question:
    "{question}"
    
    Classify the intent into EXACTLY ONE of these categories:
    - DOCUMENT_QUERY: The user is asking about the selected file content. (Only valid if a file is selected).
    - GENERAL_KNOWLEDGE: The user is asking a general conceptual question, or a question unrelated to the file.
    - CROSS_REFERENCE: The user explictly asks to compare, search across all files, or cross-reference.
    - TOOL_REQUEST: The user asks to summarize, explain specific code, or solve a problem explicitly using a tool.
    
    Return a JSON object:
    {{
        "intent": "CATEGORY",
        "confidence": 0.0 to 1.0,
        "reasoning": "brief explanation"
    }}
    """
    
    try:
        resp = _llm.generate(prompt + "\n\nOutput STRICT JSON only. No markdown.", temperature=0.0, max_tokens=200, top_p=0.9, top_k=40, seed=42)
        parsed = _llm.safe_json_extract(resp.text)
        return parsed if isinstance(parsed, dict) else {"intent": "GENERAL_KNOWLEDGE", "confidence": 0.5, "reasoning": "Invalid JSON from model."}
    except Exception as e:
        # Fallback logic
        print(f"Intent classification error: {e}")
        if selected_file:
            return {"intent": "DOCUMENT_QUERY", "confidence": 0.5, "reasoning": "Fallback to document query as file is selected."}
        else:
            return {"intent": "GENERAL_KNOWLEDGE", "confidence": 0.5, "reasoning": "Fallback to general knowledge."}
