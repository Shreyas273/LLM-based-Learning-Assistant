import json
from enum import Enum
from pydantic import BaseModel, Field
from services.llm.llm_router import LLMRouter

class TaskType(str, Enum):
    DOCUMENT_QA = "document_qa"
    FULL_SUMMARY = "full_summary"
    COMPARISON = "comparison"
    GENERAL_KNOWLEDGE = "general_knowledge"

class RoutingDecision(BaseModel):
    task_type: TaskType
    target_documents: list[str] = Field(default_factory=list)
    confidence: float

_llm = LLMRouter()

def reformulate_query(query: str, last_topic: str = None, last_question: str = None) -> str:
    """
    Rewrites vague or follow-up queries into fully standalone, explicit search queries.
    Uses session context to fill in "it", "this", or missing subjects.
    """
    if set(query.split()) <= {"explain", "summarize", "it", "this", "briefly", "what", "is"}:
        # Query is extremely vague, definitely needs heavy rewriting if there is context
        pass
        
    prompt = f"""
    You are an expert Query Reformulation Engine.

    Your task is to rewrite incoming user queries to be fully self-contained, highly specific, and optimized for semantic vector search.
    If the user's query contains pronouns ("it", "they", "this", "that") or is a vague follow-up (e.g. "What are the advantages?"), you MUST replace the pronouns/vagueness with the explicit subject from the Conversation Context.
    
    <CONVERSATION_CONTEXT>
    Previous Topic: {last_topic or 'None'}
    Previous Question: {last_question or 'None'}
    </CONVERSATION_CONTEXT>
    
    <USER_QUERY>
    {query}
    </USER_QUERY>
    
    If the USER_QUERY is already fully standalone and specific, just return it unchanged.
    Output ONLY the rewritten query string. No extra text, no thoughts.
    """
    
    try:
        response = _llm.generate(prompt, temperature=0.0, max_tokens=120, top_p=0.9, top_k=40, seed=42)
        rewritten = response.text.strip()
        print(f"[REFORMULATOR] Original: '{query}' -> Rewritten: '{rewritten}'")
        return rewritten
    except Exception as e:
        # On quota or model failure, just fall back to the original query
        print(f"LLM error in reformulate_query, using original query. Error: {e}")
        return query

def classify_task_intent(standalone_query: str, available_documents: list[str] = None) -> TaskType:
    """
    Classifies the explicitly reformulated query into one of our predefined, robust route paths.
    Using structured JSON output to guarantee strict categorization.
    """
    available_docs_str = ", ".join(available_documents) if available_documents else "None"
    
    prompt = f"""
    You are an expert Intent Router for an AI learning assistant.
    Analyze the user's query and classify it into exactly one of these TaskTypes:
    
    1. "document_qa": The user is asking a specific factual question that requires searching the document(s). (e.g., "What is the consensus algorithm in MongoDB?", "How do B-Trees work?")
    2. "full_summary": The user explicitly wants a high-level overview, summary, or explanation of an ENTIRE document. (e.g., "Summarize the MongoDB Replication PDF", "Explain this whole document")
    3. "comparison": The user explicitly asks to compare two or more specific subjects or documents. (e.g., "Compare MongoDB and SQL", "What is the difference between...")
    4. "general_knowledge": The user is asking a broad question clearly unrelated to any uploaded material, or asking you to write code, or conversational chit-chat.

    Available Uploaded Documents: [{available_docs_str}]
    User Query: "{standalone_query}"

    Respond strictly with valid JSON matching this schema:
    {{
      "task_type": "string (one of: document_qa, full_summary, comparison, general_knowledge)",
      "confidence": "number (0.0 to 1.0)",
      "target_documents": ["string (list of document names explicitly or implicitly targeted, empty if none)"]
    }}
    """
    
    try:
        response = _llm.generate(prompt + "\n\nOutput STRICT JSON only. No markdown.", temperature=0.0, max_tokens=300, top_p=0.9, top_k=40, seed=42)
        raw_json = _llm.safe_json_extract(response.text.strip())
        if not isinstance(raw_json, dict):
            raise ValueError("Invalid JSON from router model")
        task_type_str = raw_json.get("task_type", "document_qa")
        
        # Enforce enum
        task_type = TaskType(task_type_str) if task_type_str in [e.value for e in TaskType] else TaskType.DOCUMENT_QA
        
        return RoutingDecision(
            task_type=task_type,
            target_documents=raw_json.get("target_documents", []),
            confidence=raw_json.get("confidence", 0.9)
        )
    except Exception as e:
        # On quota/model failure or bad JSON, fall back to a simple deterministic rule
        print(f"[ROUTER] Classification failed or quota exceeded: {e}. Falling back to heuristic routing.")
        if available_documents:
            return RoutingDecision(
                task_type=TaskType.DOCUMENT_QA,
                target_documents=available_documents,
                confidence=0.0,
            )
        else:
            return RoutingDecision(
                task_type=TaskType.GENERAL_KNOWLEDGE,
                target_documents=[],
                confidence=0.0,
            )