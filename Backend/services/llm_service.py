from services.semantic_service import get_relevant_chunks, get_confidence_score
from services.cache_service import get_cached_answer, store_answer 
from utils.json_safe import convert_numpy
from typing import Any, Dict, List, Optional, Tuple
import json
import re
from services.llm.llm_router import LLMRouter

_llm = LLMRouter()

# --- Production Study Modes ---
MODE_TEMPLATES = {
    "story": {
        "system_instruction": "You are a creative storyteller. Explain complex concepts through engaging narratives.",
        "format_template": """
        STORY TEMPLATE:
        1. **Title**: A creative title for the story.
        2. **Characters**: List 2-3 characters involved.
        3. **The Story**: The explanation woven into a narrative (analogy).
        4. **The Lesson**: A summary of the technical concept explained in the story.
        """,
        "compatibility": ["fact_check_lax"] # Facts matter less than analogy here, but we still want truth.
    },
    "flowchart": {
        "system_instruction": "You are a process analyst. Break down systems into logical steps.",
        "format_template": """
        FLOWCHART TEMPLATE:
        1. **Start Node**: The initial state or input.
        2. **Process Steps**: Numbered steps (1, 2, 3...) detailing the logic.
        3. **Decision Points**: If/Else conditions (e.g., "If X, goto Step 4").
        4. **End Node**: The final output or state.
        """,
        "compatibility": ["strict"]
    },
    "exam_prep": {
        "system_instruction": "You are an exam coach. Focus on high-yield topics and scoring points.",
        "format_template": """
        EXAM PREP TEMPLATE:
        1. **Core Definition**: The standard academic definition.
        2. **Key Formula/Syntax**: relevant equations or code snippet.
        3. **3 Critical Points**: The most important facts to memorize.
        4. **Typical Exam Question**: A sample question asked in interviews/exams.
        """,
        "compatibility": ["strict"]
    },
    "socratic": {
        "system_instruction": "You are a Socratic tutor. Do not give the answer directly. Guide with questions.",
        "format_template": """
        SOCRATIC TEMPLATE:
        1. **Goal**: State what we are trying to figure out.
        2. **Guiding Question 1**: A question to prompt the student's thinking.
        3. **Hint**: A subtle clue if they are stuck.
        4. **Next Step**: Encouragement to solve the next part.
        """,
        "compatibility": ["interactive"]
    },
    "default": {
        "system_instruction": "You are an expert teaching assistant.",
        "format_template": """
        STANDARD TEMPLATE:
        - **Headings**: Use markdown headers for key sections.
        - **Explanation**: Clear, concise, and accurate text.
        - **Examples**: Short relevant examples.
        - **Key Takeaway**: One sentence summary.
        """,
        "compatibility": ["strict"]
    }
}

def get_mode_config(mode, enable_fact_check=True):
    """
    Returns (system_instruction, format_template) safely.
    Demotes creative modes if strict fact-checking is rigidly enforced (optional policy).
    """
    config = MODE_TEMPLATES.get(mode, MODE_TEMPLATES["default"])
    
    # Compatibility Check
    # If we had a mode that was purely fictional/joke, we might disable it here if fact_check is True.
    # For now, Story mode is fine as an analogy, but we enforce RAG constraints.
    
    return config["system_instruction"], config["format_template"]

def _safe_json_extract(text: str) -> Any:
    """
    Extract a JSON object/list from model text safely.
    Returns parsed JSON or {} on failure.
    """
    if not text:
        return {}
    raw = text.strip()
    raw = re.sub(r"```json\s*|\s*```", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except Exception:
        return {}

def rewrite_query(question: str, previous_topic: Optional[str] = None) -> str:
    """
    Rewrite user query into a standalone retrieval query.
    Backward compatible: on any failure returns original question.
    """
    try:
        prompt = f"""
You rewrite student questions into short, standalone search queries for retrieving chunks from PDFs.

Rules:
- Output ONLY the rewritten query as plain text.
- Keep it under 25 words.
- Preserve key entities, acronyms, and constraints.
- If the question is already standalone, return it unchanged.
- If there is a prior topic, use it only to resolve pronouns ("it", "this", "that").

Previous topic: {previous_topic or ""}
Question: {question}
"""
        resp = _llm.generate(prompt, temperature=0.0, max_tokens=128, top_p=0.9, top_k=40, seed=42)
        rewritten = (resp.text or "").strip()
        if not rewritten:
            return question
        # Avoid weird quoted output
        rewritten = rewritten.strip().strip('"').strip("'").strip()
        return rewritten or question
    except Exception:
        return question

def _is_comparison_question(q: str) -> bool:
    if not q:
        return False
    s = q.lower()
    triggers = [
        "compare", "comparison", "differentiate", "difference between", "differences between",
        "vs", "versus", "pros and cons", "advantages and disadvantages",
    ]
    return any(t in s for t in triggers)

def _build_citation_sources(context_data: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Builds a citation-tagged context string and a normalized sources list.
    Sources are normalized to include keys required downstream:
      {pdf, page, relevance_score} plus backward-compatible fields.
    """
    sources: List[Dict[str, Any]] = []
    parts: List[str] = []

    for idx, item in enumerate(context_data, start=1):
        meta = item.get("metadata") or {}
        pdf = str(meta.get("pdf_name") or meta.get("source") or "").strip()
        page = meta.get("page_number", meta.get("page", None))
        try:
            page = None if page is None else int(page)
        except Exception:
            page = None
        score = float(item.get("score") or 0.0)

        citation_id = f"S{idx}"
        parts.append(f"[{citation_id} | pdf={pdf} | page={page}] {item.get('text','')}")

        # Backward compatible: keep raw metadata merged in.
        src_obj = dict(meta)
        src_obj.update({
            "citation_id": citation_id,
            "pdf": pdf,
            "page": page,
            "relevance_score": score,
            # legacy-friendly aliases
            "pdf_name": pdf or src_obj.get("pdf_name") or src_obj.get("source"),
            "page_number": page if page is not None else src_obj.get("page_number", src_obj.get("page", None)),
        })
        sources.append(src_obj)

    context = "\n\n".join(parts).strip()
    return context, sources

def ask_llm(
    question,
    relevant_context=None,
    file_filter=None,
    difficulty: str = "medium",
    enable_fact_check: bool = True,
    study_approach: str = "default",
    previous_topic: str = None,
):
    cached_answer = get_cached_answer(question)
    if cached_answer:
        return cached_answer
    
    # 1. Logic Layer: Retrieval & Context Prep
    if relevant_context is None:
        from services.semantic_service import retrieve_context

        rewritten = rewrite_query(question, previous_topic=previous_topic)

        # Multi-hop retrieval for comparison questions (keeps API backward compatible).
        if _is_comparison_question(question):
            try:
                hop_prompt = f"""
You break comparison questions into 2-4 focused retrieval sub-queries.
Return STRICT JSON only:
{{
  "queries": ["subquery 1", "subquery 2"]
}}

Question: {question}
Rewritten base query: {rewritten}
"""
                hop_resp = _llm.generate(hop_prompt + "\n\nOutput STRICT JSON only. No markdown.", temperature=0.0, max_tokens=300, top_p=0.9, top_k=40, seed=42)
                hop_json = _safe_json_extract((hop_resp.text or "").strip())
                subqueries = hop_json.get("queries") if isinstance(hop_json, dict) else None
                if not isinstance(subqueries, list) or not subqueries:
                    subqueries = [rewritten]
            except Exception:
                subqueries = [rewritten]

            merged: List[Dict[str, Any]] = []
            seen = set()
            for sq in subqueries[:4]:
                sq_text = str(sq).strip()
                if not sq_text:
                    continue
                hop_ctx = retrieve_context(sq_text, file_filter, top_k=4, previous_topic=previous_topic)
                for it in hop_ctx:
                    meta = it.get("metadata") or {}
                    # De-dup by (source,page,chunk_id,text hash)
                    key = (
                        str(meta.get("source") or meta.get("pdf_name") or ""),
                        str(meta.get("page") or meta.get("page_number") or ""),
                        str(meta.get("chunk_id") or ""),
                        hash((it.get("text") or "").strip()),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(it)
            context_data = sorted(merged, key=lambda x: float(x.get("score") or 0.0), reverse=True)[:8]
        else:
            context_data = retrieve_context(rewritten, file_filter, previous_topic=previous_topic)
        
        # Retrieval Confidence Warning (Internal)
        if context_data and context_data[0]['score'] < 50.0:
            context_data = [] # Low confidence, proceed but maybe warn in logs
            
        context, sources = _build_citation_sources(context_data)
        confidence_score = context_data[0]['score'] if context_data else 0.0
    else:
        # Externally provided context (e.g. Cross-Ref)
        context = "\n".join(relevant_context)
        confidence_score = 0.0 
        sources = []

    if not context:
        context = "No specific study materials provided. Answer based on your general knowledge."
        confidence_score = 50.0

    # 2. Style Layer: Configuration
    # Fallback Mechanism: If a complex mode fails, we retry with default.
    attempts = [(study_approach, "primary")]
    if study_approach != "default":
        attempts.append(("default", "fallback"))
    
    last_error = None

    for mode, attempt_type in attempts:
        try:
            system_role, format_template = get_mode_config(mode, enable_fact_check)
            
            # 3. Prompt Construction (Decoupled)
            # - System Role: Identity & Constraints
            # - Context: The Knowledge (Truth)
            # - Template: The Output Structure
            
            prompt = f"""
<SYSTEM_ROLE>
{system_role}
You are helpful, accurate, and an expert educational assistant.
</SYSTEM_ROLE>

<STRICT_RULES>
1. Topic Continuity: If the question appears short or refers to previous discussion, assume it is about the most recently discussed topic. Do not switch documents unless explicitly requested.
2. Grounding: Answer ONLY based on the facts present in the provided <CONTEXT>.
3. NO HALLUCINATION: If the exact answer cannot be confidently derived from the <CONTEXT>, you MUST reply with: "The provided study materials do not contain the answer." Do not guess under any circumstances, unless explicitly falling back.
4. Fallback: Only if the context is completely empty or completely irrelevant, use your General Knowledge but explicitly prepend "Based on general knowledge, ...".
5. Level: Maintain a '{difficulty}' difficulty level appropriate for the student.
6. Formatting: You MUST follow the structure defined in the <OUTPUT_FORMAT> section exactly.
7. Citations: When you state a factual claim derived from context, cite the supporting snippet using bracketed source IDs like [S1] or [S2]. Use the IDs exactly as provided in the context headers.
</STRICT_RULES>

<OUTPUT_FORMAT>
{format_template}
</OUTPUT_FORMAT>

<CONTEXT>
{context}
</CONTEXT>

<STUDENT_QUERY>
{question}
</STUDENT_QUERY>
"""
            
            # 4. Deterministic Generation (Placement-Proof Level)
            try:
                response = _llm.generate(
                    prompt,
                    temperature=0.1,  # deterministic-ish for RAG
                    top_p=0.8,
                    top_k=40,
                    max_tokens=1200,
                    seed=42,
                )
            except Exception as e:
                # Hard fallback: Non-Gemini mode using raw context only
                print("Gemini quota or API error in ask_llm, using non-Gemini fallback:", e)
                if context:
                    snippet = context[:1200]
                    fallback_answer = (
                        "⚠️ Gemini API is currently unavailable.\n\n"
                        "Here is a direct excerpt-style answer from your study materials:\n\n"
                        f"{snippet}"
                    )
                    return fallback_answer, confidence_score, sources, {}
                else:
                    return (
                        "⚠️ Gemini API is currently unavailable and no study materials are indexed. "
                        "Please try again later.",
                        0.0,
                        [],
                        {},
                    )

            answer = response.text.strip()
            
            # Validation: If answer is empty or too short, treat as failure for creative modes
            if len(answer) < 50 and mode != "socratic":
                raise ValueError("Generated answer too short or empty.")

            # 5. Verification (Fact Check) - Only apply to successful generation
            from services.hallucination_control import verify_response_claims, semantic_similarity_check
            
            verification_result = {"status": "Unchecked"}
            final_confidence = confidence_score

            # Run Fact Check if enabled
            if enable_fact_check and context and len(context) > 100:
                try:
                    semantic_score = semantic_similarity_check(answer, context)
                    verification_analysis = verify_response_claims(answer, context, question)
                    
                    unsupported = verification_analysis.get("unsupported_count", 0)
                    if unsupported > 0:
                        answer += f"\n\n⚠️ **Warning**: {unsupported} claims in this answer could not be verified by the source."
                    
                    # Composite Score
                    sem_conf = semantic_score * 100
                    ver_conf = (verification_analysis.get("supported_count", 0) / max(1, len(verification_analysis.get("claims", [])))) * 100
                    final_confidence = (confidence_score * 0.4) + (sem_conf * 0.3) + (ver_conf * 0.3)
                    
                    verification_result = verification_analysis
                    verification_result["semantic_score"] = semantic_score
                    verification_result["status"] = "Verified" if unsupported == 0 else "Warning"
                except Exception as fc_error:
                    print(f"Verification Error: {fc_error}")

            # Return success
            if attempt_type == "fallback":
                answer = f"**Note: Detailed {study_approach} mode generation failed. Showing standard explanation instead.**\n\n{answer}"
            
            safe_data = convert_numpy((answer, final_confidence, sources, verification_result))
            store_answer(question, safe_data)
            return answer, final_confidence, sources, verification_result

        except Exception as e:
            print(f"Error in ask_llm ({mode} mode): {e}")
            last_error = e
            # Loop continues to fallback if available
            
    # If all attempts fail
    return f"I encountered an error generating the answer: {str(last_error)}", 0.0, [], {}

def classify_topic(question):
    prompt = f"""
    Classify this question into one topic:
    DSA, OS, DBMS, CN, OOP

    Question:
    {question}

    Return only topic name.
    """
    resp = _llm.generate(prompt, temperature=0.0, max_tokens=20, top_p=0.9, top_k=40, seed=42)
    return resp.text.strip()

def detect_misconception(question, student_answer, context=None):
    """Enhanced misconception detection with context"""
    prompt = f"""
    You are an expert educational psychologist. Analyze the student's answer for misconceptions.
    
    Question: {question}
    Student Answer: {student_answer}
    {f'Context/Reference: {context}' if context else ''}
    
    Identify:
    1. Misconception type (Conceptual, Logical, Formula-based, Terminology, Procedural)
    2. Confidence level (High/Medium/Low)
    3. Root cause of misconception
    4. Correct explanation
    5. Targeted example to address the misconception
    
    Format your response as JSON:
    {{
        "misconception_type": "type",
        "confidence": "level",
        "root_cause": "explanation",
        "correct_explanation": "clear explanation",
        "targeted_example": "specific example",
        "improvement_suggestion": "how to improve"
    }}
    """
    
    try:
        response = _llm.generate(prompt, temperature=0.1, max_tokens=900, top_p=0.9, top_k=40, seed=42)
        raw_text = response.text.strip()

        # Ensure we always return JSON the frontend can safely parse
        # If the model added any commentary around JSON, try to extract the JSON block.
        import json
        import re

        # Try direct parse first
        try:
            json.loads(raw_text)
            return raw_text
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from the text
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass

        # Fallback: wrap as JSON with a single raw_analysis field
        safe_wrapper = json.dumps({"raw_analysis": raw_text})
        return safe_wrapper
    except Exception as e:
        import json

        return json.dumps({"error": f"Error analyzing misconception: {str(e)}"})

def extract_concept_blueprint(concept, context=None):
    """
    Step 1: Extract Canonical Concept Blueprint (Key Facts) from source.
    """
    if not context:
        return ["No source context available. Standard definition applies."]

    prompt = f"""
    You are an expert curriculum designer. Extract the "Canonical Blueprint" for the concept: '{concept}'.
    
    Source Material:
    {context}
    
    Task:
    Extract 3-5 distinct, non-overlapping key facts or principles that are essential to understanding this concept.
    Return ONLY a JSON list of strings.
    
    Example:
    ["Gradient descent minimizes the cost function.", "It updates parameters iteratively in the opposite direction of the gradient.", "The learning rate controls the step size."]
    """
    try:
        response = _llm.generate(prompt, temperature=0.1, max_tokens=500, top_p=0.9, top_k=40, seed=42)
        import json
        import re
        
        text = response.text.strip()
        # Clean markdown code blocks if present
        text = re.sub(r"```json\s*|\s*```", "", text)
        
        return json.loads(text)
    except Exception as e:
        print(f"Blueprint extraction failed: {e}")
        return ["Error extracting blueprint."]

def calculate_semantic_similarity(text1, text2):
    """
    Step 2: Calculate Semantic Similarity using Sentence Transformers
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        # Load model (should load once globally in production)
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        
        score = util.pytorch_cos_sim(embedding1, embedding2).item()
        return score
    except Exception as e:
        print(f"Semantic scoring failed: {e}")
        return 0.5 # Default neutral score

def validate_explanation(concept, user_explanation, context=None):
    """
    Validate student explanation against blueprint and source context.
    Returns structured JSON analysis.
    """
    import json
    import re
    
    # 1. Get Blueprint
    blueprint = extract_concept_blueprint(concept, context)
    
    # 2. Semantic Check (if context available)
    semantic_score = 0.0
    if context:
        semantic_score = calculate_semantic_similarity(user_explanation, context)
    
    # 3. LLM Critique
    prompt = f"""
    You are a strict professor grading a student's explanation.
    
    Concept: {concept}
    Canonical Blueprint (Key points that MUST be covered):
    {json.dumps(blueprint, indent=2)}
    
    Student Explanation:
    "{user_explanation}"
    
    Source Context (for reference):
    {context[:1000] if context else "Standard academic definition"}
    
    Task:
    1. Score concept coverage (0-100) based on the blueprint.
    2. Identify SPECIFIC errors (Terminology, Causal, Logic, etc.).
    3. Provide constructive feedback.
    
    Return STRICT JSON:
    {{
        "concept_coverage_score": 0-100,
        "missing_concepts": ["concept 1", "concept 2"],
        "incorrect_statements": ["statement 1"],
        "terminology_errors": ["error 1"],
        "overall_feedback": "Short feedback summary.",
        "recommendation": "Study X more."
    }}
    """
    
    try:
        response = _llm.generate(prompt, temperature=0.1, max_tokens=900, top_p=0.9, top_k=40, seed=42)
        text = response.text.strip()
        text = re.sub(r"```json\s*|\s*```", "", text) # clean
        analysis = json.loads(text)
        
        # Hybrid Scoring
        # Final Score = (LLM Coverage * 0.7) + (Semantic Score * 100 * 0.3)
        final_score = (analysis.get("concept_coverage_score", 0) * 0.7) + (semantic_score * 100 * 0.3)
        
        analysis["final_score"] = round(final_score, 1)
        analysis["semantic_similarity"] = round(semantic_score, 2)
        analysis["blueprint"] = blueprint
        
        return analysis

    except Exception as e:
        print(f"Validation failed: {e}")
        return {
            "concept_coverage_score": 0,
            "missing_concepts": [],
            "incorrect_statements": [],
            "terminology_errors": [],
            "overall_feedback": "Validation failed due to internal error.",
            "recommendation": "Please try again.",
            "final_score": 0,
            "semantic_similarity": semantic_score,
            "blueprint": blueprint
        }
        
def ask_llm_for_json(prompt):
    """
    Helper to get structured JSON from LLM.
    """
    import json
    import re
    
    try:
        resp = _llm.generate(prompt + "\n\nOutput STRICT JSON only. No markdown.", temperature=0.0, max_tokens=900, top_p=0.9, top_k=40, seed=42)
        return _llm.safe_json_extract(resp.text.strip())
    except Exception as e:
        print("Error in ask_llm_for_json:", e)
        return {}
# --- Hybrid Topic Classification ---

class TopicClassifier:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.util = util
            self.topics = {
                "DSA": "Data Structures, Algorithms, sorting, searching, time complexity, graphs, trees, dynamic programming, recursion, arrays, linked lists, stacks, queues, hashing, heaps",
                "OS": "Operating Systems, processes, threads, scheduling, memory management, deadlock, concurrency, file systems, virtualization, kernel, system calls, synchronization, mutual exclusion",
                "DBMS": "Database Management Systems, SQL, normalization, transactions, ACID, indexing, relational model, NoSQL, query optimization, ER diagrams, keys, constraints",
                "CN": "Computer Networks, TCP/IP, OSI model, routing, protocols, HTTP, DNS, sockets, bandwidth, latency, wireless, security, encryption, firewalls",
                "OOP": "Object Oriented Programming, classes, objects, inheritance, polymorphism, encapsulation, abstraction, interfaces, design patterns, SOLID principles",
                "AI": "Artificial Intelligence, machine learning, neural networks, deep learning, NLP, computer vision, reinforcement learning, supervised learning, unsupervised learning",
            }
            self.topic_embeddings = {
                label: self.model.encode(desc, convert_to_tensor=True) 
                for label, desc in self.topics.items()
            }
            print("Topic Classifier Initialized (Embeddings Precomputed)")
        except Exception as e:
            print(f"Topic Classifier Init Failed: {e}")
            self.model = None

    def classify(self, question):
        if not self.model:
            return self.classify_fallback(question)

        try:
            # 1. Encode Question
            q_emb = self.model.encode(question, convert_to_tensor=True)

            # 2. Calculate Similarity
            scores = {}
            for label, t_emb in self.topic_embeddings.items():
                scores[label] = float(self.util.cos_sim(q_emb, t_emb)[0][0])

            # 3. Sort Scores
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top1_label, top1_score = sorted_scores[0]
            top2_label, top2_score = sorted_scores[1]

            # 4. Rules
            # Rule A: Low Confidence -> General
            if top1_score < 0.35: # Conservative threshold
                return "GENERAL", top1_score, sorted_scores

            # Rule B: Ambiguity Check (Tie-Breaker)
            # If difference is small (e.g. < 0.05), it's ambiguous.
            if (top1_score - top2_score) < 0.05:
                # Use LLM to break tie
                print(f"Ambiguous Topic: {top1_label} vs {top2_label} (Diff: {top1_score - top2_score:.3f}). Calling LLM...")
                return self.classify_fallback(question, candidates=[top1_label, top2_label]), top1_score, sorted_scores

            return top1_label, top1_score, sorted_scores

        except Exception as e:
            print(f"Classification Error: {e}")
            return "GENERAL", 0.0, []

    def classify_fallback(self, question, candidates=None):
        """
        Uses LLM for classification (Fallback or Tie-Break).
        """
        options = str(candidates) if candidates else "[DSA, OS, DBMS, CN, OOP, AI, General]"
        prompt = f"""
        Classify this computer science question into ONE of these topics: {options}
        
        Question: "{question}"
        
        Return the Label ONLY. Do not explain.
        """
        try:
            response = _llm.generate(prompt, temperature=0.0, max_tokens=20, top_p=0.9, top_k=40, seed=42)
            text = response.text.strip().upper()
            
            # Normalize
            valid_topics = ["DSA", "OS", "DBMS", "CN", "OOP", "AI", "GENERAL"]
            for t in valid_topics:
                if t in text:
                    return t
            return "GENERAL"
        except Exception as e:
            print(f"LLM Classification Failed: {e}")
            return "GENERAL"

# Global Instance
topic_classifier = TopicClassifier()

def classify_topic(question):
    """
    Wrapper for Hybrid Classifier.
    Returns Dictionary: { "label": str, "confidence": str, "score": float }
    """
    label, score, _ = topic_classifier.classify(question)
    
    # Map score to UX confidence level
    confidence_level = "Low"
    if score > 0.7: confidence_level = "High"
    elif score > 0.5: confidence_level = "Medium"
    
    return {
        "label": label,
        "confidence": confidence_level,
        "score": round(score, 2),
        "primary": label
    }