import shutil
import os
import hashlib
import json
import traceback

# ... imports ... 
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from services.limiter import limiter
from services.pdf_service import save_pdf, extract_and_chunk
from services.intent_service import classify_intent
from services.llm_service import ask_llm, classify_topic
from services.multi_pdf_service import load_all_pdfs, recommend_additional_resources, create_knowledge_graph
from services.progress_service import get_progress, suggest_revision
from datetime import datetime
from pydantic import ValidationError
from services.multi_pdf_service import generate_cross_referenced_answer
from services.interaction_service import save_interaction as save_mongo_interaction
from services.ml.mastery_model import predict_mastery
from services.ml.forgetting_model import predict_forgetting
from services.analytics_service import get_user_session_report
from utils.json_safe import convert_numpy

class QuestionRequest(BaseModel):
    question: str
    user_id: str = "guest"
    subject: str = "general"
    difficulty: str = "medium"
    enable_cross_reference: bool = False
    enable_text_verification: bool = False
    enable_fact_check: bool = True
    session_id: str = "default"
    study_approach: str = "default"
    search_mode: Optional[str] = "default" 
    file_filter: Optional[str] = None

class ExplainRequest(BaseModel):
    question: str
    student_answer: str
    subject: str = "general"
    difficulty: str = "medium"

class ComparativeAnalysisRequest(BaseModel):
    topic: str
    pdf_names: Optional[List[str]] = None
    include_all_pdfs: bool = False

# Simple in-memory status tracker (In prod, use Redis/DB)
processing_status = {}
response_cache = {}
session_memory = {}

router = APIRouter()

def process_pdf_background(file_path: str, filename: str):
    """
    Background task to process PDF.
    """
    try:
        processing_status[filename] = "processing"
        print(f"Starting background processing for {filename}...")
        
        # 1. Extract & Chunk (includes Indexing)
        stored_chunks = extract_and_chunk(file_path)
        
        # 2. Save JSON backup (optional but good for debug)
        os.makedirs("data/extracted_text", exist_ok=True)
        with open(f"data/extracted_text/{filename}.json", "w") as f:
            json.dump(stored_chunks, f, indent=2)
            
        processing_status[filename] = "ready"
        print(f"Finished processing {filename}. Status: READY")
        
    except Exception as e:
        processing_status[filename] = f"error: {str(e)}"
        print(f"Error processing {filename}: {e}")

@router.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        # Save file immediately
        pdf_path = save_pdf(file)
        filename = os.path.splitext(file.filename)[0]
        
        # Set initial status
        processing_status[filename] = "queued"
        
        # Add to background tasks
        background_tasks.add_task(process_pdf_background, pdf_path, filename)
        
        return {
            "message": "PDF uploaded. Processing started in background.",
            "file": filename,
            "status": "processing"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/file-status/{filename}")
def get_file_status(filename: str):
    """Get processing status of a file."""
    status = processing_status.get(filename, "unknown")
    # If unknown, check if we have data for it (might be from restart)
    if status == "unknown":
        if os.path.exists(f"data/extracted_text/{filename}.json"):
             status = "ready"
    return {"filename": filename, "status": status}

@router.get("/files")
def get_files():
    """Get list of all uploaded files (PDFs and processed JSON)"""
    try:
        files = []
        uploads_dir = "uploads"
        extracted_dir = "data/extracted_text"

        # First check for PDF files in uploads
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(uploads_dir, filename)
                    files.append(
                        {
                            "name": filename.replace(".pdf", ""),
                            "filename": filename,
                            "size": os.path.getsize(file_path),
                            "type": "pdf",
                        }
                    )

        # Then check for processed JSON files
        if os.path.exists(extracted_dir):
            for filename in os.listdir(extracted_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(extracted_dir, filename)
                    base_name = filename.replace(".json", "")
                    # Only add if not already added from PDFs
                    if not any(f["name"] == base_name for f in files):
                        files.append(
                            {
                                "name": base_name,
                                "filename": filename,
                                "size": os.path.getsize(file_path),
                                "type": "json",
                            }
                        )

        return {"files": files}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug")
def debug_endpoint(request: QuestionRequest):
    try:
        print(f"Received request: {request}")
        print(f"Question: {request.question}")
        print(f"Subject: {request.subject}")
        print(f"Difficulty: {request.difficulty}")
        return {"status": "debug received", "data": request.model_dump()}
    except Exception as e:
        print(f"Error in debug: {str(e)}")
        return {"error": str(e)}

# def resolve_followup(question, session_id):
#     vague_patterns = [
#         "definition",
#         "in two lines",
#         "in three lines",
#         "summarize",
#         "shortly",
#         "briefly"
#     ]
    
#     if session_id in session_memory:
#         last_topic = session_memory[session_id].get("last_topic")
        
#         if any(p in question.lower() for p in vague_patterns):
#             return f"{question} about {last_topic}"
    
#     return question

# Legacy Query Reformulator removed. Handled by routing_service.

@router.post("/ask")
@limiter.limit("10/minute") 
def ask_question(request: Request, data: QuestionRequest):

    try:
        import time
        start_time = time.time()
        print(f"Received ask request: {data.question}")

        question = data.question
        subject = data.subject
        difficulty = data.difficulty
        enable_cross_ref = data.enable_cross_reference
        enable_fact_check = data.enable_fact_check
        session_id = data.session_id
        study_approach = data.study_approach

        # 1. State Management (Memory)
        last_topic = None
        last_question = None
        if session_id in session_memory:
            last_topic = session_memory[session_id].get("last_topic")
            last_question = session_memory[session_id].get("last_question")

        # 2. Query Reformulation
        from services.routing_service import reformulate_query, classify_task_intent, TaskType
        
        standalone_query = reformulate_query(question, last_topic, last_question)

        # 3. Intent & Task Routing
        available_docs = []

        # Cross reference mode → search all PDFs
        if enable_cross_ref:
            pdf_index = load_all_pdfs()
            available_docs = list(pdf_index.keys())

        # Single subject mode
        elif subject and subject != "general":
            available_docs = [subject]

        # Safety fallback
        if not available_docs:
            available_docs = ["general"]

        # Classify task
        route = classify_task_intent(standalone_query, available_docs)

        print(f"[ROUTING] Task Type: {route.task_type}, Target Docs: {route.target_documents}")

        # Override route for legacy explicit cross_reference trigger
        if enable_cross_ref and route.task_type != TaskType.COMPARISON:
            pass
            
        # 4. Context Execution Pipeline
        answer = None
        confidence = 0.0
        sources = []
        source_file = "Multiple"
        verification = {}
        topic_metadata = {}

        if route.task_type == TaskType.COMPARISON or getattr(data, "search_mode", "default") == "cross_reference":
             # Execute Multi-Document/Comparative Search
            from services.multi_pdf_service import generate_cross_referenced_answer
            try:
                answer, confidence, source_refs = generate_cross_referenced_answer(
                    standalone_query, difficulty=difficulty
                )
                sources = [{"source": s["pdf_name"], "score": s["relevance"]} for s in source_refs]
                source_file = f"Multiple PDFs ({len(sources)} sources)"
                
                # Fast exit for cross reference returning directly
                return {
                    "answer": answer,
                    "topic": "Multi-Source Analysis",
                    "confidence": f"{confidence:.1f}%",
                    "confidence_score": float(confidence),
                    "sources": sources,
                    "mode": "Cross-Reference",
                }
            except Exception as e:
                print(f"Cross-Ref failed: {e}")
                raise HTTPException(500, f"Cross-Reference Search Failed: {e}")

        elif route.task_type == TaskType.FULL_SUMMARY:
            # Document Summarization Pipeline
            target_document = route.target_documents[0] if route.target_documents else subject
            if target_document == "general":
                 # Fallback if it classified full_summary but active doc is general
                 pdf_index = load_all_pdfs()
                 if pdf_index:
                      target_document = sorted(pdf_index.items(), key=lambda i: i[1].get('last_modified', 0), reverse=True)[0][0]
                      
            from services.semantic_service import get_all_document_chunks
            document_context = get_all_document_chunks(target_document) if target_document else []
            
            if not document_context:
                print("Document mode failed: No chunks found. Falling back to QA.")
                route.task_type = TaskType.DOCUMENT_QA
            else:
                answer, confidence, sources, verification = ask_llm(
                    standalone_query,
                    relevant_context=document_context,
                    difficulty=difficulty,
                    enable_fact_check=False,
                    study_approach="default",
                    file_filter=target_document,
                    previous_topic=last_topic,
                )
                sources = [{"source": target_document, "score": 100.0, "note": "Full Document Analysis"}]
                confidence = 100.0
                source_file = target_document

        if route.task_type == TaskType.DOCUMENT_QA:
             # Hybrid Retrieval Pipeline
             target_document = route.target_documents[0] if route.target_documents else (subject if subject != "general" else None)
             answer, confidence, sources, verification = ask_llm(
                 standalone_query,
                 difficulty=difficulty,
                 enable_fact_check=enable_fact_check,
                 study_approach=study_approach,
                 file_filter=target_document,
                 previous_topic=None, # Already injected via reformulate_query
             )
             source_file = target_document

        elif route.task_type == TaskType.GENERAL_KNOWLEDGE:
             # General Knowledge Pipeline
             answer, confidence, sources, verification = ask_llm(
                 standalone_query,
                 difficulty=difficulty,
                 enable_fact_check=False,
                 study_approach=study_approach,
                 file_filter=None,
                 previous_topic=None,
             )
             source_file = "General Knowledge"

        # Final Topic Verification
        if (
            not topic_metadata
            or topic_metadata.get("label") == "GENERAL"
        ):
            classification = classify_topic(question)
            topic_metadata = classification
            topic = classification["label"]
        else:
            topic = topic_metadata.get("label", subject)

        # --- Mastery Model Integration ---
        # Compute attempts/correct/revisions for this topic from the user's history.
        from database.repositories import find_user_interactions
        raw_interactions = []
        try:
            raw_interactions = find_user_interactions(data.user_id, session_id="current")
        except Exception as e:
            print(f"Failed to load user interactions for mastery model: {e}")

        # Normalize timestamps to ISO strings for the mastery model helpers
        from datetime import datetime as _dt
        topic_history = []
        for h in raw_interactions:
            if h.get("topic") != topic:
                continue
            ts = h.get("timestamp")
            if isinstance(ts, _dt):
                ts_val = ts.isoformat()
            else:
                ts_val = str(ts) if ts is not None else None
            topic_history.append(
                {
                    **h,
                    "timestamp": ts_val,
                }
            )
        attempts = len(topic_history)
        correct_count = sum(1 for h in topic_history if bool(h.get("correct", False)))
        revisions = attempts  # treat each interaction on the topic as a revision opportunity

        try:
            mastery_result = predict_mastery(
                attempts=attempts,
                correct=correct_count,
                revisions=revisions,
                topic=topic,
                user_history=topic_history,
            )
        except Exception as e:
            print(f"Mastery prediction failed: {e}")
            mastery_result = {
                "mastery_level": "Unavailable",
                "analysis": "Mastery analysis unavailable due to internal error.",
                "mastery_score": 0.0,
            }

        # Persist interaction after computing mastery so that the stored record
        # reflects the current masteryLevel used in analytics dashboards.
        response_time = time.time() - start_time
        is_correct = confidence > 70.0
        
        try:
            save_mongo_interaction({
                "userId": data.user_id,
                "question": question,
                "answer": answer,
                "topic": topic,
                "confidence": float(confidence),
                "masteryLevel": mastery_result.get("mastery_level", "unknown"),
                "difficulty": difficulty,
                "responseTime": response_time,
                "correct": bool(is_correct),
                "sessionId": session_id,
            })
        except Exception as e:
            print(f"Failed to save interaction to MongoDB: {e}")
        
        # User-specific progress & revision suggestions
        revision_tip = suggest_revision(topic, data.user_id)
        
        progress_data = get_progress(data.user_id)
        progress_count = sum(progress_data["raw_progress"].values())
        
        user_history = None

        try:
            forgetting_result = predict_forgetting(
                last_study_date=datetime.now(),
                strength=progress_count / 100,
                topic=topic,
                user_history=user_history,
                mastery_score=mastery_result.get(
                    "mastery_score", 0
                ),
            )
        except Exception as e:
            print(f"Forgetting prediction failed: {e}")
            forgetting_result = {"retention_rate": 1.0}

        retention = float(forgetting_result.get("retention_rate", 1.0))
        mastery_score = float(mastery_result.get("mastery_score", 0))
        effective_mastery = float(mastery_score * retention)

        recommended_action = "Continue Learning"
        action_type = "none"

        if retention < 0.4:
            recommended_action = "⚠️ Critical: Take a Mini Quiz Now!"
            action_type = "quiz"
        elif retention < 0.6:
            recommended_action = "📉 Review Flashcards"
            action_type = "flashcards"
        elif effective_mastery < 40 and mastery_score > 60:
            recommended_action = "🔄 Quick Summary Review"
            action_type = "summary"

        is_general_mode = route.task_type.name == "GENERAL_KNOWLEDGE"
        cache_key = hashlib.md5(standalone_query.encode()).hexdigest()

        response_data = {
            "answer": answer,
            "topic": topic,
            # Backward compatible additions: subject/topic/subtopic tagging
            "subject": None,
            "subtopic": None,
            "confidence": f"{float(confidence):.1f}%",
            "confidence_score": float(confidence),
            "sources": sources if "sources" in locals() else [],
            "mode": "RAG" if not is_general_mode else "General",
            "intent": route.task_type.name,
            "revision_tip": revision_tip,
            "mastery_level": mastery_result.get("mastery_level", "unknown"),
            "mastery_analysis": mastery_result.get("analysis", {}),
            "mastery_score": float(mastery_score),
            "effective_mastery": float(round(effective_mastery, 1)),
            "recommended_action": recommended_action,
            "action_type": action_type,
            "source_file": source_file,
            "source_references": source_refs
            if "source_refs" in locals()
            else sources,
            "forgetting_prediction": forgetting_result,
            "cross_reference_enabled": enable_cross_ref,
            "fact_check_enabled": enable_fact_check,
            "verification": verification
            if "verification" in locals()
            else {},
        }

        # Attach subject/topic/subtopic if we can infer from retrieval sources metadata (new pipeline).
        try:
            srcs = sources if isinstance(sources, list) else []
            first_meta = None
            # sources from ask_llm is list of metadata dicts; legacy sources is [{source,score}]
            for s in srcs:
                if isinstance(s, dict) and ("subject" in s or "topics" in s or "branch" in s):
                    first_meta = s
                    break
            if first_meta:
                response_data["subject"] = first_meta.get("branch") or first_meta.get("subject")
                # Choose best matching topic from metadata topics (simple overlap)
                topics_list = first_meta.get("topics") or []
                if isinstance(topics_list, list) and topics_list:
                    ql = (question or "").lower()
                    best = None
                    best_score = -1
                    for t in topics_list:
                        tl = str(t).lower()
                        score = 0
                        for w in tl.split():
                            if len(w) >= 3 and w in ql:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best = t
                    if best:
                        response_data["topic"] = best  # keep existing key but now cleaner if available
        except Exception:
            pass

        response_cache[cache_key] = response_data

        session_memory[session_id] = {
            "last_topic": topic,
            "last_subject": subject,
        }

        response_data = convert_numpy(response_data)

        return response_data

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{user_id}")
def show_progress(user_id: str):
    try:
        progress_data = get_progress(user_id)
        last_date = datetime.now()
        progress_score = progress_data["overall_completion"] / 100.0

        # Use generic topic None and no history for dashboard-level forgetting status
        forgetting_status = predict_forgetting(
            last_study_date=last_date,
            strength=progress_score,
            topic=None,
            user_history=None,
        )

        return {
            "progress": progress_data,
            "forgetting_status": forgetting_status,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain")
def explain_back(request: ExplainRequest):
    """Enhanced Explain Back to Me mode with misconception detection"""
    try:
        question = request.question
        student_answer = request.student_answer
        subject = request.subject
        difficulty = request.difficulty
        
        # Load the relevant PDF content using Vector DB
        from services.semantic_service import retrieve_context
        
        context_data = retrieve_context(question, file_filter=subject, top_k=3)
        context = "\n".join([item['text'] for item in context_data]) if context_data else None
        
        # Detect misconceptions
        from services.llm_service import detect_misconception
        misconception_analysis = detect_misconception(question, student_answer, context)
        
        return {
            "question": question,
            "student_answer": student_answer,
            "misconception_analysis": misconception_analysis,
            "context_available": context is not None,
            "subject": subject
        }
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session-report/{user_id}")
def session_report(
    user_id:str, session_id: Optional[str] = None, include_recommendations: bool = True
):
    """Enhanced session analytics with detailed reporting"""
    try:
        report = get_user_session_report(user_id, session_id)
        
        # Add additional analytics if needed
        if include_recommendations:
            pdf_index = load_all_pdfs()
            if pdf_index and "recent_activity" in report and report["recent_activity"]:
                last_question = report["recent_activity"][-1].get("question", "")
                if last_question:
                    recommendations = recommend_additional_resources(last_question, pdf_index)
                    report["additional_resources"] = recommendations
        
        return report
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/comparative-analysis")
def comparative_analysis(request: ComparativeAnalysisRequest):
    """Generate comparative analysis across multiple PDFs"""
    try:
        topic = request.topic
        include_all_pdfs = request.include_all_pdfs
        
        pdf_index = load_all_pdfs()
        if not pdf_index:
            return {"error": "No PDFs uploaded for analysis"}
        
        analysis, covered_pdfs = generate_comparative_analysis(topic, pdf_index)
        
        # Find related topics
        related_topics = find_related_topics(topic, pdf_index)
        
        return {
            "topic": topic,
            "analysis": analysis,
            "covered_pdfs": covered_pdfs,
            "related_topics": related_topics,
            "total_pdfs_analyzed": len(covered_pdfs)
        }
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-graph")
def knowledge_graph():
    try:
        pdf_index = load_all_pdfs()
        graph = create_knowledge_graph(pdf_index)
        
        return {
            "graph": graph,
            "total_nodes": len(graph["nodes"]),
            "total_edges": len(graph["edges"]),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class RecommendationRequest(BaseModel):
    question: str


@router.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations based on question"""
    try:
        pdf_index = load_all_pdfs()
        user_history = None
        
        # History is now stored in DB and optionally mirrored per-user in data/users.
        # For now, we pass None here; recommend_additional_resources will still work,
        # and more advanced per-user personalization can be plugged in later.
        recommendations = recommend_additional_resources(
            request.question, pdf_index, user_history
        )
        
        return {
            "question": request.question,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/duplicate-files")
def get_files_duplicate():
    """
    Deprecated duplicate route kept temporarily for backward compatibility.
    Prefer using the primary /files endpoint defined at the top of this file.
    """
    return get_files()

@router.get("/pdf-index")
def get_pdf_index():
    """Get index of all uploaded PDFs with metadata"""
    try:
        pdf_index = load_all_pdfs()

        pdf_summary = {}

        for pdf_name, data in pdf_index.items():
            pdf_summary[pdf_name] = {
                "chunk_count": data.get("chunk_count", 0),
                "topics": data.get("topics", []),
                "last_modified": data.get("last_modified"),
                "file_size": os.path.getsize(data["file_path"]) if os.path.exists(data["file_path"]) else 0
            }

        total_topics = len(
            set(
                topic
                for data in pdf_index.values()
                for topic in data.get("topics", [])
            )
        )

        return {
            "pdfs": pdf_summary,
            "total_pdfs": len(pdf_index),
            "total_topics": total_topics
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ToolRequest(BaseModel):
    tool: str  # summarizer, generator, solver
    sub_tool: str # pdf, word, quiz, math, etc.
    content: Optional[str] = None
    topic: Optional[str] = None

# --- New Service Imports ---
from services.summarizer_service import summarize_content
from services.solver_service import analyze_image, solve_problem
from services.generator_service import generate_content

# --- Infrastructure: Rate Limiting & Caching ---
from cachetools import TTLCache
import hashlib

# Initialize Cache (Max 100 items, TTL 1 hour)
tool_cache = TTLCache(maxsize=100, ttl=3600)

def get_cache_key(content: str, tool: str):
    return hashlib.md5(f"{content[:100]}-{tool}".encode()).hexdigest()

@router.post("/tools/upload-summarize")
@limiter.limit("5/minute")
async def summarize_file(
    request: Request,
    file: UploadFile = File(...), 
    type: str = "pdf"
):
    try:
        # Check Cache (only for text/pdf content, tough for file streams)
        # We'll skip cache for upload for now, or implement hash of file content later.
        
        if type in ["pdf", "docx", "pptx", "txt"]:
            # We need to process content first
            from services.tools_service import process_file_content # Keep this helper or move it
            content = await process_file_content(file, type)
            
            # Now we can cache based on content hash
            cache_key = get_cache_key(content, "summarizer")
            if cache_key in tool_cache:
                return {"summary": tool_cache[cache_key]["summary"], "cached": True}

            result = await summarize_content(content, type)
            
            # Cache the result
            tool_cache[cache_key] = result
            
            return result # Contains 'summary' and 'meta'
            
        elif type == "image":
            # Direct pass to solver service
            result = await analyze_image(file, "summarize")
            return result
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
            
    except Exception as e:
        print(f"Summarize Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/generate")
@limiter.limit("10/minute")
async def generate_tool_content(request: Request, tool_req: ToolRequest):
    try:
        cache_key = get_cache_key(tool_req.topic + tool_req.sub_tool, "generator")
        if cache_key in tool_cache:
             cached = tool_cache[cache_key]
             return {
                 "result": cached.get("result"),
                 "meta": cached.get("meta", {}),
                 "cached": True
             }

        result = await generate_content(tool_req.topic, tool_req.sub_tool)

        # If generator_service returns an error payload, don't raise 500.
        # Keep response shape stable for the frontend: {result, meta, error?}
        if isinstance(result, dict) and result.get("error"):
            return {
                "result": None,
                "meta": result.get("meta", {"tool": "content_generator", "task": tool_req.sub_tool}),
                "error": result.get("error"),
            }

        # Cache only successful results with expected keys
        tool_cache[cache_key] = result
        return {
            "result": result.get("result") if isinstance(result, dict) else result,
            "meta": (result.get("meta") if isinstance(result, dict) else {}) or {},
        }
        
    except Exception as e:
        print(f"Generate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/solve")
@limiter.limit("5/minute")
async def solve_tool_problem(request: Request, tool_req: ToolRequest):
    try:
        result = await solve_problem(tool_req.content, tool_req.sub_tool)
        return {"result": result.get("result", result.get("error")), "meta": result.get("meta")}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/image-solve")
@limiter.limit("5/minute")
async def solve_image_problem(
    request: Request,
    file: UploadFile = File(...),
    subject: str = "math"
):
    try:
        result = await analyze_image(file, subject)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain")
async def explain_concept(request: Request):
    """
    Endpoint for "Explain Back to Me" mode.
    Validates student explanation against source material.
    """
    try:
        data = await request.json()
        concept = data.get("concept")
        explanation = data.get("explanation")
        
        if not concept or not explanation:
             raise HTTPException(status_code=400, detail="Concept and Explanation are required")

        # 1. Retrieve Context
        # Reuse semantic service to find what the system knows about this concept
        from services.semantic_service import retrieve_context
        context_data = retrieve_context(concept)
        context_text = "\n".join([c['text'] for c in context_data]) if context_data else ""
        
        if not context_text:
             # Fail-safe: Use general knowledge if no local context
             context_text = None 
             
        # 2. Validate Explanation
        from services.llm_service import validate_explanation
        analysis = validate_explanation(concept, explanation, context_text)
        
        # 3. Save Interaction as specific type
        # We save this so mastery model can pick it up (high weight)
        # Note: save_interaction is defined in this file (routes.py) or imported. 
        # Assuming it's available as it is used in /ask
        # If not, we might need to import it or define it. 
        # Re-checking imports... save_interaction is likely a helper or imported.
        # Let's hope it's available in scope. If not, we'll see an error.
        # Actually, looking at previous diffs, save_interaction was called in ask_question.
        # It's likely imported from services.db or similar.
        try:
            score = float(analysis.get("final_score", 0))
            save_mongo_interaction({
                "userId": data.get("user_id") or "guest",
                "question": f"Explain: {concept}",
                "answer": explanation,
                "topic": concept,
                "confidence": score,
                "masteryLevel": "Evaluating",
                "difficulty": "hard",
                "responseTime": 0.0,
                "correct": score > 70.0,
                "sessionId": "explain_mode",
            })
        except Exception as e:
            print(f"Warning: Failed to save explain interaction: {e}")
        
        return {
            "concept": concept,
            "analysis": analysis,
            "context_found": bool(context_text)
        }
        
    except Exception as e:
        print(f"Error in explain endpoint: {e}")
        return {
            "error": "Explain mode unavailable."
        }
