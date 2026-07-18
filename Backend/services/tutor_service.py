from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.llm_service import ask_llm_for_json, rewrite_query


DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def _clamp_difficulty(diff: str) -> str:
    d = (diff or "medium").lower().strip()
    return d if d in DIFFICULTY_ORDER else "medium"


def _step_difficulty(current: str, delta: int) -> str:
    cur = _clamp_difficulty(current)
    i = DIFFICULTY_ORDER.index(cur)
    j = max(0, min(len(DIFFICULTY_ORDER) - 1, i + int(delta)))
    return DIFFICULTY_ORDER[j]


def _now_iso() -> str:
    # lightweight iso string without importing datetime everywhere
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


@dataclass
class TutorTurn:
    tutor_question: str
    student_answer: Optional[str] = None
    tutor_feedback: Optional[str] = None
    tutor_next_question: Optional[str] = None
    reinforcement: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    topic: Optional[str] = None
    concept: Optional[str] = None
    score: float = 0.0  # 0..100
    correct: bool = False
    created_at: str = field(default_factory=_now_iso)


@dataclass
class TutorSession:
    session_id: str
    user_id: str
    topic: Optional[str] = None
    concept: Optional[str] = None
    difficulty: str = "medium"
    file_filter: Optional[str] = None
    previous_topic: Optional[str] = None
    turns: List[TutorTurn] = field(default_factory=list)


# In-memory session store (compatible with current app patterns).
# If you want persistence later, we can move this to Redis/Mongo.
_SESSIONS: Dict[str, TutorSession] = {}


def start_tutor_session(
    user_id: str,
    *,
    topic: Optional[str] = None,
    concept: Optional[str] = None,
    difficulty: str = "medium",
    file_filter: Optional[str] = None,
    previous_topic: Optional[str] = None,
) -> TutorSession:
    """
    Create a new Tutor session and generate the first tutor question.
    Conversation structure starts with:
      Tutor → Question
    """
    session_id = str(uuid.uuid4())
    sess = TutorSession(
        session_id=session_id,
        user_id=user_id,
        topic=topic,
        concept=concept,
        difficulty=_clamp_difficulty(difficulty),
        file_filter=file_filter,
        previous_topic=previous_topic,
    )

    first_q = generate_tutor_question(sess)
    sess.turns.append(
        TutorTurn(
            tutor_question=first_q,
            difficulty=sess.difficulty,
            topic=sess.topic,
            concept=sess.concept,
        )
    )
    _SESSIONS[session_id] = sess
    return sess


def get_session(session_id: str) -> Optional[TutorSession]:
    return _SESSIONS.get(session_id)


def generate_tutor_question(session: TutorSession) -> str:
    """
    Socratic questioning + follow-up question generation.
    Returns a single tutor question string.
    """
    # Build a short context of the last turn for continuity.
    last = session.turns[-1] if session.turns else None
    last_q = (last.tutor_question if last else "") or ""
    last_a = (last.student_answer if last else "") or ""

    # Decide what we are tutoring "about".
    target = session.concept or session.topic or session.previous_topic or "the current topic"
    base_query = f"{target}".strip()

    prompt = f"""
You are an AI Tutor running in "Socratic Mode".
Your job is to teach by asking ONE good question at a time (no multi-part questions).

Constraints:
- Output STRICT JSON only (no markdown).
- Do NOT provide the final answer. Ask a question.
- Ask at {session.difficulty} difficulty.
- Prefer a question that tests understanding of a core concept, then builds toward applications.
- If the student struggled in the last answer, ask a simpler bridging question and provide a tiny hint inside the question (in parentheses).

Session topic/concept: {base_query}

Last tutor question: {last_q}
Last student answer: {last_a}

Return:
{{
  "tutor_question": "string"
}}
"""
    data = ask_llm_for_json(prompt)
    q = ""
    if isinstance(data, dict):
        q = str(data.get("tutor_question", "")).strip()
    if not q:
        # Hard fallback: generic Socratic prompt.
        q = f"What do you think is the key idea behind {base_query}? (Try defining it in your own words.)"
    return q


def _grade_student_answer(session: TutorSession, tutor_question: str, student_answer: str) -> Dict[str, Any]:
    """
    Lightweight grading + misconception spotting to power adaptive difficulty and reinforcement.
    """
    target = session.concept or session.topic or session.previous_topic or "the topic"

    prompt = f"""
You are grading a student's answer in an AI tutor session.

Return STRICT JSON only (no markdown).

Tutor question: {tutor_question}
Student answer: {student_answer}
Target concept/topic: {target}
Difficulty: {session.difficulty}

Tasks:
1) Decide if the answer is correct enough for this difficulty.
2) Give short, supportive feedback (2-4 sentences).
3) Extract 2-4 reinforcement points (key facts the student should remember).
4) Provide a 0-100 score.
5) If incorrect, identify the likely misconception in 1 sentence.

Return:
{{
  "correct": true/false,
  "score": 0-100,
  "feedback": "string",
  "reinforcement": ["point 1", "point 2"],
  "misconception": "string or empty"
}}
"""
    data = ask_llm_for_json(prompt)
    if not isinstance(data, dict):
        return {
            "correct": False,
            "score": 0,
            "feedback": "Thanks—let’s refine that. Focus on the core definition and one concrete example.",
            "reinforcement": [],
            "misconception": "",
        }
    # normalize
    correct = bool(data.get("correct", False))
    try:
        score = float(data.get("score", 0))
    except Exception:
        score = 0.0
    feedback = str(data.get("feedback", "")).strip() or "Good effort—let’s tighten the reasoning step by step."
    reinforcement = data.get("reinforcement", [])
    if not isinstance(reinforcement, list):
        reinforcement = []
    reinforcement = [str(x).strip() for x in reinforcement if str(x).strip()][:5]
    misconception = str(data.get("misconception", "") or "").strip()
    return {
        "correct": correct,
        "score": max(0.0, min(100.0, score)),
        "feedback": feedback,
        "reinforcement": reinforcement,
        "misconception": misconception,
    }


def _adapt_difficulty(session: TutorSession) -> str:
    """
    Adaptive difficulty using recent performance:
    - If last 2 turns are strong (>=75 and correct): increase
    - If last 2 turns are weak (<=45): decrease
    - Else: keep
    """
    recent = session.turns[-3:] if len(session.turns) >= 3 else session.turns[:]
    if not recent:
        return session.difficulty

    # consider only turns that have been answered/graded
    answered = [t for t in recent if t.student_answer is not None and t.tutor_feedback is not None]
    if len(answered) < 2:
        return session.difficulty

    last2 = answered[-2:]
    if all(t.correct and t.score >= 75 for t in last2):
        return _step_difficulty(session.difficulty, +1)
    if all(t.score <= 45 for t in last2):
        return _step_difficulty(session.difficulty, -1)
    return session.difficulty


def tutor_step(
    session_id: str,
    *,
    student_answer: str,
) -> Dict[str, Any]:
    """
    Main tutor loop step.

    Given the student's answer to the latest tutor question, returns:
      Tutor → Feedback
      Tutor → Next Question

    Output shape:
    {
      "session_id": str,
      "difficulty": str,
      "turn": {
        "tutor_question": str,
        "student_answer": str,
        "tutor_feedback": str,
        "tutor_next_question": str,
        "reinforcement": [str]
      }
    }
    """
    sess = _SESSIONS.get(session_id)
    if not sess or not sess.turns:
        raise ValueError("Tutor session not found. Start a new session first.")

    current_turn = sess.turns[-1]
    current_turn.student_answer = student_answer

    grade = _grade_student_answer(sess, current_turn.tutor_question, student_answer)
    current_turn.correct = bool(grade["correct"])
    current_turn.score = float(grade["score"])
    current_turn.tutor_feedback = grade["feedback"]

    # Concept reinforcement: always provide key points; if incorrect add misconception reminder.
    reinforcement = list(grade.get("reinforcement") or [])
    if (not current_turn.correct) and grade.get("misconception"):
        reinforcement = reinforcement + [f"Common pitfall: {grade['misconception']}"]
    current_turn.reinforcement = reinforcement[:6]

    # Update adaptive difficulty for next question
    sess.difficulty = _adapt_difficulty(sess)

    # Follow-up question generation (Socratic)
    next_q = generate_tutor_question(sess)
    current_turn.tutor_next_question = next_q

    # Append next turn placeholder so the next call has a "current question"
    sess.turns.append(
        TutorTurn(
            tutor_question=next_q,
            difficulty=sess.difficulty,
            topic=sess.topic,
            concept=sess.concept,
        )
    )

    return {
        "session_id": sess.session_id,
        "difficulty": sess.difficulty,
        "turn": {
            "tutor_question": current_turn.tutor_question,
            "student_answer": current_turn.student_answer,
            "tutor_feedback": current_turn.tutor_feedback,
            "tutor_next_question": current_turn.tutor_next_question,
            "reinforcement": current_turn.reinforcement,
            "score": current_turn.score,
            "correct": current_turn.correct,
        },
    }


def get_current_question(session_id: str) -> Dict[str, Any]:
    """
    Returns the current tutor question (the last appended placeholder turn).
    Useful for the initial "Tutor → Question" display.
    """
    sess = _SESSIONS.get(session_id)
    if not sess or not sess.turns:
        raise ValueError("Tutor session not found. Start a new session first.")
    t = sess.turns[-1]
    return {
        "session_id": sess.session_id,
        "difficulty": sess.difficulty,
        "tutor_question": t.tutor_question,
        "topic": sess.topic,
        "concept": sess.concept,
    }

