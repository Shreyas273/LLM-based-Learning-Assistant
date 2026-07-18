from datetime import datetime
from typing import Dict, Any, List
from collections import Counter

from database.repositories import find_user_interactions
from services.memory_service import (
    analyze_time_patterns,
    analyze_learning_progress,
    analyze_difficulty_progression,
    calculate_engagement_metrics,
    generate_recommendations,
    create_empty_report,
)

from services.ml.mastery_model import detect_learning_style, compute_topic_mastery_score
from services.ml.forgetting_model import predict_forgetting


def build_history_from_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize Mongo interaction documents into the 'history' structure
    expected by existing analytics helpers.
    """
    history: List[Dict[str, Any]] = []
    for doc in docs:
        q = doc.get("question") or ""
        a = doc.get("answer") or ""
        ts = doc.get("timestamp")
        ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        history.append(
            {
                "question": q,
                "answer": a,
                "topic": doc.get("topic"),
                "confidence": float(doc.get("confidence", 0)),
                "mastery_level": doc.get("masteryLevel", "unknown"),
                "timestamp": ts_iso,
                "session_id": doc.get("sessionId"),
                "response_time": doc.get("responseTime"),
                "correct": bool(doc.get("correct", False)),
                "difficulty": doc.get("difficulty", "medium"),
                "question_length": len(q),
                "answer_length": len(a),
            }
        )
    return history


def detect_weak_topics(history: List[Dict[str, Any]], *, min_attempts: int = 2) -> List[Dict[str, Any]]:
    """
    Identify weak concepts/topics based on:
    - low confidence
    - incorrect answers/explanations (correct=False)
    - repeated mistakes (multiple incorrect attempts)
    Returns list of {topic, weakness_score, signals}.
    """
    if not history:
        return []

    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for h in history:
        t = h.get("topic")
        if not t:
            continue
        by_topic.setdefault(t, []).append(h)

    weak: List[Dict[str, Any]] = []
    for topic, items in by_topic.items():
        attempts = len(items)
        if attempts < min_attempts:
            continue

        confidences = [float(i.get("confidence", 0) or 0) for i in items]
        avg_conf = sum(confidences) / max(1, len(confidences))

        incorrect = sum(1 for i in items if i.get("correct") is False)
        incorrect_rate = incorrect / max(1, attempts)

        # repeated mistakes: >=2 incorrect OR last 3 contain >=2 incorrect
        recent = items[-3:] if len(items) >= 3 else items
        recent_incorrect = sum(1 for i in recent if i.get("correct") is False)
        repetition_factor = 1.0 if (incorrect >= 2 or recent_incorrect >= 2) else 0.0

        # Combine into weakness score (0..1), then scale to 0..100 for readability
        low_conf_component = max(0.0, (70.0 - avg_conf) / 70.0)  # only penalize below 70
        weakness = (low_conf_component * 0.5) + (incorrect_rate * 0.4) + (repetition_factor * 0.1)
        weakness_score = round(min(1.0, max(0.0, weakness)) * 100.0, 1)

        if weakness_score <= 0:
            continue

        weak.append(
            {
                "topic": topic,
                "weakness_score": weakness_score,
                "signals": {
                    "attempts": attempts,
                    "avg_confidence": round(avg_conf, 1),
                    "incorrect_count": incorrect,
                    "incorrect_rate": round(incorrect_rate, 2),
                    "repeated_mistakes": bool(repetition_factor > 0),
                },
            }
        )

    weak.sort(key=lambda x: x["weakness_score"], reverse=True)
    return weak


def generate_weekly_plan(history: List[Dict[str, Any]], weak_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate a simple 7-day study plan using:
    - mastery score (computed from history)
    - forgetting curve (predict_forgetting)
    - weak topics ranking
    """
    if not history:
        return []

    # Pick top weak topics (up to 4) to focus the week.
    focus = [w["topic"] for w in (weak_topics or [])[:4]]
    if not focus:
        # Fallback: most frequent topics
        topics = [h.get("topic") for h in history if h.get("topic")]
        focus = [t for t, _ in Counter(topics).most_common(3)]

    # Compute per-topic retention urgency
    by_topic = {}
    for t in focus:
        topic_items = [h for h in history if h.get("topic") == t]
        last_ts = None
        for it in reversed(topic_items):
            if it.get("timestamp"):
                last_ts = it["timestamp"]
                break
        mastery_score = compute_topic_mastery_score(t, history)
        forget = predict_forgetting(
            last_study_date=last_ts or datetime.now().isoformat(),
            strength=min(1.0, len(topic_items) / 100.0),
            topic=t,
            user_history=history,
            mastery_score=mastery_score,
        )
        by_topic[t] = {"mastery_score": mastery_score, "forgetting": forget}

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plan: List[Dict[str, Any]] = []

    # Scheduling heuristic:
    # - High priority topics (retention < 0.4) appear 3x/week
    # - Medium (0.4-0.7) appear 2x/week
    # - Low appear 1x/week (light review)
    slots: List[str] = []
    for t in focus:
        r = float(by_topic[t]["forgetting"].get("retention_rate", 1.0) or 1.0)
        if r < 0.4:
            slots += [t, t, t]
        elif r < 0.7:
            slots += [t, t]
        else:
            slots += [t]
    if not slots:
        slots = focus[:]

    # Fill 7 days by cycling through slots
    for i, day in enumerate(day_names):
        t = slots[i % len(slots)]
        meta = by_topic.get(t, {})
        forgetting = meta.get("forgetting", {}) or {}
        mastery_score = float(meta.get("mastery_score", 0.0) or 0.0)

        # Activity selection:
        # - lower mastery => more reading + examples
        # - higher mastery but low retention => spaced repetition + quick quiz
        activities = []
        if mastery_score < 40:
            activities = ["Read core notes", "Work 3 examples", "Write 5 key points"]
        elif mastery_score < 70:
            activities = ["Revise summary", "Solve 5 practice questions", "Review mistakes"]
        else:
            activities = ["Quick spaced repetition review", "Solve 3 hard questions", "Teach-back summary"]

        plan.append(
            {
                "day": day,
                "focus_topic": t,
                "estimated_time": forgetting.get("estimated_study_time", "15 minutes"),
                "urgency": forgetting.get("urgency", "No Revision Needed"),
                "activities": activities,
            }
        )

    return plan


def build_learning_analytics_payload(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Structured output requested by the user:
    {
      learning_style,
      weak_topics,
      recommended_actions,
      weekly_plan
    }
    """
    style = detect_learning_style(history)
    weak = detect_weak_topics(history)
    weekly = generate_weekly_plan(history, weak)

    # Recommended actions synthesized from weaknesses + learning style
    recommended_actions: List[Dict[str, Any]] = []
    style_label = style.get("learning_style", "Reading learner")

    if weak:
        for w in weak[:5]:
            topic = w["topic"]
            score = w.get("weakness_score", 0)
            signals = w.get("signals", {})
            action = "Revise basics + do targeted practice"
            if signals.get("repeated_mistakes"):
                action = "Review mistakes + do error-focused drills"
            if signals.get("avg_confidence", 100) < 50:
                action = "Start with fundamentals + worked examples"
            recommended_actions.append(
                {
                    "topic": topic,
                    "priority": "high" if score >= 70 else "medium",
                    "action": action,
                }
            )
    else:
        recommended_actions.append(
            {"topic": None, "priority": "low", "action": "Keep consistent practice and spaced repetition"}
        )

    # Style-specific nudges
    style_addon = None
    if style_label == "Visual learner":
        style_addon = "Use flowcharts/diagrams and concept maps during revision."
    elif style_label == "Problem-solving learner":
        style_addon = "Focus on practice sets; review solutions and maintain an error log."
    else:
        style_addon = "Use structured notes, definitions, and short self-summaries after each session."

    return {
        "learning_style": style_label,
        "weak_topics": weak[:8],
        "recommended_actions": recommended_actions,
        "weekly_plan": weekly,
        "style_tip": style_addon,
    }


def get_user_session_report(user_id: str, session_id: str | None = None) -> Dict[str, Any]:
    """
    High-level analytics service.
    Returns the same structure as memory_service.generate_session_report,
    but pulls data via the repository layer.
    """
    docs = find_user_interactions(user_id, session_id=session_id)
    if not docs:
        return create_empty_report()

    history = build_history_from_docs(docs)

    total_questions = len(history)
    topics = [item.get("topic") for item in history if item.get("topic")]
    topic_count = dict(Counter(topics))

    confidence_scores = [item.get("confidence", 0) for item in history]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    mastery_levels = [item.get("mastery_level", "unknown") for item in history]
    mastery_distribution = dict(Counter(mastery_levels))

    time_analysis = analyze_time_patterns(history)
    progress_analysis = analyze_learning_progress(history)
    difficulty_analysis = analyze_difficulty_progression(history)
    engagement_metrics = calculate_engagement_metrics(history)

    report = {
        "session_id": session_id or "current",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_questions_asked": total_questions,
            "topic_wise_distribution": topic_count,
            "average_confidence": round(avg_confidence, 2),
            "mastery_distribution": mastery_distribution,
        },
        "time_analysis": time_analysis,
        "learning_progress": progress_analysis,
        "difficulty_analysis": difficulty_analysis,
        "engagement_metrics": engagement_metrics,
        "recent_activity": history[-5:] if len(history) >= 5 else history,
        "recommendations": generate_recommendations(history),
    }

    # Backward compatible: add new learning analytics payload as an extra field.
    report["learning_analytics"] = build_learning_analytics_payload(history)
    return report

