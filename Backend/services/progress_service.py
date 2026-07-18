from config import TOPICS
from services.memory_service import generate_session_report
from services.analytics_service import get_user_session_report


def get_progress(user_id: str):
    """
    Compute per-user progress strictly from that user's interactions.
    """
    report = generate_session_report(user_id)
    # Topic-wise counts are already user-filtered in generate_session_report
    topic_counts = report.get("summary", {}).get("topic_wise_distribution", {}) or {}
    
    # Ensure all configured topics exist in the map
    data = {topic: int(topic_counts.get(topic, 0)) for topic in TOPICS}
    
    percentages = {}
    weak_topics = []
    
    for topic, count in data.items():
        # Interpret 100 interactions as 100% completion
        percent = min(count, 100)
        percentages[topic] = round(percent, 2)
        if percent < 40:
            weak_topics.append(topic)
    
    overall = round(sum(percentages.values()) / max(len(percentages), 1), 2)
    return {
        "raw_progress": data,
        "percentage": percentages,
        "overall_completion": overall,
        "revision_suggested_for": weak_topics,
    }


def suggest_revision(topic: str, user_id: str):
    """
    Generate a revision suggestion for a topic based only on this user's data.
    """
    data = get_progress(user_id)["raw_progress"]
    score = data.get(topic, 0)
    
    if score < 20:
        return f"{topic}: Very weak. Revise basics + examples."
    elif score < 50:
        return f"{topic}: Average. Practice more questions."
    elif score < 80:
        return f"{topic}: Good. Do advanced problems."
    else:
        return f"{topic}: Strong. Focus on revision only."


def get_learning_analytics(user_id: str, session_id: str | None = None):
    """
    Structured analytics output (backward compatible additive API).
    Returns:
    {
      learning_style,
      weak_topics,
      recommended_actions,
      weekly_plan
    }
    """
    report = get_user_session_report(user_id, session_id=session_id)
    la = (report or {}).get("learning_analytics") or {}
    # Return exactly the requested shape (and nothing else) from this helper.
    return {
        "learning_style": la.get("learning_style"),
        "weak_topics": la.get("weak_topics", []),
        "recommended_actions": la.get("recommended_actions", []),
        "weekly_plan": la.get("weekly_plan", []),
    }