import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any
from database.repositories import interactions_collection


def generate_session_report(user_id: str, session_id: str = None) -> Dict:
    """
    Comprehensive session analytics report using MongoDB `interactions` collection.
    Filters strictly by userId (per-user analytics).
    """
    try:
        coll = interactions_collection
        query: Dict[str, Any] = {"userId": user_id}
        if session_id and session_id != "current":
            query["sessionId"] = session_id

        # Sort newest first for "current", otherwise ascending
        sort_dir = -1 if session_id == "current" else 1
        cursor = coll.find(query).sort("timestamp", sort_dir)
        interactions = list(cursor)

        if not interactions:
            return create_empty_report()

        if session_id == "current":
            interactions = interactions[:50]

        history: List[Dict[str, Any]] = []
        for doc in interactions:
            q = doc.get("question") or ""
            a = doc.get("answer") or ""
            ts = doc.get("timestamp")
            if isinstance(ts, datetime):
                ts_iso = ts.isoformat()
            else:
                ts_iso = str(ts)

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

        # Basic metrics
        total_questions = len(history)
        topics = [item.get("topic") for item in history if item.get("topic")]
        topic_count = dict(Counter(topics))

        # Advanced analytics
        confidence_scores = [item.get("confidence", 0) for item in history]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        mastery_levels = [item.get("mastery_level", "unknown") for item in history]
        mastery_distribution = dict(Counter(mastery_levels))

        # Time analytics
        time_analysis = analyze_time_patterns(history)

        # Learning progress
        progress_analysis = analyze_learning_progress(history)

        # Difficulty analysis
        difficulty_analysis = analyze_difficulty_progression(history)

        # Engagement metrics
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

        return report
    except Exception as e:
        return {
            "error": f"Error generating report: {str(e)}",
            "session_id": session_id or "current",
            "generated_at": datetime.now().isoformat(),
        }

def create_empty_report():
    return {
        "session_id": "current",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_questions_asked": 0,
            "topic_wise_distribution": {},
            "average_confidence": 0,
            "mastery_distribution": {}
        },
        "time_analysis": {},
        "learning_progress": {},
        "difficulty_analysis": {},
        "engagement_metrics": {},
        "recent_activity": [],
        "recommendations": []
    }

def generate_session_id():
    return str(datetime.now().timestamp())

def analyze_time_patterns(history):
    """Analyze time patterns in learning sessions"""
    if not history:
        return {}
    
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in history if "timestamp" in item]
    
    if not timestamps:
        return {}
    
    # Group by hour of day
    hour_distribution = {}
    for ts in timestamps:
        hour = ts.hour
        hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
    
    # Group by day of week
    day_distribution = {}
    for ts in timestamps:
        day = ts.strftime("%A")
        day_distribution[day] = day_distribution.get(day, 0) + 1
    
    # Calculate session intervals
    if len(timestamps) > 1:
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 for i in range(len(timestamps)-1)]
        avg_interval = sum(intervals) / len(intervals)
    else:
        avg_interval = 0
    
    return {
        "hour_distribution": hour_distribution,
        "day_distribution": day_distribution,
        "average_interval_minutes": round(avg_interval, 2),
        "most_active_hour": max(hour_distribution, key=hour_distribution.get) if hour_distribution else None,
        "most_active_day": max(day_distribution, key=day_distribution.get) if day_distribution else None
    }

def analyze_learning_progress(history):
    """Analyze learning progress over time"""
    if not history:
        return {}
    
    # Sort by timestamp
    sorted_history = sorted([item for item in history if "timestamp" in item], 
                          key=lambda x: x["timestamp"])
    
    # Track confidence progression
    confidence_progression = []
    topic_progression = {}
    
    for i, item in enumerate(sorted_history):
        if "confidence" in item:
            confidence_progression.append({
                "session_number": i + 1,
                "confidence": item["confidence"],
                "timestamp": item["timestamp"]
            })
        
        if "topic" in item and "confidence" in item:
            topic = item["topic"]
            if topic not in topic_progression:
                topic_progression[topic] = []
            topic_progression[topic].append(item["confidence"])
    
    # Calculate improvement trends
    improvement_trends = {}
    for topic, confidences in topic_progression.items():
        if len(confidences) > 1:
            improvement = confidences[-1] - confidences[0]
            improvement_trends[topic] = improvement
    
    return {
        "confidence_progression": confidence_progression,
        "topic_progression": topic_progression,
        "improvement_trends": improvement_trends,
        "overall_trend": "improving" if len(confidence_progression) > 1 and 
                         confidence_progression[-1]["confidence"] > confidence_progression[0]["confidence"] 
                         else "stable"
    }

def analyze_difficulty_progression(history):
    """Analyze difficulty level progression"""
    if not history:
        return {}
    
    # Count difficulty levels
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    difficulty_timeline = []
    
    for item in history:
        # Extract difficulty
        difficulty = item.get("difficulty", "medium").lower()
        if difficulty not in difficulty_counts:
             difficulty_counts[difficulty] = 0
        difficulty_counts[difficulty] += 1
        
        difficulty_timeline.append({
            "timestamp": item.get("timestamp"),
            "difficulty": difficulty,
            "confidence": item.get("confidence", 0)
        })
    
    return {
        "difficulty_distribution": difficulty_counts,
        "difficulty_timeline": difficulty_timeline,
        "preferred_difficulty": max(difficulty_counts, key=difficulty_counts.get) if difficulty_counts else "medium",
        "difficulty_progression": "increasing" if len(difficulty_timeline) > 5 else "stable"
    }

def calculate_engagement_metrics(history):
    """Calculate various engagement metrics"""
    if not history:
        return {}
    
    total_interactions = len(history)
    
    # Question complexity (based on length)
    question_lengths = [item.get("question_length", 0) for item in history]
    avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    
    # Answer quality (based on length and confidence)
    answer_lengths = [item.get("answer_length", 0) for item in history]
    avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
    
    # Session duration estimation
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in history if "timestamp" in item]
    if len(timestamps) > 1:
        session_duration = (timestamps[-1] - timestamps[0]).total_seconds() / 60  # minutes
    else:
        session_duration = 0
    
    # Topic diversity
    topics = set(item.get("topic", "Unknown") for item in history if "topic" in item)
    topic_diversity = len(topics)
    
    return {
        "total_interactions": total_interactions,
        "average_question_length": round(avg_question_length, 2),
        "average_answer_length": round(avg_answer_length, 2),
        "estimated_session_duration_minutes": round(session_duration, 2),
        "topic_diversity": topic_diversity,
        "engagement_score": min(100, (avg_question_length * 0.3 + avg_answer_length * 0.4 + 
                                   topic_diversity * 10 + min(session_duration, 60) * 0.3))
    }

def generate_recommendations(history):
    """Generate personalized learning recommendations"""
    if not history:
        return []
    
    recommendations = []
    
    # Analyze weak topics
    topics = [item.get("topic", "Unknown") for item in history if "topic" in item]
    topic_counts = dict(Counter(topics))
    
    if topic_counts:
        least_practiced = min(topic_counts, key=topic_counts.get)
        recommendations.append(f"Focus more on {least_practiced} - it's your least practiced topic")
    
    # Analyze confidence levels
    confidences = [item.get("confidence", 0) for item in history if "confidence" in item]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < 50:
            recommendations.append("Your confidence levels are low - consider reviewing fundamentals")
        elif avg_confidence > 80:
            recommendations.append("Great confidence! Try more challenging questions")
    
    # Analyze session patterns
    timestamps = [datetime.fromisoformat(item["timestamp"]) for item in history if "timestamp" in item]
    if timestamps:
        recent_activity = [ts for ts in timestamps if (datetime.now() - ts).days <= 7]
        if len(recent_activity) < 3:
            recommendations.append("Try to study more consistently - aim for at least 3 sessions per week")
    
    return recommendations