import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import logging
import time
import re

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def predict_mastery(attempts, correct, revisions, topic=None, user_history=None):
    """
    Advanced ML-based mastery prediction with multiple features
    """
    start_time = time.time()
    try:
        # Safety Check: Insufficient Data
        if attempts < 1 and (not user_history or len(user_history) == 0):
             logger.info(f"Insufficient data for topic: {topic}")
             return {
                "mastery_level": "Insufficient Data",
                "mastery_score": 0,
                "confidence": 0.0,
                "analysis": {
                    "strengths": [],
                    "areas_for_improvement": ["Start learning to generate data"],
                    "learning_patterns": {},
                    "next_steps": ["Start a study session", "Take a quiz"]
                },
                "recommendations": ["Start a study session", "Take a quiz"]
            }

        # Load user history from DB if not provided
        if not user_history:
             try:
                 from services.memory_service import generate_session_report
                 report = generate_session_report("current") # get recent
                 user_history = report.get("recent_activity", [])
             except Exception as e:
                 logger.warning(f"Failed to load user history: {e}")
                 user_history = []
        
        # Extract features
        features = extract_mastery_features(attempts, correct, revisions, topic, user_history)
        
        # Log extraction time & features
        duration = time.time() - start_time
        logger.info(f"Mastery prediction for '{topic}' took {duration:.4f}s. Features: {features}")

        # Use trained model if available, otherwise fallback to heuristic
        if os.path.exists("data/enhanced_mastery_model.pkl"):
            mastery_level = predict_with_enhanced_model(features)
        elif os.path.exists("data/mastery_model.pkl"):
            mastery_level = predict_with_model(features)
        else:
            mastery_level = heuristic_mastery_prediction(features)
        
        # Generate detailed mastery analysis
        analysis = generate_mastery_analysis(features, mastery_level, topic, user_history)
        
        return {
            "mastery_level": mastery_level,
            "mastery_score": features.get("mastery_score", 0),
            "confidence": features.get("prediction_confidence", 0.5),
            "analysis": analysis,
            "recommendations": generate_mastery_recommendations(features, mastery_level)
        }
        
    except Exception as e:
        logger.error(f"Error in mastery prediction: {e}", exc_info=True)
        return {
            "mastery_level": "Unavailable",
            "mastery_score": 0,
            "confidence": 0,
            "analysis": {
                "strengths": [], 
                "areas_for_improvement": [], 
                "learning_patterns": {}, 
                "next_steps": ["Continue studying"]
            },
            "recommendations": []
        }

def extract_mastery_features(attempts, correct, revisions, topic, user_history):
    """
    Extract comprehensive features for mastery prediction with Rolling Window & Difficulty Weighting
    """
    # 1. Rolling Window Filter (Last 30 Days)
    recent_history = filter_recent_history(user_history, days=30)
    
    # 2. Filter by Topic
    topic_history = [h for h in recent_history if h.get("topic") == topic]
    
    # 3. Calculate Weighted Accuracy based on Difficulty
    weighted_score = 0
    total_potential = 0
    
    difficulty_weights = {"easy": 1, "medium": 2, "hard": 3}
    
    for item in topic_history:
        diff = item.get("difficulty", "medium").lower()
        weight = difficulty_weights.get(diff, 2)
        
        is_correct = item.get("correct", False)
        # Fallback to confidence if correct not available (legacy data)
        if "correct" not in item:
             is_correct = item.get("confidence", 0) > 70.0
             
        if is_correct:
            weighted_score += weight
        
        total_potential += weight
        
    weighted_accuracy = weighted_score / total_potential if total_potential > 0 else 0
    
    # 4. Response Time (Inverse Speed)
    response_times = [item.get("response_time") for item in topic_history if item.get("response_time")]
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        # Normalize: heuristic 10s is fast (1.0), 60s is slow (0.0)
        speed_score = max(0, min(1, 1 - (avg_time - 10) / 50)) 
    else:
        speed_score = 0.5 # Default
        
    # 5. Retention / Revision
    # Simple revision count from history not just passed arg
    revision_count = len(topic_history) # Total interactions in window
    
    features = {
        "attempts": len(topic_history),
        "weighted_accuracy": weighted_accuracy,
        "speed_score": speed_score,
        "revision_count": revision_count,
        # Legacy for compat if needed, but we rely on weighted_accuracy now
        "accuracy": weighted_accuracy 
    }
    
    # Add temporal features if history available (using generic history)
    if user_history:
        temporal_features = extract_temporal_features(user_history, topic)
        features.update(temporal_features)
    
    # Add difficulty progression features
    if user_history:
        difficulty_features = extract_difficulty_features(user_history, topic)
        features.update(difficulty_features)
    
    # Calculate composite probabilistc score
    features["mastery_score"] = calculate_probabilistic_mastery(features)
    
    return features

def filter_recent_history(history, days=30):
    if not history: return []
    cutoff = datetime.now() - timedelta(days=days)
    recent = []
    for item in history:
        try:
            # Handle string timestamps
            ts_str = item.get("timestamp")
            if ts_str:
                ts = datetime.fromisoformat(ts_str)
                if ts >= cutoff:
                    recent.append(item)
        except:
            pass
    return recent

def extract_temporal_features(user_history, topic):
    """
    Extract temporal learning behaviour features
    """
    if not user_history:
        return {
            "learning_consistency": 0,
            "practice_frequency": 0
        }

    topic_history = [h for h in user_history if h.get("topic") == topic]

    if not topic_history:
        return {
            "learning_consistency": 0,
            "practice_frequency": 0
        }

    timestamps = []
    for item in topic_history:
        try:
            if "timestamp" in item:
                timestamps.append(datetime.fromisoformat(item["timestamp"]))
        except:
            pass

    if len(timestamps) < 2:
        return {
            "learning_consistency": 0.5,
            "practice_frequency": len(timestamps)
        }

    timestamps.sort()

    gaps = []
    for i in range(1, len(timestamps)):
        gaps.append((timestamps[i] - timestamps[i-1]).days)

    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    learning_consistency = max(0, min(1, 1 / (1 + avg_gap)))

    practice_frequency = len(timestamps) / 30

    return {
        "learning_consistency": learning_consistency,
        "practice_frequency": practice_frequency
    }

def extract_difficulty_features(user_history, topic):
    """
    Extract difficulty progression features
    """
    if not user_history:
        return {
            "challenge_ratio": 0,
            "difficulty_progression": 0
        }

    topic_history = [h for h in user_history if h.get("topic") == topic]

    if not topic_history:
        return {
            "challenge_ratio": 0,
            "difficulty_progression": 0
        }

    hard = 0
    total = 0

    for item in topic_history:
        diff = item.get("difficulty", "medium")

        if diff == "hard":
            hard += 1

        total += 1

    challenge_ratio = hard / total if total else 0

    return {
        "challenge_ratio": challenge_ratio,
        "difficulty_progression": challenge_ratio
    }

def calculate_probabilistic_mastery(features):
    """
    Calculate Probability of Mastery P(M|Features) using Logistic Regression-like logic
    Sigmoid(z) where z = w1*acc + w2*speed + w3*diff + bias
    """
    # Weights (tuned heuristically)
    w_acc = 4.0      # Accuracy is most important
    w_speed = 1.0    # Speed matters
    w_cons = 1.5     # Consistency matters
    w_vol = 0.5      # Volume (log scale)
    bias = -3.0      # Shift so 50% acc isn't 50% mastery (need > 60-70%)
    
    acc = features.get("weighted_accuracy", 0)
    speed = features.get("speed_score", 0.5)
    cons = features.get("learning_consistency", 0)
    vol = np.log1p(features.get("revision_count", 0)) # Log volume
    
    z = (w_acc * acc) + (w_speed * speed) + (w_cons * cons) + (w_vol * vol) + bias
    
    probability = 1 / (1 + np.exp(-z))
    
    return probability * 100 # Scale to 0-100 for UI compatibility

# Temporal and Difficulty extraction functions remain largely the same, 
# assuming user_history structure matches what memory_service returns (list of dicts).

def calculate_mastery_score(features):
    """
    Calculate composite mastery score from all features
    """
    weights = {
        "accuracy": 0.35, # Increased weight
        "revision_frequency": 0.15,
        "learning_consistency": 0.15,
        "practice_frequency": 0.15,
        "challenge_ratio": 0.1,
        "difficulty_progression": 0.1
    }
    
    score = 0
    for feature, weight in weights.items():
        value = features.get(feature, 0)
        # Normalize values to 0-1 range
        if feature == "accuracy":
            normalized_value = value
        elif feature == "revision_frequency":
            normalized_value = min(value / 0.5, 1)  # Normalize to max 0.5
        elif feature in ["learning_consistency", "challenge_ratio", "adaptability_score"]:
            normalized_value = max(0, min(value, 1))
        elif feature == "practice_frequency":
            normalized_value = min(value / 0.5, 1)  # Normalize to max 0.5 per day
        else:
            normalized_value = max(-1, min(value, 1)) * 0.5 + 0.5  # Normalize -1 to 1 -> 0 to 1
        
        score += normalized_value * weight
    
    return min(100, score * 100)  # Convert to 0-100 scale

def predict_with_enhanced_model(features):
    """
    Predict mastery using enhanced ML model
    """
    try:
        with open("data/enhanced_mastery_model.pkl", "rb") as f:
            model_data = pickle.load(f)
            
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]
        
        # Prepare feature vector with proper feature names
        feature_dict = {}
        for name in feature_names:
            feature_dict[name] = features.get(name, 0)
        
        # Create DataFrame with proper column names
        feature_df = pd.DataFrame([feature_dict], columns=feature_names)
        
        # Scale features
        scaled_features = scaler.transform(feature_df)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        confidence = max(model.predict_proba(scaled_features)[0])
        
        features["prediction_confidence"] = confidence
        
        return prediction
        
    except Exception as e:
        print(f"Error using enhanced ML model: {e}")
        return heuristic_mastery_prediction(features)

def predict_with_model(features):
    """
    Predict mastery using trained ML model
    """
    try:
        with open("data/mastery_model.pkl", "rb") as f:
            model_data = pickle.load(f)
            
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]
        
        # Prepare feature vector with proper feature names
        feature_dict = {}
        for name in feature_names:
            feature_dict[name] = features.get(name, 0)
        
        # Create DataFrame with proper column names
        feature_df = pd.DataFrame([feature_dict], columns=feature_names)
        
        # Scale features
        scaled_features = scaler.transform(feature_df)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        confidence = max(model.predict_proba(scaled_features)[0])
        
        features["prediction_confidence"] = confidence
        
        return prediction
        
    except Exception as e:
        print(f"Error using ML model: {e}")
        return heuristic_mastery_prediction(features)

def heuristic_mastery_prediction(features):
    """
    Fallback heuristic prediction
    """
def heuristic_mastery_prediction(features):
    """
    Fallback heuristic prediction based on probabilistic score
    """
    mastery_score = features.get("mastery_score", 0)
    
    # Probability Levels
    # Score is 0-100
    if mastery_score >= 80:
        return "Expert"
    elif mastery_score >= 50:
        return "Intermediate"
    else:
        return "Novice"

def calculate_effective_mastery(mastery_score, retention_probability):
    """
    Calculate Effective Mastery = Mastery Score * Retention Probability
    Returns 0.0 - 100.0
    """
    # Guards
    mastery = max(0, min(100, mastery_score))
    retention = max(0, min(1.0, retention_probability))
    
    return mastery * retention

def generate_mastery_analysis(features, mastery_level, topic, user_history):
    """
    Generate detailed mastery analysis
    """
    analysis = {
        "strengths": [],
        "areas_for_improvement": [],
        "learning_patterns": {},
        "next_steps": []
    }
    
    # Analyze strengths
    if features.get("accuracy", 0) >= 0.8:
        analysis["strengths"].append("High accuracy in answers")
    if features.get("learning_consistency", 0) >= 0.7:
        analysis["strengths"].append("Consistent learning pattern")
    if features.get("challenge_ratio", 0) >= 0.3:
        analysis["strengths"].append("Willingness to tackle challenges")
    
    # Analyze areas for improvement
    if features.get("accuracy", 0) < 0.6:
        analysis["areas_for_improvement"].append("Focus on improving answer accuracy")
    if features.get("revision_frequency", 0) < 0.2:
        analysis["areas_for_improvement"].append("Increase revision frequency")
    if features.get("learning_consistency", 0) < 0.5:
        analysis["areas_for_improvement"].append("Maintain more consistent study schedule")
    
    # Learning patterns
    analysis["learning_patterns"] = {
        "best_time": identify_best_study_time(user_history),
        "optimal_difficulty": identify_optimal_difficulty(user_history),
        "preferred_topics": identify_preferred_topics(user_history)
    }
    
    # Next steps based on mastery level
    if mastery_level in ["Weak", "Developing"]:
        analysis["next_steps"].append("Focus on fundamental concepts")
        analysis["next_steps"].append("Practice with easier questions")
    elif mastery_level == "Average":
        analysis["next_steps"].append("Mix of practice and revision")
        analysis["next_steps"].append("Try medium difficulty questions")
    else:
        analysis["next_steps"].append("Challenge with advanced problems")
        analysis["next_steps"].append("Help others with explanations")
    
    return analysis

def generate_mastery_recommendations(features, mastery_level):
    """
    Generate personalized recommendations
    """
    recommendations = []
    
    # Based on mastery level
    if mastery_level == "Weak":
        recommendations.append("Start with basic concepts and build foundation")
        recommendations.append("Use visual aids and examples for better understanding")
    elif mastery_level == "Developing":
        recommendations.append("Practice more questions to build confidence")
        recommendations.append("Review mistakes and understand corrections")
    elif mastery_level == "Average":
        recommendations.append("Challenge yourself with medium-difficulty problems")
        recommendations.append("Teach concepts to others for reinforcement")
    elif mastery_level == "Good":
        recommendations.append("Tackle complex problems and edge cases")
        recommendations.append("Explore advanced topics and applications")
    else:  # Strong
        recommendations.append("Mentor others and create study materials")
        recommendations.append("Explore interdisciplinary connections")
    
    # Based on specific features
    if features.get("revision_frequency", 0) < 0.2:
        recommendations.append("Implement spaced repetition for better retention")
    
    if features.get("learning_consistency", 0) < 0.5:
        recommendations.append("Create a regular study schedule")
    
    if features.get("challenge_ratio", 0) < 0.2:
        recommendations.append("Gradually increase difficulty to build resilience")
    
    return recommendations

def identify_best_study_time(history):
    """
    Identify most productive study times
    """
    if not history:
        return "Not enough data"
    
    hour_performance = {}
    for item in history:
        if "timestamp" in item and "confidence" in item:
            hour = datetime.fromisoformat(item["timestamp"]).hour
            if hour not in hour_performance:
                hour_performance[hour] = []
            hour_performance[hour].append(item["confidence"])
    
    best_hour = None
    best_avg = 0
    
    for hour, confidences in hour_performance.items():
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence > best_avg:
            best_avg = avg_confidence
            best_hour = hour
    
    if best_hour:
        if 6 <= best_hour < 12:
            return f"Morning ({best_hour}:00-{best_hour+1}:00)"
        elif 12 <= best_hour < 18:
            return f"Afternoon ({best_hour}:00-{best_hour+1}:00)"
        else:
            return f"Evening ({best_hour}:00-{best_hour+1}:00)"
    
    return "Not enough data"

def identify_optimal_difficulty(history):
    """
    Identify optimal difficulty level for user
    """
    if not history:
        return "Not enough data"
    
    difficulty_performance = {"easy": [], "medium": [], "hard": []}
    
    for item in history:
        question = item.get("question", "").lower()
        confidence = item.get("confidence", 0)
        
        if "easy" in question:
            difficulty_performance["easy"].append(confidence)
        elif "hard" in question:
            difficulty_performance["hard"].append(confidence)
        else:
            difficulty_performance["medium"].append(confidence)
    
    best_difficulty = "medium"
    best_avg = 0
    
    for difficulty, confidences in difficulty_performance.items():
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence > best_avg:
                best_avg = avg_confidence
                best_difficulty = difficulty
    
    return best_difficulty.capitalize()

def identify_preferred_topics(history):
    """
    Identify topics user engages with most
    """
    if not history:
        return []
    
    topic_counts = {}
    for item in history:
        topic = item.get("topic", "Unknown")
        if topic != "Unknown":
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Sort by frequency and return top 3
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, count in sorted_topics[:3]]


def detect_learning_style(user_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Classify learner style from interaction history patterns.

    Styles:
    - Visual learner: often asks for diagrams/flowcharts/visualization.
    - Reading learner: often asks for explanations/notes/summaries/definitions.
    - Problem-solving learner: often asks to solve problems, code, practice questions.

    Backward compatible: returns a dict with label + scores.
    """
    if not user_history:
        return {
            "learning_style": "Reading learner",
            "scores": {"visual": 0.0, "reading": 0.0, "problem_solving": 0.0},
            "evidence": {"samples": []},
        }

    visual_kw = [
        "flowchart", "diagram", "draw", "visual", "mind map", "chart", "graph", "illustrate",
    ]
    reading_kw = [
        "explain", "explanation", "notes", "theory", "definition", "summarize", "summary",
        "in detail", "elaborate", "write", "describe",
    ]
    problem_kw = [
        "solve", "calculate", "derive", "prove", "implement", "code", "dry run",
        "practice", "quiz", "mcq", "exercise", "question", "program",
    ]

    def score_text(text: str, kws: List[str]) -> int:
        t = (text or "").lower()
        return sum(1 for k in kws if k in t)

    recent = user_history[-30:] if len(user_history) > 30 else user_history
    v = r = p = 0
    samples = []
    for item in recent:
        q = item.get("question", "") or ""
        v += score_text(q, visual_kw)
        r += score_text(q, reading_kw)
        p += score_text(q, problem_kw)
        if (len(samples) < 5) and (v or r or p):
            samples.append(q[:140])

    # Secondary signal: difficulty + correctness (problem-solvers tend to attempt harder items)
    hard_attempts = sum(1 for it in recent if str(it.get("difficulty", "medium")).lower() == "hard")
    incorrect = sum(1 for it in recent if (it.get("correct") is False))
    total = max(1, len(recent))

    p += int((hard_attempts / total) * 6)
    r += int(((1 - (incorrect / total)) * 2))  # consistent correctness often correlates with reading/explaining

    scores = {
        "visual": float(v),
        "reading": float(r),
        "problem_solving": float(p),
    }
    best = max(scores.items(), key=lambda x: x[1])[0]
    label_map = {
        "visual": "Visual learner",
        "reading": "Reading learner",
        "problem_solving": "Problem-solving learner",
    }
    return {
        "learning_style": label_map[best],
        "scores": scores,
        "evidence": {"samples": samples},
    }


def compute_topic_mastery_score(topic: str, user_history: List[Dict[str, Any]]) -> float:
    """
    Compute a topic mastery score (0-100) using existing feature pipeline.
    This is a lightweight wrapper for analytics/planning (no model training).
    """
    if not topic:
        return 0.0
    features = extract_mastery_features(
        attempts=0,
        correct=0,
        revisions=0,
        topic=topic,
        user_history=user_history or [],
    )
    try:
        return float(features.get("mastery_score", 0.0))
    except Exception:
        return 0.0