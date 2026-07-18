import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def predict_forgetting(last_study_date, strength, topic=None, user_history=None, mastery_score=0):
    """
    Enhanced Ebbinghaus Forgetting Curve with Dynamic Stability (S)
    R = e^(-t/S)
    """
    start_time = time.time()
    try:
        if isinstance(last_study_date, str):
            last_study_date = datetime.fromisoformat(last_study_date)
        
        # MATH GUARD: Ensure days_passed is non-negative
        if not last_study_date:
             # Fallback if date missing
             days_passed = 0
             last_study_date = datetime.now()
        else:
             delta = datetime.now() - last_study_date
             days_passed = max(0, delta.days)
        
        # Calculate Dynamic Stability (S)
        # S = Base * Mastery * Repetition * Difficulty
        stability = calculate_dynamic_stability(topic, user_history, mastery_score)
        
        # MATH GUARD: Ensure Stability is positive
        stability = max(0.1, stability)
        
        # Calculate Retention (R)
        # R = e^(-t/S)
        if stability > 0:
            retention = math.exp(-days_passed / stability)
        else:
            retention = 0
            
        # Log Calculation
        duration = time.time() - start_time
        logger.info(f"Forgetting prediction for '{topic}': S={stability:.2f}, R={retention:.2f}, Time={duration:.4f}s")
            
        # Generate spaced repetition schedule based on S
        schedule = generate_spaced_repetition_schedule(last_study_date, stability, retention)
        
        # Determine urgency
        if retention < 0.4: # Higher threshold for urgency with dynamic S
            urgency = "Revise Today"
            priority = "high"
        elif retention < 0.7:
            urgency = "Revise Soon"
            priority = "medium"
        else:
            urgency = "No Revision Needed"
            priority = "low"
        
        return {
            "retention_rate": round(retention, 3),
            "days_since_study": days_passed,
            "stability": round(stability, 2),
            "urgency": urgency,
            "priority": priority,
            "next_review_date": schedule.get("next_review"),
            "spaced_repetition_schedule": schedule,
            "estimated_study_time": schedule.get("estimated_time", "15 minutes")
        }
    except Exception as e:
        logger.error(f"Error in predict_forgetting: {e}", exc_info=True)
        return {
            "retention_rate": 0,
            "days_since_study": 0,
            "stability": 0,
            "urgency": "Unknown",
            "priority": "low",
            "estimated_time": "15 minutes"
        }

def calculate_dynamic_stability(topic, user_history, mastery_score):
    """
    Calculate Stability (S) based on personalization factors
    """
    base_stability = 1.0 # Default stability (days to reach 37% retention)
    
    if not user_history or not topic:
        return base_stability
        
    # 1. Mastery Factor (0.5 to 2.5) - Driven by the ML mastery model
    mastery_factor = 0.5 + (mastery_score / 100) * 2.0 
    
    # 2. Repetition Factor (Log scale)
    topic_history = [h for h in user_history if h.get("topic") == topic]
    revision_count = len(topic_history)
    repetition_factor = 1.0 + math.log1p(revision_count) # ln(1+N)
    
    # 3. Difficulty Modifier
    # Check avg difficulty of topic in history
    difficulties = [h.get("difficulty", "medium") for h in topic_history]
    if difficulties:
         # Crude mode
         counts = {d: difficulties.count(d) for d in set(difficulties)}
         mode_diff = max(counts, key=counts.get)
         if mode_diff == "hard":
             difficulty_modifier = 0.8
         elif mode_diff == "easy":
             difficulty_modifier = 1.2
         else:
             difficulty_modifier = 1.0
    else:
        difficulty_modifier = 1.0

    # 4. Revision History Adjustment (Success/Failure)
    # Look at last validation result if available
    revision_adjustment = 1.0
    if topic_history:
        last_item = topic_history[-1]
        confidence = last_item.get("confidence", 0)
        correct = last_item.get("correct", False) # New field
        
        # If explicitly correct or high confidence
        if correct or confidence > 80:
            revision_adjustment = 1.4 # Boost stability by 40%
        elif confidence < 50:
            revision_adjustment = 0.8 # Decay stability by 20%
            
    # Composite Stability
    S = base_stability * mastery_factor * repetition_factor * difficulty_modifier * revision_adjustment
    
    return max(0.5, S) # Minimum stability of half a day

def calculate_difficulty_factor(history):
    """
    Calculate factor based on difficulty progression
    """
    if not history:
        return 1.0
    
    # Analyze difficulty progression
    recent_items = history[-10:] if len(history) >= 10 else history
    
    easy_count = sum(1 for item in recent_items if "easy" in item.get("question", "").lower())
    hard_count = sum(1 for item in recent_items if "hard" in item.get("question", "").lower())
    
    total = easy_count + hard_count
    if total == 0:
        return 1.0
    
    hard_ratio = hard_count / total
    
    # Users who attempt harder questions may have better retention
    if hard_ratio >= 0.3:
        return 1.15
    elif hard_ratio >= 0.1:
        return 1.05
    else:
        return 1.0

def calculate_topic_mastery_factor(history, topic):
    """
    Calculate mastery factor for specific topic
    """
    if not history or not topic:
        return 1.0
    
    topic_items = [item for item in history if item.get("topic") == topic]
    if not topic_items:
        return 1.0
    
    confidences = [item.get("confidence", 0) for item in topic_items if "confidence" in item]
    if not confidences:
        return 1.0
    
    avg_confidence = sum(confidences) / len(confidences)
    
    # Higher confidence in topic improves retention
    if avg_confidence >= 80:
        return 1.2
    elif avg_confidence >= 60:
        return 1.1
    elif avg_confidence >= 40:
        return 1.0
    else:
        return 0.9

def calculate_learning_style_factor(history):
    """
    Calculate factor based on learning patterns
    """
    if not history:
        return 1.0
    
    # Analyze question patterns
    question_lengths = [len(item.get("question", "")) for item in history[-20:]]
    
    if not question_lengths:
        return 1.0
    
    avg_length = sum(question_lengths) / len(question_lengths)
    
    # Users who ask more detailed questions may have better retention
    if avg_length >= 100:
        return 1.1
    elif avg_length >= 50:
        return 1.05
    else:
        return 1.0

def generate_spaced_repetition_schedule(last_study_date, stability, current_retention):
    """
    Generate optimal spaced repetition schedule based on Stability (S)
    """
    if isinstance(last_study_date, str):
        last_study_date = datetime.fromisoformat(last_study_date)
    
    # Next ideal review is when retention drops to 90% (or other threshold)
    # R = e^(-t/S) -> ln(R) = -t/S -> t = -S * ln(R)
    # Target Retention = 0.9
    optimal_interval_days = -stability * math.log(0.9)
    
    # Enforce minimums
    optimal_interval_days = max(1, int(optimal_interval_days))
    next_review = last_study_date + timedelta(days=optimal_interval_days)
    
    # Generate subsequent intervals for visualization
    intervals = []
    current_S = stability
    for _ in range(5):
        # Assume successful review increases S by 2.5x (standard SM-2 like growth)
        t = -current_S * math.log(0.9)
        intervals.append(max(1, int(t)))
        current_S *= 2.5 
    
    # Estimate time
    if optimal_interval_days <= 1:
        estimated_time = "20 minutes"
    elif optimal_interval_days <= 7:
        estimated_time = "10 minutes"
    else:
        estimated_time = "5 minutes"
    
    return {
        "next_review": next_review.isoformat(),
        "review_intervals": intervals,
        "estimated_time": estimated_time,
        "schedule_type": "dynamic stability"
    }