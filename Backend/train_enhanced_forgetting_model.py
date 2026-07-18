import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import json
import os
from typing import Dict, List, Tuple

from database.repositories import interactions_collection

class RetentionDataCollector:
    def __init__(self):
        self.retention_data = []
        
    def simulate_real_retention_data(self, num_users=500, days_per_user=90):
        """Generate realistic retention data with individual variations"""
        print(f"🔄 Generating realistic retention data for {num_users} users over {days_per_user} days...")
        
        data = []
        
        for user_id in range(num_users):
            # User-specific learning characteristics
            base_strength = np.random.gamma(2, 5)  # More realistic distribution
            learning_rate = np.random.beta(2, 2)  # Learning ability
            forgetting_rate = np.random.beta(2, 3)  # Individual forgetting rate
            consistency = np.random.beta(3, 2)  # Study consistency
            
            # Generate learning sessions
            for day in range(days_per_user):
                # Study probability based on consistency
                if np.random.random() < consistency:
                    # Study session occurs
                    study_strength = base_strength * learning_rate * (1 + np.random.normal(0, 0.1))
                    
                    # Track retention over time for this session
                    for hours_after in [1, 6, 24, 72, 168, 336, 720]:  # Up to 1 month
                        # Enhanced forgetting curve with individual factors
                        base_retention = np.exp(-hours_after / (study_strength + 1))
                        
                        # Apply individual forgetting rate adjustment
                        individual_adjustment = 1 - (forgetting_rate * 0.3)
                        
                        # Add realistic noise and factors
                        noise = np.random.normal(0, 0.05)
                        time_of_day_effect = 0.05 * np.sin(hours_after * np.pi / 12)  # Daily variation
                        difficulty_factor = np.random.uniform(0.8, 1.2)  # Topic difficulty
                        
                        retention = base_retention * individual_adjustment + noise + time_of_day_effect
                        retention = retention * difficulty_factor
                        retention = max(0.05, min(1.0, retention))  # Keep within realistic bounds
                        
                        data.append({
                            'user_id': user_id,
                            'day': day,
                            'hours_after_study': hours_after,
                            'retention_rate': retention,
                            'base_strength': base_strength,
                            'learning_rate': learning_rate,
                            'forgetting_rate': forgetting_rate,
                            'consistency': consistency,
                            'study_strength': study_strength,
                            'difficulty_factor': difficulty_factor
                        })
        
        return pd.DataFrame(data)
    
    def collect_real_interaction_data(self):
        """
        Collect data from real user interactions stored in MongoDB.

        We approximate retention by measuring how a user's confidence changes
        between consecutive interactions on the same topic. For each pair of
        consecutive interactions (i, j) for a user and topic:
          - hours_after_study = hours between timestamps
          - retention_rate   = normalized confidence at j (0–1)
        """
        print("📊 Collecting real interaction data from MongoDB...")

        try:
            cursor = interactions_collection.find({}).sort("timestamp", 1)
            docs = list(cursor)
        except Exception as e:
            print(f"⚠️ Failed to load interactions from MongoDB: {e}")
            return pd.DataFrame([])

        if not docs:
            print("ℹ️ No interactions found in MongoDB; returning empty real-data frame.")
            return pd.DataFrame([])

        records = []

        # Group interactions by (userId, topic)
        by_user_topic: Dict[Tuple[str, str], List[dict]] = {}
        for doc in docs:
            user_id = str(doc.get("userId", "unknown"))
            topic = doc.get("topic") or "Unknown"
            ts = doc.get("timestamp")
            if not isinstance(ts, datetime):
                try:
                    ts = datetime.fromisoformat(str(ts))
                except Exception:
                    continue

            key = (user_id, topic)
            by_user_topic.setdefault(key, []).append(
                {
                    "timestamp": ts,
                    "confidence": float(doc.get("confidence", 0.0)),
                    "difficulty": doc.get("difficulty", "medium"),
                }
            )

        # For each user/topic, build (hours_after_study, retention_rate) samples
        for (user_id, topic), history in by_user_topic.items():
            if len(history) < 2:
                continue

            # Ensure chronological order
            history.sort(key=lambda x: x["timestamp"])

            for i in range(len(history) - 1):
                t0 = history[i]["timestamp"]
                t1 = history[i + 1]["timestamp"]
                delta_hours = (t1 - t0).total_seconds() / 3600.0
                if delta_hours <= 0:
                    continue

                conf_next = history[i + 1]["confidence"]
                retention = max(0.05, min(1.0, conf_next / 100.0))

                records.append(
                    {
                        "user_id": user_id,
                        "topic": topic,
                        "hours_after_study": delta_hours,
                        "retention_rate": retention,
                    }
                )

        if not records:
            print("ℹ️ Not enough interaction pairs to derive retention; returning empty frame.")
            return pd.DataFrame([])

        df = pd.DataFrame(records)
        print(f"✅ Collected {len(df)} real retention samples from {len(by_user_topic)} user-topic groups.")
        return df

class EnhancedForgettingCurveModel:
    def __init__(self):
        self.parameters = {}
        self.individual_factors = {}
        self.model_type = "enhanced_ebbinghaus"
        
    def enhanced_forgetting_curve(self, t, R0, S, F, C):
        """
        Enhanced forgetting curve with additional parameters
        R(t) = R0 * exp(-t/S) * (1 - F) + C
        
        Where:
        R(t) = retention at time t
        R0 = initial retention
        S = memory strength
        F = forgetting rate modifier
        C = long-term retention baseline
        """
        return R0 * np.exp(-t / S) * (1 - F) + C
    
    def fit_enhanced_model(self, data):
        """Fit enhanced forgetting curve model"""
        print("🔧 Fitting Enhanced Forgetting Curve Model...")
        
        # Group by user for individual fitting
        user_params = {}
        
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            
            if len(user_data) < 5:
                continue
            
            try:
                # Fit enhanced curve for this user
                popt, pcov = curve_fit(
                    self.enhanced_forgetting_curve,
                    user_data['hours_after_study'],
                    user_data['retention_rate'],
                    p0=[0.95, 15, 0.1, 0.05],  # Initial guesses
                    bounds=([0.5, 1, 0, 0], [1.0, 50, 0.5, 0.2])
                )
                
                user_params[user_id] = {
                    'R0': popt[0],
                    'S': popt[1],
                    'F': popt[2],
                    'C': popt[3],
                    'fit_quality': np.sqrt(np.diag(pcov)).mean()
                }
                
            except Exception as e:
                # Use default parameters if fitting fails
                user_params[user_id] = {
                    'R0': 0.95,
                    'S': 15,
                    'F': 0.1,
                    'C': 0.05,
                    'fit_quality': 1.0
                }
        
        # Calculate population parameters
        R0_values = [params['R0'] for params in user_params.values()]
        S_values = [params['S'] for params in user_params.values()]
        F_values = [params['F'] for params in user_params.values()]
        C_values = [params['C'] for params in user_params.values()]
        
        self.parameters = {
            'population_R0': np.mean(R0_values),
            'population_S': np.mean(S_values),
            'population_F': np.mean(F_values),
            'population_C': np.mean(C_values),
            'R0_std': np.std(R0_values),
            'S_std': np.std(S_values),
            'F_std': np.std(F_values),
            'C_std': np.std(C_values),
            'individual_params': user_params
        }
        
        print(f"✅ Enhanced Population Parameters:")
        print(f"   Initial Retention (R0): {self.parameters['population_R0']:.3f} ± {self.parameters['R0_std']:.3f}")
        print(f"   Memory Strength (S): {self.parameters['population_S']:.2f} ± {self.parameters['S_std']:.2f}")
        print(f"   Forgetting Rate (F): {self.parameters['population_F']:.3f} ± {self.parameters['F_std']:.3f}")
        print(f"   Baseline Retention (C): {self.parameters['population_C']:.3f} ± {self.parameters['C_std']:.3f}")
        print(f"   Fitted {len(user_params)} individual curves")
        
        return self.parameters
    
    def predict_enhanced_retention(self, hours_after, user_id=None, individual_factors=None):
        """Predict retention using enhanced model"""
        if user_id and user_id in self.parameters.get('individual_params', {}):
            # Use individual parameters
            params = self.parameters['individual_params'][user_id]
            R0, S, F, C = params['R0'], params['S'], params['F'], params['C']
        else:
            # Use population parameters
            R0 = self.parameters.get('population_R0', 0.95)
            S = self.parameters.get('population_S', 15)
            F = self.parameters.get('population_F', 0.1)
            C = self.parameters.get('population_C', 0.05)
        
        # Apply individual adjustments if provided
        if individual_factors:
            consistency_factor = individual_factors.get('learning_consistency', 1.0)
            mastery_factor = individual_factors.get('topic_mastery', 1.0)
            
            # Adjust parameters based on individual factors
            R0 *= mastery_factor
            S *= consistency_factor
            F *= (2 - consistency_factor)  # Less consistent users forget faster
        
        retention = self.enhanced_forgetting_curve(hours_after, R0, S, F, C)
        return max(0.05, min(1.0, retention))
    
    def generate_adaptive_spaced_repetition(self, last_study_date, strength, current_retention, user_factors=None):
        """Generate adaptive spaced repetition schedule"""
        print("📅 Generating Adaptive Spaced Repetition Schedule...")
        
        schedule = {
            'intervals': [],
            'next_review': None,
            'estimated_time': None,
            'adaptation_reason': None
        }
        
        # Adaptive base intervals based on performance
        if current_retention > 0.8:
            base_intervals = [1, 4, 10, 25, 60, 120, 240]  # Longer intervals for high retention
            adaptation = "high_retention"
        elif current_retention > 0.6:
            base_intervals = [1, 3, 7, 18, 45, 90, 180]  # Standard intervals
            adaptation = "standard"
        elif current_retention > 0.4:
            base_intervals = [1, 2, 5, 12, 30, 60, 120]  # Shorter intervals
            adaptation = "medium_retention"
        else:
            base_intervals = [0.5, 1, 3, 8, 20, 40, 80]  # Very short intervals
            adaptation = "low_retention"
        
        # Adjust intervals based on user factors
        if user_factors:
            consistency = user_factors.get('learning_consistency', 0.5)
            multiplier = 0.5 + consistency  # 0.5 to 1.5 multiplier
            adjusted_intervals = [int(interval * multiplier) for interval in base_intervals]
        else:
            adjusted_intervals = base_intervals
        
        # Find next review time using enhanced prediction
        next_review_hours = None
        for interval in adjusted_intervals:
            predicted_retention = self.predict_enhanced_retention(interval, individual_factors=user_factors)
            if predicted_retention < 0.7:  # Review when retention drops below 70%
                next_review_hours = interval
                break
        
        if not next_review_hours:
            next_review_hours = adjusted_intervals[-1]
        
        # Calculate next review date
        next_review = last_study_date + timedelta(hours=next_review_hours)
        
        # Adaptive study time estimation
        if current_retention > 0.8:
            estimated_time = "10 minutes"
        elif current_retention > 0.6:
            estimated_time = "15 minutes"
        elif current_retention > 0.4:
            estimated_time = "20 minutes"
        else:
            estimated_time = "30 minutes"
        
        schedule['intervals'] = adjusted_intervals[:5]
        schedule['next_review'] = next_review.isoformat()
        schedule['estimated_time'] = estimated_time
        schedule['adaptation_reason'] = adaptation
        schedule['next_review_hours'] = next_review_hours
        
        return schedule
    
    def evaluate_model_accuracy(self, test_data):
        """Evaluate enhanced model accuracy"""
        print("🧪 Evaluating Enhanced Model Accuracy...")
        
        predictions = []
        actual = []
        
        for _, row in test_data.iterrows():
            pred = self.predict_enhanced_retention(row['hours_after_study'])
            predictions.append(pred)
            actual.append(row['retention_rate'])
        
        # Calculate metrics
        mse = np.mean((np.array(predictions) - np.array(actual)) ** 2)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actual)))
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((np.array(actual) - np.array(predictions)) ** 2)
        ss_tot = np.sum((np.array(actual) - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"✅ Enhanced Model Performance:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²: {r2:.6f}")
        
        # Plot results
        self._plot_model_evaluation(actual, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actual': actual
        }
    
    def _plot_model_evaluation(self, actual, predicted):
        """Plot model evaluation results"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Predicted vs Actual
        plt.subplot(1, 3, 1)
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Retention')
        plt.ylabel('Predicted Retention')
        plt.title('Predicted vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        plt.subplot(1, 3, 2)
        residuals = np.array(predicted) - np.array(actual)
        plt.scatter(predicted, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Retention')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution
        plt.subplot(1, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/enhanced_forgetting_model_evaluation.png')
        plt.close()
        print("📈 Model evaluation plots saved to data/enhanced_forgetting_model_evaluation.png")
    
    def save_enhanced_model(self, filepath='data/enhanced_forgetting_model.json'):
        """Save enhanced model parameters"""
        os.makedirs('data', exist_ok=True)
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        model_data = {
            'parameters': convert_numpy_types(self.parameters),
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat(),
            'version': 'enhanced_v2.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"💾 Enhanced model saved to {filepath}")

def train_enhanced_forgetting_model():
    """Main training function for enhanced forgetting model"""
    print("🧠 Starting Enhanced Forgetting Curve Model Training Pipeline")
    print("=" * 60)
    
    # Initialize data collector
    collector = RetentionDataCollector()
    
    # Prefer real interaction data if available; otherwise fall back to simulation.
    train_data = collector.collect_real_interaction_data()
    if train_data is None or len(train_data) == 0:
        print("ℹ️ Falling back to simulated retention data (no sufficient real data yet).")
        train_data = collector.simulate_real_retention_data(500, 90)
    
    # Initialize and train enhanced model
    model = EnhancedForgettingCurveModel()
    parameters = model.fit_enhanced_model(train_data)
    
    # Generate test data for evaluation (use simulation for controlled testing)
    test_data = collector.simulate_real_retention_data(100, 30)
    
    # Evaluate model
    evaluation_results = model.evaluate_model_accuracy(test_data)
    
    # Save enhanced model
    model.save_enhanced_model()
    
    print("\n🎉 Enhanced Training Complete!")
    print(f"Final Model R²: {evaluation_results['r2']:.6f}")
    print(f"Final Model MAE: {evaluation_results['mae']:.6f}")
    
    return model, parameters, evaluation_results

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Train the enhanced forgetting model
    model, parameters, results = train_enhanced_forgetting_model()
