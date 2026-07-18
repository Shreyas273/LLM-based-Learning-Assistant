import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import json
import os

class ForgettingCurveTrainer:
    def __init__(self):
        self.parameters = {}
        self.individual_factors = {}
        
    def ebbinghaus_forgetting_curve(self, t, R0, S):
        """
        Ebbinghaus forgetting curve formula
        R(t) = R0 * exp(-t/S)
        where:
        R(t) = retention at time t
        R0 = initial retention (at t=0)
        S = strength of memory (time constant)
        """
        return R0 * np.exp(-t / S)
    
    def generate_synthetic_data(self, num_users=100, days_per_user=30):
        """Generate synthetic forgetting curve data"""
        print(f"🔄 Generating forgetting data for {num_users} users over {days_per_user} days...")
        
        data = []
        
        for user_id in range(num_users):
            # Individual learning characteristics
            base_strength = np.random.uniform(5, 20)  # Memory strength
            learning_rate = np.random.uniform(0.8, 1.2)  # Individual learning rate
            consistency = np.random.uniform(0.5, 1.0)  # Study consistency
            
            for day in range(days_per_user):
                # Simulate study sessions
                if np.random.random() < consistency:  # Probability of studying
                    # Generate retention data
                    for hours_after in [1, 6, 24, 72, 168]:  # 1hr, 6hr, 1day, 3day, 1week
                        # Add noise to simulate real-world variation
                        noise = np.random.normal(0, 0.1)
                        
                        # Calculate retention with individual factors
                        retention = self.ebbinghaus_forgetting_curve(
                            hours_after, 
                            100 * learning_rate,  # Initial retention
                            base_strength * learning_rate
                        )
                        
                        retention = max(0, min(100, retention + noise * 100))
                        
                        data.append({
                            'user_id': user_id,
                            'day': day,
                            'hours_after_study': hours_after,
                            'retention_rate': retention,
                            'base_strength': base_strength,
                            'learning_rate': learning_rate,
                            'consistency': consistency,
                            'study_session': True
                        })
        
        return pd.DataFrame(data)
    
    def fit_forgetting_curve(self, data):
        """Fit forgetting curve parameters to data"""
        print("🔧 Fitting Forgetting Curve Parameters...")
        
        # Group by user to fit individual curves
        user_params = {}
        
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            
            if len(user_data) < 3:  # Need at least 3 points for fitting
                continue
            
            try:
                # Fit curve for this user
                popt, pcov = curve_fit(
                    self.ebbinghaus_forgetting_curve,
                    user_data['hours_after_study'],
                    user_data['retention_rate'],
                    p0=[100, 10],  # Initial guess: R0=100, S=10
                    bounds=([50, 1], [100, 50])  # Reasonable bounds
                )
                
                user_params[user_id] = {
                    'R0': popt[0],
                    'S': popt[1],
                    'fit_quality': np.sqrt(np.diag(pcov)).mean()
                }
                
            except:
                # Use default parameters if fitting fails
                user_params[user_id] = {
                    'R0': 100,
                    'S': 10,
                    'fit_quality': 1.0
                }
        
        # Calculate population averages
        R0_values = [params['R0'] for params in user_params.values()]
        S_values = [params['S'] for params in user_params.values()]
        
        self.parameters = {
            'population_R0': np.mean(R0_values),
            'population_S': np.mean(S_values),
            'R0_std': np.std(R0_values),
            'S_std': np.std(S_values),
            'individual_params': user_params
        }
        
        print(f"✅ Population Parameters:")
        print(f"   Initial Retention (R0): {self.parameters['population_R0']:.2f} ± {self.parameters['R0_std']:.2f}")
        print(f"   Memory Strength (S): {self.parameters['population_S']:.2f} ± {self.parameters['S_std']:.2f}")
        print(f"   Fitted {len(user_params)} individual curves")
        
        return self.parameters
    
    def calculate_individual_factors(self, data):
        """Calculate individual learning factors"""
        print("🧮 Calculating Individual Learning Factors...")
        
        individual_factors = {}
        
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            
            if len(user_data) < 5:
                continue
            
            # Calculate various metrics
            avg_retention = user_data['retention_rate'].mean()
            retention_std = user_data['retention_rate'].std()
            study_frequency = len(user_data) / 30  # Sessions per day
            
            # Learning consistency (inverse of retention variance)
            consistency = 1 / (1 + retention_std) if retention_std > 0 else 1
            
            # Difficulty preference (based on retention patterns)
            difficulty_preference = 'adaptive'  # Could be calculated from question types
            
            # Topic mastery (could be calculated from topic-specific data)
            topic_mastery = avg_retention / 100  # Normalize to 0-1
            
            individual_factors[user_id] = {
                'learning_consistency': consistency,
                'difficulty_preference': difficulty_preference,
                'topic_mastery': topic_mastery,
                'study_frequency': study_frequency,
                'avg_retention': avg_retention
            }
        
        self.individual_factors = individual_factors
        print(f"✅ Calculated factors for {len(individual_factors)} users")
        
        return individual_factors
    
    def predict_retention(self, hours_after, user_id=None, individual_factors=None):
        """Predict retention rate for given time after study"""
        if user_id and user_id in self.parameters.get('individual_params', {}):
            # Use individual parameters
            params = self.parameters['individual_params'][user_id]
            R0, S = params['R0'], params['S']
        else:
            # Use population parameters
            R0 = self.parameters.get('population_R0', 100)
            S = self.parameters.get('population_S', 10)
        
        # Apply individual adjustments if provided
        if individual_factors:
            consistency_factor = individual_factors.get('learning_consistency', 1.0)
            mastery_factor = individual_factors.get('topic_mastery', 1.0)
            
            # Adjust parameters based on individual factors
            R0 *= mastery_factor
            S *= consistency_factor
        
        retention = self.ebbinghaus_forgetting_curve(hours_after, R0, S)
        return max(0, min(100, retention))
    
    def generate_spaced_repetition_schedule(self, last_study_date, strength, current_retention):
        """Generate optimal spaced repetition schedule"""
        print("📅 Generating Spaced Repetition Schedule...")
        
        schedule = {
            'intervals': [],
            'next_review': None,
            'estimated_time': None
        }
        
        # Standard spaced repetition intervals (in hours)
        base_intervals = [1, 6, 24, 72, 168, 336, 720]  # 1hr, 6hr, 1d, 3d, 1w, 2w, 1mo
        
        # Adjust intervals based on current performance
        if current_retention > 80:
            # Good retention - increase intervals
            multiplier = 1.5
        elif current_retention < 50:
            # Poor retention - decrease intervals
            multiplier = 0.7
        else:
            multiplier = 1.0
        
        adjusted_intervals = [int(interval * multiplier) for interval in base_intervals]
        
        # Find next review time
        for interval in adjusted_intervals:
            predicted_retention = self.predict_retention(interval)
            if predicted_retention < 70:  # Review when retention drops below 70%
                next_review_hours = interval
                break
        else:
            next_review_hours = adjusted_intervals[-1]
        
        # Calculate next review date
        next_review = last_study_date + timedelta(hours=next_review_hours)
        
        # Estimate study time based on retention
        if current_retention > 80:
            estimated_time = "10 minutes"
        elif current_retention > 60:
            estimated_time = "15 minutes"
        else:
            estimated_time = "25 minutes"
        
        schedule['intervals'] = adjusted_intervals[:5]  # First 5 intervals
        schedule['next_review'] = next_review.isoformat()
        schedule['estimated_time'] = estimated_time
        
        return schedule
    
    def plot_forgetting_curves(self):
        """Plot forgetting curves for visualization"""
        print("📊 Generating Forgetting Curve Plots...")
        
        plt.figure(figsize=(12, 8))
        
        # Time points (hours after study)
        time_points = np.logspace(0, 3, 100)  # 1 hour to 1000 hours
        
        # Plot population average curve
        population_retention = [
            self.predict_retention(t) for t in time_points
        ]
        plt.plot(time_points, population_retention, 'b-', linewidth=3, label='Population Average')
        
        # Plot sample individual curves
        sample_users = list(self.parameters.get('individual_params', {}).keys())[:5]
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, user_id in enumerate(sample_users):
            individual_retention = [
                self.predict_retention(t, user_id) for t in time_points
            ]
            plt.plot(time_points, individual_retention, '--', 
                    color=colors[i], alpha=0.7, label=f'User {user_id}')
        
        plt.xscale('log')
        plt.xlabel('Hours After Study')
        plt.ylabel('Retention Rate (%)')
        plt.title('Forgetting Curves - Population vs Individuals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/forgetting_curves.png')
        plt.close()
        print("📈 Forgetting curves plot saved to data/forgetting_curves.png")
    
    def save_model(self, filepath='data/forgetting_model_params.json'):
        """Save trained model parameters"""
        os.makedirs('data', exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
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
            'individual_factors': convert_numpy_types(self.individual_factors),
            'training_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"💾 Model parameters saved to {filepath}")
    
    def load_model(self, filepath='data/forgetting_model_params.json'):
        """Load trained model parameters"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.parameters = model_data['parameters']
        self.individual_factors = model_data['individual_factors']
        
        print(f"📂 Model loaded from {filepath}")
        print(f"📅 Training date: {model_data.get('training_date', 'Unknown')}")
    
    def test_model(self, test_samples=50):
        """Test the forgetting curve model"""
        print(f"🧪 Testing forgetting curve model with {test_samples} samples...")
        
        # Generate test data
        test_data = self.generate_synthetic_data(test_samples, 7)
        
        # Test predictions
        predictions = []
        actual = []
        
        for _, row in test_data.iterrows():
            pred = self.predict_retention(row['hours_after_study'])
            predictions.append(pred)
            actual.append(row['retention_rate'])
        
        # Calculate metrics
        mse = np.mean((np.array(predictions) - np.array(actual)) ** 2)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actual)))
        
        print(f"✅ Mean Squared Error: {mse:.4f}")
        print(f"✅ Mean Absolute Error: {mae:.4f}")
        
        # Plot prediction vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predictions, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Retention (%)')
        plt.ylabel('Predicted Retention (%)')
        plt.title('Forgetting Curve Model: Predicted vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/forgetting_model_test.png')
        plt.close()
        print("📈 Test results plot saved to data/forgetting_model_test.png")
        
        return {
            'mse': mse,
            'mae': mae,
            'predictions': predictions,
            'actual': actual
        }

def train_forgetting_model():
    """Main training function for forgetting curve model"""
    print("🧠 Starting Forgetting Curve Model Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ForgettingCurveTrainer()
    
    # Generate training data
    train_data = trainer.generate_synthetic_data(200, 60)
    
    # Fit forgetting curve
    parameters = trainer.fit_forgetting_curve(train_data)
    
    # Calculate individual factors
    individual_factors = trainer.calculate_individual_factors(train_data)
    
    # Generate plots
    trainer.plot_forgetting_curves()
    
    # Save model
    trainer.save_model()
    
    # Test model
    test_results = trainer.test_model(100)
    
    print("\n🎉 Training Complete!")
    print(f"Final Test MSE: {test_results['mse']:.4f}")
    print(f"Final Test MAE: {test_results['mae']:.4f}")
    
    return trainer, parameters, individual_factors, test_results

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Train the forgetting curve model
    trainer, parameters, individual_factors, test_results = train_forgetting_model()
