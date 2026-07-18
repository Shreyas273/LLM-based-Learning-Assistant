import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import sys

# Add the services directory to the path
sys.path.append('services')

from train_mastery_model import MasteryModelTrainer
from train_forgetting_model import ForgettingCurveTrainer

class ModelTester:
    def __init__(self):
        self.mastery_trainer = None
        self.forgetting_trainer = None
        
    def load_models(self):
        """Load both trained models"""
        print("📂 Loading trained models...")
        
        try:
            # Load mastery model
            self.mastery_trainer = MasteryModelTrainer()
            self.mastery_trainer.load_model()
            
            # Load forgetting model
            self.forgetting_trainer = ForgettingCurveTrainer()
            self.forgetting_trainer.load_model()
            
            print("✅ Both models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def test_mastery_model(self):
        """Comprehensive testing of mastery model"""
        print("\n🎯 Testing Mastery Model")
        print("=" * 40)
        
        # Test with different user profiles
        test_profiles = [
            {
                'name': 'Beginner Student',
                'attempts': 5,
                'correct': 2,
                'revisions': 1,
                'expected_level': 'Weak'
            },
            {
                'name': 'Average Student',
                'attempts': 20,
                'correct': 12,
                'revisions': 8,
                'expected_level': 'Average'
            },
            {
                'name': 'Advanced Student',
                'attempts': 50,
                'correct': 45,
                'revisions': 15,
                'expected_level': 'Strong'
            },
            {
                'name': 'Inconsistent Learner',
                'attempts': 30,
                'correct': 18,
                'revisions': 20,
                'expected_level': 'Developing'
            }
        ]
        
        results = []
        
        for profile in test_profiles:
            print(f"\n👤 Testing: {profile['name']}")
            
            # Generate synthetic history for this profile
            history = self._generate_user_history(profile)
            
            # Test prediction
            from services.ml.mastery_model import predict_mastery
            prediction = predict_mastery(
                attempts=profile['attempts'],
                correct=profile['correct'],
                revisions=profile['revisions'],
                user_history=history
            )
            
            print(f"   Expected: {profile['expected_level']}")
            print(f"   Predicted: {prediction['mastery_level']}")
            print(f"   Confidence: {prediction['confidence']:.3f}")
            print(f"   Score: {prediction['mastery_score']:.1f}")
            
            results.append({
                'profile': profile['name'],
                'expected': profile['expected_level'],
                'predicted': prediction['mastery_level'],
                'confidence': prediction['confidence'],
                'score': prediction['mastery_score']
            })
        
        # Calculate accuracy
        correct = sum(1 for r in results if r['expected'] == r['predicted'])
        accuracy = correct / len(results)
        
        print(f"\n📊 Profile Test Accuracy: {accuracy:.2f} ({correct}/{len(results)})")
        
        return results
    
    def test_forgetting_model(self):
        """Comprehensive testing of forgetting model"""
        print("\n🧠 Testing Forgetting Model")
        print("=" * 40)
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Just Studied',
                'hours_after': 1,
                'expected_retention': 85
            },
            {
                'name': 'One Day Later',
                'hours_after': 24,
                'expected_retention': 60
            },
            {
                'name': 'One Week Later',
                'hours_after': 168,
                'expected_retention': 30
            },
            {
                'name': 'One Month Later',
                'hours_after': 720,
                'expected_retention': 15
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            print(f"\n⏰ Testing: {scenario['name']}")
            
            # Test prediction
            predicted = self.forgetting_trainer.predict_retention(scenario['hours_after'])
            
            print(f"   Hours after study: {scenario['hours_after']}")
            print(f"   Expected retention: {scenario['expected_retention']:.1f}%")
            print(f"   Predicted retention: {predicted:.1f}%")
            print(f"   Error: {abs(predicted - scenario['expected_retention']):.1f}%")
            
            results.append({
                'scenario': scenario['name'],
                'hours_after': scenario['hours_after'],
                'expected': scenario['expected_retention'],
                'predicted': predicted,
                'error': abs(predicted - scenario['expected_retention'])
            })
        
        # Calculate average error
        avg_error = np.mean([r['error'] for r in results])
        
        print(f"\n📊 Average Prediction Error: {avg_error:.2f}%")
        
        return results
    
    def test_spaced_repetition(self):
        """Test spaced repetition scheduling"""
        print("\n📅 Testing Spaced Repetition Scheduling")
        print("=" * 40)
        
        # Test different mastery levels
        mastery_levels = [
            {'retention': 90, 'name': 'High Mastery'},
            {'retention': 70, 'name': 'Medium Mastery'},
            {'retention': 40, 'name': 'Low Mastery'}
        ]
        
        results = []
        
        for level in mastery_levels:
            print(f"\n🎯 Testing: {level['name']} (Retention: {level['retention']}%)")
            
            # Generate schedule
            last_study = datetime.now()
            schedule = self.forgetting_trainer.generate_spaced_repetition_schedule(
                last_study_date=last_study,
                strength=10,
                current_retention=level['retention']
            )
            
            next_review = datetime.fromisoformat(schedule['next_review'])
            hours_until_review = (next_review - last_study).total_seconds() / 3600
            
            print(f"   Next review: {next_review.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Hours until review: {hours_until_review:.1f}")
            print(f"   Estimated study time: {schedule['estimated_time']}")
            print(f"   Review intervals: {schedule['intervals'][:3]}...")
            
            results.append({
                'mastery_level': level['name'],
                'retention': level['retention'],
                'hours_until_review': hours_until_review,
                'study_time': schedule['estimated_time']
            })
        
        return results
    
    def test_integration(self):
        """Test integration between both models"""
        print("\n🔗 Testing Model Integration")
        print("=" * 40)
        
        # Simulate a learning session
        print("📚 Simulating Learning Session...")
        
        # User starts with low mastery
        attempts, correct, revisions = 0, 0, 0
        history = []
        
        for session in range(5):
            print(f"\n📖 Study Session {session + 1}")
            
            # Simulate learning progress
            attempts += np.random.randint(3, 8)
            correct += np.random.randint(2, 6)
            revisions += np.random.randint(0, 3)
            
            # Test mastery prediction
            from services.ml.mastery_model import predict_mastery
            mastery_pred = predict_mastery(attempts, correct, revisions, user_history=history)
            
            # Test forgetting prediction
            forgetting_pred = self.forgetting_trainer.predict_retention(24)  # 24 hours after
            
            # Add to history
            history.append({
                'timestamp': datetime.now().isoformat(),
                'mastery_level': mastery_pred['mastery_level'],
                'confidence': mastery_pred['confidence'],
                'retention_prediction': forgetting_pred
            })
            
            print(f"   Mastery Level: {mastery_pred['mastery_level']} (Score: {mastery_pred['mastery_score']:.1f})")
            print(f"   Confidence: {mastery_pred['confidence']:.3f}")
            print(f"   Predicted Retention (24h): {forgetting_pred:.1f}%")
        
        print(f"\n📈 Learning Progress Summary:")
        print(f"   Total Attempts: {attempts}")
        print(f"   Final Mastery: {history[-1]['mastery_level']}")
        print(f"   Final Confidence: {history[-1]['confidence']:.3f}")
        
        return history
    
    def _generate_user_history(self, profile):
        """Generate synthetic user history for testing"""
        history = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(profile['attempts']):
            timestamp = base_date + timedelta(days=i * 30 // profile['attempts'])
            
            # Simulate confidence based on profile
            if profile['name'] == 'Beginner Student':
                confidence = np.random.uniform(0.3, 0.6)
            elif profile['name'] == 'Average Student':
                confidence = np.random.uniform(0.5, 0.8)
            elif profile['name'] == 'Advanced Student':
                confidence = np.random.uniform(0.7, 0.95)
            else:  # Inconsistent
                confidence = np.random.uniform(0.2, 0.9)
            
            history.append({
                'timestamp': timestamp.isoformat(),
                'confidence': confidence * 100,
                'topic': f'topic_{i % 5}',
                'mastery_level': 'unknown'
            })
        
        return history
    
    def generate_test_report(self, mastery_results, forgetting_results, spaced_results, integration_results):
        """Generate comprehensive test report"""
        print("\n📋 Generating Test Report")
        print("=" * 40)
        
        report = {
            'test_date': datetime.now().isoformat(),
            'mastery_model': {
                'profile_tests': mastery_results,
                'accuracy': sum(1 for r in mastery_results if r['expected'] == r['predicted']) / len(mastery_results)
            },
            'forgetting_model': {
                'scenario_tests': forgetting_results,
                'average_error': np.mean([r['error'] for r in forgetting_results])
            },
            'spaced_repetition': spaced_results,
            'integration_test': integration_results
        }
        
        # Save report
        with open('data/model_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("💾 Test report saved to data/model_test_report.json")
        
        # Create summary plots
        self._create_summary_plots(mastery_results, forgetting_results)
        
        return report
    
    def _create_summary_plots(self, mastery_results, forgetting_results):
        """Create summary visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Mastery Model Predictions
        profiles = [r['profile'] for r in mastery_results]
        predicted = [r['predicted'] for r in mastery_results]
        expected = [r['expected'] for r in mastery_results]
        
        x = np.arange(len(profiles))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, expected, width, label='Expected', alpha=0.7)
        axes[0, 0].bar(x + width/2, predicted, width, label='Predicted', alpha=0.7)
        axes[0, 0].set_xlabel('User Profile')
        axes[0, 0].set_ylabel('Mastery Level')
        axes[0, 0].set_title('Mastery Model: Expected vs Predicted')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(profiles, rotation=45)
        axes[0, 0].legend()
        
        # Plot 2: Forgetting Model Predictions
        scenarios = [r['scenario'] for r in forgetting_results]
        predicted_retention = [r['predicted'] for r in forgetting_results]
        expected_retention = [r['expected'] for r in forgetting_results]
        
        axes[0, 1].plot(scenarios, expected_retention, 'o-', label='Expected', linewidth=2)
        axes[0, 1].plot(scenarios, predicted_retention, 's--', label='Predicted', linewidth=2)
        axes[0, 1].set_xlabel('Time After Study')
        axes[0, 1].set_ylabel('Retention Rate (%)')
        axes[0, 1].set_title('Forgetting Model: Expected vs Predicted')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Model Confidence
        axes[1, 0].bar(profiles, [r['confidence'] for r in mastery_results], alpha=0.7)
        axes[1, 0].set_xlabel('User Profile')
        axes[1, 0].set_ylabel('Prediction Confidence')
        axes[1, 0].set_title('Mastery Model Confidence Scores')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Forgetting Model Error
        axes[1, 1].bar(scenarios, [r['error'] for r in forgetting_results], alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Time After Study')
        axes[1, 1].set_ylabel('Prediction Error (%)')
        axes[1, 1].set_title('Forgetting Model Prediction Error')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/model_test_summary.png')
        plt.close()
        print("📈 Test summary plots saved to data/model_test_summary.png")
    
    def run_complete_test_suite(self):
        """Run complete test suite for both models"""
        print("🚀 Starting Complete Model Test Suite")
        print("=" * 50)
        
        # Load models
        if not self.load_models():
            return False
        
        # Run all tests
        mastery_results = self.test_mastery_model()
        forgetting_results = self.test_forgetting_model()
        spaced_results = self.test_spaced_repetition()
        integration_results = self.test_integration()
        
        # Generate report
        report = self.generate_test_report(mastery_results, forgetting_results, spaced_results, integration_results)
        
        print("\n🎉 Complete Test Suite Finished!")
        print(f"📊 Mastery Model Accuracy: {report['mastery_model']['accuracy']:.2f}")
        print(f"📊 Forgetting Model Avg Error: {report['forgetting_model']['average_error']:.2f}%")
        
        return report

def main():
    """Main function to run all tests"""
    print("🧪 AI Learning Assistant - Model Testing Suite")
    print("=" * 50)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize tester
    tester = ModelTester()
    
    # Run complete test suite
    report = tester.run_complete_test_suite()
    
    if report:
        print("\n✅ All tests completed successfully!")
        print("📁 Check data/ directory for detailed reports and plots")
    else:
        print("\n❌ Tests failed. Please train models first.")

if __name__ == "__main__":
    main()
