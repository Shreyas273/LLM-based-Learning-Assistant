import numpy as np
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class MasteryModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic training data for mastery prediction"""
        print(f"🔄 Generating {num_samples} synthetic training samples...")
        
        data = []
        for i in range(num_samples):
            # Generate realistic learning patterns
            attempts = np.random.randint(1, 50)
            accuracy = np.random.beta(2, 5)  # Beta distribution for realistic accuracy
            correct = int(attempts * accuracy)
            revisions = np.random.randint(0, max(1, attempts // 2))  # Fixed: ensure at least 1
            
            # Temporal features
            learning_consistency = np.random.beta(3, 2)  # Most users are somewhat consistent
            practice_frequency = np.random.exponential(0.3)  # Exponential for practice frequency
            
            # Difficulty features
            challenge_ratio = np.random.beta(1, 3)  # Most users prefer easier questions
            difficulty_progression = np.random.normal(0, 0.3)  # Normal distribution for progression
            
            # Calculate mastery score
            mastery_score = self._calculate_mastery_score({
                'accuracy': accuracy,
                'revision_frequency': revisions / attempts if attempts > 0 else 0,
                'learning_consistency': learning_consistency,
                'practice_frequency': practice_frequency,
                'challenge_ratio': challenge_ratio,
                'difficulty_progression': difficulty_progression
            })
            
            # Determine mastery level based on score
            if mastery_score >= 80:
                mastery_level = 'Strong'
            elif mastery_score >= 60:
                mastery_level = 'Good'
            elif mastery_score >= 40:
                mastery_level = 'Average'
            elif mastery_score >= 20:
                mastery_level = 'Developing'
            else:
                mastery_level = 'Weak'
            
            sample = {
                'attempts': attempts,
                'correct': correct,
                'revisions': revisions,
                'accuracy': accuracy,
                'revision_frequency': revisions / attempts if attempts > 0 else 0,
                'learning_consistency': learning_consistency,
                'practice_frequency': practice_frequency,
                'challenge_ratio': challenge_ratio,
                'difficulty_progression': difficulty_progression,
                'mastery_score': mastery_score,
                'mastery_level': mastery_level
            }
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def _calculate_mastery_score(self, features):
        """Calculate mastery score from features"""
        weights = {
            'accuracy': 0.3,
            'revision_frequency': 0.2,
            'learning_consistency': 0.15,
            'practice_frequency': 0.15,
            'challenge_ratio': 0.1,
            'difficulty_progression': 0.1
        }
        
        score = 0
        for feature, weight in weights.items():
            value = features.get(feature, 0)
            # Normalize values to 0-1 range
            if feature == 'accuracy':
                normalized_value = value
            elif feature == 'revision_frequency':
                normalized_value = min(value / 0.5, 1)
            elif feature in ['learning_consistency', 'challenge_ratio']:
                normalized_value = max(0, min(value, 1))
            elif feature == 'practice_frequency':
                normalized_value = min(value / 0.5, 1)
            else:
                normalized_value = max(-1, min(value, 1)) * 0.5 + 0.5
            
            score += normalized_value * weight
        
        return min(100, score * 100)
    
    def prepare_features(self, df):
        """Prepare features for training"""
        feature_columns = [
            'attempts', 'correct', 'revisions', 'accuracy', 'revision_frequency',
            'learning_consistency', 'practice_frequency', 'challenge_ratio',
            'difficulty_progression'
        ]
        
        X = df[feature_columns]
        y = df['mastery_level']
        
        self.feature_names = feature_columns
        return X, y
    
    def train(self, X, y):
        """Train the mastery prediction model"""
        print("🚀 Training Mastery Prediction Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✅ Training Accuracy: {train_score:.4f}")
        print(f"✅ Test Accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"✅ Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed classification report
        y_pred = self.model.predict(X_test_scaled)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, y_test.unique())
        
        # Feature importance
        self._plot_feature_importance()
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_score': cv_scores.mean(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm, classes):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix - Mastery Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('data/mastery_confusion_matrix.png')
        plt.close()
        print("📈 Confusion matrix saved to data/mastery_confusion_matrix.png")
    
    def _plot_feature_importance(self):
        """Plot feature importance"""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance - Mastery Prediction")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [self.feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('data/mastery_feature_importance.png')
        plt.close()
        print("📈 Feature importance plot saved to data/mastery_feature_importance.png")
    
    def save_model(self, filepath='data/mastery_model.pkl'):
        """Save trained model"""
        os.makedirs('data', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='data/mastery_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        print(f"📂 Model loaded from {filepath}")
        print(f"📅 Training date: {model_data.get('training_date', 'Unknown')}")
    
    def test_model(self, test_samples=100):
        """Test the trained model with new data"""
        print(f"🧪 Testing model with {test_samples} new samples...")
        
        # Generate test data
        test_df = self.generate_synthetic_data(test_samples)
        X_test, y_test = self.prepare_features(test_df)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        print(f"✅ Test Samples: {len(y_test)}")
        
        # Show some predictions
        print("\n🔍 Sample Predictions:")
        for i in range(min(5, len(y_test))):
            confidence = np.max(y_pred_proba[i])
            print(f"  True: {y_test.iloc[i]}, Predicted: {y_pred[i]}, Confidence: {confidence:.3f}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_test.values,
            'probabilities': y_pred_proba
        }

def train_mastery_model():
    """Main training function for mastery model"""
    print("🎯 Starting Mastery Model Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = MasteryModelTrainer()
    
    # Generate training data
    train_df = trainer.generate_synthetic_data(2000)
    
    # Prepare features
    X, y = trainer.prepare_features(train_df)
    
    # Train model
    results = trainer.train(X, y)
    
    # Save model
    trainer.save_model()
    
    # Test model
    test_results = trainer.test_model(200)
    
    print("\n🎉 Training Complete!")
    print(f"Final Test Accuracy: {test_results['accuracy']:.4f}")
    
    return trainer, results, test_results

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Train the mastery model
    trainer, results, test_results = train_mastery_model()
