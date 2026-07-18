import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

class SimpleEnhancedMasteryTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15)
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_diverse_training_data(self, num_samples=3000):
        """Generate diverse training data without excessive complexity"""
        print(f"🔄 Generating {num_samples} diverse training samples...")
        
        data = []
        
        # Student archetypes with balanced distribution
        archetypes = ['beginner', 'average', 'advanced', 'inconsistent', 'struggler']
        archetype_weights = [0.2, 0.3, 0.2, 0.2, 0.1]  # Balanced weights
        
        for i in range(num_samples):
            # Select archetype with weighted probability
            archetype = np.random.choice(archetypes, p=archetype_weights)
            
            # Generate learning patterns based on archetype
            if archetype == 'beginner':
                attempts = np.random.randint(5, 30)
                accuracy = np.random.uniform(0.2, 0.5)
                consistency = np.random.uniform(0.2, 0.5)
                challenge_ratio = np.random.uniform(0.0, 0.2)
            elif archetype == 'average':
                attempts = np.random.randint(10, 60)
                accuracy = np.random.uniform(0.5, 0.8)
                consistency = np.random.uniform(0.4, 0.7)
                challenge_ratio = np.random.uniform(0.1, 0.4)
            elif archetype == 'advanced':
                attempts = np.random.randint(20, 100)
                accuracy = np.random.uniform(0.8, 0.95)
                consistency = np.random.uniform(0.7, 0.9)
                challenge_ratio = np.random.uniform(0.3, 0.6)
            elif archetype == 'inconsistent':
                attempts = np.random.randint(5, 80)
                accuracy = np.random.uniform(0.3, 0.8)
                consistency = np.random.uniform(0.1, 0.4)
                challenge_ratio = np.random.uniform(0.0, 0.5)
            else:  # struggler
                attempts = np.random.randint(5, 40)
                accuracy = np.random.uniform(0.1, 0.4)
                consistency = np.random.uniform(0.3, 0.6)
                challenge_ratio = np.random.uniform(0.0, 0.1)
            
            correct = int(attempts * accuracy)
            revisions = np.random.randint(0, max(1, attempts // 3))
            
            # Other features
            learning_consistency = consistency
            practice_frequency = np.random.exponential(0.2) * consistency
            difficulty_progression = np.random.normal(0, 0.1)
            time_since_last_practice = np.random.exponential(24)
            adaptability_score = np.random.beta(2, 2) * consistency
            
            # Calculate mastery score
            mastery_score = self._calculate_mastery_score({
                'accuracy': accuracy,
                'revision_frequency': revisions / attempts if attempts > 0 else 0,
                'learning_consistency': learning_consistency,
                'practice_frequency': practice_frequency,
                'challenge_ratio': challenge_ratio,
                'difficulty_progression': difficulty_progression,
                'adaptability_score': adaptability_score
            })
            
            # Determine mastery level
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
                'time_since_last_practice': time_since_last_practice,
                'adaptability_score': adaptability_score,
                'mastery_score': mastery_score,
                'mastery_level': mastery_level
            }
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def _calculate_mastery_score(self, features):
        """Calculate mastery score"""
        weights = {
            'accuracy': 0.25,
            'revision_frequency': 0.15,
            'learning_consistency': 0.15,
            'practice_frequency': 0.10,
            'challenge_ratio': 0.10,
            'difficulty_progression': 0.10,
            'adaptability_score': 0.10
        }
        
        score = 0
        for feature, weight in weights.items():
            value = features.get(feature, 0)
            
            if feature == 'accuracy':
                normalized_value = value
            elif feature == 'revision_frequency':
                normalized_value = min(value / 0.5, 1)
            elif feature in ['learning_consistency', 'challenge_ratio', 'adaptability_score']:
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
            'difficulty_progression', 'time_since_last_practice', 'adaptability_score'
        ]
        
        X = df[feature_columns]
        y = df['mastery_level']
        
        self.feature_names = feature_columns
        return X, y
    
    def train_model(self, X, y):
        """Train the model"""
        print("🚀 Training Enhanced Mastery Model...")
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"📊 Class distribution: {dict(class_counts)}")
        
        # Handle class imbalance
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index
        X_filtered = X[y.isin(valid_classes)]
        y_filtered = y[y.isin(valid_classes)]
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42
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
        
        # Detailed evaluation
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
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm, classes):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Enhanced Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('data/enhanced_mastery_confusion_matrix.png')
        plt.close()
        print("📈 Confusion matrix saved")
    
    def _plot_feature_importance(self):
        """Plot feature importance"""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Enhanced Feature Importance")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [self.feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('data/enhanced_mastery_feature_importance.png')
        plt.close()
        print("📈 Feature importance plot saved")
    
    def save_model(self, filepath='data/enhanced_mastery_model.pkl'):
        """Save trained model"""
        os.makedirs('data', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'version': 'enhanced_v2.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Enhanced model saved to {filepath}")

def train_simple_enhanced_model():
    """Main training function"""
    print("🎯 Starting Simple Enhanced Mastery Model Training")
    print("=" * 50)
    
    trainer = SimpleEnhancedMasteryTrainer()
    train_df = trainer.generate_diverse_training_data(3000)
    
    X, y = trainer.prepare_features(train_df)
    results = trainer.train_model(X, y)
    trainer.save_model()
    
    print("\n🎉 Training Complete!")
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
    
    return trainer, results

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    trainer, results = train_simple_enhanced_model()
