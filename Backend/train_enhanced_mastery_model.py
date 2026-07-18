import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pickle

class EnhancedMasteryModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_model_type = None
        
    def generate_diverse_training_data(self, num_samples=5000):
        """Generate more diverse and realistic training data"""
        print(f"🔄 Generating {num_samples} diverse training samples...")
        
        data = []
        
        # Define different student archetypes
        archetypes = {
            'beginner': {'accuracy_range': (0.2, 0.5), 'consistency_range': (0.2, 0.5), 'challenge_range': (0.0, 0.2)},
            'average': {'accuracy_range': (0.5, 0.8), 'consistency_range': (0.4, 0.7), 'challenge_range': (0.1, 0.4)},
            'advanced': {'accuracy_range': (0.8, 0.95), 'consistency_range': (0.7, 0.9), 'challenge_range': (0.3, 0.6)},
            'inconsistent': {'accuracy_range': (0.3, 0.8), 'consistency_range': (0.1, 0.4), 'challenge_range': (0.0, 0.5)},
            'struggler': {'accuracy_range': (0.1, 0.4), 'consistency_range': (0.3, 0.6), 'challenge_range': (0.0, 0.1)}
        }
        
        for i in range(num_samples):
            # Randomly select archetype
            archetype = np.random.choice(list(archetypes.keys()))
            archetype_params = archetypes[archetype]
            
            # Generate base learning patterns
            attempts = np.random.randint(5, 100)
            accuracy = np.random.uniform(*archetype_params['accuracy_range'])
            correct = int(attempts * accuracy)
            revisions = np.random.randint(0, max(1, attempts // 3))
            
            # Temporal features with archetype-specific patterns
            learning_consistency = np.random.uniform(*archetype_params['consistency_range'])
            practice_frequency = np.random.exponential(0.2) * learning_consistency
            
            # Difficulty features based on archetype
            challenge_ratio = np.random.uniform(*archetype_params['challenge_range'])
            
            # Difficulty progression (some students improve, others plateau)
            if archetype in ['advanced', 'average']:
                difficulty_progression = np.random.normal(0.1, 0.1)
            else:
                difficulty_progression = np.random.normal(0, 0.15)
            
            # Add noise and edge cases
            if np.random.random() < 0.1:  # 10% edge cases
                attempts = np.random.randint(1, 5)  # Very few attempts
                accuracy = np.random.uniform(0, 1)
            
            # Calculate mastery score with more sophisticated formula
            mastery_score = self._calculate_enhanced_mastery_score({
                'accuracy': accuracy,
                'revision_frequency': revisions / attempts if attempts > 0 else 0,
                'learning_consistency': learning_consistency,
                'practice_frequency': practice_frequency,
                'challenge_ratio': challenge_ratio,
                'difficulty_progression': difficulty_progression,
                'attempts': attempts,
                'archetype': archetype
            })
            
            # Determine mastery level with more nuanced thresholds
            mastery_level = self._determine_mastery_level(mastery_score, archetype, attempts)
            
            # Add additional features for better prediction
            time_since_last_practice = np.random.exponential(24)  # Hours since last practice
            adaptability_score = np.random.beta(2, 2) * learning_consistency
            
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
                'mastery_level': mastery_level,
                'archetype': archetype
            }
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def _calculate_enhanced_mastery_score(self, features):
        """Enhanced mastery score calculation with archetype consideration"""
        weights = {
            'accuracy': 0.25,
            'revision_frequency': 0.15,
            'learning_consistency': 0.15,
            'practice_frequency': 0.10,
            'challenge_ratio': 0.10,
            'difficulty_progression': 0.10,
            'adaptability_score': 0.10,
            'attempts_factor': 0.05
        }
        
        score = 0
        for feature, weight in weights.items():
            value = features.get(feature, 0)
            
            # Normalize based on feature type
            if feature == 'accuracy':
                normalized_value = value
            elif feature == 'revision_frequency':
                normalized_value = min(value / 0.5, 1)
            elif feature in ['learning_consistency', 'challenge_ratio', 'adaptability_score']:
                normalized_value = max(0, min(value, 1))
            elif feature == 'practice_frequency':
                normalized_value = min(value / 0.5, 1)
            elif feature == 'difficulty_progression':
                normalized_value = max(-1, min(value, 1)) * 0.5 + 0.5
            elif feature == 'attempts_factor':
                # Bonus points for more practice (diminishing returns)
                normalized_value = min(np.log(features.get('attempts', 1)) / 5, 1)
            else:
                normalized_value = 0
            
            score += normalized_value * weight
        
        # Apply archetype adjustment
        archetype = features.get('archetype', 'average')
        if archetype == 'advanced':
            score *= 1.1
        elif archetype == 'struggler':
            score *= 0.9
        elif archetype == 'inconsistent':
            score *= 0.95
        
        return min(100, score * 100)
    
    def _determine_mastery_level(self, mastery_score, archetype, attempts):
        """Determine mastery level with archetype-specific adjustments"""
        # Base thresholds
        thresholds = {
            'beginner': [15, 35, 55, 75],
            'average': [25, 45, 65, 80],
            'advanced': [35, 55, 75, 90],
            'inconsistent': [20, 40, 60, 75],
            'struggler': [10, 30, 50, 70]
        }
        
        # Get archetype-specific thresholds
        arch_thresholds = thresholds.get(archetype, thresholds['average'])
        
        # Adjust for attempts (more data = more confidence in classification)
        if attempts < 10:
            # With few attempts, tend toward average
            mastery_score = mastery_score * 0.7 + 50 * 0.3
        
        if mastery_score >= arch_thresholds[3]:
            return 'Strong'
        elif mastery_score >= arch_thresholds[2]:
            return 'Good'
        elif mastery_score >= arch_thresholds[1]:
            return 'Average'
        elif mastery_score >= arch_thresholds[0]:
            return 'Developing'
        else:
            return 'Weak'
    
    def prepare_enhanced_features(self, df):
        """Prepare enhanced features for training"""
        feature_columns = [
            'attempts', 'correct', 'revisions', 'accuracy', 'revision_frequency',
            'learning_consistency', 'practice_frequency', 'challenge_ratio',
            'difficulty_progression', 'time_since_last_practice', 'adaptability_score'
        ]
        
        X = df[feature_columns]
        y = df['mastery_level']
        
        self.feature_names = feature_columns
        return X, y
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best model"""
        print("🔧 Performing hyperparameter tuning...")
        
        # Define models and parameter grids
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        
        best_score = 0
        best_model = None
        best_params = None
        best_model_name = None
        
        for model_name, model_info in models.items():
            print(f"  📊 Tuning {model_name}...")
            grid_search = GridSearchCV(
                model_info['model'], 
                model_info['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_model_name = model_name
            
            print(f"     Best {model_name} CV Score: {grid_search.best_score_:.4f}")
        
        print(f"✅ Best model: {best_model_name} with CV Score: {best_score:.4f}")
        print(f"   Best params: {best_params}")
        
        self.model = best_model
        self.best_model_type = best_model_name
        
        return best_model, best_params
    
    def train_enhanced_model(self, X, y):
        """Train enhanced mastery model with hyperparameter tuning"""
        print("🚀 Training Enhanced Mastery Prediction Model...")
        
        # Check class distribution and handle imbalance
        class_counts = y.value_counts()
        print(f"📊 Class distribution: {dict(class_counts)}")
        
        # Remove classes with too few samples for stratification
        min_samples = 2
        valid_classes = class_counts[class_counts >= min_samples].index
        X_filtered = X[y.isin(valid_classes)]
        y_filtered = y[y.isin(valid_classes)]
        
        if len(y_filtered) < len(y):
            print(f"⚠️  Removed classes with < {min_samples} samples")
            print(f"   New class distribution: {dict(y_filtered.value_counts())}")
        
        # Split data with stratification (if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
        except ValueError:
            # Fallback without stratification if still problematic
            print("⚠️  Using non-stratified split due to class imbalance")
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        best_model, best_params = self.hyperparameter_tuning(X_train_scaled, y_train)
        
        # Train final model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✅ Enhanced Training Accuracy: {train_score:.4f}")
        print(f"✅ Enhanced Test Accuracy: {test_score:.4f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        print("\n📊 Enhanced Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_enhanced_confusion_matrix(cm, y_test.unique())
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self._plot_enhanced_feature_importance()
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'model_type': self.best_model_type,
            'best_params': best_params,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': cm
        }
    
    def _plot_enhanced_confusion_matrix(self, cm, classes):
        """Plot enhanced confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'Enhanced Confusion Matrix - {self.best_model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('data/enhanced_mastery_confusion_matrix.png')
        plt.close()
        print("📈 Enhanced confusion matrix saved to data/enhanced_mastery_confusion_matrix.png")
    
    def _plot_enhanced_feature_importance(self):
        """Plot enhanced feature importance"""
        if not hasattr(self.model, 'feature_importances_'):
            return
            
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Enhanced Feature Importance - {self.best_model_type}")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [self.feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('data/enhanced_mastery_feature_importance.png')
        plt.close()
        print("📈 Enhanced feature importance plot saved to data/enhanced_mastery_feature_importance.png")
    
    def save_enhanced_model(self, filepath='data/enhanced_mastery_model.pkl'):
        """Save enhanced trained model"""
        os.makedirs('data', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.best_model_type,
            'training_date': datetime.now().isoformat(),
            'version': 'enhanced_v2.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Enhanced model saved to {filepath}")
    
    def load_enhanced_model(self, filepath='data/enhanced_mastery_model.pkl'):
        """Load enhanced trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_model_type = model_data.get('model_type', 'RandomForest')
        
        print(f"📂 Enhanced model loaded from {filepath}")
        print(f"📅 Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"🤖 Model type: {self.best_model_type}")
        print(f"📦 Version: {model_data.get('version', 'Unknown')}")

def train_enhanced_mastery_model():
    """Main training function for enhanced mastery model"""
    print("🎯 Starting Enhanced Mastery Model Training Pipeline")
    print("=" * 60)
    
    # Initialize trainer
    trainer = EnhancedMasteryModelTrainer()
    
    # Generate diverse training data
    train_df = trainer.generate_diverse_training_data(5000)
    
    # Prepare enhanced features
    X, y = trainer.prepare_enhanced_features(train_df)
    
    # Train enhanced model
    results = trainer.train_enhanced_model(X, y)
    
    # Save enhanced model
    trainer.save_enhanced_model()
    
    print("\n🎉 Enhanced Training Complete!")
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Best Model Type: {results['model_type']}")
    
    return trainer, results

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Train the enhanced mastery model
    trainer, results = train_enhanced_mastery_model()
