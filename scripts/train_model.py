"""
Train ML Model for IoT Sign Language Glove
===========================================

This script trains a machine learning model using collected glove data.

Features:
- Loads CSV data from collected samples
- Preprocesses and normalizes sensor values
- Trains Random Forest classifier (proven to work well with flex sensors)
- Evaluates model performance
- Saves trained model for deployment

Usage:
    python train_model.py --data data/my_glove_data --output models/my_model.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import sys
import argparse

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class GloveModelTrainer:
    """Trains ML model for ASL recognition from glove sensor data"""
    
    def __init__(self, data_dir, model_name='my_glove_model'):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load all CSV files from the data directory"""
        print("\n📂 Loading data...")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.data_dir}")
            return False
        
        print(f"   Found {len(csv_files)} CSV files")
        
        # Load all CSV files
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️  Error loading {csv_file.name}: {e}")
        
        if not dfs:
            print("❌ No data could be loaded")
            return False
        
        # Combine all data
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(self.df)} samples total")
        
        # Show dataset statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(self.df)}")
        print(f"   Features: {[col for col in self.df.columns if col.startswith('flex_')]}")
        print(f"\n   Samples per letter:")
        for letter, count in self.df['label'].value_counts().sort_index().items():
            print(f"      {letter}: {count} samples")
        
        return True
    
    def preprocess_data(self):
        """Preprocess the data for training"""
        print("\n🔧 Preprocessing data...")
        
        # Extract features (flex sensors)
        feature_cols = [col for col in self.df.columns if col.startswith('flex_')]
        X = self.df[feature_cols].values
        y = self.df['label'].values
        
        print(f"   Features: {feature_cols}")
        print(f"   Shape: {X.shape}")
        
        # Check for any NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("⚠️  Found NaN or Inf values, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1000, neginf=0)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y  # Ensure balanced split
        )
        
        # Normalize features
        print("   Normalizing features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✅ Training set: {len(self.X_train)} samples")
        print(f"✅ Test set: {len(self.X_test)} samples")
        
        return True
    
    def train_random_forest(self, tune_hyperparameters=True):
        """
        Train Random Forest classifier
        
        Random Forest works very well with sensor data and is fast to train.
        """
        print("\n🌲 Training Random Forest classifier...")
        
        if tune_hyperparameters:
            print("   🔍 Tuning hyperparameters (this may take a few minutes)...")
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, 
                param_grid, 
                cv=3, 
                n_jobs=-1, 
                scoring='f1_macro',
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            
            print(f"   ✅ Best parameters: {grid_search.best_params_}")
            print(f"   ✅ Best CV score: {grid_search.best_score_:.3f}")
        else:
            # Use default parameters (faster)
            print("   Using default parameters...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
        
        print("✅ Training complete!")
        
    def train_gradient_boosting(self):
        """
        Train Gradient Boosting classifier (alternative to Random Forest)
        
        Sometimes works better but takes longer to train.
        """
        print("\n🚀 Training Gradient Boosting classifier...")
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("✅ Training complete!")
    
    def evaluate(self):
        """Evaluate the trained model"""
        print("\n📊 Evaluating model...")
        
        # Predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Accuracy
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        
        print(f"\n   Training Accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
        print(f"   Test Accuracy:     {test_acc:.3f} ({test_acc*100:.1f}%)")
        
        # F1 Score
        train_f1 = f1_score(self.y_train, y_pred_train, average='macro')
        test_f1 = f1_score(self.y_test, y_pred_test, average='macro')
        
        print(f"\n   Training F1 Score: {train_f1:.3f}")
        print(f"   Test F1 Score:     {test_f1:.3f}")
        
        # Classification report
        print("\n📋 Classification Report (Test Set):")
        print(classification_report(self.y_test, y_pred_test))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        
        # Feature importance (if Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            print("\n🔍 Feature Importance:")
            feature_names = [f'Finger_{i+1}' for i in range(5)]
            importances = self.model.feature_importances_
            for name, importance in zip(feature_names, importances):
                print(f"   {name}: {importance:.3f}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        labels = sorted(self.df['label'].unique())
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Saved confusion matrix to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, output_dir='models'):
        """Save the trained model, scaler, and metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\n💾 Saving model...")
        
        # Save model
        model_file = output_path / f"{self.model_name}_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"   ✅ Model: {model_file}")
        
        # Save scaler
        scaler_file = output_path / f"{self.model_name}_scaler_{timestamp}.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✅ Scaler: {scaler_file}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'timestamp': timestamp,
            'model_type': type(self.model).__name__,
            'n_features': self.X_train.shape[1],
            'n_samples': len(self.df),
            'labels': sorted(self.df['label'].unique().tolist()),
            'data_directory': str(self.data_dir)
        }
        
        metadata_file = output_path / f"{self.model_name}_metadata_{timestamp}.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"   ✅ Metadata: {metadata_file}")
        
        print(f"\n✨ Model saved successfully!")
        print(f"\n📦 To use this model:")
        print(f"   1. Copy these files to your mobile/desktop app:")
        print(f"      - {model_file.name}")
        print(f"      - {scaler_file.name}")
        print(f"   2. Load them using pickle in your prediction code")
        
        return model_file
    
    def run_full_pipeline(self, model_type='rf', tune=True, save=True):
        """
        Run the complete training pipeline
        
        Args:
            model_type: 'rf' (Random Forest) or 'gb' (Gradient Boosting)
            tune: Whether to tune hyperparameters
            save: Whether to save the model
        """
        # Load data
        if not self.load_data():
            return False
        
        # Preprocess
        if not self.preprocess_data():
            return False
        
        # Train
        if model_type == 'rf':
            self.train_random_forest(tune_hyperparameters=tune)
        elif model_type == 'gb':
            self.train_gradient_boosting()
        else:
            print(f"❌ Unknown model type: {model_type}")
            return False
        
        # Evaluate
        metrics = self.evaluate()
        
        # Plot confusion matrix
        cm_file = f"models/{self.model_name}_confusion_matrix.png"
        self.plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_file)
        
        # Save
        if save:
            self.save_model()
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Test Accuracy: {metrics['test_accuracy']*100:.1f}%")
        print(f"✅ Test F1 Score: {metrics['test_f1']:.3f}")
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train ML model for IoT Sign Language Glove'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/my_glove_data',
        help='Directory containing collected CSV data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['rf', 'gb'],
        default='rf',
        help='Model type: rf (Random Forest) or gb (Gradient Boosting)'
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning (faster but less optimal)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='my_glove_model',
        help='Name for the model'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🧤 IoT Sign Language Glove - Model Training")
    print("="*60)
    print(f"📂 Data directory: {args.data}")
    print(f"💾 Output directory: {args.output}")
    print(f"🎯 Model type: {args.model_type}")
    print(f"🔧 Hyperparameter tuning: {'OFF' if args.no_tune else 'ON'}")
    
    # Create trainer
    trainer = GloveModelTrainer(args.data, model_name=args.name)
    
    # Run pipeline
    success = trainer.run_full_pipeline(
        model_type=args.model_type,
        tune=not args.no_tune,
        save=True
    )
    
    if not success:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

