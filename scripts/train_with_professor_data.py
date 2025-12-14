"""
Train ML models using the professor's real glove data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import os

def load_data(filepath="data/professor_data_combined.csv"):
    """Load the combined professor data"""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        print("Please run load_professor_data.py first to generate the combined dataset.")
        return None
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} samples")
    return df

def extract_gesture_label(full_label):
    """Extract clean gesture name from full label"""
    # e.g., "15_46_23_TestSubject01_1_Single_Thumb" -> "Single_Thumb"
    parts = full_label.split('_')
    
    # Find "TestSubject" in parts
    test_subject_idx = None
    for i, part in enumerate(parts):
        if part.startswith('TestSubject'):
            test_subject_idx = i
            break
    
    if test_subject_idx is None:
        return full_label
    
    # Gesture name is after TestSubjectXX_number_
    # e.g., ["15", "46", "23", "TestSubject01", "1", "Single", "Thumb"]
    # We want parts after index test_subject_idx + 1 (skip the number)
    if test_subject_idx + 2 < len(parts):
        gesture = '_'.join(parts[test_subject_idx + 2:])
        return gesture
    
    return full_label

def prepare_data(df):
    """Prepare features and labels for ML"""
    print("\nPreparing data for training...")
    
    # Use normalized values as features
    feature_cols = ['ch0_norm', 'ch1_norm', 'ch2_norm', 'ch3_norm', 'ch4_norm']
    
    # Check if normalized columns exist
    if not all(col in df.columns for col in feature_cols):
        print("Warning: Normalized columns not found, using raw values...")
        feature_cols = ['ch0_raw', 'ch1_raw', 'ch2_raw', 'ch3_raw', 'ch4_raw']
    
    X = df[feature_cols].values
    
    # Clean up gesture labels
    df['gesture'] = df['class_label'].apply(extract_gesture_label)
    y = df['gesture'].values
    
    # Get user IDs for cross-validation
    users = df['user_id'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Unique gestures: {len(np.unique(y))}")
    print(f"Gestures: {sorted(np.unique(y))}")
    print(f"Users: {sorted(np.unique(users))}")
    
    return X, y, users

def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train a baseline Random Forest model"""
    print("\n" + "="*60)
    print("TRAINING BASELINE MODEL (Random Forest)")
    print("="*60)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf

def train_mlp_model(X_train, y_train, X_test, y_test):
    """Train an MLP (Multi-Layer Perceptron) neural network"""
    print("\n" + "="*60)
    print("TRAINING MLP MODEL")
    print("="*60)
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        verbose=True
    )
    
    mlp.fit(X_train_scaled, y_train_encoded)
    
    # Evaluate
    y_pred_encoded = mlp.predict(X_test_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return mlp, scaler, label_encoder

def leave_one_user_out_cv(X, y, users):
    """
    Leave-One-User-Out Cross-Validation
    This is the gold standard for evaluating gesture recognition systems
    because it tests if the model can generalize to NEW users
    """
    print("\n" + "="*60)
    print("LEAVE-ONE-USER-OUT CROSS-VALIDATION")
    print("="*60)
    print("This tests if the model can work for users it has never seen before!")
    print("(This is what your professor will care about!)")
    print("="*60)
    
    logo = LeaveOneGroupOut()
    unique_users = np.unique(users)
    
    accuracies = []
    
    for train_idx, test_idx in logo.split(X, y, users):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        test_user = users[test_idx[0]]
        
        # Train Random Forest (faster than MLP for CV)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"Test user: {test_user:15s} | Accuracy: {acc*100:5.2f}%")
    
    print("\n" + "="*60)
    print(f"AVERAGE LOUO ACCURACY: {np.mean(accuracies)*100:.2f}% (+/- {np.std(accuracies)*100:.2f}%)")
    print("="*60)
    print("\nWhat this means:")
    if np.mean(accuracies) > 0.85:
        print("[EXCELLENT] The model generalizes very well to new users!")
    elif np.mean(accuracies) > 0.70:
        print("[GOOD] The model works reasonably well for new users.")
    elif np.mean(accuracies) > 0.50:
        print("[FAIR] The model has some generalization, but needs improvement.")
    else:
        print("[POOR] The model is overfitting to training users.")
    
    return accuracies

def main():
    """Main training pipeline"""
    # Change to iot-sign-glove root directory for relative paths
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))  # Go to iot-sign-glove root
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare features and labels
    X, y, users = prepare_data(df)
    
    # Simple train-test split (80/20)
    print("\n" + "="*60)
    print("SIMPLE TRAIN-TEST SPLIT (80/20)")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    
    # Train baseline Random Forest
    rf_model = train_baseline_model(X_train, y_train, X_test, y_test)
    
    # Train MLP
    mlp_model, scaler, label_encoder = train_mlp_model(X_train, y_train, X_test, y_test)
    
    # Leave-One-User-Out CV (the real test!)
    louo_accuracies = leave_one_user_out_cv(X, y, users)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Implement windowing for temporal features")
    print("2. Train LSTM/TCN for sequence modeling")
    print("3. Export model to TFLite for mobile deployment")
    print("4. Test with your real glove when it arrives!")
    
    # Save models
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(mlp_model, "models/mlp_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    print("\n[OK] Models saved to models/ directory")

if __name__ == "__main__":
    main()

