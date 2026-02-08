#!/usr/bin/env python3
"""
Train a comprehensive ASL alphabet model using the external dataset
Supports all 26 letters (A-Z) with proper validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sys
import time

# ALL 26 ASL letters
ALL_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Path to the external dataset
DATASET_PATH = Path(__file__).parent.parent.parent / "ASL-Deep-Learning-Model" / "ASL-Sensor-Dataglove-Dataset"

def extract_features_from_window(window):
    """
    Extract statistical features from a window of sensor samples
    Returns: 25 features (5 stats × 5 fingers)
    """
    features = []
    
    for finger_idx in range(5):
        finger_values = window[:, finger_idx]
        features.extend([
            np.mean(finger_values),    # Mean
            np.std(finger_values),     # Standard deviation
            np.min(finger_values),     # Minimum
            np.max(finger_values),     # Maximum
            np.max(finger_values) - np.min(finger_values)  # Range
        ])
    
    return np.array(features)

def load_all_data(letters, max_users=25):
    """
    Load data for specified letters from all users
    
    Args:
        letters: List of lowercase letter names (e.g., ['a', 'b', 'c'])
        max_users: Maximum number of users to load (1-25)
    
    Returns:
        X: Feature array (n_samples, 25)
        y: Label array (n_samples,)
        groups: User ID array (n_samples,) for cross-validation
    """
    X_all = []
    y_all = []
    groups_all = []
    
    WINDOW_SIZE = 200  # 2 seconds at 100Hz
    STRIDE = 200       # No overlap to prevent data leakage
    
    print(f"\nLoading {len(letters)} letters from {max_users} users...")
    
    for user_id in range(1, max_users + 1):
        user_folder = DATASET_PATH / f"{user_id:03d}"
        user_samples = 0
        
        for letter in letters:
            gesture_file = user_folder / f"{letter}.csv"
            
            if not gesture_file.exists():
                print(f"  Warning: {gesture_file.name} not found for user {user_id:03d}")
                continue
            
            try:
                df = pd.read_csv(gesture_file)
                flex_data = df[['flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5']].values
                
                # Extract non-overlapping windows
                for start_idx in range(0, len(flex_data) - WINDOW_SIZE + 1, STRIDE):
                    window = flex_data[start_idx:start_idx + WINDOW_SIZE]
                    features = extract_features_from_window(window)
                    
                    X_all.append(features)
                    y_all.append(letter.upper())
                    groups_all.append(user_id)
                    user_samples += 1
                    
            except Exception as e:
                print(f"  Error loading {gesture_file}: {e}")
        
        if user_samples > 0:
            print(f"  User {user_id:03d}: {user_samples} samples")
    
    return np.array(X_all), np.array(y_all), np.array(groups_all)

def train_with_validation(X, y, groups, n_estimators=200):
    """
    Train model with Leave-One-User-Out cross-validation
    """
    print("\n" + "=" * 70)
    print("Leave-One-User-Out Cross-Validation")
    print("=" * 70)
    
    logo = LeaveOneGroupOut()
    fold_accuracies = []
    all_predictions = []
    all_true_labels = []
    
    n_folds = len(np.unique(groups))
    print(f"Training {n_folds} folds (each testing on 1 held-out user)...\n")
    
    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_user = groups[test_idx][0]
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,              # Slightly deeper for more classes
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',   # Handle class imbalance
            n_jobs=-1,
            verbose=0
        )
        
        rf.fit(X_train, y_train)
        
        # Evaluate on held-out user
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        
        print(f"Fold {fold_num:2d} - Test User {test_user:03d}: "
              f"Accuracy = {accuracy:5.1%}  "
              f"({len(test_idx):3d} samples)")
    
    return fold_accuracies, all_predictions, all_true_labels

def main():
    print("=" * 70)
    print("Training Full ASL Alphabet Recognition Model")
    print("=" * 70)
    print("\nUsing: ASL-Deep-Learning-Model Dataset")
    print("Letters: All 26 (A-Z)")
    print("Users: 25")
    print("Validation: Leave-One-User-Out (LOUO)")
    
    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found at {DATASET_PATH}")
        print("\nPlease ensure you have cloned the repository:")
        print("  cd C:\\Users\\Yigit\\Desktop\\iot-sign-language-desktop")
        print("  git clone https://github.com/Subhrasameerdash/ASL-Deep-Learning-Model")
        return 1
    
    start_time = time.time()
    
    # Load data for ALL 26 letters from ALL 25 users
    print("\n" + "=" * 70)
    print("Loading Dataset")
    print("=" * 70)
    
    X, y, groups = load_all_data(ALL_LETTERS, max_users=25)
    
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"Total samples: {len(X):,}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Letters: {len(np.unique(y))}")
    print(f"Users: {len(np.unique(groups))}")
    print(f"Samples per letter: {len(X) // len(np.unique(y)):.0f} (avg)")
    print(f"Samples per user: {len(X) // len(np.unique(groups)):.0f} (avg)")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for letter, count in sorted(zip(unique, counts)):
        print(f"  {letter}: {count:4d} samples")
    
    # Cross-validation
    fold_accuracies, all_predictions, all_true_labels = train_with_validation(X, y, groups)
    
    # Results
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print("\n" + "=" * 70)
    print("Cross-Validation Results")
    print("=" * 70)
    print(f"Mean Accuracy: {mean_accuracy:.2%} ± {std_accuracy:.2%}")
    print(f"Min Accuracy:  {np.min(fold_accuracies):.2%}")
    print(f"Max Accuracy:  {np.max(fold_accuracies):.2%}")
    
    print("\n" + "=" * 70)
    print("Overall Classification Report")
    print("=" * 70)
    print(classification_report(all_true_labels, all_predictions))
    
    print("\n" + "=" * 70)
    print("Confusion Matrix")
    print("=" * 70)
    cm = confusion_matrix(all_true_labels, all_predictions)
    print(cm)
    
    # Identify most confused pairs
    print("\n" + "=" * 70)
    print("Top 10 Most Confused Letter Pairs")
    print("=" * 70)
    letters = sorted(np.unique(y))
    confusions = []
    for i in range(len(letters)):
        for j in range(len(letters)):
            if i != j and cm[i][j] > 0:
                confusions.append((letters[i], letters[j], cm[i][j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    for true_letter, pred_letter, count in confusions[:10]:
        print(f"  {true_letter} -> {pred_letter}: {count} times")
    
    # Train final model on ALL data
    print("\n" + "=" * 70)
    print("Training Final Model on All Users")
    print("=" * 70)
    
    rf_final = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=1
    )
    
    print("Training...")
    rf_final.fit(X, y)
    
    # Save model
    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "rf_asl_full_alphabet.pkl"
    joblib.dump(rf_final, model_path)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n[SUCCESS] Model saved to: {model_path}")
    print(f"Total training time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("Top 15 Most Important Features")
    print("=" * 70)
    feature_names = []
    for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
        for stat in ['mean', 'std', 'min', 'max', 'range']:
            feature_names.append(f"{finger}_{stat}")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_final.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.head(15).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Model: Random Forest with {rf_final.n_estimators} trees")
    print(f"Letters recognized: 26 (A-Z)")
    print(f"Training samples: {len(X):,}")
    print(f"Training users: {len(np.unique(groups))}")
    print(f"Expected accuracy on new users: {mean_accuracy:.1%}")
    print(f"Model file: {model_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

