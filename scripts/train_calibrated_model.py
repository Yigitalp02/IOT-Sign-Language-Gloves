#!/usr/bin/env python3
"""
Train ASL model with PROPER calibration matching OUR sensor baselines.

The external dataset uses different sensor calibration (-64 to 900).
We'll transform their data to match OUR baselines (440-900 range) so the
model works with our synthetic simulator and future hardware.
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

# Letters to train on
TARGET_LETTERS = ['a', 'd', 'e', 'f', 'i', 's', 'y']

# Path to external dataset
DATASET_PATH = Path(__file__).parent.parent.parent / "ASL-Deep-Learning-Model" / "ASL-Sensor-Dataglove-Dataset"

# OUR sensor calibration (what our simulator uses)
OUR_BASELINES = {'thumb': 440, 'index': 612, 'middle': 618, 'ring': 548, 'pinky': 528}
OUR_MAXBENDS = {'thumb': 650, 'index': 900, 'middle': 900, 'ring': 850, 'pinky': 800}

# THEIR sensor calibration (from external dataset analysis)
THEIR_BASELINES = {'thumb': 20, 'index': -11, 'middle': -34, 'ring': -64, 'pinky': -67}
THEIR_MAXBENDS = {'thumb': 115, 'index': 91, 'middle': 102, 'ring': 148, 'pinky': 117}


def transform_sensor_values(their_values, finger_name):
    """
    Transform external dataset sensor values to OUR calibration range.
    
    Process:
    1. Normalize their values to 0-1 using their baseline/maxbend
    2. Denormalize to our range using our baseline/maxbend
    
    Args:
        their_values: numpy array of sensor values from external dataset
        finger_name: 'thumb', 'index', 'middle', 'ring', or 'pinky'
    
    Returns:
        Transformed values in our sensor range
    """
    their_baseline = THEIR_BASELINES[finger_name]
    their_maxbend = THEIR_MAXBENDS[finger_name]
    
    our_baseline = OUR_BASELINES[finger_name]
    our_maxbend = OUR_MAXBENDS[finger_name]
    
    # Step 1: Normalize to 0-1 using their calibration
    normalized = (their_values - their_baseline) / (their_maxbend - their_baseline)
    normalized = np.clip(normalized, 0, 1)  # Clamp to valid range
    
    # Step 2: Denormalize to our calibration
    our_values = our_baseline + normalized * (our_maxbend - our_baseline)
    
    return our_values


def extract_features_from_window(window):
    """Extract 25 features from a window of sensor samples."""
    features = []
    
    for finger_idx in range(5):
        finger_values = window[:, finger_idx]
        features.extend([
            np.mean(finger_values),
            np.std(finger_values),
            np.min(finger_values),
            np.max(finger_values),
            np.max(finger_values) - np.min(finger_values)
        ])
    
    return np.array(features)


def load_and_transform_data(letters, max_users=25):
    """
    Load external dataset and transform to OUR sensor calibration.
    
    Args:
        letters: List of lowercase letter names
        max_users: Maximum number of users to load
    
    Returns:
        X: Feature array (n_samples, 25)
        y: Label array (n_samples,)
        groups: User ID array (n_samples,)
    """
    X_all = []
    y_all = []
    groups_all = []
    
    WINDOW_SIZE = 200
    STRIDE = 200  # No overlap
    
    print(f"\nLoading and transforming {len(letters)} letters from {max_users} users...")
    print("Transforming sensor values to OUR calibration range...")
    
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    for user_id in range(1, max_users + 1):
        user_folder = DATASET_PATH / f"{user_id:03d}"
        user_samples = 0
        
        for letter in letters:
            gesture_file = user_folder / f"{letter}.csv"
            
            if not gesture_file.exists():
                continue
            
            try:
                # Load their data
                df = pd.read_csv(gesture_file)
                
                # Transform each finger's values to our calibration
                transformed_data = np.zeros((len(df), 5))
                for i, (col, finger) in enumerate(zip(['flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5'], 
                                                       finger_names)):
                    their_values = df[col].values
                    transformed_data[:, i] = transform_sensor_values(their_values, finger)
                
                # Extract windows
                for start_idx in range(0, len(transformed_data) - WINDOW_SIZE + 1, STRIDE):
                    window = transformed_data[start_idx:start_idx + WINDOW_SIZE]
                    features = extract_features_from_window(window)
                    
                    X_all.append(features)
                    y_all.append(letter.upper())
                    groups_all.append(user_id)
                    user_samples += 1
                    
            except Exception as e:
                print(f"  Error loading {gesture_file}: {e}")
        
        if user_samples > 0:
            print(f"  User {user_id:03d}: {user_samples} windows")
    
    return np.array(X_all), np.array(y_all), np.array(groups_all)


def train_with_validation(X, y, groups):
    """Train model with Leave-One-User-Out cross-validation."""
    print("\n" + "=" * 70)
    print("Leave-One-User-Out Cross-Validation")
    print("=" * 70)
    
    logo = LeaveOneGroupOut()
    fold_accuracies = []
    all_predictions = []
    all_true_labels = []
    
    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_user = groups[test_idx][0]
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            verbose=0
        )
        
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        
        print(f"Fold {fold_num:2d} - Test User {test_user:03d}: Accuracy = {accuracy:5.1%}")
    
    return fold_accuracies, all_predictions, all_true_labels


def main():
    print("=" * 70)
    print("Training ASL Model with PROPER Sensor Calibration")
    print("=" * 70)
    print("\nTransforming external dataset to match OUR sensor baselines:")
    print(f"  Our range: Thumb {OUR_BASELINES['thumb']}-{OUR_MAXBENDS['thumb']}, "
          f"Index {OUR_BASELINES['index']}-{OUR_MAXBENDS['index']}")
    print(f"  Their range: Thumb {THEIR_BASELINES['thumb']}-{THEIR_MAXBENDS['thumb']}, "
          f"Index {THEIR_BASELINES['index']}-{THEIR_MAXBENDS['index']}")
    
    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found at {DATASET_PATH}")
        return 1
    
    start_time = time.time()
    
    # Load and transform data
    X, y, groups = load_and_transform_data(TARGET_LETTERS, max_users=25)
    
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"Total samples: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Letters: {len(np.unique(y))}")
    print(f"Users: {len(np.unique(groups))}")
    print(f"Samples per letter: {len(X) // len(np.unique(y)):.0f} (avg)")
    
    # Show feature statistics to verify transformation
    print("\n" + "=" * 70)
    print("Feature Statistics (should match our simulator range)")
    print("=" * 70)
    for i, finger in enumerate(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
        mean_idx = i * 5
        finger_means = X[:, mean_idx]
        print(f"{finger:8s}: min={np.min(finger_means):6.1f}, "
              f"max={np.max(finger_means):6.1f}, "
              f"mean={np.mean(finger_means):6.1f}")
    
    # Cross-validation
    fold_accuracies, all_predictions, all_true_labels = train_with_validation(X, y, groups)
    
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print("\n" + "=" * 70)
    print("Cross-Validation Results")
    print("=" * 70)
    print(f"Mean Accuracy: {mean_accuracy:.2%} +/- {std_accuracy:.2%}")
    print(f"Range: {np.min(fold_accuracies):.2%} - {np.max(fold_accuracies):.2%}")
    
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(classification_report(all_true_labels, all_predictions))
    
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
        n_jobs=-1
    )
    
    print("Training...")
    rf_final.fit(X, y)
    
    # Save model
    output_path = Path(__file__).parent.parent / "models" / "rf_asl_calibrated.pkl"
    output_path.parent.mkdir(exist_ok=True)
    joblib.dump(rf_final, output_path)
    
    elapsed = time.time() - start_time
    
    print(f"\n[SUCCESS] Model saved: {output_path}")
    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
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
    print(f"Model: Random Forest ({rf_final.n_estimators} trees)")
    print(f"Letters: {', '.join(sorted(TARGET_LETTERS))}")
    print(f"Training samples: {len(X):,}")
    print(f"Training users: {len(np.unique(groups))}")
    print(f"Expected accuracy: {mean_accuracy:.1%}")
    print(f"Sensor calibration: MATCHED to our simulator")
    print(f"Model file: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

