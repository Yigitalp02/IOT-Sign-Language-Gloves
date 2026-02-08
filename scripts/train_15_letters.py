#!/usr/bin/env python3
"""
Train ASL model with 15 DISTINGUISHABLE letters (best we can do with flex sensors).

Letters included:
- Easy (hand shape): A, B, C, D, E, F, O, S, T
- Pinky/Thumb extended: I, Y  
- Special: K, V, W, X

Letters EXCLUDED (require orientation/position):
- G, H, J, L, M, N, P, Q, R, U, Z
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys
import time

# 15 distinguishable letters
TARGET_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'i', 'k', 'o', 's', 't', 'v', 'w', 'x', 'y']

DATASET_PATH = Path(__file__).parent.parent.parent / "ASL-Deep-Learning-Model" / "ASL-Sensor-Dataglove-Dataset"

# Calibration constants
OUR_BASELINES = {'thumb': 440, 'index': 612, 'middle': 618, 'ring': 548, 'pinky': 528}
OUR_MAXBENDS = {'thumb': 650, 'index': 900, 'middle': 900, 'ring': 850, 'pinky': 800}
THEIR_BASELINES = {'thumb': 20, 'index': -11, 'middle': -34, 'ring': -64, 'pinky': -67}
THEIR_MAXBENDS = {'thumb': 115, 'index': 91, 'middle': 102, 'ring': 148, 'pinky': 117}


def transform_sensor_values(their_values, finger_name):
    """Transform external dataset sensor values to OUR calibration range."""
    their_baseline = THEIR_BASELINES[finger_name]
    their_maxbend = THEIR_MAXBENDS[finger_name]
    
    our_baseline = OUR_BASELINES[finger_name]
    our_maxbend = OUR_MAXBENDS[finger_name]
    
    normalized = (their_values - their_baseline) / (their_maxbend - their_baseline)
    normalized = np.clip(normalized, 0, 1)
    
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
    """Load external dataset and transform to OUR sensor calibration."""
    X_all = []
    y_all = []
    groups_all = []
    
    WINDOW_SIZE = 200
    STRIDE = 200
    
    print(f"\nLoading {len(letters)} letters from {max_users} users...")
    
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    for user_id in range(1, max_users + 1):
        user_folder = DATASET_PATH / f"{user_id:03d}"
        user_samples = 0
        
        for letter in letters:
            gesture_file = user_folder / f"{letter}.csv"
            
            if not gesture_file.exists():
                continue
            
            try:
                df = pd.read_csv(gesture_file)
                
                transformed_data = np.zeros((len(df), 5))
                for i, (col, finger) in enumerate(zip(['flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5'], 
                                                       finger_names)):
                    their_values = df[col].values
                    transformed_data[:, i] = transform_sensor_values(their_values, finger)
                
                for start_idx in range(0, len(transformed_data) - WINDOW_SIZE + 1, STRIDE):
                    window = transformed_data[start_idx:start_idx + WINDOW_SIZE]
                    features = extract_features_from_window(window)
                    
                    X_all.append(features)
                    y_all.append(letter.upper())
                    groups_all.append(user_id)
                    user_samples += 1
                    
            except Exception as e:
                pass
        
        if user_samples > 0 and user_id % 5 == 0:
            print(f"  Processed {user_id} users...")
    
    return np.array(X_all), np.array(y_all), np.array(groups_all)


def main():
    print("=" * 70)
    print("Training 15-Letter ASL Model (Flex Sensors Only)")
    print("=" * 70)
    print(f"\nLetters: {', '.join([L.upper() for L in TARGET_LETTERS])}")
    
    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found at {DATASET_PATH}")
        return 1
    
    start_time = time.time()
    
    # Load data
    X, y, groups = load_and_transform_data(TARGET_LETTERS, max_users=25)
    
    print(f"\nDataset: {len(X):,} samples, {len(np.unique(y))} classes, {len(np.unique(groups))} users")
    
    # Train with LOUO validation
    print("\n" + "=" * 70)
    print("Training with Leave-One-User-Out Cross-Validation")
    print("=" * 70)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    logo = LeaveOneGroupOut()
    fold_accuracies = []
    all_predictions = []
    all_true_labels = []
    
    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        
        if fold_num % 5 == 0:
            print(f"  Validated {fold_num}/25 users...")
    
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Mean Accuracy: {mean_accuracy:.2%} +/- {std_accuracy:.2%}")
    print(f"Range: {np.min(fold_accuracies):.2%} - {np.max(fold_accuracies):.2%}")
    
    print("\n" + "=" * 70)
    print("Per-Letter Performance")
    print("=" * 70)
    print(classification_report(all_true_labels, all_predictions))
    
    # Train final model
    print("\n" + "=" * 70)
    print("Training Final Model on All Users")
    print("=" * 70)
    
    rf.fit(X, y)
    
    # Save
    output_path = Path(__file__).parent.parent / "models" / "rf_asl_15letters.pkl"
    output_path.parent.mkdir(exist_ok=True)
    joblib.dump(rf, output_path)
    
    elapsed = time.time() - start_time
    
    print(f"\n[SUCCESS] Model saved: {output_path}")
    print(f"Accuracy: {mean_accuracy:.2%}")
    print(f"Letters: {len(TARGET_LETTERS)}")
    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

