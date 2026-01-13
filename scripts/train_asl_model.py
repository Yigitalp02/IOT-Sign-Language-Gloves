"""
Train ASL-Specific Model
=========================

Relabels the professor's dataset with ASL letters and trains
a model specifically for A, F, E, I, D, S, T recognition.

This should achieve higher accuracy than the general gesture model
because it's tailored to the specific ASL letters we care about.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from load_professor_data import load_all_data
from windowed_features import load_and_window_data


# Mapping from professor's gestures to ASL letters
GESTURE_TO_ASL = {
    'Single_Thumb': 'D',
    'Single_Index': '1',  # Not an ASL letter we care about
    'Single_Middle': '3',  # Not an ASL letter we care about
    'Single_Ring': '4',  # Not an ASL letter we care about
    'Single_Pinkie': 'I',
    'Grasp': 'A',  # A, S, T are all the same
    'FourFinger_Grasp': 'E',
    'Thumb2Index': 'F',
    'Thumb2Middle': 'F',  # Similar to F
    'Thumb2Ring': 'F',  # Similar to F
    'Thumb2Pinkie': 'A',  # Similar to closed fist
}

# Only keep the ASL letters we care about
TARGET_ASL_LETTERS = ['A', 'D', 'E', 'F', 'I']


def relabel_to_asl(gestures):
    """Convert gesture names to ASL letters."""
    asl_labels = []
    
    for gesture in gestures:
        # Gesture names are clean: "Grasp", "Single_Thumb", etc.
        if gesture in GESTURE_TO_ASL:
            asl_labels.append(GESTURE_TO_ASL[gesture])
        else:
            asl_labels.append('UNKNOWN')
    
    return np.array(asl_labels)


def main():
    print("=" * 70)
    print("Training ASL-Specific Model")
    print("=" * 70)
    print("\nTarget ASL Letters: A, D, E, F, I")
    print("  A = Closed fist (Grasp, Thumb2Pinkie)")
    print("  D = Thumb pointing (Single_Thumb)")
    print("  E = Four fingers tucked (FourFinger_Grasp)")
    print("  F = OK sign (Thumb2Index, Thumb2Middle, Thumb2Ring)")
    print("  I = Pinky extended (Single_Pinkie)")
    
    # Load data
    print("\n[1/5] Loading and creating windowed features...")
    X, y_gestures, users = load_and_window_data(window_size=1.0, stride=0.5)
    
    print(f"  Total windows: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Original gestures: {len(np.unique(y_gestures))}")
    
    # Relabel to ASL
    print("\n[2/5] Relabeling to ASL letters...")
    y_asl = relabel_to_asl(y_gestures)
    
    # Filter to only target ASL letters
    mask = np.isin(y_asl, TARGET_ASL_LETTERS)
    X_filtered = X[mask]
    y_filtered = y_asl[mask]
    users_filtered = users[mask]
    
    print(f"  Filtered to {len(X_filtered)} windows")
    print(f"  ASL classes: {sorted(np.unique(y_filtered))}")
    
    # Show class distribution
    print("\n  Class distribution:")
    for letter in sorted(np.unique(y_filtered)):
        count = np.sum(y_filtered == letter)
        percentage = count / len(y_filtered) * 100
        print(f"    {letter}: {count:5d} windows ({percentage:5.1f}%)")
    
    # Train model
    print("\n[3/5] Training ASL-specific Random Forest...")
    print("  Using optimized hyperparameters:")
    print("    - n_estimators: 150")
    print("    - max_depth: 30")
    print("    - min_samples_split: 2")
    print("    - min_samples_leaf: 1")
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',  # Fix class imbalance!
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # LOUO Cross-validation
    print("\n[4/5] Running Leave-One-User-Out cross-validation...")
    logo = LeaveOneGroupOut()
    scores = cross_val_score(
        model, X_filtered, y_filtered,
        groups=users_filtered,
        cv=logo,
        n_jobs=-1,
        verbose=0
    )
    
    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100
    
    print(f"\n  LOUO Cross-Validation Results:")
    print(f"    Mean Accuracy: {mean_acc:.2f}%")
    print(f"    Std Deviation: ±{std_acc:.2f}%")
    print(f"    Min Accuracy: {np.min(scores)*100:.2f}%")
    print(f"    Max Accuracy: {np.max(scores)*100:.2f}%")
    print(f"\n    Per-user accuracy:")
    for i, score in enumerate(scores, 1):
        print(f"      User {i:2d}: {score*100:5.1f}%")
    
    # Train final model on all data
    print("\n  Training final model on all data...")
    model.fit(X_filtered, y_filtered)
    
    train_acc = model.score(X_filtered, y_filtered) * 100
    print(f"    Training accuracy: {train_acc:.2f}%")
    
    # Feature importance
    print("\n  Top 10 most important features:")
    feature_names = []
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    stats = ['Mean', 'Std', 'Min', 'Max', 'Velocity', 'Acceleration']
    for finger in fingers:
        for stat in stats:
            feature_names.append(f"{finger}_{stat}")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"    {i+1:2d}. {feature_names[idx]:20s} {importances[idx]:.4f}")
    
    # Save model
    print("\n[5/5] Saving ASL-specific model...")
    output_dir = Path(__file__).parent.parent / "models" / "windowed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "rf_asl_model.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved to: {model_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nASL Model Summary:")
    print(f"  - Target Classes: {len(np.unique(y_filtered))} ASL letters")
    print(f"  - Classes: {', '.join(sorted(np.unique(y_filtered)))}")
    print(f"  - LOUO Accuracy: {mean_acc:.2f}% (±{std_acc:.2f}%)")
    print(f"  - Training Accuracy: {train_acc:.2f}%")
    print(f"  - Features: {X_filtered.shape[1]}")
    print(f"  - Training Samples: {len(X_filtered)} windows")
    print(f"  - Users: {len(np.unique(users_filtered))}")
    print(f"\nComparison with General Model:")
    print(f"  - General model: 73.18% (11 classes)")
    print(f"  - ASL model: {mean_acc:.2f}% (5 classes)")
    print(f"  - Improvement: {mean_acc - 73.18:+.2f}%")
    print(f"\nTo use this model in the app, update main.rs:")
    print(f'  model_path = "models/windowed/rf_asl_model.pkl"')


if __name__ == "__main__":
    main()

