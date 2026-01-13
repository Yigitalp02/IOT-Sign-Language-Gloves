"""
Train ultra-simple model using ONLY MEAN VALUES.

For static poses, mean values might be enough to distinguish gestures.
This creates a 5-feature model (one mean per finger).
"""

import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
import sys

sys.path.insert(0, str(Path(__file__).parent))
from windowed_features import load_and_window_data


# Gesture to ASL mapping
GESTURE_TO_ASL = {
    'Single_Thumb': 'D',
    'Single_Pinkie': 'I',
    'Grasp': 'A',
    'FourFinger_Grasp': 'E',
    'Thumb2Index': 'F',
    'Thumb2Middle': 'F',
    'Thumb2Ring': 'F',
    'Thumb2Pinkie': 'A',
}

TARGET_ASL = ['A', 'D', 'E', 'F', 'I']


def extract_mean_features(X_original):
    """
    Extract only MEAN values from the 30-feature vectors.
    
    Original 30 features per window (6 per finger):
    [mean, std, min, max, velocity, acceleration] × 5 fingers
    
    We want only:
    [mean] × 5 fingers = 5 features
    """
    # Take every 6th element starting from 0 (the mean values)
    indices = [i*6 for i in range(5)]  # [0, 6, 12, 18, 24]
    X_mean_only = X_original[:, indices]
    
    return X_mean_only


def relabel_to_asl(gestures):
    """Convert gesture names to ASL letters."""
    return np.array([GESTURE_TO_ASL.get(g, 'UNKNOWN') for g in gestures])


def main():
    print("=" * 70)
    print("Training MEAN-ONLY Model (5 features)")
    print("=" * 70)
    print("\nUsing ONLY finger mean values for static pose recognition")
    
    # Load windowed data (30 features)
    print("\n[1/4] Loading windowed data...")
    X_30, y_gestures, users = load_and_window_data(window_size=1.0, stride=0.5)
    
    # Extract only mean features
    print("\n[2/4] Extracting mean-only features...")
    X = extract_mean_features(X_30)
    print(f"  Reduced from {X_30.shape[1]} to {X.shape[1]} features")
    
    # Relabel to ASL
    print("\n  Relabeling to ASL...")
    y_asl = relabel_to_asl(y_gestures)
    
    # Filter to target ASL letters
    mask = np.isin(y_asl, TARGET_ASL)
    X = X[mask]
    y = y_asl[mask]
    users = users[mask]
    
    print(f"  Filtered to {len(X)} windows")
    print(f"  Classes: {sorted(np.unique(y))}")
    
    # Show distribution
    print("\n  Class distribution:")
    for letter in sorted(np.unique(y)):
        count = np.sum(y == letter)
        pct = count / len(y) * 100
        print(f"    {letter}: {count:5d} ({pct:5.1f}%)")
    
    # Train
    print("\n[3/4] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,  # Simpler tree for simpler features
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # LOUO CV
    print("\n  Leave-One-User-Out cross-validation...")
    logo = LeaveOneGroupOut()
    scores = cross_val_score(model, X, y, groups=users, cv=logo, n_jobs=-1)
    
    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100
    
    print(f"\n  LOUO Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
    print(f"  Per-user: {[f'{s*100:.1f}%' for s in scores]}")
    
    # Train final
    print("\n  Training final model...")
    model.fit(X, y)
    train_acc = model.score(X, y) * 100
    print(f"  Training accuracy: {train_acc:.2f}%")
    
    # Feature importance
    print("\n  Feature importance (finger means):")
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    importances = model.feature_importances_
    for i, finger in enumerate(fingers):
        print(f"    {finger:8s}: {importances[i]:.4f}")
    
    # Save
    print("\n[4/4] Saving model...")
    output_dir = Path(__file__).parent.parent / "models" / "windowed"
    model_path = output_dir / "rf_mean_only.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nModel Summary:")
    print(f"  - Features: {X.shape[1]} (only finger means)")
    print(f"  - LOUO Accuracy: {mean_acc:.2f}%")
    print(f"  - Classes: {len(np.unique(y))}")
    print(f"  - Best for: Static poses / frozen snapshots")


if __name__ == "__main__":
    main()

