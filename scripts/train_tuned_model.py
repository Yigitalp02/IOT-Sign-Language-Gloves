"""
Train and save the Random Forest model with optimized hyperparameters.

This uses the best parameters found by tune_hyperparameters.py:
- n_estimators: 150
- max_depth: 30  
- min_samples_split: 2
- min_samples_leaf: 1

Expected LOUO accuracy: ~74.63%
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

def main():
    print("=" * 70)
    print("Training Optimized Random Forest Model")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading dataset...")
    all_data = load_all_data()
    print(f"  Loaded {len(all_data)} recordings")
    
    # Create windowed features (1.0s windows, 0.5s stride)
    print("\n[2/4] Creating windowed features...")
    X, y, users = load_and_window_data(
        window_size=1.0,
        stride=0.5
    )
    print(f"  Generated {len(X)} windows")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Users: {len(np.unique(users))}")
    
    # Create model with OPTIMIZED hyperparameters
    print("\n[3/4] Training Random Forest with OPTIMIZED parameters...")
    print("  Parameters:")
    print("    - n_estimators: 150")
    print("    - max_depth: 30")
    print("    - min_samples_split: 2")
    print("    - min_samples_leaf: 1")
    print("    - Using all CPU cores")
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Perform LOUO cross-validation to verify accuracy
    print("\n  Running Leave-One-User-Out cross-validation...")
    logo = LeaveOneGroupOut()
    scores = cross_val_score(
        model, X, y, 
        groups=users, 
        cv=logo, 
        n_jobs=-1,
        verbose=0
    )
    
    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100
    
    print(f"\n  LOUO Cross-Validation Results:")
    print(f"    Mean Accuracy: {mean_acc:.2f}%")
    print(f"    Std Deviation: ±{std_acc:.2f}%")
    print(f"    Per-fold: {[f'{s*100:.1f}%' for s in scores]}")
    
    # Train on ALL data for deployment
    print("\n  Training final model on all data...")
    model.fit(X, y)
    
    train_acc = model.score(X, y) * 100
    print(f"    Training accuracy: {train_acc:.2f}%")
    
    # Save model
    print("\n[4/4] Saving model...")
    output_dir = Path(__file__).parent.parent / "models" / "windowed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "rf_tuned_74pct.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel Summary:")
    print(f"  - LOUO Accuracy: {mean_acc:.2f}% (±{std_acc:.2f}%)")
    print(f"  - Training Accuracy: {train_acc:.2f}%")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Classes: {len(np.unique(y))}")
    print(f"  - Estimators: {model.n_estimators}")
    print(f"  - Max Depth: {model.max_depth}")
    print(f"\nTo use this model in the app, update main.rs to:")
    print(f'  model_path = "models/windowed/rf_tuned_74pct.pkl"')

if __name__ == "__main__":
    main()

