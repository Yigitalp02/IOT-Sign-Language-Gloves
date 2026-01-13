"""
Train a simple Random Forest model for demo purposes.
Saves a model compatible with the current 30-feature setup.
"""

import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from load_professor_data import load_all_data
from windowed_features import create_windowed_dataset

def main():
    print("=" * 70)
    print("Training Simple Model for Demo")
    print("=" * 70)
    
    # Load all data
    print("\n[1/4] Loading dataset...")
    all_data = load_all_data()
    print(f"  Loaded {len(all_data)} recordings")
    
    # Create windowed features (1.0s windows, 0.5s stride)
    print("\n[2/4] Creating windowed features...")
    X, y, users = create_windowed_dataset(
        all_data,
        window_size=1.0,
        stride=0.5,
        sample_rate=100
    )
    print(f"  Generated {len(X)} windows")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Classes: {np.unique(y)}")
    
    # Train Random Forest on ALL data (no split, just for demo)
    print("\n[3/4] Training Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    train_acc = model.score(X, y)
    print(f"  Training accuracy: {train_acc*100:.2f}%")
    
    # Save model
    print("\n[4/4] Saving model...")
    output_dir = Path(__file__).parent.parent / "models" / "windowed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "demo_model.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nModel info:")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Classes: {len(np.unique(y))}")
    print(f"  - Estimators: {model.n_estimators}")
    print(f"  - Training accuracy: {train_acc*100:.2f}%")
    print(f"\nUse this model in predict_realtime.py by specifying:")
    print(f"  --model models/windowed/demo_model.pkl")

if __name__ == "__main__":
    main()

