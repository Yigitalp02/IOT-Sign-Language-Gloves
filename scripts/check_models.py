"""Check feature dimensions of trained models."""

import joblib
import glob
from pathlib import Path

models_dir = Path(__file__).parent.parent / "models" / "windowed"
models = list(models_dir.glob("*.pkl"))

print("=" * 70)
print("Trained Models Feature Dimensions")
print("=" * 70)

for model_path in sorted(models):
    if 'scaler' in model_path.name or 'label' in model_path.name:
        continue
    
    try:
        model = joblib.load(model_path)
        
        # Try to get feature count
        if hasattr(model, 'n_features_in_'):
            features = model.n_features_in_
        elif hasattr(model, 'n_features_'):
            features = model.n_features_
        else:
            features = "Unknown"
        
        # Get model type
        model_type = type(model).__name__
        
        print(f"\n{model_path.name}")
        print(f"  Type: {model_type}")
        print(f"  Features: {features}")
        
    except Exception as e:
        print(f"\n{model_path.name}")
        print(f"  Error: {e}")

print("\n" + "=" * 70)

