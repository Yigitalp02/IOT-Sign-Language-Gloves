"""
Predict using MEAN-ONLY model (5 features).

For static poses where velocity/acceleration don't help.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


def normalize_sensors(samples_df):
    """
    Normalize sensor values from raw (400-650) to 0-1 scale.
    
    Uses typical baselines and max bends from professor's data.
    """
    # Typical baseline values (straight fingers)
    baselines = {
        'ch0': 440,  # Thumb
        'ch1': 612,  # Index
        'ch2': 618,  # Middle
        'ch3': 548,  # Ring
        'ch4': 528,  # Pinky
    }
    
    # Typical max bend values (fully bent fingers)
    maxbends = {
        'ch0': 650,  # Thumb
        'ch1': 900,  # Index
        'ch2': 900,  # Middle
        'ch3': 850,  # Ring
        'ch4': 800,  # Pinky
    }
    
    normalized_df = samples_df.copy()
    
    for i in range(5):
        ch = f'ch{i}'
        if ch not in samples_df.columns:
            return None
        
        baseline = baselines[ch]
        maxbend = maxbends[ch]
        
        # Normalize: (value - baseline) / (maxbend - baseline)
        normalized_df[f'{ch}_norm'] = ((samples_df[ch] - baseline) / (maxbend - baseline)).clip(0, 1)
    
    return normalized_df


def extract_mean_features(normalized_df):
    """Extract only mean values for each finger (5 features total)."""
    features = []
    
    for i in range(5):
        ch_col = f'ch{i}_norm'
        if ch_col not in normalized_df.columns:
            return None
        
        mean_val = np.mean(normalized_df[ch_col].values)
        features.append(mean_val)
    
    return np.array(features)


def predict_simple(samples_df, model_path=None):
    """Predict using only mean features - optimized for static poses."""
    
    # Load model
    if model_path is None:
        script_dir = Path(__file__).parent
        model_path = script_dir.parent / "models" / "windowed" / "rf_mean_only.pkl"
    
    if not Path(model_path).exists():
        return "ERROR", 0.0, f"Model not found: {model_path}", None
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return "ERROR", 0.0, f"Failed to load model: {e}", None
    
    # Normalize first!
    normalized_df = normalize_sensors(samples_df)
    
    if normalized_df is None:
        return "ERROR", 0.0, "Normalization failed", None
    
    # Extract mean features
    features = extract_mean_features(normalized_df)
    
    if features is None:
        return "ERROR", 0.0, "Feature extraction failed", None
    
    # Predict
    try:
        features_2d = features.reshape(1, -1)
        prediction = model.predict(features_2d)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_2d)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        return prediction, confidence, "Success", features
    
    except Exception as e:
        return "ERROR", 0.0, f"Prediction failed: {e}", None


def main():
    parser = argparse.ArgumentParser(description="Mean-only prediction for static poses")
    parser.add_argument("--file", required=True, help="CSV file with sensor samples")
    parser.add_argument("--model", help="Path to model file")
    
    args = parser.parse_args()
    
    # Load samples
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Validate
    required_cols = ['ch0', 'ch1', 'ch2', 'ch3', 'ch4']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Need columns {required_cols}")
        sys.exit(1)
    
    # Predict
    letter, confidence, status, normalized_features = predict_simple(df, args.model)
    
    if letter == "ERROR":
        print(f"ERROR: {status}")
        sys.exit(1)
    
    # Calculate mean values for debug
    raw_means = [df[f'ch{i}'].mean() for i in range(5)]
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    # Output
    print(f"GESTURE:{letter}")
    print(f"ASL_LETTER:{letter}")  # Already ASL letter
    print(f"CONFIDENCE:{confidence:.2f}")
    print(f"SAMPLES:{len(df)}")
    print(f"FEATURES:5_mean_only")
    
    # Debug output
    print(f"\n[DEBUG] Raw mean values:")
    for name, mean in zip(finger_names, raw_means):
        print(f"  {name:8s}: {mean:6.1f}")
    
    print(f"\n[DEBUG] Normalized features (what model sees):")
    if normalized_features is not None:
        for name, norm_val in zip(finger_names, normalized_features):
            category = "LOW" if norm_val < 0.3 else ("HIGH" if norm_val > 0.7 else "MID")
            print(f"  {name:8s}: {norm_val:5.3f}  ({category})")


if __name__ == "__main__":
    main()

