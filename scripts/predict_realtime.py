"""
Real-Time Gesture Prediction
=============================

Takes a buffer of sensor samples and predicts the gesture/ASL letter.

Usage:
    python predict_realtime.py --samples "timestamp,ch0,ch1,ch2,ch3,ch4\n..."
    
Or from file:
    python predict_realtime.py --file buffer.csv
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings

# Suppress warnings and verbose output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Feature extraction (same logic as windowed_features.py)
def extract_features(samples_df, sample_rate=100):
    """
    Extract 30 features from a window of samples.
    
    Features per channel (6 × 5 = 30):
    - mean, std, min, max, mean_velocity, mean_acceleration
    """
    features = []
    
    for i in range(5):
        ch_col = f'ch{i}'
        
        if ch_col not in samples_df.columns:
            print(f"ERROR: Column {ch_col} not found!")
            return None
        
        values = samples_df[ch_col].values
        
        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Velocity (1st derivative)
        if len(values) > 1:
            velocity = np.diff(values)
            mean_velocity = np.mean(velocity) * sample_rate
        else:
            mean_velocity = 0
        
        # Acceleration (2nd derivative)
        if len(values) > 2:
            acceleration = np.diff(velocity)
            mean_acceleration = np.mean(acceleration) * (sample_rate ** 2)
        else:
            mean_acceleration = 0
        
        features.extend([
            mean_val, std_val, min_val, max_val,
            mean_velocity, mean_acceleration
        ])
    
    return np.array(features)


def predict_gesture(samples_df, model_path=None):
    """
    Predict gesture from sensor samples.
    
    Args:
        samples_df: DataFrame with columns [timestamp, ch0, ch1, ch2, ch3, ch4]
        model_path: Path to trained model (.pkl file)
    
    Returns:
        predicted_gesture: String (e.g., "Single_Thumb", "Grasp")
        confidence: Float (0-1)
    """
    
    # Find model file
    if model_path is None:
        script_dir = Path(__file__).parent
        # Try demo model first (compatible with current features)
        model_path = script_dir.parent / "models" / "windowed" / "demo_model.pkl"
        if not model_path.exists():
            # Fallback to other models
            model_path = script_dir.parent / "models" / "windowed" / "rf_windowed_model.pkl"
    
    if not Path(model_path).exists():
        return "ERROR", 0.0, f"Model not found: {model_path}"
    
    # Extract features
    features = extract_features(samples_df)
    
    if features is None:
        return "ERROR", 0.0, "Feature extraction failed"
    
    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return "ERROR", 0.0, f"Failed to load model: {e}"
    
    # Predict
    try:
        # Reshape for single prediction
        features_2d = features.reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(features_2d)[0]
        
        # Get confidence (probability)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_2d)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        return prediction, confidence, "Success"
    
    except Exception as e:
        return "ERROR", 0.0, f"Prediction failed: {e}"


def map_gesture_to_asl(gesture_name):
    """
    Map gesture name to ASL letter.
    Handles both short names and numbered prefixes (e.g., "6_Grasp" or "Grasp")
    """
    # Remove number prefix if present (e.g., "6_Grasp" -> "Grasp")
    clean_name = gesture_name.split('_', 1)[-1] if '_' in gesture_name else gesture_name
    
    # Comprehensive mapping based on professor's dataset
    asl_mapping = {
        "Grasp": "A",
        "FourFinger_Grasp": "E", 
        "Thumb2Index": "F",
        "Single_Pinkie": "I",
        "Single_Thumb": "D",
        "Single_Index": "1",
        "Single_Middle": "3",
        "Single_Ring": "4",
        "Thumb2Middle": "F",  # Similar to F
        "Thumb2Ring": "F",    # Similar to F
        "Thumb2Pinkie": "A",  # Similar to grasp/closed fist
    }
    
    # Try exact match first
    if gesture_name in asl_mapping:
        return asl_mapping[gesture_name]
    
    # Try without prefix
    if clean_name in asl_mapping:
        return asl_mapping[clean_name]
    
    # Return original if no mapping found
    return gesture_name


def main():
    parser = argparse.ArgumentParser(description="Real-time gesture prediction")
    parser.add_argument("--file", help="CSV file with sensor samples")
    parser.add_argument("--samples", help="Raw CSV string")
    parser.add_argument("--model", help="Path to model file (.pkl)")
    
    args = parser.parse_args()
    
    # Load samples
    if args.file:
        try:
            df = pd.read_csv(args.file)
        except Exception as e:
            print(f"ERROR: Failed to load file: {e}")
            sys.exit(1)
    
    elif args.samples:
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(args.samples))
        except Exception as e:
            print(f"ERROR: Failed to parse samples: {e}")
            sys.exit(1)
    
    else:
        print("ERROR: Must provide --file or --samples")
        sys.exit(1)
    
    # Validate data
    required_cols = ['ch0', 'ch1', 'ch2', 'ch3', 'ch4']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: CSV must have columns: {required_cols}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)
    
    if len(df) < 50:
        print(f"WARNING: Only {len(df)} samples (recommend 100+ for 1 second window)")
    
    # Predict
    gesture, confidence, status = predict_gesture(df, args.model)
    
    if status != "Success":
        print(f"ERROR: {status}")
        sys.exit(1)
    
    # Map to ASL letter
    asl_letter = map_gesture_to_asl(gesture)
    
    # Output result (parseable by desktop app)
    print(f"GESTURE:{gesture}")
    print(f"ASL_LETTER:{asl_letter}")
    print(f"CONFIDENCE:{confidence:.2f}")
    print(f"SAMPLES:{len(df)}")


if __name__ == "__main__":
    main()

