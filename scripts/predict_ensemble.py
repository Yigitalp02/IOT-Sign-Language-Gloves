"""
Ensemble Prediction with Sliding Window Voting
===============================================

Makes multiple predictions from overlapping windows and votes for
more stable and confident results.

Usage:
    python predict_ensemble.py --file buffer.csv
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
from collections import Counter

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Feature extraction (same as predict_realtime.py)
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


def map_gesture_to_asl(gesture_name):
    """Map gesture name to ASL letter."""
    # Remove number prefix if present
    clean_name = gesture_name.split('_', 1)[-1] if '_' in gesture_name else gesture_name
    
    asl_mapping = {
        "Grasp": "A",
        "FourFinger_Grasp": "E", 
        "Thumb2Index": "F",
        "Single_Pinkie": "I",
        "Single_Thumb": "D",
        "Single_Index": "1",
        "Single_Middle": "3",
        "Single_Ring": "4",
        "Thumb2Middle": "F",
        "Thumb2Ring": "F",
        "Thumb2Pinkie": "A",
    }
    
    if gesture_name in asl_mapping:
        return asl_mapping[gesture_name]
    if clean_name in asl_mapping:
        return asl_mapping[clean_name]
    
    return gesture_name


def predict_ensemble(samples_df, model_path=None):
    """
    Predict using multiple sliding windows with voting.
    
    Strategy:
    1. Create multiple 100-sample windows with 25-sample stride
    2. Predict each window
    3. Vote for most common prediction
    4. Average confidence scores
    5. Boost confidence based on agreement ratio
    """
    
    # Load model
    if model_path is None:
        script_dir = Path(__file__).parent
        model_path = script_dir.parent / "models" / "windowed" / "rf_asl_model.pkl"
    
    if not Path(model_path).exists():
        return "ERROR", 0.0, "Model not found", []
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return "ERROR", 0.0, f"Failed to load model: {e}", []
    
    # Sliding window parameters
    window_size = 100  # 1 second at 100Hz
    stride = 25  # 0.25 second stride (75% overlap)
    
    predictions_list = []
    
    # Extract predictions from multiple windows
    for start in range(0, len(samples_df) - window_size + 1, stride):
        window = samples_df.iloc[start:start + window_size]
        
        if len(window) < window_size:
            continue
        
        # Extract features
        features = extract_features(window)
        if features is None:
            continue
        
        try:
            # Predict
            features_2d = features.reshape(1, -1)
            prediction = model.predict(features_2d)[0]
            
            # Get confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_2d)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 1.0
            
            predictions_list.append({
                'gesture': prediction,
                'confidence': confidence,
                'window_start': start
            })
        
        except Exception as e:
            continue
    
    if not predictions_list:
        return "ERROR", 0.0, "No valid predictions", []
    
    # Vote: count predictions for each gesture
    gesture_votes = Counter([p['gesture'] for p in predictions_list])
    
    # Get winner
    winner_gesture, winner_count = gesture_votes.most_common(1)[0]
    total_predictions = len(predictions_list)
    agreement_ratio = winner_count / total_predictions
    
    # Calculate average confidence for the winning gesture
    winner_confidences = [
        p['confidence'] for p in predictions_list 
        if p['gesture'] == winner_gesture
    ]
    avg_confidence = np.mean(winner_confidences)
    
    # Boost confidence based on agreement
    # High agreement (90%+) = boost confidence
    # Low agreement (30%-)  = reduce confidence
    confidence_boost = 0.7 + 0.3 * agreement_ratio
    final_confidence = avg_confidence * confidence_boost
    final_confidence = min(final_confidence, 1.0)  # Cap at 100%
    
    # Create voting summary
    voting_summary = f"{winner_count}/{total_predictions} windows ({agreement_ratio*100:.0f}% agree)"
    
    return winner_gesture, final_confidence, voting_summary, predictions_list


def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction with voting")
    parser.add_argument("--file", required=True, help="CSV file with sensor samples")
    parser.add_argument("--model", help="Path to model file (.pkl)")
    
    args = parser.parse_args()
    
    # Load samples
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"ERROR: Failed to load file: {e}")
        sys.exit(1)
    
    # Validate columns
    required_cols = ['ch0', 'ch1', 'ch2', 'ch3', 'ch4']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: CSV must have columns: {required_cols}")
        sys.exit(1)
    
    # Run ensemble prediction
    gesture, confidence, voting_summary, predictions = predict_ensemble(df, args.model)
    
    if gesture == "ERROR":
        print(f"ERROR: {voting_summary}")
        sys.exit(1)
    
    # Map to ASL letter
    asl_letter = map_gesture_to_asl(gesture)
    
    # Output for app parsing
    print(f"GESTURE:{gesture}")
    print(f"ASL_LETTER:{asl_letter}")
    print(f"CONFIDENCE:{confidence:.2f}")
    print(f"SAMPLES:{len(df)}")
    print(f"VOTING:{voting_summary}")
    print(f"WINDOWS:{len(predictions)}")
    
    # Debug output (optional)
    if len(predictions) > 1:
        print(f"\n[DEBUG] Individual window predictions:")
        for i, pred in enumerate(predictions[:10]):  # Show first 10
            print(f"  Window {i+1}: {pred['gesture']} ({pred['confidence']:.2f})")
        if len(predictions) > 10:
            print(f"  ... and {len(predictions) - 10} more")


if __name__ == "__main__":
    main()

