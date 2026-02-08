"""
ML-based ASL prediction using trained Random Forest model.

Uses the rf_asl_external_proper.pkl model (88.6% accuracy on 7 letters: A, D, E, F, I, S, Y)
Trained with Leave-One-User-Out validation on external ASL dataset.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def extract_features_from_samples(samples_df):
    """
    Extract 25 features from sensor samples (same as training).
    
    Features: mean, std, min, max, range for each of 5 fingers
    
    Args:
        samples_df: DataFrame with columns ch0-ch4 (thumb, index, middle, ring, pinky)
    
    Returns:
        numpy array of shape (25,)
    """
    features = []
    
    for finger_col in ['ch0', 'ch1', 'ch2', 'ch3', 'ch4']:
        if finger_col not in samples_df.columns:
            raise ValueError(f"Missing column: {finger_col}")
        
        finger_values = samples_df[finger_col].values
        
        # Calculate statistics
        features.extend([
            np.mean(finger_values),    # Mean
            np.std(finger_values),     # Standard deviation
            np.min(finger_values),     # Minimum
            np.max(finger_values),     # Maximum
            np.max(finger_values) - np.min(finger_values)  # Range
        ])
    
    return np.array(features)


def get_pattern_description(letter, features):
    """
    Generate a human-readable description of the pattern.
    
    Args:
        letter: Predicted letter
        features: 25-element feature array
    
    Returns:
        Description string
    """
    # Extract means (every 5th element starting at 0)
    means = features[::5]  # [thumb_mean, index_mean, middle_mean, ring_mean, pinky_mean]
    
    # Simple normalization to 0-1 for description
    baselines = np.array([440, 612, 618, 548, 528])
    maxbends = np.array([650, 900, 900, 850, 800])
    normalized = (means - baselines) / (maxbends - baselines)
    normalized = np.clip(normalized, 0, 1)
    
    # Count bent fingers (>0.5 = bent)
    bent_count = np.sum(normalized > 0.5)
    
    descriptions = {
        'A': f"Closed fist ({bent_count}/5 fingers bent)",
        'B': f"Open hand ({bent_count}/5 fingers bent)",
        'C': f"Curved hand ({bent_count}/5 fingers bent)",
        'D': f"Index pointing ({bent_count}/5 fingers bent)",
        'E': f"Partial fist ({bent_count}/5 fingers bent)",
        'F': f"OK sign ({bent_count}/5 fingers bent)",
        'I': f"Pinky extended ({bent_count}/5 fingers bent)",
        'K': f"Peace sign variant ({bent_count}/5 fingers bent)",
        'O': f"Fingers forming circle ({bent_count}/5 fingers bent)",
        'S': f"Closed fist ({bent_count}/5 fingers bent)",
        'T': f"Thumb between fingers ({bent_count}/5 fingers bent)",
        'V': f"Peace sign ({bent_count}/5 fingers bent)",
        'W': f"Three fingers up ({bent_count}/5 fingers bent)",
        'X': f"Hooked index ({bent_count}/5 fingers bent)",
        'Y': f"Shaka sign ({bent_count}/5 fingers bent)"
    }
    
    return descriptions.get(letter, f"{bent_count}/5 fingers bent")


def boost_confidence_with_pattern_check(letter, confidence, features):
    """
    Boost confidence for weak letters if their pattern matches expectations.
    
    This helps improve confidence scores for letters that the model often confuses,
    but when the sensor pattern is actually a good match.
    
    Args:
        letter: Predicted letter
        confidence: Model's original confidence (0-1)
        features: 25-element feature array
    
    Returns:
        Boosted confidence (0-1)
    """
    # Extract means
    means = features[::5]
    
    # Normalize
    baselines = np.array([440, 612, 618, 548, 528])
    maxbends = np.array([650, 900, 900, 850, 800])
    normalized = (means - baselines) / (maxbends - baselines)
    normalized = np.clip(normalized, 0, 1)
    
    # Pattern verification for weak letters
    # Format: (thumb, index, middle, ring, pinky) - 1=bent, 0=straight, 0.5=moderate
    expected_patterns = {
        'T': (0, 0, 1, 1, 1),  # Thumb + index straight, others bent
        'C': (0, 1, 1, 1, 1),  # Thumb straight, curved fingers
        'O': (1, 1, 1, 1, 1),  # All moderate/bent (circle)
        'X': (0, 1, 1, 1, 1),  # Hooked index, others bent
        'S': (1, 1, 1, 1, 1),  # All bent (fist with thumb wrapped)
        'E': (1, 1, 1, 1, 1),  # All bent
        'A': (0, 1, 1, 1, 1),  # Thumb straight, fingers bent
    }
    
    if letter not in expected_patterns:
        return confidence  # No boost for strong letters
    
    expected = np.array(expected_patterns[letter])
    
    # Convert normalized to binary (1=bent, 0=straight)
    actual = (normalized > 0.5).astype(int)
    
    # Calculate match score
    match_score = np.sum(actual == expected) / 5.0
    
    # Boost logic:
    # - If pattern matches well (4/5 or 5/5) and confidence is low-moderate (20-75%)
    # - Boost confidence to 70-85% range
    if match_score >= 0.8 and 0.20 <= confidence <= 0.80:
        # Pattern matches! Boost confidence
        # Scale boost based on match quality and original confidence
        if match_score == 1.0:
            # Perfect match (5/5) - boost to 75-85%
            boosted = 0.75 + (confidence - 0.20) * 0.167  # 75-85% range
        else:
            # Good match (4/5) - boost to 70-80%
            boosted = 0.70 + (confidence - 0.20) * 0.167  # 70-80% range
        
        print(f"[CONFIDENCE BOOST] {letter}: {confidence:.1%} -> {boosted:.1%} (pattern match: {match_score:.0%})")
        return boosted
    
    return confidence


def predict_ml(samples_df):
    """
    Predict ASL letter using trained ML model.
    
    Args:
        samples_df: DataFrame with sensor data
    
    Returns:
        (letter, confidence, description)
    """
    # Try to find the model - check multiple locations
    model_locations = [
        # NEW: 15-letter model (best for demos!)
        Path(__file__).parent.parent / "models" / "rf_asl_15letters.pkl",
        # Bundled path (same directory as script in release)
        Path(__file__).parent / "rf_asl_15letters.pkl",
        # Fallback: 7-letter calibrated model
        Path(__file__).parent.parent / "models" / "rf_asl_calibrated.pkl",
        # Fallback: Absolute path
        Path("iot-sign-glove") / "models" / "rf_asl_15letters.pkl"
    ]
    
    model_path = None
    for path in model_locations:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        paths_checked = "\n  - ".join([str(p) for p in model_locations])
        raise FileNotFoundError(f"Model not found. Checked:\n  - {paths_checked}")
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    # Extract features
    features = extract_features_from_samples(samples_df)
    features = features.reshape(1, -1)  # Shape: (1, 25)
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Get prediction probabilities for confidence
    probabilities = model.predict_proba(features)[0]
    confidence = np.max(probabilities)
    
    # BOOST confidence for weak letters if pattern matches
    confidence = boost_confidence_with_pattern_check(prediction, confidence, features.flatten())
    
    # Generate description
    description = get_pattern_description(prediction, features.flatten())
    
    return prediction, confidence, description


def main():
    parser = argparse.ArgumentParser(description='ML-based ASL prediction')
    parser.add_argument('--file', type=str, required=True, help='Path to CSV file with sensor data')
    args = parser.parse_args()
    
    try:
        # Load CSV data
        df = pd.read_csv(args.file)
        
        if len(df) < 50:
            print("ASL_LETTER:NONE")
            print("CONFIDENCE:0.00")
            print(f"SAMPLES:{len(df)}")
            print("DESCRIPTION:Insufficient data (need at least 50 samples)")
            return 1
        
        # Predict
        letter, confidence, description = predict_ml(df)
        
        # Output in format expected by Rust backend
        print(f"ASL_LETTER:{letter}")
        print(f"CONFIDENCE:{confidence:.2f}")
        print(f"SAMPLES:{len(df)}")
        print(f"DESCRIPTION:{description}")
        
        return 0
        
    except FileNotFoundError as e:
        print("ASL_LETTER:ERROR")
        print("CONFIDENCE:0.00")
        print("SAMPLES:0")
        print(f"DESCRIPTION:{str(e)}")
        return 1
    except Exception as e:
        print("ASL_LETTER:ERROR")
        print("CONFIDENCE:0.00")
        print("SAMPLES:0")
        print(f"DESCRIPTION:Prediction error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

