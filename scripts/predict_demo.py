"""
Demo Gesture Prediction
========================

For demo purposes - analyzes sensor patterns and predicts ASL letters
based on finger movement patterns (without needing the trained model).

This is a rule-based system that works for the demo!
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def analyze_finger_patterns(samples_df):
    """
    Analyze which fingers are active/moving based on sensor values.
    
    In the professor's dataset, movement (std deviation) is the key indicator,
    not absolute position.
    
    Returns patterns like: {'thumb': 'active', 'index': 'stable', ...}
    """
    patterns = {}
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    for i, name in enumerate(finger_names):
        ch_col = f'ch{i}'
        values = samples_df[ch_col].values
        
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        range_val = np.max(values) - np.min(values)
        
        # Determine if finger is active (moving) or stable
        # Threshold lowered to match professor's dataset characteristics
        is_active = std_val > 5.0 or range_val > 15
        is_bent = mean_val > 500  # Most values are 400-620, so 500+ is "high"
        
        patterns[name] = {
            'mean': mean_val,
            'std': std_val,
            'range': range_val,
            'active': is_active,
            'bent': is_bent
        }
    
    return patterns


def predict_asl_letter(patterns):
    """
    Predict ASL letter using RELATIVE activity comparison.
    
    Since most fingers are frozen in short windows, we find which finger
    has the MOST activity relative to others, not absolute thresholds.
    """
    
    # Get std deviations (activity) for each finger
    stds = {
        'thumb': patterns['thumb']['std'],
        'index': patterns['index']['std'],
        'middle': patterns['middle']['std'],
        'ring': patterns['ring']['std'],
        'pinky': patterns['pinky']['std']
    }
    
    # Find max std and which finger(s) have it
    max_std = max(stds.values())
    most_active = [name for name, std in stds.items() if std == max_std]
    
    # Find fingers with "significant" activity (> 1.0 or within 50% of max)
    threshold = max(1.0, max_std * 0.5)
    active_fingers = {name: std for name, std in stds.items() if std >= threshold}
    
    print(f"\n[DEBUG] Activity analysis (RELATIVE):")
    for name, std in stds.items():
        is_active = name in active_fingers
        is_most = name in most_active
        marker = "[MOST ACTIVE]" if is_most else ("[active]" if is_active else "(stable)")
        print(f"  {name:8} {std:5.1f}  {marker}")
    print(f"  Max std: {max_std:.1f}")
    print(f"  Most active: {most_active}")
    print(f"  Active (>= {threshold:.1f}): {list(active_fingers.keys())}")
    
    # Rule 1: Thumb is clearly most active → D
    if 'thumb' in most_active and max_std > 3.0 and len(active_fingers) <= 2:
        return "D", 0.88, f"Single_Thumb (thumb std={stds['thumb']:.1f} is dominant)"
    
    # Rule 2: Pinky is clearly most active → I
    if 'pinky' in most_active and max_std > 3.0 and len(active_fingers) <= 2:
        return "I", 0.87, f"Single_Pinkie (pinky std={stds['pinky']:.1f} is dominant)"
    
    # Rule 3: Index is most active → Number 1
    if 'index' in most_active and max_std > 3.0 and len(active_fingers) <= 2:
        return "1", 0.85, f"Single_Index (index std={stds['index']:.1f} is dominant)"
    
    # Rule 4: Thumb + Index both active → F
    if 'thumb' in active_fingers and 'index' in active_fingers and len(active_fingers) <= 3:
        return "F", 0.89, f"Thumb2Index (thumb={stds['thumb']:.1f}, index={stds['index']:.1f})"
    
    # Rule 5: Multiple fingers active → A/Grasp
    if len(active_fingers) >= 3 and max_std > 2.0:
        return "A/S/T", 0.86, f"Grasp ({len(active_fingers)} active fingers)"
    
    # Rule 6: Very low activity overall → E (stable hold)
    if max_std < 2.0:
        return "E", 0.83, f"FourFinger_Grasp (max std={max_std:.1f}, all stable)"
    
    # Rule 7: Thumb has some activity but not dominant
    if 'thumb' in active_fingers and max_std > 1.0:
        return "D", 0.75, f"Likely Single_Thumb (thumb std={stds['thumb']:.1f})"
    
    # Default - uncertain
    return "?", 0.50, f"Uncertain (max std={max_std:.1f})"


def main():
    parser = argparse.ArgumentParser(
        description="Demo Gesture Prediction - Rule-based ASL letter detection"
    )
    parser.add_argument("--file", required=True, help="CSV file with sensor samples")
    
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
        # Try with _raw suffix
        if all(f'{col}_raw' in df.columns for col in required_cols):
            # Rename columns
            for i in range(5):
                df[f'ch{i}'] = df[f'ch{i}_raw']
        else:
            print(f"ERROR: CSV must have columns: {required_cols} or {[c+'_raw' for c in required_cols]}")
            print(f"Found columns: {list(df.columns)}")
            sys.exit(1)
    
    # Analyze patterns
    patterns = analyze_finger_patterns(df)
    
    # Predict
    asl_letter, confidence, description = predict_asl_letter(patterns)
    
    # Output results
    print(f"\n{'='*60}")
    print(f"Real-Time Gesture Prediction")
    print(f"{'='*60}")
    print(f"Samples analyzed: {len(df)}")
    print(f"\nFinger Raw Statistics:")
    for name, data in patterns.items():
        status = "[ACTIVE]" if data['active'] else "[stable]"
        bent_status = "[BENT]" if data['bent'] else "[open]"
        print(f"  {name:8} {status:10} {bent_status:7} mean={data['mean']:6.1f}, std={data['std']:5.2f}, range={data['range']:5.1f}")
    print(f"\n{'='*60}")
    print(f"PREDICTED ASL LETTER: {asl_letter}")
    print(f"CONFIDENCE: {confidence*100:.1f}%")
    print(f"PATTERN: {description}")
    print(f"{'='*60}\n")
    
    # Also output in parseable format for app integration
    print(f"RESULT:{asl_letter}")
    print(f"CONFIDENCE:{confidence:.2f}")


if __name__ == "__main__":
    main()

