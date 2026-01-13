"""
Rule-based ASL prediction for simulator demo.

Maps raw sensor patterns to ASL letters based on simple rules.
Designed to work with static poses from the simulator.
"""

import sys
import argparse
import numpy as np
import pandas as pd


def normalize_simple(raw_value, finger_idx):
    """Simple normalization: baseline to max bend."""
    baselines = [440, 612, 618, 548, 528]  # Thumb, Index, Middle, Ring, Pinky
    maxbends = [650, 900, 900, 850, 800]
    
    baseline = baselines[finger_idx]
    maxbend = maxbends[finger_idx]
    
    normalized = (raw_value - baseline) / (maxbend - baseline)
    return max(0.0, min(1.0, normalized))


def predict_asl_rule_based(samples_df):
    """
    Rule-based ASL prediction using finger patterns.
    
    Returns: (letter, confidence, description)
    """
    # Calculate mean raw values
    means = [samples_df[f'ch{i}'].mean() for i in range(5)]
    
    # Normalize
    norm = [normalize_simple(means[i], i) for i in range(5)]
    
    thumb, index, middle, ring, pinky = norm
    
    # Count how many fingers are bent (HIGH > 0.5)
    bent = [f > 0.5 for f in norm]
    bent_count = sum(bent)
    
    # Rule-based classification
    if bent_count == 0:
        # All fingers straight
        return "NONE", 0.5, "All fingers extended (resting position)"
    
    elif bent_count == 5:
        # All fingers bent (fist)
        # A: Thumb tucked and bent (>0.7)
        # S: Thumb wrapped, less bent (<0.7)
        if thumb > 0.7:
            return "A", 0.9, "Closed fist (thumb tucked)"
        else:
            return "S", 0.85, "Closed fist (thumb wrapped)"
    
    elif bent_count == 4:
        # Four fingers bent, one extended
        if not bent[0]:  # Thumb extended
            return "D", 0.85, "Thumb extended, others bent"
        elif not bent[4]:  # Pinky extended
            return "I", 0.85, "Pinky extended, others bent"
        elif not bent[1]:  # Index extended
            return "G", 0.8, "Index extended (not in target set)"
        else:
            return "UNKNOWN", 0.3, f"{bent_count} fingers bent"
    
    elif bent_count == 3:
        # Three fingers bent
        if not bent[0] and not bent[1]:
            # Thumb and index straight (D - index finger pointing up)
            return "D", 0.85, "Index finger extended (thumb up)"
        elif not bent[0] and not bent[4]:
            # Thumb and pinky straight (Y)
            return "Y", 0.85, "Thumb and pinky extended"
        elif not bent[2] and not bent[3] and not bent[4]:
            # Middle, ring, pinky extended
            return "W", 0.7, "Three fingers extended (not in target)"
        else:
            return "E", 0.6, "Partial fist"
    
    elif bent_count == 2:
        # Two fingers bent
        if bent[0] and bent[1]:
            # Both thumb and index bent - distinguish F vs T
            # F: Thumb and index similarly bent (~0.7 each)
            # T: Index more bent than thumb (index ~0.9, thumb ~0.5)
            if abs(thumb - index) < 0.2:
                # Similar bend values = F (OK sign)
                return "F", 0.85, "Thumb and index extended (OK sign)"
            else:
                # Index more bent = T (thumb tucked)
                return "T", 0.85, "Thumb tucked between fingers"
        else:
            return "OPEN", 0.6, "Two fingers bent"
    
    elif bent_count <= 1:
        # Mostly extended
        if bent[0] and not any(bent[1:]):
            # Only thumb bent
            return "E", 0.7, "Four fingers extended, thumb tucked"
        else:
            return "OPEN", 0.5, "Mostly extended"
    
    else:
        return "UNKNOWN", 0.3, "Unclear pattern"


def main():
    parser = argparse.ArgumentParser(description="Rule-based ASL prediction")
    parser.add_argument("--file", help="CSV file with sensor samples")
    parser.add_argument("--samples", help="Raw CSV string")
    
    args = parser.parse_args()
    
    # Load data
    if args.file:
        try:
            df = pd.read_csv(args.file)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    elif args.samples:
        from io import StringIO
        df = pd.read_csv(StringIO(args.samples), header=None, names=['timestamp', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4'])
    else:
        print("ERROR: Need --file or --samples")
        sys.exit(1)
    
    # Validate
    required_cols = ['ch0', 'ch1', 'ch2', 'ch3', 'ch4']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Need columns {required_cols}")
        sys.exit(1)
    
    # Predict
    letter, confidence, description = predict_asl_rule_based(df)
    
    # Calculate stats
    means = [df[f'ch{i}'].mean() for i in range(5)]
    norms = [normalize_simple(means[i], i) for i in range(5)]
    
    # Output
    print(f"ASL_LETTER:{letter}")
    print(f"CONFIDENCE:{confidence:.2f}")
    print(f"SAMPLES:{len(df)}")
    print(f"DESCRIPTION:{description}")
    
    # Debug
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    print(f"\n[DEBUG] Raw means:")
    for name, mean in zip(finger_names, means):
        print(f"  {name:8s}: {mean:6.1f}")
    
    print(f"\n[DEBUG] Normalized (0=straight, 1=bent):")
    for name, norm in zip(finger_names, norms):
        state = "BENT" if norm > 0.5 else "straight"
        print(f"  {name:8s}: {norm:5.3f}  ({state})")


if __name__ == "__main__":
    main()

