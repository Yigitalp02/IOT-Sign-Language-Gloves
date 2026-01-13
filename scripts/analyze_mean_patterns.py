"""
Analyze what the mean values look like for each ASL class.
This will show if the classes are actually separable by mean values.
"""

import numpy as np
import sys
from pathlib import Path

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
    """Extract only mean values (first feature of each finger)."""
    indices = [i*6 for i in range(5)]  # [0, 6, 12, 18, 24]
    return X_original[:, indices]


def relabel_to_asl(gestures):
    """Convert gesture names to ASL letters."""
    return np.array([GESTURE_TO_ASL.get(g, 'UNKNOWN') for g in gestures])


def main():
    print("=" * 70)
    print("Analyzing Mean Value Patterns for ASL Classes")
    print("=" * 70)
    
    # Load data
    print("\nLoading windowed data...")
    X_30, y_gestures, users = load_and_window_data(window_size=1.0, stride=0.5)
    
    # Extract means
    X_means = extract_mean_features(X_30)
    
    # Relabel
    y_asl = relabel_to_asl(y_gestures)
    
    # Filter
    mask = np.isin(y_asl, TARGET_ASL)
    X = X_means[mask]
    y = y_asl[mask]
    
    print(f"Total windows: {len(X)}")
    
    # Analyze each class
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    print("\n" + "=" * 70)
    print("Mean Values by ASL Letter (normalized 0-1 scale)")
    print("=" * 70)
    
    for letter in sorted(TARGET_ASL):
        mask = y == letter
        class_data = X[mask]
        count = np.sum(mask)
        
        print(f"\n{letter} ({count} windows):")
        print("-" * 60)
        
        for i, finger in enumerate(finger_names):
            mean = np.mean(class_data[:, i])
            std = np.std(class_data[:, i])
            min_val = np.min(class_data[:, i])
            max_val = np.max(class_data[:, i])
            
            # Categorize
            if mean < 0.3:
                category = "LOW (extended)"
            elif mean > 0.7:
                category = "HIGH (bent)"
            else:
                category = "MID"
            
            print(f"  {finger:8s}: {mean:.3f} +/- {std:.3f}  [{min_val:.3f}, {max_val:.3f}]  {category}")
    
    # Show overlaps
    print("\n" + "=" * 70)
    print("Pattern Summary")
    print("=" * 70)
    
    for letter in sorted(TARGET_ASL):
        mask = y == letter
        class_data = X[mask]
        means = np.mean(class_data, axis=0)
        
        pattern = []
        for i, m in enumerate(means):
            if m < 0.3:
                pattern.append("LOW")
            elif m > 0.7:
                pattern.append("HIGH")
            else:
                pattern.append("MID")
        
        pattern_str = ", ".join([f"{finger_names[i]}:{pattern[i]}" for i in range(5)])
        print(f"{letter}: {pattern_str}")
    
    print("\n" + "=" * 70)
    print("Expected ASL Patterns vs Actual Data")
    print("=" * 70)
    
    expected = {
        'A': "All bent (all HIGH)",
        'D': "Index extended, thumb up (thumb MID/LOW, others HIGH)",
        'E': "Four fingers tucked (all HIGH)",
        'F': "OK sign - thumb touches index (thumb HIGH, index HIGH)",
        'I': "Pinky extended (pinky MID/LOW, others HIGH)",
    }
    
    print("\nExpected ASL:")
    for letter, desc in expected.items():
        print(f"  {letter}: {desc}")
    
    print("\nActual in training data:")
    for letter in sorted(TARGET_ASL):
        mask = y == letter
        class_data = X[mask]
        means = np.mean(class_data, axis=0)
        print(f"  {letter}: Thumb={means[0]:.2f}, Index={means[1]:.2f}, Middle={means[2]:.2f}, Ring={means[3]:.2f}, Pinky={means[4]:.2f}")


if __name__ == "__main__":
    main()

