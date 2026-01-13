"""
Generate windowed features for 1.0s windows with 30 features
Optimized configuration based on window size comparison
"""

import sys
import os

# Change to iot-sign-glove root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(script_dir))

# Import after changing directory
from windowed_features import load_and_window_data, analyze_windowed_data
import numpy as np

print("="*60)
print("GENERATING OPTIMIZED 1.0s WINDOW FEATURES")
print("="*60)
print("Configuration: 1.0s window, 0.5s stride")
print("Features: 30 per window (6 per channel)")
print("="*60)

# Generate features
X, y, users = load_and_window_data(
    window_size=1.0,
    stride=0.5
)

# Analyze
analyze_windowed_data(X, y, users)

# Save
output_file = "data/windowed_features_1.0s_0.5s.npz"
np.savez_compressed(
    output_file,
    X=X,
    y=y,
    users=users,
    window_size=1.0,
    stride=0.5
)
print(f"\n[OK] Saved windowed features to: {output_file}")
print("="*60)




