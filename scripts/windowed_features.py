"""
Windowed Feature Extraction for Gesture Recognition

This script implements sliding window feature extraction to capture
temporal patterns in the sensor data.

Instead of classifying single samples, we use 0.5-1.0 second windows
to extract statistical features, which improves accuracy significantly.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import os

def extract_window_features(window_data: pd.DataFrame, sample_rate: int = 100) -> np.ndarray:
    """
    Extract statistical and temporal features from a time window
    
    For each of the 5 sensor channels, we extract:
    
    Static Features:
    - Mean (average value)
    - Std (variability)
    - Min (minimum value)
    - Max (maximum value)
    
    Dynamic Features (inspired by professor's approach):
    - Velocity (mean rate of change) - captures movement speed
    - Acceleration (mean change in velocity) - captures smoothness
    
    Total: 5 channels × 6 features = 30 features per window
    """
    features = []
    
    # Extract features for each channel
    for i in range(5):
        ch_col = f'ch{i}_norm'
        if ch_col not in window_data.columns:
            ch_col = f'ch{i}_raw'  # Fallback to raw if normalized not available
        
        values = window_data[ch_col].values
        
        # Static statistical features
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Dynamic temporal features (NEW!)
        # Velocity: First derivative (rate of change)
        if len(values) > 1:
            velocity = np.diff(values)  # Calculate differences
            mean_velocity = np.mean(velocity) * sample_rate  # Scale by sample rate
            std_velocity = np.std(velocity) * sample_rate
        else:
            mean_velocity = 0
            std_velocity = 0
        
        # Acceleration: Second derivative (change in velocity)
        # Helps distinguish smooth vs jerky movements
        if len(values) > 2:
            acceleration = np.diff(np.diff(values))  # Second derivative
            mean_acceleration = np.mean(acceleration) * (sample_rate ** 2)
        else:
            mean_acceleration = 0
        
        # Combine all features for this channel
        features.extend([
            mean_val,           # Average position
            std_val,            # Variability
            min_val,            # Minimum bend
            max_val,            # Maximum bend
            mean_velocity,      # Movement speed (NEW!)
            mean_acceleration,  # Movement smoothness (NEW!)
        ])
    
    return np.array(features)

def create_sliding_windows(
    df: pd.DataFrame,
    window_size: float = 0.5,
    stride: float = 0.25,
    sample_rate: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows from continuous sensor data
    
    Args:
        df: DataFrame with sensor data
        window_size: Window size in seconds (default: 0.5s)
        stride: Stride/step size in seconds (default: 0.25s = 50% overlap)
        sample_rate: Sampling rate in Hz (default: 100Hz)
    
    Returns:
        X: Feature matrix (n_windows, n_features)
        y: Labels (n_windows,)
        users: User IDs (n_windows,)
    """
    window_samples = int(window_size * sample_rate)
    stride_samples = int(stride * sample_rate)
    
    print(f"Creating sliding windows:")
    print(f"  Window size: {window_size}s ({window_samples} samples)")
    print(f"  Stride: {stride}s ({stride_samples} samples)")
    print(f"  Overlap: {(1 - stride/window_size)*100:.0f}%")
    
    X_windows = []
    y_windows = []
    user_windows = []
    
    # Group by user and gesture to maintain continuity
    grouped = df.groupby(['user_id', 'gesture'])
    
    for (user, gesture), group in grouped:
        # Sort by timestamp to ensure temporal order
        group = group.sort_values('timestamp')
        
        # Create sliding windows
        for start_idx in range(0, len(group) - window_samples + 1, stride_samples):
            end_idx = start_idx + window_samples
            window_data = group.iloc[start_idx:end_idx]
            
            # Skip if window doesn't have enough samples
            if len(window_data) < window_samples:
                continue
            
            # Extract features (with velocity & acceleration)
            features = extract_window_features(window_data, sample_rate)
            
            X_windows.append(features)
            y_windows.append(gesture)
            user_windows.append(user)
    
    X = np.array(X_windows)
    y = np.array(y_windows)
    users = np.array(user_windows)
    
    print(f"\nCreated {len(X):,} windows")
    print(f"Feature dimension: {X.shape[1]}")
    
    return X, y, users

def load_and_window_data(
    data_file: str = "data/Data/professor_data_combined.csv",
    window_size: float = 0.5,
    stride: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data and create windowed features
    
    Args:
        data_file: Path to combined dataset
        window_size: Window size in seconds
        stride: Stride size in seconds
    
    Returns:
        X: Feature matrix
        y: Labels
        users: User IDs
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} samples")
    
    # Extract gesture labels (remove timestamp prefix)
    def extract_gesture_label(full_label):
        parts = full_label.split('_')
        for i, part in enumerate(parts):
            if part.startswith('TestSubject'):
                if i + 2 < len(parts):
                    return '_'.join(parts[i + 2:])
        return full_label
    
    df['gesture'] = df['class_label'].apply(extract_gesture_label)
    df['timestamp'] = df['timestamp_ms'] / 1000.0  # Convert to seconds
    
    print(f"Unique gestures: {df['gesture'].nunique()}")
    print(f"Unique users: {df['user_id'].nunique()}")
    
    # Create windows
    X, y, users = create_sliding_windows(df, window_size, stride)
    
    return X, y, users

def analyze_windowed_data(X: np.ndarray, y: np.ndarray, users: np.ndarray):
    """Print statistics about windowed data"""
    print("\n" + "="*60)
    print("WINDOWED DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal windows: {len(X):,}")
    print(f"Features per window: {X.shape[1]}")
    print(f"Unique gestures: {len(np.unique(y))}")
    print(f"Unique users: {len(np.unique(users))}")
    
    print(f"\nWindows per gesture:")
    unique_gestures, counts = np.unique(y, return_counts=True)
    for gesture, count in sorted(zip(unique_gestures, counts), key=lambda x: -x[1]):
        print(f"  {gesture:25s}: {count:6,} windows")
    
    print(f"\nWindows per user:")
    unique_users, counts = np.unique(users, return_counts=True)
    for user, count in sorted(zip(unique_users, counts), key=lambda x: -x[1]):
        print(f"  {user}: {count:6,} windows")
    
    print(f"\nFeature statistics:")
    print(f"  Mean: {X.mean():.4f}")
    print(f"  Std:  {X.std():.4f}")
    print(f"  Min:  {X.min():.4f}")
    print(f"  Max:  {X.max():.4f}")

if __name__ == "__main__":
    # Change to iot-sign-glove root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    # Test with different window configurations
    configs = [
        (0.5, 0.25),  # 0.5s window, 50% overlap
        (1.0, 0.5),   # 1.0s window, 50% overlap
        (0.75, 0.25), # 0.75s window, 67% overlap
    ]
    
    for window_size, stride in configs:
        print("\n" + "="*60)
        print(f"CONFIGURATION: Window={window_size}s, Stride={stride}s")
        print("="*60)
        
        X, y, users = load_and_window_data(
            window_size=window_size,
            stride=stride
        )
        
        analyze_windowed_data(X, y, users)
        
        # Save windowed data
        output_file = f"data/windowed_features_{window_size}s_{stride}s.npz"
        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            users=users,
            window_size=window_size,
            stride=stride
        )
        print(f"\n[OK] Saved windowed features to: {output_file}")

