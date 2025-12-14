"""
Load and analyze the professor's glove data
Converts from semicolon format to our standard format
"""

import pandas as pd
import numpy as np
import glob
import os

def load_calibration(subject_folder):
    """Load calibration file to get baseline values"""
    calib_files = glob.glob(f"{subject_folder}/*_Calibration.csv")
    if not calib_files:
        return None
    
    df = pd.read_csv(calib_files[0], sep=';')
    
    # Calculate baseline (average of calibration data)
    baseline = {
        'thumb': df['thumb'].mean(),
        'index': df['index'].mean(),
        'middle': df['middle'].mean(),
        'ring': df['ring'].mean(),
        'pinkie': df['pinkie'].mean(),
    }
    
    return baseline

def load_gesture_file(filepath, subject_id):
    """Load a single gesture CSV file"""
    # Extract gesture name from filename
    # e.g., "2024_05_23___15_46_23_TestSubject01_1_Single_Thumb.csv" -> "Single_Thumb"
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    
    # Find the gesture name (after the number)
    gesture_name = '_'.join(parts[5:]).replace('.csv', '')
    
    # Load data
    df = pd.read_csv(filepath, sep=';')
    
    # Rename columns to match our format
    df = df.rename(columns={
        'thumb': 'ch0_raw',
        'index': 'ch1_raw',
        'middle': 'ch2_raw',
        'ring': 'ch3_raw',
        'pinkie': 'ch4_raw',
    })
    
    # Add metadata
    df['user_id'] = subject_id
    df['class_label'] = gesture_name
    df['timestamp_ms'] = (df['timestamp'] * 1000).astype(int)
    
    return df, gesture_name

def normalize_data(df, baseline, maxbend=None):
    """Apply normalization to sensor data"""
    if maxbend is None:
        # Estimate max bend from data (use 90th percentile)
        maxbend = {
            'ch0_raw': df['ch0_raw'].quantile(0.9),
            'ch1_raw': df['ch1_raw'].quantile(0.9),
            'ch2_raw': df['ch2_raw'].quantile(0.9),
            'ch3_raw': df['ch3_raw'].quantile(0.9),
            'ch4_raw': df['ch4_raw'].quantile(0.9),
        }
    
    # Normalize: (raw - baseline) / (maxbend - baseline)
    for i in range(5):
        ch = f'ch{i}_raw'
        b = baseline.get(ch, df[ch].min())
        m = maxbend.get(ch, df[ch].max())
        df[f'ch{i}_norm'] = ((df[ch] - b) / (m - b)).clip(0, 1)
    
    return df

def load_all_data(data_folder="data/Data"):
    """Load all professor's data"""
    all_data = []
    
    subjects = sorted(glob.glob(f"{data_folder}/TestSubject*"))
    
    print(f"Loading data from {len(subjects)} subjects...")
    print("=" * 60)
    
    for subject_folder in subjects:
        subject_id = os.path.basename(subject_folder)
        print(f"\n{subject_id}:")
        
        # Load calibration
        baseline = load_calibration(subject_folder)
        if baseline:
            print(f"  Baseline: thumb={baseline['thumb']:.1f}, index={baseline['index']:.1f}, middle={baseline['middle']:.1f}, ring={baseline['ring']:.1f}, pinkie={baseline['pinkie']:.1f}")
        else:
            print(f"  No calibration found, will estimate from data")
            baseline = None
        
        # Load all gesture files
        gesture_files = glob.glob(f"{subject_folder}/*.csv")
        gesture_files = [f for f in gesture_files if 'Calibration' not in f]
        
        for filepath in gesture_files:
            try:
                df, gesture_name = load_gesture_file(filepath, subject_id)
                
                # Normalize if we have baseline
                if baseline:
                    baseline_dict = {f'ch{i}_raw': list(baseline.values())[i] for i in range(5)}
                    df = normalize_data(df, baseline_dict)
                
                all_data.append(df)
                print(f"    [OK] {gesture_name}: {len(df)} samples")
            except Exception as e:
                print(f"    [ERROR] Error loading {os.path.basename(filepath)}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Total files loaded: {len(all_data)}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Total samples: {len(combined):,}")
        print(f"Gestures: {sorted(combined['class_label'].unique())}")
        print(f"Users: {sorted(combined['user_id'].unique())}")
        return combined
    else:
        return None

def analyze_dataset(df):
    """Analyze the loaded dataset"""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal samples: {len(df):,}")
    print(f"Total users: {df['user_id'].nunique()}")
    print(f"Total gestures: {df['class_label'].nunique()}")
    
    print(f"\nGestures and sample counts:")
    gesture_counts = df.groupby('class_label').size().sort_values(ascending=False)
    for gesture, count in gesture_counts.items():
        print(f"  {gesture:25s}: {count:7,} samples ({count/100:.1f} seconds at 100Hz)")
    
    print(f"\nSamples per user:")
    user_counts = df.groupby('user_id').size().sort_values(ascending=False)
    for user, count in user_counts.items():
        print(f"  {user}: {count:6,} samples")
    
    print(f"\nSensor value ranges:")
    for i in range(5):
        ch = f'ch{i}_raw'
        print(f"  CH{i}: min={df[ch].min():.0f}, max={df[ch].max():.0f}, mean={df[ch].mean():.0f}")
    
    if 'ch0_norm' in df.columns:
        print(f"\nNormalized value ranges:")
        for i in range(5):
            ch = f'ch{i}_norm'
            print(f"  CH{i}_norm: min={df[ch].min():.3f}, max={df[ch].max():.3f}, mean={df[ch].mean():.3f}")

if __name__ == "__main__":
    # Load all data
    df = load_all_data("Data")
    
    if df is not None:
        # Analyze
        analyze_dataset(df)
        
        # Save to single file for easy ML training
        output_file = "data/professor_data_combined.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved combined dataset to: {output_file}")
        print(f"\nYou can now use this for ML training!")
    else:
        print("\nNo data loaded!")

