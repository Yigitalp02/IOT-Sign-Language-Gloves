"""
Create ASL Letter Mapping from Professor's Dataset
====================================================

This script analyzes the professor's 11-gesture dataset and creates
a mapping to ASL letters based on finger bend patterns.

Strategy:
1. Load raw sensor data from each gesture class
2. Analyze which fingers are bent vs extended
3. Map finger patterns to ASL letters
4. Create new training labels for ASL recognition
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_gesture_patterns(data_folder: Path):
    """
    Analyze each gesture to understand finger patterns.
    
    Returns a dictionary mapping gesture names to average finger states.
    """
    patterns = {}
    
    gesture_types = [
        "1_Single_Thumb",
        "2_Single_Index", 
        "3_Single_Middle",
        "4_Single_Ring",
        "5_Single_Pinkie",
        "6_Grasp",
        "7_FourFinger_Grasp",
        "8_Thumb2Index",
        "9_Thumb2Middle",
        "10_Thumb2Ring",
        "11_Thumb2Pinkie"
    ]
    
    print("="*60)
    print("Analyzing Finger Patterns in Professor's Dataset")
    print("="*60)
    
    for gesture in gesture_types:
        print(f"\nAnalyzing: {gesture}")
        
        # Collect data from all users for this gesture
        all_samples = []
        
        for user_id in range(1, 13):  # 12 users
            user_folder = data_folder / f"TestSubject{user_id:02d}"
            
            # Find file for this gesture
            for csv_file in user_folder.glob(f"*{gesture}.csv"):
                # Read CSV (semicolon-delimited)
                df = pd.read_csv(csv_file, sep=';')
                
                # Extract finger sensor values (columns: thumb, index, middle, ring, pinkie)
                if len(df) > 0:
                    finger_values = df[['thumb', 'index', 'middle', 'ring', 'pinkie']].values
                    all_samples.append(finger_values)
                break
        
        if all_samples:
            # Combine all samples
            combined = np.vstack(all_samples)
            
            # Calculate statistics
            mean_values = combined.mean(axis=0)
            std_values = combined.std(axis=0)
            min_values = combined.min(axis=0)
            max_values = combined.max(axis=0)
            
            patterns[gesture] = {
                'mean': mean_values,
                'std': std_values,
                'min': min_values,
                'max': max_values,
                'samples': len(combined)
            }
            
            # Print analysis
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            print(f"  Samples: {len(combined)}")
            print(f"  Average sensor values:")
            for i, name in enumerate(finger_names):
                print(f"    {name:8} -> {mean_values[i]:6.1f} +/- {std_values[i]:5.1f} (range: {min_values[i]:4.0f}-{max_values[i]:4.0f})")
    
    return patterns


def create_asl_mapping(patterns):
    """
    Create ASL letter mapping based on gesture patterns.
    
    ASL letters that can be mapped from the 11 gestures:
    - A: Closed fist (Grasp)
    - S: Closed fist (Grasp) 
    - E: Fingers tucked (FourFinger_Grasp)
    - F: Thumb and index touch (Thumb2Index)
    - T: Similar to closed fist (Grasp)
    
    NOTE: Many ASL letters require finger combinations not present in dataset!
    """
    
    print("\n" + "="*60)
    print("ASL Letter Mapping")
    print("="*60)
    
    asl_mapping = {
        # Direct matches
        "6_Grasp": ["A", "S", "T"],  # Closed fist variations
        "8_Thumb2Index": ["F"],       # OK sign / pinch
        "7_FourFinger_Grasp": ["E"],  # Tucked fingers
        
        # Partial matches (need verification)
        "5_Single_Pinkie": ["I"],     # Only if pinky is extended
        "1_Single_Thumb": ["D"],      # Only if thumb is up, others closed
        
        # Missing - require multiple fingers extended:
        # V, W, K, H, U, R, etc. - NOT in dataset
    }
    
    print("\n[OK] Direct Mappings:")
    for gesture, letters in asl_mapping.items():
        print(f"  {gesture:25} -> {', '.join(letters)}")
    
    print("\n[MISSING] ASL Letters (require finger combinations):")
    missing = ["B", "C", "G", "H", "K", "L", "M", "N", "O", "P", "Q", "R", "U", "V", "W", "X", "Y", "Z"]
    print(f"  {', '.join(missing)}")
    print(f"\n  These require multiple fingers extended simultaneously,")
    print(f"  which is not captured in the current 11-gesture dataset.")
    
    return asl_mapping


def create_relabeled_dataset(data_folder: Path, output_folder: Path, asl_mapping):
    """
    Create a new dataset with ASL letter labels instead of gesture names.
    """
    output_folder.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating Relabeled Dataset for ASL Recognition")
    print("="*60)
    
    relabeled_count = 0
    skipped_count = 0
    
    for gesture, asl_letters in asl_mapping.items():
        # For simplicity, use the first ASL letter mapping
        asl_label = asl_letters[0]
        
        print(f"\nRelabeling {gesture} -> {asl_label}")
        
        # Process all users
        for user_id in range(1, 13):
            user_folder = data_folder / f"TestSubject{user_id:02d}"
            
            # Find file for this gesture
            for csv_file in user_folder.glob(f"*{gesture}.csv"):
                # Read original CSV
                df = pd.read_csv(csv_file, sep=';')
                
                # Save with new label
                output_file = output_folder / f"User{user_id:02d}_{asl_label}_{csv_file.stem}.csv"
                df.to_csv(output_file, index=False)
                
                relabeled_count += 1
                print(f"  [OK] {csv_file.name} -> {output_file.name}")
                break
    
    print(f"\n[DONE] Relabeled {relabeled_count} files")
    print(f"[SAVED] To: {output_folder}")
    
    return relabeled_count


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    data_folder = script_dir.parent / "data" / "Data"
    output_folder = script_dir.parent / "data" / "ASL_Labeled"
    
    if not data_folder.exists():
        print(f"ERROR: Data folder not found: {data_folder}")
        sys.exit(1)
    
    # Step 1: Analyze gesture patterns
    print("\n[Step 1] Analyzing gesture patterns...\n")
    patterns = analyze_gesture_patterns(data_folder)
    
    # Step 2: Create ASL mapping
    print("\n[Step 2] Creating ASL letter mapping...\n")
    asl_mapping = create_asl_mapping(patterns)
    
    # Step 3: Create relabeled dataset
    print("\n[Step 3] Creating relabeled dataset...\n")
    relabeled_count = create_relabeled_dataset(data_folder, output_folder, asl_mapping)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"[OK] Successfully mapped {len(asl_mapping)} gestures to ASL letters")
    print(f"[OK] Created {relabeled_count} relabeled files")
    print(f"[NOTE] Only ~6 ASL letters can be recognized with current data")
    print(f"[INFO] To recognize more letters, you'll need data with finger combinations")
    print("="*60)


if __name__ == "__main__":
    main()

