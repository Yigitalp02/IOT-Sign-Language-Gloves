"""
IoT Glove Simulator
===================
Simulates the IoT glove by reading CSV data from the professor's dataset
and sending it over a serial port for the desktop app to capture.

Usage:
    python glove_simulator.py [OPTIONS]

Options:
    --port COM3                Serial port to use (default: COM3)
    --baud 115200              Baud rate (default: 115200)
    --user 1                   Test subject ID (1-12) (default: 1)
    --gesture 1_Single_Thumb   Gesture to simulate (default: 1_Single_Thumb)
    --speed 1.0                Playback speed multiplier (default: 1.0)
    --loop                     Loop the gesture repeatedly

Available Gestures:
    1_Single_Thumb, 2_Single_Index, 3_Single_Middle, 4_Single_Ring,
    5_Single_Pinkie, 6_Grasp, 7_FourFinger_Grasp, 8_Thumb2Index,
    9_Thumb2Middle, 10_Thumb2Ring, 11_Thumb2Pinkie, 0_Calibration

Requirements:
    pip install pyserial
"""

import sys
import time
import argparse
import os
from pathlib import Path

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed!")
    print("Install it with: pip install pyserial")
    sys.exit(1)

# Available gestures from professor's dataset
GESTURES = [
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
    "11_Thumb2Pinkie",
    "0_Calibration"
]

# ASL letter to gesture mapping (based on finger patterns)
ASL_TO_GESTURE = {
    # Direct mappings
    "A": "6_Grasp",              # Closed fist
    "S": "6_Grasp",              # Closed fist (same as A)
    "T": "6_Grasp",              # Closed fist with thumb between fingers
    "F": "8_Thumb2Index",        # Thumb touches index (OK sign)
    "E": "7_FourFinger_Grasp",   # Four fingers tucked
    "I": "5_Single_Pinkie",      # Pinky extended
    "D": "1_Single_Thumb",       # Thumb extended
    
    # Numbers that might map
    "9": "8_Thumb2Index",        # OK sign (same as F)
}

# Combined list of available options (ASL letters + gestures)
ALL_GESTURES = list(ASL_TO_GESTURE.keys()) + GESTURES


def find_csv_file(data_folder: Path, user_id: int, gesture: str):
    """Find the CSV file for a specific user and gesture."""
    user_folder = data_folder / f"TestSubject{user_id:02d}"
    
    if not user_folder.exists():
        print(f"ERROR: User folder not found: {user_folder}")
        print(f"\nAvailable user folders:")
        for folder in sorted(data_folder.glob("TestSubject*")):
            print(f"  - {folder.name}")
        return None
    
    # Look for files containing the gesture name
    for csv_file in user_folder.glob("*.csv"):
        filename = csv_file.name
        # Check if gesture pattern is in filename
        # e.g., "2024_05_23___15_46_23_TestSubject01_1_Single_Thumb.csv"
        if f"_{gesture}.csv" in filename:
            return csv_file
    
    print(f"ERROR: No CSV file found for user {user_id}, gesture '{gesture}'")
    print(f"Searched in: {user_folder}")
    print(f"\nAvailable gestures for this user:")
    for csv_file in sorted(user_folder.glob("*.csv")):
        # Extract gesture name from filename
        parts = csv_file.stem.split('_')
        if len(parts) >= 4:
            gesture_part = '_'.join(parts[4:])  # e.g., "1_Single_Thumb"
            print(f"  - {gesture_part}")
    return None


def find_active_gesture_window(samples, window_size=200):
    """
    Find the most active window in the recording.
    
    Recordings contain: [rest] -> [gesture] -> [rest]
    We want the middle part where the gesture is actively being performed.
    
    Returns the window with the highest standard deviation (most movement).
    """
    if len(samples) <= window_size:
        return samples
    
    import numpy as np
    
    # Extract sensor values
    all_values = np.array([[s[1], s[2], s[3], s[4], s[5]] for s in samples])
    
    # Find window with highest total variation (std across all sensors)
    best_start = 0
    max_variation = 0
    
    for start in range(0, len(samples) - window_size, 10):  # Check every 10 samples
        window = all_values[start:start+window_size]
        
        # Calculate total variation (sum of std for each sensor)
        variation = np.sum(np.std(window, axis=0))
        
        if variation > max_variation:
            max_variation = variation
            best_start = start
    
    # Extract the most active window
    active_window = samples[best_start:best_start+window_size]
    
    print(f"[INFO] Selected active window: samples {best_start} to {best_start+window_size}")
    print(f"       (from {len(samples)} total samples)")
    print(f"       Variation score: {max_variation:.1f}")
    
    return active_window


def read_csv_data(csv_file: Path, extract_active=True, window_size=200):
    """Read sensor data from CSV file.
    
    Professor's CSV format (semicolon-delimited):
    timestamp;cycle;thumb;index;middle;ring;pinkie;thumb_y;index_y;middle_y;ring_y;pinkie_y
    
    Args:
        csv_file: Path to CSV file
        extract_active: If True, extract the most active window (gesture part)
        window_size: Size of window to extract (default 200 = 2 seconds @ 100Hz)
    """
    samples = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        # Skip header
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(';')
            if len(parts) < 7:  # Need at least timestamp + cycle + 5 sensors
                continue
            
            try:
                # Format: timestamp;cycle;thumb;index;middle;ring;pinkie;...
                timestamp = int(float(parts[0]) * 1000)  # Convert to milliseconds
                ch0 = int(parts[2])  # thumb
                ch1 = int(parts[3])  # index
                ch2 = int(parts[4])  # middle
                ch3 = int(parts[5])  # ring
                ch4 = int(parts[6])  # pinkie
                
                samples.append((timestamp, ch0, ch1, ch2, ch3, ch4))
            except (ValueError, IndexError):
                continue
    
    if extract_active and len(samples) > window_size:
        samples = find_active_gesture_window(samples, window_size)
    
    return samples


def simulate_glove(port_name: str, baud_rate: int, samples: list, 
                   speed: float = 1.0, loop: bool = False):
    """Send sensor data over serial port."""
    
    print(f"\n{'='*60}")
    print(f"IoT Glove Simulator")
    print(f"{'='*60}")
    print(f"Port: {port_name}")
    print(f"Baud Rate: {baud_rate}")
    print(f"Samples: {len(samples)}")
    print(f"Speed: {speed}x")
    print(f"Loop: {'Yes' if loop else 'No'}")
    print(f"{'='*60}\n")
    
    # Open serial port
    try:
        ser = serial.Serial(port_name, baud_rate, timeout=1)
        print(f"[OK] Connected to {port_name}")
        time.sleep(2)  # Wait for connection to stabilize
    except serial.SerialException as e:
        print(f"ERROR: Could not open serial port {port_name}")
        print(f"Error: {e}")
        print("\nAvailable ports:")
        from serial.tools import list_ports
        for port in list_ports.comports():
            print(f"  - {port.device} ({port.description})")
        return
    
    print("\n[START] Starting data transmission...")
    print("        (Press Ctrl+C to stop)\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"[SEND] Iteration {iteration} - Sending {len(samples)} samples...")
            
            for i, (timestamp, ch0, ch1, ch2, ch3, ch4) in enumerate(samples):
                # Format: timestamp,ch0,ch1,ch2,ch3,ch4
                message = f"{timestamp},{ch0},{ch1},{ch2},{ch3},{ch4}\n"
                ser.write(message.encode('utf-8'))
                
                # Progress indicator every 50 samples
                if (i + 1) % 50 == 0:
                    progress = (i + 1) / len(samples) * 100
                    print(f"       Progress: {i+1}/{len(samples)} ({progress:.1f}%)")
                
                # Sleep to simulate 100Hz sampling rate (10ms between samples)
                # Adjust by speed multiplier
                time.sleep(0.01 / speed)
            
            print(f"[OK] Iteration {iteration} complete!\n")
            
            if not loop:
                break
            
            time.sleep(1)  # Brief pause between loops
    
    except KeyboardInterrupt:
        print("\n\n[STOP] Stopped by user")
    
    finally:
        ser.close()
        print("[OK] Serial port closed")


def main():
    parser = argparse.ArgumentParser(
        description="IoT Glove Simulator - Simulates glove data over serial port",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available ASL Letters (mapped to gestures):
  A, S, T (closed fist) -> 6_Grasp
  F, 9 (OK sign)        -> 8_Thumb2Index
  E (tucked fingers)    -> 7_FourFinger_Grasp
  I (pinky up)          -> 5_Single_Pinkie
  D (thumb up)          -> 1_Single_Thumb

Available Gestures (descriptive names):
  {chr(10).join(f'  - {g}' for g in GESTURES)}

Examples:
  # Using ASL letters:
  python glove_simulator.py --port COM3 --gesture A --loop
  python glove_simulator.py --port COM3 --gesture F --user 2
  
  # Using descriptive names:
  python glove_simulator.py --port COM3 --gesture 1_Single_Thumb --loop
  python glove_simulator.py --user 5 --gesture 6_Grasp --speed 2.0
"""
    )
    parser.add_argument("--port", default="COM3", help="Serial port (default: COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--user", type=int, default=1, choices=range(1, 13), 
                        help="Test subject ID (1-12) (default: 1)")
    parser.add_argument("--gesture", default="A", 
                        help="ASL letter (A,S,T,F,E,I,D,9) or gesture name (default: A)")
    parser.add_argument("--speed", type=float, default=1.0, 
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--loop", action="store_true", 
                        help="Loop the gesture repeatedly")
    
    args = parser.parse_args()
    
    # Resolve ASL letter to gesture name
    gesture_input = args.gesture
    
    if gesture_input in ASL_TO_GESTURE:
        # It's an ASL letter - map to gesture
        actual_gesture = ASL_TO_GESTURE[gesture_input]
        print(f"\n[ASL LETTER] '{gesture_input}' -> Gesture: {actual_gesture}")
    elif gesture_input in GESTURES:
        # It's a gesture name - use directly
        actual_gesture = gesture_input
    else:
        print(f"ERROR: Unknown gesture or ASL letter: {gesture_input}")
        print(f"\nAvailable ASL letters: {', '.join(sorted(ASL_TO_GESTURE.keys()))}")
        print(f"Available gestures: {', '.join(GESTURES)}")
        sys.exit(1)
    
    # Update args with resolved gesture
    args.gesture = actual_gesture
    
    # Find data folder
    script_dir = Path(__file__).parent
    data_folder = script_dir.parent / "data" / "Data"
    
    if not data_folder.exists():
        print(f"ERROR: Data folder not found: {data_folder}")
        print("Make sure the professor's dataset is in: iot-sign-glove/data/Data/")
        sys.exit(1)
    
    # Find CSV file
    csv_file = find_csv_file(data_folder, args.user, args.gesture)
    if not csv_file:
        sys.exit(1)
    
    print(f"\n[*] Loading: {csv_file.name}")
    
    # Read data
    samples = read_csv_data(csv_file)
    if not samples:
        print("ERROR: No valid samples found in CSV file")
        sys.exit(1)
    
    print(f"[OK] Loaded {len(samples)} samples")
    
    # Simulate
    simulate_glove(args.port, args.baud, samples, args.speed, args.loop)


if __name__ == "__main__":
    main()

