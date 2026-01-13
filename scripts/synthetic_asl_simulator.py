"""
Synthetic ASL data generator for demo.

Generates CLEAR ASL hand patterns with realistic sensor values.
"""

import serial
import time
import argparse
import random

# ASL Letter definitions (normalized 0-1, where 0=straight, 1=fully bent)
ASL_PATTERNS = {
    'A': {'thumb': 0.9, 'index': 0.9, 'middle': 0.9, 'ring': 0.9, 'pinky': 0.9},  # Closed fist (thumb tucked)
    'D': {'thumb': 0.1, 'index': 0.1, 'middle': 0.9, 'ring': 0.9, 'pinky': 0.9},  # Index + thumb up
    'E': {'thumb': 0.9, 'index': 0.3, 'middle': 0.3, 'ring': 0.3, 'pinky': 0.3},  # Four fingers tucked
    'F': {'thumb': 0.7, 'index': 0.7, 'middle': 0.1, 'ring': 0.1, 'pinky': 0.1},  # OK sign (thumb+index)
    'I': {'thumb': 0.9, 'index': 0.9, 'middle': 0.9, 'ring': 0.9, 'pinky': 0.1},  # Pinky up
    'S': {'thumb': 0.6, 'index': 0.9, 'middle': 0.9, 'ring': 0.9, 'pinky': 0.9},  # Fist (thumb wrapped, less bent)
    'Y': {'thumb': 0.1, 'index': 0.9, 'middle': 0.9, 'ring': 0.9, 'pinky': 0.1},  # Thumb + Pinky extended
}

# Raw sensor ranges (ADC values)
BASELINES = {'thumb': 440, 'index': 612, 'middle': 618, 'ring': 548, 'pinky': 528}
MAXBENDS = {'thumb': 650, 'index': 900, 'middle': 900, 'ring': 850, 'pinky': 800}


def denormalize(normalized_value, finger):
    """Convert normalized (0-1) back to raw ADC value."""
    baseline = BASELINES[finger]
    maxbend = MAXBENDS[finger]
    return int(baseline + (normalized_value * (maxbend - baseline)))


def add_noise(value, noise_level=3):
    """Add realistic sensor noise."""
    return value + random.randint(-noise_level, noise_level)


def generate_asl_sample(letter, timestamp):
    """Generate a single sensor sample for an ASL letter."""
    if letter not in ASL_PATTERNS:
        letter = 'A'  # Default
    
    pattern = ASL_PATTERNS[letter]
    
    # Convert to raw values
    ch0 = denormalize(pattern['thumb'], 'thumb')
    ch1 = denormalize(pattern['index'], 'index')
    ch2 = denormalize(pattern['middle'], 'middle')
    ch3 = denormalize(pattern['ring'], 'ring')
    ch4 = denormalize(pattern['pinky'], 'pinky')
    
    # Add noise for realism
    ch0 = add_noise(ch0)
    ch1 = add_noise(ch1)
    ch2 = add_noise(ch2)
    ch3 = add_noise(ch3)
    ch4 = add_noise(ch4)
    
    return (timestamp, ch0, ch1, ch2, ch3, ch4)


def simulate_asl(port_name, baud_rate, letter, loop=False):
    """Simulate ASL letter over serial port."""
    
    print("=" * 60)
    print(f"Synthetic ASL Simulator - Letter '{letter}'")
    print("=" * 60)
    print(f"Port: {port_name}")
    print(f"Pattern: {ASL_PATTERNS.get(letter, 'Unknown')}")
    print("=" * 60)
    
    # Open serial port
    try:
        ser = serial.Serial(port_name, baud_rate, timeout=1)
        print(f"\n[OK] Connected to {port_name}")
        time.sleep(2)
    except serial.SerialException as e:
        print(f"\nERROR: Could not open port {port_name}")
        print(f"Error: {e}")
        return
    
    print(f"\n[START] Sending ASL '{letter}' data...")
    print("        (Press Ctrl+C to stop)\n")
    
    try:
        timestamp = 0
        iteration = 0
        
        while True:
            iteration += 1
            print(f"[SEND] Cycle {iteration} - Sending 200 samples...")
            
            for i in range(200):  # Send 200 samples (2 seconds @ 100Hz)
                timestamp += 10  # 10ms per sample
                
                # Generate sample
                ts, ch0, ch1, ch2, ch3, ch4 = generate_asl_sample(letter, timestamp)
                
                # Send over serial
                message = f"{ts},{ch0},{ch1},{ch2},{ch3},{ch4}\n"
                ser.write(message.encode('utf-8'))
                
                # Sleep for 100Hz (10ms)
                time.sleep(0.01)
            
            print(f"[OK] Sent 200 samples\n")
            
            if not loop:
                break
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n[STOP] Stopped by user")
    
    finally:
        ser.close()
        print("[OK] Port closed")


def main():
    parser = argparse.ArgumentParser(description="Synthetic ASL Simulator")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3)")
    parser.add_argument("--gesture", required=True, help="ASL letter (A, D, E, F, I, S, T)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--loop", action="store_true", help="Loop continuously")
    
    args = parser.parse_args()
    
    # Validate letter
    letter = args.gesture.upper()
    if letter not in ASL_PATTERNS:
        print(f"ERROR: Unknown ASL letter '{letter}'")
        print(f"Available: {', '.join(sorted(ASL_PATTERNS.keys()))}")
        return
    
    # Show pattern
    pattern = ASL_PATTERNS[letter]
    print(f"\nASL '{letter}' Pattern (0=straight, 1=bent):")
    for finger, value in pattern.items():
        state = "BENT" if value > 0.5 else "straight"
        print(f"  {finger:8s}: {value:.1f} ({state})")
    
    # Run simulator
    simulate_asl(args.port, args.baud, letter, args.loop)


if __name__ == "__main__":
    main()

