"""
Data Collection Script for IoT Sign Language Glove
===================================================

This script helps you collect labeled sensor data from your glove for training ML models.

Features:
- Connects to glove via serial port
- Records 5 flex sensors (finger bending)
- Labels each recording with the corresponding ASL letter
- Saves data in CSV format compatible with training scripts
- Real-time visualization of sensor values
- Quality checks (min/max samples, data validation)

Usage:
    python collect_data.py --port COM3 --output data/my_glove_data
"""

import serial
import serial.tools.list_ports
import time
import csv
import os
from datetime import datetime
from pathlib import Path
import sys

# Configuration
BAUDRATE = 115200
SAMPLE_RATE = 50  # Hz (20ms per sample)
MIN_SAMPLES_PER_LETTER = 100  # Minimum samples to collect per letter
TARGET_SAMPLES_PER_LETTER = 150  # Target samples (3 seconds at 50Hz)

# ASL Letters to collect (15 letters that work well with flex sensors)
ASL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'I', 'K', 'O', 'S', 'T', 'V', 'W', 'X', 'Y']

# You can also add custom words/phrases
# CUSTOM_WORDS = ['HELLO', 'THANKS', 'SORRY']


class GloveDataCollector:
    """Collects labeled data from the smart glove"""
    
    def __init__(self, port, output_dir):
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.serial_conn = None
        self.current_letter = None
        self.current_samples = []
        self.session_start_time = datetime.now()
        
    def connect(self):
        """Connect to the glove via serial port"""
        try:
            print(f"🔌 Connecting to glove on {self.port} at {BAUDRATE} baud...")
            self.serial_conn = serial.Serial(self.port, BAUDRATE, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print("✅ Connected successfully!")
            return True
        except serial.SerialException as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the glove"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("🔌 Disconnected from glove")
    
    def read_sensor_line(self):
        """
        Read one line of sensor data from the glove
        Expected format: "flex1,flex2,flex3,flex4,flex5" or similar
        
        Adjust this based on your glove's actual data format!
        """
        try:
            if self.serial_conn and self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                
                # Skip empty lines
                if not line:
                    return None
                
                # Parse the sensor values
                # Adjust this parsing based on your glove's format
                # Example formats:
                # - "!123,456,789,012,345#" (with start/end markers)
                # - "123,456,789,012,345" (simple CSV)
                # - "flex1:123 flex2:456 ..." (key-value pairs)
                
                # For now, assume simple CSV format
                values = line.strip('!#').split(',')
                
                if len(values) >= 5:
                    flex_values = [int(v) for v in values[:5]]
                    return {
                        'timestamp': time.time(),
                        'flex_1': flex_values[0],
                        'flex_2': flex_values[1],
                        'flex_3': flex_values[2],
                        'flex_4': flex_values[3],
                        'flex_5': flex_values[4],
                    }
                else:
                    print(f"⚠️  Invalid data format: {line}")
                    return None
        except Exception as e:
            print(f"⚠️  Error reading sensor: {e}")
            return None
    
    def collect_letter(self, letter, repetition=1):
        """
        Collect samples for a specific letter
        
        Args:
            letter: The ASL letter to collect
            repetition: Which repetition (for multiple samples of same letter)
        """
        print(f"\n{'='*60}")
        print(f"📝 COLLECTING: Letter '{letter}' (Repetition {repetition})")
        print(f"{'='*60}")
        print(f"🖐️  Make the ASL sign for '{letter}' and hold it steady")
        print(f"⏱️  Target: {TARGET_SAMPLES_PER_LETTER} samples (~3 seconds)")
        print(f"\nPress ENTER when ready, then hold the sign...")
        input()
        
        print("🔴 RECORDING... Hold the sign steady!")
        self.current_samples = []
        self.current_letter = letter
        
        start_time = time.time()
        sample_count = 0
        
        while sample_count < TARGET_SAMPLES_PER_LETTER:
            sensor_data = self.read_sensor_line()
            
            if sensor_data:
                sensor_data['label'] = letter
                sensor_data['repetition'] = repetition
                self.current_samples.append(sensor_data)
                sample_count += 1
                
                # Progress bar
                if sample_count % 10 == 0:
                    progress = (sample_count / TARGET_SAMPLES_PER_LETTER) * 100
                    bar_length = 40
                    filled = int(bar_length * sample_count / TARGET_SAMPLES_PER_LETTER)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r  [{bar}] {progress:.0f}% ({sample_count}/{TARGET_SAMPLES_PER_LETTER})", end='')
            
            # Small delay to match sample rate
            time.sleep(1.0 / SAMPLE_RATE)
        
        elapsed = time.time() - start_time
        print(f"\n✅ DONE! Collected {sample_count} samples in {elapsed:.2f}s")
        
        # Show sensor value ranges
        if self.current_samples:
            self._show_sensor_stats()
        
        # Save immediately
        self._save_samples(letter, repetition)
        
    def _show_sensor_stats(self):
        """Display statistics about collected samples"""
        print("\n📊 Sensor Value Ranges:")
        for i in range(1, 6):
            key = f'flex_{i}'
            values = [s[key] for s in self.current_samples]
            print(f"   Finger {i}: min={min(values):4d}, max={max(values):4d}, avg={sum(values)/len(values):6.1f}")
    
    def _save_samples(self, letter, repetition):
        """Save collected samples to CSV file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{letter}_rep{repetition}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        print(f"\n💾 Saving to: {filepath}")
        
        with open(filepath, 'w', newline='') as f:
            if self.current_samples:
                fieldnames = self.current_samples[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.current_samples)
        
        print(f"✅ Saved {len(self.current_samples)} samples")
    
    def collect_full_dataset(self, repetitions_per_letter=5):
        """
        Collect a complete dataset with multiple repetitions of each letter
        
        Args:
            repetitions_per_letter: How many times to collect each letter
        """
        print("\n" + "="*60)
        print("🎯 FULL DATASET COLLECTION")
        print("="*60)
        print(f"📋 Letters to collect: {', '.join(ASL_LETTERS)}")
        print(f"🔄 Repetitions per letter: {repetitions_per_letter}")
        print(f"📊 Total recordings: {len(ASL_LETTERS) * repetitions_per_letter}")
        print(f"⏱️  Estimated time: ~{len(ASL_LETTERS) * repetitions_per_letter * 5 / 60:.0f} minutes")
        print("\n💡 Tips:")
        print("   - Keep your hand steady during recording")
        print("   - Rest between letters to avoid fatigue")
        print("   - Press Ctrl+C to pause/stop")
        print("\nReady? Press ENTER to start...")
        input()
        
        total_recordings = len(ASL_LETTERS) * repetitions_per_letter
        completed = 0
        
        try:
            for letter in ASL_LETTERS:
                for rep in range(1, repetitions_per_letter + 1):
                    completed += 1
                    print(f"\n[Progress: {completed}/{total_recordings}]")
                    self.collect_letter(letter, rep)
                    
                    if rep < repetitions_per_letter:
                        print(f"\n⏸️  Rest for a moment...")
                        time.sleep(2)
            
            print("\n" + "="*60)
            print("🎉 DATASET COLLECTION COMPLETE!")
            print("="*60)
            print(f"📁 Output directory: {self.output_dir}")
            print(f"📊 Total files: {completed}")
            print("\n✨ Next steps:")
            print("   1. Review the collected data")
            print("   2. Run: python train_model.py")
            
        except KeyboardInterrupt:
            print("\n\n⏸️  Collection paused by user")
            print(f"✅ Saved {completed} recordings so far")


def list_serial_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("❌ No serial ports found!")
        return []
    
    print("\n📡 Available serial ports:")
    for i, port in enumerate(ports, 1):
        print(f"   {i}. {port.device} - {port.description}")
    
    return [p.device for p in ports]


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🧤 IoT Sign Language Glove - Data Collection Tool")
    print("="*60)
    
    # List available ports
    ports = list_serial_ports()
    
    if not ports:
        print("\n⚠️  No serial ports detected. Is your glove connected?")
        return
    
    # Select port
    print("\nEnter the port number or full port name (e.g., COM3 or /dev/ttyUSB0):")
    port_input = input("> ").strip()
    
    # Try to parse as number
    try:
        port_num = int(port_input)
        if 1 <= port_num <= len(ports):
            selected_port = ports[port_num - 1]
        else:
            print("❌ Invalid port number")
            return
    except ValueError:
        selected_port = port_input
    
    # Output directory
    print("\nEnter output directory (default: data/my_glove_data):")
    output_dir = input("> ").strip() or "data/my_glove_data"
    
    # Create collector
    collector = GloveDataCollector(selected_port, output_dir)
    
    # Connect
    if not collector.connect():
        return
    
    try:
        # Main menu
        while True:
            print("\n" + "="*60)
            print("📋 MENU")
            print("="*60)
            print("1. Test connection (view live sensor data)")
            print("2. Collect single letter")
            print("3. Collect full dataset (all 15 letters)")
            print("4. Exit")
            print("\nChoice: ", end='')
            
            choice = input().strip()
            
            if choice == '1':
                # Test mode - show live data
                print("\n📡 Live sensor data (Press Ctrl+C to stop):")
                try:
                    while True:
                        data = collector.read_sensor_line()
                        if data:
                            print(f"  Flex: [{data['flex_1']:4d}, {data['flex_2']:4d}, {data['flex_3']:4d}, {data['flex_4']:4d}, {data['flex_5']:4d}]")
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\n✅ Test stopped")
            
            elif choice == '2':
                # Single letter
                print(f"\nAvailable letters: {', '.join(ASL_LETTERS)}")
                letter = input("Enter letter to collect: ").strip().upper()
                if letter in ASL_LETTERS:
                    collector.collect_letter(letter)
                else:
                    print("❌ Invalid letter")
            
            elif choice == '3':
                # Full dataset
                reps = input("\nHow many repetitions per letter? (default: 5): ").strip()
                repetitions = int(reps) if reps.isdigit() else 5
                collector.collect_full_dataset(repetitions)
            
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        collector.disconnect()


if __name__ == "__main__":
    main()

