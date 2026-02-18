"""
Data Validation and Quality Check Tool
=======================================

Validates collected glove data to ensure quality before training.

Checks:
- File count and naming
- Sample counts per file
- Sensor value ranges
- Missing or corrupt data
- Label distribution
- Data consistency

Usage:
    python validate_data.py --data data/my_glove_data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys


class DataValidator:
    """Validates collected glove data"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.issues = []
        self.warnings = []
        self.stats = {}
        
    def validate(self):
        """Run all validation checks"""
        print("\n🔍 Validating data...")
        print(f"📂 Directory: {self.data_dir}\n")
        
        # Check 1: Directory exists
        if not self.data_dir.exists():
            self.issues.append(f"Directory does not exist: {self.data_dir}")
            return False
        
        # Check 2: Find CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            self.issues.append("No CSV files found")
            return False
        
        print(f"✅ Found {len(csv_files)} CSV files\n")
        
        # Check 3: Validate each file
        valid_files = 0
        total_samples = 0
        labels_count = {}
        
        for csv_file in csv_files:
            is_valid, samples, label = self._validate_file(csv_file)
            if is_valid:
                valid_files += 1
                total_samples += samples
                labels_count[label] = labels_count.get(label, 0) + 1
        
        print(f"\n✅ Valid files: {valid_files}/{len(csv_files)}")
        print(f"✅ Total samples: {total_samples}")
        
        # Check 4: Label distribution
        print(f"\n📊 Label Distribution:")
        expected_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'I', 'K', 'O', 'S', 'T', 'V', 'W', 'X', 'Y']
        
        for letter in sorted(labels_count.keys()):
            count = labels_count[letter]
            status = "✅" if count >= 5 else "⚠️ "
            print(f"   {status} {letter}: {count} files")
        
        # Check for missing letters
        missing_letters = set(expected_letters) - set(labels_count.keys())
        if missing_letters:
            self.warnings.append(f"Missing letters: {', '.join(sorted(missing_letters))}")
        
        # Check for letters with too few samples
        few_samples = [l for l, c in labels_count.items() if c < 5]
        if few_samples:
            self.warnings.append(f"Letters with <5 files: {', '.join(sorted(few_samples))}")
        
        # Store stats
        self.stats = {
            'total_files': len(csv_files),
            'valid_files': valid_files,
            'total_samples': total_samples,
            'labels': labels_count
        }
        
        # Print summary
        self._print_summary()
        
        return len(self.issues) == 0
    
    def _validate_file(self, csv_file):
        """Validate a single CSV file"""
        try:
            df = pd.read_csv(csv_file)
            
            # Check required columns
            required_cols = ['timestamp', 'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.issues.append(f"{csv_file.name}: Missing columns {missing_cols}")
                return False, 0, None
            
            # Check sample count
            sample_count = len(df)
            if sample_count < 50:
                self.warnings.append(f"{csv_file.name}: Only {sample_count} samples (expected ~150)")
            
            # Check for NaN values
            if df.isnull().any().any():
                self.issues.append(f"{csv_file.name}: Contains NaN values")
                return False, 0, None
            
            # Check sensor value ranges
            for i in range(1, 6):
                col = f'flex_{i}'
                values = df[col].values
                
                # Check if all values are the same (sensor stuck)
                if np.all(values == values[0]):
                    self.warnings.append(f"{csv_file.name}: Sensor {i} has constant values (may be stuck)")
                
                # Check for unrealistic ranges
                if values.min() < 0 or values.max() > 1023:
                    self.warnings.append(f"{csv_file.name}: Sensor {i} has out-of-range values")
            
            # Get label
            label = df['label'].iloc[0]
            
            return True, sample_count, label
            
        except Exception as e:
            self.issues.append(f"{csv_file.name}: Error reading file - {e}")
            return False, 0, None
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("📋 VALIDATION SUMMARY")
        print("="*60)
        
        if self.issues:
            print("\n❌ ISSUES:")
            for issue in self.issues:
                print(f"   - {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✅ All checks passed! Data looks good.")
        elif not self.issues:
            print("\n✅ No critical issues found (warnings are non-blocking)")
        else:
            print("\n❌ Please fix the issues above before training")
        
        print("\n" + "="*60)
    
    def generate_report(self, output_file='data_validation_report.txt'):
        """Generate a text report"""
        with open(output_file, 'w') as f:
            f.write("Data Validation Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Directory: {self.data_dir}\n")
            f.write(f"Total files: {self.stats.get('total_files', 0)}\n")
            f.write(f"Valid files: {self.stats.get('valid_files', 0)}\n")
            f.write(f"Total samples: {self.stats.get('total_samples', 0)}\n\n")
            
            f.write("Label Distribution:\n")
            for label, count in sorted(self.stats.get('labels', {}).items()):
                f.write(f"  {label}: {count} files\n")
            
            f.write("\nIssues:\n")
            if self.issues:
                for issue in self.issues:
                    f.write(f"  - {issue}\n")
            else:
                f.write("  None\n")
            
            f.write("\nWarnings:\n")
            if self.warnings:
                for warning in self.warnings:
                    f.write(f"  - {warning}\n")
            else:
                f.write("  None\n")
        
        print(f"\n💾 Report saved to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Validate collected glove data'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Directory containing collected CSV data'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Output file for validation report'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = DataValidator(args.data)
    
    # Run validation
    is_valid = validator.validate()
    
    # Generate report if requested
    if args.report:
        validator.generate_report(args.report)
    
    # Exit code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

