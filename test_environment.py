"""
Quick Environment Test
======================

Tests that all required dependencies are installed and working.

Usage:
    python test_environment.py
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - NOT INSTALLED")
        print(f"   Install with: pip install {package_name or module_name}")
        return False

def main():
    """Test all dependencies"""
    print("\n" + "="*60)
    print("🧪 Testing Environment")
    print("="*60)
    
    print(f"\n📍 Python Version: {sys.version}")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("⚠️  Warning: Python 3.7+ recommended")
    else:
        print("✅ Python version OK")
    
    print("\n📦 Checking dependencies:\n")
    
    # Test all required packages
    all_ok = True
    all_ok &= test_import('numpy')
    all_ok &= test_import('pandas')
    all_ok &= test_import('sklearn', 'scikit-learn')
    all_ok &= test_import('serial', 'pyserial')
    all_ok &= test_import('matplotlib')
    all_ok &= test_import('seaborn')
    
    print("\n" + "="*60)
    
    if all_ok:
        print("✅ All dependencies installed!")
        print("\n🎉 Ready to collect data and train models!")
        print("\nNext steps:")
        print("  1. Connect your glove via USB")
        print("  2. Run: python scripts/collect_data.py")
    else:
        print("❌ Some dependencies are missing")
        print("\n📦 Install all dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

