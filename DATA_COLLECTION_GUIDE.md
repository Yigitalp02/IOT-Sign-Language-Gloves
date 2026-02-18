# 📋 Data Collection Guide for IoT Sign Language Glove

## 🎯 Goal
Collect high-quality sensor data from your smart glove to train an accurate ASL recognition model.

---

## 🛠️ **Setup**

### 1. **Hardware Setup**
- ✅ Connect your glove to the computer via USB
- ✅ Make sure all 5 flex sensors are working
- ✅ Note which COM port the glove is connected to (e.g., COM3, /dev/ttyUSB0)

### 2. **Software Setup**
```powershell
# Navigate to the project directory
cd C:\Users\Yigit\Desktop\iot-sign-language-desktop\iot-sign-glove

# Create/activate virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install pyserial scikit-learn matplotlib seaborn

# Test the glove connection
python scripts/collect_data.py
```

---

## 📊 **What Data to Collect**

### **Target Letters** (15 letters that work well with flex sensors)
```
A, B, C, D, E, F, I, K, O, S, T, V, W, X, Y
```

### **Why These Letters?**
- These letters have **distinct finger bending patterns**
- They are **distinguishable** using only flex sensors (no IMU needed)
- They form **meaningful words** (e.g., DEAF, TAXI, WAVY, DIVA, SODA)

### **How Much Data?**
- **Minimum**: 5 repetitions per letter × 15 letters = **75 recordings**
- **Recommended**: 10-20 repetitions per letter = **150-300 recordings**
- **Each recording**: ~150 samples (3 seconds at 50Hz)

---

## 🎬 **Collection Process**

### **Step 1: Test Connection**
```powershell
python scripts/collect_data.py
```
- Choose option `1` (Test connection)
- Verify sensor values are changing when you move your fingers
- Expected format: `Flex: [123, 456, 789, 012, 345]`

### **Step 2: Collect Single Letter (Practice)**
- Choose option `2` (Collect single letter)
- Try collecting letter `A` first
- Hold the sign **steady** for 3 seconds
- Review the saved data

### **Step 3: Collect Full Dataset**
- Choose option `3` (Collect full dataset)
- Enter number of repetitions (start with 5, then do more later)
- Follow the prompts for each letter

---

## 💡 **Tips for High-Quality Data**

### ✅ **DO:**
- **Hold the sign steady** during recording (avoid shaking)
- **Rest between letters** to avoid fatigue
- **Vary hand position slightly** between repetitions (makes model more robust)
- **Record in different sessions** (different times of day, hand conditions)
- **Check sensor values** - make sure they're not stuck at 0 or maxed out

### ❌ **DON'T:**
- Don't move your hand during recording
- Don't rush - take your time between letters
- Don't record when tired or uncomfortable
- Don't ignore warnings about invalid data

---

## 📁 **Data Organization**

Your data will be saved as:
```
data/my_glove_data/
  ├── A_rep1_20260218_143022.csv
  ├── A_rep2_20260218_143045.csv
  ├── B_rep1_20260218_143112.csv
  └── ...
```

Each CSV file contains:
- `timestamp`: When the sample was recorded
- `flex_1` to `flex_5`: Sensor values for each finger
- `label`: The letter being signed
- `repetition`: Which repetition number

---

## 🔍 **Data Quality Checks**

### **After collecting, verify:**
1. **File count**: Should have `15 letters × N repetitions` files
2. **Sample count**: Each file should have ~150 samples
3. **Sensor ranges**: Values should vary (not all the same)
4. **No errors**: No files with error messages or empty data

### **Quick validation:**
```python
import pandas as pd
from pathlib import Path

# Load and check data
data_dir = Path('data/my_glove_data')
csv_files = list(data_dir.glob('*.csv'))

print(f"Total files: {len(csv_files)}")

for csv_file in csv_files[:5]:  # Check first 5 files
    df = pd.read_csv(csv_file)
    print(f"{csv_file.name}: {len(df)} samples")
```

---

## 🚀 **Next Steps: Training**

Once you've collected enough data:

```powershell
# Train the model
python scripts/train_model.py --data data/my_glove_data --name my_glove_v1

# This will:
# 1. Load all your CSV files
# 2. Train a Random Forest classifier
# 3. Evaluate accuracy
# 4. Save the model to models/
```

### **Expected Results:**
- **Good accuracy**: 85-95% on test set
- **Fast inference**: <10ms per prediction
- **Lightweight**: ~1-5MB model file

---

## 🐛 **Troubleshooting**

### **Problem: "No serial ports found"**
- ✅ Check USB connection
- ✅ Install USB-to-Serial drivers if needed
- ✅ On Windows: Check Device Manager for COM ports

### **Problem: "Invalid data format"**
- ✅ Check your Arduino code's serial output format
- ✅ Update the `read_sensor_line()` function in `collect_data.py`
- ✅ Expected format: `flex1,flex2,flex3,flex4,flex5`

### **Problem: Sensor values not changing**
- ✅ Check flex sensor connections
- ✅ Verify sensor wiring (voltage divider circuit)
- ✅ Test each sensor individually

### **Problem: Accuracy is low after training**
- ✅ Collect more data (at least 10 reps per letter)
- ✅ Check for mislabeled data
- ✅ Ensure consistent hand positions
- ✅ Try different model hyperparameters

---

## 📈 **Iterative Improvement**

### **Round 1: Initial Model**
1. Collect 5 reps × 15 letters = 75 recordings
2. Train baseline model
3. Test accuracy

### **Round 2: Targeted Improvement**
1. Check confusion matrix - which letters are confused?
2. Collect **more data** for confused letters
3. Retrain

### **Round 3: Real-World Testing**
1. Test with real users
2. Collect edge cases (unusual hand sizes, positions)
3. Retrain with augmented dataset

---

## 🎯 **Success Criteria**

Your model is ready when:
- ✅ **Test accuracy** ≥ 90%
- ✅ **Consistent predictions** for each letter
- ✅ **Fast inference** (<50ms on mobile)
- ✅ **Works in real-time** during live testing

---

## 📞 **Need Help?**

If you encounter issues:
1. Check the console output for specific error messages
2. Review the collected CSV files manually
3. Test the glove connection in isolation
4. Adjust the data collection script as needed

**Good luck! 🍀**

