# 🧤 IoT Sign Glove - Data Collection & Model Training

This directory contains tools for collecting sensor data from your smart glove and training machine learning models for ASL recognition.

---

## 🚀 **Quick Start**

### **1. Setup Environment**

```powershell
# Navigate to this directory
cd C:\Users\Yigit\Desktop\iot-sign-language-desktop\iot-sign-glove

# Create virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### **2. Collect Data from Your Glove**

```powershell
python scripts/collect_data.py
```

Follow the interactive prompts to:
- Test connection
- Collect individual letters
- Collect full dataset (all 15 letters)

### **3. Validate Data Quality**

```powershell
python scripts/validate_data.py --data data/my_glove_data
```

### **4. Train Model**

```powershell
python scripts/train_model.py --data data/my_glove_data --name my_glove_v1
```

---

## 📁 **Directory Structure**

```
iot-sign-glove/
├── data/                           # Collected sensor data
│   ├── my_glove_data/             # Your new glove data
│   │   ├── A_rep1_20260218.csv
│   │   ├── A_rep2_20260218.csv
│   │   └── ...
│   └── ASL_Labeled/               # External datasets (if any)
│
├── models/                         # Trained models
│   ├── my_glove_v1_20260218.pkl
│   ├── my_glove_v1_scaler_20260218.pkl
│   └── ...
│
├── scripts/                        # Data collection & training scripts
│   ├── collect_data.py            # 📊 Collect data from glove
│   ├── train_model.py             # 🎯 Train ML model
│   └── validate_data.py           # ✅ Validate data quality
│
├── DATA_COLLECTION_GUIDE.md       # 📖 Detailed collection guide
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🎯 **Supported ASL Letters**

The following 15 letters work well with flex sensors only (no IMU needed):

```
A, B, C, D, E, F, I, K, O, S, T, V, W, X, Y
```

These letters:
- Have **distinct finger bending patterns**
- Form **meaningful words** (DEAF, TAXI, WAVY, DIVA, SODA, etc.)
- Achieve **high accuracy** with simple ML models

---

## 📊 **Data Collection**

### **Recommended Data Volume**

| Level | Repetitions/Letter | Total Recordings | Estimated Time |
|-------|-------------------|------------------|----------------|
| Minimum | 5 | 75 | ~20 min |
| Recommended | 10 | 150 | ~40 min |
| Optimal | 20+ | 300+ | ~80 min |

### **Data Format**

Each CSV file contains:
- `timestamp`: Unix timestamp
- `flex_1` to `flex_5`: Sensor values (0-1023)
- `label`: ASL letter
- `repetition`: Repetition number

Example:
```csv
timestamp,flex_1,flex_2,flex_3,flex_4,flex_5,label,repetition
1708268123.5,120,580,720,770,580,A,1
1708268123.52,120,578,718,768,579,A,1
...
```

---

## 🤖 **Model Training**

### **Available Models**

1. **Random Forest** (recommended)
   - Fast training & inference
   - Good accuracy (85-95%)
   - Small model size (~1-5MB)
   - Works well with limited data

2. **Gradient Boosting** (alternative)
   - Slightly better accuracy
   - Slower training
   - Larger model size

### **Training Options**

```powershell
# Basic training (fast)
python scripts/train_model.py --data data/my_glove_data

# With hyperparameter tuning (slower, better accuracy)
python scripts/train_model.py --data data/my_glove_data --model-type rf

# Gradient Boosting
python scripts/train_model.py --data data/my_glove_data --model-type gb

# Custom name
python scripts/train_model.py --data data/my_glove_data --name glove_v2_optimized
```

### **Expected Results**

A well-trained model should achieve:
- **Test Accuracy**: 85-95%
- **Training Time**: 2-10 minutes
- **Inference Time**: <10ms per prediction
- **Model Size**: 1-5MB

---

## 🔍 **Validation & Quality Checks**

### **Validate Before Training**

```powershell
python scripts/validate_data.py --data data/my_glove_data --report validation_report.txt
```

This checks:
- ✅ File count and naming
- ✅ Sample counts (should be ~150 per file)
- ✅ Sensor value ranges
- ✅ Missing or corrupt data
- ✅ Label distribution
- ✅ Data consistency

---

## 🛠️ **Troubleshooting**

### **"No serial ports found"**
- Check USB connection
- Install USB-to-Serial drivers
- Verify port in Device Manager (Windows) or `ls /dev/tty*` (Linux/Mac)

### **"Invalid data format"**
- Check your Arduino's serial output format
- Update `read_sensor_line()` in `collect_data.py`
- Expected: `flex1,flex2,flex3,flex4,flex5`

### **Low model accuracy**
- Collect more data (aim for 10+ reps per letter)
- Check for mislabeled data
- Ensure consistent hand positions
- Review confusion matrix for confused letters

---

## 📖 **Detailed Guides**

- 📘 [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) - Complete data collection workflow
- 📗 See script help: `python scripts/collect_data.py --help`
- 📕 See training help: `python scripts/train_model.py --help`

---

## 🚢 **Deploy Model to Mobile App**

Once trained, copy the model files to your mobile app:

```powershell
# Copy model and scaler
cp models/my_glove_v1_20260218.pkl ../mobile/assets/models/
cp models/my_glove_v1_scaler_20260218.pkl ../mobile/assets/models/
```

Then update your mobile app's API to use the new model!

---

## 🎓 **Next Steps**

1. ✅ **Tomorrow**: Connect your glove and test serial connection
2. ✅ Collect initial dataset (5-10 reps per letter)
3. ✅ Validate data quality
4. ✅ Train baseline model
5. ✅ Test in mobile app
6. ✅ Iterate: Collect more data for confused letters
7. ✅ Deploy improved model

---

## 📞 **Need Help?**

Check the console output for specific error messages and refer to the troubleshooting section above.

**Happy training! 🎉**
