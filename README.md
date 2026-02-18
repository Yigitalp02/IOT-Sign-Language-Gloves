# IoT Sign Glove - Data Collection and Model Training

Tools for collecting sensor data from smart glove hardware and training machine learning models for ASL recognition.

**Version**: 1.0.0  
**Last Updated**: February 2026

---

## Overview

This directory contains Python scripts and utilities for:
- Collecting labeled sensor data from Arduino-based glove
- Validating data quality
- Training Random Forest and Gradient Boosting models
- Evaluating model performance
- Exporting models for deployment

---

## Quick Start

### 1. Setup Environment

```powershell
# Navigate to this directory
cd C:\Users\Yigit\Desktop\iot-sign-language-desktop\iot-sign-glove

# Create virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Data from Glove

```powershell
python scripts/collect_data.py
```

Follow interactive prompts to:
- Test serial connection
- Collect individual letters
- Collect full dataset (all 15 letters)

### 3. Validate Data Quality

```powershell
python scripts/validate_data.py --data data/my_glove_data
```

### 4. Train Model

```powershell
python scripts/train_model.py --data data/my_glove_data --name my_glove_v1
```

---

## Directory Structure

```
iot-sign-glove/
├── data/                           # Collected sensor data
│   ├── my_glove_data/             # Your glove recordings
│   │   ├── A_rep1_20260218.csv
│   │   ├── A_rep2_20260218.csv
│   │   └── ...
│   └── ASL_Labeled/               # External datasets (optional)
│
├── models/                         # Trained models
│   ├── my_glove_v1_20260218.pkl
│   ├── my_glove_v1_scaler_20260218.pkl
│   └── confusion_matrix.png
│
├── scripts/                        # Python scripts
│   ├── collect_data.py            # Data collection from serial
│   ├── train_model.py             # Model training
│   └── validate_data.py           # Data validation
│
├── DATA_COLLECTION_GUIDE.md       # Detailed collection guide
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Supported ASL Letters

The following 15 letters work well with flex sensors (no IMU needed):

```
A, B, C, D, E, F, I, K, O, S, T, V, W, X, Y
```

These letters:
- Have distinct finger bending patterns
- Form meaningful words (DEAF, TAXI, WAVY, DIVA, SODA, etc.)
- Achieve high accuracy with simple ML models

---

## Data Collection

### Recommended Data Volume

| Level | Reps/Letter | Total Recordings | Estimated Time |
|-------|-------------|------------------|----------------|
| Minimum | 5 | 75 | ~20 min |
| Recommended | 10 | 150 | ~40 min |
| Optimal | 20+ | 300+ | ~80 min |

### Data Format

Each CSV file contains:
- `timestamp`: Unix timestamp (seconds)
- `flex_1` to `flex_5`: Sensor values (0-1023)
- `label`: ASL letter (A-Y)
- `repetition`: Repetition number (1-N)

Example:
```csv
timestamp,flex_1,flex_2,flex_3,flex_4,flex_5,label,repetition
1708268123.5,120,580,720,770,580,A,1
1708268123.52,120,578,718,768,579,A,1
```

### Collection Best Practices

1. **Calibrate first**: Record baseline (open hand) and max bend (closed fist)
2. **Hold steady**: Maintain position for full recording duration
3. **Add variation**: Tilt wrist, vary finger tightness across repetitions
4. **Check quality**: Validate data before training
5. **Collect multiple sessions**: Record over different days/times

---

## Model Training

### Available Models

**1. Random Forest (Recommended)**
- Fast training and inference
- Good accuracy (85-95%)
- Small model size (~1-5MB)
- Works well with limited data
- Robust to noise

**2. Gradient Boosting (Alternative)**
- Slightly better accuracy
- Slower training
- Larger model size
- More prone to overfitting

### Training Options

```powershell
# Basic training
python scripts/train_model.py --data data/my_glove_data

# With hyperparameter tuning
python scripts/train_model.py --data data/my_glove_data --tune

# Gradient Boosting model
python scripts/train_model.py --data data/my_glove_data --model-type gb

# Custom name
python scripts/train_model.py --data data/my_glove_data --name glove_v2_optimized

# Full options
python scripts/train_model.py \
  --data data/my_glove_data \
  --model-type rf \
  --name production_model \
  --tune \
  --output models/
```

### Expected Results

A well-trained model should achieve:
- **Test Accuracy**: 85-95%
- **Cross-Validation**: 80-90%
- **Training Time**: 2-10 minutes
- **Inference Time**: <10ms per prediction
- **Model Size**: 1-5MB

---

## Validation and Quality Checks

### Validate Before Training

```powershell
python scripts/validate_data.py \
  --data data/my_glove_data \
  --report validation_report.txt
```

This checks:
- File count and naming conventions
- Sample counts per file (~150 expected)
- Sensor value ranges (0-1023)
- Missing or corrupt data
- Label distribution balance
- Data consistency across recordings

### Validation Output

```
Validation Report
-----------------
Total files: 75
Total samples: 11,250
Letters found: 15
Samples per letter: 750 (min: 720, max: 780)
Sensor ranges: OK (0-1023)
Missing values: 0
Corrupt files: 0
Data quality: PASS
```

---

## Troubleshooting

### "No serial ports found"
**Problem**: Arduino not detected  
**Solution**:
- Check USB connection
- Install CH340 driver (Arduino Nano)
- Verify port in Device Manager (Windows) or `ls /dev/tty*` (Linux/Mac)
- Close other programs using serial port

### "Invalid data format"
**Problem**: Serial data doesn't match expected format  
**Solution**:
- Check Arduino sketch output format
- Expected: `flex1,flex2,flex3,flex4,flex5\n`
- Update `read_sensor_line()` in `collect_data.py` if needed

### "Low model accuracy"
**Problem**: Test accuracy below 70%  
**Solution**:
- Collect more data (aim for 10+ reps per letter)
- Check for mislabeled recordings
- Ensure consistent hand positions
- Review confusion matrix for problematic letters
- Try hyperparameter tuning with `--tune` flag

### "Out of memory during training"
**Problem**: Training crashes with memory error  
**Solution**:
- Reduce number of samples per recording
- Use fewer repetitions initially
- Close other applications
- Try Gradient Boosting with fewer estimators

---

## Detailed Guides

- **DATA_COLLECTION_GUIDE.md**: Complete data collection workflow with tips
- **Script help**: `python scripts/collect_data.py --help`
- **Training help**: `python scripts/train_model.py --help`
- **Validation help**: `python scripts/validate_data.py --help`

---

## Deployment

### Deploy Model to API Server

Once trained, deploy the model:

```powershell
# Copy model files to API server
scp models/my_glove_v1_20260218.pkl user@api.server:/opt/stack/asl-ml-server/models/
scp models/my_glove_v1_scaler_20260218.pkl user@api.server:/opt/stack/asl-ml-server/models/

# SSH to server and restart API
ssh user@api.server
cd /opt/stack/asl-ml-server
sudo docker compose restart asl-ml-api
```

### Test Deployed Model

```bash
# Test prediction with deployed model
curl -X POST https://api.ybilgin.com/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"flex_sensors": [[512, 678, 345, 890, 234]]}'
```

---

## Script Reference

### collect_data.py

Interactive data collection from serial port.

```powershell
python scripts/collect_data.py [OPTIONS]

Options:
  --port COM3          # Specify COM port
  --baud 115200        # Baud rate (default: 115200)
  --samples 150        # Samples per recording (default: 150)
  --output data/       # Output directory
```

### train_model.py

Train ML model from collected data.

```powershell
python scripts/train_model.py [OPTIONS]

Options:
  --data PATH          # Data directory (required)
  --model-type rf|gb   # Random Forest or Gradient Boosting
  --name NAME          # Model name
  --tune               # Enable hyperparameter tuning
  --output PATH        # Output directory for model
```

### validate_data.py

Validate data quality before training.

```powershell
python scripts/validate_data.py [OPTIONS]

Options:
  --data PATH          # Data directory (required)
  --report FILE        # Save validation report to file
  --verbose            # Show detailed output
```

---

## Next Steps

1. **Tomorrow**: Connect glove and test serial connection
2. Collect initial dataset (5-10 reps per letter)
3. Validate data quality
4. Train baseline model
5. Test in desktop/mobile app
6. Iterate: Collect more data for confused letters
7. Deploy improved model to production

---

## Python Dependencies

Key dependencies (see `requirements.txt` for full list):
- `pyserial` - Serial port communication
- `scikit-learn` - Machine learning models
- `xgboost` - Gradient boosting (optional)
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting confusion matrices
- `tqdm` - Progress bars

Install all:
```powershell
pip install -r requirements.txt
```

---

## Data Privacy

All collected data stays local on your machine. No data is automatically uploaded to cloud services. When deploying models, only the trained model files (not raw data) are transferred to the API server.

---

## Related Projects

- **Desktop App**: `../` - Desktop application for data collection
- **Mobile App**: `../mobile/` - Mobile companion app
- **API Server**: `../ASL-ML-Inference-API/` - Cloud ML inference server

---

## Documentation

- **DATA_COLLECTION_GUIDE.md**: Detailed collection workflow
- **PROJECT_STATE.md**: Complete project overview
- **Script help**: Use `--help` flag with any script

---

## Academic Context

This toolset is part of a Computer Science graduation project focused on:
- IoT sensor integration
- Machine learning for gesture recognition
- Data collection and quality assurance
- Model training and evaluation
- Edge computing and embedded systems

---

## License

MIT License - Part of Computer Science Graduation Project

**Author**: Yigit Alp Bilgin  
**Year**: 2026

For questions or support, refer to PROJECT_STATE.md or contact the project team.
