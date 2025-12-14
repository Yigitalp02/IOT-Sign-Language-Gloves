# Trained Models

This directory contains trained machine learning models for gesture recognition.

## Files (gitignored due to size)

- `rf_model.pkl` (156 MB) - Random Forest classifier
- `mlp_model.pkl` (0.27 MB) - Multi-Layer Perceptron neural network
- `scaler.pkl` (<1 MB) - Feature scaler for normalization
- `label_encoder.pkl` (<1 MB) - Label encoder for gesture classes

## Performance

| Model | Accuracy (80/20 split) | LOUO Accuracy |
|-------|------------------------|---------------|
| Random Forest | 82.00% | 56.11% ± 7.81% |
| MLP | 71.13% | Not tested |

## Regenerate Models

To train models from scratch:

```bash
cd iot-sign-glove
python scripts/train_with_professor_data.py
```

This will:
1. Load the professor's dataset (1.26M samples)
2. Train Random Forest and MLP models
3. Perform Leave-One-User-Out cross-validation
4. Save models to this directory

## Model Details

### Random Forest
- 100 trees
- 5 input features (normalized sensor channels)
- 11 output classes (gestures)
- Training time: ~2 minutes

### MLP
- Architecture: 5 → 128 → 64 → 32 → 11
- Activation: ReLU
- Optimizer: Adam
- Training time: ~5 minutes

## Usage

```python
import joblib

# Load models
rf_model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Predict
X_scaled = scaler.transform(X)
y_pred = rf_model.predict(X_scaled)
gesture = label_encoder.inverse_transform(y_pred)
```



