# Windowed Feature Models

This directory contains trained models using windowed features with temporal dynamics.

## Models

Trained with:
- **Window size**: 0.5 seconds (50 samples at 100 Hz)
- **Features**: Mean, Std, Min, Max, Velocity, Acceleration per sensor (30 features total)
- **Preprocessing**: Butterworth low-pass filter (10 Hz cutoff) for noise reduction
- **Validation**: Leave-One-User-Out cross-validation

## Performance (LOUO)

- `gb_model_0.5s.pkl`: Gradient Boosting - **68.61%** accuracy (±9.71%)
- `rf_model_0.5s.pkl`: Random Forest - **68.31%** accuracy (±9.20%)
- `mlp_model_0.5s.pkl`: MLP Neural Network - Lower LOUO performance

## Files

- `*_model_0.5s.pkl`: Trained model files (gitignored - too large)
- `scaler.pkl`: StandardScaler for feature normalization (gitignored)
- `label_encoder.pkl`: Label encoder for gesture names (gitignored)

## Results

Compared to baseline (56.11% single-point):
- **+12.50% improvement** with windowed features
- **7.5x better than random guessing** (9% → 68.61%)

Training date: December 2025

