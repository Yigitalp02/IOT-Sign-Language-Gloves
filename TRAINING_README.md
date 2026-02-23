# ASL Glove Model Training

## Quick Start

```bash
# From iot-sign-glove directory
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
python scripts/train_model.py --input data/Data/glove_data_NORMALIZED_B_A_C_D_E_F_I_K_O_S_T_V_W_X_Y_2026-02-23-10-45-28.csv
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | `data/Data/glove_data_NORMALIZED_*.csv` | Path to normalized CSV |
| `--output`, `-o` | `models/rf_asl_15letters_normalized.pkl` | Output model path |
| `--window-size` | 50 | Samples per window (50 = 1 sec at 50Hz) |
| `--stride` | 25 | Stride between windows (50% overlap) |
| `--tune` | off | Run hyperparameter tuning (slower, may improve accuracy) |
| `--seed` | 42 | Random seed |

## Data Format

CSV must have columns: `label`, `ch0_norm`, `ch1_norm`, `ch2_norm`, `ch3_norm`, `ch4_norm`

Values should be normalized 0-1 (0 = straight, 1 = bent).

## Model Performance (Lab Data, Feb 2026)

- **Test accuracy**: ~95%
- **Window size**: 50 samples (1 second at 50Hz) - good for real-time
- **Classes**: A, B, C, D, E, F, I, K, O, S, T, V, W, X, Y

## Deploy to API

1. Copy model to server:
   ```bash
   scp models/rf_asl_15letters_normalized.pkl user@server:/opt/stack/ai-models/
   ```

2. Set `MODEL_PATH=/models/rf_asl_15letters_normalized.pkl` in Docker environment

3. Restart API container

4. In desktop app `.env`, add:
   ```
   VITE_USE_NORMALIZED_MODEL=true
   ```

5. Calibrate the glove (per user) before predicting

## Local Model (Development)

Use the local model for development without hitting the cloud API:

1. Start the local server:
   ```bash
   cd iot-sign-glove
   pip install -r requirements.txt  # includes fastapi, uvicorn
   python scripts/serve_local_model.py
   ```

2. In the desktop app, enable "Use local model (dev)" switch

3. Predictions go to http://localhost:8765 instead of the cloud API
