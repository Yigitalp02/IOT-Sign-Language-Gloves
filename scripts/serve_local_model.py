#!/usr/bin/env python3
"""
Local ASL Model Server - Development Only
Serves the trained model on localhost. Same /predict interface as the cloud API.
Use when "Use local model (dev)" switch is ON in the desktop app.

Usage:
  cd iot-sign-glove && python scripts/serve_local_model.py
  # Server runs at http://localhost:8765
"""

import time
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
# Priority: newest 97.7% model → 96% model → generic fallback
_candidates = [
    "rf_asl_15letters_normalized_97pct_seed1_feb26.pkl",
    "rf_asl_15letters_normalized_96pct_seed1.pkl",
    "rf_asl_15letters_normalized.pkl",
]
MODEL_PATH = next((MODELS_DIR / c for c in _candidates if (MODELS_DIR / c).exists()), None)

# Feature extraction — must match train_model.py exactly
# 45 features: mean, std, min, max, range × 9 channels (5 flex + 4 quaternion)
def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    # Pad legacy 5-column input with identity quaternion
    if window.shape[1] < 9:
        pad = np.zeros((window.shape[0], 9 - window.shape[1]))
        pad[:, 0] = 1.0   # qw = 1 (identity)
        window = np.hstack([window, pad])
    features = []
    for i in range(9):
        v = window[:, i].astype(float)
        v = v[~np.isnan(v)]
        std_val = float(np.std(v)) if len(v) >= 2 else 0.0
        features.extend([
            float(np.mean(v)) if len(v) > 0 else 0.0,
            std_val,
            float(np.min(v))  if len(v) > 0 else 0.0,
            float(np.max(v))  if len(v) > 0 else 0.0,
            float(np.max(v) - np.min(v)) if len(v) > 1 else 0.0,
        ])
    return np.array(features, dtype=np.float64)

app = FastAPI(title="ASL Local Model (Dev)", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
model_name = "not loaded"

@app.on_event("startup")
def startup():
    global model, model_name
    if MODEL_PATH and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        model_name = MODEL_PATH.stem
        meta_path = MODEL_PATH.with_suffix(".meta.joblib")
        acc = ""
        if meta_path.exists():
            try:
                meta = joblib.load(meta_path)
                acc = f" ({meta.get('test_accuracy', 0)*100:.1f}% test)"
            except Exception:
                pass
        print(f"Model loaded: {model_name}{acc}")
    else:
        print(f"ERROR: No model found in {MODELS_DIR}")

class SensorData(BaseModel):
    flex_sensors: Union[List[List[float]], List[float]]
    timestamp: Optional[float] = None
    device_id: Optional[str] = "desktop-app"

class PredictionResponse(BaseModel):
    letter: str
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time_ms: float
    model_name: str
    timestamp: float

@app.post("/predict", response_model=PredictionResponse)
def predict(sensor_data: SensorData):
    start = time.time()
    if model is None:
        return PredictionResponse(
            letter="?",
            confidence=0,
            all_probabilities={},
            processing_time_ms=0,
            model_name="not-loaded",
            timestamp=time.time(),
        )
    arr = np.array(sensor_data.flex_sensors)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    features = extract_features_from_window(arr).reshape(1, -1)
    pred = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        classes = model.classes_
        prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
        conf = float(max(probs))
    else:
        prob_dict = {str(pred): 1.0}
        conf = 1.0
    return PredictionResponse(
        letter=str(pred),
        confidence=conf,
        all_probabilities=prob_dict,
        processing_time_ms=(time.time() - start) * 1000,
        model_name=model_name,
        timestamp=time.time(),
    )

@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "model_name": model_name,
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting local ASL model server at http://localhost:8765")
    print("Use 'Use local model (dev)' switch in the desktop app to connect.")
    uvicorn.run(app, host="0.0.0.0", port=8765)
