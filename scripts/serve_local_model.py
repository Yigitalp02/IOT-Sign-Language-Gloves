#!/usr/bin/env python3
"""
Local ASL Model Server - Development Only
Serves the trained model on localhost. Same /predict interface as the cloud API.
Use when "Use local model (dev)" switch is ON in the desktop app.

Usage:
  cd iot-sign-glove && python scripts/serve_local_model.py
  # Server runs at http://localhost:8765
"""
import warnings
# Suppress noisy sklearn/joblib version-mismatch warnings from RF parallel prediction
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with",
    category=UserWarning,
    module="sklearn",
)

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
# Priority order: newest format first, legacy fallbacks last
_candidates = [
    "rf_asl_v2_gravity_cascade.pkl",           # v2 two-stage gravity (new)
    "rf_asl_21letters_imu_two_staged.pkl",     # professor's cascade
    "rf_asl_15letters_normalized_97pct_45feat_seed1_feb26.pkl",
    "rf_asl_21letters_imu.pkl",
    "rf_asl_15letters_normalized_97pct_seed1_feb26.pkl",
    "rf_asl_15letters_normalized_96pct_seed1.pkl",
    "rf_asl_15letters_normalized.pkl",
]
MODEL_PATH = next((MODELS_DIR / c for c in _candidates if (MODELS_DIR / c).exists()), None)

# ── Feature extraction ────────────────────────────────────────────────────────
# v2 gravity-cascade model uses:
#   Stage 1 — 25 features: mean, std, min, max, range  ×  5 flex channels
#   Stage 2 — 4 features:  mean/std of fwd_z and up_z  (yaw-invariant gravity)
# Professor two-staged model uses:
#   Stage 1 — 25 flex features
#   Stage 2 — 29 features: 25 flex + mean(qw, qx, qy, qz)
# Legacy single-model uses 45 features (5 stats × 9 channels).

def _safe_stats(v: np.ndarray):
    """Return (mean, std, min, max, range) handling empty / single-element arrays."""
    v = v[~np.isnan(v)].astype(float)
    if len(v) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (float(np.mean(v)),
            float(np.std(v)) if len(v) >= 2 else 0.0,
            float(np.min(v)),
            float(np.max(v)),
            float(np.max(v) - np.min(v)) if len(v) > 1 else 0.0)

def extract_features_stage1(window: np.ndarray) -> np.ndarray:
    """25 features — flex channels 0-4 only. IMU is ignored."""
    feats = []
    for i in range(5):
        feats.extend(_safe_stats(window[:, i]))
    return np.array(feats, dtype=np.float64)

def extract_gravity_features(window: np.ndarray) -> np.ndarray:
    """
    6 yaw-invariant gravity features used by v2 Stage 2.

    fwd_z   = 2*(qx*qz + qw*qy)   z-component of hand's forward axis
    up_z    = 1 - 2*(qx^2 + qy^2) z-component of hand's up axis
    right_z = 2*(qy*qz - qw*qx)   z-component of hand's right axis (wrist roll)

    All three are invariant to yaw — only tilt relative to gravity matters.
    right_z is critical for separating P vs Q vs L (differ in wrist roll).
    """
    if window.shape[1] >= 9:
        qw = window[:, 5].astype(float)
        qx = window[:, 6].astype(float)
        qy = window[:, 7].astype(float)
        qz = window[:, 8].astype(float)
    else:
        qw = np.ones(len(window))
        qx = qy = qz = np.zeros(len(window))

    fwd_z   = 2.0 * (qx * qz + qw * qy)
    up_z    = 1.0 - 2.0 * (qx**2 + qy**2)
    right_z = 2.0 * (qy * qz - qw * qx)

    return np.array([
        float(np.mean(fwd_z)),   float(np.std(fwd_z)),
        float(np.mean(up_z)),    float(np.std(up_z)),
        float(np.mean(right_z)), float(np.std(right_z)),
    ], dtype=np.float64)

def extract_features_stage2(window: np.ndarray) -> np.ndarray:
    """29 features — 25 flex stats + mean of each of 4 IMU channels (cols 5-8).
    Used by professor's cascade format (disamb_dgq / disamb_kp)."""
    feats = list(extract_features_stage1(window))
    if window.shape[1] >= 9:
        for i in range(5, 9):
            v = window[:, i].astype(float)
            v = v[~np.isnan(v)]
            feats.append(float(np.mean(v)) if len(v) > 0 else (1.0 if i == 5 else 0.0))
    else:
        feats.extend([1.0, 0.0, 0.0, 0.0])
    return np.array(feats, dtype=np.float64)

def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """Legacy path: 45 features (5 stats × 9 channels). Used by single-model files."""
    if window.shape[1] < 9:
        pad = np.zeros((window.shape[0], 9 - window.shape[1]))
        pad[:, 0] = 1.0
        window = np.hstack([window, pad])
    feats = []
    for i in range(9):
        feats.extend(_safe_stats(window[:, i]))
    return np.array(feats, dtype=np.float64)

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
    imu: Optional[List[float]] = None   # [w, x, y, z] — current quaternion for stage-2 disambiguation
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

    # If a current IMU quaternion was sent, append it as a constant column to every row
    # so extract_features_stage2 can compute mean(qw,qx,qy,qz) from real orientation data.
    if sensor_data.imu and len(sensor_data.imu) == 4:
        imu_cols = np.tile(sensor_data.imu, (arr.shape[0], 1))   # (N, 4)
        arr = np.hstack([arr, imu_cols])                          # → (N, 9)

    # ── v2 gravity-cascade model ───────────────────────────────────────────────
    if isinstance(model, dict) and model.get("format") == "v2_gravity_cascade":
        s1  = model["stage_1_model"]
        f1  = extract_features_stage1(arr).reshape(1, -1)

        probs1    = s1.predict_proba(f1)[0]
        prob_dict = {str(c): float(p) for c, p in zip(s1.classes_, probs1)}
        pred      = str(s1.predict(f1)[0])
        conf      = float(max(probs1))

        # Stage 2: only runs when Stage 1 lands on a confusable family
        families  = model.get("families", {})
        s2_models = model.get("stage_2_models", {})
        fg        = extract_gravity_features(arr).reshape(1, -1)

        for fam_name, members in families.items():
            if pred in members and fam_name in s2_models:
                clf  = s2_models[fam_name]
                p2   = clf.predict_proba(fg)[0]
                pred = str(clf.predict(fg)[0])
                conf = float(max(p2))
                for c, p in zip(clf.classes_, p2):
                    prob_dict[str(c)] = float(p)
                break

    # ── Two-staged model (professor's format) ─────────────────────────────────
    elif isinstance(model, dict) and "stage_1_model" in model:
        s1 = model["stage_1_model"]
        f1 = extract_features_stage1(arr).reshape(1, -1)
        f2 = extract_features_stage2(arr).reshape(1, -1)

        probs1    = s1.predict_proba(f1)[0]
        prob_dict = {str(c): float(p) for c, p in zip(s1.classes_, probs1)}
        pred      = str(s1.predict(f1)[0])
        conf      = float(max(probs1))

        # ── Format A: disamb_dgq + disamb_kp (latest model) ──────────────────
        if "disamb_dgq" in model:
            dgq_clf    = model["disamb_dgq"]
            kp_clf     = model["disamb_kp"]
            dgq_letters = [str(c) for c in dgq_clf.classes_]
            kp_letters  = [str(c) for c in kp_clf.classes_]

            if pred in dgq_letters:
                p2    = dgq_clf.predict_proba(f2)[0]
                pred  = str(dgq_clf.predict(f2)[0])
                conf  = float(max(p2))
                for c, p in zip(dgq_clf.classes_, p2):
                    prob_dict[str(c)] = float(p)
            elif pred in kp_letters:
                p2    = kp_clf.predict_proba(f2)[0]
                pred  = str(kp_clf.predict(f2)[0])
                conf  = float(max(p2))
                for c, p in zip(kp_clf.classes_, p2):
                    prob_dict[str(c)] = float(p)

        # ── Format B: stage_2_disambiguators + stage_2_verifiers (older) ─────
        elif "stage_2_disambiguators" in model:
            disambiguators = model["stage_2_disambiguators"]
            dg_family   = [str(l) for l in model.get("dg_family", [])]
            kp_family   = [str(l) for l in model.get("kp_family", [])]
            verifiers   = model.get("stage_2_verifiers", {})
            ver_letters = [str(l) for l in model.get("verifier_letters", [])]
            threshold   = float(model.get("verifier_threshold", 0.65))
            sorted_idx  = probs1.argsort()[::-1]
            top_classes = [str(s1.classes_[i]) for i in sorted_idx]

            if pred in dg_family and "DG" in disambiguators:
                clf = disambiguators["DG"]
                p2  = clf.predict_proba(f2)[0]
                pred = str(clf.predict(f2)[0]); conf = float(max(p2))
                for c, p in zip(clf.classes_, p2): prob_dict[str(c)] = float(p)
            elif pred in kp_family and "KP" in disambiguators:
                clf = disambiguators["KP"]
                p2  = clf.predict_proba(f2)[0]
                pred = str(clf.predict(f2)[0]); conf = float(max(p2))
                for c, p in zip(clf.classes_, p2): prob_dict[str(c)] = float(p)
            elif pred in ver_letters and pred in verifiers:
                clf     = verifiers[pred]
                vp      = clf.predict_proba(f2)[0]
                yes_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
                yes_conf = float(vp[yes_idx]) if yes_idx >= 0 else 0.0
                if yes_conf < threshold:
                    for cand in top_classes[1:3]:
                        if cand not in ver_letters:
                            pred = cand; conf = prob_dict.get(pred, 0.0); break

        # ── Format C: single stage_2_model (earliest format) ─────────────────
        elif "stage_2_model" in model:
            triggers = model.get("imu_trigger_letters") or model.get("trigger_letters", [])
            if pred in triggers:
                clf   = model["stage_2_model"]
                p2    = clf.predict_proba(f2)[0]
                pred  = str(clf.predict(f2)[0]); conf = float(max(p2))
                for c, p in zip(clf.classes_, p2): prob_dict[str(c)] = float(p)

    # ── Legacy single-model format ─────────────────────────────────────────────
    else:
        features = extract_features_from_window(arr).reshape(1, -1)
        pred = model.predict(features)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            prob_dict = {str(c): float(p) for c, p in zip(model.classes_, probs)}
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
