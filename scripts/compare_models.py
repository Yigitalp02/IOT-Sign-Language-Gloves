#!/usr/bin/env python3
"""
Compare New (Normalized) Model vs Legacy (Flex) Model
Runs both on the same test data and reports accuracy, F1, confusion matrix.

No API key needed if you have rf_asl_15letters.pkl in iot-sign-glove/models/.
If the legacy model is missing, use API_KEY + --api-url to compare via cloud API.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_model import extract_features_from_window, split_by_recording

# Flex sensor ranges used by API model (from App.tsx MODEL_BASELINES/MODEL_MAXBENDS)
FLEX_BASELINES = [440, 612, 618, 548, 528]   # straight (lower)
FLEX_MAXBENDS = [650, 900, 900, 850, 800]     # bent (higher)


def normalized_to_flex(norm_window: np.ndarray) -> np.ndarray:
    """Convert normalized 0-1 window to flex sensor range for API model."""
    flex = np.zeros_like(norm_window)
    for c in range(5):
        flex[:, c] = FLEX_BASELINES[c] + norm_window[:, c] * (FLEX_MAXBENDS[c] - FLEX_BASELINES[c])
    return np.clip(flex, 0, 1023).astype(int)


def predict_api(norm_window: np.ndarray, api_url: str, api_key: str) -> str:
    """Send flex-converted window to API, return predicted letter."""
    flex_data = normalized_to_flex(norm_window)
    payload = {"flex_sensors": flex_data.tolist(), "device_id": "compare-script"}
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    try:
        import urllib.request
        req = urllib.request.Request(
            f"{api_url}/predict",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            return str(result["letter"])
    except Exception as e:
        return f"ERR:{e}"


def main():
    parser = argparse.ArgumentParser(description="Compare new vs API model")
    parser.add_argument(
        "--input", "-i",
        default="data/Data/glove_data_NORMALIZED_B_A_C_D_E_F_I_K_O_S_T_V_W_X_Y_2026-02-23-10-45-28.csv",
        help="Normalized CSV path",
    )
    parser.add_argument("--model", "-m", default="models/rf_asl_15letters_normalized.pkl")
    parser.add_argument("--legacy-model", default="models/rf_asl_15letters.pkl", help="Legacy flex model (run locally if exists)")
    parser.add_argument("--api-url", default=os.getenv("API_URL", "https://api.ybilgin.com"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--skip-api", action="store_true", help="Skip API comparison (API unreachable)")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-test", type=int, default=0, help="Max test samples (0=all)")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.input
    model_path = PROJECT_ROOT / args.model

    if not csv_path.exists():
        print(f"Error: Input not found: {csv_path}")
        return 1
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1

    print("=" * 60)
    print("Model Comparison: New (Normalized) vs API (Legacy Flex)")
    print("=" * 60)
    print(f"Input: {csv_path}")
    print(f"New model: {model_path}")
    print(f"API: {args.api_url}")
    print()

    # Load data (features + raw windows in same order)
    print("Loading data...")

    # Build raw windows too (same order as X,y) for API calls
    df = pd.read_csv(csv_path)
    channel_cols = ["ch0_norm", "ch1_norm", "ch2_norm", "ch3_norm", "ch4_norm"]
    X_list, windows_list, y_list, rec_ids_list = [], [], [], []
    for letter in df["label"].unique():
        subset = df[df["label"] == letter].reset_index(drop=True)
        data = subset[channel_cols].values.astype(np.float64)
        n_recs = max(1, len(data) // 150)
        for rec_idx in range(n_recs):
            start = rec_idx * 150
            end = min(start + 150, len(data))
            rec_data = data[start:end]
            if len(rec_data) < args.window_size:
                continue
            for i in range(0, len(rec_data) - args.window_size + 1, args.stride):
                win = rec_data[i : i + args.window_size]
                windows_list.append(win)
                X_list.append(extract_features_from_window(win))
                y_list.append(letter)
                rec_ids_list.append((letter, rec_idx))
    X_all = np.array(X_list)
    windows_all = np.array(windows_list)
    y_all = np.array(y_list)
    rec_ids_all = rec_ids_list

    X_train, X_test, y_train, y_test = split_by_recording(
        X_all, y_all, rec_ids_all, test_ratio=0.2
    )
    _, windows_test, _, _ = split_by_recording(
        windows_all, y_all, rec_ids_all, test_ratio=0.2
    )
    if args.max_test > 0:
        X_test = X_test[: args.max_test]
        y_test = y_test[: args.max_test]
        windows_test = windows_test[: args.max_test]

    print(f"Test samples: {len(y_test)}")
    print("Data format: New <- 0-1 norm | Legacy <- flex 440-900")
    print()

    # Load new model
    new_model = joblib.load(model_path)
    print("New model loaded:", model_path.stem)
    print()

    # Predict with new model: expects 25 features from NORMALIZED 0-1 windows
    # X_test = extract_features_from_window(norm_win) — already in correct format
    y_pred_new = new_model.predict(X_test)
    acc_new = accuracy_score(y_test, y_pred_new)
    f1_new = f1_score(y_test, y_pred_new, average="weighted")
    print()
    print("--- NEW MODEL (Normalized) ---")
    print(f"Accuracy: {acc_new:.4f}")
    print(f"F1 (weighted): {f1_new:.4f}")
    print(classification_report(y_test, y_pred_new))
    print("Confusion matrix:")
    labels = sorted(set(y_test) | set(y_pred_new))
    cm_new = confusion_matrix(y_test, y_pred_new, labels=labels)
    print("     ", " ".join(f"{l:>4}" for l in labels))
    for i, l in enumerate(labels):
        print(f" {l}  ", " ".join(f"{cm_new[i, j]:4d}" for j in range(len(labels))))
    print()

    # Predict with legacy model: LOCAL first (no API key needed), else API
    legacy_model_path = PROJECT_ROOT / args.legacy_model
    valid = np.zeros(len(y_test), dtype=bool)
    acc_legacy = 0.0
    y_pred_legacy = None

    if legacy_model_path.exists():
        # Run legacy model locally - no API needed!
        # Legacy expects 25 features from FLEX 440-900 windows (not 0-1)
        print("--- LEGACY MODEL (Flex) - LOCAL ---")
        legacy_model = joblib.load(legacy_model_path)
        print(f"Loaded: {legacy_model_path.stem}")
        X_legacy = np.array([
            extract_features_from_window(normalized_to_flex(w))  # convert 0-1 → flex, then extract features
            for w in windows_test
        ])
        y_pred_legacy = legacy_model.predict(X_legacy)
        valid = np.ones(len(y_test), dtype=bool)
        acc_legacy = accuracy_score(y_test, y_pred_legacy)
        f1_legacy = f1_score(y_test, y_pred_legacy, average="weighted")
        print(f"Accuracy: {acc_legacy:.4f}")
        print(f"F1 (weighted): {f1_legacy:.4f}")
        print(classification_report(y_test, y_pred_legacy))
        cm_legacy = confusion_matrix(y_test, y_pred_legacy, labels=labels)
        print("Confusion matrix:")
        print("     ", " ".join(f"{l:>4}" for l in labels))
        for i, l in enumerate(labels):
            print(f" {l}  ", " ".join(f"{cm_legacy[i, j]:4d}" for j in range(len(labels))))
    elif not args.skip_api and args.api_key:
        # Fallback: call cloud API — expects raw flex samples (440-900), API extracts features
        print("--- LEGACY MODEL (Flex) - via API ---")
        print("Legacy model not found locally. Calling API...")
        y_pred_legacy = []
        err_count = 0
        for i, win in enumerate(windows_test):
            pred = predict_api(win, args.api_url, args.api_key)
            if pred.startswith("ERR:"):
                err_count += 1
                y_pred_legacy.append("?")
            else:
                y_pred_legacy.append(pred)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(windows_test)}...")
        y_pred_legacy = np.array(y_pred_legacy)
        valid = y_pred_legacy != "?"
        if valid.sum() > 0:
            acc_legacy = accuracy_score(y_test[valid], y_pred_legacy[valid])
            f1_legacy = f1_score(y_test[valid], y_pred_legacy[valid], average="weighted")
            print(f"Accuracy: {acc_legacy:.4f} (valid: {valid.sum()})")
            print(f"F1 (weighted): {f1_legacy:.4f}")
            if err_count:
                print(f"API errors: {err_count}")
            print(classification_report(y_test[valid], y_pred_legacy[valid]))
            cm_legacy = confusion_matrix(y_test[valid], y_pred_legacy[valid], labels=labels)
            print("Confusion matrix:")
            print("     ", " ".join(f"{l:>4}" for l in labels))
            for i, l in enumerate(labels):
                print(f" {l}  ", " ".join(f"{cm_legacy[i, j]:4d}" for j in range(len(labels))))
        else:
            print("All API calls failed.")
    elif args.skip_api:
        print("--- LEGACY MODEL --- Skipped (--skip-api)")
        if not legacy_model_path.exists():
            print(f"  (Legacy model not found: {legacy_model_path})")
    else:
        print("--- LEGACY MODEL --- Skipped")
        if not legacy_model_path.exists():
            print(f"  Legacy model not found: {legacy_model_path}")
            print("  Add rf_asl_15letters.pkl to iot-sign-glove/models/ to compare locally (no API key needed).")
        else:
            print("  Set API_KEY to compare via cloud API.")

    print()
    print("=" * 60)
    if valid.sum() > 0:
        winner = "New (normalized)" if acc_new > acc_legacy else "Legacy (flex)"
        print(f"Winner on this test set: {winner} model")
        print(f"  New:    {acc_new:.1%}  |  Legacy: {acc_legacy:.1%}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
