#!/usr/bin/env python3
"""
Compare original (300 trees) vs tuned (800 trees) model on the SAME test set.
Ensures a fair head-to-head comparison.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_model import extract_features_from_window, load_and_prepare_data, split_by_recording

def main():
    csv_path = PROJECT_ROOT / "data/Data/glove_data_NORMALIZED_B_A_C_D_E_F_I_K_O_S_T_V_W_X_Y_2026-02-23-10-45-28.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return 1

    print("=" * 60)
    print("Model Config Comparison: Original (300) vs Tuned (800)")
    print("=" * 60)
    print(f"Data: {csv_path.name}")
    print()

    # Load data (same logic as train_model)
    X, y, rec_ids = load_and_prepare_data(str(csv_path), window_size=50, stride=25)
    X_train, X_test, y_train, y_test = split_by_recording(X, y, rec_ids, test_ratio=0.2, seed=42)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print()

    # 1. Train original config (300 trees)
    print("Training ORIGINAL config (300 trees, max_depth=25)...")
    model_orig = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model_orig.fit(X_train, y_train)
    acc_orig = accuracy_score(y_test, model_orig.predict(X_test))
    f1_orig = f1_score(y_test, model_orig.predict(X_test), average="weighted")
    print(f"  Test Accuracy: {acc_orig:.4f} ({acc_orig*100:.1f}%)")
    print(f"  F1 (weighted): {f1_orig:.4f}")
    print()

    # 2. Load tuned model (current saved model)
    tuned_path = PROJECT_ROOT / "models/rf_asl_15letters_normalized.pkl"
    if not tuned_path.exists():
        print(f"Error: Tuned model not found at {tuned_path}")
        return 1
    print("Loading TUNED model (800 trees from --tune)...")
    model_tuned = joblib.load(tuned_path)
    acc_tuned = accuracy_score(y_test, model_tuned.predict(X_test))
    f1_tuned = f1_score(y_test, model_tuned.predict(X_test), average="weighted")
    print(f"  Test Accuracy: {acc_tuned:.4f} ({acc_tuned*100:.1f}%)")
    print(f"  F1 (weighted): {f1_tuned:.4f}")
    print()

    # Summary
    print("=" * 60)
    print("HEAD-TO-HEAD (same test set, same split):")
    print("=" * 60)
    print(f"  Original (300 trees): {acc_orig:.4f}  ({acc_orig*100:.1f}%)")
    print(f"  Tuned (800 trees):    {acc_tuned:.4f}  ({acc_tuned*100:.1f}%)")
    diff = acc_tuned - acc_orig
    if abs(diff) < 0.005:
        print(f"  >> Essentially the same ({diff*100:+.1f}% difference)")
    elif diff > 0:
        print(f"  >> Tuned wins by {diff*100:.1f}%")
    else:
        print(f"  >> Original wins by {-diff*100:.1f}%")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
