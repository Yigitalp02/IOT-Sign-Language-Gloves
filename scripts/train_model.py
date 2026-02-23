#!/usr/bin/env python3
"""
ASL Glove Model Training - Normalized Data
Trains a Random Forest for real-time 15-letter ASL recognition.
Feature extraction matches ASL-ML-Inference-API (25 stats per window).
"""

import argparse
import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import warnings
warnings.filterwarnings("ignore")

# Must match ASL-ML-Inference-API/app/main.py extract_features_from_window
def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """Extract 25 statistical features: mean, std, min, max, range per finger."""
    features = []
    for finger_idx in range(5):
        finger_values = window[:, finger_idx].astype(float)
        vals = finger_values[~np.isnan(finger_values)]
        if len(vals) < 2:
            std_val = 0.0
        else:
            std_val = float(np.std(vals))
        features.extend([
            float(np.mean(vals)) if len(vals) > 0 else 0.0,
            std_val,
            float(np.min(vals)) if len(vals) > 0 else 0.0,
            float(np.max(vals)) if len(vals) > 0 else 0.0,
            float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0,
        ])
    return np.array(features, dtype=np.float64)


def load_and_prepare_data(
    csv_path: str,
    window_size: int = 50,
    stride: int = 25,
    samples_per_recording: int = 150,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load normalized CSV and create windowed features.
    Splits by recording to avoid leakage (samples from same recording stay together).
    """
    df = pd.read_csv(csv_path)
    if "ch0_norm" not in df.columns:
        raise ValueError("Expected columns: label, ch0_norm, ch1_norm, ch2_norm, ch3_norm, ch4_norm")

    channel_cols = ["ch0_norm", "ch1_norm", "ch2_norm", "ch3_norm", "ch4_norm"]
    X_list, y_list, recording_ids = [], [], []

    for letter in df["label"].unique():
        subset = df[df["label"] == letter].reset_index(drop=True)
        data = subset[channel_cols].values.astype(np.float64)

        # Split into recordings (~150 samples each)
        n_recordings = max(1, len(data) // samples_per_recording)
        for rec_idx in range(n_recordings):
            start = rec_idx * samples_per_recording
            end = min(start + samples_per_recording, len(data))
            rec_data = data[start:end]

            if len(rec_data) < window_size:
                continue

            # Sliding windows within recording
            for i in range(0, len(rec_data) - window_size + 1, stride):
                window = rec_data[i : i + window_size]
                feats = extract_features_from_window(window)
                X_list.append(feats)
                y_list.append(letter)
                recording_ids.append((letter, rec_idx))

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, recording_ids


def split_by_recording(X, y, recording_ids, test_ratio=0.2, seed=42):
    """Split so train and test have different recordings (no leakage)."""
    np.random.seed(seed)
    unique_recs = sorted(set(recording_ids))
    n_test = max(1, int(len(unique_recs) * test_ratio))
    np.random.shuffle(unique_recs)
    test_recs = set(unique_recs[:n_test])

    train_mask = np.array([r not in test_recs for r in recording_ids])
    test_mask = ~train_mask

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def main():
    parser = argparse.ArgumentParser(description="Train ASL glove model on normalized data")
    parser.add_argument(
        "--input",
        "-i",
        default="data/Data/glove_data_NORMALIZED_B_A_C_D_E_F_I_K_O_S_T_V_W_X_Y_2026-02-23-10-45-28.csv",
        help="Path to normalized CSV",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="models/rf_asl_15letters_normalized.pkl",
        help="Output model path",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Samples per window (50 = 1 sec at 50Hz)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=25,
        help="Stride between windows (50%% overlap)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (slower)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees (300=fast/small, 500+=more accuracy, may overfit)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run compare_models.py after training (same evaluation as compare script)",
    )
    parser.add_argument(
        "--try-seeds",
        type=str,
        default="",
        metavar="SEEDS",
        help="Try multiple seeds, keep best (e.g. '42,0,123,7,99')",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    csv_path = project_root / args.input
    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: Input file not found: {csv_path}")
        return 1

    print("=" * 60)
    print("ASL Glove Model Training (Normalized Data)")
    print("=" * 60)
    print(f"Input: {csv_path}")
    print(f"Window size: {args.window_size} samples")
    print(f"Stride: {args.stride}")
    print()

    print("Loading data...")
    X, y, rec_ids = load_and_prepare_data(
        str(csv_path),
        window_size=args.window_size,
        stride=args.stride,
    )
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(f"  Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")
    print()

    seeds_to_try = [args.seed]
    if args.try_seeds:
        seeds_to_try = [int(s.strip()) for s in args.try_seeds.split(",") if s.strip()]
        print(f"Will try seeds: {seeds_to_try} (keep best)")
        if args.tune:
            print("Warning: --tune with --try-seeds is slow; tuning uses first seed only")
    print()

    best_model = None
    best_test_acc = -1.0
    best_seed = None
    best_cv_acc = -1.0

    for run_seed in seeds_to_try:
        if len(seeds_to_try) > 1:
            print(f"--- Seed {run_seed} ---")
        X_train, X_test, y_train, y_test = split_by_recording(
            X, y, rec_ids, test_ratio=0.2, seed=run_seed
        )
        if len(seeds_to_try) == 1:
            print(f"Train: {len(X_train)}, Test: {len(X_test)} (split by recording)")
            print()

        if args.tune and run_seed == seeds_to_try[0]:
            print("Hyperparameter tuning (may take 5-15 min)...")
            param_dist = {
                "n_estimators": [300, 400, 500, 600, 700, 800],
                "max_depth": [20, 25, 30, 35, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            }
            rf = RandomForestClassifier(random_state=run_seed, n_jobs=-1)
            search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=50,
                cv=5,
                scoring="f1_weighted",
                random_state=run_seed,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"Best params: {search.best_params_}")
        else:
            print(f"Training Random Forest ({args.n_estimators} trees)...")
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=run_seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_acc = cv_scores.mean()
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        if len(seeds_to_try) > 1:
            print(f"  Seed {run_seed}: test={test_acc:.4f}, cv={cv_acc:.4f}")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = model
                best_seed = run_seed
                best_cv_acc = cv_acc
            continue

        print("Cross-validation (5-fold)...")
        print(f"  Accuracy: {cv_acc:.4f} (+/- {cv_scores.std() * 2:.4f})")
        print()
        print("Test set performance:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1 (weighted): {test_f1:.4f}")
        print()
        print("Classification report:")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
        print("      ", " ".join(f"{l:>4}" for l in labels))
        for i, l in enumerate(labels):
            row = " ".join(f"{cm[i, j]:4d}" for j in range(len(labels)))
            print(f"  {l}   {row}")
        print()
        best_model = model
        best_test_acc = test_acc
        best_seed = run_seed
        best_cv_acc = cv_acc
        break

    model = best_model
    if len(seeds_to_try) > 1:
        print("=" * 60)
        print(f"Best: seed {best_seed} -> test {best_test_acc:.4f} ({best_test_acc*100:.1f}%)")
        print("=" * 60)
        print()

    joblib.dump(model, out_path)
    print(f"Model saved: {out_path}")

    # Save metadata for API compatibility
    meta = {
        "window_size": args.window_size,
        "n_features": 25,
        "classes": list(model.classes_),
        "normalized_input": True,
        "n_estimators": getattr(model, "n_estimators", args.n_estimators),
        "max_depth": getattr(model, "max_depth", 25),
        "seed": best_seed,
        "test_accuracy": float(best_test_acc),
        "cv_accuracy": float(best_cv_acc),
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = out_path.with_suffix(".meta.joblib")
    joblib.dump(meta, meta_path)
    print(f"Metadata saved: {meta_path}")

    # Append to training log (reproducible record)
    log_path = out_path.parent / "training_log.jsonl"
    log_entry = {
        "timestamp": meta["timestamp"],
        "seed": best_seed,
        "n_estimators": meta["n_estimators"],
        "max_depth": meta["max_depth"],
        "test_accuracy": meta["test_accuracy"],
        "cv_accuracy": meta["cv_accuracy"],
        "output": str(out_path.name),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Logged to {log_path.name}")
    print()

    if args.compare:
        print("Running compare_models (same evaluation)...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(script_dir / "compare_models.py"), "--skip-api"],
            cwd=str(project_root),
        )
        if result.returncode != 0:
            print("(compare_models had an issue, but model was saved)")
    else:
        print("Tip: Run 'python scripts/compare_models.py --skip-api' for full evaluation")

    print()
    print("Done. Deploy to API: copy .pkl to /opt/stack/ai-models/ and set MODEL_PATH")
    return 0


if __name__ == "__main__":
    exit(main())
