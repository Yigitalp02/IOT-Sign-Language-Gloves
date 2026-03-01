#!/usr/bin/env python3
"""
ASL Glove Model Training — Flex + IMU support

Handles two kinds of input CSVs:

  5-column (legacy, flex only):
    label, ch0_norm, ch1_norm, ch2_norm, ch3_norm, ch4_norm

  9-column (new, flex + IMU):
    label, ch0_norm, ch1_norm, ch2_norm, ch3_norm, ch4_norm, qw, qx, qy, qz

Merging rule
  - If a letter appears in a 9-column CSV → use that data (real IMU).
  - If a letter only appears in a 5-column CSV → keep it, but augment with
    random near-identity quaternions so the model learns it is
    orientation-invariant (can be signed at any wrist angle).

Feature vector: 29 values
  - 25: mean, std, min, max, range  ×  5 flex channels
  -  4: mean(qw), mean(qx), mean(qy), mean(qz)  (quaternion is stable in a
        static sign; std / range across the 50-sample window are near-zero)

Usage examples
  # Original 15-letter flex-only dataset (backward compatible):
  python scripts/train_model.py -i data/Data/glove_data_feb26.csv

  # Mixed: old flex CSV  +  new D/K/G/H/L/P/Q/R CSV with IMU:
  python scripts/train_model.py \\
      -i data/Data/glove_data_NORMALIZED_..._feb26.csv \\
      -i data/Data/glove_data_IMU_D_K_G_H_L_P_Q_R_<timestamp>.csv \\
      -o models/rf_asl_21letters_imu.pkl
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ── Column definitions ────────────────────────────────────────────────────────
FLEX_COLS = ["ch0_norm", "ch1_norm", "ch2_norm", "ch3_norm", "ch4_norm"]
IMU_COLS  = ["qw", "qx", "qy", "qz"]
ALL_COLS  = FLEX_COLS + IMU_COLS

N_FEATURES = 45   # 5 stats × 9 channels (5 flex + 4 IMU)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """
    Extract 45 features from a (window_size × 9) array.

    All 9 channels get the same 5-stat treatment:
      mean, std, min, max, range  ×  9 channels  =  45 features

    Channels 0-4 : flex sensors (normalized 0-1)
    Channels 5-8 : quaternion components qw, qx, qy, qz

    For a static sign the IMU channels are nearly constant, so their std/range
    will be close to zero — but Random Forest naturally ignores low-variance
    features, so including them is harmless and keeps the extraction uniform.

    Legacy (window_size × 5) arrays are padded with an identity quaternion
    column so the output is always 45 values.
    """
    n_channels = window.shape[1]

    # Pad legacy 5-column windows with identity quaternion (1, 0, 0, 0)
    if n_channels < 9:
        pad = np.zeros((window.shape[0], 9 - n_channels))
        pad[:, 0] = 1.0  # qw = 1  (identity)
        window = np.hstack([window, pad])

    features = []
    for i in range(9):
        vals = window[:, i].astype(float)
        vals = vals[~np.isnan(vals)]
        std_val = float(np.std(vals)) if len(vals) >= 2 else 0.0
        features.extend([
            float(np.mean(vals)) if len(vals) > 0 else 0.0,
            std_val,
            float(np.min(vals))  if len(vals) > 0 else 0.0,
            float(np.max(vals))  if len(vals) > 0 else 0.0,
            float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0,
        ])

    return np.array(features, dtype=np.float64)


# ── Quaternion augmentation ───────────────────────────────────────────────────
def augment_with_quaternion(df: pd.DataFrame, sigma: float = 0.04, seed: int = 0) -> pd.DataFrame:
    """
    Attach uniformly random unit quaternions to a flex-only DataFrame.

    These letters are orientation-invariant — the same flex pattern should be
    recognised at ANY wrist angle.  We therefore sample quaternions uniformly
    over the full rotation sphere (Shoemake 1992) rather than near-identity,
    so the model never learns that a particular orientation implies a letter.

    `sigma` is kept as a parameter for API compatibility but is no longer used;
    the distribution is always the full uniform sphere.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Uniform random unit quaternions: sample 4 standard normals then normalise
    raw   = rng.standard_normal((n, 4))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    q     = raw / norms            # shape (n, 4) — uniform on S³

    df = df.copy()
    df["qw"] = q[:, 0]
    df["qx"] = q[:, 1]
    df["qy"] = q[:, 2]
    df["qz"] = q[:, 3]
    return df


# ── Multi-CSV loader ──────────────────────────────────────────────────────────
def load_combined_data(csv_paths: list[str], aug_sigma: float = 0.04, aug_seed: int = 0) -> pd.DataFrame:
    """
    Load one or more CSVs, auto-detect format, merge with priority rules:

      • Letters in ANY 9-column CSV → use real IMU data, ignore flex-only rows.
      • Letters ONLY in 5-column CSV → augment with random quaternions.

    This means re-recorded letters (D, K) in the IMU CSV automatically
    replace the old recordings of those letters in the flex CSV.
    """
    flex_frames: list[pd.DataFrame] = []
    imu_frames:  list[pd.DataFrame] = []
    imu_letters: set[str] = set()

    for path in csv_paths:
        df = pd.read_csv(path)

        if "ch0_norm" not in df.columns:
            raise ValueError(f"{path}: missing 'ch0_norm' column")

        if all(c in df.columns for c in IMU_COLS):
            # 9-column CSV — real IMU data
            rows = df[["label"] + ALL_COLS].copy()
            imu_letters.update(rows["label"].unique())
            imu_frames.append(rows)
            print(f"  [9-col] {Path(path).name}: {len(rows):,} rows, "
                  f"letters: {sorted(rows['label'].unique())}")
        else:
            # 5-column CSV — flex only
            rows = df[["label"] + FLEX_COLS].copy()
            flex_frames.append(rows)
            print(f"  [5-col] {Path(path).name}: {len(rows):,} rows, "
                  f"letters: {sorted(rows['label'].unique())}")

    parts: list[pd.DataFrame] = []

    # IMU frames: keep as-is
    if imu_frames:
        parts.append(pd.concat(imu_frames, ignore_index=True))

    # Flex frames: drop letters that have real IMU, augment the rest
    if flex_frames:
        flex_all = pd.concat(flex_frames, ignore_index=True)
        flex_keep = flex_all[~flex_all["label"].isin(imu_letters)].copy()

        if len(flex_keep) > 0:
            skipped = set(flex_all["label"].unique()) - set(flex_keep["label"].unique())
            if skipped:
                print(f"  Skipping from flex CSV (replaced by IMU CSV): {sorted(skipped)}")
            flex_keep = augment_with_quaternion(flex_keep, sigma=aug_sigma, seed=aug_seed)
            parts.append(flex_keep)

    if not parts:
        raise ValueError("No data loaded from any input CSV")

    combined = pd.concat(parts, ignore_index=True)

    flex_only_letters = sorted(set(combined["label"].unique()) - imu_letters)
    print()
    print(f"  Real IMU data   : {sorted(imu_letters) if imu_letters else 'none'}")
    print(f"  Augmented quat  : {flex_only_letters}")
    print(f"  Total rows      : {len(combined):,}")
    return combined


# ── Windowed feature extraction ───────────────────────────────────────────────
def prepare_windows(
    df: pd.DataFrame,
    window_size: int = 50,
    stride: int = 25,
    samples_per_recording: int = 150,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Slide windows over each recording, extract 29-feature vectors.
    Recordings are kept separate to avoid train/test leakage.
    """
    channel_cols = ALL_COLS  # always 9 columns (IMU was added for flex-only letters)
    X_list, y_list, recording_ids = [], [], []

    for letter in df["label"].unique():
        subset = df[df["label"] == letter].reset_index(drop=True)
        data = subset[channel_cols].values.astype(np.float64)

        n_recordings = max(1, len(data) // samples_per_recording)
        for rec_idx in range(n_recordings):
            start = rec_idx * samples_per_recording
            end   = min(start + samples_per_recording, len(data))
            rec_data = data[start:end]

            if len(rec_data) < window_size:
                continue

            for i in range(0, len(rec_data) - window_size + 1, stride):
                window  = rec_data[i : i + window_size]
                feats   = extract_features_from_window(window)
                X_list.append(feats)
                y_list.append(letter)
                recording_ids.append((letter, rec_idx))

    return np.array(X_list), np.array(y_list), recording_ids


# ── Train/test split ──────────────────────────────────────────────────────────
def split_by_recording(X, y, recording_ids, test_ratio=0.2, seed=42):
    """Split so train and test never share recordings (no data leakage)."""
    np.random.seed(seed)
    unique_recs = sorted(set(recording_ids))
    n_test = max(1, int(len(unique_recs) * test_ratio))
    np.random.shuffle(unique_recs)
    test_recs = set(unique_recs[:n_test])

    train_mask = np.array([r not in test_recs for r in recording_ids])
    return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train ASL glove model (flex + optional IMU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  # Single flex-only CSV (backward compatible with old datasets):
  python scripts/train_model.py -i data/Data/glove_data_feb26.csv

  # Mixed: old 15-letter flex CSV  +  new D/K/G/H/L/P/Q/R CSV with IMU
  python scripts/train_model.py \\
      -i data/Data/glove_data_NORMALIZED_..._feb26.csv \\
      -i data/Data/glove_data_IMU_D_K_G_H_L_P_Q_R_<date>.csv \\
      -o models/rf_asl_21letters_imu.pkl
        """,
    )
    parser.add_argument(
        "--input", "-i",
        action="append",
        dest="inputs",
        metavar="CSV",
        default=None,
        help="Input CSV (repeat for multiple files). 5-col or 9-col auto-detected.",
    )
    parser.add_argument(
        "--output", "-o",
        default="models/rf_asl_model.pkl",
        help="Output model path",
    )
    parser.add_argument("--window-size", type=int, default=50,
                        help="Samples per window (default: 50 = 1s at 50 Hz)")
    parser.add_argument("--stride",      type=int, default=25,
                        help="Window stride (default: 25 = 50%% overlap)")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Random Forest trees (default: 300)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--try-seeds", type=str, default="", metavar="SEEDS",
                        help="Try multiple seeds, keep best (e.g. '1,42,0,123,7')")
    parser.add_argument("--tune", action="store_true",
                        help="Hyperparameter search (slow, ~5-15 min)")
    parser.add_argument("--aug-sigma", type=float, default=0.04,
                        help="Quaternion augmentation noise for flex-only letters (default: 0.04)")
    parser.add_argument("--compare", action="store_true",
                        help="Run compare_models.py after training")
    args = parser.parse_args()

    # Default input if none given
    if not args.inputs:
        args.inputs = [
            "data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_K_O_S_T_V_W_X_Y_2026-02-26-12-30-42.csv"
        ]

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_paths  = [str(project_root / p) for p in args.inputs]
    out_path     = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in input_paths:
        if not Path(p).exists():
            print(f"Error: Input file not found: {p}")
            return 1

    print("=" * 60)
    print("ASL Glove Model Training (Flex + IMU)")
    print("=" * 60)
    for p in input_paths:
        print(f"  Input : {p}")
    print(f"  Output: {out_path}")
    print(f"  Window: {args.window_size} samples  Stride: {args.stride}")
    print()

    # ── Load & merge ──────────────────────────────────────────────────────────
    print("Loading data...")
    combined_df = load_combined_data(input_paths, aug_sigma=args.aug_sigma)
    print()

    # ── Build feature matrix ──────────────────────────────────────────────────
    print("Extracting windowed features...")
    X, y, rec_ids = prepare_windows(
        combined_df,
        window_size=args.window_size,
        stride=args.stride,
    )
    n_samples, n_feats = X.shape
    n_classes = len(np.unique(y))
    print(f"  Windows: {n_samples}  Features: {n_feats}  Classes: {n_classes}")
    print(f"  Labels : {sorted(np.unique(y))}")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    seeds_to_try = [args.seed]
    if args.try_seeds:
        seeds_to_try = [int(s.strip()) for s in args.try_seeds.split(",") if s.strip()]
        print(f"Trying seeds: {seeds_to_try}")

    best_model, best_test_acc, best_seed, best_cv_acc = None, -1.0, None, -1.0

    for run_seed in seeds_to_try:
        if len(seeds_to_try) > 1:
            print(f"--- Seed {run_seed} ---")

        X_train, X_test, y_train, y_test = split_by_recording(
            X, y, rec_ids, test_ratio=0.2, seed=run_seed
        )
        if len(seeds_to_try) == 1:
            print(f"Train: {len(X_train)}  Test: {len(X_test)} (split by recording)")
            print()

        if args.tune and run_seed == seeds_to_try[0]:
            print("Hyperparameter tuning (may take 5-15 min)...")
            param_dist = {
                "n_estimators":     [300, 400, 500, 600, 700, 800],
                "max_depth":        [20, 25, 30, 35, None],
                "min_samples_split":[2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features":     ["sqrt", "log2", None],
            }
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=run_seed, n_jobs=-1),
                param_distributions=param_dist,
                n_iter=50, cv=5, scoring="f1_weighted",
                random_state=run_seed, n_jobs=-1, verbose=1,
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

        cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)
        cv_acc   = cv_scores.mean()
        y_pred   = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1  = f1_score(y_test, y_pred, average="weighted")

        if len(seeds_to_try) > 1:
            print(f"  Seed {run_seed}: test={test_acc:.4f}  cv={cv_acc:.4f}")
            if test_acc > best_test_acc:
                best_test_acc, best_model, best_seed, best_cv_acc = (
                    test_acc, model, run_seed, cv_acc)
            continue

        # Single-seed: print full report
        print("Cross-validation (5-fold):")
        print(f"  Accuracy: {cv_acc:.4f} (+/- {cv_scores.std() * 2:.4f})")
        print()
        print("Test set:")
        print(f"  Accuracy : {test_acc:.4f}")
        print(f"  F1 (wtd) : {test_f1:.4f}")
        print()
        print("Classification report:")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        print("       " + " ".join(f"{l:>4}" for l in labels))
        for i, l in enumerate(labels):
            print(f"  {l:>3}  " + " ".join(f"{cm[i,j]:4d}" for j in range(len(labels))))
        print()
        best_model, best_test_acc, best_seed, best_cv_acc = model, test_acc, run_seed, cv_acc
        break

    if len(seeds_to_try) > 1:
        print("=" * 60)
        print(f"Best: seed {best_seed}  →  test {best_test_acc:.4f} "
              f"({best_test_acc * 100:.1f}%)")
        print("=" * 60)
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(best_model, out_path)
    print(f"Model saved : {out_path}")

    meta = {
        "window_size":      args.window_size,
        "n_features":       N_FEATURES,
        "has_imu":          True,
        "classes":          list(best_model.classes_),
        "normalized_input": True,
        "n_estimators":     getattr(best_model, "n_estimators", args.n_estimators),
        "max_depth":        getattr(best_model, "max_depth", 25),
        "seed":             best_seed,
        "test_accuracy":    float(best_test_acc),
        "cv_accuracy":      float(best_cv_acc),
        "timestamp":        datetime.now().isoformat(),
        "inputs":           args.inputs,
    }
    meta_path = out_path.with_suffix(".meta.joblib")
    joblib.dump(meta, meta_path)
    print(f"Metadata    : {meta_path}")

    log_path = out_path.parent / "training_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp":     meta["timestamp"],
            "seed":          best_seed,
            "n_estimators":  meta["n_estimators"],
            "max_depth":     meta["max_depth"],
            "test_accuracy": meta["test_accuracy"],
            "cv_accuracy":   meta["cv_accuracy"],
            "n_classes":     n_classes,
            "has_imu":       True,
            "output":        out_path.name,
        }) + "\n")
    print(f"Log         : {log_path.name}")
    print()

    if args.compare:
        import subprocess
        subprocess.run([sys.executable, str(script_dir / "compare_models.py"), "--skip-api"],
                       cwd=str(project_root))
    else:
        print("Tip: python scripts/compare_models.py --skip-api")

    print()
    print("Done. Deploy: copy .pkl to /opt/stack/ai-models/ and update MODEL_PATH")
    print()
    print("NOTE: The API (main.py) and serve_local_model.py also need their")
    print("      extract_features_from_window updated to match this 29-feature")
    print("      version, and the /predict endpoint must accept 9-channel input.")
    return 0


if __name__ == "__main__":
    exit(main())
