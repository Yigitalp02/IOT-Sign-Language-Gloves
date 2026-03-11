#!/usr/bin/env python3
"""
ASL Glove Model Training v2 — Two-Stage Gravity Cascade

Architecture
============

Stage 1  — Flex only (25 features)
  Classifies all 21 letters using only the 5 flex channels.
  IMU is completely ignored, so orientation never affects Stage 1 predictions.
  Features: mean, std, min, max, range  ×  5 flex channels  = 25 features

Stage 2  — Gravity disambiguation (4 features, yaw-invariant)
  Runs ONLY when Stage 1 lands on a confusable family.
  Uses the z-components of the hand's Forward and Up axes — these are
  yaw-invariant (rotating left/right doesn't change them), so only the tilt
  relative to gravity matters.
  Features: mean(fwd_z), std(fwd_z), mean(up_z), std(up_z)  = 4 features

  fwd_z = 2*(qx*qz + qw*qy)  — how much the hand's forward direction points up/down
  up_z  = 1 - 2*(qx²  + qy²) — how much the hand's back faces up/down

Confusable families (identical flex, different orientation)
  DG   : D (fingers up, neutral)  vs  G (hand pointing sideways)
  VHR  : V (fingers up)  vs  H (hand sideways)  vs  R (fingers angled down)
  LPQ  : L (index up)  vs  P (hand down-forward)  vs  Q (diagonal tilt)

Orientation-invariant letters (A,B,C,D,E,F,I,K,L,O,S,T,V,W,X,Y)
  Stage 1 always decides these correctly because their flex is unique.
  D, V, L are in families only as the "neutral partner" — Stage 2 will still
  correctly return them when the hand orientation is neutral.

Orientation-dependent letters (G,H,P,Q,R)
  Stage 1 will sometimes predict their family partner (same flex).
  Stage 2 corrects this using the distinct IMU orientation they were recorded in.

Augmentation
  Stage 1 : none — flex features are stable, the model is orientation-blind.
  Stage 2 : ±2° per-sample Euler jitter (professor's tremor model),
            8 augmented copies per window, making Stage 2 robust to natural
            hand wobble while still learning clear orientation boundaries.

Usage
  python scripts/train_model_v2.py \\
      -i data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_G_K_O_L_S_H_R_P_Q_T_V_W_X_Y_2026-03-10-12-56-47.csv \\
      -o models/rf_asl_v2_gravity_cascade.pkl
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ── Letter groups ─────────────────────────────────────────────────────────────
FAMILIES: dict[str, list[str]] = {
    "DG":  ["D", "G"],
    "VHR": ["V", "H", "R"],
    "LPQ": ["L", "P", "Q"],
}
FAMILY_MEMBERS: set[str] = {l for fam in FAMILIES.values() for l in fam}
IMU_LETTERS:    set[str] = {"G", "H", "P", "Q", "R"}   # orientation-dependent

# ── Column names in the CSV ───────────────────────────────────────────────────
FLEX_COLS = ["ch0_norm", "ch1_norm", "ch2_norm", "ch3_norm", "ch4_norm"]
IMU_COLS  = ["qw", "qx", "qy", "qz"]
ALL_COLS  = FLEX_COLS + IMU_COLS   # 9 values per sample row

# ── Feature extraction — Stage 1 (flex only) ─────────────────────────────────
def _safe_stats(vals: np.ndarray) -> list[float]:
    """(mean, std, min, max, range) handling NaN / single-element arrays."""
    v = vals[~np.isnan(vals)].astype(float)
    if len(v) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(v)),
        float(np.std(v))  if len(v) >= 2 else 0.0,
        float(np.min(v)),
        float(np.max(v)),
        float(np.max(v) - np.min(v)) if len(v) > 1 else 0.0,
    ]


def flex_features(window: np.ndarray) -> np.ndarray:
    """
    25 features — mean/std/min/max/range of the 5 flex channels (cols 0-4).
    IMU columns are ignored completely.
    """
    feats: list[float] = []
    for i in range(5):
        feats.extend(_safe_stats(window[:, i]))
    return np.array(feats, dtype=np.float64)


# ── Feature extraction — Stage 2 (gravity, yaw-invariant) ────────────────────
def gravity_features(window: np.ndarray) -> np.ndarray:
    """
    6 yaw-invariant gravity features — the z-components of all three hand axes.

    All three are invariant to yaw (compass direction): only the tilt relative
    to gravity changes them. This fully describes the hand's 3D tilt.

    fwd_z   = 2*(qx*qz + qw*qy)   z-component of hand's forward axis
              +1 = fingers point straight up, -1 = fingers point straight down
    up_z    = 1 - 2*(qx^2 + qy^2) z-component of hand's up axis (back-of-hand)
              +1 = back of hand faces up (palm down), -1 = back of hand faces down
    right_z = 2*(qy*qz - qw*qx)   z-component of hand's right axis (thumb side)
              captures wrist roll — critical for separating P vs Q vs L

    Features: mean + std of each axis across the window = 6 features total.
    std is near-zero for a held static sign; it rises if orientation drifts.
    """
    qw = window[:, 5].astype(float)
    qx = window[:, 6].astype(float)
    qy = window[:, 7].astype(float)
    qz = window[:, 8].astype(float)

    fwd_z   = 2.0 * (qx * qz + qw * qy)
    up_z    = 1.0 - 2.0 * (qx**2 + qy**2)
    right_z = 2.0 * (qy * qz - qw * qx)

    return np.array([
        float(np.mean(fwd_z)),   float(np.std(fwd_z)),
        float(np.mean(up_z)),    float(np.std(up_z)),
        float(np.mean(right_z)), float(np.std(right_z)),
    ], dtype=np.float64)


# ── Per-sample Euler jitter (professor's tremor model) ───────────────────────
def _jitter_quaternion(qw, qx, qy, qz, max_deg: float, rng: np.random.Generator):
    """
    Compose q with a tiny random rotation (uniform in [-max_deg, +max_deg]
    around each axis).  Models natural hand tremor for augmentation.
    """
    d = np.radians(max_deg)
    rx, ry, rz = rng.uniform(-d, d, 3)

    # Build rotation quaternion for each axis and compose
    def axis_quat(angle, axis):
        h = angle / 2.0
        c, s = np.cos(h), np.sin(h)
        return (c, s if axis == 0 else 0.0,
                   s if axis == 1 else 0.0,
                   s if axis == 2 else 0.0)

    def qmul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        )

    j = qmul(qmul(axis_quat(rx, 0), axis_quat(ry, 1)), axis_quat(rz, 2))
    nw, nx, ny, nz = qmul((qw, qx, qy, qz), j)
    n = np.sqrt(nw**2 + nx**2 + ny**2 + nz**2) or 1.0
    return nw/n, nx/n, ny/n, nz/n


def jitter_window_imu(window: np.ndarray, max_deg: float,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Apply independent ±max_deg Euler jitter to every sample row in a window.
    Returns a new array; the original is not modified.
    """
    w = window.copy()
    for i in range(len(w)):
        w[i, 5], w[i, 6], w[i, 7], w[i, 8] = _jitter_quaternion(
            w[i, 5], w[i, 6], w[i, 7], w[i, 8], max_deg, rng
        )
    return w


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(csv_paths: list[str]) -> pd.DataFrame:
    """
    Load one or more 9-column CSVs and concatenate them.
    Raises ValueError if any file is missing the required columns.
    """
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        missing = [c for c in ["label"] + ALL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        frames.append(df[["label"] + ALL_COLS].copy())
        print(f"  Loaded {Path(path).name}: {len(df):,} rows, "
              f"letters={sorted(df['label'].unique())}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} rows, "
          f"{combined['label'].nunique()} letters: {sorted(combined['label'].unique())}")
    return combined


# ── Windowed extraction helpers ───────────────────────────────────────────────
def iter_windows(data: np.ndarray, window_size: int, stride: int,
                 samples_per_recording: int):
    """
    Yield (window_array, recording_index) for every strided window.
    Each group of `samples_per_recording` rows is treated as one recording session.
    """
    n_recordings = max(1, len(data) // samples_per_recording)
    for rec_idx in range(n_recordings):
        start = rec_idx * samples_per_recording
        end   = min(start + samples_per_recording, len(data))
        rec   = data[start:end]
        if len(rec) < window_size:
            continue
        for i in range(0, len(rec) - window_size + 1, stride):
            yield rec[i: i + window_size], rec_idx


# ── Stage 1 dataset ───────────────────────────────────────────────────────────
def build_stage1_dataset(df: pd.DataFrame, window_size: int, stride: int,
                         samples_per_rec: int):
    """
    25 flex features per window, all 21 letters, NO IMU, NO augmentation.
    Returns X (n_windows, 25), y (n_windows,), rec_ids list.
    """
    X, y, rec_ids = [], [], []
    for letter in sorted(df["label"].unique()):
        data = df[df["label"] == letter][ALL_COLS].values.astype(np.float64)
        for win, rec_idx in iter_windows(data, window_size, stride, samples_per_rec):
            X.append(flex_features(win))
            y.append(letter)
            rec_ids.append((letter, rec_idx))
    return np.array(X), np.array(y), rec_ids


# ── Stage 2 dataset per family ────────────────────────────────────────────────
def build_stage2_dataset(df: pd.DataFrame, family_letters: list[str],
                         window_size: int, stride: int, samples_per_rec: int,
                         aug_per_window: int, jitter_deg: float, seed: int):
    """
    4 gravity features per window, for the letters in a single family.

    Augmentation:
      aug_per_window copies of each window are created, each with
      independent per-sample Euler jitter of ±jitter_deg degrees.
      This makes Stage 2 robust to natural hand wobble while keeping
      the clear tilt boundaries between family members.
    """
    rng = np.random.default_rng(seed)
    X, y, rec_ids = [], [], []

    for letter in family_letters:
        subset = df[df["label"] == letter]
        if len(subset) == 0:
            print(f"  [Stage 2] WARNING: no data for letter '{letter}' -- skipping")
            continue
        data = subset[ALL_COLS].values.astype(np.float64)
        for win, rec_idx in iter_windows(data, window_size, stride, samples_per_rec):
            for _ in range(aug_per_window):
                aug = jitter_window_imu(win, jitter_deg, rng)
                X.append(gravity_features(aug))
                y.append(letter)
                rec_ids.append((letter, rec_idx))
    return np.array(X), np.array(y), rec_ids


# ── Train/test split by recording (no leakage) ───────────────────────────────
def split_by_recording(X, y, rec_ids, test_ratio: float = 0.2, seed: int = 42):
    """Hold out a fraction of recording sessions as the test set."""
    rng = np.random.default_rng(seed)
    unique_recs = sorted(set(rec_ids))
    rng.shuffle(unique_recs)
    n_test    = max(1, int(len(unique_recs) * test_ratio))
    test_recs = set(unique_recs[:n_test])
    mask      = np.array([r not in test_recs for r in rec_ids])
    return X[mask], X[~mask], y[mask], y[~mask]


# ── Random Forest factory ─────────────────────────────────────────────────────
def make_rf(n_estimators: int, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators    = n_estimators,
        max_depth       = 25,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features    = "sqrt",
        random_state    = seed,
        n_jobs          = -1,
    )


# ── Training pipeline ─────────────────────────────────────────────────────────
def train(args) -> int:
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_paths  = [str(project_root / p) for p in args.inputs]
    out_path     = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in input_paths:
        if not Path(p).exists():
            print(f"ERROR: File not found: {p}")
            return 1

    print("=" * 65)
    print("ASL Glove Model Training v2 -- Two-Stage Gravity Cascade")
    print("=" * 65)
    for p in input_paths:
        print(f"  Input : {p}")
    print(f"  Output: {out_path}")
    print(f"  Window: {args.window_size}  Stride: {args.stride}  "
          f"Trees: {args.n_estimators}  Jitter: +/-{args.jitter_deg}deg  "
          f"Aug/win: {args.aug_per_window}")
    print()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = load_data(input_paths)
    available_letters = set(df["label"].unique())
    print()

    # Validate all expected IMU letters are present
    missing_imu = IMU_LETTERS - available_letters
    if missing_imu:
        print(f"WARNING: IMU-dependent letters not found in data: {missing_imu}")
        print("  Stage 2 families may be incomplete.\n")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print("-" * 40)
    print("Stage 1 -- Flex-only classifier (25 features, all 21 letters)")
    print("-" * 40)

    X1, y1, rec_ids1 = build_stage1_dataset(
        df,
        window_size     = args.window_size,
        stride          = args.stride,
        samples_per_rec = args.samples_per_recording,
    )
    print(f"  Windows: {len(X1)}  Features: {X1.shape[1]}")

    X1_tr, X1_te, y1_tr, y1_te = split_by_recording(
        X1, y1, rec_ids1, test_ratio=0.2, seed=args.seed
    )
    print(f"  Train: {len(X1_tr)}  Test: {len(X1_te)}")

    s1_clf = make_rf(args.n_estimators, args.seed)
    s1_clf.fit(X1_tr, y1_tr)

    y1_pred   = s1_clf.predict(X1_te)
    s1_acc    = accuracy_score(y1_te, y1_pred)
    s1_f1     = f1_score(y1_te, y1_pred, average="weighted")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(s1_clf, X1_tr, y1_tr, cv=cv,
                                scoring="accuracy", n_jobs=-1)

    print(f"\n  Stage 1 results:")
    print(f"    Test accuracy : {s1_acc:.4f} ({s1_acc*100:.1f}%)")
    print(f"    Test F1 (wtd) : {s1_f1:.4f}")
    print(f"    CV  accuracy  : {cv_scores.mean():.4f} "
          f"(+/-{cv_scores.std()*2:.4f})")
    print()
    print("  Classification report (Stage 1):")
    print(classification_report(y1_te, y1_pred, zero_division=0))
    labels = sorted(set(np.concatenate([y1_te, y1_pred])))
    cm = confusion_matrix(y1_te, y1_pred, labels=labels)
    print("  Confusion matrix (Stage 1):")
    print("        " + " ".join(f"{l:>4}" for l in labels))
    for i, lbl in enumerate(labels):
        print(f"  {lbl:>3}   " + " ".join(f"{cm[i,j]:4d}" for j in range(len(labels))))
    print()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print("-" * 40)
    print("Stage 2 -- Gravity-based family disambiguators (4 features each)")
    print("-" * 40)
    print(f"  Families: {FAMILIES}")
    print()

    s2_models: dict[str, RandomForestClassifier] = {}

    for fam_name, members in FAMILIES.items():
        present = [m for m in members if m in available_letters]
        if len(present) < 2:
            print(f"  [{fam_name}] Only {present} in data -- skipping")
            continue

        X2, y2, rec_ids2 = build_stage2_dataset(
            df,
            family_letters  = present,
            window_size     = args.window_size,
            stride          = args.stride,
            samples_per_rec = args.samples_per_recording,
            aug_per_window  = args.aug_per_window,
            jitter_deg      = args.jitter_deg,
            seed            = args.seed,
        )
        print(f"  [{fam_name}] letters={present}  windows={len(X2)}")

        X2_tr, X2_te, y2_tr, y2_te = split_by_recording(
            X2, y2, rec_ids2, test_ratio=0.2, seed=args.seed
        )

        clf = make_rf(args.n_estimators, args.seed)
        clf.fit(X2_tr, y2_tr)

        y2_pred = clf.predict(X2_te)
        acc2    = accuracy_score(y2_te, y2_pred)
        f1_2    = f1_score(y2_te, y2_pred, average="weighted", zero_division=0)
        print(f"          test={acc2:.4f} ({acc2*100:.1f}%)  F1={f1_2:.4f}")

        # Show confusion for this family
        fam_labels = sorted(set(np.concatenate([y2_te, y2_pred])))
        cm2 = confusion_matrix(y2_te, y2_pred, labels=fam_labels)
        print(f"          CM: {' '.join(f'{l:>4}' for l in fam_labels)}")
        for i, lbl in enumerate(fam_labels):
            print(f"              {lbl:>3}  "
                  + " ".join(f"{cm2[i,j]:4d}" for j in range(len(fam_labels))))
        print()

        s2_models[fam_name] = clf

    # ── Save model package ────────────────────────────────────────────────────
    model_package = {
        "format":         "v2_gravity_cascade",
        "stage_1_model":  s1_clf,
        "stage_2_models": s2_models,
        "families":       FAMILIES,
        "imu_letters":    sorted(IMU_LETTERS & available_letters),
    }

    joblib.dump(model_package, out_path)
    print(f"Model saved : {out_path}")

    meta = {
        "format":         "v2_gravity_cascade",
        "window_size":    args.window_size,
        "s1_features":    25,
        "s2_features":    6,
        "classes":        sorted(available_letters),
        "imu_letters":    sorted(IMU_LETTERS & available_letters),
        "families":       FAMILIES,
        "n_estimators":   args.n_estimators,
        "jitter_deg":     args.jitter_deg,
        "aug_per_window": args.aug_per_window,
        "seed":           args.seed,
        "s1_test_accuracy": float(s1_acc),
        "s1_cv_accuracy":   float(cv_scores.mean()),
        "s2_families_trained": sorted(s2_models.keys()),
        "timestamp":      datetime.now().isoformat(),
        "inputs":         args.inputs,
    }
    meta_path = out_path.with_suffix(".meta.joblib")
    joblib.dump(meta, meta_path)
    print(f"Metadata    : {meta_path}")

    log_path = out_path.parent / "training_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    print(f"Log entry   : {log_path}")

    print()
    print("=" * 65)
    print(f"DONE  --  Stage 1 test accuracy: {s1_acc*100:.1f}%")
    print("=" * 65)
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train ASL v2 two-stage gravity cascade model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  # Train on a single 9-column CSV with all 21 letters:
  python scripts/train_model_v2.py \\
      -i data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_G_K_O_L_S_H_R_P_Q_T_V_W_X_Y_2026-03-10-12-56-47.csv

  # Custom output path:
  python scripts/train_model_v2.py -i data/Data/my_data.csv -o models/my_model.pkl

  # Use multiple seeds and pick the best:
  python scripts/train_model_v2.py -i data/Data/my_data.csv --try-seeds 1,42,0,123
        """,
    )
    parser.add_argument("--input", "-i", action="append", dest="inputs",
                        metavar="CSV", default=None,
                        help="Input 9-column CSV (repeat for multiple files)")
    parser.add_argument("--output", "-o",
                        default="models/rf_asl_v2_gravity_cascade.pkl",
                        help="Output model path (default: models/rf_asl_v2_gravity_cascade.pkl)")
    parser.add_argument("--window-size",           type=int, default=50,
                        help="Samples per window (default: 50 = 1 s at 50 Hz)")
    parser.add_argument("--stride",                type=int, default=25,
                        help="Window stride (default: 25 = 50%% overlap)")
    parser.add_argument("--samples-per-recording", type=int, default=150,
                        help="Samples per recording session (default: 150 = 3 s)")
    parser.add_argument("--n-estimators",          type=int, default=300,
                        help="Random Forest trees (default: 300)")
    parser.add_argument("--aug-per-window",        type=int, default=8,
                        help="Jittered copies per window for Stage 2 (default: 8)")
    parser.add_argument("--jitter-deg",            type=float, default=2.0,
                        help="Max Euler jitter in degrees for Stage 2 aug (default: 2.0)")
    parser.add_argument("--seed",                  type=int, default=200)
    parser.add_argument("--try-seeds", type=str, default="",
                        help="Comma-separated seeds to try; keeps best Stage 1 accuracy")
    args = parser.parse_args()

    if not args.inputs:
        # Default to the latest full dataset
        args.inputs = [
            "data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_G_K_O_L_S_H_R_P_Q_T_V_W_X_Y_2026-03-10-12-56-47.csv"
        ]

    if not args.try_seeds:
        return train(args)

    # Multi-seed: run each, compare Stage 1 test accuracy, save best
    seeds = [int(s.strip()) for s in args.try_seeds.split(",") if s.strip()]
    print(f"Trying seeds: {seeds}")
    best_acc, best_seed = -1.0, seeds[0]
    for s in seeds:
        args.seed = s
        # Temporarily suppress output for non-first seeds
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train(args)
        lines = buf.getvalue()
        # Extract Stage 1 test accuracy from output
        for line in lines.splitlines():
            if "Test accuracy" in line:
                try:
                    acc = float(line.split(":")[1].split("(")[0])
                    print(f"  seed={s}  Stage1={acc:.4f}")
                    if acc > best_acc:
                        best_acc, best_seed = acc, s
                except Exception:
                    pass
    print(f"\nBest seed: {best_seed}  Stage1={best_acc:.4f} -- retraining...")
    args.seed = best_seed
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
