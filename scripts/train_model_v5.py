#!/usr/bin/env python3
"""
ASL Glove Model Training v5 — Expanded Multi-Family Cascade

What's different from v3
========================
v3 had 3 disambiguation families: DG, VHR, LPQ
v5 expands to 5 families:

  VHRU : V, H, R, U  — two-finger extended shapes; wrist orientation separates them
  AT   : A, T        — closed fists; gravity separates thumb placement
  ES   : E, S        — closed fists; gravity separates orientation
  DG   : D, G        — index-up vs index-sideways
  LPQ  : L, P, Q     — thumb+index; wrist roll separates them

Key design principle (inherited from v3)
  Stage 1  — flex only (25 features, collapsed super-labels)
             Family members share a super-label → Stage 1 never has to separate
             letters with identical flex patterns.
  Stage 2  — gravity features (6 features, yaw-invariant) per family
             fwd_z, up_z, right_z (mean+std each) distinguish wrist orientations.

Selective data loading
======================
OLD CSV → keep only letters already good in v3: C, F, I, O, X, Y, K, D, G
NEW CSV → keep only re-recorded / new letters:  B, W, V, H, R, S, E, A, T, U, L, P, Q

This replaces badly-recorded letters with fresh recordings while keeping
the already-accurate letters from the previous dataset.

Model format string: "v5_multifamily"

Usage
=====
  python scripts/train_model_v5.py
  python scripts/train_model_v5.py -o models/my_v5_test.pkl
  python scripts/train_model_v5.py --n-estimators 500 --aug-per-window 12
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ── Letter families ────────────────────────────────────────────────────────────
FAMILIES: dict[str, list[str]] = {
    "VHRU": ["V", "H", "R", "U"],
    "AT":   ["A", "T"],
    "ES":   ["E", "S"],
    "DG":   ["D", "G"],
    "LPQ":  ["L", "P", "Q"],
}

FAMILY_LABEL_MAP: dict[str, str] = {
    letter: fam_name
    for fam_name, members in FAMILIES.items()
    for letter in members
}

# ── Column names ───────────────────────────────────────────────────────────────
FLEX_COLS = ["ch0_norm", "ch1_norm", "ch2_norm", "ch3_norm", "ch4_norm"]
IMU_COLS  = ["qw", "qx", "qy", "qz"]
ALL_COLS  = FLEX_COLS + IMU_COLS

# ── Default data sources (selectively pick letters from each file) ─────────────
_OLD_CSV = (
    "data/Data/"
    "glove_data_NORMALIZED_A_B_C_D_E_F_I_G_K_O_L_S_H_R_P_Q_T_V_W_X_Y_"
    "2026-03-10-12-56-47.csv"
)
_NEW_CSV = (
    "data/Data/"
    "glove_data_NORMALIZED_B_W_V_H_R_S_E_A_T_U_L_P_Q_"
    "2026-04-03-11-58-32.csv"
)
# Letters that are already accurate in the old dataset — keep from old CSV
_OLD_LETTERS = ["C", "F", "I", "O", "X", "Y", "K", "D", "G"]
# Re-recorded / new letters — take from the new CSV
_NEW_LETTERS = ["B", "W", "V", "H", "R", "S", "E", "A", "T", "U", "L", "P", "Q"]


# ── Feature extraction ─────────────────────────────────────────────────────────
def _safe_stats(vals: np.ndarray) -> list[float]:
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
    """25 features — mean/std/min/max/range of the 5 flex channels only."""
    feats: list[float] = []
    for i in range(5):
        feats.extend(_safe_stats(window[:, i]))
    return np.array(feats, dtype=np.float64)


def gravity_features(window: np.ndarray) -> np.ndarray:
    """6 yaw-invariant gravity features (same as v3 Stage 2).

    fwd_z   = 2*(qx*qz + qw*qy)   — fingers pointing up/down
    up_z    = 1 - 2*(qx^2 + qy^2) — back-of-hand facing up/down
    right_z = 2*(qy*qz - qw*qx)   — wrist roll

    Features: mean + std of each = 6 total.
    Yaw-invariant: only tilt relative to gravity matters.
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


# ── Quaternion jitter for Stage 2 augmentation ────────────────────────────────
def _jitter_quaternion(qw, qx, qy, qz, max_deg: float,
                       rng: np.random.Generator):
    d = np.radians(max_deg)
    rx, ry, rz = rng.uniform(-d, d, 3)

    def axis_quat(angle, axis):
        h = angle / 2.0
        c, s = np.cos(h), np.sin(h)
        return (c,
                s if axis == 0 else 0.0,
                s if axis == 1 else 0.0,
                s if axis == 2 else 0.0)

    def qmul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw)

    j = qmul(qmul(axis_quat(rx, 0), axis_quat(ry, 1)), axis_quat(rz, 2))
    nw, nx, ny, nz = qmul((qw, qx, qy, qz), j)
    n = np.sqrt(nw**2 + nx**2 + ny**2 + nz**2) or 1.0
    return nw/n, nx/n, ny/n, nz/n


def jitter_window_imu(window: np.ndarray, max_deg: float,
                      rng: np.random.Generator) -> np.ndarray:
    w = window.copy()
    for i in range(len(w)):
        w[i, 5], w[i, 6], w[i, 7], w[i, 8] = _jitter_quaternion(
            w[i, 5], w[i, 6], w[i, 7], w[i, 8], max_deg, rng
        )
    return w


# ── Selective data loading ─────────────────────────────────────────────────────
def load_selective(
    file_letter_pairs: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    """Load multiple CSV files keeping only the specified letters from each.

    Args:
        file_letter_pairs: [(csv_path, [letters_to_keep]), ...]
            Pass an empty list for letters_to_keep to keep all letters.
    """
    frames = []
    for path, keep_letters in file_letter_pairs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        missing = [c for c in ["label"] + ALL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name}: missing columns {missing}")
        if keep_letters:
            before = len(df)
            df = df[df["label"].isin(keep_letters)]
            skipped = sorted(set(pd.read_csv(path, usecols=["label"])["label"].unique()) - set(keep_letters))
            print(f"  {p.name}")
            print(f"    kept   : {sorted(df['label'].unique())}  ({len(df):,} rows)")
            if skipped:
                print(f"    skipped: {skipped}")
        else:
            print(f"  {p.name}: all {len(df):,} rows kept")
        frames.append(df[["label"] + ALL_COLS].copy())

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} rows, "
          f"{combined['label'].nunique()} letters: {sorted(combined['label'].unique())}")
    return combined


# ── Windowed extraction helpers ────────────────────────────────────────────────
def iter_windows(data: np.ndarray, window_size: int, stride: int,
                 samples_per_recording: int):
    n_recordings = max(1, len(data) // samples_per_recording)
    for rec_idx in range(n_recordings):
        start = rec_idx * samples_per_recording
        end   = min(start + samples_per_recording, len(data))
        rec   = data[start:end]
        if len(rec) < window_size:
            continue
        for i in range(0, len(rec) - window_size + 1, stride):
            yield rec[i: i + window_size], rec_idx


# ── Stage 1 dataset ────────────────────────────────────────────────────────────
def build_stage1_dataset(df: pd.DataFrame, window_size: int, stride: int,
                         samples_per_rec: int):
    """25 flex features per window. Family members share their super-label."""
    X, y, rec_ids = [], [], []
    for true_letter in sorted(df["label"].unique()):
        collapsed = FAMILY_LABEL_MAP.get(true_letter, true_letter)
        data = df[df["label"] == true_letter][ALL_COLS].values.astype(np.float64)
        for win, rec_idx in iter_windows(data, window_size, stride, samples_per_rec):
            X.append(flex_features(win))
            y.append(collapsed)
            rec_ids.append((collapsed, rec_idx))
    return np.array(X), np.array(y), rec_ids


# ── Stage 2 dataset ────────────────────────────────────────────────────────────
def build_stage2_dataset(df: pd.DataFrame, family_letters: list[str],
                         window_size: int, stride: int, samples_per_rec: int,
                         aug_per_window: int, jitter_deg: float, seed: int):
    """6 gravity features per window for one family, augmented with jitter."""
    rng = np.random.default_rng(seed)
    X, y, rec_ids = [], [], []
    for letter in family_letters:
        subset = df[df["label"] == letter]
        if len(subset) == 0:
            print(f"  [Stage 2] WARNING: no data for '{letter}' -- skipping")
            continue
        data = subset[ALL_COLS].values.astype(np.float64)
        for win, rec_idx in iter_windows(data, window_size, stride, samples_per_rec):
            for _ in range(aug_per_window):
                aug = jitter_window_imu(win, jitter_deg, rng)
                X.append(gravity_features(aug))
                y.append(letter)
                rec_ids.append((letter, rec_idx))
    return np.array(X), np.array(y), rec_ids


# ── Train/test split by recording session (no leakage) ────────────────────────
def split_by_recording(X, y, rec_ids, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    unique_recs = sorted(set(rec_ids))
    rng.shuffle(unique_recs)
    n_test    = max(1, int(len(unique_recs) * test_ratio))
    test_recs = set(unique_recs[:n_test])
    mask      = np.array([r not in test_recs for r in rec_ids])
    return X[mask], X[~mask], y[mask], y[~mask]


# ── Random Forest factory ──────────────────────────────────────────────────────
def make_rf(n_estimators: int, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = 25,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = "sqrt",
        random_state      = seed,
        n_jobs            = -1,
    )


# ── Training pipeline ──────────────────────────────────────────────────────────
def train(args) -> int:
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    out_path     = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ASL Glove Model Training v5 -- Expanded Multi-Family Cascade")
    print("=" * 65)
    print(f"  Output : {out_path}")
    print(f"  Window : {args.window_size}  Stride: {args.stride}  "
          f"Trees: {args.n_estimators}  Jitter: +/-{args.jitter_deg}deg  "
          f"Aug/win: {args.aug_per_window}")
    print()
    print(f"  Families ({len(FAMILIES)}):")
    for name, members in FAMILIES.items():
        print(f"    {name:6s}: {members}")
    print()

    # ── Load selectively ──────────────────────────────────────────────────────
    print("Loading data (selective per-file)...")
    file_letter_pairs = [
        (str(project_root / _OLD_CSV), _OLD_LETTERS),
        (str(project_root / _NEW_CSV), _NEW_LETTERS),
    ]
    df = load_selective(file_letter_pairs)
    available_letters = set(df["label"].unique())
    print()

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print("-" * 40)
    print("Stage 1 -- Collapsed flex classifier")
    print("  Family members share a super-label; Stage 1 never tries to")
    print("  separate flex-identical letters.")
    print("-" * 40)

    X1, y1, rec_ids1 = build_stage1_dataset(
        df,
        window_size     = args.window_size,
        stride          = args.stride,
        samples_per_rec = args.samples_per_recording,
    )
    collapsed_classes = sorted(set(y1))
    print(f"  Stage 1 classes ({len(collapsed_classes)}): {collapsed_classes}")
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

    print(f"\n  Stage 1 results (collapsed classes):")
    print(f"    Test accuracy : {s1_acc:.4f} ({s1_acc*100:.1f}%)")
    print(f"    Test F1 (wtd) : {s1_f1:.4f}")
    print(f"    CV  accuracy  : {cv_scores.mean():.4f} "
          f"(+/-{cv_scores.std()*2:.4f})")
    print()
    print("  Classification report (Stage 1 — collapsed):")
    print(classification_report(y1_te, y1_pred, zero_division=0))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print("-" * 40)
    print("Stage 2 -- Gravity-based family disambiguators (6 features each)")
    print("-" * 40)

    s2_models: dict[str, RandomForestClassifier] = {}

    for fam_name, members in FAMILIES.items():
        present = [m for m in members if m in available_letters]
        if len(present) < 2:
            print(f"  [{fam_name}] Only {present} present -- skipping")
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
        print(f"\n  [{fam_name}] letters={present}  windows={len(X2)}")

        X2_tr, X2_te, y2_tr, y2_te = split_by_recording(
            X2, y2, rec_ids2, test_ratio=0.2, seed=args.seed
        )
        print(f"    Train: {len(X2_tr)}  Test: {len(X2_te)}")

        clf = make_rf(args.n_estimators, args.seed)
        clf.fit(X2_tr, y2_tr)

        y2_pred = clf.predict(X2_te)
        acc2    = accuracy_score(y2_te, y2_pred)
        f1_2    = f1_score(y2_te, y2_pred, average="weighted", zero_division=0)
        print(f"    Test acc: {acc2:.4f} ({acc2*100:.1f}%)  F1: {f1_2:.4f}")
        print(classification_report(y2_te, y2_pred, zero_division=0,
                                    target_names=sorted(set(y2_te))))

        s2_models[fam_name] = clf

    # ── Save model ────────────────────────────────────────────────────────────
    model_package = {
        "format":           "v5_multifamily",
        "stage_1_model":    s1_clf,
        "stage_2_models":   s2_models,
        "families":         FAMILIES,
        "family_label_map": FAMILY_LABEL_MAP,
    }
    joblib.dump(model_package, out_path)
    print(f"\nModel saved : {out_path}")

    meta = {
        "format":               "v5_multifamily",
        "window_size":          args.window_size,
        "s1_features":          25,
        "s2_features":          6,
        "s1_classes":           collapsed_classes,
        "all_letters":          sorted(available_letters),
        "families":             {k: v for k, v in FAMILIES.items()},
        "family_label_map":     FAMILY_LABEL_MAP,
        "n_estimators":         args.n_estimators,
        "jitter_deg":           args.jitter_deg,
        "aug_per_window":       args.aug_per_window,
        "seed":                 args.seed,
        "s1_test_accuracy":     float(s1_acc),
        "s1_cv_accuracy":       float(cv_scores.mean()),
        "s2_families_trained":  sorted(s2_models.keys()),
        "timestamp":            datetime.now().isoformat(),
        "data_old_csv_letters": _OLD_LETTERS,
        "data_new_csv_letters": _NEW_LETTERS,
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
    print(f"DONE  --  Stage 1 test accuracy (collapsed): {s1_acc*100:.1f}%")
    print("  Non-family letters (B, C, F, I, K, O, W, X, Y) are")
    print("  orientation-invariant — confidence unaffected by wrist tilt.")
    print("=" * 65)
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train ASL v5 expanded multi-family cascade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", "-o",
                        default="models/rf_asl_v5_multifamily.pkl")
    parser.add_argument("--window-size",           type=int,   default=50)
    parser.add_argument("--stride",                type=int,   default=25)
    parser.add_argument("--samples-per-recording", type=int,   default=150)
    parser.add_argument("--n-estimators",          type=int,   default=300)
    parser.add_argument("--aug-per-window",        type=int,   default=8)
    parser.add_argument("--jitter-deg",            type=float, default=2.0)
    parser.add_argument("--seed",                  type=int,   default=200)
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
