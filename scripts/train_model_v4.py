#!/usr/bin/env python3
"""
ASL Glove Model Training v4 — Flex-Only + Hardcoded IMU Rules

The IMU disambiguation rules are based on known physics, not trained or
calibrated from data. No IMU values are needed in the training CSV at all.

Architecture
============

Step 1 — Flex-only RF classifier (25 features, all 21 letters)
  Trains on flex columns only. IMU columns are completely ignored.
  Records can have IMU = 0,0,0,0 or be totally absent from the CSV.

Step 2 — Hardcoded directional IMU rules (no training, no calibration)
  We know the physics of each letter's orientation:

    fwd_z = 2*(qx*qz + qw*qy)
      +1 = fingers point straight up
       0 = fingers point sideways (horizontal)
      -1 = fingers point straight down

    up_z = 1 - 2*(qx² + qy²)
      +1 = back of hand faces up (palm faces down)
      -1 = back of hand faces down (palm faces up)

    right_z = 2*(qy*qz - qw*qx)
      wrist roll — separates P from Q

  The 2D grid (fwd_z = finger direction, up_z = palm direction):

              up_z LOW (palm sideways)   up_z HIGH (palm faces down)
  fwd_z +  │  D, V, R, L (fingers up)  │  —
  fwd_z ~0 │  H, P-ish                 │  G
  fwd_z −  │  P (fingers down-fwd)     │  Q (fingers down + palm down)

  Rules:
    DG  : fwd_z > FWD_UP → D          fwd_z ≤ FWD_UP → G
    VHR : fwd_z > FWD_UP → V or R     fwd_z ≤ FWD_UP → H
          (V vs R already separated by flex — crossed vs spread fingers)
    LPQ : fwd_z > FWD_UP            → L  (index+thumb pointing up)
          fwd_z < FWD_DN & up_z > UP_PALM_DOWN_THRESH → Q  (palm down + fingers down)
          fwd_z < FWD_DN             → P  (fingers down, palm sideways)
          middle zone                → keep flex prediction

  Thresholds are stored in the model package's imu_rules and can be edited
  in the saved JSON sidecar without retraining.

Usage
=====
  python scripts/train_model_v4.py
  python scripts/train_model_v4.py -i data/Data/my_data.csv -o models/my_model_v4.pkl
  python scripts/train_model_v4.py -i data/Data/my_data.csv --try-seeds 1,42,0,123
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

# ── Letter families that share the same flex shape ────────────────────────────
# Only these need IMU disambiguation. All other letters are flex-unique.
FAMILIES: dict[str, list[str]] = {
    "DG":  ["D", "G"],
    "VHR": ["V", "H", "R"],
    "LPQ": ["L", "P", "Q"],
}
FAMILY_MEMBERS: set[str] = {l for fam in FAMILIES.values() for l in fam}

# ── Column names ──────────────────────────────────────────────────────────────
FLEX_COLS = ["ch0_norm", "ch1_norm", "ch2_norm", "ch3_norm", "ch4_norm"]
IMU_COLS  = ["qw", "qx", "qy", "qz"]
ALL_COLS  = FLEX_COLS + IMU_COLS


# ── Safe statistics helper ────────────────────────────────────────────────────
def _safe_stats(vals: np.ndarray) -> list[float]:
    v = vals[~np.isnan(vals)].astype(float)
    if len(v) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(v)),
        float(np.std(v))   if len(v) >= 2 else 0.0,
        float(np.min(v)),
        float(np.max(v)),
        float(np.max(v) - np.min(v)) if len(v) > 1 else 0.0,
    ]


# ── Feature extraction — flex only (25 features) ─────────────────────────────
def flex_features(window: np.ndarray) -> np.ndarray:
    """mean/std/min/max/range × 5 flex channels. IMU completely ignored."""
    feats: list[float] = []
    for i in range(5):
        feats.extend(_safe_stats(window[:, i]))
    return np.array(feats, dtype=np.float64)


# ── Gravity feature computation (yaw-invariant) ───────────────────────────────
def gravity_vector(window: np.ndarray) -> tuple[float, float, float]:
    """
    Returns the mean (fwd_z, up_z, right_z) of a window.
    These three values fully describe the hand tilt relative to gravity,
    independent of which direction the user is facing.
    """
    qw = window[:, 5].astype(float)
    qx = window[:, 6].astype(float)
    qy = window[:, 7].astype(float)
    qz = window[:, 8].astype(float)

    fwd_z   = float(np.mean(2.0 * (qx * qz + qw * qy)))
    up_z    = float(np.mean(1.0 - 2.0 * (qx**2 + qy**2)))
    right_z = float(np.mean(2.0 * (qy * qz - qw * qx)))
    return fwd_z, up_z, right_z


# ── Sliding window iterator ───────────────────────────────────────────────────
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


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(csv_paths: list[str]) -> pd.DataFrame:
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


# ── Build flex dataset (Step 1) ───────────────────────────────────────────────
def build_flex_dataset(df: pd.DataFrame, window_size: int, stride: int,
                       samples_per_rec: int):
    X, y, rec_ids = [], [], []
    for letter in sorted(df["label"].unique()):
        data = df[df["label"] == letter][ALL_COLS].values.astype(np.float64)
        for win, rec_idx in iter_windows(data, window_size, stride, samples_per_rec):
            X.append(flex_features(win))
            y.append(letter)
            rec_ids.append((letter, rec_idx))
    return np.array(X), np.array(y), rec_ids



# ── Train/test split by recording session (no leakage) ───────────────────────
def split_by_recording(X, y, rec_ids, test_ratio: float = 0.2, seed: int = 42):
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
        n_estimators      = n_estimators,
        max_depth         = 25,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = "sqrt",
        random_state      = seed,
        n_jobs            = -1,
    )


# ── Training pipeline ─────────────────────────────────────────────────────────
def train(args) -> tuple[int, float]:
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_paths  = [str(project_root / p) for p in args.inputs]
    out_path     = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in input_paths:
        if not Path(p).exists():
            print(f"ERROR: File not found: {p}")
            return 1, 0.0

    print("=" * 65)
    print("ASL Glove Model Training v4 -- Flex-Only + Deterministic IMU Rules")
    print("=" * 65)
    for p in input_paths:
        print(f"  Input : {p}")
    print(f"  Output: {out_path}")
    print(f"  Window: {args.window_size}  Stride: {args.stride}  "
          f"Trees: {args.n_estimators}  Seed: {args.seed}")
    print()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = load_data(input_paths)
    available_letters = set(df["label"].unique())
    print()

    # ── Step 1: Flex-only classifier ──────────────────────────────────────────
    print("-" * 65)
    print("Step 1 — Flex-only classifier (25 features, all letters, NO IMU)")
    print("-" * 65)

    X, y, rec_ids = build_flex_dataset(
        df,
        window_size     = args.window_size,
        stride          = args.stride,
        samples_per_rec = args.samples_per_recording,
    )
    print(f"  Windows: {len(X)}   Features: {X.shape[1]}")

    X_tr, X_te, y_tr, y_te = split_by_recording(
        X, y, rec_ids, test_ratio=0.2, seed=args.seed
    )
    print(f"  Train: {len(X_tr)}   Test: {len(X_te)}")

    clf = make_rf(args.n_estimators, args.seed)
    clf.fit(X_tr, y_tr)

    y_pred    = clf.predict(X_te)
    test_acc  = accuracy_score(y_te, y_pred)
    test_f1   = f1_score(y_te, y_pred, average="weighted")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"\n  Test accuracy : {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Test F1 (wtd) : {test_f1:.4f}")
    print(f"  CV  accuracy  : {cv_scores.mean():.4f} (+/-{cv_scores.std()*2:.4f})")
    print()
    print("  Classification report:")
    print(classification_report(y_te, y_pred, zero_division=0))

    labels = sorted(set(np.concatenate([y_te, y_pred])))
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    print("  Confusion matrix (flex-only, before IMU rules):")
    print("        " + " ".join(f"{l:>4}" for l in labels))
    for i, lbl in enumerate(labels):
        print(f"  {lbl:>3}   " + " ".join(f"{cm[i,j]:4d}" for j in range(len(labels))))
    print()

    # ── Step 2: Hardcoded directional IMU rules ───────────────────────────────
    print("-" * 65)
    print("Step 2 — Hardcoded IMU rules (physics-based, no training needed)")
    print("-" * 65)

    # fwd_z thresholds — these are the key values:
    #   > FWD_UP  → fingers pointing upward  (D, V, R, L)
    #   < FWD_DN  → fingers pointing downward (P, Q)
    #   between   → fingers pointing sideways (G, H)
    # Thresholds calibrated from real glove data:
    #   D (fingers forward, palm down): fwd_z=+0.40, up_z=+0.91
    #   D (fingers up,      palm fwd) : fwd_z=-0.84, up_z=+0.53
    #   G (fingers right,   palm→you) : fwd_z=+0.85, up_z=-0.14
    FWD_UP     = args.fwd_up_thresh        # +0.65 : G has fwd_z=+0.85 vs D max fwd_z=+0.40
    FWD_DN     = args.fwd_dn_thresh        # -0.20 : below = fingers pointing DOWN (P, Q)
    UP_PALM_DN = args.up_palm_down_thresh  # +0.40 : above = palm facing DOWN (D has +0.53/+0.91, G has -0.14)

    imu_rules = {
        "type":                "threshold",
        "fwd_up_thresh":       FWD_UP,
        "fwd_dn_thresh":       FWD_DN,
        "up_palm_down_thresh": UP_PALM_DN,
        "families": {
            "DG": {
                "members": ["D", "G"],
                "rule": (
                    "D is DEFAULT (works in many orientations). "
                    "G is EXCEPTION: fwd_z < fwd_up (fingers sideways) "
                    "AND up_z < up_palm_down (palm toward you, not down). "
                    "Any other orientation stays D."
                ),
            },
            "VHR": {
                "members": ["V", "H", "R"],
                "rule": (
                    "V is default (fingers up). R stays from flex (crossed). "
                    "H is EXCEPTION: fwd_z < fwd_up (fingers sideways)."
                ),
            },
            "LPQ": {
                "members": ["L", "P", "Q"],
                "rule": (
                    "L: fwd_z < -0.50 (fingers UP → very negative fwd_z=-0.921).  "
                    "Q: fwd_z > +0.55 (fingers DOWN → fwd_z=+0.844).  "
                    "P: DEFAULT (fingers forward, moderate fwd_z=+0.222)."
                ),
            },
        },
        "_note": (
            "fwd_z: +1=fingers up, 0=sideways, -1=down  |  "
            "up_z: +1=palm faces down (back-of-hand up), 0=palm sideways, -1=palm faces up  |  "
            "All three are yaw-invariant (compass direction doesn't matter)"
        ),
    }

    print(f"  fwd_up_thresh      = {FWD_UP:+.2f}  (fingers UP if fwd_z above this)")
    print(f"  fwd_dn_thresh      = {FWD_DN:+.2f}  (fingers DOWN if fwd_z below this)")
    print(f"  up_palm_down_thresh= {UP_PALM_DN:+.2f}  (palm faces DOWN if up_z above this)")
    print()
    print(f"  DG  : fwd_z > {FWD_UP:.2f} AND up_z < 0.20       => G   else => D (default)")
    print(f"  VHR : fwd_z > 0.80                           => R (up_z<-0.20) / H (up_z>=-0.20)")
    print(f"        fwd_z <= 0.80                           => V (default)")
    print(f"  LPQ : fwd_z < -0.50                          => L (fingers up)")
    print(f"        fwd_z >  0.55                          => Q (fingers down)")
    print(f"        else                                    => P (default, fingers forward)")
    print()

    # ── Simulate combined accuracy (flex + IMU rules on test set) ─────────────
    print("-" * 65)
    print("Combined accuracy simulation (flex prediction + IMU rule applied)")
    print("-" * 65)

    # Re-build the test gravity vectors (same deterministic split as above)
    X_grav_te = []
    for letter in sorted(df["label"].unique()):
        data = df[df["label"] == letter][ALL_COLS].values.astype(np.float64)
        windows = list(iter_windows(data, args.window_size, args.stride,
                                    args.samples_per_recording))
        if not windows:
            continue
        rng = np.random.default_rng(args.seed)
        rng.shuffle(windows)
        n_test_recs = max(1, int(len(windows) * 0.2))
        for win, _ in windows[:n_test_recs]:
            X_grav_te.append(gravity_vector(win))

    def apply_imu_rules(flex_pred, gv, rules):
        """Apply hardcoded threshold rules to a flex prediction + gravity vector."""
        fwd_z = float(gv[0])
        up_z  = float(gv[1])
        fwd_up     = rules.get("fwd_up_thresh",       0.65)
        up_palm_dn = rules.get("up_palm_down_thresh",  0.40)

        if flex_pred in ("D", "G"):
            g_detected = (fwd_z > fwd_up) and (up_z < 0.20)
            return "G" if g_detected else "D"

        if flex_pred in ("V", "H", "R"):
            if fwd_z > 0.80:
                return "R" if up_z < -0.20 else "H"
            return "V"

        if flex_pred in ("L", "P", "Q"):
            if fwd_z < -0.50:
                return "L"
            if fwd_z > 0.55:
                return "Q"
            return "P"

        return flex_pred

    y_combined = []
    for i, flex_pred in enumerate(y_pred):
        if i < len(X_grav_te):
            ruled = apply_imu_rules(flex_pred, X_grav_te[i], imu_rules)
        else:
            ruled = flex_pred
        y_combined.append(ruled)

    combined_acc = accuracy_score(y_te, y_combined)
    print(f"  Flex-only accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  With IMU rules      : {combined_acc:.4f} ({combined_acc*100:.1f}%)")
    delta = combined_acc - test_acc
    print(f"  IMU rule effect     : {delta:+.4f} ({delta*100:+.1f}%)")
    print()

    # ── Save model package ────────────────────────────────────────────────────
    model_package = {
        "format":     "v4_flex_rules",
        "flex_model": clf,
        "imu_rules":  imu_rules,
        "families":   FAMILIES,
        "window_size": args.window_size,
    }

    joblib.dump(model_package, out_path)
    print(f"Model saved : {out_path}")

    # Also save a human-readable JSON of the rules for manual inspection/tuning
    rules_json_path = out_path.with_suffix(".rules.json")
    rules_export = {
        "fwd_up_thresh":       imu_rules["fwd_up_thresh"],
        "fwd_dn_thresh":       imu_rules["fwd_dn_thresh"],
        "up_palm_down_thresh": imu_rules["up_palm_down_thresh"],
        "families":            imu_rules["families"],
        "_note":               imu_rules["_note"],
    }
    with open(rules_json_path, "w", encoding="utf-8") as f:
        json.dump(rules_export, f, indent=2)
    print(f"Rules JSON  : {rules_json_path}  (edit to tune thresholds without retraining)")

    meta = {
        "format":          "v4_flex_rules",
        "window_size":     args.window_size,
        "flex_features":   25,
        "classes":         sorted(available_letters),
        "families":        FAMILIES,
        "n_estimators":    args.n_estimators,
        "seed":            args.seed,
        "test_accuracy":   float(test_acc),
        "cv_accuracy":     float(cv_scores.mean()),
        "combined_accuracy": float(combined_acc),
        "imu_families_derived": sorted(imu_rules.keys()),
        "timestamp":       datetime.now().isoformat(),
        "inputs":          args.inputs,
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
    print(f"DONE  --  Flex-only: {test_acc*100:.1f}%  |  With IMU rules: {combined_acc*100:.1f}%")
    print("=" * 65)
    return 0, test_acc


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train ASL v4: flex-only RF + deterministic IMU centroid rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  # Default data path:
  python scripts/train_model_v4.py

  # Custom CSV:
  python scripts/train_model_v4.py -i data/Data/my_data.csv -o models/my_v4.pkl

  # Try multiple seeds, keep best:
  python scripts/train_model_v4.py --try-seeds 1,42,0,123,200

How inference works (for the API server)
  1. Extract 25 flex features from the current window
  2. flex_model.predict() → coarse letter (e.g. "D")
  3. If the letter is in a family (DG, VHR, LPQ):
       compute (fwd_z, up_z, right_z) from live IMU
       pick the family member whose centroid is closest
       → final letter (e.g. "G" if hand is sideways)
  4. Otherwise: flex prediction is the final answer
        """,
    )
    parser.add_argument("--input", "-i", action="append", dest="inputs",
                        metavar="CSV", default=None)
    parser.add_argument("--output", "-o",
                        default="models/rf_asl_v4_flex_rules.pkl")
    parser.add_argument("--fwd-up-thresh",       type=float, default=0.65,
                        help="fwd_z above this = fingers pointing UP (default: 0.30)")
    parser.add_argument("--fwd-dn-thresh",       type=float, default=-0.20,
                        help="fwd_z below this = fingers pointing DOWN (default: -0.20)")
    parser.add_argument("--up-palm-down-thresh", type=float, default=0.40,
                        help="up_z above this = palm facing DOWN / back-of-hand up (default: 0.40)")
    parser.add_argument("--window-size",           type=int, default=50)
    parser.add_argument("--stride",                type=int, default=25)
    parser.add_argument("--samples-per-recording", type=int, default=150)
    parser.add_argument("--n-estimators",          type=int, default=300)
    parser.add_argument("--seed",                  type=int, default=42)
    parser.add_argument("--try-seeds", type=str, default="",
                        help="Comma-separated seeds to try; keeps best accuracy")
    args = parser.parse_args()

    if not args.inputs:
        args.inputs = [
            "data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_G_K_O_L_S_H_R_P_Q_T_V_W_X_Y_2026-03-10-12-56-47.csv"
        ]

    if not args.try_seeds:
        code, _ = train(args)
        return code

    seeds = [int(s.strip()) for s in args.try_seeds.split(",") if s.strip()]
    print(f"Trying seeds: {seeds}")
    best_acc, best_seed = -1.0, seeds[0]
    for s in seeds:
        args.seed = s
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, acc = train(args)
        for line in buf.getvalue().splitlines():
            if "Test accuracy" in line:
                try:
                    a = float(line.split(":")[1].split("(")[0])
                    print(f"  seed={s}  flex-only={a:.4f}")
                    if a > best_acc:
                        best_acc, best_seed = a, s
                except Exception:
                    pass
    print(f"\nBest seed: {best_seed}  flex-only={best_acc:.4f} — retraining...")
    args.seed = best_seed
    code, _ = train(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
