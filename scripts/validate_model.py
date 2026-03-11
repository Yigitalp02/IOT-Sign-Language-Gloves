#!/usr/bin/env python3
"""
Proper k-fold cross-validation by recording session.

Splits the 20 recording sessions per letter into k folds and rotates the
test fold — every session appears in the test set exactly once. This is
the only way to get a trustworthy accuracy estimate when your dataset is
structured as multiple sessions per class.

A single 80/20 random split can give wildly different numbers depending
on which 4 sessions land in the test set (we saw 77%-100% for DG just by
changing the random seed). k-fold gives the average AND the variance,
so you know how reliable the number is.

Usage:
    python scripts/validate_model.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from train_model_v2 import (
    load_data, build_stage1_dataset, build_stage2_dataset,
    make_rf, FAMILIES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = str(PROJECT_ROOT / "data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_G_K_O_L_S_H_R_P_Q_T_V_W_X_Y_2026-03-10-12-56-47.csv")

SEED         = 200
K_FOLDS      = 5
WINDOW       = 50
STRIDE       = 25
SAMPLES_REC  = 150
N_TREES      = 300
AUG_PER_WIN  = 8
JITTER_DEG   = 2.0


def kfold_by_recording(X, y, rec_ids, k=5, seed=42):
    """
    Yield (X_train, X_test, y_train, y_test) for each fold.

    Groups windows by their (letter, rec_idx) pair.  Within each class,
    recording sessions are split into k roughly-equal buckets.  Each fold
    holds out one bucket from every class as the test set, training on the
    rest.  This ensures:
      - Every session appears in test exactly once
      - Class balance is preserved across folds
      - No window from the same session is split between train and test
    """
    rng = np.random.default_rng(seed)
    rec_ids_arr = np.array(rec_ids)

    # Group indices by class then by recording
    from collections import defaultdict
    class_recs: dict[str, list] = defaultdict(list)
    for i, (letter, rec_idx) in enumerate(rec_ids):
        class_recs[letter].append((rec_idx, i))

    # For each class, assign recording sessions to folds round-robin
    fold_indices: list[list[int]] = [[] for _ in range(k)]
    for letter, pairs in class_recs.items():
        # Unique sessions for this class, shuffled
        unique_sessions = sorted(set(r for r, _ in pairs))
        rng.shuffle(unique_sessions)
        session_to_fold = {s: i % k for i, s in enumerate(unique_sessions)}
        for rec_idx, win_idx in pairs:
            fold_indices[session_to_fold[rec_idx]].append(win_idx)

    for fold in range(k):
        test_idx  = np.array(fold_indices[fold])
        train_idx = np.array([i for f in range(k) if f != fold
                               for i in fold_indices[f]])
        yield (X[train_idx], X[test_idx],
               y[train_idx], y[test_idx])


def run_kfold(name, X, y, rec_ids, k, seed):
    """Run k-fold CV, print per-fold + summary statistics."""
    print(f"\n  {name}  ({k}-fold CV by recording session, seed={seed})")
    print(f"  {'Fold':>5}  {'Accuracy':>8}  {'N_test':>6}")
    print("  " + "-" * 25)

    accs = []
    all_true, all_pred = [], []

    for fold_i, (X_tr, X_te, y_tr, y_te) in enumerate(
            kfold_by_recording(X, y, rec_ids, k=k, seed=seed)):
        clf = make_rf(N_TREES, seed + fold_i)   # slightly varied seed per fold
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        accs.append(acc)
        all_true.extend(y_te)
        all_pred.extend(y_pred)
        print(f"  {fold_i+1:>5}  {acc:>8.4f}  {len(y_te):>6}")

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    print(f"  {'mean':>5}  {mean_acc:>8.4f}")
    print(f"  {'std' :>5}  {std_acc :>8.4f}  (low std = result is reliable)")

    # Full classification report across all folds
    labels = sorted(set(all_true))
    print(f"\n  Classification report (all folds combined):")
    print(classification_report(all_true, all_pred, labels=labels,
                                zero_division=0,
                                target_names=[str(l) for l in labels]))

    if len(labels) <= 5:
        cm = confusion_matrix(all_true, all_pred, labels=labels)
        print(f"  Confusion matrix:")
        print("         " + "  ".join(f"{l:>4}" for l in labels))
        for i, lbl in enumerate(labels):
            print(f"  {lbl:>5}   " + "  ".join(f"{cm[i,j]:4d}" for j in range(len(labels))))

    return mean_acc, std_acc


print("=" * 65)
print("Model Validation -- k-Fold Cross-Validation by Recording Session")
print("=" * 65)
print(f"Dataset : {Path(CSV).name}")
print(f"k       : {K_FOLDS}  (each fold = 1/5 of recording sessions per class)")
print(f"Seed    : {SEED}")
print()

print("Loading data ...")
df = load_data([CSV])
available = set(df["label"].unique())
print()

# ── Stage 1 ───────────────────────────────────────────────────────────────────
print("=" * 40)
print("STAGE 1 -- Flex-only (25 features, all 21 letters)")
print("=" * 40)
X1, y1, r1 = build_stage1_dataset(df, WINDOW, STRIDE, SAMPLES_REC)
s1_mean, s1_std = run_kfold("Stage 1", X1, y1, r1, K_FOLDS, SEED)

# ── Stage 2 per family ────────────────────────────────────────────────────────
print()
print("=" * 40)
print("STAGE 2 -- Gravity disambiguation (6 features, per family)")
print("=" * 40)

s2_results = {}
for fam_name, members in FAMILIES.items():
    present = [m for m in members if m in available]
    if len(present) < 2:
        print(f"\n  [{fam_name}] skipped (not enough letters in data)")
        continue
    X2, y2, r2 = build_stage2_dataset(
        df, present, WINDOW, STRIDE, SAMPLES_REC, AUG_PER_WIN, JITTER_DEG, SEED
    )
    mean2, std2 = run_kfold(f"Stage 2 [{fam_name}] {present}",
                             X2, y2, r2, K_FOLDS, SEED)
    s2_results[fam_name] = (mean2, std2)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Stage 1       : {s1_mean:.4f}  (+/- {s1_std:.4f})")
for fam, (m, s) in s2_results.items():
    reliability = "reliable" if s < 0.05 else "VARIABLE -- more data needed"
    print(f"  Stage 2 [{fam:>3}] : {m:.4f}  (+/- {s:.4f})  <- {reliability}")
print()
print("Interpretation:")
print("  mean  = expected real-world accuracy (not a lucky split)")
print("  std   = how much the result varies across different test sets")
print("  std < 0.03 = trustworthy;  std > 0.07 = need more data or re-recording")
