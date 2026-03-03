#!/usr/bin/env python3
"""
Multi-algorithm comparison on the combined 21-letter dataset.

Tests: Random Forest, Gradient Boosting, SVM, KNN, MLP, ExtraTrees
Uses the same windowed features and recording-based train/test split
as train_model.py so results are directly comparable.

Usage:
  python scripts/compare_algorithms.py
"""

import sys
import time
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_model import load_combined_data, prepare_windows, split_by_recording

# ── Dataset ───────────────────────────────────────────────────────────────────
INPUTS = [
    "data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_K_O_S_T_V_W_X_Y_2026-02-26-12-30-42.csv",
    "data/Data/glove_data_NORMALIZED_B_W_D_K_G_H_L_P_Q_R_2026-03-02-08-59-19.csv",
]

# ── Classifiers to compare ────────────────────────────────────────────────────
def build_classifiers():
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, HistGradientBoostingClassifier,
    )
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    return [
        ("Random Forest (300)",
         RandomForestClassifier(n_estimators=300, max_depth=25,
                                min_samples_split=5, min_samples_leaf=2,
                                max_features="sqrt", random_state=0, n_jobs=-1)),

        ("Extra Trees (300)",
         ExtraTreesClassifier(n_estimators=300, max_depth=25,
                              min_samples_split=5, min_samples_leaf=2,
                              max_features="sqrt", random_state=0, n_jobs=-1)),

        ("Hist Gradient Boosting",
         HistGradientBoostingClassifier(max_iter=300, max_depth=10,
                                        learning_rate=0.1, random_state=0)),

        ("Gradient Boosting (100)",
         GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                    learning_rate=0.1, random_state=0)),

        ("SVM (RBF)",
         SVC(kernel="rbf", C=10, gamma="scale", random_state=0)),

        ("KNN (k=5)",
         KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)),

        ("MLP (256-128)",
         MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                       learning_rate_init=0.001, random_state=0)),
    ]


def main():
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    print("=" * 65)
    print("Algorithm Comparison — 21-letter ASL dataset (flex + IMU)")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    input_paths = [str(PROJECT_ROOT / p) for p in INPUTS]
    df = load_combined_data(input_paths)

    print("\nExtracting windowed features...")
    X, y, rec_ids = prepare_windows(df, window_size=50, stride=25)
    print(f"  Windows: {len(X)}  Features: {X.shape[1]}  Classes: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = split_by_recording(
        X, y, rec_ids, test_ratio=0.2, seed=0
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print()

    # ── Warn about dataset size ───────────────────────────────────────────────
    print("NOTE: Dataset is small (single user, single session).")
    print("High accuracy reflects within-session consistency,")
    print("not necessarily cross-user or cross-day generalisation.")
    print()

    # ── Run each classifier ───────────────────────────────────────────────────
    header = f"{'Algorithm':<28} {'Test Acc':>9} {'Test F1':>9} {'CV Acc':>9} {'Time':>8}"
    print(header)
    print("-" * 65)

    results = []
    classifiers = build_classifiers()

    for name, clf in classifiers:
        t0 = time.time()

        # 5-fold CV on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        try:
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv,
                                        scoring="accuracy", n_jobs=-1)
            cv_acc = cv_scores.mean()
        except Exception:
            cv_acc = float("nan")

        # Train on full training set
        clf.fit(X_train, y_train)
        y_pred   = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1  = f1_score(y_test, y_pred, average="weighted")
        elapsed  = time.time() - t0

        results.append((name, test_acc, test_f1, cv_acc, elapsed, clf, y_pred))
        print(f"{name:<28} {test_acc:>9.4f} {test_f1:>9.4f} {cv_acc:>9.4f} {elapsed:>7.1f}s")

    # ── Best model detail ──────────────────────────────────────────────────────
    results.sort(key=lambda r: r[1], reverse=True)
    best_name, best_acc, best_f1, best_cv, _, best_clf, best_pred = results[0]

    print()
    print("=" * 65)
    print(f"Best: {best_name}  ->  test {best_acc:.4f} ({best_acc*100:.1f}%)")
    print("=" * 65)
    print()
    print(f"Classification report ({best_name}):")
    print(classification_report(y_test, best_pred))

    # ── Per-class breakdown ────────────────────────────────────────────────────
    labels = sorted(np.unique(y_test))
    print("Per-letter accuracy (test set, best model):")
    for letter in labels:
        mask = y_test == letter
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_test[mask], best_pred[mask])
        bar = "#" * int(acc * 20)
        print(f"  {letter}  [{bar:<20}] {acc*100:5.1f}%  ({mask.sum()} windows)")

    # ── Overfitting check ──────────────────────────────────────────────────────
    print()
    print("Overfitting check (CV acc vs test acc):")
    print(f"{'Algorithm':<28} {'CV':>8} {'Test':>8} {'Gap':>8}")
    print("-" * 50)
    for name, test_acc, _, cv_acc, *_ in results:
        gap = cv_acc - test_acc
        flag = " <-- possible overfit" if gap > 0.05 else ""
        print(f"{name:<28} {cv_acc:>8.4f} {test_acc:>8.4f} {gap:>+8.4f}{flag}")


if __name__ == "__main__":
    main()
