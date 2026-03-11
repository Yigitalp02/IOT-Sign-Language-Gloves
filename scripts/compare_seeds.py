#!/usr/bin/env python3
"""
Quick seed comparison for train_model_v2.
Tries every seed in SEEDS, prints Stage 1, Stage 2 per-family, and combined score.
Retrain the best seed at the end.

Usage:
    python scripts/compare_seeds.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from pathlib import Path
from sklearn.metrics import accuracy_score
from train_model_v2 import (
    load_data, build_stage1_dataset, build_stage2_dataset,
    split_by_recording, make_rf, FAMILIES,
)
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = str(PROJECT_ROOT / "data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_G_K_O_L_S_H_R_P_Q_T_V_W_X_Y_2026-03-10-12-56-47.csv")
OUT  = PROJECT_ROOT / "models/rf_asl_v2_gravity_cascade.pkl"

SEEDS        = [0, 1, 7, 13, 21, 42, 77, 99, 123, 200, 314, 999]
WINDOW       = 50
STRIDE       = 25
SAMPLES_REC  = 150
N_TREES      = 300
AUG_PER_WIN  = 8
JITTER_DEG   = 2.0
TEST_RATIO   = 0.2

print(f"Loading data from {Path(CSV).name} ...")
df = load_data([CSV])
print()

X1, y1, r1 = build_stage1_dataset(df, WINDOW, STRIDE, SAMPLES_REC)

header = f"{'seed':>5}  {'S1':>6}  {'DG':>6}  {'VHR':>6}  {'LPQ':>6}  {'avgS2':>6}  {'combined':>8}"
print(header)
print("-" * len(header))

results = []
for seed in SEEDS:
    # Stage 1
    X1_tr, X1_te, y1_tr, y1_te = split_by_recording(X1, y1, r1, TEST_RATIO, seed)
    clf1 = make_rf(N_TREES, seed)
    clf1.fit(X1_tr, y1_tr)
    s1_acc = accuracy_score(y1_te, clf1.predict(X1_te))

    # Stage 2 per family
    s2_acc = {}
    s2_clfs = {}
    for fam, members in FAMILIES.items():
        X2, y2, r2 = build_stage2_dataset(
            df, members, WINDOW, STRIDE, SAMPLES_REC, AUG_PER_WIN, JITTER_DEG, seed
        )
        X2_tr, X2_te, y2_tr, y2_te = split_by_recording(X2, y2, r2, TEST_RATIO, seed)
        clf2 = make_rf(N_TREES, seed)
        clf2.fit(X2_tr, y2_tr)
        s2_acc[fam]  = accuracy_score(y2_te, clf2.predict(X2_te))
        s2_clfs[fam] = clf2

    avg_s2   = sum(s2_acc.values()) / len(s2_acc)
    combined = (s1_acc + avg_s2) / 2.0

    results.append({
        "seed":     seed,
        "s1":       s1_acc,
        "dg":       s2_acc.get("DG",  0.0),
        "vhr":      s2_acc.get("VHR", 0.0),
        "lpq":      s2_acc.get("LPQ", 0.0),
        "avg_s2":   avg_s2,
        "combined": combined,
        "clf1":     clf1,
        "clf2s":    s2_clfs,
    })

    print(
        f"{seed:>5}  {s1_acc:>6.4f}  "
        f"{s2_acc.get('DG',0):>6.4f}  "
        f"{s2_acc.get('VHR',0):>6.4f}  "
        f"{s2_acc.get('LPQ',0):>6.4f}  "
        f"{avg_s2:>6.4f}  {combined:>8.4f}"
    )

# Sort by combined score
results.sort(key=lambda x: x["combined"], reverse=True)

print()
print("Top 5 by combined score:")
print(header)
print("-" * len(header))
for r in results[:5]:
    print(
        f"{r['seed']:>5}  {r['s1']:>6.4f}  "
        f"{r['dg']:>6.4f}  {r['vhr']:>6.4f}  {r['lpq']:>6.4f}  "
        f"{r['avg_s2']:>6.4f}  {r['combined']:>8.4f}"
    )

best = results[0]
print(f"\nBest seed: {best['seed']}  (combined={best['combined']:.4f})")
print(f"  Stage 1 : {best['s1']:.4f}")
print(f"  DG      : {best['dg']:.4f}")
print(f"  VHR     : {best['vhr']:.4f}")
print(f"  LPQ     : {best['lpq']:.4f}")

# Save the best model
model_package = {
    "format":         "v2_gravity_cascade",
    "stage_1_model":  best["clf1"],
    "stage_2_models": best["clf2s"],
    "families":       FAMILIES,
    "imu_letters":    sorted({"G", "H", "P", "Q", "R"} & set(df["label"].unique())),
}
joblib.dump(model_package, OUT)
print(f"\nBest model saved to {OUT}")
