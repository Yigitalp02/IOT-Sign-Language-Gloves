#!/usr/bin/env python3
"""
IMU Centroid Calibration — One-time tool

Run this ONCE on a CSV that has reliable IMU data (your existing recordings).
It computes the median gravity vector for each ambiguous letter and saves a
centroids.json file.

After this, you can retrain the flex model on any new data (no IMU needed)
and the centroids.json stays valid as long as the glove hardware doesn't change.

Usage
=====
  python scripts/calibrate_imu_centroids.py
  python scripts/calibrate_imu_centroids.py -i data/Data/my_old_data.csv
  python scripts/calibrate_imu_centroids.py -o models/my_centroids.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES: dict[str, list[str]] = {
    "DG":  ["D", "G"],
    "VHR": ["V", "H", "R"],
    "LPQ": ["L", "P", "Q"],
}

FLEX_COLS = ["ch0_norm", "ch1_norm", "ch2_norm", "ch3_norm", "ch4_norm"]
IMU_COLS  = ["qw", "qx", "qy", "qz"]
ALL_COLS  = FLEX_COLS + IMU_COLS

WINDOW_SIZE = 50
STRIDE      = 25
SAMPLES_PER_REC = 150


def iter_windows(data, window_size, stride, samples_per_rec):
    n = max(1, len(data) // samples_per_rec)
    for rec in range(n):
        s = rec * samples_per_rec
        e = min(s + samples_per_rec, len(data))
        chunk = data[s:e]
        if len(chunk) < window_size:
            continue
        for i in range(0, len(chunk) - window_size + 1, stride):
            yield chunk[i: i + window_size]


def gravity_vector(window: np.ndarray) -> tuple[float, float, float]:
    """Mean yaw-invariant gravity components for a window."""
    qw = window[:, 5].astype(float)
    qx = window[:, 6].astype(float)
    qy = window[:, 7].astype(float)
    qz = window[:, 8].astype(float)
    fwd_z   = float(np.mean(2.0 * (qx * qz + qw * qy)))
    up_z    = float(np.mean(1.0 - 2.0 * (qx**2 + qy**2)))
    right_z = float(np.mean(2.0 * (qy * qz - qw * qx)))
    return fwd_z, up_z, right_z


def compute_centroids(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    missing = [c for c in ["label"] + ALL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    available = set(df["label"].unique())
    print(f"Loaded {Path(csv_path).name}: {len(df):,} rows, letters={sorted(available)}")
    print()

    result = {}
    for fam_name, members in FAMILIES.items():
        present = [m for m in members if m in available]
        if len(present) < 2:
            print(f"[{fam_name}] Only {present} in data — skipping")
            continue

        centroids = {}
        print(f"[{fam_name}]  members={present}")
        for letter in present:
            data = df[df["label"] == letter][ALL_COLS].values.astype(np.float64)

            # Check if IMU data looks valid (not all zeros)
            imu_data = data[:, 5:]
            imu_nonzero = np.count_nonzero(np.abs(imu_data) > 0.01)
            if imu_nonzero < 10:
                print(f"  {letter}: WARNING — IMU data looks like all-zeros, "
                      f"skipping centroid for this letter")
                continue

            vecs = [gravity_vector(w) for w in iter_windows(
                data, WINDOW_SIZE, STRIDE, SAMPLES_PER_REC)]
            if not vecs:
                print(f"  {letter}: not enough windows — skipping")
                continue

            arr = np.array(vecs)
            fwd  = float(np.median(arr[:, 0]))
            up   = float(np.median(arr[:, 1]))
            right = float(np.median(arr[:, 2]))
            centroids[letter] = {
                "fwd_z":   round(fwd,   4),
                "up_z":    round(up,    4),
                "right_z": round(right, 4),
            }
            print(f"  {letter}:  fwd_z={fwd:+.3f}  up_z={up:+.3f}  right_z={right:+.3f}")

        if len(centroids) >= 2:
            letters = list(centroids.keys())
            print(f"  Centroid separations:")
            for i in range(len(letters)):
                ci = np.array(list(centroids[letters[i]].values()))
                for j in range(i + 1, len(letters)):
                    cj = np.array(list(centroids[letters[j]].values()))
                    sep = float(np.linalg.norm(ci - cj))
                    quality = "OK" if sep >= 0.3 else "LOW — may cause errors"
                    print(f"    {letters[i]}-{letters[j]}: {sep:.3f}  [{quality}]")

        result[fam_name] = {
            "members":   present,
            "centroids": centroids,
        }
        print()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute IMU gravity centroids from an existing CSV with reliable IMU data"
    )
    parser.add_argument("--input", "-i",
                        default="data/Data/glove_data_NORMALIZED_A_B_C_D_E_F_I_G_K_O_L_S_H_R_P_Q_T_V_W_X_Y_2026-03-10-12-56-47.csv")
    parser.add_argument("--output", "-o",
                        default="models/imu_centroids.json")
    args = parser.parse_args()

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    csv_path     = str(project_root / args.input)
    out_path     = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        return 1

    print("=" * 60)
    print("IMU Centroid Calibration")
    print("=" * 60)
    centroids = compute_centroids(csv_path)

    output = {
        "_note": (
            "Auto-derived from training data. Edit fwd_z/up_z/right_z values "
            "to manually tune. fwd_z: +1=fingers up, -1=fingers down. "
            "up_z: +1=back-of-hand up (palm down). right_z: wrist roll."
        ),
        "families": centroids,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Centroids saved: {out_path}")
    print()
    print("Next step: train the flex model on any data (IMU not required):")
    print(f"  python scripts/train_model_v4.py -i <new_flex_only_data.csv>")
    print(f"  (it will auto-load {out_path.name} for IMU disambiguation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
