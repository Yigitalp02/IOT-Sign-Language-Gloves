#!/usr/bin/env python3
"""
Build serve_local_model_one.py into a standalone server using PyInstaller.

Uses --onedir (NOT --onefile) because --onefile extracts itself into a random
Windows temp folder on every launch. Windows Defender / Storage Sense can delete
that temp folder while the server is still running, causing random mid-session
crashes. --onedir writes the files once to a permanent location and never
re-extracts, making it completely stable.

Output layout after this script runs:
  src-tauri/resources/model-server/
    serve_local_model_one.exe   ← main launcher
    _internal/                  ← Python DLLs, packages, embedded model

Tauri bundles "resources/model-server/**/*" and resolve_resource() finds the exe.

Usage (from project root or iot-sign-glove/):
    python build_model_server.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE        = Path(__file__).resolve().parent          # iot-sign-glove/
SCRIPTS_DIR = HERE / "scripts"
DIST_DIR    = HERE / "dist"
BUILD_DIR   = HERE / "pyinstaller_build"
SPEC_DIR    = HERE

SCRIPT     = SCRIPTS_DIR / "serve_local_model_one.py"
MODELS_DIR = HERE / "models"

# Destination inside src-tauri so Tauri can bundle without a ../ path
# (Tauri converts ../ to _up_/ which breaks resolve_resource at runtime).
TAURI_RESOURCES = HERE.parent / "src-tauri" / "resources" / "model-server"

_MODEL_PRIORITY = [
    "rf_asl_v5_multifamily.pkl",
    "rf_asl_v3_combined.pkl",
    "rf_asl_v4_flex_rules.pkl",
    "rf_asl_21letters_imu_two_staged.pkl",
]


def find_best_model():
    for name in _MODEL_PRIORITY:
        p = MODELS_DIR / name
        if p.exists():
            return p
    return None


def main():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found – installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    sep = ";" if sys.platform == "win32" else ":"
    add_data: list[str] = []

    model_path = find_best_model()
    if model_path:
        add_data += ["--add-data", f"{model_path}{sep}models"]
        meta = model_path.with_suffix("").with_suffix(".meta.joblib")
        if meta.exists():
            add_data += ["--add-data", f"{meta}{sep}models"]
        print(f"Embedding model: {model_path.name}")
    else:
        print("WARNING: No model file found – server will start but return '?' for predictions.")
        print(f"         Place a .pkl model in: {MODELS_DIR}")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir",                          # stable: no temp-dir extraction
        "--name", "serve_local_model_one",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(SPEC_DIR),
        "--noconfirm",
        "--log-level", "WARN",
        # sklearn is loaded dynamically by joblib when unpickling the model,
        # so PyInstaller won't detect it via static import analysis.
        "--collect-all", "sklearn",
    ] + add_data + [str(SCRIPT)]

    print("\nBuilding standalone server (--onedir) …")
    subprocess.run(cmd, check=True, cwd=str(HERE))

    # PyInstaller --onedir creates dist/serve_local_model_one/ (a directory)
    out_dir = DIST_DIR / "serve_local_model_one"
    if not out_dir.is_dir():
        print(f"\nWARNING: Expected output directory not found at {out_dir}")
        return

    total_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"\nBuilt: {out_dir}/  ({total_mb:.0f} MB total)")

    # Copy the whole directory into src-tauri/resources/model-server/
    if TAURI_RESOURCES.exists():
        shutil.rmtree(TAURI_RESOURCES)
    shutil.copytree(out_dir, TAURI_RESOURCES)
    print(f"Copied to: {TAURI_RESOURCES}/")


if __name__ == "__main__":
    main()
