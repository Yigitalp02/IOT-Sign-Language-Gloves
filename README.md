# IoT Sign Glove — Pipeline Starter

A minimal, **hardware-ready** pipeline to collect data, train baselines, and report results for a sign-language glove project.

## Quick Start
```bash
# 1) Create venv and activate (PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) Run synthetic baseline
python scripts/baseline_synth.py