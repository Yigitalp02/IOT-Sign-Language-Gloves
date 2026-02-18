# Data Directory

This directory contains sensor data collected from the smart glove.

## 🚫 Data files are NOT stored in Git

CSV and NPZ data files are too large for GitHub and are excluded via `.gitignore`.

## 📊 Data Structure

After collecting data with `scripts/collect_data.py`, you'll see:

```
data/
├── my_glove_data/
│   ├── A_rep1_20260218_143022.csv
│   ├── A_rep2_20260218_143045.csv
│   ├── B_rep1_20260218_143112.csv
│   └── ...
└── README.md (this file)
```

## 📁 Expected CSV Format

```csv
timestamp,flex_1,flex_2,flex_3,flex_4,flex_5,label,repetition
1708268123.5,120,580,720,770,580,A,1
1708268123.52,120,578,718,768,579,A,1
...
```

## 🔄 How to Get Data

### Collect Your Own Data (Primary Method)
```bash
python scripts/collect_data.py
```

Follow the interactive prompts to collect data from your glove.

### Download Sample Dataset (Optional)
If available, sample datasets can be downloaded from:
- GitHub Releases
- Google Drive / Dropbox link
- Research paper supplementary materials

## 💾 Data Sizes

Typical sizes:
- Single letter recording: ~30-50 KB
- Full dataset (15 letters × 10 reps): ~5-10 MB
- Large datasets with multiple subjects: 50-500 MB

## ✅ Data Quality

Validate your collected data:
```bash
python scripts/validate_data.py --data data/my_glove_data
```

