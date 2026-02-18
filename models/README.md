# Models Directory

This directory contains trained machine learning models for ASL recognition.

## 🚫 Models are NOT stored in Git

Model files (`.pkl`, `.h5`) are too large for GitHub and are excluded via `.gitignore`.

## 📦 Available Models

After training with `scripts/train_model.py`, you'll find:

- `my_glove_model_YYYYMMDD_HHMMSS.pkl` - Trained classifier
- `my_glove_model_scaler_YYYYMMDD_HHMMSS.pkl` - Feature scaler
- `my_glove_model_metadata_YYYYMMDD_HHMMSS.pkl` - Model metadata

## 🔄 How to Get Models

### Option 1: Train Your Own (Recommended)
```bash
# Collect data with your glove
python scripts/collect_data.py

# Train model
python scripts/train_model.py --data data/my_glove_data
```

### Option 2: Download Pre-trained Models
If available, download from:
- GitHub Releases
- Google Drive / Dropbox link
- Model hosting service

## 📊 Model Sizes (Why they're excluded)

Typical sizes:
- Random Forest: 50-100 MB
- Gradient Boosting: 30-80 MB
- Scaler: <1 MB
- Deep Learning: 100+ MB

GitHub limit: 100 MB per file

## 💡 Tip

For deployment, compress models or use model quantization to reduce size.

