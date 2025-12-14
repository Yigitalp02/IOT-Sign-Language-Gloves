"""
Generate Confusion Matrix Visualizations from Trained Models

This script loads trained models and generates confusion matrices
for presentation/thesis materials.

Usage:
    python scripts/generate_confusion_matrix.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Configuration
DATA_PATH = 'data/Data/professor_data_combined.csv'
WINDOWED_FEATURES_PATH = 'data/windowed_features_0.5s_0.25s.npz'
MODELS_DIR = 'models/windowed/'
OUTPUT_DIR = 'results/confusion_matrices/'

# Gesture labels (in order)
GESTURE_LABELS = [
    'FourFinger_Grasp',
    'Grasp',
    'Single_Index',
    'Single_Middle',
    'Single_Pinkie',
    'Single_Ring',
    'Single_Thumb',
    'Thumb2Index',
    'Thumb2Middle',
    'Thumb2Pinkie',
    'Thumb2Ring'
]

# Short labels for better visualization
GESTURE_LABELS_SHORT = [
    '4FGrasp',
    'Grasp',
    'Index',
    'Middle',
    'Pinkie',
    'Ring',
    'Thumb',
    'T2Index',
    'T2Middle',
    'T2Pinkie',
    'T2Ring'
]


def load_windowed_data():
    """Load windowed features and labels"""
    print(f"Loading windowed features from {WINDOWED_FEATURES_PATH}...")
    data = np.load(WINDOWED_FEATURES_PATH)
    
    X = data['X']
    y = data['y']
    users = data['users']
    
    print(f"Loaded {len(X)} windows from {len(np.unique(users))} users")
    print(f"Feature shape: {X.shape}")
    
    return X, y, users


def load_model(model_name):
    """Load a trained model"""
    model_path = os.path.join(MODELS_DIR, f'{model_name}')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def load_preprocessors():
    """Load label encoder and scaler"""
    le_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return label_encoder, scaler


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """
    Generate and save a beautiful confusion matrix visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model (for title)
        save_path: Path to save the figure
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=GESTURE_LABELS_SHORT,
                yticklabels=GESTURE_LABELS_SHORT,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'{model_name} - Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Gesture', fontsize=12)
    ax1.set_xlabel('Predicted Gesture', fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Plot 2: Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=GESTURE_LABELS_SHORT,
                yticklabels=GESTURE_LABELS_SHORT,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title(f'{model_name} - Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Gesture', fontsize=12)
    ax2.set_xlabel('Predicted Gesture', fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved confusion matrix to: {save_path}")
    plt.close()


def generate_classification_report(y_true, y_pred, model_name, save_path):
    """Generate and save detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=GESTURE_LABELS)
    
    report_path = save_path.replace('.png', '.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(f"Classification Report: {model_name}\n")
        f.write("="*80 + "\n\n")
        f.write(report)
    
    print(f"Saved classification report to: {report_path}")


def evaluate_model(model_name):
    """Evaluate a specific model and generate visualizations"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}\n")
    
    # Load data
    X, y, users = load_windowed_data()
    
    # Load model and preprocessors
    model = load_model(model_name)
    label_encoder, scaler = load_preprocessors()
    
    # Prepare test data (80/20 split, same as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    y_test_encoded = label_encoder.transform(y_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Generate confusion matrix
    output_path = os.path.join(OUTPUT_DIR, f'{model_name.replace(".pkl", "")}_confusion_matrix.png')
    plot_confusion_matrix(y_test_encoded, y_pred, 
                         model_name.replace('.pkl', '').replace('_', ' ').upper(),
                         output_path)
    
    # Generate classification report
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    generate_classification_report(y_test_labels, y_pred_labels,
                                  model_name.replace('.pkl', ''),
                                  output_path)
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    return accuracy


def main():
    """Generate confusion matrices for all trained models"""
    print("Confusion Matrix Generator")
    print("="*80)
    
    models_to_evaluate = [
        'rf_model_0.5s.pkl',
        'gb_model_0.5s.pkl',
        'mlp_model_0.5s.pkl',
    ]
    
    results = {}
    
    for model_name in models_to_evaluate:
        try:
            accuracy = evaluate_model(model_name)
            results[model_name] = accuracy
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for model_name, accuracy in results.items():
        print(f"{model_name:<30} {accuracy*100:>6.2f}%")
    
    print(f"\nConfusion matrices saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

