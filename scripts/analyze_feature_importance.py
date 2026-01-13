"""
Analyze feature importance to identify which features contribute most to predictions

This helps us:
1. Understand what the model finds important
2. Remove weak/redundant features
3. Potentially improve generalization
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt

def get_feature_names():
    """Generate names for all 30 features"""
    feature_names = []
    channel_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinkie']
    
    for i, ch_name in enumerate(channel_names):
        feature_names.extend([
            f'{ch_name}_mean',
            f'{ch_name}_std',
            f'{ch_name}_min',
            f'{ch_name}_max',
            f'{ch_name}_velocity',
            f'{ch_name}_acceleration',
        ])
    
    return feature_names

def analyze_rf_importance(model, feature_names):
    """Analyze Random Forest feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n" + "="*60)
    print("RANDOM FOREST - FEATURE IMPORTANCE")
    print("="*60)
    print("\nTop 15 Most Important Features:")
    print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12} {'Cumulative'}")
    print("-"*60)
    
    cumulative = 0
    for rank, idx in enumerate(indices[:15], 1):
        cumulative += importances[idx]
        print(f"{rank:<6} {feature_names[idx]:<25} {importances[idx]:>10.4f}  {cumulative:>10.2%}")
    
    print("\n" + "-"*60)
    print(f"Top 15 features account for: {cumulative:.2%} of total importance")
    
    print("\n\nBottom 10 Least Important Features:")
    print(f"{'Rank':<6} {'Feature':<25} {'Importance'}")
    print("-"*60)
    
    for rank, idx in enumerate(indices[-10:], 1):
        print(f"{rank:<6} {feature_names[idx]:<25} {importances[idx]:>10.4f}")
    
    return importances, indices

def test_reduced_feature_set(X, y, users, feature_names, top_n_features):
    """Test model performance with only top N features"""
    print("\n\n" + "="*60)
    print(f"TESTING WITH TOP {top_n_features} FEATURES")
    print("="*60)
    
    # Train a full model to get feature importance
    rf_full = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("\nTraining full model to determine feature importance...")
    rf_full.fit(X, y)
    
    # Get top features
    importances = rf_full.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_n_features]
    
    print(f"\nTop {top_n_features} features selected:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Create reduced dataset
    X_reduced = X[:, top_indices]
    
    # Test with LOUO
    print(f"\nRunning LOUO with {top_n_features} features...")
    gkf = LeaveOneGroupOut()
    scores = []
    
    rf_reduced = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_reduced, y, groups=users)):
        X_train, X_test = X_reduced[train_idx], X_reduced[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf_reduced.fit(X_train, y_train)
        score = rf_reduced.score(X_test, y_test)
        scores.append(score)
        
        test_user = users[test_idx][0]
        print(f"  [{fold_idx+1:2d}/12] {test_user}: {score*100:5.2f}%")
    
    mean_score = np.mean(scores) * 100
    std_score = np.std(scores) * 100
    
    print("\n" + "-"*60)
    print(f"LOUO Accuracy with {top_n_features} features: {mean_score:.2f}% (±{std_score:.2f}%)")
    print("-"*60)
    
    return mean_score, std_score, top_indices

def analyze_feature_types(importances, feature_names):
    """Analyze importance by feature type"""
    print("\n\n" + "="*60)
    print("FEATURE TYPE ANALYSIS")
    print("="*60)
    
    # Group by feature type
    types = {
        'Statistical': ['mean', 'std', 'min', 'max'],
        'Dynamic': ['velocity', 'acceleration']
    }
    
    type_importance = {}
    for type_name, keywords in types.items():
        total = 0
        count = 0
        for i, fname in enumerate(feature_names):
            if any(kw in fname for kw in keywords):
                total += importances[i]
                count += 1
        type_importance[type_name] = (total, count, total/count if count > 0 else 0)
    
    print("\nImportance by Feature Type:")
    print(f"{'Type':<15} {'Total':<12} {'Count':<8} {'Average'}")
    print("-"*60)
    for type_name, (total, count, avg) in type_importance.items():
        print(f"{type_name:<15} {total:>10.4f}  {count:>6}   {avg:>10.4f}")
    
    # Group by finger
    print("\n\nImportance by Finger:")
    print(f"{'Finger':<15} {'Total':<12} {'Count':<8} {'Average'}")
    print("-"*60)
    
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinkie']
    finger_importance = {}
    
    for finger in fingers:
        total = 0
        count = 0
        for i, fname in enumerate(feature_names):
            if finger in fname:
                total += importances[i]
                count += 1
        finger_importance[finger] = (total, count, total/count if count > 0 else 0)
        print(f"{finger:<15} {total:>10.4f}  {count:>6}   {total/count:>10.4f}")

if __name__ == "__main__":
    # Change to iot-sign-glove root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    print("="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    print("Analyzing which features contribute most to predictions")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = np.load('data/windowed_features_1.0s_0.5s.npz')
    X = data['X']
    y = data['y']
    users = data['users']
    
    print(f"Loaded: {len(X):,} windows, {X.shape[1]} features")
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Load trained model (or train new one)
    model_path = 'models/windowed/random_forest.pkl'
    if os.path.exists(model_path):
        print(f"\nLoading trained model from: {model_path}")
        rf = joblib.load(model_path)
    else:
        print("\nTraining new Random Forest model...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=2
        )
        rf.fit(X, y)
    
    # Analyze importance
    importances, indices = analyze_rf_importance(rf, feature_names)
    
    # Analyze by type and finger
    analyze_feature_types(importances, feature_names)
    
    # Test reduced feature sets
    baseline_accuracy = 73.13  # From previous run
    
    print("\n\n" + "="*60)
    print("TESTING REDUCED FEATURE SETS")
    print("="*60)
    print(f"Baseline (30 features): {baseline_accuracy:.2f}%")
    print("\nLet's test if we can maintain accuracy with fewer features...")
    
    # Test different feature counts
    for n_features in [25, 20, 15]:
        mean_score, std_score, top_indices = test_reduced_feature_set(
            X, y, users, feature_names, n_features
        )
        
        diff = mean_score - baseline_accuracy
        if diff >= -1.0:  # Within 1% of baseline
            print(f"\n[OK] {n_features} features: {mean_score:.2f}% (change: {diff:+.2f}%)")
        else:
            print(f"\n[WARNING] {n_features} features: {mean_score:.2f}% (drop: {diff:.2f}%)")
    
    print("\n\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)




