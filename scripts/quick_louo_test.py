"""
Quick LOUO test - skip 80/20 training, go straight to LOUO validation
"""

import numpy as np
import os
import time
from datetime import timedelta
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def load_windowed_data(window_size=0.5, stride=0.25):
    """Load pre-computed windowed features"""
    filename = f"data/windowed_features_{window_size}s_{stride}s.npz"
    
    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}")
    print(f"File: {filename}")
    
    data = np.load(filename)
    X = data['X']
    y = data['y']
    users = data['users']
    
    print(f"[OK] Loaded {len(X):,} windows")
    print(f"[OK] Feature dimension: {X.shape[1]}")
    print(f"[OK] Unique gestures: {len(np.unique(y))}")
    print(f"[OK] Unique users: {len(np.unique(users))}")
    
    return X, y, users

def louo_validation(X, y, users, model_name="RF"):
    """Run Leave-One-User-Out cross-validation"""
    print(f"\n{'='*60}")
    print(f"LEAVE-ONE-USER-OUT: {model_name}")
    print(f"{'='*60}")
    print(f"Total users: {len(np.unique(users))}")
    print(f"Feature dimension: {X.shape[1]}")
    
    logo = LeaveOneGroupOut()
    scores = []
    user_names = []
    
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, users)):
        fold_start = time.time()
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get test user name
        test_user = users[test_idx[0]]
        user_names.append(test_user)
        
        # Train model
        if model_name == "RF":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:  # GB
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        
        # Progress update
        elapsed = time.time() - start_time
        fold_time = time.time() - fold_start
        avg_time_per_fold = elapsed / (fold_idx + 1)
        remaining_folds = len(np.unique(users)) - (fold_idx + 1)
        eta = remaining_folds * avg_time_per_fold
        
        progress_pct = ((fold_idx + 1) / len(np.unique(users))) * 100
        
        print(f"[{fold_idx+1}/{len(np.unique(users))}] ({progress_pct:.1f}%) "
              f"{test_user:15s}: {acc*100:.2f}% | "
              f"Fold time: {timedelta(seconds=int(fold_time))} | "
              f"ETA: {timedelta(seconds=int(eta))}")
    
    total_time = time.time() - start_time
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\n{'='*60}")
    print(f"LOUO RESULTS - {model_name}")
    print(f"{'='*60}")
    print(f"Average Accuracy: {avg_score*100:.2f}% (±{std_score*100:.2f}%)")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    
    print(f"\nPer-user breakdown:")
    for user, score in sorted(zip(user_names, scores), key=lambda x: -x[1]):
        print(f"  {user}: {score*100:.2f}%")
    
    return avg_score, std_score

if __name__ == "__main__":
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    print(f"\n{'='*60}")
    print(f"QUICK LOUO VALIDATION TEST")
    print(f"{'='*60}")
    print(f"Testing new features:")
    print(f"  - Jerk (3rd derivative)")
    print(f"  - RMS (signal power)")
    print(f"  - Zero-crossing rate")
    print(f"  - Recency features (last 3 samples)")
    print(f"  - Z-score normalization (per-user)")
    print(f"{'='*60}")
    
    # Load data
    X, y, users = load_windowed_data()
    
    if X is None:
        exit(1)
    
    # Run RF LOUO (fast)
    print("\n\n")
    print("="*60)
    print("STEP 1/2: RANDOM FOREST LOUO")
    print("="*60)
    rf_acc, rf_std = louo_validation(X, y, users, "RF")
    
    # Run GB LOUO (slow)
    print("\n\n")
    print("="*60)
    print("STEP 2/2: GRADIENT BOOSTING LOUO")
    print("="*60)
    gb_acc, gb_std = louo_validation(X, y, users, "GB")
    
    # Final summary
    print("\n\n")
    print("="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\nRandom Forest LOUO:      {rf_acc*100:.2f}% (±{rf_std*100:.2f}%)")
    print(f"Gradient Boosting LOUO:  {gb_acc*100:.2f}% (±{gb_std*100:.2f}%)")
    print(f"\nPrevious Results (30 features, min-max norm):")
    print(f"  Random Forest:         68.31%")
    print(f"  Gradient Boosting:     68.61%")
    print(f"\nImprovement:")
    print(f"  Random Forest:         +{(rf_acc*100 - 68.31):.2f}%")
    print(f"  Gradient Boosting:     +{(gb_acc*100 - 68.61):.2f}%")
    print("="*60)







