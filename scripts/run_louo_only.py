"""
Run only Leave-One-User-Out CV to test generalization

This skips the 80/20 training and goes straight to the most important test:
Can the models generalize to completely NEW users?
"""

import numpy as np
import os
import time
from datetime import timedelta
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

def load_windowed_data():
    """Load windowed features"""
    filename = "data/windowed_features_0.5s_0.25s.npz"
    print(f"Loading: {filename}")
    data = np.load(filename)
    X, y, users = data['X'], data['y'], data['users']
    print(f"[OK] Loaded {len(X):,} windows, {X.shape[1]} features")
    return X, y, users

def louo_cv(X, y, users, model_name='RF'):
    """Leave-One-User-Out Cross-Validation"""
    print(f"\n{'='*60}")
    print(f"LEAVE-ONE-USER-OUT CV - {model_name}")
    print(f"{'='*60}")
    print(f"Testing generalization to NEW users...")
    print(f"Total users: {len(np.unique(users))}\n")
    
    logo = LeaveOneGroupOut()
    accuracies = []
    start_time = time.time()
    
    for i, (train_idx, test_idx) in enumerate(logo.split(X, y, users)):
        iter_start = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_user = users[test_idx[0]]
        
        # Train model
        if model_name == 'RF':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                random_state=42,
                n_jobs=-1
            )
        else:  # GB
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=7,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        # Progress info
        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        remaining = iter_time * (12 - i - 1)
        
        print(f"[{i+1}/12] {test_user:15s}: {acc*100:5.2f}% | "
              f"Time: {timedelta(seconds=int(iter_time))} | "
              f"ETA: {timedelta(seconds=int(remaining))}")
    
    total_time = time.time() - start_time
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Average LOUO Accuracy: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    
    # Compare to baseline
    baseline = 0.5611
    improvement = (mean_acc - baseline) * 100
    print(f"\nComparison:")
    print(f"  Baseline (single-point): 56.11%")
    print(f"  Windowed ({model_name}):       {mean_acc*100:.2f}%")
    print(f"  Improvement:             {improvement:+.2f}%")
    
    if mean_acc > 0.70:
        print(f"\n[EXCELLENT] State-of-the-art performance!")
    elif mean_acc > 0.65:
        print(f"\n[VERY GOOD] Strong generalization!")
    elif mean_acc > 0.60:
        print(f"\n[GOOD] Windowing significantly helped!")
    
    return accuracies, mean_acc

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    print("="*60)
    print("LEAVE-ONE-USER-OUT CROSS-VALIDATION")
    print("="*60)
    print("Testing if windowed models generalize to NEW users")
    print("Estimated time: 10-15 minutes total")
    print("="*60)
    
    X, y, users = load_windowed_data()
    
    # Test both models
    rf_acc, rf_mean = loou_cv(X, y, users, 'RF')
    gb_acc, gb_mean = louo_cv(X, y, users, 'GB')
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\nModel Performance (LOUO):")
    print(f"  Random Forest:       {rf_mean*100:.2f}%")
    print(f"  Gradient Boosting:   {gb_mean*100:.2f}%")
    print(f"\nBaseline comparison:")
    print(f"  Single-point (old):  56.11%")
    print(f"  Best windowed:       {max(rf_mean, gb_mean)*100:.2f}%")
    print(f"  Improvement:         {(max(rf_mean, gb_mean)-0.5611)*100:+.2f}%")
    print(f"{'='*60}")
    
    # Save results
    os.makedirs("models/windowed", exist_ok=True)
    np.savez("models/windowed/louo_results.npz",
             rf_accuracies=rf_acc,
             gb_accuracies=gb_acc,
             rf_mean=rf_mean,
             gb_mean=gb_mean)
    print(f"\n[OK] Results saved to models/windowed/loou_results.npz")

if __name__ == "__main__":
    main()











