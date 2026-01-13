"""
Compare different window sizes to find optimal configuration
Tests: 0.5s, 0.75s, 1.0s windows
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
import time
import joblib

def evaluate_window_size(window_file, window_name):
    """Evaluate LOUO performance for a specific window size"""
    print("\n" + "="*60)
    print(f"TESTING: {window_name}")
    print("="*60)
    
    # Load data
    print(f"Loading: {window_file}")
    data = np.load(window_file)
    X = data['X']
    y = data['y']
    users = data['users']
    
    print(f"Windows: {len(X):,} | Features: {X.shape[1]} | Users: {len(np.unique(users))}")
    
    # Initialize models
    N_CORES = joblib.cpu_count()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=N_CORES,
        verbose=0  # Silent
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        verbose=0  # Silent
    )
    
    # Leave-One-User-Out Cross-Validation
    gkf = GroupKFold(n_splits=len(np.unique(users)))
    
    rf_scores = []
    gb_scores = []
    
    print("\nRunning LOUO Cross-Validation...")
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=users)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        test_user = users[test_idx][0]
        
        # Train and evaluate RF
        rf.fit(X_train, y_train)
        rf_acc = rf.score(X_test, y_test)
        rf_scores.append(rf_acc)
        
        # Train and evaluate GB
        gb.fit(X_train, y_train)
        gb_acc = gb.score(X_test, y_test)
        gb_scores.append(gb_acc)
        
        progress = (fold_idx + 1) / len(np.unique(users)) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (fold_idx + 1)) * (len(np.unique(users)) - fold_idx - 1)
        
        print(f"[{fold_idx+1:2d}/12] {test_user}: RF={rf_acc*100:5.2f}% | GB={gb_acc*100:5.2f}% | "
              f"Elapsed: {int(elapsed//60):02d}:{int(elapsed%60):02d} | ETA: {int(eta//60):02d}:{int(eta%60):02d}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    rf_mean = np.mean(rf_scores) * 100
    rf_std = np.std(rf_scores) * 100
    gb_mean = np.mean(gb_scores) * 100
    gb_std = np.std(gb_scores) * 100
    
    print("\n" + "-"*60)
    print("RESULTS:")
    print(f"  Random Forest:       {rf_mean:.2f}% (±{rf_std:.2f}%)")
    print(f"  Gradient Boosting:   {gb_mean:.2f}% (±{gb_std:.2f}%)")
    print(f"  Total time: {int(total_time//60):02d}:{int(total_time%60):02d}")
    print("-"*60)
    
    return {
        'window': window_name,
        'rf_mean': rf_mean,
        'rf_std': rf_std,
        'gb_mean': gb_mean,
        'gb_std': gb_std,
        'time': total_time,
        'rf_scores': rf_scores,
        'gb_scores': gb_scores
    }

if __name__ == "__main__":
    import os
    
    # Change to iot-sign-glove root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    print("="*60)
    print("WINDOW SIZE COMPARISON - LOUO Cross-Validation")
    print("="*60)
    print("Testing different temporal window sizes")
    print("Goal: Find optimal window for gesture recognition")
    print("="*60)
    
    # Test configurations
    configs = [
        ('data/windowed_features_0.5s_0.25s.npz', '0.5s window, 0.25s stride'),
        ('data/windowed_features_0.75s_0.25s.npz', '0.75s window, 0.25s stride'),
        ('data/windowed_features_1.0s_0.5s.npz', '1.0s window, 0.5s stride'),
    ]
    
    results = []
    for window_file, window_name in configs:
        result = evaluate_window_size(window_file, window_name)
        results.append(result)
    
    # Print summary comparison
    print("\n\n" + "="*60)
    print("FINAL COMPARISON - ALL WINDOW SIZES")
    print("="*60)
    print(f"{'Window':<25} {'RF LOUO':<20} {'GB LOUO':<20} {'Time'}")
    print("-"*60)
    
    for r in results:
        print(f"{r['window']:<25} {r['rf_mean']:5.2f}% (±{r['rf_std']:4.2f}%)     "
              f"{r['gb_mean']:5.2f}% (±{r['gb_std']:4.2f}%)     "
              f"{int(r['time']//60):2d}:{int(r['time']%60):02d}")
    
    # Determine best
    best_rf = max(results, key=lambda x: x['rf_mean'])
    best_gb = max(results, key=lambda x: x['gb_mean'])
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print(f"Best for Random Forest:     {best_rf['window']} ({best_rf['rf_mean']:.2f}%)")
    print(f"Best for Gradient Boosting: {best_gb['window']} ({best_gb['gb_mean']:.2f}%)")
    print("="*60)




