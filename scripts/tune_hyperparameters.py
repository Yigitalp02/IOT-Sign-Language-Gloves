"""
Hyperparameter Tuning for Random Forest and Gradient Boosting

This script tests different hyperparameter combinations to find the optimal settings
that maximize LOUO (Leave-One-User-Out) accuracy.

We'll tune:
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Gradient Boosting: n_estimators, max_depth, learning_rate
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
import joblib
import itertools

def evaluate_model(model, X, y, users, model_name="Model"):
    """Evaluate model using LOUO cross-validation"""
    gkf = LeaveOneGroupOut()
    scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=users)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    mean_score = np.mean(scores) * 100
    std_score = np.std(scores) * 100
    
    return mean_score, std_score, scores

def tune_random_forest(X, y, users):
    """Tune Random Forest hyperparameters"""
    print("\n" + "="*60)
    print("TUNING RANDOM FOREST HYPERPARAMETERS")
    print("="*60)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print(f"\nParameter space:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Estimated time: {total_combinations * 0.3:.0f} - {total_combinations * 0.5:.0f} minutes")
    
    # Baseline (current settings)
    print("\n" + "-"*60)
    print("BASELINE (Current Settings)")
    print("-"*60)
    baseline_params = {
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }
    
    print(f"Parameters: {baseline_params}")
    print("Evaluating...")
    
    rf_baseline = RandomForestClassifier(**baseline_params)
    baseline_score, baseline_std, _ = evaluate_model(rf_baseline, X, y, users, "RF Baseline")
    
    print(f"\nBaseline LOUO: {baseline_score:.2f}% (±{baseline_std:.2f}%)")
    
    # Test promising combinations (reduced search)
    print("\n" + "-"*60)
    print("TESTING PROMISING COMBINATIONS")
    print("-"*60)
    
    best_score = baseline_score
    best_params = baseline_params.copy()
    best_std = baseline_std
    
    # Reduced grid search - test most promising combinations
    test_configs = [
        # More trees
        {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 5, 'min_samples_leaf': 2},
        # Deeper trees
        {'n_estimators': 200, 'max_depth': 40, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 2},
        # Less regularization
        {'n_estimators': 200, 'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 1},
        # More regularization
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4},
        # Balanced
        {'n_estimators': 250, 'max_depth': 35, 'min_samples_split': 5, 'min_samples_leaf': 2},
        # More trees + deeper
        {'n_estimators': 300, 'max_depth': 40, 'min_samples_split': 5, 'min_samples_leaf': 2},
    ]
    
    for idx, params in enumerate(test_configs, 1):
        params_with_defaults = {
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0,
            **params
        }
        
        print(f"\n[{idx}/{len(test_configs)}] Testing: n_estimators={params['n_estimators']}, "
              f"max_depth={params['max_depth']}, "
              f"min_samples_split={params['min_samples_split']}, "
              f"min_samples_leaf={params['min_samples_leaf']}")
        
        start_time = time.time()
        rf = RandomForestClassifier(**params_with_defaults)
        score, std, _ = evaluate_model(rf, X, y, users, f"RF Config {idx}")
        elapsed = time.time() - start_time
        
        improvement = score - baseline_score
        print(f"   Result: {score:.2f}% (±{std:.2f}%) | "
              f"Change: {improvement:+.2f}% | Time: {elapsed/60:.1f}m")
        
        if score > best_score:
            best_score = score
            best_params = params_with_defaults
            best_std = std
            print(f"   [NEW BEST!]")
    
    print("\n" + "="*60)
    print("RANDOM FOREST TUNING RESULTS")
    print("="*60)
    print(f"Baseline: {baseline_score:.2f}% (±{baseline_std:.2f}%)")
    print(f"Best:     {best_score:.2f}% (±{best_std:.2f}%)")
    print(f"Improvement: {best_score - baseline_score:+.2f}%")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        if param not in ['random_state', 'n_jobs', 'verbose']:
            print(f"  {param}: {value}")
    
    return best_params, best_score, best_std

def tune_gradient_boosting(X, y, users):
    """Tune Gradient Boosting hyperparameters"""
    print("\n\n" + "="*60)
    print("TUNING GRADIENT BOOSTING HYPERPARAMETERS")
    print("="*60)
    
    # Baseline (current settings)
    print("\n" + "-"*60)
    print("BASELINE (Current Settings)")
    print("-"*60)
    baseline_params = {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': 0
    }
    
    print(f"Parameters: {baseline_params}")
    print("Evaluating... (this takes longer)")
    
    gb_baseline = GradientBoostingClassifier(**baseline_params)
    baseline_score, baseline_std, _ = evaluate_model(gb_baseline, X, y, users, "GB Baseline")
    
    print(f"\nBaseline LOUO: {baseline_score:.2f}% (±{baseline_std:.2f}%)")
    
    # Test promising combinations
    print("\n" + "-"*60)
    print("TESTING PROMISING COMBINATIONS")
    print("-"*60)
    
    best_score = baseline_score
    best_params = baseline_params.copy()
    best_std = baseline_std
    
    test_configs = [
        # More estimators
        {'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.1},
        {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1},
        # Deeper trees
        {'n_estimators': 100, 'max_depth': 9, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
        # Different learning rates
        {'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.05},
        {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.15},
        # Balanced
        {'n_estimators': 150, 'max_depth': 9, 'learning_rate': 0.1},
    ]
    
    for idx, params in enumerate(test_configs, 1):
        params_with_defaults = {
            'random_state': 42,
            'verbose': 0,
            **params
        }
        
        print(f"\n[{idx}/{len(test_configs)}] Testing: n_estimators={params['n_estimators']}, "
              f"max_depth={params['max_depth']}, "
              f"learning_rate={params['learning_rate']}")
        
        start_time = time.time()
        gb = GradientBoostingClassifier(**params_with_defaults)
        score, std, _ = evaluate_model(gb, X, y, users, f"GB Config {idx}")
        elapsed = time.time() - start_time
        
        improvement = score - baseline_score
        print(f"   Result: {score:.2f}% (±{std:.2f}%) | "
              f"Change: {improvement:+.2f}% | Time: {elapsed/60:.1f}m")
        
        if score > best_score:
            best_score = score
            best_params = params_with_defaults
            best_std = std
            print(f"   [NEW BEST!]")
    
    print("\n" + "="*60)
    print("GRADIENT BOOSTING TUNING RESULTS")
    print("="*60)
    print(f"Baseline: {baseline_score:.2f}% (±{baseline_std:.2f}%)")
    print(f"Best:     {best_score:.2f}% (±{best_std:.2f}%)")
    print(f"Improvement: {best_score - baseline_score:+.2f}%")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        if param not in ['random_state', 'verbose']:
            print(f"  {param}: {value}")
    
    return best_params, best_score, best_std

if __name__ == "__main__":
    import os
    
    # Change to iot-sign-glove root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    print("Finding optimal parameters for Random Forest and Gradient Boosting")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = np.load('data/windowed_features_1.0s_0.5s.npz')
    X = data['X']
    y = data['y']
    users = data['users']
    
    print(f"Loaded: {len(X):,} windows, {X.shape[1]} features, {len(np.unique(users))} users")
    
    # Tune Random Forest
    rf_best_params, rf_best_score, rf_best_std = tune_random_forest(X, y, users)
    
    # Tune Gradient Boosting
    gb_best_params, gb_best_score, gb_best_std = tune_gradient_boosting(X, y, users)
    
    # Final summary
    print("\n\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print("\nRandom Forest:")
    print(f"  Original: 73.13%")
    print(f"  Tuned:    {rf_best_score:.2f}% (±{rf_best_std:.2f}%)")
    print(f"  Improvement: {rf_best_score - 73.13:+.2f}%")
    
    print("\nGradient Boosting:")
    print(f"  Original: 72.87%")
    print(f"  Tuned:    {gb_best_score:.2f}% (±{gb_best_std:.2f}%)")
    print(f"  Improvement: {gb_best_score - 72.87:+.2f}%")
    
    print("\n" + "="*60)
    print("TUNING COMPLETE!")
    print("="*60)




