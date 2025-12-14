"""
Train accurate ML models using windowed features with detailed progress tracking

This version prioritizes accuracy and provides clear progress updates
"""

import numpy as np
import os
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib

class ProgressTracker:
    """Track and display progress for long-running operations"""
    def __init__(self, total_steps, name="Process"):
        self.total_steps = total_steps
        self.current_step = 0
        self.name = name
        self.start_time = time.time()
    
    def update(self, step_name=""):
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * time_per_step
            
            progress_pct = (self.current_step / self.total_steps) * 100
            
            print(f"[{self.current_step}/{self.total_steps}] ({progress_pct:.1f}%) "
                  f"{step_name} | Elapsed: {timedelta(seconds=int(elapsed))} | "
                  f"ETA: {timedelta(seconds=int(eta))}")

def load_windowed_data(window_size=0.5, stride=0.25):
    """Load pre-computed windowed features"""
    filename = f"data/windowed_features_{window_size}s_{stride}s.npz"
    
    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}")
    print(f"File: {filename}")
    
    if not os.path.exists(filename):
        print(f"\nError: {filename} not found!")
        print("Run: python scripts/windowed_features.py")
        return None, None, None
    
    data = np.load(filename)
    X = data['X']
    y = data['y']
    users = data['users']
    
    print(f"[OK] Loaded {len(X):,} windows")
    print(f"[OK] Feature dimension: {X.shape[1]}")
    print(f"[OK] Unique gestures: {len(np.unique(y))}")
    print(f"[OK] Unique users: {len(np.unique(users))}")
    
    return X, y, users

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with progress tracking"""
    print(f"\n{'='*60}")
    print(f"TRAINING: RANDOM FOREST")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Trees: 200 | Max depth: 30")
    print(f"\nEstimated time: 2-5 minutes")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=2  # Shows tree-by-tree progress
    )
    
    print("\nTraining Random Forest...")
    rf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"\n[OK] Training complete in {timedelta(seconds=int(train_time))}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"RANDOM FOREST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"\nPer-Gesture Performance:")
    print(classification_report(y_test, y_pred))
    
    return rf

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with progress tracking"""
    print(f"\n{'='*60}")
    print(f"TRAINING: GRADIENT BOOSTING")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Estimators: 100 | Max depth: 7")
    print(f"\nEstimated time: 5-10 minutes")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        verbose=1  # Shows progress
    )
    
    print("\nTraining Gradient Boosting...")
    gb.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"\n[OK] Training complete in {timedelta(seconds=int(train_time))}")
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = gb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"GRADIENT BOOSTING RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"\nPer-Gesture Performance:")
    print(classification_report(y_test, y_pred))
    
    return gb

def train_mlp(X_train, y_train, X_test, y_test):
    """Train MLP with detailed progress tracking"""
    print(f"\n{'='*60}")
    print(f"TRAINING: NEURAL NETWORK (MLP)")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Architecture: 40 -> 256 -> 128 -> 64 -> 11")
    print(f"Max iterations: 300 (with early stopping)")
    print(f"\nEstimated time: 10-15 minutes")
    print(f"Note: Each iteration takes ~5-10 seconds")
    print(f"{'='*60}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[OK] Features scaled")
    
    start_time = time.time()
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        batch_size=256,  # Larger batches for faster training
        max_iter=300,    # Reduced from 500
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,  # Stop if no improvement for 10 iterations
        verbose=True,     # Shows iteration-by-iteration progress
        tol=1e-4
    )
    
    print(f"\nTraining MLP...")
    print(f"Progress will show iteration loss and validation score.")
    print(f"Training will stop early if validation doesn't improve for 10 iterations.\n")
    
    mlp.fit(X_train_scaled, y_train_encoded)
    
    train_time = time.time() - start_time
    print(f"\n[OK] Training complete in {timedelta(seconds=int(train_time))}")
    print(f"[OK] Completed {mlp.n_iter_} iterations")
    
    # Evaluate
    print("\nEvaluating...")
    y_pred_encoded = mlp.predict(X_test_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MLP RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"\nPer-Gesture Performance:")
    print(classification_report(y_test, y_pred))
    
    return mlp, scaler, label_encoder

def leave_one_user_out_cv(X, y, users, model_type='rf'):
    """LOUO CV with progress tracking"""
    print(f"\n{'='*60}")
    print(f"LEAVE-ONE-USER-OUT CROSS-VALIDATION")
    print(f"{'='*60}")
    print(f"Model: {model_type.upper()}")
    print(f"Total users: {len(np.unique(users))}")
    print(f"This tests generalization to NEW users!\n")
    
    if model_type == 'rf':
        print(f"Estimated time: 5-10 minutes (all 12 users)\n")
    elif model_type == 'gb':
        print(f"Estimated time: 15-20 minutes (all 12 users)\n")
    
    logo = LeaveOneGroupOut()
    unique_users = np.unique(users)
    total_users = len(unique_users)
    
    tracker = ProgressTracker(total_users, "LOUO CV")
    accuracies = []
    
    for i, (train_idx, test_idx) in enumerate(logo.split(X, y, users)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_user = users[test_idx[0]]
        
        # Train model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        tracker.update(f"{test_user:15s}: {acc*100:5.2f}%")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{'='*60}")
    print(f"LOUO RESULTS")
    print(f"{'='*60}")
    print(f"Average Accuracy: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")
    
    # Compare to baseline
    baseline = 0.5611
    improvement = (mean_acc - baseline) * 100
    print(f"\nComparison:")
    print(f"  Baseline (single-point): 56.11%")
    print(f"  Windowed features:       {mean_acc*100:.2f}%")
    print(f"  Improvement:             {improvement:+.2f}%")
    
    if mean_acc > 0.70:
        print(f"\n[EXCELLENT] State-of-the-art performance!")
    elif mean_acc > 0.65:
        print(f"\n[VERY GOOD] Strong generalization to new users.")
    elif mean_acc > 0.60:
        print(f"\n[GOOD] Windowing significantly improved accuracy.")
    else:
        print(f"\n→ Modest improvement. Consider adding more features.")
    
    return accuracies

def main():
    """Main training pipeline with progress tracking"""
    total_start = time.time()
    
    # Change to iot-sign-glove root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    print(f"\n{'='*60}")
    print(f"WINDOWED FEATURE TRAINING - ACCURATE VERSION")
    print(f"{'='*60}")
    print(f"This will take approximately 30-45 minutes total")
    print(f"You'll see detailed progress for each step")
    print(f"{'='*60}")
    
    # Load data
    X, y, users = load_windowed_data(window_size=0.5, stride=0.25)
    if X is None:
        return
    
    # Train-test split
    print(f"\n{'='*60}")
    print(f"PREPARING DATA (80/20 SPLIT)")
    print(f"{'='*60}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Training: {len(X_train):,} windows")
    print(f"[OK] Testing:  {len(X_test):,} windows")
    
    # Train models
    print(f"\n{'='*60}")
    print(f"STEP 1/3: TRAIN MODELS ON 80/20 SPLIT")
    print(f"{'='*60}")
    
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    gb_model = train_gradient_boosting(X_train, y_train, X_test, y_test)
    mlp_model, scaler, label_encoder = train_mlp(X_train, y_train, X_test, y_test)
    
    # LOUO CV
    print(f"\n{'='*60}")
    print(f"STEP 2/3: LEAVE-ONE-USER-OUT CROSS-VALIDATION")
    print(f"{'='*60}")
    print(f"This is the REAL test - can models work for NEW users?")
    
    print(f"\n\nRunning RF LOUO...")
    rf_louo = leave_one_user_out_cv(X, y, users, model_type='rf')
    
    print(f"\n\nRunning GB LOUO...")
    gb_louo = leave_one_user_out_cv(X, y, users, model_type='gb')
    
    # Save models
    print(f"\n{'='*60}")
    print(f"STEP 3/3: SAVING MODELS")
    print(f"{'='*60}")
    
    os.makedirs("models/windowed", exist_ok=True)
    
    joblib.dump(rf_model, "models/windowed/rf_model_0.5s.pkl")
    joblib.dump(gb_model, "models/windowed/gb_model_0.5s.pkl")
    joblib.dump(mlp_model, "models/windowed/mlp_model_0.5s.pkl")
    joblib.dump(scaler, "models/windowed/scaler.pkl")
    joblib.dump(label_encoder, "models/windowed/label_encoder.pkl")
    
    print(f"[OK] Saved all models to models/windowed/")
    
    # Final summary
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    
    print(f"\nFINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Model Performance (Leave-One-User-Out):")
    print(f"  Random Forest:       {np.mean(rf_louo)*100:.2f}% (±{np.std(rf_louo)*100:.2f}%)")
    print(f"  Gradient Boosting:   {np.mean(gb_louo)*100:.2f}% (±{np.std(gb_louo)*100:.2f}%)")
    print(f"\nBaseline comparison:")
    print(f"  Single-point (old):  56.11%")
    print(f"  Windowed (new):      {np.mean(rf_louo)*100:.2f}%")
    print(f"  Improvement:         {(np.mean(rf_louo)-0.5611)*100:+.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

