# Feature Improvements V2 - Balanced Approach

## Problem with V1 (60 features):
- **SEVERE OVERFITTING**
- 80/20 split: 97% accuracy ✅
- LOUO: 60.74% accuracy ❌ (WORSE than 68.31% baseline!)
- Cause: Too many user-specific features with only 12 users

---

## V2 Changes - The Sweet Spot (35 features):

### **KEPT (Working features):**
1. ✅ Mean, std, min, max per sensor (4 features)
2. ✅ Velocity (1st derivative) per sensor  
3. ✅ Acceleration (2nd derivative) per sensor
4. ✅ **Jerk (3rd derivative)** per sensor - NEW! From professor's paper
5. ✅ **Min-max normalization** - Better generalization
6. ✅ **Butterworth filter** - Noise reduction

**Total: 5 sensors × 7 features = 35 features**

### **REMOVED (Overfitting culprits):**
1. ❌ **Recency features (last 3 samples)** - Too user-specific, memorized individual styles
2. ❌ **RMS** - Redundant with mean/std
3. ❌ **Zero-crossing rate** - Needs more training data
4. ❌ **Z-score normalization** - Made features too user-specific

---

## Expected Results:

**Previous best (30 features):**
- RF LOUO: 68.31%
- GB LOUO: 68.61%

**V2 target (35 features):**
- RF LOUO: **70-72%** (modest improvement)
- GB LOUO: **71-73%** (modest improvement)

**Why this should work:**
- Jerk adds meaningful temporal information (forceful vs gentle)
- Fewer features = less overfitting
- Min-max normalization = better generalization
- Still captures all important dynamics

---

## What to Tell Professor:

> "I implemented jerk (3rd derivative) from your paper, which captures whether movements are forceful or gentle. This increased features from 30 to 35.
>
> I initially tried adding more features (RMS, zero-crossing rate, recency) but encountered severe overfitting - the model memorized user-specific patterns instead of learning generalizable gestures.
>
> The balanced approach with 35 features adds meaningful temporal information (jerk) while maintaining good generalization to new users."

---

## Technical Details:

### Feature Extraction (per sensor):
```python
# Static (4)
mean, std, min, max

# Dynamic (3)  
velocity = 1st derivative × sample_rate
acceleration = 2nd derivative × sample_rate²
jerk = 3rd derivative × sample_rate³  # NEW!
```

### Jerk Physical Meaning:
- **High jerk**: Forceful, sudden movements (e.g., Grasp)
- **Low jerk**: Gentle, smooth movements (e.g., Peace sign)
- **From professor's FingerPhase.py line 294**

### Normalization:
```python
# Min-max (better for generalization)
normalized = (raw - calibration_min) / (calibration_max - calibration_min)
```

---

## Timeline:
- V0 (Baseline): 30 features, 68.31% LOUO ✅
- V1 (Overfitted): 60 features, 60.74% LOUO ❌
- **V2 (Balanced): 35 features, target 70-72% LOUO** ⏳







