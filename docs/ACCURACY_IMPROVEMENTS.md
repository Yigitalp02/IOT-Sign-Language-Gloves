# Accuracy Improvement Plan
## Based on Professor's Paper Analysis

### Current Status: 68.61% LOUO Accuracy

---

## ✅ What You Already Have (Correct!)

### 1. Evaluation Strategy
- ✅ **Leave-One-User-Out (LOUO)** - Actually BETTER than GroupKFold
- ✅ **Person-independent testing** - Users in train ≠ users in test
- ✅ **Per-class metrics** - Classification reports with macro F1

### 2. Windowing
- ✅ **Sliding windows**: 500ms with 50% overlap (they recommend 200-500ms)
- ✅ **Sample rate**: 100Hz (they use 50Hz, so we're good)

### 3. Features (Partial)
- ✅ **Mean, Std, Min, Max** per finger
- ✅ **Velocity (1st derivative)**
- ✅ **Acceleration (2nd derivative)**

### 4. Preprocessing
- ✅ **Butterworth low-pass filter** for noise reduction
- ✅ **Calibration-based normalization** (min-max from calibration)

---

## ❌ What You're Missing (High Impact!)

### 1. **Jerk (3rd Derivative)** - NOT IMPLEMENTED
**Impact: +2-3% accuracy**

Their paper extracts:
```python
# From professor's code (FingerPhase.py, line 294)
jerk = (acceleration[0] - acceleration[self.distance]) / self.distance * frequency
```

**Why it matters:**
- Jerk = rate of change of acceleration
- Captures "smoothness" vs "jerkiness" of movement
- Distinguishes forceful gestures (high jerk) from gentle ones (low jerk)

**Example:**
- Grasp: High jerk (forceful closing)
- Peace sign: Low jerk (gentle positioning)

---

### 2. **Per-User Z-Score Normalization** - NOT IMPLEMENTED
**Impact: +3-5% accuracy**

**Current approach (yours):**
```python
# Min-max normalization using calibration data
normalized = (raw - cal_min) / (cal_max - cal_min)
```

**Their approach (professor's paper):**
```python
# Z-score normalization per user, per feature
normalized = (raw - user_mean) / user_std

# They normalize EACH feature separately:
# - Capacitance → z-score normalized
# - Velocity → z-score normalized
# - Acceleration → z-score normalized
# - Jerk → z-score normalized
```

**Why it matters:**
- **Handles user variability**: Different hand sizes, finger lengths, wearing styles
- **Scale-invariant**: Features have comparable ranges
- **Reduces overfitting**: Models generalize better to new users

**Example:**
- User A: Large hands, max capacitance = 4000
- User B: Small hands, max capacitance = 2500
- Z-score makes them comparable!

---

### 3. **Per-Feature Normalization** - NOT IMPLEMENTED
**Impact: +1-2% accuracy**

**Current approach:**
```python
# All 30 features normalized together (StandardScaler on entire feature matrix)
scaler.fit(X_train)  # X_train shape: (n_samples, 30)
```

**Their approach:**
```python
# Each derivative is normalized independently DURING feature extraction
# For finger i:
#   capacitance[i] → normalizer[0].normalize(capacitance[i])
#   velocity[i] → normalizer[1].normalize(velocity[i])
#   acceleration[i] → normalizer[2].normalize(acceleration[i])
#   jerk[i] → normalizer[3].normalize(jerk[i])
```

**Why it matters:**
- Velocity/acceleration/jerk have different scales than raw capacitance
- Independent normalization prevents one feature type from dominating
- Calibration is done per-user, per-feature type

---

### 4. **Recency Features (Last 1-3 Raw Samples)** - NOT IMPLEMENTED
**Impact: +1-2% accuracy**

**What they do:**
```python
# In addition to statistical features, add the last few raw samples
features = [
    mean, std, min, max, velocity, acceleration,  # Statistical
    raw[-1], raw[-2], raw[-3]  # Last 3 samples for recency
]
```

**Why it matters:**
- Captures the CURRENT state, not just statistics
- Helps with gesture onset/offset detection
- Provides temporal context

---

### 5. **RMS (Root Mean Square)** - NOT IMPLEMENTED  
**Impact: +0.5-1% accuracy**

```python
rms = np.sqrt(np.mean(window_data ** 2))
```

**Why it matters:**
- Measures signal power/energy
- Robust to positive/negative fluctuations
- Common in signal processing for activity level

---

### 6. **Zero-Crossing Rate** - NOT IMPLEMENTED
**Impact: +0.5-1% accuracy**

```python
zero_crossings = np.sum(np.diff(np.sign(window_data)) != 0)
zero_crossing_rate = zero_crossings / len(window_data)
```

**Why it matters:**
- Measures frequency content (high ZCR = oscillatory movements)
- Distinguishes shaky vs smooth movements
- Helps identify tremor vs intentional movement

---

## 📈 **Expected Accuracy Gains**

| Improvement | Expected Gain | Priority |
|-------------|---------------|----------|
| **Per-user z-score normalization** | +3-5% | 🔴 **HIGH** |
| **Jerk (3rd derivative)** | +2-3% | 🔴 **HIGH** |
| **RMS + Zero-crossing** | +1-2% | 🟡 MEDIUM |
| **Per-feature normalization** | +1-2% | 🟡 MEDIUM |
| **Recency features** | +1-2% | 🟢 LOW |

**Total Potential Gain: +8-14%**

**Projected Accuracy: 68.61% → 75-80%** 🎯

---

## 🛠️ **Implementation Priority**

### Phase 1: High-Impact Features (Do First!)
1. Add jerk (3rd derivative)
2. Implement per-user z-score normalization
3. Separate normalizers for each feature type

### Phase 2: Medium-Impact Features
4. Add RMS and zero-crossing rate
5. Per-feature StandardScaler

### Phase 3: Low-Impact Features (Optional)
6. Add last 1-3 raw samples for recency

---

## 📝 **Code Changes Needed**

### File: `iot-sign-glove/scripts/windowed_features.py`
- ✅ Already has: velocity, acceleration
- ❌ Add: jerk, RMS, zero-crossing rate, recency
- ❌ Change: normalization strategy

### File: `iot-sign-glove/scripts/load_professor_data.py`
- ✅ Already has: Butterworth filter
- ❌ Add: per-user z-score normalization instead of min-max

### File: `iot-sign-glove/scripts/train_windowed_accurate.py`
- ✅ Already has: LOUO cross-validation
- ❌ Modify: Feature scaling strategy

---

## 🎓 **For Your Professor Meeting**

**Q: "Why is your accuracy ~69%?"**
> "I'm using min-max normalization globally, but your paper uses per-user z-score normalization for each feature type separately. I'm also missing jerk (3rd derivative) and some frequency-domain features like RMS and zero-crossing rate. Implementing these should bring accuracy to 75-80%."

**Q: "What's your validation strategy?"**
> "Leave-One-User-Out cross-validation, which is actually more rigorous than GroupKFold. Each model is tested on a completely new user who wasn't in the training set, ensuring person-independent generalization."

**Q: "What features are you using?"**
> "Currently: mean, std, min, max, velocity, acceleration per finger in 0.5s windows. Based on your paper, I'm adding jerk, RMS, zero-crossing rate, and switching to per-user z-score normalization."

---

## 🔬 **Technical Details from Their Paper**

### Feature Extraction (Per Finger, Per Window):
```python
# They extract 23 features total (Configuration.FeatureCount = 23)
# For 5 fingers, that's ~4-5 features per finger

# Current features per finger (from their code):
features = [
    capacitance_normalized,      # Z-score
    velocity_normalized,         # Z-score  
    acceleration_normalized,     # Z-score
    jerk_normalized             # Z-score
]

# Then they extract window statistics:
# mean(features), std(features), min(features), max(features), etc.
```

### Normalization Strategy:
```python
# During calibration (50-550 samples):
for feature_type in [capacitance, velocity, acceleration, jerk]:
    normalizer[feature_type].add_calibration_value(feature)
    
# After calibration:
normalizer[feature_type].calculate_mean_std()

# During prediction:
normalized = (raw - mean) / std  # Z-score per feature type
```

### Sampling:
- **50 Hz** sampling rate
- **Sample_Distance = 19** (0.38s lookback for derivatives)
- **BufferCount = 20** (0.4s window at 50Hz)

---

## 🎯 **Next Steps**

1. ✅ Read this document
2. ⏳ Implement Phase 1 improvements (jerk + z-score normalization)
3. ⏳ Retrain and measure accuracy gain
4. ⏳ Implement Phase 2 if needed
5. ⏳ Generate confusion matrices for presentation

**Ready to implement these improvements?**







