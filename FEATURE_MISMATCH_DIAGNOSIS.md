# Feature Mismatch Diagnosis & Solutions

## Current Issue

**Error**: LSTM/XGBoost models expect 42 features but only 41 are available
- Missing feature: `volume_ma`
- Actual features generated: 41
- Expected features: 42

---

## Root Causes

### 1. **Volume Column Missing or Incorrectly Named**
- `compute_features()` tries to generate `volume_ma` from volume data
- If the raw data doesn't have a 'volume' column, this fails
- yfinance might return volume as 'Volume' or under MultiIndex columns

### 2. **Feature Engineering Mismatch**
- Models were trained with specific features in specific order
- Current data generation differs from training data
- Metadata.json lists 42 expected features
- Actual generation produces 41

### 3. **Scaler Fitting Issue**
- The scaler (MinMaxScaler) was fit on 42 features
- When we pad with zeros, it breaks the feature scaling
- Padding zeros doesn't work well with normalized data

---

## Current Workarounds Applied

✅ **Added graceful fallbacks:**
1. Detect missing features
2. Use only available features
3. Pad with zeros if needed (suboptimal but functional)
4. Print detailed error messages

✅ **Better error handling:**
- Show which features are missing
- Use LSTM if XGBoost fails, or vice versa
- Display full error traces for debugging

---

## Solutions to Try (In Order)

### Solution 1: Check If Volume Data Exists
```python
# In fetch_data cell, add:
print(f'Columns in entry_df: {list(entry_df.columns)}')
print(f'Has volume: {"volume" in entry_df.columns}')
```

**If YES → Problem is elsewhere**
**If NO → See Solution 2**

### Solution 2: Check Raw Data Before Feature Engineering
```python
# Check raw XAUUSD data
yf_sym = SYMBOL_MAP.get('XAUUSD', 'XAUUSD')
raw_df = yf.download(yf_sym, period='7d', interval='1h', progress=False)
print(f'Raw columns: {list(raw_df.columns)}')
print(f'Raw shape: {raw_df.shape}')
```

### Solution 3: Fix compute_features() to Handle Missing Volume
If volume is missing, update `src/feature_engineering.py`:

```python
def compute_features(df):
    # ... existing code ...
    
    # Volume features (skip if volume not available)
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1)
    else:
        # Fallback: use constant values
        df['volume_ma'] = df['close']  # Placeholder
        df['volume_ratio'] = 1.0
```

### Solution 4: Update Metadata to Match Current Features
If volume_ma will never be available:

```python
# Update results/advanced_models/metadata.json
# Change "features" array to remove 'volume_ma'
# Update feature_count from 42 to 41
```

### Solution 5: Retrain Models Without volume_ma
If volume_ma is not critical:

```bash
python training/03_train_models_xauusd_optimized.py
# Will retrain on 41 features instead of 42
```

---

## Recommended Fix

**Option A (Quick Fix - Recommended for testing):**
1. Run the diagnostic in fetch_data cell
2. Check if volume exists
3. If not, update compute_features() to skip/placeholder volume_ma
4. Notebook will use 41 features with padding

**Option B (Proper Fix - Best long-term):**
1. Verify what features are actually available in live data
2. Update metadata to reflect actual feature list
3. Retrain models on the correct feature set
4. Remove padding (models trained on real features only)

**Option C (Hybrid - Balanced):**
1. Use Solution 3 (add volume placeholder)
2. Retrain models once with stable feature set
3. Use retr ained models going forward

---

## Testing the Fix

Run notebook and check for these messages:

**Good (Working):**
```
[OK] LSTM loaded: 42 features
[OK] XGBoost loaded: 42 features
[ENSEMBLE] Combined LSTM/XGB: 0.675
[1] Running ML Ensemble Forecast...
    Direction: BUY
    Confidence: 72% (LSTM: 0.65 | XGBoost: 0.70)
```

**Acceptable (Fallback working):**
```
[WARN] LSTM: Missing 1 features: ['volume_ma']
[WARN] LSTM feature shape mismatch: 41 vs 42
[ENSEMBLE] Using XGBoost only (LSTM failed: Input shape mismatch)
```

**Failed (Both models down):**
```
[ENSEMBLE ERROR] Both models failed
  LSTM: [LSTM ERROR] Input 0 with name...
  XGB: [XGBOOST ERROR] ...
[STOP] Cannot proceed without valid ML forecast
```

---

## Prevention

Going forward:

1. **Always log feature generation:**
   ```python
   feat_df = compute_features(df)
   print(f"Generated {feat_df.shape[1]} features: {list(feat_df.columns)}")
   ```

2. **Match metadata to reality:**
   - After retraining models, update metadata.json immediately
   - Include feature list in metadata

3. **Add feature validation:**
   ```python
   required_features = json.load(open('metadata.json'))['features']
   available = [f for f in required_features if f in df.columns]
   if len(available) < len(required_features):
       print(f"[ERROR] Missing {len(required_features)-len(available)} features")
   ```

---

## Next Steps

1. **Run fetch_data cell** with diagnostics enabled
2. **Check diagnostic output** to see if volume_ma is truly missing
3. **Apply appropriate solution** based on findings
4. **Test with LIVE=False** (backtest mode) first
5. **Monitor for errors** and adjust as needed

Once you run the diagnostic and see what features are actually being generated, let me know what the output says and we can apply the appropriate fix!
