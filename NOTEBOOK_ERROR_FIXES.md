# Notebook Error Fixes

## Issue: TypeError - NoneType object is not subscriptable

**Error:**
```
TypeError: 'NoneType' object is not subscriptable
Cell In[23], line 55: sl, atr = smart_stop_loss(entry_df, fc['direction'], ...)
```

**Root Cause:** `ensemble_forecast()` returned `None` instead of a valid forecast dictionary.

---

## Solutions Implemented

### Engine 1 (notebooks/05_signal_engine.ipynb)

#### 1. **Enhanced Model Loading (load_models cell)**
- Added file existence checks before loading models
- Better error messages showing which file is missing
- Warnings if models fail to load

```python
# Check files exist
if not model_path.exists():
    print(f'[ERROR] LSTM model not found: {model_path}')
    return False
```

#### 2. **Improved Ensemble Forecast (ensemble_forecast cell)**
- Added error tracking with messages from individual models
- Fallback to single model if one fails
- Better error diagnostics

```python
def ensemble_forecast(df):
    lstm_pred, lstm_err = run_lstm_prediction(df)
    xgb_pred, xgb_err = run_xgboost_prediction(df)
    
    if lstm_pred is None and xgb_pred is None:
        print(f'[ENSEMBLE ERROR] Both models failed: {lstm_err} {xgb_err}')
        return None
    
    # Fallback if one model fails
    if lstm_pred is None or xgb_pred is None:
        ensemble_pred = lstm_pred if lstm_pred is not None else xgb_pred
    else:
        ensemble_pred = lstm_pred * 0.4 + xgb_pred * 0.6
```

#### 3. **Robust Signal Generation (generate_signal cell)**
- Added `None` check before accessing `fc` dictionary
- Graceful error handling with debug information
- Only proceeds if valid forecast is available

```python
if fc is None:
    print('[ERROR] Ensemble forecast failed - models may not be loaded')
    print('[DEBUG] LSTM OK:', lstm_ok, '| XGBoost OK:', xgb_ok)
    print('[STOP] Cannot proceed without valid ML forecast')
else:
    # Process signal normally
    # ...
```

---

### Engine 2 (notebooks/06_signal_engine.ipynb)

#### 1. **Enhanced Time-Series Forecast (forecast_function cell)**
- Added try-catch around entire forecast
- Returns 'UNKNOWN' instead of None on error
- Better error messages

```python
def forecast_ts(close_prices, method='ensemble'):
    try:
        # ... forecast logic ...
        return direction, confidence
    except Exception as e:
        print(f'[FORECAST ERROR] {str(e)[:60]}')
        return 'UNKNOWN', 30  # Safe fallback
```

#### 2. **Safe Signal Generation (signal_generation cell)**
- `generate_signal()` returns `None` if direction is 'UNKNOWN'
- `live_scanner()` already has safe None checks
- No subscripting of None values possible

---

## What to Do If You Still Get Errors

### If models don't load:
1. Check file paths exist:
   ```bash
   ls -la results/advanced_models/best_model_lstm_final.h5
   ls -la results/advanced_models/best_model_xgb_final.pkl
   ls -la results/advanced_models/scaler_final.pkl
   ls -la results/advanced_models/metadata.json
   ```

2. If files missing, check working directory is correct (should be notebook root, not notebooks/)

### If compute_features fails:
1. Check data is properly loaded in fetch_data cell
2. Run diagnostics:
   ```python
   from src.feature_engineering import compute_features
   df = entry_df.copy()
   try:
       feat = compute_features(df)
       print(f"Features computed: {feat.shape}")
   except Exception as e:
       print(f"Feature error: {e}")
   ```

### If yfinance fails:
1. Check internet connection
2. Verify SYMBOL_MAP has XAUUSD entry
3. Check Yahoo Finance API is accessible

---

## Testing the Fixes

Run the notebooks in order:

**Engine 1:**
```jupyter
# In notebook 05_signal_engine.ipynb:
1. Run "imports" cell → should see "[OK] Imports complete"
2. Run "config" cell → should see configuration summary
3. Run "load_models" cell → should see both models load successfully
4. Run "pattern_recognition" cell → initialize engine
5. Run "smart_sl" cell → initialize SL system
6. Run "ensemble_forecast" cell → test forecast function
7. Run "fetch_data" cell → download XAUUSD data
8. Run "generate_signal" cell → should generate signal or skip with reason
```

**Expected Output (if successful):**
```
[OK] LSTM loaded: 42 features, 54.0% accuracy
[OK] XGBoost loaded: 42 features, 57.2% accuracy

[STATUS] Models: LSTM=True XGBoost=True

...

[1] Running ML Ensemble Forecast...
    Direction: BUY
    Confidence: 75% (LSTM: 0.62 | XGBoost: 0.78)

[2] Pattern Recognition...
    Candlestick Patterns: 50% quality
    ...
```

**Expected Output (if models fail):**
```
[ERROR] LSTM load failed: [Errno 2] No such file or directory
[STATUS] Models: LSTM=False XGBoost=True

...

[1] Running ML Ensemble Forecast...
[ERROR] Ensemble forecast failed - models may not be loaded
[DEBUG] LSTM OK: False | XGBoost OK: True
[STOP] Cannot proceed without valid ML forecast
```

---

## Files Modified

- `notebooks/05_signal_engine.ipynb`
  - load_models cell: Added file checks
  - ensemble_forecast cell: Added error tracking and fallback
  - generate_signal cell: Added None checks

- `notebooks/06_signal_engine.ipynb`
  - forecast_function cell: Added try-catch and safe fallback

---

## Summary

Both notebooks now have **defensive error handling** that:
- ✓ Checks if files exist before loading
- ✓ Catches and reports errors with diagnostics
- ✓ Never subscripts None values
- ✓ Falls back gracefully when models unavailable
- ✓ Provides clear error messages for debugging

**Next Steps:**
1. Run notebooks and verify models load
2. If models don't load, check file paths and working directory
3. If compute_features fails, debug feature engineering independently
4. Once working, test with LIVE=False first (backtest mode)
5. Then enable LIVE=True for continuous scanning
