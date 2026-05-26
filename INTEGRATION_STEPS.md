# INTEGRATION STEPS — How to Apply Fixes to Notebook

## 🎯 Goal
Replace the broken live scanner (cell-11) with the corrected 45-feature version and add performance diagnostics.

---

## 📝 STEP 1: Fix Config (5 min)

Edit **cell_config** (current settings):

```python
# CHANGE THIS:
REQUIRE_MODEL_CONSENSUS = True  # Impossible with only 2 models!

# TO THIS:
REQUIRE_MODEL_CONSENSUS = False  # Only 2 models loaded (LSTM + XGBoost)
```

**Why?** The config was written for 3 models (AutoETS, Chronos, LSTM) but only LSTM + XGBoost are loaded.

---

## 🔧 STEP 2: Replace Live Scanner (5 min)

**Old (Broken) Code Location:** 
- Cell ID: `cell-11` 
- Problem: Uses `quick_features()` with only 8 features
- Result: Model predictions unreliable

**Replace with:**

Copy the entire code from `LIVE_SCANNER_CORRECTED.py` into cell-11, replacing everything.

**Key improvements in corrected version:**
```python
# OLD (BROKEN)
def quick_features(df):
    # Returns only 8 features
    return {'ema_20', 'ema_50', 'rsi_14', ...}

# NEW (CORRECT)
from src.feature_engineering import compute_features
feat_df = compute_features(df)  # Returns 45 features!
X_raw = feat_df_clean[feature_list].iloc[-1:].values  # Proper feature extraction
```

---

## 📊 STEP 3: Add Diagnostics Cell (2 min)

**Location:** Add new cell AFTER all other cells

**Copy from:** `DIAGNOSTICS_CELL.py`

**What it does:**
- Reads `results/signals_log.csv`
- Computes Sharpe ratio
- Computes Calmar ratio
- Computes Profit Factor
- Shows recent trades
- Gives recommendations

---

## ✅ STEP 4: Restart & Test (10 min)

1. **Kernel → Restart & Clear Output**
2. **Run All Cells** (Ctrl+Shift+Enter)
3. **Wait for models to load:**
   ```
   Loading LSTM model ... OK  (45 features)
   LSTM accuracy: 78.1%  |  AUC: 0.8579
   Loading XGBoost model ... OK  (45 features)
   XGBoost accuracy: 78.1%  |  AUC: 0.8579
   ```

4. **Run cell-11 (corrected scanner):**
   - Change: `LIVE = False` → `LIVE = True`
   - Run the cell
   - Wait for first signal

---

## 📈 STEP 5: Monitor Signals (30 min)

**Expected output when signal found:**
```
[12:34:56][  1] Running ML forecast on 500 bars...

======================================================================
SIGNAL: BUY  XAUUSD [1H]
Entry:     4523.20000
SL:        4507.20000  (16.00 pts)           ← Tight SL (good!)
TP1/2/3:   4539.70 / 4549.20 / 4557.30      ← Good TP progression
R:R:       1:1.96                            ← R:R ratio ~2:1 (good!)
Conf:      73%                               ← Confidence score
Models:    LSTM: 1.58% | XGBoost: 11.98% | Ensemble: 6.78%
======================================================================
```

**Good signs:**
- ✅ SL is tight (11-16 pts, not 25+)
- ✅ TP/SL ratio is ~2:1
- ✅ Confidence is 60-95%
- ✅ Both models producing predictions

**Bad signs:**
- ❌ "Forecast returned None" (feature mismatch still)
- ❌ Feature count mismatch errors
- ❌ SL > 20 pts (too wide)

---

## 📋 STEP 6: Run Diagnostics (5 min)

After 5+ signals have closed:

1. **Run DIAGNOSTICS_CELL** (new cell added in step 3)
2. **Check output:**
   ```
   PERFORMANCE METRICS
   ══════════════════
   Total Trades       : 5
   Winning Trades     : 3 (60%)  ← Should be > 50%
   Cumulative PnL     : +2.34%
   Sharpe Ratio       : 0.82    ← Should be > 0.3
   Calmar Ratio       : 0.45    ← Should be > 0.2
   Profit Factor      : 1.82    ← Should be > 1.2
   ```

3. **Compare to baseline:**
   - Previous (broken): 25% win rate, -2% PnL
   - Expected (fixed): 50-55% win rate, +2-5% PnL

---

## 🔍 TROUBLESHOOTING

### "Feature count mismatch"
```
[ERROR] XGBoost: feature count mismatch 8 != 50
```
**Fix:** Make sure you replaced the ENTIRE cell-11 with corrected version.

### "LSTM: no complete bars after dropna"
```
[LSTM: no complete bars after dropna()]
```
**Cause:** First few bars have NaN (lagged indicators)  
**Fix:** Normal—wait 50+ bars before first signal expected

### "Model loading failed"
```
[ERROR] Failed to load models: Unable to synchronously open file
```
**Cause:** File path is wrong  
**Fix:** Check that paths use `../results/advanced_models/` (relative path)

### Still 25% win rate
```
Total Trades: 10
Winning Trades: 2 (20%)
Sharpe Ratio: -0.15
```
**Problem:** Feature fix didn't help  
**Options:**
1. Check that actual features in signals match expected 45
2. Verify models were trained on **recent** XAUUSD data (not months old)
3. Consider Phase 2: Retrain on 7 years of multi-asset data

---

## 📝 QUICK CHECKLIST

- [ ] Changed `REQUIRE_MODEL_CONSENSUS = False` in config
- [ ] Replaced cell-11 with corrected live scanner
- [ ] Added DIAGNOSTICS_CELL as new cell
- [ ] Kernel → Restart & Run All
- [ ] Waited for models to load (should show 78.1% accuracy)
- [ ] Set `LIVE = True` in cell-11
- [ ] Got first signal with tight SL (~15 pts)
- [ ] Let 5+ signals close
- [ ] Ran DIAGNOSTICS_CELL
- [ ] Saw Sharpe > 0.3 or Profit Factor > 1.2
- [ ] Confirmed win rate improved from 25% baseline

---

## ⏱️ ESTIMATED TIME
- Steps 1-5: **20 minutes**
- Step 6 (diagnostics): **Requires 5+ closed signals (~2-3 hours trading)**

---

## 🚨 CRITICAL POINTS

1. **The old live scanner used 8 features—models expect 45**
   - This wasn't a small bug, it was fundamental
   - Fixes should show dramatic improvement

2. **Use `LIVE = True` only after verifying setup**
   - Test with real market data
   - Watch first 5 signals carefully

3. **Performance metrics are KEY**
   - Don't trust win rate alone
   - Check Sharpe & Calmar
   - Verify Profit Factor > 1.0

4. **If performance doesn't improve**
   - Problem may be model age (trained on old data)
   - Consider Phase 2: Retrain on 7 years data + multi-asset

---

## 📞 SUPPORT

If issues arise, check:
1. CRITICAL_FIXES_SUMMARY.md — Full technical details
2. DIAGNOSTICS_CELL.py — Detailed error messages
3. Feature mismatch errors — Almost certainly a copy-paste issue in cell-11
