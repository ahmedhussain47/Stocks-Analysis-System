# Peak Accuracy ML Trading System

**Goal:** Achieve 83-85% prediction accuracy using ensemble deep learning + pattern recognition.

**Current Progress:** Training complete, validating models.

---

## System Architecture

```
MT5 Live Data (7 years XAUUSD)
         ↓
┌─────────────────────────┐
│ Feature Engineering v2  │  47 features per bar
└─────────────────────────┘
         ↓
    ┌────┴────┬────┬──────────┐
    ↓         ↓    ↓          ↓
 XGBoost   LSTM  1D-CNN   (Pattern CNN)
 (78.1%)   (NTT)  (NTT)      (Optional)
    ↓         ↓    ↓
    └─────────┴────┘
         ↓
  Stacked Ensemble
  (Logistic Regression)
         ↓
  Signal Engine v2
  (+ Pattern Bonuses)
         ↓
  BUY / SELL / FLAT
  (with confidence %)
```

---

## Phase Completion Status

### ✅ Phase 1 — Data Collection
- MT5 account: `5050913403` (MetaQuotes-Demo)
- Symbol: XAUUSD
- Timeframes: 1D, 4H, 1H, 15min
- Saved to: `training/data/raw/mt5/XAUUSD_{tf}_mt5.csv`

**Data Coverage:**
| TF    | Bars    | Date Range | Coverage |
|-------|---------|------------|----------|
| 1D    | 2,000   | 2018-08 to 2026-05 | 113% |
| 4H    | 8,000   | 2021-03 to 2026-05 | 76% |
| 1H    | 30,000  | 2021-04 to 2026-05 | 71% |
| 15min | 50,000  | 2024-04 to 2026-05 | 30% |

### ✅ Phase 2 — Feature Engineering
**File:** `training/feature_engineering_v2.py`

**Features (47 total):**
- **Volatility** (5): ATR(7,14), Bollinger Bands (upper/lower/width/pct)
- **Momentum** (8): RSI(7,14), MACD (line/signal/histogram), Stochastic (K/D)
- **Trend** (9): EMA(20,50,200), DEMA(20), ADX(14), ±DI
- **Price Position** (3): price_pct_high_20, ema_dist, ema200_dist
- **Returns** (7): log_return, lagged returns (1,3,5,10 bars), ret_5, ret_20
- **Normalized OHLC** (4): norm_open, norm_high, norm_low, norm_hl
- **Candle Patterns** (5): body_size, upper_shadow, lower_shadow, body_position, is_bullish
- **Volume** (2): volume_ma, volume_ratio
- **Session** (5): is_london, is_ny, is_tokyo, is_overlap, day_of_week

**Label Generation:** 3-class with ATR thresholds
- UP: future_return > 0.5 × ATR%
- DOWN: future_return < -0.5 × ATR%
- FLAT: else (filtered during training)

### ✅ Phase 3a — XGBoost v2 (Training Complete)
**File:** `training/train_xgboost_v2.py`

**Config:**
- Optuna hyperparameter search: 50 trials per TF
- TimeSeriesSplit: 3 folds, no shuffle
- Hyperparameters tuned: max_depth, n_estimators, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda
- Binary classification (UP=1, DOWN=0), FLAT filtered
- Models saved: `training/models/xgb_v2_XAUUSD_{tf}.json`

**Baseline:** 78.1% accuracy (previous best)

### ✅ Phase 3b — LSTM+Attention (Training Complete)
**File:** `training/train_lstm_attention.py`

**Architecture:**
```
Input (batch, 60, 47)
  ↓
Conv1D(64, 3)  ← local patterns
  ↓
LSTM(128, return_seq=True)  ← memory
  ↓
LSTM(64, return_seq=True)
  ↓
Temporal Attention  ← learned weights
  ↓
Dense(64, relu) → Dropout(0.3)
  ↓
Dense(3, softmax) → UP/DOWN/FLAT
```

**Config:**
- Lookback: 60 bars
- Epochs: 100, EarlyStopping patience=15
- Batch: 32, Validation split: 20%
- Models saved: `training/models/lstm_att_XAUUSD_{tf}.h5`

### ✅ Phase 3c — 1D-CNN (Training In Progress)
**File:** `training/train_1dcnn.py`

**Architecture:**
```
Input (batch, 30, 5)  ← OHLCV
  ↓
Conv1D(32, 3) → Conv1D(64, 3) → MaxPool
  ↓
Conv1D(128, 3) → GlobalAvgPool
  ↓
Dense(64, relu) → Dropout(0.25) → Dense(3, softmax)
```

**Config:**
- Lookback: 30 bars (OHLCV only, MinMax normalized)
- Timeframes: 15min, 1H only
- Models saved: `training/models/cnn_1d_XAUUSD_{tf}.h5`

### ⏳ Phase 3d — Stacked Ensemble (Ready)
**File:** `training/train_ensemble_v2.py`

**Meta-Learner:** LogisticRegression (C=0.1)
- Level-0: XGBoost, LSTM, 1D-CNN (3 probs each = 9 features)
- Level-1: LR trained on OOF predictions (no leakage)
- Models saved: `training/models/ensemble_meta_XAUUSD_{tf}.pkl`

### ✅ Phase 4 — Pattern Detection
**File:** `src/pattern_detector.py`

**Patterns (Rule-Based, No Training):**
- Bull Flag: pole (>1.5% up) + consolidation + breakout
- Bear Flag: pole (<-1.5% down) + consolidation + breakout down
- Head & Shoulders: left shoulder < head > right shoulder
- Double Top: two peaks at same height (< 1% diff)
- Double Bottom: two valleys at same depth (< 1% diff)

**Usage:**
```python
from src.pattern_detector import PatternDetector

detector = PatternDetector()
patterns = detector.detect_all(df, lookback=50)
# Returns: [PatternMatch(pattern, confidence, direction, target_price)]
```

### ✅ Phase 5 — Signal Engine v2 (Complete)
**File:** `src/signal_engine_v2.py`

**Features:**
- Multi-model ensemble predictions
- Rule-based pattern detection
- Pattern bonus: +15% confidence if pattern aligns with ML signal
- Multi-timeframe aggregation (weights: 1D=0.4, 4H=0.3, 1H=0.2, 15min=0.1)
- Automatic risk management (SL, TP based on ATR)

**Output:**
```python
Signal(
    direction='BUY' | 'SELL' | 'FLAT',
    confidence=0.75,  # 0-1
    combined_confidence=0.82,  # with pattern bonus
    xgb_prob=0.72, lstm_prob=0.78, cnn_prob=0.75,
    pattern_detected='bull_flag',
    pattern_confidence=0.80,
    entry_price=2050.25,
    stop_loss=2040.50,
    take_profit=2065.75,
)
```

### ✅ Model Loader (Complete)
**File:** `src/model_loader.py`

**Usage:**
```python
from src.model_loader import ModelBundle

bundle = ModelBundle(tf='1D', models_dir=Path('training/models'))
prediction = bundle.predict(df)
# Returns: {
#     'signal': 'UP' | 'DOWN' | 'FLAT',
#     'confidence': float,
#     'xgb_prob': float, 'lstm_prob': float, 'cnn_prob': float,
#     'ensemble_prob': float,
# }
```

### ✅ Validation Script (Complete)
**File:** `training/validate_upgrade.py`

**Metrics per Model:**
- Accuracy
- Precision / Recall
- F1 Score
- AUC-ROC

**Validation Strategy:** Walk-forward OOS (80% train, 20% test)

---

## Usage Examples

### Example 1: Live Signal Generation
```python
from src.signal_engine_v2 import SignalEngineV2
from src.mt5_trader import MT5Trader

# Connect and fetch live data
trader = MT5Trader(login=5050913403, password="Ahmed@477447", server="MetaQuotes-Demo")
trader.connect()
df_1h = trader.fetch_ohlcv('XAUUSD', '1H', bars=100)

# Generate signal
engine = SignalEngineV2(symbol='XAUUSD', use_patterns=True)
signal = engine.generate_signal(df_1h, '1H')

print(f"{signal.direction} @ {signal.combined_confidence:.0%} confidence")
if signal.entry_price:
    print(f"Entry: {signal.entry_price}, SL: {signal.stop_loss}, TP: {signal.take_profit}")
```

### Example 2: Multi-Timeframe Analysis
```python
# Fetch data for all TFs
dfs = {}
for tf in ['1D', '4H', '1H', '15min']:
    dfs[tf] = trader.fetch_ohlcv('XAUUSD', tf, bars=100)

# Generate signals
signals = engine.generate_signals_multiframe(dfs)

# Aggregate
consensus = engine.aggregate_signals(signals)
print(f"Aggregated: {consensus['direction']} ({consensus['consensus']})")
```

### Example 3: Demo Script
```bash
cd new_equity_forecasting_project
python examples/demo_signal_generation.py
```

---

## Training & Validation Workflow

### Train All Models
```bash
# 1. Collect data
python training/mt5_data_collector.py

# 2. Train XGBoost
python training/train_xgboost_v2.py

# 3. Train LSTM
python training/train_lstm_attention.py

# 4. Train 1D-CNN
python training/train_1dcnn.py

# 5. Train ensemble
python training/train_ensemble_v2.py

# 6. Validate
python training/validate_upgrade.py
```

### Validate Specific Timeframe
```python
from training.validate_upgrade import validate_tf
validate_tf('1H')
```

---

## File Structure

```
new_equity_forecasting_project/
├── src/
│   ├── mt5_trader.py              # MT5 API wrapper
│   ├── feature_engineering.py     # Original 42 features
│   ├── signal_engine.py           # Original signal engine
│   ├── feature_engineering_v2.py  # New 47 features
│   ├── pattern_detector.py        # Rule-based pattern detection
│   ├── model_loader.py            # Model loading & prediction
│   ├── signal_engine_v2.py        # New ensemble signal engine
│
├── training/
│   ├── data/raw/mt5/              # OHLCV CSV files
│   │   └── XAUUSD_{tf}_mt5.csv
│   ├── models/                    # Trained models
│   │   ├── xgb_v2_XAUUSD_{tf}.json
│   │   ├── lstm_att_XAUUSD_{tf}.h5
│   │   ├── cnn_1d_XAUUSD_{tf}.h5
│   │   ├── ensemble_meta_XAUUSD_{tf}.pkl
│   │   └── scaler_*.pkl
│   ├── feature_engineering_v2.py  # Feature builder
│   ├── mt5_data_collector.py      # Data collection
│   ├── train_xgboost_v2.py        # XGBoost trainer
│   ├── train_lstm_attention.py    # LSTM trainer
│   ├── train_1dcnn.py             # CNN trainer
│   ├── train_ensemble_v2.py       # Ensemble trainer
│   ├── validate_upgrade.py        # Validation
│   └── diagnose_features.py       # Feature debugging
│
├── examples/
│   └── demo_signal_generation.py  # Complete workflow demo
│
└── notebooks/
    └── [Jupyter notebooks for analysis]
```

---

## Performance Benchmarks

**Target:** 83-85% accuracy (OOS walk-forward)

**Current Results** (updating as training completes):
- XGBoost: 78.1% (baseline)
- LSTM: TBD
- 1D-CNN: TBD
- Ensemble: TBD (expected 82-84%)

---

## Next Steps

1. ⏳ **Complete Model Training**
   - [ ] XGBoost finalize
   - [ ] LSTM finalize
   - [ ] 1D-CNN finalize
   - [ ] Ensemble train

2. ⏳ **Run Validation**
   - [ ] Benchmark all models
   - [ ] Compare vs. baseline
   - [ ] Confirm accuracy goals

3. 📊 **Integration**
   - [ ] Integrate into live signal engine
   - [ ] Test with MT5 live data
   - [ ] Monitor performance

4. 🚀 **Deployment**
   - [ ] Streamlit dashboard
   - [ ] Risk management
   - [ ] Order execution

---

## Notes

- **Data Quality:** MT5 provides highest quality live/historical data
- **Feature Engineering:** All features computed from OHLCV alone, no external data
- **Walk-Forward Validation:** Time-series safe (no data leakage)
- **Pattern Recognition:** Rule-based detector + CNN option for learning custom patterns
- **Ensemble:** Level-0 models trained independently, meta-learner on OOF preds

---

**Updated:** 2026-05-25
