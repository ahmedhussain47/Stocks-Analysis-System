# Quick Start — Peak Accuracy ML Trading System

## 5-Minute Setup

### 1. Verify Models Are Trained
```bash
ls training/models/
# Should show: xgb_v2_*, lstm_att_*, cnn_1d_*, ensemble_*, scaler_*
```

### 2. Run Signal Generation Demo
```bash
python examples/demo_signal_generation.py
```

Expected output:
```
Peak Accuracy ML Trading System — Signal Generation Demo
======================================================================

[1/4] Connecting to MT5...
[OK] MT5 Connected

[2/4] Fetching live data for XAUUSD...
  1D   ... [OK] 100 bars
  4H   ... [OK] 100 bars
  1H   ... [OK] 100 bars
  15min... [OK] 100 bars

[3/4] Initializing signal engine...
[OK] Signal engine ready

[4/4] Generating signals across timeframes...
  1D   ... [OK] BUY (78%)
  4H   ... [OK] SELL (65%)
  1H   ... [OK] BUY (82%)
  15min... [OK] FLAT (42%)

======================================================================
Aggregated Signal:
======================================================================
Direction:  BUY
Confidence: 75%
Consensus:  Mixed signals
Buy Count:  2
Sell Count: 1
```

---

## Common Tasks

### Check Model Accuracy
```bash
python training/validate_upgrade.py
```

Shows OOS accuracy for each model on each timeframe.

### Train New Models
```bash
# Collect 7 years of data
python training/mt5_data_collector.py

# Train XGBoost (15 min)
python training/train_xgboost_v2.py

# Train LSTM (30 min)
python training/train_lstm_attention.py

# Train CNN (20 min)
python training/train_1dcnn.py

# Train ensemble
python training/train_ensemble_v2.py
```

### Use in Your Code

```python
from src.signal_engine_v2 import SignalEngineV2
from src.mt5_trader import MT5Trader
import pandas as pd

# Fetch data
trader = MT5Trader(login=5050913403, password="Ahmed@477447")
trader.connect()
df = trader.fetch_ohlcv('XAUUSD', '1H', bars=100)

# Generate signal
engine = SignalEngineV2()
signal = engine.generate_signal(df, '1H')

# Check result
print(f"Signal: {signal.direction}")
print(f"Confidence: {signal.combined_confidence:.0%}")
print(f"Entry: {signal.entry_price}")
print(f"Stop: {signal.stop_loss}")
print(f"Target: {signal.take_profit}")

trader.disconnect()
```

---

## Understanding Signals

Each signal contains:

| Field | Meaning | Range |
|-------|---------|-------|
| `direction` | Trade action | BUY, SELL, FLAT |
| `confidence` | ML model confidence | 0-1 (50% = 0.5) |
| `combined_confidence` | With pattern bonus | 0-1 |
| `xgb_prob`, `lstm_prob`, `cnn_prob` | Individual model probs | 0-1 (>0.5 = UP) |
| `pattern_detected` | Chart pattern found | bull_flag, bear_flag, etc. |
| `pattern_confidence` | Pattern strength | 0-1 |
| `entry_price` | Current price | $ |
| `stop_loss` | Risk limit | $ |
| `take_profit` | Profit target | $ |

---

## Multi-Timeframe Strategy

Get signals from all timeframes, then aggregate:

```python
# Fetch all timeframes
dfs = {
    '1D': trader.fetch_ohlcv('XAUUSD', '1D', bars=100),
    '4H': trader.fetch_ohlcv('XAUUSD', '4H', bars=100),
    '1H': trader.fetch_ohlcv('XAUUSD', '1H', bars=100),
    '15min': trader.fetch_ohlcv('XAUUSD', '15min', bars=100),
}

# Generate signals
signals = engine.generate_signals_multiframe(dfs)

# Aggregate
consensus = engine.aggregate_signals(signals)
print(f"{consensus['direction']} ({consensus['consensus']})")
```

Aggregation weighting:
- 1D: 40% weight (most important)
- 4H: 30% weight
- 1H: 20% weight
- 15min: 10% weight

---

## System Components

### Feature Engineering (47 features)
```python
from training.feature_engineering_v2 import build_features_v2

df_features = build_features_v2(df, timeframe='1H')
# Returns: DataFrame with 47 technical indicators
```

### Pattern Detection (Rule-Based)
```python
from src.pattern_detector import PatternDetector

detector = PatternDetector()
patterns = detector.detect_all(df, lookback=50)
for p in patterns:
    print(f"{p.pattern}: {p.confidence:.0%} confidence, {p.direction} move")
```

### Model Predictions (Individual)
```python
from src.model_loader import ModelBundle

models = ModelBundle(tf='1H', models_dir=Path('training/models'))
pred = models.predict(df)
# {'signal': 'UP'/'DOWN'/'FLAT', 'confidence': 0-1, ...}
```

---

## Troubleshooting

### MT5 Connection Failed
```python
# Check credentials
trader = MT5Trader(login=5050913403, password="Ahmed@477447", server="MetaQuotes-Demo")
if trader.connect():
    print("OK")
else:
    print(f"Error: {trader.connect()}")
```

### Models Not Found
```bash
ls training/models/
# If empty, run: python training/train_xgboost_v2.py
```

### Features Have NaN
```python
# Features are forward-filled automatically
# If still issues, use fill approach:
df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
```

### Predictions Constant (All FLAT)
```python
# Check data quality
print(len(df))  # Need at least 100 bars
print(df['close'].describe())  # Check price range
```

---

## Performance Expectations

**Accuracy Goals:**
- XGBoost: 78-80% (baseline)
- LSTM: 80-82% (sequences)
- 1D-CNN: 79-81% (patterns)
- **Ensemble: 82-85%** (combined)

**Confidence:**
- >70% = Strong signal
- 50-70% = Mixed
- <50% = FLAT (no trade)

---

## Architecture

```
Real-time OHLC Data
       ↓
47 Technical Features
       ↓
┌─────┬──────┬─────┐
│ XGB │ LSTM │ CNN │
└──┬──┴──┬───┴──┬──┘
   │     │      │
   └─────┴──────┘
         ↓
   Ensemble Meta-Learner
         ↓
   Signal Engine
   (+ Pattern Bonus)
         ↓
   BUY / SELL / FLAT
   (with confidence %)
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/signal_engine_v2.py` | Main signal generation |
| `src/model_loader.py` | Load & use trained models |
| `src/pattern_detector.py` | Pattern detection |
| `training/feature_engineering_v2.py` | Build 47 features |
| `examples/demo_signal_generation.py` | Complete workflow demo |

---

**Ready to trade?** Run `python examples/demo_signal_generation.py` 🚀
