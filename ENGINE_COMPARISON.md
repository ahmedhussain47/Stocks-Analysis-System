# Signal Engine Comparison: v1 vs v2

## Overview

Two separate signal engines run in **parallel isolation** for direct comparison:

| **Component** | **Engine v1 (Original)** | **Engine v2 (Baseline)** |
|---|---|---|
| **Models** | LSTM (77.25%) + XGBoost (73.65%) | AutoETS (52.5%) + Chronos (51.3%) |
| **Notebook** | `notebooks/05_signal_engine.ipynb` | `notebooks/06_signal_engine_v2_auto_chronos.ipynb` |
| **Output CSV** | `results/signals_log.csv` | `results/signals_log_v2.csv` |
| **Scan Interval** | 500s | 510s (offset to avoid collision) |
| **Risk Management** | ATR_SL_MULT = 1.1 (10% wider buffer) | ATR_SL_MULT = 1.1 (same) |
| **Tier 2 Features** | All (pullback, momentum, risk adj, session, volume/ATR, retest, adaptive) | All (identical) |

## Model Differences

### Engine v1: High-Accuracy Ensemble
- **LSTM**: Deep learning on 45 engineered features → 77.25% accuracy
- **XGBoost**: Gradient boosting on 50 features (OHLCV + indicators) → 73.65% accuracy
- **Weights**: LSTM 60% (trending: 65%), XGBoost 40% (trending: 35%)
- **Use case**: Better for trending markets with strong directional signals

### Engine v2: Lightweight Baseline
- **AutoETS**: Exponential smoothing on log returns → 52.5% accuracy
- **Chronos**: Zero-shot transformer on close prices → 51.3% accuracy
- **Weights**: AutoETS 50%, Chronos 50% (equal)
- **Use case**: Faster inference, lower compute, good baseline for comparison

## Running Both Engines

### Option A: Sequential (One at a time)
```bash
# Run v1 only
jupyter notebook notebooks/05_signal_engine.ipynb
# Kernel → Restart & Run All

# Run v2 only
jupyter notebook notebooks/06_signal_engine_v2_auto_chronos.ipynb
# Kernel → Restart & Run All
```

### Option B: Parallel (Both at once)
Open two Jupyter tabs:
- Tab 1: `05_signal_engine.ipynb` → Run All
- Tab 2: `06_signal_engine_v2_auto_chronos.ipynb` → Run All

Both will scan every 500s/510s and log signals independently.

## Comparing Results

After collecting ~10-20 signal outcomes in each engine:

```python
# In Python or new Jupyter cell
from src.engine_comparison import compare_engines
compare_engines()

# Or from terminal
python -c "from src.engine_comparison import compare_engines; compare_engines()"
```

**Output:**
```
================================================================================
SIGNAL ENGINE COMPARISON: v1 (LSTM+XGBoost) vs v2 (AutoETS+Chronos)
================================================================================

[v1] LSTM + XGBoost (High-Accuracy Ensemble)
--------------------------------------------------------------------------------
  Total signals      : 42
  Resolved trades    : 12
  Win rate           : 58.3%
  Avg PnL/trade      : +0.0142%
  Total PnL          : +0.1704%
  Sharpe ratio       : 1.245
  Max drawdown       : 0.0523
  Profit factor      : 2.34x
  Avg confidence     : 78.2%

[v2] AutoETS + Chronos (Simpler Baseline)
--------------------------------------------------------------------------------
  Total signals      : 38
  Resolved trades    : 10
  Win rate           : 40.0%
  Avg PnL/trade      : +0.0098%
  Total PnL          : +0.0980%
  Sharpe ratio       : 0.821
  Max drawdown       : 0.0834
  Profit factor      : 1.45x
  Avg confidence     : 72.1%

[COMPARISON]
--------------------------------------------------------------------------------
  Win rate delta     : -18.3%  (v1 better)
  Total PnL delta    : -0.0724%  (v1 better)
  Sharpe delta       : -0.424  (v1 better)

  Verdict:
    -> v1 (LSTM+XGBoost) performs better (3 of 3 metrics)
```

## Interpretation

### If v1 is better (expected):
- ✓ LSTM+XGBoost justify their higher accuracy
- ✓ Complex models provide real edge in live trading
- ✓ Keep using Engine v1, continue optimizing Tier 2 filters

### If v2 is surprisingly good:
- ⚠ AutoETS+Chronos might be overfitting to historical data
- ⚠ Simpler models with lower compute cost may be preferred
- → Run longer (30+ trades) to confirm, as AutoETS/Chronos have higher variance

### If they're statistically tied (rare):
- ⚠ Model accuracy doesn't translate to live edge
- → Focus on Tier 2 filters (pullback detection, session bias, risk adjustment)
- → Possibly switch to technical analysis only

## File Separation

```
results/
├── signals_log.csv           [v1 - LSTM+XGBoost]
├── signals_log_v2.csv        [v2 - AutoETS+Chronos]
├── ensemble_state.json       [v1 ensemble weights]
└── advanced_models/
    ├── best_model_lstm.h5    [v1 LSTM]
    ├── best_model_xgb.json   [v1 XGBoost]
    └── metadata.json

notebooks/
├── 05_signal_engine.ipynb               [v1 engine]
└── 06_signal_engine_v2_auto_chronos.ipynb [v2 engine]

src/
├── feature_engineering.py   [Tier 2: shared indicators, pullback, etc]
├── tier2_enhancements.py    [Tier 2: momentum, risk, session filters]
├── engine_comparison.py      [NEW: side-by-side metrics]
└── ...
```

## Configuration

Both engines share identical Tier 2 settings (via `src/`):
- Confidence threshold: 62% (RANGING aware)
- ATR_SL_MULT: 1.1x (10% wider buffer)
- Pullback: 30-80% retracement detection
- Momentum: MACD + RSI scoring
- Session: TOKYO/LONDON/NY/SYDNEY filtering
- News blackout: ±10min around high-impact events
- Dynamic risk: Position sizing by 20-trade win rate + drawdown

## Next Steps

1. **Day 1**: Start both engines with the same XAUUSD 1H config
2. **Collect**: ~10-20 resolved trades per engine (~5-10 hours of scanning)
3. **Compare**: Run `compare_engines()` to see which performs better
4. **Decide**: Based on results, keep v1 or investigate why v2 wins (unlikely)
5. **Optimize**: Focus on Tier 2 filters (shared across both engines)

## Important Notes

- **Different input scales**: v1 uses engineered features (scaled), v2 uses raw returns/prices
- **Model latency**: v1 is fast (LSTM/XGBoost in-process), v2 is slower (StatsForecast + Chronos overhead)
- **Ensemble weights**: v1 adapts to regime (TRENDING: 65/35, RANGING: 55/45), v2 is static (50/50)
- **Confidence calibration**: Both engines use the same calibrator (fit on historical data), so v2 confidence may be miscalibrated for weaker models

## Troubleshooting

**"signals_log_v2.csv not found"**
- Run v2 notebook cell_live for at least one scan cycle
- Check that `SCAN_INTERVAL_S` is set correctly (510s for v2)

**"Chronos load failed"**
- First load is slow (~30s), subsequent loads are instant (cached)
- Check that `src/new_models.py` has `ChronosZeroShotForecaster`

**"StatsForecast error: incomplete data"**
- AutoETS needs at least 10 log returns
- If market just opened, wait a few bars

**CSV schema mismatch**
- Delete old `signals_log_v2.csv` and restart v2 engine
- New CSV will be created with correct schema on first signal
