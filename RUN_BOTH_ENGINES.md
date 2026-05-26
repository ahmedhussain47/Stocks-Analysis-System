# Run Both Engines in Parallel

## Setup: Two Jupyter Tabs

**Terminal/PowerShell:**
```bash
cd path/to/new_equity_forecasting_project
jupyter notebook
```

## Tab 1: Engine v1 (LSTM+XGBoost)

In Jupyter, open: `notebooks/05_signal_engine.ipynb`

**Do NOT edit config** — keep:
- ASSET = 'XAUUSD'
- ENTRY_TF = '1H'
- SCAN_INTERVAL_S = 500

Then: **Kernel → Restart & Run All**

Wait for output:
```
[OK] Fetching XAUUSD data ...
[OK] Loading LSTM model ...
[OK] Loading XGBoost model ...
[OK] Live scanner initialized
*** TIER 2 MODE: AutoETS + Chronos ***
Live scanner ON: XAUUSD 1H, every 500s ...
```

## Tab 2: Engine v2 (AutoETS+Chronos)

In same Jupyter window, open new tab: `notebooks/06_signal_engine_v2_auto_chronos.ipynb`

**Do NOT edit config** — keep:
- ASSET = 'XAUUSD'
- ENTRY_TF = '1H'
- SCAN_INTERVAL_S = 510  (offset from v1)

Then: **Kernel → Restart & Run All**

Wait for output:
```
[OK] Fetching XAUUSD data ...
[OK] Loading Chronos-Bolt-Tiny ...
[OK] Live scanner initialized
*** ENGINE v2 (AutoETS+Chronos)***
Live scanner ON: XAUUSD 1H, every 510s ...
```

## Live Monitoring

Both tabs will now:
- Scan XAUUSD 1H every 500-510s (offset prevents collision)
- Log signals independently:
  - v1 → `results/signals_log.csv`
  - v2 → `results/signals_log_v2.csv`
- Run Tier 2 filters identically (same entry/TP/SL logic, different models)

**You'll see output like:**
```
[06:45:23 UTC][1] *** SELL XAUUSD @ 4522.800 | Conf: 86% | TP: 4510.2 | SL: 4534.6 [v1]
[06:45:33 UTC][1] *** BUY XAUUSD @ 4521.500 | Conf: 79% | TP: 4535.8 | SL: 4513.1 [v2]
[06:46:17 UTC][2] *** SELL XAUUSD @ 4520.100 | Conf: 92% | TP: 4508.5 | SL: 4531.9 [v1]
```

Each signal will have a `[v1]` or `[v2]` tag (added automatically by each engine).

## Stop Both Engines

**In each Jupyter tab:** `Kernel → Interrupt` (or Ctrl+C)

## After 10-20 Trades Per Engine (~5-10 hours)

**Compare side-by-side:**
```python
from src.engine_comparison import compare_engines
compare_engines()
```

This will show:
- Win rate, total PnL, Sharpe ratio, drawdown for each
- Which engine performed better
- Statistical verdict (v1 better, v2 better, or tied)

---

## Tips

1. **Start together**: Both Tab 1 and Tab 2 → Run All at same time for clean comparison
2. **Don't interrupt manually**: Let them run for full 24-48 hours (more samples = more accurate comparison)
3. **Monitor signals**: Check that both are actually generating signals (not stuck)
4. **If one crashes**: Restart that tab only; other engine keeps running independently
5. **CSV errors**: If either CSV fails, delete it and restart that engine (auto-recreates with correct schema)

---

## Expected Pattern

**First hour:**
- v1: 3-5 signals (LSTM+XGBoost are selective, high confidence)
- v2: 2-4 signals (AutoETS+Chronos less selective, lower confidence)

**After 6 hours:**
- v1: 15-20 signals with ~2-4 outcomes
- v2: 12-18 signals with ~1-3 outcomes

**After 24 hours:**
- v1: 30-50 signals with ~10-15 outcomes
- v2: 25-40 signals with ~8-12 outcomes

At 10+ outcomes each, you can reliably compare (Sharpe ratio, win rate, drawdown).

