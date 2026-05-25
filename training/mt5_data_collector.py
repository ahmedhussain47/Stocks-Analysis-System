"""
MT5 Data Collector — Fetch 7 years of XAUUSD across multiple timeframes.

Saves to: training/data/raw/mt5/XAUUSD_{tf}_mt5.csv
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Set UTF-8 encoding for stdout on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.mt5_trader import MT5Trader

# ── Configuration ──────────────────────────────────────────────────────────
MT5_LOGIN = 5050913403
MT5_PASS = "Ahmed@477447"
MT5_SERVER = "MetaQuotes-Demo"

ASSET = 'XAUUSD'
LOOKBACK_YEARS = 7

TIMEFRAMES = ['1D', '4H', '1H', '15min']

DATA_DIR = Path(__file__).parent / 'data' / 'raw' / 'mt5'
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {DATA_DIR}")

# ── Connect to MT5 ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Connecting to MT5...")
print("="*70)

trader = MT5Trader(
    login=MT5_LOGIN,
    password=MT5_PASS,
    server=MT5_SERVER,
    demo_mode=True
)

if not trader.connect():
    print("[FAIL] MT5 connection failed!")
    sys.exit(1)

summary = trader.get_account_summary()
print("[OK] MT5 Connected")
print(f"  Account equity: ${summary.get('equity', 0):,.2f}")

# ── Calculate date range ───────────────────────────────────────────────────
print("\n" + "="*70)
print("Calculating date range...")
print("="*70)

date_to = datetime.now(timezone.utc)
date_from = date_to - timedelta(days=365 * LOOKBACK_YEARS)

print(f"Target: {LOOKBACK_YEARS} years back")
print(f"Requested from: {date_from.strftime('%Y-%m-%d')}")
print(f"Requested to:   {date_to.strftime('%Y-%m-%d')}")
print(f"Note: MT5 will return whatever historical data is available\n")

# ── Fetch data for each timeframe ──────────────────────────────────────────
print("="*70)
print(f"Fetching {ASSET} across timeframes...")
print("="*70)

results = {}

for tf in TIMEFRAMES:
    print(f"\n[{tf}] Fetching...", end=" ", flush=True)

    try:
        # Use fetch_ohlcv with large bar count instead of copy_rates_range
        # This fetches the most recent N bars from current time backwards
        bar_counts = {
            '1D': 2000,      # ~8 years of daily bars
            '4H': 8000,      # ~13 years of 4H bars
            '1H': 30000,     # ~3 years of hourly bars
            '15min': 50000,  # ~1.4 years of 15min bars (MT5 limit)
        }

        bars = bar_counts.get(tf, 1000)
        df = trader.fetch_ohlcv(
            symbol=ASSET,
            timeframe=tf,
            bars=bars
        )

        if df.empty:
            print("[FAIL] No data!")
            print("     (MT5 may not have historical data for this timeframe)")
            results[tf] = None
            continue
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        results[tf] = None
        continue

    print(f"[OK] Got {len(df):,} bars")

    # Calculate expected bars
    if tf == '1D':
        expected = LOOKBACK_YEARS * 252  # ~252 trading days per year
    elif tf == '4H':
        expected = LOOKBACK_YEARS * 252 * 6  # ~6 4H bars per day
    elif tf == '1H':
        expected = LOOKBACK_YEARS * 252 * 24  # ~24 1H bars per day
    elif tf == '15min':
        expected = LOOKBACK_YEARS * 252 * 96  # ~96 15min bars per day
    else:
        expected = None

    if expected:
        coverage = (len(df) / expected) * 100
        print(f"       Coverage: {coverage:.1f}% of expected {expected:,} bars")
        if coverage < 70:
            print(f"       [WARN] Low coverage!")

    # Save to CSV
    csv_path = DATA_DIR / f"{ASSET}_{tf}_mt5.csv"
    df.to_csv(csv_path)
    print(f"       Saved: {csv_path.name}")

    results[tf] = {
        'path': csv_path,
        'rows': len(df),
        'start': df.index[0],
        'end': df.index[-1],
    }

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for tf in TIMEFRAMES:
    if results[tf] is None:
        print(f"{tf:6s}: [FAIL]")
    else:
        r = results[tf]
        print(f"{tf:6s}: {r['rows']:>7,} bars | {r['start'].strftime('%Y-%m-%d')} to {r['end'].strftime('%Y-%m-%d')}")

print(f"\nAll data saved to: {DATA_DIR}")

trader.disconnect()
print("\n[OK] Done!")
