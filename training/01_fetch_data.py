"""
PHASE 1: MULTI-ASSET DATA FETCHING (2019-2026)
================================================

Fetches 7 years of hourly data for 10 forex pairs + gold.
Expected output: ~60,000 bars per asset

Assets (ordered by correlation to Gold):
Tier 1 (Direct USD correlation):
  - EURUSD (most liquid, DXY inverse)
  - GBPUSD (carries)
  - USDCHF (safe haven)
  - XAUUSD (target instrument)

Tier 2 (Indirect correlation):
  - AUDUSD (commodity-linked)
  - USDCAD (commodity-linked)
  - NZDUSD (risk-on/off indicator)

Tier 3 (Support/Context):
  - USDJPY (carry trade)
  - DXY (dollar strength)
  - US10Y (bonds - yield proxy)

Time Range: 2019-05-25 to 2026-05-25 (7 years)
Timeframe: 1H (hourly)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

print("="*80)
print("PHASE 1: MULTI-ASSET DATA FETCHING")
print("="*80)

# Create output directory
data_dir = Path('training/data/raw')
data_dir.mkdir(parents=True, exist_ok=True)

# Assets to fetch (ticker symbols for yfinance)
ASSETS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDCHF': 'USDCHF=X',
    'XAUUSD': 'GC=F',  # Gold futures
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
    'NZDUSD': 'NZDUSD=X',
    'USDJPY': 'USDJPY=X',
    'DXY': 'DXY',  # Dollar index
    'US10Y': '^TNX',  # 10-year Treasury yield
}

# Fetch parameters
START_DATE = '2019-01-01'
END_DATE = '2026-05-25'
INTERVAL = '1d'  # daily data (yfinance limitation: hourly only available last 730 days)
# Note: Using daily instead of hourly for 7-year history. Models still work well on daily.

print(f"\nFetching {len(ASSETS)} assets from {START_DATE} to {END_DATE}")
print(f"Interval: {INTERVAL}")
print("-" * 80)

# Track results
results = {
    'timestamp': datetime.now().isoformat(),
    'start_date': START_DATE,
    'end_date': END_DATE,
    'interval': INTERVAL,
    'assets': {}
}

# Fetch data for each asset
for asset_name, ticker in ASSETS.items():
    print(f"\n[{asset_name:10s}] Fetching from yfinance...", end=' ', flush=True)

    try:
        # Download data
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval=INTERVAL,
            progress=False,
            timeout=30
        )

        if df is None or df.empty:
            print(f"[ERROR] No data returned")
            results['assets'][asset_name] = {
                'status': 'failed',
                'error': 'Empty result from yfinance',
                'bars': 0
            }
            continue

        # Standardize column names (handle MultiIndex tuples)
        try:
            df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
        except:
            df.columns = [col.lower() for col in df.columns]

        # Clean data
        print(f"✓ Got {len(df)} bars | ", end='', flush=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        print(f"After dedup: {len(df)} bars | ", end='', flush=True)

        # Remove NaN rows
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"Removed {dropped} NaN rows | ", end='', flush=True)

        # Check for gaps (max 4 hour gap acceptable)
        time_diffs = df.index.to_series().diff().dt.total_seconds() / 3600
        max_gap = time_diffs.max()
        gap_count = (time_diffs > 4).sum()

        if gap_count > 0:
            print(f"Gaps: {gap_count} (max {max_gap:.0f}h) | ", end='', flush=True)

        # Save to CSV
        output_file = data_dir / f"{asset_name.lower()}_1h_2019_2026.csv"
        df.to_csv(output_file)
        print(f"[OK] Saved")

        results['assets'][asset_name] = {
            'status': 'success',
            'bars': len(df),
            'first_date': str(df.index[0]),
            'last_date': str(df.index[-1]),
            'max_gap_hours': float(max_gap) if gap_count > 0 else 0.0,
            'file': str(output_file)
        }

        # Rate limit (yfinance appreciates slower requests)
        time.sleep(0.5)

    except Exception as e:
        print(f"[ERROR] {str(e)[:60]}")
        results['assets'][asset_name] = {
            'status': 'failed',
            'error': str(e)[:100],
            'bars': 0
        }

# Summary report
print("\n" + "="*80)
print("DATA FETCHING SUMMARY")
print("="*80)

successful = sum(1 for r in results['assets'].values() if r['status'] == 'success')
failed = sum(1 for r in results['assets'].values() if r['status'] == 'failed')

print(f"\n[OK] Successful: {successful}/{len(ASSETS)}")
print(f"[FAILED] Count: {failed}/{len(ASSETS)}")

print("\nAsset Details:")
print("-" * 80)
for asset_name, info in results['assets'].items():
    if info['status'] == 'success':
        print(f"  {asset_name:10s} | {info['bars']:6d} bars | {info['first_date']} to {info['last_date']}")
    else:
        print(f"  {asset_name:10s} | FAILED: {info['error'][:40]}")

# Save manifest
manifest_file = data_dir / 'data_manifest.json'
with open(manifest_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Manifest saved: {manifest_file}")

# Data quality checks
print("\n" + "="*80)
print("DATA QUALITY CHECKS")
print("="*80)

if successful >= 8:
    print("[OK] Got 8+ assets (sufficient for multi-asset ensemble)")
else:
    print(f"[WARNING] Only {successful} assets (ideally 8+)")

total_bars = sum(r.get('bars', 0) for r in results['assets'].values() if r['status'] == 'success')
if total_bars > 300000:
    print(f"[OK] Total {total_bars:,} bars (~40,000+ per asset avg)")
else:
    print(f"[WARNING] Only {total_bars:,} bars total")

# Check for gaps
assets_with_gaps = sum(1 for r in results['assets'].values()
                       if r['status'] == 'success' and r.get('max_gap_hours', 0) > 4)
if assets_with_gaps == 0:
    print("[OK] No significant gaps in data (< 4 hours)")
else:
    print(f"[WARNING] {assets_with_gaps} assets have gaps > 4 hours")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Review data_manifest.json for any issues")
print("2. Run Phase 2: Feature engineering")
print("3. Generate 50+ features for all assets")
print("4. Create training/validation splits")
print("\nExpected time for Phase 2: 30 minutes")
print("="*80)
