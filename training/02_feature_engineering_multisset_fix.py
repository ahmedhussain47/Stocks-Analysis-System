"""
PHASE 2B: MULTI-ASSET FEATURE ENGINEERING (FIXED)
==================================================

Train on XAUUSD but enrich with cross-asset correlation features.
This uses data from 9 other assets to understand market context.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 2B: MULTI-ASSET FEATURE ENGINEERING (FIXED)")
print("="*80)

data_dir = Path('training/data/raw')
processed_dir = Path('training/data/processed')

ASSETS = ['EURUSD', 'GBPUSD', 'USDCHF', 'XAUUSD', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDJPY', 'DXY', 'US10Y']
TARGET_ASSET = 'XAUUSD'

print(f"\nLoading assets...")
data = {}
for asset in ASSETS:
    file_path = data_dir / f"{asset.lower()}_1h_2019_2026.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data[asset] = df
        print(f"  [OK] {asset:10s}: {len(df):6d} bars | {df.index[0].date()} to {df.index[-1].date()}")

# Get XAUUSD as base
xauusd = data[TARGET_ASSET].copy()
print(f"\n[TARGET] XAUUSD: {len(xauusd)} bars ({xauusd.index[0]} to {xauusd.index[-1]})")

# ─────────────────────────────────────────────────────────────────────────
# ADD XAUUSD TECHNICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────

def add_technical_features(df):
    """Add technical indicators"""
    close = df['close']

    # EMAs
    df['ema_20'] = close.ewm(span=20).mean()
    df['ema_50'] = close.ewm(span=50).mean()
    df['ema_200'] = close.ewm(span=200).mean()
    df['dema_20'] = 2 * close.ewm(span=20).mean() - close.ewm(span=20).mean().ewm(span=20).mean()

    # RSI
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_7'] = 100 - (100 / (1 + (delta.clip(lower=0)).rolling(7).mean() / (-delta.clip(upper=0)).rolling(7).mean().replace(0, np.nan)))

    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # ATR
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - close.shift()).abs()
    tr3 = (df['low'] - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(alpha=1/14).mean()
    df['atr_7'] = tr.ewm(alpha=1/7).mean()

    # ADX
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14).fillna(0)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14).fillna(0)
    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    adx = di_diff.rolling(14).mean() / di_sum.replace(0, np.nan) * 100
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['adx_14'] = adx

    # Bollinger Bands
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['bb_upper'] = sma + 2 * std
    df['bb_mid'] = sma
    df['bb_lower'] = sma - 2 * std
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_pct'] = (close - df['bb_lower']) / df['bb_width'].replace(0, np.nan)

    # Ichimoku
    h_tenkan = df['high'].rolling(9).max()
    l_tenkan = df['low'].rolling(9).min()
    h_kijun = df['high'].rolling(26).max()
    l_kijun = df['low'].rolling(26).min()
    df['ichimoku_tenkan'] = (h_tenkan + l_tenkan) / 2
    df['ichimoku_kijun'] = (h_kijun + l_kijun) / 2

    return df

print("\nAdding XAUUSD technical features...")
xauusd = add_technical_features(xauusd)

# Price action
xauusd['dist_ema20'] = xauusd['close'] - xauusd['ema_20']
xauusd['dist_ema50'] = xauusd['close'] - xauusd['ema_50']
xauusd['dist_ema200'] = xauusd['close'] - xauusd['ema_200']
xauusd['price_pct_high_20'] = (xauusd['close'] - xauusd['low'].rolling(20).min()) / (xauusd['high'].rolling(20).max() - xauusd['low'].rolling(20).min()).replace(0, np.nan)

# Returns
xauusd['ret_1'] = xauusd['close'].pct_change(1)
xauusd['ret_5'] = xauusd['close'].pct_change(5)
xauusd['ret_20'] = xauusd['close'].pct_change(20)
xauusd['log_ret_lag_1'] = np.log(xauusd['close'] / xauusd['close'].shift(1))
xauusd['log_ret_lag_2'] = np.log(xauusd['close'] / xauusd['close'].shift(2))
xauusd['log_ret_lag_3'] = np.log(xauusd['close'] / xauusd['close'].shift(3))
xauusd['log_ret_lag_5'] = np.log(xauusd['close'] / xauusd['close'].shift(5))

# Volatility
xauusd['volatility_5'] = xauusd['close'].pct_change().rolling(5).std()
xauusd['volatility_10'] = xauusd['close'].pct_change().rolling(10).std()
xauusd['volatility_20'] = xauusd['close'].pct_change().rolling(20).std()

# Volume
xauusd['volume_ma'] = xauusd['volume'].rolling(20).mean()
xauusd['volume_ratio'] = xauusd['volume'] / (xauusd['volume'].rolling(20).mean() + 1)

print(f"  [OK] {len(xauusd.columns)} columns generated")

# ─────────────────────────────────────────────────────────────────────────
# ADD CROSS-ASSET FEATURES
# ─────────────────────────────────────────────────────────────────────────

print("\nAdding cross-asset features...")

# Resample other assets to match XAUUSD dates (daily -> align to XAUUSD dates)
for asset_name in ['EURUSD', 'GBPUSD', 'USDJPY', 'DXY', 'US10Y']:
    if asset_name in data:
        df = data[asset_name].copy()

        # Interpolate/align to XAUUSD dates
        df = df.reindex(xauusd.index, method='ffill')

        # Calculate correlation (rolling 20-period)
        corr = xauusd['close'].rolling(20).corr(df['close'])
        xauusd[f'corr_{asset_name}'] = corr

        # Momentum relative to XAUUSD
        xauusd[f'ret_rel_{asset_name}'] = df['close'].pct_change() - xauusd['close'].pct_change()

        print(f"  [OK] Added {asset_name} correlation + relative momentum")

# ─────────────────────────────────────────────────────────────────────────
# FINALIZE & SAVE
# ─────────────────────────────────────────────────────────────────────────

# Drop NaN (from warmup period)
initial = len(xauusd)
xauusd = xauusd.dropna()
dropped = initial - len(xauusd)

print(f"\nDropped {dropped} NaN rows from warmup | {len(xauusd)} clean samples")

# Get feature columns (exclude OHLCV)
exclude = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
feature_cols = [col for col in xauusd.columns if col not in exclude]

print(f"[OK] Final feature count: {len(feature_cols)}")

# Split
train_date = pd.Timestamp('2025-01-01')
xauusd_train = xauusd[xauusd.index < train_date].copy()
xauusd_test = xauusd[xauusd.index >= train_date].copy()

print(f"[OK] Train: {len(xauusd_train)} | Test: {len(xauusd_test)}")

# Save
xauusd.to_csv(processed_dir / 'features_full_multisset.csv')
xauusd_train.to_csv(processed_dir / 'features_train_multisset.csv')
xauusd_test.to_csv(processed_dir / 'features_test_multisset.csv')

# Metadata
meta = {
    'timestamp': datetime.now().isoformat(),
    'target_asset': TARGET_ASSET,
    'enriched_with_assets': ['EURUSD', 'GBPUSD', 'USDJPY', 'DXY', 'US10Y'],
    'features': feature_cols,
    'feature_count': len(feature_cols),
    'total_bars': len(xauusd),
    'train_bars': len(xauusd_train),
    'test_bars': len(xauusd_test),
    'period': f'{xauusd.index[0]} to {xauusd.index[-1]}'
}

with open(processed_dir / 'feature_metadata_multisset.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n[OK] Saved multi-asset enriched XAUUSD dataset")
print(f"     features_full_multisset.csv")
print(f"     features_train_multisset.csv")
print(f"     features_test_multisset.csv")
print(f"     feature_metadata_multisset.json")
print("\n" + "="*80)
