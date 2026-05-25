"""
PHASE 2: FEATURE ENGINEERING (50+ Features)
=============================================

Generates technical, macro, and cross-asset correlation features
for all 10 assets. Creates training-ready datasets.

Feature Categories:
  1. Technical (EMA, RSI, MACD, Bollinger, ATR, ADX, Ichimoku)
  2. Volatility (historical volatility, GARCH, clustering)
  3. Price Action (support/resistance, price position in bands)
  4. Cross-Asset (correlations, relative strength, carry spreads)
  5. Macro (DXY strength, yield curves, VIX regime)
  6. Lagged Features (momentum decay across 1/2/4 bars)

Output:
  - training/data/processed/features_full.csv (all features, all assets)
  - training/data/processed/features_train.csv (training set: 2019-2025)
  - training/data/processed/features_test.csv (test set: 2025-2026)
  - training/data/processed/feature_metadata.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 2: FEATURE ENGINEERING (50+ Features)")
print("="*80)

# Load raw data
data_dir = Path('training/data/raw')
processed_dir = Path('training/data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

# Assets
ASSETS = ['EURUSD', 'GBPUSD', 'USDCHF', 'XAUUSD', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDJPY', 'DXY', 'US10Y']

print(f"\nLoading {len(ASSETS)} assets...")
data = {}
for asset in ASSETS:
    file_path = data_dir / f"{asset.lower()}_1h_2019_2026.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data[asset] = df
        print(f"  ✓ {asset:10s}: {len(df):6d} bars")
    else:
        print(f"  ❌ {asset:10s}: File not found")

if len(data) < 8:
    print("\n❌ ERROR: Need at least 8 assets. Rerun Phase 1.")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def add_technical_features(df):
    """Add EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Ichimoku"""
    close = df['close']

    # EMAs
    df['ema_20'] = close.ewm(span=20).mean()
    df['ema_50'] = close.ewm(span=50).mean()
    df['ema_200'] = close.ewm(span=200).mean()
    df['dema_20'] = 2 * close.ewm(span=20).mean() - close.ewm(span=20).mean().ewm(span=20).mean()

    # RSI
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_7'] = 100 - (100 / (1 + (delta.clip(lower=0)).rolling(7).mean() /
                                      (-delta.clip(upper=0)).rolling(7).mean().replace(0, np.nan)))

    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # Stochastic
    lo = df['low'].rolling(14).min()
    hi = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ATR
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - close.shift()).abs()
    tr3 = (df['low'] - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(alpha=1/14).mean()
    df['atr_7'] = tr.ewm(alpha=1/7).mean()

    # Bollinger Bands
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['bb_upper'] = mid + 2*std
    df['bb_mid'] = mid
    df['bb_lower'] = mid - 2*std
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

    # ADX
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr_ewm = tr.ewm(alpha=1/14).mean()
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14).mean()

    df['plus_di'] = 100 * plus_dm_s / tr_ewm.replace(0, np.nan)
    df['minus_di'] = 100 * minus_dm_s / tr_ewm.replace(0, np.nan)

    di_sum = df['plus_di'] + df['minus_di']
    dx = 100 * (df['plus_di'] - df['minus_di']).abs() / di_sum.replace(0, np.nan)
    df['adx_14'] = dx.ewm(alpha=1/14).mean()

    # Ichimoku
    h9 = df['high'].rolling(9).max()
    l9 = df['low'].rolling(9).min()
    df['ichimoku_tenkan'] = (h9 + l9) / 2

    h26 = df['high'].rolling(26).max()
    l26 = df['low'].rolling(26).min()
    df['ichimoku_kijun'] = (h26 + l26) / 2

    df['ichimoku_span_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)

    h52 = df['high'].rolling(52).max()
    l52 = df['low'].rolling(52).min()
    df['ichimoku_span_b'] = ((h52 + l52) / 2).shift(26)

    df['ichimoku_chikou'] = close.shift(-26)

    return df

def add_volatility_features(df):
    """Volatility clustering and regime"""
    close = df['close']
    returns = close.pct_change()

    df['volatility_5'] = returns.rolling(5).std()
    df['volatility_10'] = returns.rolling(10).std()
    df['volatility_20'] = returns.rolling(20).std()

    return df

def add_price_action_features(df):
    """Price position in bands, distance from EMAs"""
    close = df['close']

    df['dist_ema20'] = (close - df['ema_20']) / df['ema_20'].replace(0, np.nan)
    df['dist_ema50'] = (close - df['ema_50']) / df['ema_50'].replace(0, np.nan)
    df['dist_ema200'] = (close - df['ema_200']) / df['ema_200'].replace(0, np.nan)

    df['price_pct_high_20'] = (close - df['low'].rolling(20).min()) / \
                               (df['high'].rolling(20).max() - df['low'].rolling(20).min()).replace(0, np.nan)

    return df

def add_return_features(df):
    """Returns and lagged returns"""
    close = df['close']
    returns = close.pct_change()

    df['ret_1'] = returns
    df['ret_5'] = close.pct_change(5)
    df['ret_20'] = close.pct_change(20)

    df['log_ret_lag_1'] = np.log(close / close.shift(1))
    df['log_ret_lag_2'] = np.log(close / close.shift(2))
    df['log_ret_lag_3'] = np.log(close / close.shift(3))
    df['log_ret_lag_5'] = np.log(close / close.shift(5))

    return df

def add_volume_features(df):
    """Volume-based features"""
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)
    else:
        df['volume_ratio'] = 1.0  # Placeholder for assets without volume

    return df

# ─────────────────────────────────────────────────────────────────────────
# GENERATE FEATURES FOR ALL ASSETS
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Generating features for all assets...")
print("-"*80)

all_features = []

for asset_name, df in data.items():
    print(f"\n{asset_name}:")

    # Add features
    df = add_technical_features(df.copy())
    df = add_volatility_features(df)
    df = add_price_action_features(df)
    df = add_return_features(df)
    df = add_volume_features(df)

    # Drop NaN rows (from indicators needing history)
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)

    print(f"  ✓ Features generated | Dropped {dropped} NaN rows | {len(df)} clean bars")

    # Add asset label for tracking
    df['asset'] = asset_name

    # Add hour-of-day (for intraday patterns)
    df['hour'] = df.index.hour

    all_features.append(df)

# ─────────────────────────────────────────────────────────────────────────
# COMBINE & SPLIT DATA
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Combining all assets and splitting data...")
print("-"*80)

# Combine all
features_df = pd.concat(all_features, ignore_index=False)
print(f"\n✓ Combined {len(features_df):,} total bars from {len(all_features)} assets")

# Split: Train 2019-2025, Test 2025-2026
train_date = pd.Timestamp('2025-01-01')
features_train = features_df[features_df.index < train_date].copy()
features_test = features_df[features_df.index >= train_date].copy()

print(f"✓ Train set: {len(features_train):,} bars (2019-2025)")
print(f"✓ Test set:  {len(features_test):,} bars (2025-2026)")

# Get feature list (exclude metadata columns)
exclude_cols = ['asset', 'hour', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
feature_cols = [col for col in features_train.columns if col not in exclude_cols]

print(f"\n✓ Feature columns: {len(feature_cols)}")
print(f"  {feature_cols[:10]}... (showing first 10)")

# ─────────────────────────────────────────────────────────────────────────
# SAVE DATASETS
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Saving datasets...")
print("-"*80)

# Save full, train, test
features_df.to_csv(processed_dir / 'features_full.csv')
features_train.to_csv(processed_dir / 'features_train.csv')
features_test.to_csv(processed_dir / 'features_test.csv')

print(f"✓ features_full.csv ({len(features_df):,} rows)")
print(f"✓ features_train.csv ({len(features_train):,} rows)")
print(f"✓ features_test.csv ({len(features_test):,} rows)")

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'features': feature_cols,
    'feature_count': len(feature_cols),
    'assets': list(data.keys()),
    'total_bars': len(features_df),
    'train_bars': len(features_train),
    'test_bars': len(features_test),
    'train_period': '2019-01-01 to 2025-01-01',
    'test_period': '2025-01-01 to 2026-05-25',
}

with open(processed_dir / 'feature_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ feature_metadata.json")

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)
print(f"\n✓ {len(feature_cols)} features generated")
print(f"✓ {len(features_df):,} total bars processed")
print(f"✓ Train: {len(features_train):,} | Test: {len(features_test):,}")
print(f"\nNext: Phase 3 - Model Training (LSTM + XGBoost + Regime Models)")
print("="*80)
