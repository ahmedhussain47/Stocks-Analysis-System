"""Quick diagnostic to check data and features."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from training.feature_engineering_v2 import build_features_v2, get_feature_names, atr as calc_atr

DATA_DIR = Path(__file__).parent / 'data' / 'raw' / 'mt5'
ASSET = 'XAUUSD'
TF = '1D'

# Load data
csv_path = DATA_DIR / f"{ASSET}_{TF}_mt5.csv"
df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"First 5 rows:")
print(df.head())
print()

# Build features
print(f"Building features...")
X = build_features_v2(df, TF)

print(f"Features shape: {X.shape}")
print(f"Features columns: {len(X.columns)}")
print(f"Feature NaN counts:")
nan_counts = X.isna().sum()
print(nan_counts[nan_counts > 0].sort_values(ascending=False))
print()

print(f"Features with 100% NaN: {(X.isna().sum() == len(X)).sum()}")
print(f"Rows with any NaN: {X.isna().any(axis=1).sum()}")
print(f"Rows with NO NaN: {(~X.isna().any(axis=1)).sum()}")
print()

# Create labels
print(f"Creating labels...")
h = 3
close = df['close'].values

atr_vals = calc_atr(df, 14).values

future_close = np.full_like(close, np.nan, dtype=float)
future_close[:len(close)-h] = close[h:]
future_return = (future_close - close) / close

threshold = 0.5 * (atr_vals / close)

print(f"ATR values NaN count: {np.isnan(atr_vals).sum()}")
print(f"Future return NaN count: {np.isnan(future_return).sum()}")
print(f"Threshold NaN count: {np.isnan(threshold).sum()}")
print()

labels = pd.Series(0, index=df.index)
valid_mask = ~np.isnan(future_return) & ~np.isnan(threshold)

labels[valid_mask & (future_return > threshold)] = 1
labels[valid_mask & (future_return < -threshold)] = -1
labels[~valid_mask] = np.nan

print(f"Label distribution (including NaN):")
print(f"  UP (1): {(labels == 1).sum()}")
print(f"  DOWN (-1): {(labels == -1).sum()}")
print(f"  FLAT (0): {(labels == 0).sum()}")
print(f"  NaN: {labels.isna().sum()}")
print()

# Check combined
valid_idx = ~(X.isna().any(axis=1) | labels.isna())
print(f"Combined valid rows: {valid_idx.sum()} / {len(df)}")
