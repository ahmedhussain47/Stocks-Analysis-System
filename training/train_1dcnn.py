"""
1D-CNN for OHLCV Patterns — Fast convolutional model for 15min & 1H timeframes.

Input: 30-bar windows with 5 channels (OHLCV).
Output: 3-class prediction (UP/DOWN/FLAT).
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for stdout on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json

from training.feature_engineering_v2 import atr as calc_atr

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / 'data' / 'raw' / 'mt5'
MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ASSET = 'XAUUSD'
TIMEFRAMES = ['15min', '1H']  # 1D-CNN for intraday TFs only

# CNN config
LOOKBACK = 30  # 30-bar windows
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15

# Label horizons
LABEL_HORIZONS = {
    '15min': 4,
    '1H': 4,
}

ATR_THRESHOLD = 0.5

# ── Load Data ──────────────────────────────────────────────────────────────
def load_data(tf: str) -> pd.DataFrame:
    """Load CSV data for a timeframe."""
    csv_path = DATA_DIR / f"{ASSET}_{tf}_mt5.csv"
    if not csv_path.exists():
        print(f"[FAIL] File not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
    return df

def create_labels(df: pd.DataFrame, tf: str, atr_pct: float = ATR_THRESHOLD) -> pd.Series:
    """Create 3-class labels: UP (1), DOWN (-1), FLAT (0)."""
    h = LABEL_HORIZONS[tf]
    close = df['close'].values

    atr_vals = calc_atr(df, 14).values

    future_close = np.full_like(close, np.nan, dtype=float)
    future_close[:len(close)-h] = close[h:]
    future_return = (future_close - close) / close

    threshold = atr_pct * (atr_vals / close)

    labels = pd.Series(0, index=df.index)
    valid_mask = ~np.isnan(future_return) & ~np.isnan(threshold)

    labels[valid_mask & (future_return > threshold)] = 1
    labels[valid_mask & (future_return < -threshold)] = -1
    labels[~valid_mask] = np.nan

    return labels

def create_ohlcv_sequences(df: pd.DataFrame, y: np.ndarray, lookback: int = 30):
    """Create (batch, lookback, 5) sequences from OHLCV data."""
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    data = df[ohlcv_cols].values

    # Normalize each column independently (0-1)
    scalers = {}
    for i, col in enumerate(ohlcv_cols):
        scaler = MinMaxScaler()
        data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
        scalers[col] = scaler

    X_seq, y_seq = [], []

    for i in range(len(data) - lookback):
        X_seq.append(data[i:i+lookback])
        y_seq.append(y[i+lookback])

    return np.array(X_seq), np.array(y_seq), scalers

# ── Model Architecture ─────────────────────────────────────────────────────
def build_1dcnn_model():
    """
    1D-CNN for OHLCV patterns.

    Input: (batch, lookback, 5)  [OHLCV]
    Conv1D → Conv1D → MaxPool → Conv1D → GlobalAvgPool → Dense → Output
    """
    inputs = layers.Input(shape=(LOOKBACK, 5))

    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.15)(x)

    outputs = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ── Train per timeframe ────────────────────────────────────────────────────
def train_1dcnn_tf(tf: str):
    """Train 1D-CNN for a single timeframe."""

    print(f"\n{'='*70}")
    print(f"1D-CNN Training: {ASSET} @ {tf}")
    print(f"{'='*70}")

    # Load data
    print(f"Loading data...", end=" ", flush=True)
    df = load_data(tf)
    if df is None or len(df) < 200:
        print(f"[FAIL] Insufficient data")
        return

    print(f"[OK] {len(df):,} bars")

    # Create labels
    print(f"Creating 3-class labels...", end=" ", flush=True)
    y = create_labels(df, tf)
    print(f"[OK]")

    # Remove NaN labels
    valid_idx = ~y.isna()
    df = df[valid_idx]
    y = y[valid_idx].values

    print(f"Valid samples: {len(y):,}")

    if len(df) < LOOKBACK + 10:
        print(f"[FAIL] Not enough valid samples")
        return

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution:")
    for u, c in zip(unique, counts):
        label = {1: 'UP', -1: 'DOWN', 0: 'FLAT'}[u]
        pct = (c / len(y)) * 100
        print(f"  {label:4s}: {c:>6,} ({pct:>5.1f}%)")

    # Create OHLCV sequences
    print(f"Creating {LOOKBACK}-bar OHLCV sequences...", end=" ", flush=True)
    X_seq, y_seq, scalers = create_ohlcv_sequences(df, y, LOOKBACK)
    print(f"[OK] {len(X_seq):,} sequences")

    if len(X_seq) < 50:
        print(f"[FAIL] Not enough sequences")
        return

    # Time series split
    n_train = int(len(X_seq) * 0.8)
    n_val = int(len(X_seq) * 0.1)

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train:n_train+n_val]
    y_val = y_seq[n_train:n_train+n_val]
    X_test = X_seq[n_train+n_val:]
    y_test = y_seq[n_train+n_val:]

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Filter FLAT
    train_mask = y_train != 0
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    # Remap to binary
    y_train_binary = (y_train > 0).astype(np.int32)
    y_val_binary = (y_val > 0).astype(np.int32)
    y_test_binary = (y_test > 0).astype(np.int32)

    print(f"After filtering FLAT: {len(X_train):,} training samples")

    # Build and train
    print(f"Building 1D-CNN model...", end=" ", flush=True)
    model = build_1dcnn_model()
    print(f"[OK]")

    print(f"Training...", end=" ", flush=True)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train_binary,
        validation_data=(X_val, y_val_binary),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=0
    )

    print(f"[OK]")

    # Evaluate
    print(f"Evaluating...", end=" ", flush=True)
    train_loss, train_acc = model.evaluate(X_train, y_train_binary, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val_binary, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_binary, verbose=0)

    print(f"[OK]")
    print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Save
    print(f"Saving model...", end=" ", flush=True)

    model_path = MODELS_DIR / f'cnn_1d_XAUUSD_{tf}.h5'
    model.save(str(model_path))

    scaler_path = MODELS_DIR / f'scaler_cnn_1d_XAUUSD_{tf}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)

    # Metadata
    metadata = {
        'timeframe': tf,
        'asset': ASSET,
        'lookback': LOOKBACK,
        'channels': 5,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'epochs_trained': len(history.history['loss']),
        'training_date': datetime.now().isoformat(),
    }

    metadata_path = MODELS_DIR / f'metadata_cnn_1d_XAUUSD_{tf}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK]")
    print(f"  Model: {model_path.name}")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*70)
    print("1D-CNN Multi-Timeframe Training (15min, 1H)")
    print("="*70)

    for tf in TIMEFRAMES:
        try:
            train_1dcnn_tf(tf)
        except Exception as e:
            print(f"[FAIL] Error training {tf}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("[OK] All 1D-CNN models trained!")
    print(f"{'='*70}")
