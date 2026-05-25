"""
LSTM with Temporal Attention — Sequence model for 7-year XAUUSD predictions.

Processes 60-bar lookback windows with LSTM + learned attention weights.
Trains separate models per timeframe.
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
from tensorflow.keras import layers, Sequential, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json

from training.feature_engineering_v2 import build_features_v2, get_feature_names

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / 'data' / 'raw' / 'mt5'
MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ASSET = 'XAUUSD'
TIMEFRAMES = ['1D', '4H', '1H', '15min']

# LSTM config
LOOKBACK = 60  # 60 bars
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
VALIDATION_SPLIT = 0.2

# Label horizons (bars ahead)
LABEL_HORIZONS = {
    '1D': 3,
    '4H': 3,
    '1H': 4,
    '15min': 4,
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
    from training.feature_engineering_v2 import atr as calc_atr

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

def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 60):
    """Convert flat arrays to sequences for LSTM."""
    X_seq, y_seq = [], []

    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])

    return np.array(X_seq), np.array(y_seq)

# ── Model Architecture ─────────────────────────────────────────────────────
def build_lstm_attention_model(n_features: int):
    """
    LSTM with Temporal Attention.

    Input: (batch, lookback, n_features)
    Conv1D → 2×LSTM → Temporal Attention → Dense → Output(3 classes)
    """
    inputs = layers.Input(shape=(LOOKBACK, n_features))

    # Local pattern extraction via Conv1D
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # LSTM layers
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)

    # Temporal Attention: learned dot-product attention across time
    # Compute attention weights over the sequence
    attention = layers.Dense(1, activation='sigmoid')(x)  # (batch, lookback, 1)
    attention = layers.Flatten()(attention)
    attention = layers.Softmax()(attention)
    attention = layers.Reshape((LOOKBACK, 1))(attention)

    # Apply attention weights
    x = layers.Multiply()([x, attention])
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output: 3-class softmax
    outputs = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ── Train per timeframe ────────────────────────────────────────────────────
def train_lstm_tf(tf: str):
    """Train LSTM with Attention for a single timeframe."""

    print(f"\n{'='*70}")
    print(f"LSTM+Attention Training: {ASSET} @ {tf}")
    print(f"{'='*70}")

    # Load data
    print(f"Loading data...", end=" ", flush=True)
    df = load_data(tf)
    if df is None or len(df) < 200:
        print(f"[FAIL] Insufficient data")
        return

    print(f"[OK] {len(df):,} bars")

    # Build features
    print(f"Building {len(get_feature_names())} features...", end=" ", flush=True)
    X = build_features_v2(df, tf)
    print(f"[OK]")

    # Create labels
    print(f"Creating 3-class labels...", end=" ", flush=True)
    y = create_labels(df, tf)
    print(f"[OK]")

    # Remove NaN rows
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx].values
    y = y[valid_idx].values

    print(f"Valid samples: {len(X):,}")

    if len(X) < LOOKBACK + 10:
        print(f"[FAIL] Not enough valid samples for sequences")
        return

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution:")
    for u, c in zip(unique, counts):
        label = {1: 'UP', -1: 'DOWN', 0: 'FLAT'}[u]
        pct = (c / len(y)) * 100
        print(f"  {label:4s}: {c:>6,} ({pct:>5.1f}%)")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    print(f"Creating {LOOKBACK}-bar sequences...", end=" ", flush=True)
    X_seq, y_seq = create_sequences(X_scaled, y, LOOKBACK)
    print(f"[OK] {len(X_seq):,} sequences")

    if len(X_seq) < 50:
        print(f"[FAIL] Not enough sequences for training")
        return

    # Time series split (no shuffle)
    n_train = int(len(X_seq) * 0.8)
    n_val = int(len(X_seq) * 0.1)

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train:n_train+n_val]
    y_val = y_seq[n_train:n_train+n_val]
    X_test = X_seq[n_train+n_val:]
    y_test = y_seq[n_train+n_val:]

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Filter FLAT class for training
    train_mask = y_train != 0
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    # Remap to binary (UP=1, DOWN=0)
    y_train_binary = (y_train > 0).astype(np.int32)
    y_val_binary = (y_val > 0).astype(np.int32)
    y_test_binary = (y_test > 0).astype(np.int32)

    print(f"After filtering FLAT: {len(X_train):,} training samples")

    # Build and train model
    print(f"Building LSTM+Attention model...", end=" ", flush=True)
    model = build_lstm_attention_model(X_scaled.shape[1])
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

    # Save model
    print(f"Saving model...", end=" ", flush=True)

    model_path = MODELS_DIR / f'lstm_att_XAUUSD_{tf}.h5'
    model.save(str(model_path))

    scaler_path = MODELS_DIR / f'scaler_lstm_att_XAUUSD_{tf}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save metadata
    metadata = {
        'timeframe': tf,
        'asset': ASSET,
        'n_features': X_scaled.shape[1],
        'lookback': LOOKBACK,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'epochs_trained': len(history.history['loss']),
        'training_date': datetime.now().isoformat(),
    }

    metadata_path = MODELS_DIR / f'metadata_lstm_att_XAUUSD_{tf}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK]")
    print(f"  Model: {model_path.name}")
    print(f"  Scaler: {scaler_path.name}")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*70)
    print("LSTM+Attention Multi-Timeframe Training")
    print("="*70)

    for tf in TIMEFRAMES:
        try:
            train_lstm_tf(tf)
        except Exception as e:
            print(f"[FAIL] Error training {tf}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("[OK] All LSTM models trained!")
    print(f"{'='*70}")
