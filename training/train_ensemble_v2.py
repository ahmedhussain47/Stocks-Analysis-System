"""
Stacked Ensemble v2 — Meta-learner combining XGBoost, LSTM, 1D-CNN.

Trains logistic regression on concatenated probability outputs from level-0 models.
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
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import tensorflow as tf

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / 'data' / 'raw' / 'mt5'
MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ASSET = 'XAUUSD'
TIMEFRAMES = ['1D', '4H', '1H', '15min']

LABEL_HORIZONS = {
    '1D': 3,
    '4H': 3,
    '1H': 4,
    '15min': 4,
}

ATR_THRESHOLD = 0.5

# ── Load base model predictions ────────────────────────────────────────────
def load_model_predictions(tf: str) -> dict:
    """Load predictions from trained base models."""
    predictions = {}

    # Try XGBoost
    xgb_model_path = MODELS_DIR / f'xgb_v2_XAUUSD_{tf}.json'
    xgb_scaler_path = MODELS_DIR / f'scaler_v2_XAUUSD_{tf}.pkl'

    if xgb_model_path.exists() and xgb_scaler_path.exists():
        predictions['xgb'] = {
            'model': xgb.Booster(model_file=str(xgb_model_path)),
            'scaler': pickle.load(open(xgb_scaler_path, 'rb'))
        }
    else:
        print(f"[WARN] XGBoost model not found for {tf}")

    # Try LSTM
    lstm_model_path = MODELS_DIR / f'lstm_att_XAUUSD_{tf}.h5'
    lstm_scaler_path = MODELS_DIR / f'scaler_lstm_att_XAUUSD_{tf}.pkl'

    if lstm_model_path.exists() and lstm_scaler_path.exists():
        predictions['lstm'] = {
            'model': tf.keras.models.load_model(str(lstm_model_path)),
            'scaler': pickle.load(open(lstm_scaler_path, 'rb'))
        }
    else:
        print(f"[WARN] LSTM model not found for {tf}")

    # Try 1D-CNN
    if tf in ['15min', '1H']:
        cnn_model_path = MODELS_DIR / f'cnn_1d_XAUUSD_{tf}.h5'
        cnn_scaler_path = MODELS_DIR / f'scaler_cnn_1d_XAUUSD_{tf}.pkl'

        if cnn_model_path.exists() and cnn_scaler_path.exists():
            predictions['cnn'] = {
                'model': tf.keras.models.load_model(str(cnn_model_path)),
                'scaler': pickle.load(open(cnn_scaler_path, 'rb'))
            }
        else:
            print(f"[WARN] 1D-CNN model not found for {tf}")

    return predictions

def create_labels(df: pd.DataFrame, tf: str, atr_pct: float = ATR_THRESHOLD) -> pd.Series:
    """Create 3-class labels."""
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

def load_data(tf: str) -> pd.DataFrame:
    """Load CSV data."""
    csv_path = DATA_DIR / f"{ASSET}_{tf}_mt5.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)

# ── Train Ensemble ───────────────────────────────────────────────────────
def train_ensemble_tf(tf: str):
    """Train ensemble for a single timeframe."""

    print(f"\n{'='*70}")
    print(f"Stacked Ensemble Training: {ASSET} @ {tf}")
    print(f"{'='*70}")

    # Load data
    print(f"Loading data...", end=" ", flush=True)
    df = load_data(tf)
    if df is None or len(df) < 100:
        print(f"[FAIL] Insufficient data")
        return

    print(f"[OK] {len(df):,} bars")

    # Create labels
    print(f"Creating labels...", end=" ", flush=True)
    y = create_labels(df, tf)
    print(f"[OK]")

    valid_idx = ~y.isna()
    df = df[valid_idx]
    y = y[valid_idx].values

    if len(df) < 50:
        print(f"[FAIL] Not enough valid samples")
        return

    # Load base models
    print(f"Loading base models...", end=" ", flush=True)
    base_models = load_model_predictions(tf)

    if len(base_models) < 2:
        print(f"[FAIL] Need at least 2 base models trained")
        return

    print(f"[OK] {len(base_models)} models loaded")

    # Generate meta-features (OOF predictions)
    print(f"Generating out-of-fold predictions...", end=" ", flush=True)

    from training.feature_engineering_v2 import build_features_v2

    X = build_features_v2(df, tf)
    X_scaled = StandardScaler().fit_transform(X)

    n_samples = len(X)
    meta_features = []

    # Iterate through folds for OOF predictions
    tscv = TimeSeriesSplit(n_splits=3)

    meta_X = np.zeros((n_samples, len(base_models) * 3))  # 3 probs per model
    meta_idx = 0

    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Get predictions from each base model on validation fold
        fold_meta = np.zeros((len(val_idx), len(base_models) * 3))

        col_idx = 0
        for model_name, model_dict in base_models.items():
            model = model_dict['model']

            if model_name == 'xgb':
                # XGBoost predict_proba
                X_val_xgb = xgb.DMatrix(X_val)
                proba = model.predict(X_val_xgb)
                # Reshape if needed (usually comes as [n_samples, 3])
                if len(proba.shape) == 1:
                    proba = np.column_stack([1-proba, proba, np.zeros(len(proba))])
            else:
                # LSTM or CNN predict
                proba = model.predict(X_val, verbose=0)

            # Use first 3 columns (probabilities for classes)
            fold_meta[:, col_idx:col_idx+3] = proba[:, :3]
            col_idx += 3

        meta_X[val_idx] = fold_meta

    print(f"[OK]")

    # Filter FLAT class
    print(f"Training meta-learner...", end=" ", flush=True)

    train_mask = y != 0
    meta_X_train = meta_X[train_mask]
    y_train_binary = (y[train_mask] > 0).astype(int)

    # Train logistic regression meta-learner
    meta_learner = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    meta_learner.fit(meta_X_train, y_train_binary)

    print(f"[OK]")

    # Evaluate
    print(f"Evaluating...", end=" ", flush=True)

    y_pred = meta_learner.predict(meta_X_train)
    y_pred_proba = meta_learner.predict_proba(meta_X_train)[:, 1]

    train_acc = accuracy_score(y_train_binary, y_pred)
    train_auc = roc_auc_score(y_train_binary, y_pred_proba)

    print(f"[OK]")
    print(f"  Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}")

    # Save meta-learner
    print(f"Saving ensemble...", end=" ", flush=True)

    ensemble_path = MODELS_DIR / f'ensemble_meta_XAUUSD_{tf}.pkl'
    with open(ensemble_path, 'wb') as f:
        pickle.dump(meta_learner, f)

    metadata = {
        'timeframe': tf,
        'asset': ASSET,
        'base_models': list(base_models.keys()),
        'n_meta_features': meta_X_train.shape[1],
        'train_acc': float(train_acc),
        'train_auc': float(train_auc),
        'training_date': datetime.now().isoformat(),
    }

    metadata_path = MODELS_DIR / f'metadata_ensemble_XAUUSD_{tf}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK]")
    print(f"  Ensemble: {ensemble_path.name}")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*70)
    print("Stacked Ensemble v2 Training")
    print("="*70)

    for tf in TIMEFRAMES:
        try:
            train_ensemble_tf(tf)
        except Exception as e:
            print(f"[FAIL] Error training {tf}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("[OK] Ensemble training complete!")
    print(f"{'='*70}")
