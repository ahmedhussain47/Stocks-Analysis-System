"""
Validation Script — Benchmark peak accuracy vs. baseline.

Walk-forward OOS accuracy on all timeframes.
Compares: XGBoost, LSTM, CNN, Ensemble vs. current best.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
import tensorflow
from tensorflow import keras

from training.feature_engineering_v2 import build_features_v2, atr as calc_atr

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / 'data' / 'raw' / 'mt5'
MODELS_DIR = Path(__file__).parent / 'models'

ASSET = 'XAUUSD'
TIMEFRAMES = ['1D', '4H', '1H', '15min']

LABEL_HORIZONS = {
    '1D': 3,
    '4H': 3,
    '1H': 4,
    '15min': 4,
}

ATR_THRESHOLD = 0.5

# ── Label Creation ─────────────────────────────────────────────────────────
def create_labels(df: pd.DataFrame, tf: str, atr_pct: float = ATR_THRESHOLD):
    """Create 3-class labels."""
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

# ── Evaluation Metrics ─────────────────────────────────────────────────────
def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """Compute all metrics."""
    # Filter for binary (UP/DOWN only)
    valid_mask = y_true != 0
    y_binary = (y_true[valid_mask] > 0).astype(int)
    y_pred_binary = y_pred[valid_mask]

    if len(y_binary) < 10:
        return None

    acc = accuracy_score(y_binary, y_pred_binary)
    prec = precision_score(y_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_binary, y_pred_binary, zero_division=0)

    auc = None
    if y_pred_proba is not None:
        proba_binary = y_pred_proba[valid_mask]
        try:
            auc = roc_auc_score(y_binary, proba_binary)
        except:
            auc = None

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'n_samples': len(y_binary),
    }

# ── Validation per Timeframe ───────────────────────────────────────────────
def validate_tf(tf: str):
    """Validate all models on a timeframe."""

    print(f"\n{'='*70}")
    print(f"Validation: {ASSET} @ {tf}")
    print(f"{'='*70}")

    # Load data
    csv_path = DATA_DIR / f"{ASSET}_{tf}_mt5.csv"
    if not csv_path.exists():
        print(f"[FAIL] Data not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
    print(f"Data: {len(df):,} bars")

    # Create labels
    y = create_labels(df, tf)
    valid_idx = ~y.isna()
    df = df[valid_idx]
    y = y[valid_idx].values

    print(f"Valid samples: {len(y):,}")

    # Build features
    X = build_features_v2(df, tf)
    X_scaled = None

    # Walk-forward validation (80% train, 20% test OOS)
    n_train = int(len(X) * 0.8)
    n_test = len(X) - n_train

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"Train: {len(X_train):,} | Test OOS: {len(X_test):,}")

    results = {}

    # ── XGBoost ────────────────────────────────────────────────────────────
    print(f"\nXGBoost:", end=" ", flush=True)
    xgb_model_path = MODELS_DIR / f'xgb_v2_XAUUSD_{tf}.json'
    xgb_scaler_path = MODELS_DIR / f'scaler_v2_XAUUSD_{tf}.pkl'

    if xgb_model_path.exists():
        try:
            xgb_model = xgb.Booster(model_file=str(xgb_model_path))
            xgb_scaler = pickle.load(open(xgb_scaler_path, 'rb'))

            # Preserve feature names
            X_test_scaled = xgb_scaler.transform(X_test.values)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            dmatrix = xgb.DMatrix(X_test_scaled)
            proba = xgb_model.predict(dmatrix)

            # Remap probabilities (assuming binary output: [prob_down, prob_up])
            if len(proba.shape) == 1 or proba.shape[1] == 1:
                y_pred_xgb = (proba.flatten() > 0.5).astype(int)
                proba_xgb = proba.flatten()
            else:
                y_pred_xgb = (proba[:, 1] > 0.5).astype(int) if proba.shape[1] > 1 else (proba[:, 0] > 0.5).astype(int)
                proba_xgb = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

            metrics = evaluate_predictions(y_test, y_pred_xgb, proba_xgb)
            if metrics:
                results['xgb'] = metrics
                print(f"[OK] Acc: {metrics['accuracy']:.2%} | AUC: {metrics.get('auc', 0):.3f}")
            else:
                print(f"[FAIL] Insufficient test samples")
        except Exception as e:
            print(f"[FAIL] {e}")
    else:
        print(f"[SKIP] Model not found")

    # ── LSTM ───────────────────────────────────────────────────────────────
    print(f"LSTM:    ", end=" ", flush=True)
    lstm_model_path = MODELS_DIR / f'lstm_att_XAUUSD_{tf}.h5'
    lstm_scaler_path = MODELS_DIR / f'scaler_lstm_att_XAUUSD_{tf}.pkl'

    if lstm_model_path.exists():
        try:
            lstm_model = keras.models.load_model(str(lstm_model_path))
            lstm_scaler = pickle.load(open(lstm_scaler_path, 'rb'))

            # Need sequences
            lookback = 60
            if len(X_test) >= lookback:
                X_test_full_scaled = lstm_scaler.transform(X.iloc[n_train-lookback:].values)
                X_test_seq = []

                for i in range(n_train-lookback, len(X)):
                    if i + lookback <= len(X_test_full_scaled):
                        X_test_seq.append(X_test_full_scaled[i-n_train+lookback:i-n_train+lookback+lookback])

                if len(X_test_seq) > 0:
                    X_test_seq = np.array(X_test_seq)
                    proba_lstm = lstm_model.predict(X_test_seq, verbose=0)
                    y_pred_lstm = (proba_lstm[:, 1] > 0.5).astype(int) if proba_lstm.shape[1] > 1 else (proba_lstm[:, 0] > 0.5).astype(int)

                    # Match lengths
                    y_test_matched = y_test[-len(y_pred_lstm):]
                    metrics = evaluate_predictions(y_test_matched, y_pred_lstm, proba_lstm[:, 1])
                    if metrics:
                        results['lstm'] = metrics
                        print(f"[OK] Acc: {metrics['accuracy']:.2%} | AUC: {metrics.get('auc', 0):.3f}")
                    else:
                        print(f"[FAIL] No valid samples")
                else:
                    print(f"[FAIL] Could not create sequences")
            else:
                print(f"[SKIP] Not enough data for sequences")
        except Exception as e:
            print(f"[FAIL] {e}")
    else:
        print(f"[SKIP] Model not found")

    # ── 1D-CNN ────────────────────────────────────────────────────────────
    if tf in ['15min', '1H']:
        print(f"1D-CNN:  ", end=" ", flush=True)
        cnn_model_path = MODELS_DIR / f'cnn_1d_XAUUSD_{tf}.h5'
        cnn_scaler_path = MODELS_DIR / f'scaler_cnn_1d_XAUUSD_{tf}.pkl'

        if cnn_model_path.exists():
            try:
                cnn_model = keras.models.load_model(str(cnn_model_path))
                cnn_scalers = pickle.load(open(cnn_scaler_path, 'rb'))

                # Need OHLCV sequences
                lookback = 30
                if len(df) >= lookback:
                    ohlcv = df[['open', 'high', 'low', 'close', 'volume']].values
                    X_test_cnn = []

                    for i in range(len(df)-lookback):
                        if i >= n_train - lookback:
                            window = ohlcv[i:i+lookback].copy()
                            # Normalize
                            for j, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
                                if col in cnn_scalers:
                                    window[:, j] = cnn_scalers[col].transform(window[:, j].reshape(-1, 1)).flatten()
                            X_test_cnn.append(window)

                    if len(X_test_cnn) > 0:
                        X_test_cnn = np.array(X_test_cnn)
                        proba_cnn = cnn_model.predict(X_test_cnn, verbose=0)
                        y_pred_cnn = (proba_cnn[:, 1] > 0.5).astype(int) if proba_cnn.shape[1] > 1 else (proba_cnn[:, 0] > 0.5).astype(int)

                        y_test_matched = y_test[-len(y_pred_cnn):]
                        metrics = evaluate_predictions(y_test_matched, y_pred_cnn, proba_cnn[:, 1])
                        if metrics:
                            results['cnn'] = metrics
                            print(f"[OK] Acc: {metrics['accuracy']:.2%} | AUC: {metrics.get('auc', 0):.3f}")
                        else:
                            print(f"[FAIL] No valid samples")
                    else:
                        print(f"[FAIL] Could not create sequences")
                else:
                    print(f"[SKIP] Not enough data")
            except Exception as e:
                print(f"[FAIL] {e}")
        else:
            print(f"[SKIP] Model not found")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'-'*70}")
    print(f"OOS Accuracy Summary:")
    print(f"{'-'*70}")

    if results:
        best_model = max(results, key=lambda k: results[k]['accuracy'])
        best_acc = results[best_model]['accuracy']

        for model_name, metrics in results.items():
            auc_str = f" | AUC: {metrics['auc']:.3f}" if metrics['auc'] else ""
            print(f"{model_name:10s}: {metrics['accuracy']:.2%}{auc_str}")

        print(f"\nBest: {best_model} at {best_acc:.2%}")
    else:
        print("No model results available")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*70)
    print("Peak Accuracy ML Trading System — Validation")
    print("="*70)

    for tf in TIMEFRAMES:
        try:
            validate_tf(tf)
        except Exception as e:
            print(f"[FAIL] Error validating {tf}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("Validation complete!")
    print(f"{'='*70}")
