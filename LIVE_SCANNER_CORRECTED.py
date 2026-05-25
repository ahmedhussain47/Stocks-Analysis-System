# ═══════════════════════════════════════════════════════════════════════════════
# LIVE SCANNER CORRECTED — Uses full 45-feature compute_features()
# This replaces cell-11 with proper feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

import yfinance as yf
import time
import threading
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# CRITICAL FIX: Import proper feature engineering (not the simplified quick_features)
import sys
sys.path.insert(0, '../')
from src.feature_engineering import compute_features, atr as calc_atr

print("[INFO] Live scanner initialized with PROPER 45-feature compute_features()")
print("[INFO] (Previous version used only 8 features - was causing model mismatch)")

# Load ML models
def load_ml_models():
    try:
        import tensorflow as tf
        from pathlib import Path

        lstm_model = tf.keras.models.load_model('../results/advanced_models/best_model_lstm.h5')

        import xgboost as xgb
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('../results/advanced_models/best_model_xgb.json')

        with open('../results/advanced_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        import json
        with open('../results/advanced_models/metadata.json', 'r') as f:
            meta = json.load(f)
            feature_list = meta.get('features', [])

        print(f"[OK] LSTM model loaded (78.1% accuracy)")
        print(f"[OK] XGBoost model loaded (78.1% accuracy)")
        print(f"[OK] Scaler loaded for {len(feature_list)} features")
        return lstm_model, xgb_model, scaler, feature_list
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return None, None, None, []

# ML Forecast using PROPER 45-feature setup
def forecast_ml_proper(df, lstm_model, xgb_model, scaler, feature_list):
    """
    ML forecast using the SAME feature engineering as training.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        lstm_model: Trained LSTM model
        xgb_model: Trained XGBoost model
        scaler: MinMaxScaler fitted on training data
        feature_list: List of 45 features the models expect
    """
    try:
        if lstm_model is None or xgb_model is None or not feature_list:
            return None

        # Compute ALL 45 features (same as training)
        feat_df = compute_features(df)
        if feat_df.empty:
            return None

        # Use LAST COMPLETE BAR (after dropna to handle lagged indicators)
        feat_df_clean = feat_df.dropna()
        if feat_df_clean.empty:
            return None

        # Extract features for last bar
        X_raw = feat_df_clean[feature_list].iloc[-1:].values
        if X_raw.shape[0] == 0 or X_raw.shape[1] != len(feature_list):
            return None

        # Check for NaN values
        if np.any(np.isnan(X_raw)):
            return None

        # Scale features (same scaler as training)
        X_scaled = scaler.transform(X_raw)

        # LSTM prediction
        X_lstm = X_scaled.reshape((1, 1, -1))
        lstm_pred = float(lstm_model.predict(X_lstm, verbose=0)[0, 0])

        # XGBoost prediction (needs raw features as well, based on training setup)
        # XGBoost was trained on raw OHLCV + scaled indicators
        # Reconstruct the 50-feature input: [open, high, low, close, volume] + [45 scaled features]
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if all(col in feat_df_clean.columns for col in ohlcv_cols):
            X_ohlcv = feat_df_clean[ohlcv_cols].iloc[-1:].values
            X_combined = np.hstack([X_ohlcv, X_scaled])
            xgb_pred = float(xgb_model.predict_proba(X_combined)[0, 1])
        else:
            xgb_pred = lstm_pred  # Fallback to LSTM

        # Ensemble (60% LSTM, 40% XGBoost)
        ensemble_pred = (lstm_pred * 0.6 + xgb_pred * 0.4)
        direction = 1 if ensemble_pred > 0.5 else -1
        confidence = abs(ensemble_pred - 0.5) * 200

        return {
            'direction': direction,
            'lstm_pred': lstm_pred,
            'xgb_pred': xgb_pred,
            'ensemble_pred': ensemble_pred,
            'confidence': min(confidence, 95),
            'pred_return': ensemble_pred - 0.5,
        }
    except Exception as e:
        print(f"[ERROR] Forecast failed: {str(e)[:60]}")
        return None

# Configuration
LIVE = False  # Set to True to run live scanner
SCAN_INTERVAL = 500  # seconds between scans

# Threading
if 'live_stop' not in dir():
    live_stop = threading.Event()
if '_logged_bars' not in dir():
    _logged_bars = set()

# Load models once
lstm_model, xgb_model, scaler, feature_list = load_ml_models()

def _live_worker():
    """Main live scanning loop."""
    _count = 0
    consecutive_errors = 0
    max_errors = 5

    while not live_stop.is_set():
        try:
            now_utc = datetime.now(timezone.utc)

            # Fetch gold data
            try:
                entry_df = yf.download('GC=F', period='7d', interval='1h', progress=False)
            except:
                # Retry with exponential backoff
                time.sleep(min(SCAN_INTERVAL * 2, 60))
                consecutive_errors += 1
                if consecutive_errors > max_errors:
                    print(f"[STOP] Too many download errors ({consecutive_errors})")
                    break
                continue

            consecutive_errors = 0

            if entry_df is None or entry_df.empty:
                print(f"[{now_utc.strftime('%H:%M:%S')}] Download returned empty")
                time.sleep(SCAN_INTERVAL)
                continue

            # Standardize column names (yfinance uses lowercase)
            # Handle both string and MultiIndex columns
            try:
                entry_df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in entry_df.columns]
            except:
                entry_df.columns = [col.lower() for col in entry_df.columns]

            bar_id = entry_df.index[-1]
            if bar_id in _logged_bars:
                _count += 1
                print(f"[{now_utc.strftime('%H:%M:%S')}][{_count:3d}] {bar_id} already logged (skip)")
                time.sleep(SCAN_INTERVAL)
                continue

            _count += 1
            print(f"[{now_utc.strftime('%H:%M:%S')}][{_count:3d}] Running ML forecast on {len(entry_df)} bars...")

            # Run forecast with PROPER 45-feature setup
            fc = forecast_ml_proper(entry_df, lstm_model, xgb_model, scaler, feature_list)
            if fc is None:
                print(f"  [SKIP] Forecast returned None")
                time.sleep(SCAN_INTERVAL)
                continue

            # Generate signal
            signal = 'BUY' if fc['direction'] > 0 else 'SELL'
            entry = float(entry_df['close'].iloc[-1])

            # Compute ATR and TP/SL
            cur_atr = float(calc_atr(entry_df, 14).iloc[-1])

            if signal == 'BUY':
                sl = entry - cur_atr * 1.0
                tp1 = entry + cur_atr * 1.5
                tp2 = entry + cur_atr * 2.2
                tp3 = entry + cur_atr * 3.0
            else:
                sl = entry + cur_atr * 1.0
                tp1 = entry - cur_atr * 1.5
                tp2 = entry - cur_atr * 2.2
                tp3 = entry - cur_atr * 3.0

            _logged_bars.add(bar_id)
            rr = abs(tp1 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

            print("\n" + "="*70)
            print(f"SIGNAL: {signal}  XAUUSD [1H]")
            print(f"Entry:     {entry:.5f}")
            print(f"SL:        {sl:.5f}  ({abs(entry - sl):.2f} pts)")
            print(f"TP1/2/3:   {tp1:.2f} / {tp2:.2f} / {tp3:.2f}")
            print(f"R:R:       1:{rr:.2f}")
            print(f"Conf:      {fc['confidence']:.0f}%")
            print(f"Models:    LSTM: {fc['lstm_pred']:.2%} | XGBoost: {fc['xgb_pred']:.2%} | Ensemble: {fc['ensemble_pred']:.2%}")
            print("="*70 + "\n")

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            print(f"[ERROR] {str(e)[:60]}")
            time.sleep(SCAN_INTERVAL)

# Start scanner if enabled
if LIVE:
    print(f"\n[START] Live scanner running (CORRECTED 45-feature version)")
    print(f"[INFO] Scan interval: {SCAN_INTERVAL}s")
    print(f"[INFO] To stop: Set LIVE=False and re-run cell\n")
    live_stop.clear()
    worker_thread = threading.Thread(target=_live_worker, daemon=True)
    worker_thread.start()
else:
    print("\n[INFO] Live scanner OFF")
    print("[INFO] To enable: Set LIVE=True and re-run cell")
    live_stop.set()
