"""
GOLD SIGNAL ENGINE - Final Version (Fixed)
Uses: Historic Data + ML Models + 3-Tier System
Generates: High-Quality Signals with Optimized TP/SL
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
import pickle
import warnings
warnings.filterwarnings('ignore')

class GoldSignalEngine:
    """Complete signal generation system."""

    def __init__(self):
        self.last_signals = set()

    def load_ml_models(self):
        """Load LSTM + XGBoost."""
        try:
            import tensorflow as tf
            import xgboost as xgb

            lstm = tf.keras.models.load_model('results/advanced_models/best_model_lstm.h5')
            xgb_model = xgb.Booster()
            xgb_model.load_model('results/advanced_models/best_model_xgb.json')
            with open('results/advanced_models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            print("[OK] ML models loaded")
            return lstm, xgb_model, scaler
        except Exception as e:
            print(f"[WARNING] ML models failed: {e}")
            return None, None, None

    def compute_features(self, df):
        """Compute 8 features for ML."""
        try:
            close = df['Close'].values.flatten()

            ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
            ema50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])

            delta = pd.Series(close).diff()
            gain = float((delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1])
            loss = float((-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1])
            rsi = 100 - (100 / (1 + gain/loss)) if loss != 0 else 50.0

            macd = float(pd.Series(close).ewm(span=12).mean().iloc[-1] - pd.Series(close).ewm(span=26).mean().iloc[-1])
            atr = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
            bb_std = float(pd.Series(close).rolling(20).std().iloc[-1])
            vol_ma = float(df['Volume'].rolling(20).mean().iloc[-1])
            returns = float((close[-1] - close[-2]) / close[-2]) if close[-2] != 0 else 0.0

            return np.array([[ema20, ema50, rsi, macd, atr, bb_std, vol_ma, returns]])
        except Exception as e:
            print(f"[Features error] {e}")
            return None

    def ml_predict(self, df, lstm_model, xgb_model, scaler):
        """ML ensemble prediction."""
        try:
            if lstm_model is None or xgb_model is None:
                return None

            X = self.compute_features(df)
            if X is None:
                return None

            X_scaled = scaler.transform(X)
            X_lstm = X_scaled.reshape((1, 1, 8))
            lstm_pred = float(lstm_model.predict(X_lstm, verbose=0)[0][0])

            import xgboost as xgb
            xgb_pred = float(xgb_model.predict(xgb.DMatrix(X_scaled))[0])

            ensemble = lstm_pred * 0.6 + xgb_pred * 0.4
            direction = 1 if ensemble > 0.5 else -1
            confidence = abs(ensemble - 0.5) * 200

            return {
                'direction': direction,
                'confidence': min(confidence, 95),
                'lstm': lstm_pred,
                'xgb': xgb_pred,
            }
        except Exception as e:
            print(f"[ML error] {e}")
            return None

    def detect_patterns(self, df):
        """Detect technical patterns."""
        try:
            if len(df) < 50:
                return None

            close = df['Close'].values[-50:].flatten()
            high = df['High'].values[-50:].flatten()
            low = df['Low'].values[-50:].flatten()

            # Double Bottom
            if close[-1] > close[-2] and low[-10] > low[-20] and low[-5] < low[-10]:
                return {'pattern': 'DOUBLE_BOTTOM', 'direction': 1}

            # Double Top
            if close[-1] < close[-2] and high[-10] < high[-20] and high[-5] > high[-10]:
                return {'pattern': 'DOUBLE_TOP', 'direction': -1}

            # Support Bounce
            support = low[-20:].min()
            if abs(close[-1] - support) / support < 0.005:
                return {'pattern': 'SUPPORT_BOUNCE', 'direction': 1}

            return None
        except:
            return None

    def optimize_entry(self, df, signal):
        """Compute tight TP/SL."""
        try:
            close = float(df['Close'].iloc[-1])
            atr = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
            support = float(df['Low'].iloc[-50:].min())
            resistance = float(df['High'].iloc[-50:].max())

            if signal == 'BUY':
                entry = close
                sl = max(support * 0.99, close - atr * 1.0)
                tp1 = close + atr * 1.5
                tp2 = close + atr * 2.2
                tp3 = close + atr * 3.0
                rr = (tp1 - entry) / (entry - sl) if (entry - sl) > 0 else 0
            else:
                entry = close
                sl = min(resistance * 1.01, close + atr * 1.0)
                tp1 = close - atr * 1.5
                tp2 = close - atr * 2.2
                tp3 = close - atr * 3.0
                rr = (entry - tp1) / (sl - entry) if (sl - entry) > 0 else 0

            return {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'rr': rr}
        except Exception as e:
            print(f"[Entry error] {e}")
            return None

    def validate_signal(self, direction, df):
        """Validate against trend."""
        try:
            close = df['Close'].values.flatten()
            ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
            ema50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])
            trend = 1 if ema20 > ema50 else -1

            if direction != trend:
                return False

            atr_ratio = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1]) / close[-1]
            if atr_ratio > 0.01:
                return False

            return True
        except:
            return True

    def generate_signal(self, df, lstm_model, xgb_model, scaler):
        """Generate signal using all tiers."""

        # Tier 1: Patterns
        pattern = self.detect_patterns(df)

        # Tier 1: ML
        ml = self.ml_predict(df, lstm_model, xgb_model, scaler)

        # Determine direction
        if ml:
            direction = ml['direction']
            confidence = ml['confidence']
            source = f"ML ({confidence:.0f}%)"
        elif pattern:
            direction = pattern['direction']
            confidence = 70.0
            source = pattern['pattern']
        else:
            return None

        # Tier 2: Entry optimization
        signal = 'BUY' if direction > 0 else 'SELL'
        tp_sl = self.optimize_entry(df, signal)
        if tp_sl is None:
            return None

        # Tier 3: Validation
        if not self.validate_signal(direction, df):
            return None

        # Avoid duplicates
        signal_key = (direction, tp_sl['entry'])
        if signal_key in self.last_signals:
            return None
        self.last_signals.add(signal_key)

        return {
            'signal': signal,
            'entry': tp_sl['entry'],
            'sl': tp_sl['sl'],
            'tp1': tp_sl['tp1'],
            'tp2': tp_sl['tp2'],
            'tp3': tp_sl['tp3'],
            'rr': tp_sl['rr'],
            'confidence': confidence,
            'source': source,
        }

def main():
    print("="*70)
    print("GOLD SIGNAL ENGINE - Final Version")
    print("3-Tier: Patterns + ML + Optimization + Validation")
    print("="*70)

    engine = GoldSignalEngine()
    lstm_model, xgb_model, scaler = engine.load_ml_models()

    print("\nFetching gold data...")
    df = yf.download('GC=F', period='7d', interval='1h', progress=False)

    if df.empty or len(df) < 50:
        print("[ERROR] Insufficient data")
        return

    print(f"[OK] Loaded {len(df)} bars\n")

    signal = engine.generate_signal(df, lstm_model, xgb_model, scaler)

    if signal:
        print("="*70)
        print(f"SIGNAL: {signal['signal']}  XAUUSD [1H]")
        print(f"Entry:  {signal['entry']:.5f}")
        print(f"SL:     {signal['sl']:.5f}  ({abs(signal['entry']-signal['sl']):.2f} pts)")
        print(f"TP1/2/3: {signal['tp1']:.2f} / {signal['tp2']:.2f} / {signal['tp3']:.2f}")
        print(f"R:R:    1:{signal['rr']:.2f}")
        print(f"Conf:   {signal['confidence']:.0f}%")
        print(f"Source: {signal['source']}")
        print("="*70)
    else:
        print("[NO SIGNAL] Conditions not met")

if __name__ == '__main__':
    main()
