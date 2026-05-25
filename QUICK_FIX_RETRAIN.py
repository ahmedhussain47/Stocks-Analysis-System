"""
QUICK FIX: Retrain LSTM + XGBoost on latest XAUUSD data (May 2026)
This fixes the SELL-bias issue by training on current market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("QUICK FIX: Retraining on Latest XAUUSD Data")
print("="*60)

# ─────────────────────────────────────────────────────────────
# STEP 1: Fetch Latest Data
# ─────────────────────────────────────────────────────────────

print("\n[1/5] Fetching XAUUSD data (May 1-25, 2026)...")
try:
    df = yf.download('XAUUSD=X', start='2026-05-01', end='2026-05-26', interval='1h', progress=False)
    if df.empty:
        # Fallback: load from local if yfinance fails
        print("[WARNING] yfinance failed, using local data...")
        df = pd.read_csv('results/xauusd_1h.csv', index_col=0, parse_dates=True)
    print(f"✅ Loaded {len(df)} bars")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ─────────────────────────────────────────────────────────────
# STEP 2: Feature Engineering
# ─────────────────────────────────────────────────────────────

print("\n[2/5] Engineering features...")

df['returns'] = df['Close'].pct_change()
df['ema_20'] = df['Close'].ewm(span=20).mean()
df['ema_50'] = df['Close'].ewm(span=50).mean()
df['rsi_14'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                  -df['Close'].diff().clip(upper=0).rolling(14).mean())))
df['atr_14'] = ((df['High'] - df['Low']).rolling(14).mean())
df['bb_std'] = df['Close'].rolling(20).std()
df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
df['volume_ma'] = df['Volume'].rolling(20).mean()

df.dropna(inplace=True)

# Target: Next bar direction
df['direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1=UP, 0=DOWN

# Features for model
features = ['ema_20', 'ema_50', 'rsi_14', 'atr_14', 'bb_std', 'macd', 'volume_ma', 'returns']
X = df[features].values
y = df['direction'].values

print(f"✅ Features: {len(features)}, Samples: {len(X)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: Normalize Data
# ─────────────────────────────────────────────────────────────

print("\n[3/5] Normalizing data...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
import joblib
joblib.dump(scaler, 'results/advanced_models/scaler_xauusd_quick.pkl')

print(f"✅ Normalized shape: {X_scaled.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4: Train LSTM
# ─────────────────────────────────────────────────────────────

print("\n[4/5] Training LSTM (30 epochs)...")

# Reshape for LSTM (samples, timesteps=1, features)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

model_lstm = Sequential([
    LSTM(64, activation='relu', input_shape=(1, len(features))),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_lstm.fit(X_lstm, y, epochs=30, batch_size=16, validation_split=0.2, verbose=0)

accuracy = history.history['accuracy'][-1]
print(f"✅ LSTM trained - Accuracy: {accuracy:.2%}")

model_lstm.save('results/advanced_models/best_model_lstm_quick.h5')

# ─────────────────────────────────────────────────────────────
# STEP 5: Train XGBoost
# ─────────────────────────────────────────────────────────────

print("\n[5/5] Training XGBoost...")

model_xgb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model_xgb.fit(X_scaled, y)
xgb_score = model_xgb.score(X_scaled, y)
print(f"✅ XGBoost trained - Accuracy: {xgb_score:.2%}")

with open('results/advanced_models/best_model_xgb_quick.pkl', 'wb') as f:
    pickle.dump(model_xgb, f)

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("QUICK FIX COMPLETE!")
print("="*60)
print(f"\n✅ LSTM Accuracy: {accuracy:.2%}")
print(f"✅ XGBoost Accuracy: {xgb_score:.2%}")
print(f"✅ Training data: May 1-25, 2026 ({len(X)} bars)")
print(f"✅ Models saved to: results/advanced_models/")
print("\nNext: Update notebook to use:")
print("  - best_model_lstm_quick.h5")
print("  - best_model_xgb_quick.pkl")
print("\nThen restart scanner!")
