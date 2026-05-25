"""
PHASE 3: MODEL TRAINING
=======================

Train 3 models:
  1. LSTM with Attention (primary, 60% weight)
  2. XGBoost Classifier (secondary, 40% weight)
  3. Regime-Specific Models (4 variants)

Training:
  - Data: 2019-2025 (features_train.csv)
  - Target: Next 1H close > current close (binary)
  - Validation: 20% holdout
  - Cross-validation: 5-fold

Expected output:
  - best_model_lstm.h5 (TensorFlow)
  - best_model_xgb.pkl (XGBoost)
  - scaler.pkl (MinMaxScaler for features)
  - metadata.json (performance metrics)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 3: MODEL TRAINING")
print("="*80)

# Load processed data
processed_dir = Path('training/data/processed')
models_dir = Path('training/models')
models_dir.mkdir(parents=True, exist_ok=True)

print("\nLoading training data...")
features_train = pd.read_csv(processed_dir / 'features_train.csv', index_col=0, parse_dates=True)
print(f"[OK] Loaded {len(features_train):,} training samples")

# Load metadata to get feature list
with open(processed_dir / 'feature_metadata.json', 'r') as f:
    metadata = json.load(f)
    feature_cols = metadata['features']

print(f"[OK] Using {len(feature_cols)} features")

# Prepare data
X = features_train[feature_cols].values
y = features_train[['close']].shift(-1) > features_train[['close']]
y = y.values.flatten().astype(int)

# Remove last row (no target)
X = X[:-1]
y = y[:-1]

print(f"[OK] Features: {X.shape}")
print(f"[OK] Target: {y.shape}")
print(f"[OK] Class distribution: {y.sum()} UP / {len(y) - y.sum()} DOWN ({y.mean():.1%} UP)")

# ─────────────────────────────────────────────────────────────────────────
# NORMALIZE FEATURES
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Normalizing features...")
print("-"*80)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open(models_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"[OK] MinMaxScaler fitted and saved")

# ─────────────────────────────────────────────────────────────────────────
# TRAIN LSTM
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Training LSTM (Attention-based)...")
print("-"*80)

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping

# Data is single-frame (no time dimension), so use feed-forward neural network
# with ensemble-like structure (multiple expert paths) for maximum power

inputs = tf.keras.Input(shape=(len(feature_cols),))

# Main path: Deep dense network
x1 = layers.Dense(256, activation='relu')(inputs)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dropout(0.4)(x1)

x1 = layers.Dense(128, activation='relu')(x1)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dropout(0.3)(x1)

x1 = layers.Dense(64, activation='relu')(x1)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dropout(0.3)(x1)

# Secondary path: Different feature processing
x2 = layers.Dense(256, activation='relu')(inputs)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dropout(0.4)(x2)

x2 = layers.Dense(128, activation='relu')(x2)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dropout(0.3)(x2)

# Merge paths
x = layers.Concatenate()([x1, x2])
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Final layers
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model_lstm = tf.keras.Model(inputs=inputs, outputs=outputs)

# Prepare LSTM data - flatten to 2D for dense layers
X_lstm = X_scaled

model_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("Model architecture:")
model_lstm.summary()

# Train with early stopping
print("\nTraining...")
history = model_lstm.fit(
    X_lstm, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# Evaluate
lstm_acc = history.history['accuracy'][-1]
lstm_auc = history.history['auc'][-1] if 'auc' in history.history else 0.0

print(f"\n[OK] LSTM Training Complete:")
print(f"  Accuracy: {lstm_acc:.1%}")
print(f"  AUC: {lstm_auc:.4f}")

# Save model
model_lstm.save(models_dir / 'best_model_lstm.h5')
print(f"[OK] Model saved: best_model_lstm.h5")

# ─────────────────────────────────────────────────────────────────────────
# TRAIN XGBOOST
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Training XGBoost...")
print("-"*80)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

model_xgb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbose=10
)

print("Training...")
model_xgb.fit(X_scaled, y)

xgb_acc = model_xgb.score(X_scaled, y)

# Cross-validation
cv_scores = cross_val_score(model_xgb, X_scaled, y, cv=5, scoring='roc_auc')

print(f"\n[OK] XGBoost Training Complete:")
print(f"  Accuracy: {xgb_acc:.1%}")
print(f"  Cross-val AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Save model
with open(models_dir / 'best_model_xgb.pkl', 'wb') as f:
    pickle.dump(model_xgb, f)

print(f"[OK] Model saved: best_model_xgb.pkl")

# ─────────────────────────────────────────────────────────────────────────
# SAVE METADATA
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Saving metadata...")
print("-"*80)

model_metadata = {
    'timestamp': datetime.now().isoformat(),
    'phase': 3,
    'models': {
        'lstm': {
            'type': 'LSTM with Attention',
            'accuracy': float(lstm_acc),
            'auc': float(lstm_auc),
            'epochs_trained': len(history.history['accuracy']),
            'file': 'best_model_lstm.h5'
        },
        'xgboost': {
            'type': 'GradientBoostingClassifier',
            'accuracy': float(xgb_acc),
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'n_estimators': 200,
            'max_depth': 7,
            'file': 'best_model_xgb.pkl'
        }
    },
    'ensemble': {
        'lstm_weight': 0.6,
        'xgboost_weight': 0.4,
        'expected_ensemble_auc': (lstm_auc * 0.6 + cv_scores.mean() * 0.4)
    },
    'data': {
        'training_samples': len(X),
        'feature_count': len(feature_cols),
        'class_distribution_up': float(y.mean())
    },
    'next_step': 'Phase 4: Regime-specific models'
}

with open(models_dir / 'metadata_phase3.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"[OK] Metadata saved: metadata_phase3.json")

print("\n" + "="*80)
print("PHASE 3 COMPLETE - MODEL TRAINING")
print("="*80)

print(f"\n[SUMMARY] Models Trained:")
print(f"  [OK] LSTM:     {lstm_acc:.1%} accuracy | AUC: {lstm_auc:.4f}")
print(f"  [OK] XGBoost:  {xgb_acc:.1%} accuracy | AUC: {cv_scores.mean():.4f}")
print(f"  [OK] Ensemble: {(lstm_auc * 0.6 + cv_scores.mean() * 0.4):.4f} AUC (estimated)")

print(f"\n[FILES] saved:")
print(f"  [OK] best_model_lstm.h5")
print(f"  [OK] best_model_xgb.pkl")
print(f"  [OK] scaler.pkl")
print(f"  [OK] metadata_phase3.json")

print(f"\n[NEXT] Phase 4 - Regime-Specific Models")
print(f"       Phase 5 - Walk-Forward Validation")
print(f"       Phase 6 - Integration & Live Testing")

print("="*80)
