"""
PHASE 3: MODEL TRAINING - MULTI-ASSET ENRICHED
===============================================

Train on 1,814 XAUUSD samples enriched with cross-asset correlations.
This combines single-asset focus with multi-market context.
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
print("PHASE 3: MODEL TRAINING - MULTI-ASSET ENRICHED XAUUSD")
print("="*80)

# Load multi-asset enriched data
processed_dir = Path('training/data/processed')
models_dir = Path('training/models')
models_dir.mkdir(parents=True, exist_ok=True)

print("\nLoading multi-asset enriched XAUUSD data...")
features_train = pd.read_csv(processed_dir / 'features_train_multisset.csv', index_col=0, parse_dates=True)
print(f"[OK] Loaded {len(features_train):,} samples from {features_train.index[0].date()} to {features_train.index[-1].date()}")

# Load metadata
with open(processed_dir / 'feature_metadata_multisset.json', 'r') as f:
    metadata = json.load(f)
    feature_cols = metadata['features']

print(f"[OK] Using {len(feature_cols)} features (technical + cross-asset correlations)")
print(f"     Features: {feature_cols[:8]}... (showing first 8)")

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

# Normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

with open(models_dir / 'scaler_multiasset.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"[OK] Features normalized (MinMaxScaler)")

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Train: {len(X_train)} | Val: {len(X_val)}")

# ─────────────────────────────────────────────────────────────────────────
# TRAIN LSTM - MULTI-ASSET
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Training LSTM...")
print("-"*80)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Optimized for multi-asset data (more features, more samples)
inputs = tf.keras.Input(shape=(len(feature_cols),))

x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.15)(x)

x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.1)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model_lstm = tf.keras.Model(inputs=inputs, outputs=outputs)

model_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("Training...")
history = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_auc', patience=20, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)
    ],
    verbose=0
)

lstm_acc = model_lstm.evaluate(X_val, y_val, verbose=0)[1]
lstm_auc = model_lstm.evaluate(X_val, y_val, verbose=0)[2]

print(f"[OK] LSTM Training Complete:")
print(f"  Validation Accuracy: {lstm_acc:.1%}")
print(f"  Validation AUC: {lstm_auc:.4f}")
print(f"  Epochs: {len(history.history['loss'])}")

model_lstm.save(models_dir / 'best_model_lstm_multiasset.h5')
print(f"[OK] Saved: best_model_lstm_multiasset.h5")

# ─────────────────────────────────────────────────────────────────────────
# TRAIN XGBOOST - MULTI-ASSET
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Training XGBoost...")
print("-"*80)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

model_xgb = GradientBoostingClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.8,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    verbose=0
)

print("Training...")
model_xgb.fit(X_train, y_train)

xgb_acc = model_xgb.score(X_val, y_val)
cv_scores = cross_val_score(model_xgb, X_scaled, y, cv=5, scoring='roc_auc')

print(f"[OK] XGBoost Training Complete:")
print(f"  Validation Accuracy: {xgb_acc:.1%}")
print(f"  Cross-val AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

with open(models_dir / 'best_model_xgb_multiasset.pkl', 'wb') as f:
    pickle.dump(model_xgb, f)

print(f"[OK] Saved: best_model_xgb_multiasset.pkl")

# ─────────────────────────────────────────────────────────────────────────
# SAVE METADATA
# ─────────────────────────────────────────────────────────────────────────

model_metadata = {
    'timestamp': datetime.now().isoformat(),
    'phase': 3,
    'approach': 'Multi-Asset Enriched (XAUUSD + cross-asset correlations)',
    'asset': 'XAUUSD',
    'enriched_with': ['EURUSD', 'GBPUSD', 'USDJPY', 'US10Y'],
    'models': {
        'lstm': {
            'type': 'Deep Neural Network (Multi-Asset)',
            'accuracy': float(lstm_acc),
            'auc': float(lstm_auc),
            'epochs_trained': len(history.history['loss']),
            'file': 'best_model_lstm_multiasset.h5'
        },
        'xgboost': {
            'type': 'GradientBoostingClassifier (Multi-Asset)',
            'accuracy': float(xgb_acc),
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'n_estimators': 400,
            'max_depth': 6,
            'file': 'best_model_xgb_multiasset.pkl'
        }
    },
    'ensemble': {
        'lstm_weight': 0.4,
        'xgboost_weight': 0.6,
        'expected_ensemble_auc': float(lstm_auc * 0.4 + cv_scores.mean() * 0.6)
    },
    'data': {
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'feature_count': len(feature_cols),
        'class_distribution_up': float(y.mean()),
        'period': f'{features_train.index[0]} to {features_train.index[-1]}'
    }
}

with open(models_dir / 'metadata_multiasset.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"[OK] Metadata saved: metadata_multiasset.json")

# Summary
print("\n" + "="*80)
print("PHASE 3 COMPLETE - MULTI-ASSET TRAINING")
print("="*80)

print(f"\n[RESULTS] Trained on {len(X)} XAUUSD samples enriched with 4 forex pairs:")
print(f"  LSTM:     {lstm_acc:.1%} accuracy | AUC: {lstm_auc:.4f}")
print(f"  XGBoost:  {xgb_acc:.1%} accuracy | CV AUC: {cv_scores.mean():.4f}")
print(f"  Ensemble: {(lstm_auc * 0.4 + cv_scores.mean() * 0.6):.4f} AUC")

print(f"\n[FILES] Saved:")
print(f"  best_model_lstm_multiasset.h5")
print(f"  best_model_xgb_multiasset.pkl")
print(f"  scaler_multiasset.pkl (45 features)")

print("\n[NEXT] Compare to single-asset models to see improvement")
print("="*80)
