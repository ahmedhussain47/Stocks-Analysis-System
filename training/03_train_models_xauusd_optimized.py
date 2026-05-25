"""
PHASE 3: MODEL TRAINING - XAUUSD OPTIMIZED
=============================================

Train on 1,415 XAUUSD samples (5+ years) with optimized hyperparameters.
Target: 75%+ accuracy for both LSTM and XGBoost.
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
print("PHASE 3: MODEL TRAINING - XAUUSD OPTIMIZED")
print("="*80)

# Load processed data
processed_dir = Path('training/data/processed')
models_dir = Path('training/models')
models_dir.mkdir(parents=True, exist_ok=True)

print("\nLoading XAUUSD data...")
features_train = pd.read_csv(processed_dir / 'features_train.csv', index_col=0, parse_dates=True)
print(f"[OK] Loaded {len(features_train):,} samples from {features_train.index[0].date()} to {features_train.index[-1].date()}")

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

print(f"[OK] MinMaxScaler fitted on {len(feature_cols)} features")

# ─────────────────────────────────────────────────────────────────────────
# TRAIN LSTM - OPTIMIZED FOR XAUUSD
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Training LSTM (Deep Neural Network)...")
print("-"*80)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Train: {len(X_train)} | Val: {len(X_val)}")

# Build LSTM - optimized for 1,415 samples
inputs = tf.keras.Input(shape=(len(feature_cols),))

# Deep narrow architecture (regularized)
x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.15)(x)

x = layers.Dense(16, activation='relu')(x)
x = layers.Dropout(0.1)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model_lstm = tf.keras.Model(inputs=inputs, outputs=outputs)

model_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("Model architecture:")
model_lstm.summary()

# Train with aggressive early stopping
print("\nTraining...")
history = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[
        EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ],
    verbose=0
)

# Evaluate
lstm_acc = model_lstm.evaluate(X_val, y_val, verbose=0)[1]
lstm_auc = model_lstm.evaluate(X_val, y_val, verbose=0)[2]

print(f"\n[OK] LSTM Training Complete:")
print(f"  Validation Accuracy: {lstm_acc:.1%}")
print(f"  Validation AUC: {lstm_auc:.4f}")
print(f"  Epochs trained: {len(history.history['loss'])}")

# Save model
model_lstm.save(models_dir / 'best_model_lstm.h5')
print(f"[OK] Model saved: best_model_lstm.h5")

# ─────────────────────────────────────────────────────────────────────────
# TRAIN XGBOOST - OPTIMIZED FOR XAUUSD
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("Training XGBoost...")
print("-"*80)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

model_xgb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    verbose=0
)

print("Training...")
model_xgb.fit(X_train, y_train)

xgb_acc = model_xgb.score(X_val, y_val)

# Cross-validation
cv_scores = cross_val_score(model_xgb, X_scaled, y, cv=5, scoring='roc_auc')

print(f"\n[OK] XGBoost Training Complete:")
print(f"  Validation Accuracy: {xgb_acc:.1%}")
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
    'asset': 'XAUUSD',
    'data_period': f'{features_train.index[0].date()} to {features_train.index[-1].date()}',
    'models': {
        'lstm': {
            'type': 'Deep Neural Network',
            'accuracy': float(lstm_acc),
            'auc': float(lstm_auc),
            'epochs_trained': len(history.history['loss']),
            'file': 'best_model_lstm.h5'
        },
        'xgboost': {
            'type': 'GradientBoostingClassifier',
            'accuracy': float(xgb_acc),
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'n_estimators': 300,
            'max_depth': 5,
            'file': 'best_model_xgb.pkl'
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
        'class_distribution_up': float(y.mean())
    }
}

with open(models_dir / 'metadata_phase3.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"[OK] Metadata saved: metadata_phase3.json")

# ─────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PHASE 3 COMPLETE - XAUUSD OPTIMIZED TRAINING")
print("="*80)

print(f"\n[RESULTS] Models Trained on {len(X)} XAUUSD samples (5+ years):")
print(f"  LSTM:     {lstm_acc:.1%} accuracy | AUC: {lstm_auc:.4f}")
print(f"  XGBoost:  {xgb_acc:.1%} accuracy | CV AUC: {cv_scores.mean():.4f}")
print(f"  Ensemble: {(lstm_auc * 0.4 + cv_scores.mean() * 0.6):.4f} AUC (estimated)")

print(f"\n[FILES] saved:")
print(f"  [OK] best_model_lstm.h5")
print(f"  [OK] best_model_xgb.pkl")
print(f"  [OK] scaler.pkl (42 features)")
print(f"  [OK] metadata_phase3.json")

print("\n" + "="*80)
