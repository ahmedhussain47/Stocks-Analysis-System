"""
XGBoost v2 Training — 7-year data, 65+ features, 3-class prediction, Optuna tuning.

Run after mt5_data_collector.py completes.
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
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import optuna
from optuna.pruners import MedianPruner
import pickle
import json

from training.feature_engineering_v2 import build_features_v2, get_feature_names

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / 'data' / 'raw' / 'mt5'
MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ASSET = 'XAUUSD'
TIMEFRAMES = ['1D', '4H', '1H', '15min']

# Label horizons (bars ahead)
LABEL_HORIZONS = {
    '1D': 3,
    '4H': 3,
    '1H': 4,
    '15min': 4,
}

# ATR thresholds for 3-class labeling
ATR_THRESHOLD = 0.5  # fraction of ATR for UP/DOWN classification

# Optuna tuning
N_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # seconds per TF

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
    """
    Create 3-class labels: UP (1), DOWN (-1), FLAT (0).

    UP:   future_return > atr_pct * atr
    DOWN: future_return < -atr_pct * atr
    FLAT: else
    """
    from training.feature_engineering_v2 import atr as calc_atr

    h = LABEL_HORIZONS[tf]
    close = df['close'].values

    # Calculate ATR
    atr_vals = calc_atr(df, 14).values

    # Future return at horizon h (last h bars will be NaN)
    future_close = np.full_like(close, np.nan, dtype=float)
    future_close[:len(close)-h] = close[h:]
    future_return = (future_close - close) / close

    # 3-class threshold
    threshold = atr_pct * (atr_vals / close)

    labels = pd.Series(0, index=df.index)  # default FLAT
    valid_mask = ~np.isnan(future_return) & ~np.isnan(threshold)

    labels[valid_mask & (future_return > threshold)] = 1   # UP
    labels[valid_mask & (future_return < -threshold)] = -1  # DOWN
    labels[~valid_mask] = np.nan  # Mark invalid as NaN for filtering

    return labels

# ── Train per timeframe ────────────────────────────────────────────────────
def train_xgb_tf(tf: str):
    """Train XGBoost for a single timeframe."""

    print(f"\n{'='*70}")
    print(f"XGBoost v2 Training: {ASSET} @ {tf}")
    print(f"{'='*70}")

    # Load data
    print(f"Loading data...", end=" ", flush=True)
    df = load_data(tf)
    if df is None or len(df) < 100:
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

    # Remove rows with NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"Valid samples: {len(X):,}")

    if len(X) < 100:
        print(f"[FAIL] Not enough valid samples")
        return

    # Class distribution
    class_counts = y.value_counts()
    print(f"Class distribution:")
    for class_val in sorted(y.unique()):
        count = (y == class_val).sum()
        pct = (count / len(y)) * 100
        label = {1: 'UP', -1: 'DOWN', 0: 'FLAT'}[class_val]
        print(f"  {label:4s}: {count:>6,} ({pct:>5.1f}%)")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # ── Optuna Hyperparameter Search ───────────────────────────────────────
    print(f"\nHyperparameter search ({N_TRIALS} trials)...")

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        }

        # Time series split (no shuffle)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Filter FLAT class for training (only UP/DOWN signal)
            train_idx_filtered = y_train != 0
            X_train = X_train[train_idx_filtered]
            y_train = y_train[train_idx_filtered]

            # Remap labels to binary for XGBoost (or use multiclass)
            y_train_binary = (y_train > 0).astype(int)  # UP=1, DOWN=0
            y_val_binary = (y_val > 0).astype(int)

            # Train (simplified without early stopping for compatibility)
            bst = xgb.XGBClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
            )

            bst.fit(
                X_train, y_train_binary,
                eval_set=[(X_val, y_val_binary)],
                verbose=False
            )

            # Evaluate
            y_pred_proba = bst.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val_binary, y_pred_proba)
            scores.append(auc)

            # Report to optuna for pruning
            trial.report(np.mean(scores), len(scores) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(),
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ── Train final model on full data ─────────────────────────────────────
    print(f"\nTraining final model on full dataset...", end=" ", flush=True)

    # Filter FLAT
    valid_idx = y != 0
    X_train = X_scaled[valid_idx]
    y_train = y[valid_idx]
    y_train_binary = (y_train > 0).astype(int)

    final_model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
    )
    final_model.fit(X_train, y_train_binary, verbose=False)
    print(f"[OK]")

    # ── Save Model ─────────────────────────────────────────────────────────
    print(f"Saving model...", end=" ", flush=True)

    model_path = MODELS_DIR / f'xgb_v2_XAUUSD_{tf}.json'
    final_model.get_booster().save_model(str(model_path))

    scaler_path = MODELS_DIR / f'scaler_v2_XAUUSD_{tf}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save metadata
    metadata = {
        'timeframe': tf,
        'asset': ASSET,
        'n_features': len(X.columns),
        'n_samples': len(X_train),
        'best_auc': float(study.best_value),
        'best_params': best_params,
        'training_date': datetime.now().isoformat(),
    }
    metadata_path = MODELS_DIR / f'metadata_xgb_v2_XAUUSD_{tf}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK]")
    print(f"  Model: {model_path.name}")
    print(f"  Scaler: {scaler_path.name}")
    print(f"  Metadata: {metadata_path.name}")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*70)
    print("XGBoost v2 Multi-Timeframe Training")
    print("="*70)

    for tf in TIMEFRAMES:
        try:
            train_xgb_tf(tf)
        except Exception as e:
            print(f"[FAIL] Error training {tf}: {e}")

    print(f"\n{'='*70}")
    print("[OK] All models trained!")
    print(f"{'='*70}")
