"""
Model Loader — Load and predict with trained ensemble models.

Loads XGBoost, LSTM, and 1D-CNN per timeframe, returns ensemble predictions.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import sys
import xgboost as xgb
import tensorflow as tf

# Import feature builder
from src.feature_engineering import build_features_v2

from src.pattern_detector import PatternDetector

class ModelBundle:
    """Bundle of base models + meta-learner for a timeframe."""

    def __init__(self, tf: str, models_dir: Path):
        """
        Args:
            tf: Timeframe ('1D', '4H', '1H', '15min')
            models_dir: Path to trained models directory
        """
        self.tf = tf
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.meta_learner = None
        self._load_models()

    def _load_models(self):
        """Load all available base models."""
        # XGBoost
        xgb_model_path = self.models_dir / f'xgb_v2_XAUUSD_{self.tf}.json'
        xgb_scaler_path = self.models_dir / f'scaler_v2_XAUUSD_{self.tf}.pkl'

        if xgb_model_path.exists():
            try:
                self.models['xgb'] = xgb.Booster(model_file=str(xgb_model_path))
                self.scalers['xgb'] = pickle.load(open(xgb_scaler_path, 'rb'))
            except Exception as e:
                print(f"[WARN] Failed to load XGBoost model: {e}")

        # LSTM
        lstm_model_path = self.models_dir / f'lstm_att_XAUUSD_{self.tf}.h5'
        lstm_scaler_path = self.models_dir / f'scaler_lstm_att_XAUUSD_{self.tf}.pkl'

        if lstm_model_path.exists():
            try:
                self.models['lstm'] = tf.keras.models.load_model(str(lstm_model_path))
                self.scalers['lstm'] = pickle.load(open(lstm_scaler_path, 'rb'))
            except Exception as e:
                print(f"[WARN] Failed to load LSTM model: {e}")

        # 1D-CNN (only for short timeframes)
        if self.tf in ['15min', '1H']:
            cnn_model_path = self.models_dir / f'cnn_1d_XAUUSD_{self.tf}.h5'
            cnn_scaler_path = self.models_dir / f'scaler_cnn_1d_XAUUSD_{self.tf}.pkl'

            if cnn_model_path.exists():
                try:
                    self.models['cnn'] = tf.keras.models.load_model(str(cnn_model_path))
                    self.scalers['cnn'] = pickle.load(open(cnn_scaler_path, 'rb'))
                except Exception as e:
                    print(f"[WARN] Failed to load 1D-CNN model: {e}")

        # Ensemble meta-learner
        ensemble_path = self.models_dir / f'ensemble_meta_XAUUSD_{self.tf}.pkl'
        if ensemble_path.exists():
            try:
                self.meta_learner = pickle.load(open(ensemble_path, 'rb'))
            except Exception as e:
                print(f"[WARN] Failed to load ensemble meta-learner: {e}")

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate predictions from all models.

        Args:
            df: Recent OHLCV data (last row is current)

        Returns:
            {
                'signal': 'UP' | 'DOWN' | 'FLAT',
                'confidence': float (0-1),
                'xgb_prob': float,
                'lstm_prob': float,
                'cnn_prob': float (if available),
            }
        """
        if len(df) < 60:  # Need at least LSTM lookback
            return {
                'signal': 'FLAT',
                'confidence': 0.0,
                'xgb_prob': 0.5,
                'lstm_prob': 0.5,
                'cnn_prob': 0.5,
            }

        # Build features for current bar
        X = build_features_v2(df, self.tf)
        X_latest = X.iloc[-1:].values

        predictions = {}

        # XGBoost prediction
        if 'xgb' in self.models:
            try:
                X_xgb = self.scalers['xgb'].transform(X_latest)
                dmatrix = xgb.DMatrix(X_xgb)
                proba = self.models['xgb'].predict(dmatrix)[0]
                # proba is [prob_down, prob_flat, prob_up] or [prob_down, prob_up]
                if len(proba) == 3:
                    predictions['xgb'] = proba[2]  # prob UP
                else:
                    predictions['xgb'] = proba[1] if len(proba) > 1 else 0.5
            except Exception as e:
                predictions['xgb'] = 0.5

        # LSTM prediction (needs sequence)
        if 'lstm' in self.models:
            try:
                lookback = 60
                if len(X) >= lookback:
                    X_lstm = X.iloc[-lookback:].values
                    X_lstm = self.scalers['lstm'].transform(X_lstm)
                    X_lstm = X_lstm.reshape(1, lookback, X_lstm.shape[1])
                    proba = self.models['lstm'].predict(X_lstm, verbose=0)[0]
                    predictions['lstm'] = proba[1] if len(proba) > 1 else 0.5  # prob UP (index 1)
                else:
                    predictions['lstm'] = 0.5
            except Exception as e:
                predictions['lstm'] = 0.5

        # 1D-CNN prediction (needs 30-bar OHLCV sequence)
        if 'cnn' in self.models and self.tf in ['15min', '1H']:
            try:
                lookback = 30
                if len(df) >= lookback:
                    # Get raw OHLCV
                    ohlcv = df[['open', 'high', 'low', 'close', 'volume']].iloc[-lookback:].values

                    # Normalize each column separately using stored scalers
                    scalers_dict = self.scalers['cnn']
                    ohlcv_normalized = ohlcv.copy()
                    for i, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
                        if col in scalers_dict:
                            ohlcv_normalized[:, i] = scalers_dict[col].transform(
                                ohlcv[:, i].reshape(-1, 1)
                            ).flatten()

                    X_cnn = ohlcv_normalized.reshape(1, lookback, 5)
                    proba = self.models['cnn'].predict(X_cnn, verbose=0)[0]
                    predictions['cnn'] = proba[1] if len(proba) > 1 else 0.5
                else:
                    predictions['cnn'] = 0.5
            except Exception as e:
                predictions['cnn'] = 0.5

        # Generate meta-features if meta-learner available
        if self.meta_learner and len(predictions) >= 2:
            try:
                meta_features = []
                for model_name in ['xgb', 'lstm', 'cnn']:
                    if model_name in predictions:
                        prob = predictions[model_name]
                        meta_features.extend([1 - prob, 0, prob])  # [prob_down, prob_flat, prob_up]

                if len(meta_features) >= 6:
                    meta_features = np.array(meta_features).reshape(1, -1)
                    ensemble_prob_up = self.meta_learner.predict_proba(meta_features)[0][1]
                    predictions['ensemble'] = ensemble_prob_up
            except Exception as e:
                predictions['ensemble'] = np.mean([p for p in predictions.values()])

        # Average available predictions
        if not predictions:
            return {
                'signal': 'FLAT',
                'confidence': 0.0,
                'xgb_prob': 0.5,
                'lstm_prob': 0.5,
                'cnn_prob': 0.5,
            }

        avg_prob = np.mean(list(predictions.values()))
        confidence = abs(avg_prob - 0.5) * 2  # 0.5 confidence when prob=0.5, 1.0 when prob=0.0 or 1.0

        signal = 'UP' if avg_prob > 0.6 else ('DOWN' if avg_prob < 0.4 else 'FLAT')

        return {
            'signal': signal,
            'confidence': confidence,
            'xgb_prob': predictions.get('xgb', 0.5),
            'lstm_prob': predictions.get('lstm', 0.5),
            'cnn_prob': predictions.get('cnn', 0.5),
            'ensemble_prob': predictions.get('ensemble', avg_prob),
        }
