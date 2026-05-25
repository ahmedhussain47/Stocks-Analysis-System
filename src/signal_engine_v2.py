"""
Signal Engine v2 — Peak Accuracy ML Trading System.

Integrates:
- XGBoost + LSTM + 1D-CNN ensemble predictions
- Rule-based chart pattern detection
- Confidence scoring with pattern bonuses
- Multi-timeframe context
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys

# Fix imports for different execution contexts
try:
    from src.model_loader import ModelBundle
    from src.pattern_detector import PatternDetector, PatternMatch
except ImportError:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.model_loader import ModelBundle
    from src.pattern_detector import PatternDetector, PatternMatch

@dataclass
class Signal:
    """Trading signal with full context."""
    symbol: str
    timeframe: str
    direction: str  # 'BUY', 'SELL', 'FLAT'
    confidence: float  # 0-1
    xgb_prob: float
    lstm_prob: float
    cnn_prob: float
    ensemble_prob: Optional[float]
    pattern_detected: Optional[str] = None
    pattern_confidence: float = 0.0
    combined_confidence: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class SignalEngineV2:
    """Peak accuracy ML signal generator."""

    def __init__(
        self,
        symbol: str = 'XAUUSD',
        models_dir: Optional[Path] = None,
        use_patterns: bool = True,
        pattern_bonus: float = 0.15,  # +15% confidence if pattern aligns
    ):
        """
        Args:
            symbol: Trading pair
            models_dir: Path to trained models
            use_patterns: Enable rule-based pattern detection
            pattern_bonus: Confidence boost if pattern matches signal direction
        """
        self.symbol = symbol
        self.models_dir = models_dir or Path(__file__).parent.parent / 'training' / 'models'
        self.use_patterns = use_patterns
        self.pattern_bonus = pattern_bonus

        # Load model bundles per timeframe
        self.models = {}
        self.pattern_detector = PatternDetector(min_bars=10) if use_patterns else None

        for tf in ['1D', '4H', '1H', '15min']:
            try:
                self.models[tf] = ModelBundle(tf, self.models_dir)
            except Exception as e:
                print(f"[WARN] Could not load models for {tf}: {e}")

    def generate_signal(self, df: pd.DataFrame, timeframe: str) -> Signal:
        """
        Generate trading signal from recent OHLCV data.

        Args:
            df: OHLCV DataFrame (last row = current bar)
            timeframe: '1D', '4H', '1H', or '15min'

        Returns:
            Signal object with all predictions
        """
        if timeframe not in self.models:
            return Signal(
                symbol=self.symbol,
                timeframe=timeframe,
                direction='FLAT',
                confidence=0.0,
                xgb_prob=0.5,
                lstm_prob=0.5,
                cnn_prob=0.5,
                ensemble_prob=0.5,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
            )

        # Get ML predictions
        ml_pred = self.models[timeframe].predict(df)

        # Detect patterns if enabled
        pattern_match = None
        pattern_bonus = 0.0

        if self.use_patterns:
            patterns = self.pattern_detector.detect_all(df, lookback=50)
            if patterns:
                # Use highest confidence pattern
                pattern_match = max(patterns, key=lambda p: p.confidence)

                # Apply bonus if pattern direction matches ML signal
                if (ml_pred['signal'] == 'UP' and pattern_match.direction == 'UP') or \
                   (ml_pred['signal'] == 'DOWN' and pattern_match.direction == 'DOWN'):
                    pattern_bonus = self.pattern_bonus * pattern_match.confidence

        # Combine confidences
        combined_confidence = min(1.0, ml_pred['confidence'] + pattern_bonus)

        # Adjust signal based on combined confidence
        if combined_confidence < 0.4:
            final_direction = 'FLAT'
        elif ml_pred['signal'] == 'UP':
            final_direction = 'BUY'
        elif ml_pred['signal'] == 'DOWN':
            final_direction = 'SELL'
        else:
            final_direction = 'FLAT'

        # Calculate risk management levels
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df, period=14)

        if final_direction == 'BUY':
            entry = current_price
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2.5)
        elif final_direction == 'SELL':
            entry = current_price
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 2.5)
        else:
            entry = None
            stop_loss = None
            take_profit = None

        return Signal(
            symbol=self.symbol,
            timeframe=timeframe,
            direction=final_direction,
            confidence=ml_pred['confidence'],
            xgb_prob=ml_pred['xgb_prob'],
            lstm_prob=ml_pred['lstm_prob'],
            cnn_prob=ml_pred['cnn_prob'],
            ensemble_prob=ml_pred.get('ensemble_prob'),
            pattern_detected=pattern_match.pattern if pattern_match else None,
            pattern_confidence=pattern_match.confidence if pattern_match else 0.0,
            combined_confidence=combined_confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def generate_signals_multiframe(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate signals across multiple timeframes.

        Args:
            dfs: {'1D': df, '4H': df, '1H': df, '15min': df}

        Returns:
            {'1D': Signal, ...}
        """
        signals = {}
        for tf, df in dfs.items():
            if df is not None and len(df) > 0:
                signals[tf] = self.generate_signal(df, tf)
        return signals

    def aggregate_signals(self, signals: Dict[str, Signal]) -> Dict:
        """
        Aggregate multi-timeframe signals into consolidated recommendation.

        Strategy:
        - Higher TFs (1D, 4H) have more weight
        - All TFs must agree on direction for strong signal
        - Confidence weighted by timeframe importance
        """
        if not signals:
            return {
                'direction': 'FLAT',
                'confidence': 0.0,
                'consensus': 'No data',
            }

        # Weights: 1D=0.4, 4H=0.3, 1H=0.2, 15min=0.1
        weights = {'1D': 0.4, '4H': 0.3, '1H': 0.2, '15min': 0.1}

        weighted_conf = {}
        total_weight = 0

        for tf, signal in signals.items():
            if tf in weights and signal.direction != 'FLAT':
                weight = weights[tf]
                weighted_conf[signal.direction] = weighted_conf.get(signal.direction, 0) + \
                                                   signal.combined_confidence * weight
                total_weight += weight

        if total_weight == 0:
            return {'direction': 'FLAT', 'confidence': 0.0, 'consensus': 'No consensus'}

        # Find direction with highest weighted confidence
        best_direction = max(weighted_conf, key=weighted_conf.get)
        best_confidence = weighted_conf[best_direction] / total_weight

        # Check consensus (all signals point same direction)
        directions = [s.direction for s in signals.values() if s.direction != 'FLAT']
        consensus = len(set(directions)) == 1 and len(directions) > 0

        return {
            'direction': best_direction,
            'confidence': best_confidence,
            'consensus': 'All TFs agree' if consensus else 'Mixed signals',
            'signal_count': len(signals),
            'buy_count': len([s for s in signals.values() if s.direction == 'BUY']),
            'sell_count': len([s for s in signals.values() if s.direction == 'SELL']),
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return atr if not np.isnan(atr) else 0.0

    def format_signal(self, signal: Signal) -> str:
        """Format signal for display/logging."""
        lines = [
            f"{'='*70}",
            f"SIGNAL: {signal.symbol} @ {signal.timeframe}",
            f"{'='*70}",
            f"Direction:          {signal.direction}",
            f"Combined Confidence: {signal.combined_confidence:.2%}",
            f"ML Confidence:       {signal.confidence:.2%}",
            f"",
            f"Model Probabilities:",
            f"  XGBoost: {signal.xgb_prob:.2%}",
            f"  LSTM:    {signal.lstm_prob:.2%}",
            f"  CNN:     {signal.cnn_prob:.2%}",
            f"  Ensemble:{signal.ensemble_prob:.2%}" if signal.ensemble_prob else "",
            f"",
        ]

        if signal.pattern_detected:
            lines.extend([
                f"Pattern:             {signal.pattern_detected}",
                f"Pattern Confidence:  {signal.pattern_confidence:.2%}",
                f"",
            ])

        if signal.entry_price:
            lines.extend([
                f"Risk Management:",
                f"  Entry:  ${signal.entry_price:,.2f}",
                f"  SL:     ${signal.stop_loss:,.2f}",
                f"  TP:     ${signal.take_profit:,.2f}",
                f"  RR:     1:{abs((signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)):.2f}",
            ])

        return "\n".join([l for l in lines if l])
