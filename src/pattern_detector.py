"""
Rule-Based Pattern Detector — Identify Bull Flag, Bear Flag, H&S, etc.

Returns confidence scores for each detected pattern.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PatternMatch:
    """Detected pattern with metadata."""
    pattern: str
    confidence: float
    start_idx: int
    end_idx: int
    target_price: Optional[float] = None
    direction: str = 'NEUTRAL'  # UP, DOWN, NEUTRAL

class PatternDetector:
    """Detect classic chart patterns from OHLC data."""

    def __init__(self, min_bars: int = 10):
        """
        Args:
            min_bars: Minimum bars required for a valid pattern
        """
        self.min_bars = min_bars

    def detect_all(self, df: pd.DataFrame, lookback: int = 50) -> List[PatternMatch]:
        """
        Detect all patterns in the lookback window.

        Args:
            df: OHLCV DataFrame with timestamp index
            lookback: Number of bars to scan

        Returns:
            List of PatternMatch objects
        """
        if len(df) < lookback:
            return []

        df_window = df.iloc[-lookback:].copy()
        high = df_window['high'].values
        low = df_window['low'].values
        close = df_window['close'].values

        patterns = []

        # Detect each pattern type
        bull_flag = self._detect_bull_flag(high, low, close)
        if bull_flag:
            patterns.append(bull_flag)

        bear_flag = self._detect_bear_flag(high, low, close)
        if bear_flag:
            patterns.append(bear_flag)

        head_shoulders = self._detect_head_shoulders(high, low, close)
        if head_shoulders:
            patterns.append(head_shoulders)

        double_top = self._detect_double_top(high, low, close)
        if double_top:
            patterns.append(double_top)

        double_bottom = self._detect_double_bottom(high, low, close)
        if double_bottom:
            patterns.append(double_bottom)

        return patterns

    def _detect_bull_flag(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[PatternMatch]:
        """
        Bull Flag: Strong uptrend, consolidation, breakout up.

        Pattern:
        1. Pole: strong upward movement (>1.5% of range in 3-5 bars)
        2. Flag: downward/sideways consolidation (2-5 bars, lower than pole high)
        3. Breakout: close above pole high
        """
        if len(high) < 10:
            return None

        # Last 10 bars: look for pole + flag + breakout
        recent = close[-10:].copy()
        recent_high = high[-10:].copy()
        recent_low = low[-10:].copy()

        # Check for uptrend at start (pole)
        pole_return = (recent[3] - recent[0]) / recent[0]
        if pole_return < 0.015:  # At least 1.5% gain
            return None

        # Check for consolidation (flag)
        flag_high = recent_high[3:7].max()
        flag_low = recent_low[3:7].min()
        flag_range = (flag_high - flag_low) / flag_low

        if flag_range > 0.02:  # Flag shouldn't be too wide
            return None

        # Check for breakout
        breakout = recent[-1] > flag_high

        if not breakout:
            return None

        confidence = min(0.9, 0.6 + abs(pole_return) * 10)
        target = flag_high * 1.02  # 2% above breakout

        return PatternMatch(
            pattern='bull_flag',
            confidence=confidence,
            start_idx=0,
            end_idx=9,
            target_price=target,
            direction='UP'
        )

    def _detect_bear_flag(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[PatternMatch]:
        """
        Bear Flag: Strong downtrend, consolidation, breakout down.
        """
        if len(high) < 10:
            return None

        recent = close[-10:].copy()
        recent_high = high[-10:].copy()
        recent_low = low[-10:].copy()

        # Downtrend pole
        pole_return = (recent[3] - recent[0]) / recent[0]
        if pole_return > -0.015:  # At least 1.5% loss
            return None

        # Consolidation flag
        flag_high = recent_high[3:7].max()
        flag_low = recent_low[3:7].min()
        flag_range = (flag_high - flag_low) / flag_low

        if flag_range > 0.02:
            return None

        # Breakout down
        breakout = recent[-1] < flag_low

        if not breakout:
            return None

        confidence = min(0.9, 0.6 + abs(pole_return) * 10)
        target = flag_low * 0.98  # 2% below breakout

        return PatternMatch(
            pattern='bear_flag',
            confidence=confidence,
            start_idx=0,
            end_idx=9,
            target_price=target,
            direction='DOWN'
        )

    def _detect_head_shoulders(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[PatternMatch]:
        """
        Head & Shoulders: Left shoulder, head (higher), right shoulder (lower).
        Neckline breakout signals reversal.
        """
        if len(high) < 15:
            return None

        recent_high = high[-15:].copy()

        # Find local maxima (potential shoulders and head)
        peaks = []
        for i in range(1, len(recent_high) - 1):
            if recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i+1]:
                peaks.append((i, recent_high[i]))

        if len(peaks) < 3:
            return None

        # Check pattern: left shoulder < head > right shoulder
        if len(peaks) >= 3:
            left_idx, left_h = peaks[0]
            head_idx, head_h = peaks[1]
            right_idx, right_h = peaks[2]

            if head_h > left_h and head_h > right_h and left_h > right_h * 0.95:
                # Valid H&S shape
                confidence = 0.75
                target = low[-15:].min() * 0.98

                return PatternMatch(
                    pattern='head_shoulders',
                    confidence=confidence,
                    start_idx=0,
                    end_idx=14,
                    target_price=target,
                    direction='DOWN'
                )

        return None

    def _detect_double_top(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[PatternMatch]:
        """
        Double Top: Two peaks at similar height, valley between them.
        """
        if len(high) < 10:
            return None

        recent_high = high[-10:].copy()

        # Find peaks
        peaks = []
        for i in range(1, len(recent_high) - 1):
            if recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i+1]:
                peaks.append((i, recent_high[i]))

        if len(peaks) < 2:
            return None

        # Check if two peaks are close in height
        for i in range(len(peaks) - 1):
            idx1, h1 = peaks[i]
            idx2, h2 = peaks[i + 1]

            price_diff = abs(h1 - h2) / min(h1, h2)
            if price_diff < 0.01:  # Within 1% of each other
                confidence = 0.8
                valley = low[idx1:idx2].min()
                target = valley * 0.98

                return PatternMatch(
                    pattern='double_top',
                    confidence=confidence,
                    start_idx=0,
                    end_idx=9,
                    target_price=target,
                    direction='DOWN'
                )

        return None

    def _detect_double_bottom(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[PatternMatch]:
        """
        Double Bottom: Two valleys at similar depth, peak between them.
        """
        if len(low) < 10:
            return None

        recent_low = low[-10:].copy()

        # Find valleys
        valleys = []
        for i in range(1, len(recent_low) - 1):
            if recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i+1]:
                valleys.append((i, recent_low[i]))

        if len(valleys) < 2:
            return None

        # Check if two valleys are close in depth
        for i in range(len(valleys) - 1):
            idx1, l1 = valleys[i]
            idx2, l2 = valleys[i + 1]

            price_diff = abs(l1 - l2) / min(l1, l2)
            if price_diff < 0.01:  # Within 1% of each other
                confidence = 0.8
                peak = high[idx1:idx2].max()
                target = peak * 1.02

                return PatternMatch(
                    pattern='double_bottom',
                    confidence=confidence,
                    start_idx=0,
                    end_idx=9,
                    target_price=target,
                    direction='UP'
                )

        return None
