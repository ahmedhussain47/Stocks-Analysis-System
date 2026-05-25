"""
Demo — Complete Signal Generation Workflow

Shows how to use:
- MT5 live data
- Feature engineering
- Model predictions
- Pattern detection
- Signal generation with confidence
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.mt5_trader import MT5Trader
from src.signal_engine_v2 import SignalEngineV2

# ── Configuration ──────────────────────────────────────────────────────────
MT5_LOGIN = 5050913403
MT5_PASS = "Ahmed@477447"
MT5_SERVER = "MetaQuotes-Demo"

ASSET = 'XAUUSD'
TIMEFRAMES = ['1D', '4H', '1H', '15min']

# ── Main Demo ──────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print(f"Peak Accuracy ML Trading System — Signal Generation Demo")
    print("="*70)

    # Initialize MT5 connection
    print(f"\n[1/4] Connecting to MT5...")
    trader = MT5Trader(
        login=MT5_LOGIN,
        password=MT5_PASS,
        server=MT5_SERVER,
        demo_mode=True
    )

    if not trader.connect():
        print("[FAIL] MT5 connection failed")
        return

    print("[OK] MT5 Connected")

    # Fetch live data for all timeframes
    print(f"\n[2/4] Fetching live data for {ASSET}...")
    dfs = {}
    for tf in TIMEFRAMES:
        print(f"  {tf:6s}...", end=" ", flush=True)
        try:
            # Fetch enough bars for LSTM lookback (60 bars)
            bar_counts = {
                '1D': 100,
                '4H': 100,
                '1H': 100,
                '15min': 100,
            }
            df = trader.fetch_ohlcv(
                symbol=ASSET,
                timeframe=tf,
                bars=bar_counts.get(tf, 100)
            )

            if not df.empty:
                dfs[tf] = df
                print(f"[OK] {len(df):,} bars")
            else:
                print(f"[FAIL] No data")
        except Exception as e:
            print(f"[FAIL] {e}")

    trader.disconnect()

    if not dfs:
        print("[FAIL] Could not fetch any data")
        return

    # Initialize signal engine
    print(f"\n[3/4] Initializing signal engine...")
    try:
        engine = SignalEngineV2(
            symbol=ASSET,
            use_patterns=True,
            pattern_bonus=0.15
        )
        print("[OK] Signal engine ready")
    except Exception as e:
        print(f"[FAIL] Could not initialize engine: {e}")
        return

    # Generate signals
    print(f"\n[4/4] Generating signals across timeframes...")
    signals = {}
    for tf in TIMEFRAMES:
        if tf in dfs:
            print(f"  {tf:6s}...", end=" ", flush=True)
            try:
                signal = engine.generate_signal(dfs[tf], tf)
                signals[tf] = signal
                print(f"[OK] {signal.direction} ({signal.combined_confidence:.0%})")
            except Exception as e:
                print(f"[FAIL] {e}")

    # Aggregate signals
    print(f"\n{'='*70}")
    print("Aggregated Signal:")
    print(f"{'='*70}")

    aggregate = engine.aggregate_signals(signals)
    print(f"Direction:  {aggregate['direction']}")
    print(f"Confidence: {aggregate['confidence']:.0%}")
    print(f"Consensus:  {aggregate['consensus']}")
    print(f"Buy Count:  {aggregate['buy_count']}")
    print(f"Sell Count: {aggregate['sell_count']}")

    # Detailed signal breakdown
    print(f"\n{'='*70}")
    print("Detailed Signals:")
    print(f"{'='*70}")

    for tf in TIMEFRAMES:
        if tf in signals:
            signal = signals[tf]
            print(f"\n{signal.timeframe}:")
            print(f"  Direction:         {signal.direction}")
            print(f"  Confidence:        {signal.confidence:.0%}")
            print(f"  Combined:          {signal.combined_confidence:.0%}")
            print(f"  XGBoost:           {signal.xgb_prob:.0%}")
            print(f"  LSTM:              {signal.lstm_prob:.0%}")
            print(f"  CNN:               {signal.cnn_prob:.0%}")

            if signal.pattern_detected:
                print(f"  Pattern:           {signal.pattern_detected} ({signal.pattern_confidence:.0%})")

            if signal.entry_price:
                print(f"  Entry:             ${signal.entry_price:,.2f}")
                print(f"  Stop Loss:         ${signal.stop_loss:,.2f}")
                print(f"  Take Profit:       ${signal.take_profit:,.2f}")

    print(f"\n{'='*70}")
    print("Demo Complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
