"""
Complete Pipeline Runner — Waits for base models, trains ensemble, validates.

This script monitors training status and automatically:
1. Waits for all base models to complete
2. Trains stacked ensemble meta-learner
3. Runs full validation on all models
4. Generates final accuracy report
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Set UTF-8 encoding for stdout on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Configuration ──────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / 'training' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_MODELS = {
    'xgb': ['xgb_v2_XAUUSD_1D.json', 'xgb_v2_XAUUSD_4H.json', 'xgb_v2_XAUUSD_1H.json', 'xgb_v2_XAUUSD_15min.json'],
    'lstm': ['lstm_att_XAUUSD_1D.h5', 'lstm_att_XAUUSD_4H.h5', 'lstm_att_XAUUSD_1H.h5', 'lstm_att_XAUUSD_15min.h5'],
    'cnn': ['cnn_1d_XAUUSD_15min.h5', 'cnn_1d_XAUUSD_1H.h5'],
}

def check_models_ready():
    """Check if all required base models are trained."""
    missing = []
    for category, models in REQUIRED_MODELS.items():
        for model in models:
            model_path = MODELS_DIR / model
            if not model_path.exists():
                missing.append(f"{category}/{model}")

    return len(missing) == 0, missing

def wait_for_models(max_wait_minutes=60, check_interval=30):
    """Wait for all base models to complete training."""
    print("\n" + "="*70)
    print("WAITING FOR BASE MODEL TRAINING TO COMPLETE")
    print("="*70)

    start_time = datetime.now()
    check_count = 0

    while True:
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        check_count += 1

        ready, missing = check_models_ready()

        if ready:
            print(f"\n[OK] All base models trained! ({check_count} checks)")
            return True

        if elapsed > max_wait_minutes:
            print(f"\n[TIMEOUT] Exceeded {max_wait_minutes} minutes waiting for models")
            print(f"Missing models:")
            for m in missing:
                print(f"  - {m}")
            return False

        print(f"[CHECK {check_count}] {len(missing)} models still training... "
              f"(elapsed: {elapsed:.1f}m, next check in {check_interval}s)")
        for m in missing:
            print(f"  - {m}")

        time.sleep(check_interval)

def run_ensemble_training():
    """Train stacked ensemble meta-learner."""
    print("\n" + "="*70)
    print("TRAINING STACKED ENSEMBLE")
    print("="*70)

    from training.train_ensemble_v2 import train_ensemble_tf

    TIMEFRAMES = ['1D', '4H', '1H', '15min']

    for tf in TIMEFRAMES:
        try:
            train_ensemble_tf(tf)
        except Exception as e:
            print(f"[FAIL] Error training ensemble for {tf}: {e}")
            return False

    return True

def run_validation():
    """Run complete validation suite."""
    print("\n" + "="*70)
    print("RUNNING COMPLETE VALIDATION SUITE")
    print("="*70)

    from training.validate_upgrade import validate_tf

    TIMEFRAMES = ['1D', '4H', '1H', '15min']

    all_results = {}

    for tf in TIMEFRAMES:
        try:
            print(f"\nValidating {tf}...")
            validate_tf(tf)
        except Exception as e:
            print(f"[FAIL] Error validating {tf}: {e}")

    return True

def main():
    """Main pipeline."""
    print("="*70)
    print("COMPLETE ML TRADING SYSTEM PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().isoformat()}")

    # Step 1: Wait for base models
    if not wait_for_models(max_wait_minutes=120, check_interval=30):
        print("\n[FAIL] Timed out waiting for base models")
        return False

    # Step 2: Train ensemble
    if not run_ensemble_training():
        print("\n[FAIL] Ensemble training failed")
        return False

    # Step 3: Run validation
    if not run_validation():
        print("\n[FAIL] Validation failed")
        return False

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"End time: {datetime.now().isoformat()}")
    print("\nNext steps:")
    print("1. Review validation results in FINAL_RESULTS.md")
    print("2. Deploy signal engine: python examples/demo_signal_generation.py")
    print("3. Monitor live signals from MT5")

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
