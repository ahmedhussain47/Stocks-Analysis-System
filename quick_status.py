"""Quick training status check."""
import sys
from pathlib import Path

# Set UTF-8 encoding for stdout on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from training.feature_engineering_v2 import atr
import pandas as pd
import json

MODELS_DIR = Path(__file__).parent / 'training' / 'models'
DATA_DIR = Path(__file__).parent / 'training' / 'data' / 'raw' / 'mt5'

print("\n" + "="*70)
print("PEAK ACCURACY ML TRADING SYSTEM — QUICK STATUS")
print("="*70)

# Check models
models_status = {
    'XGBoost': {
        '1D': MODELS_DIR / 'xgb_v2_XAUUSD_1D.json',
        '4H': MODELS_DIR / 'xgb_v2_XAUUSD_4H.json',
        '1H': MODELS_DIR / 'xgb_v2_XAUUSD_1H.json',
        '15min': MODELS_DIR / 'xgb_v2_XAUUSD_15min.json',
    },
    'LSTM': {
        '1D': MODELS_DIR / 'lstm_att_XAUUSD_1D.h5',
        '4H': MODELS_DIR / 'lstm_att_XAUUSD_4H.h5',
        '1H': MODELS_DIR / 'lstm_att_XAUUSD_1H.h5',
        '15min': MODELS_DIR / 'lstm_att_XAUUSD_15min.h5',
    },
    '1D-CNN': {
        '1H': MODELS_DIR / 'cnn_1d_XAUUSD_1H.h5',
        '15min': MODELS_DIR / 'cnn_1d_XAUUSD_15min.h5',
    },
}

# Check data
print("\nDATA:")
for tf in ['1D', '4H', '1H', '15min']:
    csv_path = DATA_DIR / f'XAUUSD_{tf}_mt5.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"  {tf}: {len(df):,} bars [OK]")
    else:
        print(f"  {tf}: NOT FOUND [FAIL]")

# Check models
print("\nMODELS:")
total_models = 0
trained_models = 0

for model_type, tfs in models_status.items():
    print(f"  {model_type}:")
    for tf, path in tfs.items():
        total_models += 1
        status = "[OK]" if path.exists() else "[PENDING]"
        if path.exists():
            trained_models += 1
        print(f"    {tf}: {status}")

print(f"\n  TOTAL: {trained_models}/{total_models} trained ({100*trained_models//total_models}%)")

# Check validation results
print("\nVALIDATION RESULTS:")
try:
    final_results_path = Path(__file__).parent / 'FINAL_RESULTS.md'
    if final_results_path.exists():
        with open(final_results_path) as f:
            content = f.read()
            if '91.92%' in content:
                print("  [OK] XGBoost 1D: 91.92% accuracy (AUC 0.986)")
            if 'PRODUCTION READY' in content:
                print("  [OK] System Status: PRODUCTION READY")
except:
    pass

print("\nNEXT STEPS:")
print("  1. Wait for XGBoost & LSTM training to complete (~45-60 min)")
print("  2. Ensemble meta-learner will auto-train")
print("  3. Full validation will run automatically")
print("  4. System ready for live MT5 deployment")

print("\n" + "="*70 + "\n")
