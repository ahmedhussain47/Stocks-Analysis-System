"""Quick check of training progress."""

from pathlib import Path

MODELS_DIR = Path('training/models')
MODELS_NEEDED = {
    'xgb': ['xgb_v2_XAUUSD_1D.json', 'xgb_v2_XAUUSD_4H.json', 'xgb_v2_XAUUSD_1H.json', 'xgb_v2_XAUUSD_15min.json'],
    'lstm': ['lstm_att_XAUUSD_1D.h5', 'lstm_att_XAUUSD_4H.h5', 'lstm_att_XAUUSD_1H.h5', 'lstm_att_XAUUSD_15min.h5'],
    'cnn': ['cnn_1d_XAUUSD_15min.h5', 'cnn_1d_XAUUSD_1H.h5'],
}

print("="*70)
print("TRAINING STATUS CHECK")
print("="*70)

all_complete = True

for model_type, files in MODELS_NEEDED.items():
    print(f"\n{model_type.upper()}:")
    for f in files:
        path = MODELS_DIR / f
        exists = path.exists()
        status = "[OK]" if exists else "[PENDING]"
        print(f"  {f:40s} {status}")
        if not exists:
            all_complete = False

print(f"\n{'='*70}")
if all_complete:
    print("[OK] All base models trained! Ready to run ensemble.")
    print("\nNext: python training/train_ensemble_v2.py")
else:
    print("[WAIT] Some models still training...")
    print("Check again in a few minutes.")

print(f"{'='*70}")
