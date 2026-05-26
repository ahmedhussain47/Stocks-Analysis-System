"""
MASTER TRAINING SCRIPT
======================

Runs all 3 phases sequentially:
  Phase 1: Fetch 2019-2026 data (10 assets)
  Phase 2: Generate 50+ features
  Phase 3: Train LSTM + XGBoost

Total runtime: ~2-3 hours
"""

import subprocess
import sys
from pathlib import Path
import time

print("="*80)
print("MULTI-ASSET MODEL TRAINING PIPELINE")
print("="*80)
print("\nThis script will:")
print("  1. Fetch 7 years of data for 10 forex pairs")
print("  2. Engineer 50+ features")
print("  3. Train LSTM + XGBoost on multi-asset ensemble")
print("\nEstimated time: 2-3 hours")
print("="*80)

# Check if training directory exists
training_dir = Path('training')
if not training_dir.exists():
    print("\n❌ ERROR: 'training' directory not found")
    print("Please create it with: mkdir training")
    sys.exit(1)

scripts = [
    ('01_fetch_data.py', 'Phase 1: Fetching Data'),
    ('02_feature_engineering.py', 'Phase 2: Feature Engineering'),
    ('03_train_models.py', 'Phase 3: Training Models'),
]

print("\nRunning training pipeline...")
print("-"*80)

success_count = 0

for script_name, phase_name in scripts:
    script_path = training_dir / script_name

    if not script_path.exists():
        print(f"\n❌ {phase_name}")
        print(f"   Script not found: {script_path}")
        continue

    print(f"\n{'='*80}")
    print(f"▶️  {phase_name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=training_dir.parent,
            capture_output=False
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ {phase_name} - COMPLETE ({elapsed/60:.1f} min)")
            success_count += 1
        else:
            print(f"\n❌ {phase_name} - FAILED (exit code {result.returncode})")

    except Exception as e:
        print(f"\n❌ {phase_name} - ERROR: {e}")

print("\n" + "="*80)
print("TRAINING PIPELINE SUMMARY")
print("="*80)

if success_count == len(scripts):
    print(f"\n✅ ALL PHASES COMPLETE ({success_count}/{len(scripts)})")
    print("\nNext steps:")
    print("  1. Review models in training/models/")
    print("  2. Copy best_model_lstm.h5 → results/advanced_models/")
    print("  3. Copy best_model_xgb.pkl → results/advanced_models/")
    print("  4. Copy scaler.pkl → results/advanced_models/")
    print("  5. Run Phase 5: Walk-forward validation")
else:
    print(f"\n⚠️  PARTIAL COMPLETE ({success_count}/{len(scripts)})")
    print("\nFailed phases:")
    for i, (script_name, phase_name) in enumerate(scripts):
        if i >= success_count:
            print(f"  - {phase_name}")

print("\n" + "="*80)
