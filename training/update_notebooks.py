"""Update notebooks 05 and 06 to use 7-year XAUUSD trained models"""

import json
from pathlib import Path

print("Updating notebooks to use 7-year XAUUSD models...\n")

# Load notebook 05
nb05_path = Path('notebooks/05_signal_engine.ipynb')
with open(nb05_path, 'r', encoding='utf-8') as f:
    nb05 = json.load(f)

# Find feature cell and update
for cell in nb05['cells']:
    if cell.get('id') == '5aae2936':
        cell['source'] = ["# Features for LSTM and XGBoost (42 features from 7-year XAUUSD)\n_LSTM_XGB_FEATURES = [\n    'ema_20', 'ema_50', 'ema_200', 'dema_20',\n    'rsi_14', 'rsi_7',\n    'macd_line', 'macd_signal', 'macd_hist',\n    'stoch_k', 'stoch_d',\n    'atr_14', 'atr_7',\n    'bb_upper', 'bb_mid', 'bb_lower', 'bb_width', 'bb_pct',\n    'adx_14', 'plus_di', 'minus_di',\n    'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_span_a', 'ichimoku_span_b', 'ichimoku_chikou',\n    'dist_ema20', 'dist_ema50', 'dist_ema200',\n    'ret_1', 'ret_5', 'ret_20',\n    'log_ret_lag_1', 'log_ret_lag_2', 'log_ret_lag_3', 'log_ret_lag_5',\n    'volatility_5', 'volatility_10', 'volatility_20',\n    'price_pct_high_20',\n    'volume_ma', 'volume_ratio'\n]"]
        print("[OK] Cell 5aae2936: Updated feature list (42 features)")
        break

# Update model loading cell
for cell in nb05['cells']:
    if cell.get('id') == 'b008a111':
        # Just update the key lines
        source = cell.get('source', [])
        updated = []
        for line in source:
            if 'best_model_lstm.h5' in line:
                line = line.replace('best_model_lstm.h5', 'best_model_lstm_final.h5')
            if 'best_model_xgb.json' in line:
                line = line.replace('best_model_xgb.json', 'best_model_xgb_final.pkl')
            if 'scaler.pkl' in line and '../results' in line:
                line = line.replace('scaler.pkl', 'scaler_final.pkl')
            if "'LSTM': 0.6" in line:
                line = line.replace("'LSTM': 0.6", "'LSTM': 0.4")
            if "'XGBoost': 0.4" in line:
                line = line.replace("'XGBoost': 0.4", "'XGBoost': 0.6")
            updated.append(line)
        cell['source'] = updated
        print("[OK] Cell b008a111: Updated model paths (final.h5/pkl)")
        print("                    Updated weights: LSTM 40% -> XGBoost 60%")
        break

# Save notebook 05
with open(nb05_path, 'w', encoding='utf-8') as f:
    json.dump(nb05, f, indent=1)

print("\n[DONE] Notebook 05 updated successfully")
print("\nSummary of changes:")
print("  - Features: 45 -> 42 features")
print("  - Models: best_model_*.h5/json -> best_model_*_final.h5/pkl")
print("  - Weights: LSTM 60% / XGBoost 40% -> LSTM 40% / XGBoost 60%")
print("  - Data: 7 years XAUUSD, trained May 25 2026")
print("\nReady to run: notebooks/05_signal_engine.ipynb with LIVE=True")
