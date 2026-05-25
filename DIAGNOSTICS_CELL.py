# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS CELL — Signal Quality Analysis
# Run this AFTER all other notebook cells to see performance metrics
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from pathlib import Path
from src.performance_metrics import PerformanceMetrics

print("="*80)
print("SIGNAL ENGINE DIAGNOSTICS")
print("="*80)

# ── Load signals log ──────────────────────────────────────────────────────────
log_path = Path('results/signals_log.csv')
if not log_path.exists():
    print(f"[WARNING] {log_path} not found - no signals logged yet")
    print("\nTo generate signals, run the live scanner cell with LIVE=True")
else:
    df_log = pd.read_csv(log_path)
    print(f"\n[SIGNALS LOGGED] {len(df_log)} total entries\n")

    # Filter to completed trades only (outcome != '')
    df_completed = df_log[df_log['outcome'].notna() & (df_log['outcome'] != '')]
    print(f"[COMPLETED TRADES] {len(df_completed)} trades closed")
    print(f"[PENDING SIGNALS] {len(df_log) - len(df_completed)} signals awaiting execution")

    if len(df_completed) > 0:
        # Display recent trades
        print(f"\n[RECENT TRADES (last 5)]:")
        print("-" * 100)
        cols_display = ['timestamp', 'signal', 'entry', 'take_profit', 'stop_loss', 'confidence', 'outcome', 'pnl_pct']
        for idx, row in df_completed.tail(5).iterrows():
            try:
                pnl = float(row.get('pnl_pct', 0))
                print(f"  {row.get('timestamp', 'N/A')[:10]} | {row.get('signal', 'N/A'):6s} | "
                      f"Entry: {float(row.get('entry', 0)):.2f} | "
                      f"SL: {float(row.get('stop_loss', 0)):.2f} | "
                      f"TP: {float(row.get('take_profit', 0)):.2f} | "
                      f"Conf: {row.get('confidence', 0)}% | "
                      f"Outcome: {row.get('outcome', 'N/A'):12s} | "
                      f"PnL: {pnl:+.2%}")
            except:
                pass

        # ── Compute performance metrics ──
        metrics = PerformanceMetrics()
        for idx, row in df_completed.iterrows():
            try:
                pnl_pct = float(row.get('pnl_pct', 0))
                metrics.add_trade(pnl_pct)
            except:
                pass

        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        print(f"Total Trades       : {metrics.count()}")
        print(f"Winning Trades     : {metrics.win_count()} ({metrics.win_rate():.1%})")
        print(f"Losing Trades      : {metrics.loss_count()} ({(1-metrics.win_rate()):.1%})")
        print(f"Cumulative PnL     : {metrics.total_pnl():+.2%}")
        print(f"Avg PnL/Trade      : {metrics.avg_pnl():+.2%}")
        print(f"Std Dev (PnL)      : {metrics.std_pnl():.2%}")
        print(f"Sharpe Ratio       : {metrics.sharpe_ratio():.2f}  {'✓ Good' if metrics.sharpe_ratio() > 0.5 else '✗ Needs improvement' if metrics.sharpe_ratio() < 0.1 else '~ Acceptable'}")
        print(f"Calmar Ratio       : {metrics.calmar_ratio():.2f}  {'✓ Good' if metrics.calmar_ratio() > 0.3 else '✗ Poor' if metrics.calmar_ratio() < 0.1 else '~ Acceptable'}")
        print(f"Profit Factor      : {metrics.profit_factor():.2f}  {'✓ Profitable' if metrics.profit_factor() > 1.5 else '✗ Losing' if metrics.profit_factor() < 1.0 else '~ Marginal'}")
        print(f"Max Drawdown       : {metrics.max_drawdown():.1%}")
        print(f"Recovery Factor    : {metrics.recovery_factor():.2f}")

    else:
        print("\n[NO COMPLETED TRADES] Waiting for signals to execute...")
        print("Live scanner needs to run for signals to be generated and closed.")

# ── Model status ──────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("MODEL STATUS")
print("="*80)

try:
    import json
    meta_path = Path('results/advanced_models/metadata.json')
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        print(f"\nBest Model (from metadata):")
        print(f"  Model Type   : {meta.get('best_model', 'N/A')}")
        print(f"  Accuracy     : {meta.get('best_accuracy', 0):.1%}")
        print(f"  AUC Score    : {meta.get('best_auc', 0):.4f}")
        print(f"  Features     : {len(meta.get('features', []))}")

        # Show all models evaluated
        print(f"\nAll Models Evaluated:")
        for model_name, metrics_dict in meta.get('model_metrics', {}).items():
            print(f"  {model_name:15s} - Accuracy: {metrics_dict.get('accuracy', 0):.1%}  AUC: {metrics_dict.get('auc', 0):.4f}")
    else:
        print("[WARNING] metadata.json not found")
except Exception as e:
    print(f"[ERROR] Could not load metadata: {e}")

# ── Current market conditions ──────────────────────────────────────────────────
print("\n" + "="*80)
print("CURRENT MARKET STATE")
print("="*80)

try:
    # Reconstruct from latest data
    if len(tf_data) > 0 and ENTRY_TF in tf_data:
        entry_df = tf_data[ENTRY_TF]
        cur_price = float(entry_df['close'].iloc[-1])
        last_time = entry_df.index[-1]

        # Compute latest indicators
        feat_latest = compute_features(entry_df).dropna()
        if len(feat_latest) > 0:
            feat_row = feat_latest.iloc[-1]
            ema20 = float(feat_row.get('ema_20', cur_price))
            ema50 = float(feat_row.get('ema_50', cur_price))
            rsi = float(feat_row.get('rsi_14', 50))
            adx = float(feat_row.get('adx_14', 0))

            print(f"\nPrice      : {cur_price:.5f}")
            print(f"EMA20/50   : {ema20:.5f} / {ema50:.5f}  (Price {'ABOVE' if cur_price > ema20 else 'BELOW'} EMA20)")
            print(f"RSI(14)    : {rsi:.1f}  {'[OVERBOUGHT]' if rsi > 70 else '[OVERSOLD]' if rsi < 30 else '[NEUTRAL]'}")
            print(f"ADX(14)    : {adx:.1f}  {'[TRENDING]' if adx > 25 else '[RANGING]'}")
            print(f"Last bar   : {last_time}")

            # Latest ML forecast
            if 'fc' in dir():  # fc is from cell_forecast
                print(f"\nLatest ML Forecast:")
                print(f"  Direction : {'BUY' if fc['direction'] > 0 else 'SELL' if fc['direction'] < 0 else 'NEUTRAL'}")
                print(f"  LSTM Pred : {fc.get('lstm_pred', 0):.4f}")
                print(f"  XGB Pred  : {fc.get('xgb_pred', 0):.4f}")
        else:
            print("[WARNING] No complete feature bars available")
except Exception as e:
    print(f"[ERROR] Could not compute market state: {e}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

try:
    if len(df_completed) > 0:
        win_rate = metrics.win_rate()
        sharpe = metrics.sharpe_ratio()

        if win_rate < 0.4:
            print("❌ Win rate too low (<40%) — system is losing money")
            print("   Action: Review signal quality, check if ML models need retraining")
        elif win_rate >= 0.55:
            print("✅ Win rate is good (≥55%) — system is profitable")
        else:
            print("⚠️  Win rate marginal (40-55%) — need more samples to confirm profitability")

        if sharpe < 0.1:
            print("❌ Sharpe ratio very low — risk/reward is poor")
        elif sharpe > 1.0:
            print("✅ Sharpe ratio good (>1.0) — excellent risk-adjusted returns")

        if metrics.profit_factor() < 1.0:
            print("❌ Profit factor <1.0 — system is unprofitable")
        elif metrics.profit_factor() > 1.5:
            print("✅ Profit factor strong (>1.5) — system is healthy")
    else:
        print("⏳ No completed trades yet — run live scanner to generate signals")
        print("   Cell: cell-11 (LIVE SCANNER)")
        print("   Set LIVE=True to start scanning")
except Exception as e:
    print(f"[ERROR] Could not generate recommendations: {e}")

print("\n" + "="*80)
