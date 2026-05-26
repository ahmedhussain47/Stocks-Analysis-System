# MT5 Integration Guide (Demo Mode)

## ⚠️ SAFETY FIRST: Demo Mode Only

This setup uses **DEMO MODE** by default (`DEMO_MODE = True`). Orders are validated but NOT placed on real MT5.

---

## Setup Steps

### Step 1: Install MetaTrader5 Python API
```bash
pip install MetaTrader5
```

### Step 2: Get Your MT5 Credentials
```
Login: [Your demo account number]
Password: [Your demo password]
Server: ICMarkets-Demo  # or your broker's demo server
```

### Step 3: Test Connection (before enabling live)
```python
from src.mt5_trader import MT5Trader

trader = MT5Trader(
    login=YOUR_LOGIN,
    password=YOUR_PASSWORD,
    server="ICMarkets-Demo",
    demo_mode=True  # ✅ DEMO MODE — safe
)

if trader.connect():
    summary = trader.get_account_summary()
    print(f"Account equity: ${summary['equity']:,.2f}")
```

---

## How It Works (Demo Mode)

### Flow: Signal → Validation → Demo Log

```
1. Signal engine generates: BUY XAUUSD @ 2450.50, SL=2448.00, TP=2455.00
                           ↓
2. MT5 trader validates:  ✓ Margin OK
                           ✓ Position count OK
                           ✓ Daily loss OK
                           ✓ SL distance OK
                           ↓
3. Order action:         🧪 LOGGED (NOT SENT TO MT5)
                         Saved to: results/mt5_trades.json
                           ↓
4. Output:              [MT5] 🧪 DEMO — BUY 0.50 XAUUSD @ 2450.50
                        SL: 2448.00 (250p) | TP: 2455.00 (450p) | RR: 1.80
```

### Demo Mode Output
- ✅ Validates every order against risk rules
- ✅ Logs validated orders to `results/mt5_trades.json`
- ✅ Shows what WOULD be executed
- ✅ NO actual orders placed
- ✅ Safe for testing for days/weeks

---

## Risk Controls (Always Active)

| Control | Default | Purpose |
|---------|---------|---------|
| **Position Size** | 2% of equity | Max risk per trade |
| **Max Daily Loss** | -5% | Stop trading if down $X |
| **Min SL Distance** | 20 pips | Prevent tight stops |
| **Max Positions** | 3 open | Avoid overleverage |
| **Slippage Buffer** | 5 pips | Account for execution slips |

---

## Usage in Notebook

### In Cell (BEFORE running live scanner):

```python
# ─────────────────────────────────────────────────────────────────
# MT5 SETUP (DEMO MODE)
# ─────────────────────────────────────────────────────────────────

from src.mt5_trader import MT5Trader

# Initialize in DEMO mode
trader = MT5Trader(
    login=YOUR_DEMO_LOGIN,        # Get from your MT5 account
    password=YOUR_DEMO_PASSWORD,
    server="ICMarkets-Demo",       # Check your broker
    demo_mode=True                 # ✅ DEMO — safe
)

if not trader.connect():
    print("MT5 connection failed!")
else:
    print("MT5 ready for DEMO trading")
```

### In Live Scanner Cell:

```python
# After compute_signal generates a signal:
if f_sig:
    # Place order on MT5 (demo mode)
    order = trader.place_order(
        symbol='XAUUSD',
        signal=f_sig['signal'],
        entry_price=f_sig['entry'],
        sl_price=f_sig['sl'],
        tp_price=f_sig['tp'],
        confidence=f_sig['confidence'] / 100.0,
    )
    
    if order:
        print(f"Order logged: {order}")
        
        # Show account summary
        summary = trader.get_account_summary()
        print(f"Account equity: ${summary['equity']:,.2f} | Open: {summary['open_positions']}")
```

---

## Validation Checklist (Demo Phase)

- [ ] MT5 connection successful
- [ ] 10+ demo orders validated & logged
- [ ] Risk controls blocking risky trades
- [ ] Account summary showing correct equity
- [ ] `results/mt5_trades.json` has order logs
- [ ] Win rate & profit factor tracked
- [ ] No errors over 24h test period

---

## Moving to Live (After Demo Validation)

### ONLY after 1-2 weeks of demo validation:

**Step 1:** Change `demo_mode=False`
```python
trader = MT5Trader(..., demo_mode=False)
```

**Step 2:** Switch to LIVE server
```python
server="ICMarkets-Live"  # NOT -Demo
```

**Step 3:** Monitor first 3 days closely
- Track P/L hourly
- Check for execution slippage
- Verify TP/SL placement

**Step 4:** If issues → Go back to demo, fix, repeat validation

---

## Files Generated

| File | Purpose |
|------|---------|
| `results/mt5_trades.json` | Log of all orders (validated or executed) |
| `results/signals_log.csv` | Signal engine performance |
| `results/advanced_models/` | Model predictions & confidence |

---

## Quick Commands

```python
# Check account
summary = trader.get_account_summary()

# Close all demo positions
for ticket in list(trader.open_positions.keys()):
    trader.close_position(ticket, current_price)

# Disconnect
trader.disconnect()
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "MT5 not connected" | Check login/password/server name |
| "Order rejected: Daily loss limit" | Stop trading for the day |
| "Insufficient margin" | Reduce position size % or add funds |
| "SL too close" | Increase SL distance (min 20p) |

---

## Support

For questions:
- Check `results/mt5_trades.json` for order logs
- Review signal engine output in notebook
- Verify risk controls in `src/mt5_trader.py` line ~75-110

