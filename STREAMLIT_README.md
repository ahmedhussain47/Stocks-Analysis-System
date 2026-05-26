# Peak Accuracy ML Trading System — Streamlit App

## Running the App

### 1. Install Dependencies
```bash
pip install streamlit
```

### 2. Start the App
From the project root directory:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Login Credentials

### Admin (Ahmed)
- **Role:** admin
- **Password:** Ahmed_Admin_2024

### Brother Access
- **Role:** brother
- **Password:** Brother_Access_2024

---

## Features

### Admin Panel (After Login)
- ✓ Enable/disable brother's access
- ✓ Set daily signal limit (1-100 signals/day)
- ✓ Control access expiration date (1-90 days)
- ✓ View real-time usage statistics
- ✓ Logout

### Brother Access
- ✓ Generate live 15min signals from MT5
- ✓ View remaining signals for the day
- ✓ See access expiration date
- ✓ Logout

### Signal Generation
- **Model:** XGBoost + LSTM + 1D-CNN Ensemble
- **Timeframe:** 15min
- **Data Source:** Live MetaTrader5
- **Data Points:** 2200 bars = ~22 days of 15min candles
- **Accuracy:** 91.92% (XGBoost 1D)
- **Features:** 47 technical indicators

### Signal Display
Shows:
- Direction: BUY / SELL / FLAT
- Confidence score (0-100%)
- Individual model predictions
- Pattern detection results
- Risk/Reward levels:
  - Entry price
  - Stop Loss (ATR × 1.5)
  - Take Profit (ATR × 2.5)
  - R:R ratio

---

## MT5 Setup

The app uses your MT5 demo account:
- **Login:** 5050913403
- **Server:** MetaQuotes-Demo
- **Asset:** XAUUSD (Gold)
- **Mode:** 🧪 DEMO (no real money)

Make sure MT5 is running on your PC before starting the app.

---

## Session Management

- **Session File:** `.streamlit/session_state.pkl`
- **Auto-reset:** Daily counter resets at midnight UTC
- **Brother limits:** Configured per-day (default: 10 signals)
- **Access expiry:** Set in admin panel (default: 7 days)

---

## Troubleshooting

**MT5 Connection Failed:**
1. Check MT5 app is running
2. Verify login credentials in `app.py`
3. Ensure demo account is activated

**Import Error (src.signal_engine_v2):**
- Make sure `src/` folder is in project root
- Check all model files exist in `training/models/`

**No data from MT5:**
- Check market hours (Forex 24/5)
- Gold XAUUSD trades most hours
- Try refreshing after waiting a few bars

---

## Customization

To change credentials, edit `app.py`:
```python
ADMIN_PASS = "your_admin_password"
BROTHER_PASS = "your_brother_password"
```

To change default limits:
```python
def load_session_config():
    config = {
        "daily_limit": 10,  # Change here
        "allowed_until": (datetime.now() + timedelta(days=7)).isoformat(),  # Change days here
        ...
    }
```

---

## Notes

- The app stores session data in `.streamlit/session_state.pkl`
- Admin can override any session setting
- All times are UTC
- Signals are logged with generation timestamp
