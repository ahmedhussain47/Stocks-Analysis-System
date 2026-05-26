"""
Peak Accuracy ML Trading System — Streamlit Web App
Admin controls + brother access with session limits.
Live MT5 integration, 15min ensemble signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Setup paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    from src.signal_engine_v2 import SignalEngineV2
    import MetaTrader5 as mt5
except ImportError as e:
    st.error(f"Import Error: {e}\nMake sure src/ folder and MetaTrader5 are available.")

# ════════════════════════════════════════════════════════════════════════════
# Page Config & Constants
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Peak Accuracy ML Trading",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Admin & Brother credentials
ADMIN_USER = "admin"
ADMIN_PASS = "Ahmed_Admin_2024"
BROTHER_USER = "brother"
BROTHER_PASS = "Brother_Access_2024"

# Session file path
SESSION_FILE = ROOT / ".streamlit" / "session_state.pkl"
SESSION_FILE.parent.mkdir(exist_ok=True)

# MT5 credentials
MT5_LOGIN = 5050913403
MT5_PASS = "Ahmed@477447"
MT5_SERVER = "MetaQuotes-Demo"
ASSET = "XAUUSD"
MODELS_DIR = ROOT / 'training' / 'models'

# ════════════════════════════════════════════════════════════════════════════
# Session Management Functions
# ════════════════════════════════════════════════════════════════════════════

def load_session_config():
    """Load session limits and user status."""
    config = {
        "brother_enabled": True,
        "daily_limit": 10,
        "allowed_until": (datetime.now() + timedelta(days=7)).isoformat(),
        "used_today": 0,
        "last_reset_date": datetime.now().date().isoformat(),
    }

    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE, 'rb') as f:
                saved = pickle.load(f)
                config.update(saved)
        except Exception:
            pass

    return config


def save_session_config(config):
    """Save session limits and user status."""
    try:
        with open(SESSION_FILE, 'wb') as f:
            pickle.dump(config, f)
    except Exception as e:
        st.error(f"Failed to save config: {e}")


def is_access_allowed(user_role):
    """Check if user can generate signals."""
    config = load_session_config()

    if user_role == "admin":
        return True, "✓ Admin access granted"

    if user_role == "brother":
        if not config["brother_enabled"]:
            return False, "❌ Access disabled by admin"

        allowed_until = datetime.fromisoformat(config["allowed_until"])
        if datetime.now() > allowed_until:
            return False, f"❌ Access expired on {allowed_until.strftime('%Y-%m-%d')}"

        if config["used_today"] >= config["daily_limit"]:
            return False, f"⚠️ Daily limit reached ({config['daily_limit']} signals/day)"

        return True, f"✓ {config['daily_limit'] - config['used_today']} signals remaining today"

    return False, "Invalid user role"


def increment_signal_count():
    """Increment signal generation count for the day."""
    config = load_session_config()
    today = datetime.now().date()

    if config.get("last_reset_date") != today.isoformat():
        config["used_today"] = 0
        config["last_reset_date"] = today.isoformat()

    config["used_today"] += 1
    save_session_config(config)

# ════════════════════════════════════════════════════════════════════════════
# Sidebar — Login & Admin Panel
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔐 Access Control")

    user_role = st.radio("Role:", ["admin", "brother"], horizontal=False)
    password = st.text_input("Password:", type="password", key="login_password")

    if st.button("🔓 Login", use_container_width=True, key="login_btn"):
        if user_role == "admin" and password == ADMIN_PASS:
            st.session_state.logged_in = True
            st.session_state.user_role = "admin"
            st.success("✓ Admin logged in")
        elif user_role == "brother" and password == BROTHER_PASS:
            st.session_state.logged_in = True
            st.session_state.user_role = "brother"
            st.success("✓ Brother logged in")
        else:
            st.error("❌ Invalid credentials")
        st.rerun()

    if st.session_state.get("logged_in"):
        st.divider()

        # Admin Panel
        if st.session_state.user_role == "admin":
            st.subheader("⚙️ Admin Settings")

            config = load_session_config()

            col1, col2 = st.columns(2)
            with col1:
                brother_enabled = st.checkbox(
                    "Allow brother access",
                    value=config["brother_enabled"],
                    key="enable_brother"
                )
            with col2:
                daily_limit = st.number_input(
                    "Daily signal limit",
                    min_value=1,
                    max_value=100,
                    value=config["daily_limit"],
                    key="daily_limit_input"
                )

            allowed_days = st.slider(
                "Access expires in (days)",
                min_value=1,
                max_value=90,
                value=7,
                key="access_days"
            )

            if st.button("💾 Save Settings", use_container_width=True, key="save_btn"):
                config["brother_enabled"] = brother_enabled
                config["daily_limit"] = daily_limit
                config["allowed_until"] = (
                    datetime.now() + timedelta(days=allowed_days)
                ).isoformat()
                config["used_today"] = 0
                config["last_reset_date"] = datetime.now().date().isoformat()
                save_session_config(config)
                st.success("✓ Settings saved")

            st.divider()

            # Status display
            st.subheader("📊 Usage Status")
            st.write(f"**Brother enabled:** {'✓' if config['brother_enabled'] else '✗'}")
            st.write(f"**Daily limit:** {config['daily_limit']} signals")
            st.write(f"**Used today:** {config['used_today']}")
            exp_date = config["allowed_until"][:10]
            st.write(f"**Expires:** {exp_date}")

            if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
                st.session_state.logged_in = False
                st.session_state.user_role = None
                st.rerun()
        else:
            # Brother view
            st.subheader("👤 Session Info")
            config = load_session_config()
            remaining = config["daily_limit"] - config["used_today"]

            st.metric("Signals Today", f"{config['used_today']}/{config['daily_limit']}")
            st.metric("Remaining", remaining)

            if remaining <= 2:
                st.warning(f"⚠️ Only {remaining} signals left!")

            exp_date = config["allowed_until"][:10]
            st.write(f"**Access until:** {exp_date}")

            if st.button("🚪 Logout", use_container_width=True, key="logout_btn_brother"):
                st.session_state.logged_in = False
                st.session_state.user_role = None
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# Main Page Header
# ════════════════════════════════════════════════════════════════════════════

st.title("🏆 Peak Accuracy ML Trading System")
st.subheader("XAUUSD • 15min Timeframe • 91.92% Accuracy Ensemble")

# ════════════════════════════════════════════════════════════════════════════
# Main Content
# ════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("logged_in"):
    st.warning("👈 **Please log in from the sidebar** to generate signals")
    st.stop()

# Check access permission
allowed, message = is_access_allowed(st.session_state.user_role)

if not allowed:
    st.error(message)
    st.stop()

st.info(message)

# Signal generation section
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📊 Generate 15min Signal")

    if st.button("🟢 Generate Signal from MT5", use_container_width=True, key="gen_signal_btn"):
        try:
            with st.spinner("Connecting to MT5 and generating signal..."):
                # Initialize MT5
                if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASS, server=MT5_SERVER):
                    st.error(f"❌ MT5 Connection Failed: {mt5.last_error()}")
                    st.stop()

                # Fetch 15min data (2200 bars)
                rates = mt5.copy_rates_from_pos(ASSET, mt5.TIMEFRAME_M15, 0, 2200)

                if rates is None or len(rates) == 0:
                    st.error("❌ No data from MT5. Check market hours.")
                    mt5.shutdown()
                    st.stop()

                # Prepare DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df = df.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'})
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df.set_index('timestamp', inplace=True)

                # Generate signal using ML ensemble
                engine = SignalEngineV2(symbol=ASSET, models_dir=MODELS_DIR, use_patterns=True)
                sig = engine.generate_signal(df, '15min')

                # Store in session state
                st.session_state.signal_data = {
                    'sig': sig,
                    'df': df,
                    'generated_at': datetime.now(tz=__import__('datetime').timezone.utc).isoformat(),
                }

                # Increment counter for brother
                if st.session_state.user_role == "brother":
                    increment_signal_count()

                mt5.shutdown()
                st.success("✓ Signal generated!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)[:150]}")
            try:
                mt5.shutdown()
            except:
                pass

with col2:
    st.subheader("ℹ️ Info")
    st.write(f"**Mode:** 🧪 Demo")
    st.write(f"**Asset:** {ASSET}")
    st.write(f"**TF:** 15min")
    config = load_session_config()
    if st.session_state.user_role == "brother":
        remaining = config["daily_limit"] - config["used_today"]
        st.metric("Signals Left", remaining)

# Display signal results
if "signal_data" in st.session_state:
    st.divider()

    sig = st.session_state.signal_data['sig']
    df = st.session_state.signal_data['df']
    gen_time = st.session_state.signal_data['generated_at']

    # Main signal display
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        dir_color = "#00C853" if sig.direction == "BUY" else "#FF1744" if sig.direction == "SELL" else "#FFA726"
        dir_emoji = "▲" if sig.direction == "BUY" else "▼" if sig.direction == "SELL" else "◆"
        st.markdown(
            f"<h3 style='color:{dir_color};margin:0'>{dir_emoji} {sig.direction}</h3>",
            unsafe_allow_html=True
        )
        st.caption(f"@ ${df['close'].iloc[-1]:.2f}")

    with col_b:
        st.metric("Confidence", f"{sig.combined_confidence:.0%}")

    with col_c:
        st.metric("Pattern", sig.pattern_detected or "None")

    with col_d:
        st.metric("Ensemble Prob", f"{sig.ensemble_prob:.1%}" if sig.ensemble_prob else "N/A")

    st.divider()

    # Model predictions
    st.subheader("🤖 Model Predictions")
    pred_cols = st.columns(4)
    with pred_cols[0]:
        st.metric("XGBoost", f"{sig.xgb_prob:.1%}")
    with pred_cols[1]:
        st.metric("LSTM", f"{sig.lstm_prob:.1%}")
    with pred_cols[2]:
        st.metric("CNN", f"{sig.cnn_prob:.1%}")
    with pred_cols[3]:
        st.metric("Average", f"{(sig.xgb_prob + sig.lstm_prob + sig.cnn_prob)/3:.1%}")

    # Risk management
    if sig.direction != 'FLAT':
        st.divider()
        st.subheader("⚠️ Risk Management")
        rm_cols = st.columns(4)
        with rm_cols[0]:
            st.metric("Entry", f"${sig.entry_price:.2f}")
        with rm_cols[1]:
            st.metric("Stop Loss", f"${sig.stop_loss:.2f}")
        with rm_cols[2]:
            st.metric("Take Profit", f"${sig.take_profit:.2f}")
        with rm_cols[3]:
            risk = abs(sig.entry_price - sig.stop_loss)
            reward = abs(sig.take_profit - sig.entry_price)
            rr = reward / risk if risk > 0 else 0
            st.metric("R:R Ratio", f"1:{rr:.2f}")

    st.divider()
    st.caption(f"Generated: {gen_time[:19]} UTC | Models: 7y data | 47 features")
