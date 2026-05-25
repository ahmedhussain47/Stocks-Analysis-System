"""
Peak Accuracy ML Trading System — Streamlit App with Admin Controls
Allows controlled access to signal generation with permission-based session management.
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import time
import threading

# Fix imports - add parent directory to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_engine_v2 import SignalEngineV2
from src.mt5_trader import MT5Trader

# Configuration
ADMIN_PASSWORD = "Ahmed@477447"  # Change this!
SESSION_FILE = Path(__file__).parent / "session_control.json"
MODELS_DIR = ROOT / 'training' / 'models'

# Page config
st.set_page_config(
    page_title="Peak Accuracy ML Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Session Management
# ─────────────────────────────────────────────────────────────────────────────

def load_session_control():
    """Load session control settings from file."""
    if not os.path.exists(SESSION_FILE):
        return {
            "access_enabled": False,
            "enabled_until": None,
            "session_duration_minutes": 30,
            "max_signals_per_session": 5,
            "signals_generated": 0,
            "last_signal_time": None
        }
    with open(SESSION_FILE, 'r') as f:
        return json.load(f)

def save_session_control(config):
    """Save session control settings to file."""
    with open(SESSION_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def check_session_validity():
    """Check if current session is valid."""
    config = load_session_control()

    if not config["access_enabled"]:
        return False, "❌ Access not granted. Ask admin to enable session."

    if config["enabled_until"]:
        enabled_until = datetime.fromisoformat(config["enabled_until"])
        if datetime.now() > enabled_until:
            config["access_enabled"] = False
            save_session_control(config)
            return False, "⏰ Session expired. Ask admin to grant new access."

    if config["signals_generated"] >= config["max_signals_per_session"]:
        return False, f"📊 Limit reached ({config['signals_generated']}/{config['max_signals_per_session']} signals). Ask admin for reset."

    return True, "✅ Session valid"

def hash_password(pwd):
    """Hash password for comparison."""
    return hashlib.sha256(pwd.encode()).hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# Admin Panel
# ─────────────────────────────────────────────────────────────────────────────

def show_admin_panel():
    """Admin control panel."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Admin Panel")

    # Password protection
    admin_pwd = st.sidebar.text_input("Admin Password:", type="password", key="admin_pwd")

    if admin_pwd and hash_password(admin_pwd) == hash_password(ADMIN_PASSWORD):
        st.sidebar.success("✅ Admin authenticated")

        with st.sidebar.expander("⚙️ Session Controls", expanded=True):
            config = load_session_control()

            # Toggle access
            col1, col2 = st.columns(2)
            with col1:
                enable_access = st.checkbox(
                    "Enable Access",
                    value=config["access_enabled"],
                    help="Grant your brother permission to generate signals"
                )
            with col2:
                if enable_access:
                    duration = st.number_input(
                        "Duration (min):",
                        min_value=5,
                        max_value=480,
                        value=config["session_duration_minutes"],
                        step=5
                    )
                    config["session_duration_minutes"] = duration
                    config["enabled_until"] = (datetime.now() + timedelta(minutes=duration)).isoformat()

            # Max signals per session
            st.write("**Limits:**")
            config["max_signals_per_session"] = st.slider(
                "Max signals per session:",
                min_value=1,
                max_value=20,
                value=config["max_signals_per_session"],
                step=1
            )

            # Status
            st.write("**Current Status:**")
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.metric("Access Enabled", "✅ YES" if enable_access else "❌ NO")
            with status_col2:
                st.metric("Signals Generated", f"{config['signals_generated']}/{config['max_signals_per_session']}")

            # Save changes
            config["access_enabled"] = enable_access
            save_session_control(config)

            if st.button("🔄 Reset Signal Counter", use_container_width=True):
                config["signals_generated"] = 0
                save_session_control(config)
                st.success("Counter reset!")

            if st.button("🚫 Revoke Access Immediately", use_container_width=True, type="secondary"):
                config["access_enabled"] = False
                config["enabled_until"] = None
                save_session_control(config)
                st.warning("Access revoked!")

    elif admin_pwd:
        st.sidebar.error("❌ Wrong password")

# ─────────────────────────────────────────────────────────────────────────────
# Main Trading Interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 Peak Accuracy ML Trading System")
    st.subheader("91.92% Accuracy | Multi-Timeframe Ensemble | XAUUSD")

    # Show admin panel
    show_admin_panel()

    # Check session validity
    is_valid, msg = check_session_validity()

    if is_valid:
        st.success(msg)

        # Main trading interface
        tab1, tab2, tab3, tab4 = st.tabs(["Generate Signals", "Live Scanner", "System Info", "Session Status"])

        with tab1:
            st.header("Signal Generation")

            if st.button("🎯 Generate Trading Signals", use_container_width=True, type="primary"):
                with st.spinner("Loading system and fetching data..."):
                    try:
                        # Initialize engine
                        engine = SignalEngineV2(symbol='XAUUSD', models_dir=MODELS_DIR, use_patterns=True)

                        # Connect to MT5
                        trader = MT5Trader(
                            login=5050913403,
                            password="Ahmed@477447",
                            server="MetaQuotes-Demo",
                            demo_mode=True
                        )

                        if not trader.connect():
                            st.error("❌ MT5 connection failed")
                            return

                        # Fetch data
                        st.info("📊 Fetching live data...")
                        dfs = {}
                        for tf in ['1D', '4H', '1H', '15min']:
                            df = trader.fetch_ohlcv(symbol='XAUUSD', timeframe=tf, bars=100)
                            if not df.empty:
                                dfs[tf] = df

                        if not dfs:
                            st.error("❌ No data from MT5")
                            return

                        # Generate signals
                        st.info("🤖 Generating signals from 10 trained models...")
                        signals = {}

                        signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
                        cols = [signal_col1, signal_col2, signal_col3, signal_col4]

                        for idx, tf in enumerate(['1D', '4H', '1H', '15min']):
                            if tf in dfs:
                                sig = engine.generate_signal(dfs[tf], tf)
                                signals[tf] = sig

                                with cols[idx]:
                                    direction_color = "🟢" if sig.direction == "BUY" else ("🔴" if sig.direction == "SELL" else "⚪")
                                    st.metric(
                                        f"{tf}",
                                        f"{direction_color} {sig.direction}",
                                        f"Conf: {sig.confidence:.0%}"
                                    )

                        # Multi-timeframe consensus
                        st.divider()
                        st.subheader("Multi-Timeframe Consensus")

                        if len(signals) >= 2:
                            consensus = engine.aggregate_signals(signals)

                            consensus_col1, consensus_col2, consensus_col3 = st.columns(3)

                            with consensus_col1:
                                direction = consensus.get('direction', 'FLAT')
                                direction_color = "🟢" if direction == "BUY" else ("🔴" if direction == "SELL" else "⚪")
                                st.metric("Direction", f"{direction_color} {direction}")

                            with consensus_col2:
                                st.metric("Confidence", f"{consensus.get('confidence', 0):.0%}")

                            with consensus_col3:
                                alignment = consensus.get('alignment_count', 0)
                                st.metric("Alignment", f"{alignment}/4 TFs agree")

                            # Recommendation
                            st.divider()
                            if alignment >= 3 and direction != "FLAT":
                                st.success(f"✅ **STRONG SIGNAL**: {alignment}/4 timeframes agree → **FULL POSITION**")
                            elif alignment >= 2 and direction != "FLAT":
                                st.warning(f"⚠️ **MODERATE SIGNAL**: {alignment}/4 timeframes agree → **50-75% POSITION**")
                            else:
                                st.info(f"ℹ️ **WEAK/FLAT SIGNAL**: {alignment}/4 timeframes agree → **SKIP OR SMALL POSITION**")

                        # Update session counter
                        config = load_session_control()
                        config["signals_generated"] += 1
                        config["last_signal_time"] = datetime.now().isoformat()
                        save_session_control(config)

                        st.success(f"✅ Signals generated! ({config['signals_generated']}/{config['max_signals_per_session']} used)")

                        # Cleanup
                        trader.disconnect()

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)[:200]}")

        with tab2:
            st.header("📡 Live Signal Scanner")
            st.info("🔴 Real-time monitoring of all timeframes for trading signals")

            # Scanner controls
            col1, col2, col3 = st.columns(3)

            with col1:
                scan_interval = st.slider("Scan interval (seconds):", min_value=5, max_value=60, value=15, step=5)

            with col2:
                max_scans = st.number_input("Duration (minutes):", min_value=5, max_value=120, value=30, step=5)

            with col3:
                start_scanner = st.button("🟢 Start Live Scanning", use_container_width=True, type="primary")
                stop_scanner = st.button("🛑 Stop Scanner", use_container_width=True, type="secondary")

            # Initialize session state for scanner
            if 'scanner_active' not in st.session_state:
                st.session_state.scanner_active = False
                st.session_state.scan_logs = []
                st.session_state.last_signals = {}
                st.session_state.scan_count = 0
                st.session_state.scanner_start_time = None

            # Handle scanner start/stop
            if start_scanner:
                st.session_state.scanner_active = True
                st.session_state.scan_logs = []
                st.session_state.scan_count = 0
                st.session_state.scanner_start_time = datetime.now()

            if stop_scanner:
                st.session_state.scanner_active = False

            # Display scanner
            if st.session_state.scanner_active:
                signal_placeholder = st.empty()
                status_placeholder = st.empty()
                log_placeholder = st.empty()

                try:
                    # Single scan iteration
                    st.session_state.scan_count += 1
                    elapsed = (datetime.now() - st.session_state.scanner_start_time).total_seconds() / 60
                    end_time = st.session_state.scanner_start_time + timedelta(minutes=max_scans)

                    # Update status
                    with status_placeholder.container():
                        s_col1, s_col2, s_col3 = st.columns(3)
                        with s_col1:
                            st.metric("Scans", f"{st.session_state.scan_count}")
                        with s_col2:
                            st.metric("Elapsed", f"{elapsed:.1f}m / {max_scans}m")
                        with s_col3:
                            st.metric("Scan Interval", f"{scan_interval}s")

                    # Check if duration exceeded
                    if datetime.now() > end_time:
                        st.session_state.scanner_active = False
                        st.success(f"✅ Scanner completed! {st.session_state.scan_count} scans, {len(st.session_state.scan_logs)} signals detected")
                    else:
                        # Generate signals
                        try:
                            engine = SignalEngineV2(symbol='XAUUSD', models_dir=MODELS_DIR, use_patterns=True)
                            trader = MT5Trader(
                                login=5050913403,
                                password="Ahmed@477447",
                                server="MetaQuotes-Demo",
                                demo_mode=True
                            )

                            if trader.connect():
                                dfs = {}
                                for tf in ['1D', '4H', '1H', '15min']:
                                    df = trader.fetch_ohlcv(symbol='XAUUSD', timeframe=tf, bars=100)
                                    if not df.empty:
                                        dfs[tf] = df

                                # Generate signals
                                signals = {}
                                for tf in ['1D', '4H', '1H', '15min']:
                                    if tf in dfs:
                                        sig = engine.generate_signal(dfs[tf], tf)
                                        signals[tf] = sig

                                        # Check for new signals
                                        if tf not in st.session_state.last_signals or st.session_state.last_signals[tf].direction != sig.direction:
                                            if sig.direction != "FLAT":
                                                log_msg = f"🔔 [{datetime.now().strftime('%H:%M:%S')}] {tf}: {sig.direction} ({sig.confidence:.0%})"
                                                st.session_state.scan_logs.append(log_msg)

                                st.session_state.last_signals = signals

                                # Display current signals
                                with signal_placeholder.container():
                                    st.markdown("**Current Signals:**")
                                    sig_cols = st.columns(4)
                                    for idx, tf in enumerate(['1D', '4H', '1H', '15min']):
                                        if tf in signals:
                                            sig = signals[tf]
                                            direction_color = "🟢" if sig.direction == "BUY" else ("🔴" if sig.direction == "SELL" else "⚪")
                                            with sig_cols[idx]:
                                                st.metric(
                                                    f"{tf}",
                                                    f"{direction_color} {sig.direction}",
                                                    f"{sig.confidence:.0%}"
                                                )

                                # Display logs
                                with log_placeholder.container():
                                    st.markdown("**📋 Scan Log:**")
                                    log_area = st.empty()
                                    with log_area:
                                        log_text = "\n".join(st.session_state.scan_logs[-20:])
                                        st.code(log_text, language="text")

                                trader.disconnect()

                                # Trigger next scan after interval
                                time.sleep(scan_interval)
                                st.rerun()

                        except Exception as e:
                            st.session_state.scan_logs.append(f"❌ [{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)[:50]}")
                            time.sleep(scan_interval)
                            st.rerun()

                except Exception as e:
                    st.error(f"Scanner error: {str(e)}")
                    st.session_state.scanner_active = False

            elif st.session_state.scan_logs:
                # Show last scan results after stop
                st.info("📊 Last scan results:")
                log_text = "\n".join(st.session_state.scan_logs[-20:])
                st.code(log_text, language="text")

        with tab3:
            st.header("System Information")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Accuracy")
                st.write("""
                - **XGBoost 1D**: 91.92% (AUC: 0.986)
                - **Target**: 82-85%
                - **Achievement**: 106.7% ✓
                """)

            with col2:
                st.subheader("📈 Data")
                st.write("""
                - **Duration**: 7 years
                - **Total Bars**: 90,000+
                - **Features**: 47 per bar
                - **Timeframes**: 4 (1D, 4H, 1H, 15min)
                """)

            st.subheader("🤖 Models")
            st.write("""
            - **XGBoost**: 4 timeframes trained
            - **LSTM+Attention**: 4 timeframes trained
            - **1D-CNN**: 2 timeframes trained
            - **Ensemble**: Stacked meta-learner
            """)

        with tab3:
            st.header("Session Status")
            config = load_session_control()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Access Enabled", "✅ YES" if config["access_enabled"] else "❌ NO")

            with col2:
                st.metric("Signals Generated", f"{config['signals_generated']}/{config['max_signals_per_session']}")

            with col3:
                if config["enabled_until"]:
                    until = datetime.fromisoformat(config["enabled_until"])
                    remaining = (until - datetime.now()).total_seconds() / 60
                    if remaining > 0:
                        st.metric("Time Remaining", f"{int(remaining)}m")
                    else:
                        st.metric("Time Remaining", "Expired")
                else:
                    st.metric("Time Remaining", "N/A")

    else:
        # Access denied
        st.error(msg, icon="🚫")
        st.markdown("---")
        st.info("""
        ### ⏸️ Access Not Granted

        Your brother (Ahmed) needs to grant you permission to generate signals.

        Ask him to:
        1. Enter admin password in the sidebar
        2. Toggle "Enable Access"
        3. Set session duration
        4. Click to apply changes

        You can then generate up to the limit of signals during the session.
        """)

if __name__ == "__main__":
    main()
