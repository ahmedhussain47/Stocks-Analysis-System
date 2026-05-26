"""
MT5 Integration — Live data feed + order execution (demo/live).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List
import pandas as pd
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger(__name__)

# ── MT5 Symbol Mapping ──────────────────────────────────────────────────────
MT5_SYMBOLS = {
    'XAUUSD': 'XAUUSD',
    'XAGUSD': 'XAGUSD',
    'EURUSD': 'EURUSD',
    'GBPUSD': 'GBPUSD',
    'USDJPY': 'USDJPY',
    'USDCHF': 'USDCHF',
    'AUDUSD': 'AUDUSD',
    'USDCAD': 'USDCAD',
    'NZDUSD': 'NZDUSD',
    'GBPJPY': 'GBPJPY',
    'EURJPY': 'EURJPY',
    'EURGBP': 'EURGBP',
}


class MT5Trader:
    """Live MT5 connection for data + order execution."""

    def __init__(
        self,
        login: int,
        password: str,
        server: str = "MetaQuotes-Demo",
        demo_mode: bool = True,
    ):
        if mt5 is None:
            raise ImportError("MetaTrader5 not installed. Run: pip install MetaTrader5")

        self.login = login
        self.password = password
        self.server = server
        self.demo_mode = demo_mode
        self.connected = False
        self.open_positions: Dict = {}

    def connect(self) -> bool:
        """Connect to MT5."""
        try:
            if mt5.initialize(login=self.login, password=self.password, server=self.server):
                self.connected = True
                logger.info(f"MT5 connected: {self.server}")
                return True
            else:
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, bars: int = 500
    ) -> pd.DataFrame:
        """Fetch live OHLCV from MT5."""
        if not self.connected:
            logger.warning("MT5 not connected")
            return pd.DataFrame()

        mt5_symbol = MT5_SYMBOLS.get(symbol, symbol)
        tf_map = {
            "1min": mt5.TIMEFRAME_M1,
            "5min": mt5.TIMEFRAME_M5,
            "15min": mt5.TIMEFRAME_M15,
            "30min": mt5.TIMEFRAME_M30,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1,
        }

        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)

        try:
            rates = mt5.copy_rates_from_pos(mt5_symbol, tf, 0, bars)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data from MT5: {symbol} @ {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
            })
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df.set_index('timestamp', inplace=True)
            logger.debug(f"Fetched {len(df)} bars for {symbol} @ {timeframe}")
            return df
        except Exception as e:
            logger.error(f"MT5 fetch error: {e}")
            return pd.DataFrame()

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
        max_attempts: int = 3
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a date range from MT5."""
        if not self.connected:
            logger.warning("MT5 not connected")
            return pd.DataFrame()

        mt5_symbol = MT5_SYMBOLS.get(symbol, symbol)
        tf_map = {
            "1min": mt5.TIMEFRAME_M1,
            "5min": mt5.TIMEFRAME_M5,
            "15min": mt5.TIMEFRAME_M15,
            "30min": mt5.TIMEFRAME_M30,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1,
        }

        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)

        # Ensure datetime objects are UTC-aware
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        if date_to.tzinfo is None:
            date_to = date_to.replace(tzinfo=timezone.utc)

        all_rates = []
        attempt = 0
        current_start = date_from

        while current_start < date_to and attempt < max_attempts:
            try:
                rates = mt5.copy_rates_range(mt5_symbol, tf, current_start, date_to)

                if rates is None or len(rates) == 0:
                    logger.warning(f"No data from MT5: {symbol} @ {timeframe} from {current_start}")
                    break

                all_rates.extend(rates)

                # Move forward to avoid duplicate/missing bars
                last_time = pd.Timestamp(all_rates[-1]['time'], unit='s', tz=timezone.utc)
                current_start = last_time + pd.Timedelta(minutes=1)

                logger.debug(f"Fetched {len(rates)} bars for {symbol} @ {timeframe}. Total: {len(all_rates)}")

                if len(rates) < 300:  # Likely reached end of available data
                    break

            except Exception as e:
                attempt += 1
                logger.warning(f"Fetch attempt {attempt} failed: {e}")
                if attempt >= max_attempts:
                    logger.error(f"Max attempts reached for {symbol} @ {timeframe}")
                    break

        if not all_rates:
            logger.warning(f"No data collected for {symbol} @ {timeframe} from {date_from} to {date_to}")
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)

        # Remove duplicates (by timestamp) and sort
        if 'time' in df.columns:
            df.drop_duplicates(subset=['time'], keep='first', inplace=True)
            df.sort_values('time', inplace=True)
        else:
            logger.warning(f"No 'time' column in returned data")
            return pd.DataFrame()

        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
        })
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df.set_index('timestamp', inplace=True)

        logger.info(f"Successfully fetched {len(df)} bars for {symbol} @ {timeframe} from {date_from} to {date_to}")
        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest ask price."""
        if not self.connected:
            return None
        mt5_symbol = MT5_SYMBOLS.get(symbol, symbol)
        try:
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick:
                return float(tick.ask)
        except Exception as e:
            logger.error(f"Price fetch error: {e}")
        return None

    def get_account_summary(self) -> Dict:
        """Get account info."""
        if not self.connected:
            return {}
        try:
            info = mt5.account_info()
            return {
                'equity': float(info.equity),
                'balance': float(info.balance),
                'free_margin': float(info.margin_free),
                'margin_level': float(info.margin_level),
                'open_positions': len(self.open_positions),
            }
        except Exception as e:
            logger.error(f"Account info error: {e}")
            return {}

    def place_order(
        self,
        symbol: str,
        signal: str,  # 'BUY' or 'SELL'
        entry_price: float,
        sl_price: float,
        tp_price: float,
        volume: float,
        confidence: float = 0.5,
    ) -> Optional[Dict]:
        """Place order (demo mode: logs only)."""
        if not self.connected:
            logger.warning("MT5 not connected")
            return None

        mt5_symbol = MT5_SYMBOLS.get(symbol, symbol)
        order_type = mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL

        order_dict = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'action': signal,
            'entry': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'volume': volume,
            'confidence': confidence,
            'demo_mode': self.demo_mode,
            'status': 'LOGGED' if self.demo_mode else 'EXECUTED',
        }

        if self.demo_mode:
            logger.info(f"🧪 DEMO: {signal} {volume:.4f} {symbol} @ {entry_price:.5f}")
            return order_dict
        else:
            try:
                request = {
                    'action': mt5.TRADE_ACTION_DEAL,
                    'symbol': mt5_symbol,
                    'volume': volume,
                    'type': order_type,
                    'price': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'deviation': 10,
                    'magic': 234000,
                    'comment': f'Signal {signal} {confidence:.0%}',
                    'type_time': mt5.ORDER_TIME_GTC,
                    'type_filling': mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    order_dict['ticket'] = result.order
                    logger.info(f"Order placed: ticket={result.order}")
                    self.open_positions[result.order] = order_dict
                    return order_dict
                else:
                    logger.error(f"Order failed: {result.comment}")
                    return None
            except Exception as e:
                logger.error(f"Order placement error: {e}")
                return None

    def close_position(self, ticket: int, current_price: float) -> bool:
        """Close a position by ticket."""
        if not self.connected or self.demo_mode:
            self.open_positions.pop(ticket, None)
            return True

        try:
            position = mt5.positions_get(ticket=ticket)
            if position:
                request = {
                    'action': mt5.TRADE_ACTION_DEAL,
                    'position': ticket,
                    'symbol': position[0].symbol,
                    'volume': position[0].volume,
                    'type': mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    'price': current_price,
                    'deviation': 10,
                    'magic': 234000,
                    'comment': f'Close ticket {ticket}',
                    'type_time': mt5.ORDER_TIME_GTC,
                    'type_filling': mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.open_positions.pop(ticket, None)
                    return True
        except Exception as e:
            logger.error(f"Close error: {e}")
        return False
