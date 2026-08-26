import asyncio

# Fix for Streamlit background thread asyncio event loop initialization
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import logging
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from broker_interface import BrokerFactory
from config import BrokerType, get_config

logger = logging.getLogger("Dashboard")

st.set_page_config(
    page_title="Quantitative Bot Telemetry",
    layout="wide",
    page_icon="📈",
)


@st.cache_resource
def load_app_config():
    return get_config()


config = load_app_config()

if config.broker.broker_type == BrokerType.BINANCE:
    DEFAULT_SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "BNBUSDC"]
else:
    DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL"]


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def fetch_broker_data(symbol: str):
    broker = BrokerFactory.create_broker(config.broker)
    await broker.connect()

    balance = await broker.get_account_balance()
    positions = await broker.get_positions()
    df_bars = await broker.get_historical_klines(
        symbol=symbol, timeframe="15m", limit=60
    )

    return balance, positions, df_bars


st.title("📈 Multi-Asset Quantitative Bot Command Center")

st.sidebar.header("Active Venue Settings")
st.sidebar.write(f"**Broker:** `{config.broker.broker_type.value}`")
st.sidebar.write(f"**Mode:** `{config.broker.trading_mode.value}`")
st.sidebar.write(f"**Target Universe:** `{', '.join(DEFAULT_SYMBOLS)}`")

st.sidebar.divider()
st.sidebar.header("Risk Guardrails")
st.sidebar.write(f"**Max 24h DD:** `{config.risk.max_drawdown_pct:.1%}`")
st.sidebar.write(f"**Risk Per Trade:** `{config.risk.risk_per_trade_pct:.1%}`")
st.sidebar.write(f"**Daily Limit:** `{config.risk.max_daily_trades} trades`")

if st.sidebar.button("🔄 Refresh Telemetry"):
    st.rerun()

selected_symbol = st.selectbox("Select Active Asset Feed", DEFAULT_SYMBOLS)

try:
    balance, positions, df_bars = run_async(fetch_broker_data(selected_symbol))
except Exception as e:
    st.error(f"Failed to connect to broker endpoint [{config.broker.broker_type.value}]: {e}")
    balance = {"equity": 0.0, "cash": 0.0, "buying_power": 0.0}
    positions = []
    df_bars = pd.DataFrame()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Broker", config.broker.broker_type.value)
col2.metric("Trading Mode", config.broker.trading_mode.value)
col3.metric("Total Equity", f"${balance.get('equity', 0.0):,.2f}")
col4.metric("Buying Power", f"${balance.get('buying_power', 0.0):,.2f}")

st.divider()

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader(f"Live Price Feed — {selected_symbol}")
    if df_bars is not None and not df_bars.empty:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_bars.index,
                    open=df_bars["open"],
                    high=df_bars["high"],
                    low=df_bars["low"],
                    close=df_bars["close"],
                    name=selected_symbol,
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No candle data available for {selected_symbol} on {config.broker.broker_type.value}.")

with right_col:
    st.subheader("Open Positions")
    if positions:
        df_pos = pd.DataFrame(positions)
        display_cols = [c for c in ["symbol", "qty", "side", "entry_price", "current_price"] if c in df_pos.columns]
        st.dataframe(
            df_pos[display_cols],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No open positions reported by active broker.")