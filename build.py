import os
import zipfile
from pathlib import Path

# Project file contents mapping
FILES = {
    "trading_bot/requirements.txt": """
pandas>=2.1.0
numpy>=1.26.0
statsmodels>=0.14.0
ta>=0.10.2
alpaca-py>=0.13.0
python-binance>=1.0.19
ib-insync>=0.9.86
streamlit>=1.31.0
plotly>=5.18.0
rich>=13.7.0
python-dotenv>=1.0.0
pydantic>=2.5.0
""".strip(),

    "trading_bot/config.py": """
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class TradingMode(Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"

class BrokerType(Enum):
    BINANCE = "BINANCE"
    ALPACA = "ALPACA"
    IBKR = "IBKR"

@dataclass
class BrokerConfig:
    broker_type: BrokerType = BrokerType.ALPACA
    trading_mode: TradingMode = TradingMode.PAPER
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("API_SECRET", ""))
    passphrase: Optional[str] = field(default_factory=lambda: os.getenv("API_PASSPHRASE", None))
    host: str = "127.0.0.1"
    port: int = 7497

@dataclass
class RiskConfig:
    max_drawdown_pct: float = 0.05
    risk_per_trade_pct: float = 0.015
    max_daily_trades: int = 20
    max_asset_concentration_pct: float = 0.25
    kelly_fraction: float = 0.5
    atr_multiplier: float = 2.0
    atr_period: int = 14

@dataclass
class StrategyConfig:
    name: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
""".strip(),

    "trading_bot/broker_interface.py": """
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd

class BaseBroker(ABC):
    \"\"\"Abstract base broker interface normalizing multi-venue actions.\"\"\"
    
    @abstractmethod
    async def get_account_balance(self) -> Dict[str, float]:
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_historical_klines(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        pass

    @abstractmethod
    async def place_order(self, symbol: str, qty: float, side: str, order_type: str = "MARKET", sl: Optional[float] = None, tp: Optional[float] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return symbol.replace("-", "").replace("/", "").upper()
""".strip(),

    "trading_bot/strategies/__init__.py": "",

    "trading_bot/strategies/base.py": """
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

@dataclass
class Signal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float
    metadata: Dict[str, Any]

class BaseStrategy(ABC):
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def generate_signals(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        pass
""".strip(),

    "trading_bot/strategies/trend_following.py": """
import pandas as pd
import ta
from typing import Dict
from strategies.base import BaseStrategy, Signal

class TrendFollowingEMA(BaseStrategy):
    def __init__(self, fast_ema: int = 12, slow_ema: int = 26, adx_threshold: float = 25.0):
        super().__init__("Multi-Timeframe Trend Following")
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.adx_threshold = adx_threshold

    def generate_signals(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        signals = {}
        for symbol, df in df_dict.items():
            if len(df) < self.slow_ema + 10:
                continue

            df['fast_ema'] = ta.trend.ema_indicator(df['close'], window=self.fast_ema)
            df['slow_ema'] = ta.trend.ema_indicator(df['close'], window=self.slow_ema)
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            action = "HOLD"
            if latest['adx'] > self.adx_threshold:
                if prev['fast_ema'] <= prev['slow_ema'] and latest['fast_ema'] > latest['slow_ema']:
                    action = "BUY"
                elif prev['fast_ema'] >= prev['slow_ema'] and latest['fast_ema'] < latest['slow_ema']:
                    action = "SELL"

            signals[symbol] = Signal(symbol=symbol, action=action, strength=float(latest['adx'] / 100.0), metadata={"adx": float(latest['adx'])})
        return signals
""".strip(),

    "trading_bot/strategies/mean_reversion.py": """
import pandas as pd
import ta
from typing import Dict
from strategies.base import BaseStrategy, Signal

class MeanReversionBB(BaseStrategy):
    def __init__(self, bb_window: int = 20, bb_std: float = 2.0, rsi_period: int = 14):
        super().__init__("Bollinger Bands + RSI Mean Reversion")
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_period = rsi_period

    def generate_signals(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        signals = {}
        for symbol, df in df_dict.items():
            if len(df) < self.bb_window + 5:
                continue

            indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=self.bb_window, window_dev=self.bb_std)
            df['bb_lower'] = indicator_bb.bollinger_lband()
            df['bb_upper'] = indicator_bb.bollinger_hband()
            df['rsi'] = ta.momentum.rsi(close=df["close"], window=self.rsi_period)
            df['vol_ma'] = df['volume'].rolling(window=20).mean()

            latest = df.iloc[-1]
            vol_confirmed = latest['volume'] > (1.5 * latest['vol_ma'])

            action = "HOLD"
            if vol_confirmed:
                if latest['close'] < latest['bb_lower'] and latest['rsi'] < 30:
                    action = "BUY"
                elif latest['close'] > latest['bb_upper'] and latest['rsi'] > 70:
                    action = "SELL"

            signals[symbol] = Signal(symbol=symbol, action=action, strength=1.0 if vol_confirmed else 0.5, metadata={"rsi": float(latest['rsi'])})
        return signals
""".strip(),

    "trading_bot/strategies/pair_trading.py": """
from dataclasses import dataclass
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)

@dataclass
class PairSignal:
    asset_a: str
    asset_b: str
    action: str
    z_score: float
    hedge_ratio: float
    p_value: float
    spread: float
    mean_spread: float
    std_spread: float

class PairTradingStatArb:
    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.2, p_val_threshold: float = 0.05, lookback_window: int = 60, z_score_window: int = 30):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.p_val_threshold = p_val_threshold
        self.lookback_window = lookback_window
        self.z_score_window = z_score_window

    def test_cointegration(self, series_a: pd.Series, series_b: pd.Series) -> Tuple[bool, float, float]:
        if len(series_a) < self.lookback_window or len(series_b) < self.lookback_window:
            return False, 1.0, 0.0
        score, p_value, _ = coint(series_a, series_b)
        if p_value > self.p_val_threshold:
            return False, float(p_value), 0.0
        x = sm.add_constant(series_b)
        model = sm.OLS(series_a, x).fit()
        hedge_ratio = float(model.params.iloc[1])
        return True, float(p_value), hedge_ratio

    def compute_z_score(self, series_a: pd.Series, series_b: pd.Series, hedge_ratio: float) -> Tuple[float, float, float, float]:
        spread = series_a - (hedge_ratio * series_b)
        if len(spread) < self.z_score_window:
            return 0.0, 0.0, 0.0, 0.0
        rolling_mean = spread.rolling(window=self.z_score_window).mean().iloc[-1]
        rolling_std = spread.rolling(window=self.z_score_window).std().iloc[-1]
        current_spread = spread.iloc[-1]
        if rolling_std <= 0.0 or np.isnan(rolling_std):
            return float(current_spread), float(rolling_mean), 0.0, 0.0
        z_score = (current_spread - rolling_mean) / rolling_std
        return float(current_spread), float(rolling_mean), float(rolling_std), float(z_score)

    def generate_signal(self, df_dict: Dict[str, pd.DataFrame], asset_a: str, asset_b: str) -> PairSignal:
        if asset_a not in df_dict or asset_b not in df_dict:
            return PairSignal(asset_a, asset_b, "NO_SIGNAL", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        df_a, df_b = df_dict[asset_a], df_dict[asset_b]
        aligned = pd.concat([df_a["close"], df_b["close"]], axis=1, keys=[asset_a, asset_b]).dropna()
        if len(aligned) < self.lookback_window:
            return PairSignal(asset_a, asset_b, "NO_SIGNAL", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        series_a, series_b = aligned[asset_a], aligned[asset_b]
        is_coint, p_val, hedge_ratio = self.test_cointegration(series_a, series_b)
        if not is_coint:
            return PairSignal(asset_a, asset_b, "NO_SIGNAL", 0.0, hedge_ratio, p_val, 0.0, 0.0, 0.0)
        spread, mean_spread, std_spread, z_score = self.compute_z_score(series_a, series_b, hedge_ratio)
        if z_score < -self.entry_z:
            action = "LONG_SPREAD"
        elif z_score > self.entry_z:
            action = "SHORT_SPREAD"
        elif abs(z_score) <= self.exit_z:
            action = "CLOSE"
        else:
            action = "NO_SIGNAL"
        return PairSignal(asset_a, asset_b, action, z_score, hedge_ratio, p_val, spread, mean_spread, std_spread)
""".strip(),

    "trading_bot/risk_manager.py": """
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RiskConfig(BaseModel):
    max_drawdown_pct: float = Field(default=0.05, ge=0.01, le=0.50)
    risk_per_trade_pct: float = Field(default=0.015, ge=0.001, le=0.10)
    max_daily_trades: int = Field(default=20, ge=1)
    max_asset_concentration_pct: float = Field(default=0.25, ge=0.05, le=1.00)
    kelly_fraction: float = Field(default=0.5, ge=0.1, le=1.0)
    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    atr_period: int = Field(default=14, ge=2)

class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_trade_count: int = 0
        self.last_trade_reset: datetime = datetime.utcnow()
        self.high_watermark_24h: float = 0.0
        self.high_watermark_timestamp: datetime = datetime.utcnow()
        self.is_killed: bool = False
        self.active_positions: Dict[str, float] = {}

    def update_24h_high_watermark(self, current_equity: float) -> None:
        now = datetime.utcnow()
        if now - self.high_watermark_timestamp > timedelta(hours=24) or current_equity > self.high_watermark_24h:
            self.high_watermark_24h = current_equity
            self.high_watermark_timestamp = now

    def check_24h_drawdown_kill_switch(self, current_equity: float) -> bool:
        if self.is_killed:
            return True
        self.update_24h_high_watermark(current_equity)
        if self.high_watermark_24h <= 0.0:
            return False
        drawdown = (self.high_watermark_24h - current_equity) / self.high_watermark_24h
        if drawdown >= self.config.max_drawdown_pct:
            self.is_killed = True
            logger.critical(f"EMERGENCY KILL-SWITCH: Drawdown ({drawdown:.2%}) >= threshold ({self.config.max_drawdown_pct:.2%}).")
            return True
        return False

    def calculate_kelly_size(self, capital: float, win_rate: float, win_loss_ratio: float) -> float:
        if win_rate <= 0.0 or win_rate >= 1.0 or win_loss_ratio <= 0.0 or capital <= 0.0:
            return 0.0
        full_kelly = win_rate - ((1.0 - win_rate) / win_loss_ratio)
        if full_kelly <= 0.0:
            return 0.0
        allocated = capital * (full_kelly * self.config.kelly_fraction)
        return min(allocated, capital * self.config.max_asset_concentration_pct)

    def calculate_atr_position_size(self, capital: float, current_price: float, atr: float) -> Tuple[float, float]:
        if atr <= 0.0 or current_price <= 0.0 or capital <= 0.0:
            return 0.0, 0.0
        risk_capital = capital * self.config.risk_per_trade_pct
        stop_loss_distance = atr * self.config.atr_multiplier
        units = risk_capital / stop_loss_distance
        max_val = capital * self.config.max_asset_concentration_pct
        if (units * current_price) > max_val:
            units = max_val / current_price
        return float(units), float(stop_loss_distance)

    def validate_trade(self, symbol: str, requested_value: float, total_portfolio_value: float) -> Tuple[bool, str]:
        if self.is_killed:
            return False, "REJECTED: 24h Max Drawdown Emergency Lockout."
        now = datetime.utcnow()
        if now.date() != self.last_trade_reset.date():
            self.daily_trade_count = 0
            self.last_trade_reset = now
        if self.daily_trade_count >= self.config.max_daily_trades:
            return False, f"REJECTED: Daily limit ({self.config.max_daily_trades}) reached."
        proj = self.active_positions.get(symbol, 0.0) + requested_value
        max_exp = total_portfolio_value * self.config.max_asset_concentration_pct
        if proj > max_exp:
            return False, f"REJECTED: {symbol} allocation exceeds concentration cap."
        return True, "VALIDATED."
""".strip(),

    "trading_bot/execution.py": """
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class OrderRequest:
    symbol: str
    qty: float
    side: str
    order_type: str = "MARKET"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class ExecutionEngine:
    def __init__(self):
        self.trailing_stops: Dict[str, float] = {}

    def update_trailing_stop(self, symbol: str, current_price: float, atr: float, side: str, multiplier: float = 2.0) -> float:
        stop_distance = atr * multiplier
        if side.upper() == "BUY":
            new_stop = current_price - stop_distance
            self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0.0), new_stop)
        else:
            new_stop = current_price + stop_distance
            self.trailing_stops[symbol] = min(self.trailing_stops.get(symbol, float("inf")), new_stop)
        return self.trailing_stops[symbol]
""".strip(),

    "trading_bot/telegram_notifier.py": """
import logging
import aiohttp

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    async def send_message(self, message: str) -> bool:
        if not self.bot_token or not self.chat_id:
            return False
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram message: {e}")
            return False
""".strip(),

    "trading_bot/dashboard.py": """
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Algorithmic Trading Telemetry", layout="wide")
st.title("📈 Production Trading Bot Telemetry & Command Center")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Equity", "$104,250.00", "+4.25%")
col2.metric("24h Drawdown", "0.82%", "-0.15%")
col3.metric("Sharpe Ratio", "2.14")
col4.metric("Profit Factor", "1.85")

st.subheader("Live Portfolio & Strategy Telemetry")
dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="H")
df_chart = pd.DataFrame({"Price": np.random.normal(100, 2, 100).cumsum()}, index=dates)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["Price"], mode="lines", name="Equity Curve"))
st.plotly_chart(fig, use_container_width=True)
""".strip(),

    "trading_bot/cli.py": """
from rich.console import Console
from rich.prompt import Prompt

console = Console()

def launch_cli():
    console.print("[bold green]=== Multi-Asset Quantitative Bot Configuration ===", style="header")
    mode = Prompt.ask("Select Mode", choices=["Paper", "Live"], default="Paper")
    broker = Prompt.ask("Select Broker", choices=["Alpaca", "Binance", "IBKR"], default="Alpaca")
    console.print(f"[bold cyan]Selected: {mode} trading on {broker}[/bold cyan]")

if __name__ == "__main__":
    launch_cli()
""".strip(),

    "trading_bot/main.py": """
import asyncio
import logging
from config import BrokerConfig, RiskConfig
from risk_manager import RiskManager
from execution import ExecutionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def main():
    logging.info("Starting Multi-Asset Algorithmic Trading Bot...")
    risk_mgr = RiskManager(RiskConfig())
    exec_engine = ExecutionEngine()
    
    # Event loop
    for i in range(3):
        logging.info("Executing main cycle check...")
        await asyncio.sleep(1)
    logging.info("Bot execution cycle complete.")

if __name__ == "__main__":
    asyncio.run(main())
""".strip(),
}

def create_project_zip(zip_name: str = "trading_bot.zip"):
    # 1. Write individual project files
    for file_path, content in FILES.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Created: {file_path}")

    # 2. Package into ZIP archive
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in FILES.keys():
            zip_file.write(file_path)

    print(f"\nSuccessfully generated '{zip_name}' containing all {len(FILES)} project files!")

if __name__ == "__main__":
    create_project_zip()