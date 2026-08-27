import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


class BrokerType(str, Enum):
    BINANCE = "BINANCE"
    ALPACA = "ALPACA"
    IBKR = "IBKR"


class TradingMode(str, Enum):
    LIVE = "LIVE"
    PAPER = "PAPER"


class MarketType(str, Enum):
    SPOT = "SPOT"
    FUTURES = "FUTURES"


@dataclass
class BrokerConfig:
    broker_type: BrokerType = BrokerType.BINANCE
    trading_mode: TradingMode = TradingMode.LIVE
    market_type: MarketType = MarketType.SPOT
    api_key: str = ""
    api_secret: str = ""
    futures_leverage: int = 1
    trading_pairs: List[str] = field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    )
    host: str = "127.0.0.1"
    port: int = 7497


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    max_daily_trades: int = 10


@dataclass
class TelegramConfig:
    is_enabled: bool = True
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class AppConfig:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    allocated_capital: float = 100.0


def get_config() -> AppConfig:
    broker_type_str = os.getenv("BROKER_TYPE", "BINANCE").upper()
    trading_mode_str = os.getenv("TRADING_MODE", "LIVE").upper()
    market_type_str = os.getenv("MARKET_TYPE", "SPOT").upper()

    # Parse TRADING_PAIRS environment string into clean uppercase list
    pairs_str = os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    trading_pairs_list = [p.strip().upper() for p in pairs_str.split(",") if p.strip()]

    return AppConfig(
        broker=BrokerConfig(
            broker_type=BrokerType[broker_type_str]
            if broker_type_str in BrokerType.__members__
            else BrokerType.BINANCE,
            trading_mode=TradingMode[trading_mode_str]
            if trading_mode_str in TradingMode.__members__
            else TradingMode.LIVE,
            market_type=MarketType[market_type_str]
            if market_type_str in MarketType.__members__
            else MarketType.SPOT,
            api_key=os.getenv("API_KEY", ""),
            api_secret=os.getenv("API_SECRET", ""),
            futures_leverage=int(os.getenv("FUTURES_LEVERAGE", "1")),
            trading_pairs=trading_pairs_list,
        ),
        risk=RiskConfig(
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.02")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.15")),
            max_daily_trades=int(os.getenv("MAX_DAILY_TRADES", "10")),
        ),
        telegram=TelegramConfig(
            is_enabled=os.getenv("TELEGRAM_ENABLED", "true").lower() == "true",
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        ),
        allocated_capital=float(os.getenv("ALLOCATED_CAPITAL", "100.0")),
    )