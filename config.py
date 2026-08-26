import logging
import os
from enum import Enum
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class BrokerType(str, Enum):
    BINANCE = "BINANCE"
    ALPACA = "ALPACA"
    IBKR = "IBKR"


class BrokerConfig(BaseModel):
    """Broker connection and credential settings loaded from environment variables."""

    broker_type: BrokerType = Field(default=BrokerType.ALPACA)
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    api_key: str = Field(default="")
    api_secret: str = Field(default="")
    passphrase: Optional[str] = Field(default=None)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=7497)

    @field_validator("api_key", "api_secret")
    def validate_credentials_if_live(cls, v: str, info) -> str:
        """Enforces present credentials when live mode is selected."""
        # Note: Access context or environment directly if validation rules cross fields
        return v


class TelegramConfig(BaseModel):
    """Telegram alert configuration."""

    bot_token: Optional[str] = Field(default=None)
    chat_id: Optional[str] = Field(default=None)

    @property
    def is_enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


class RiskConfig(BaseModel):
    """Core risk guardrail and position sizing parameters."""

    max_drawdown_pct: float = Field(default=0.05, ge=0.01, le=0.50)
    risk_per_trade_pct: float = Field(default=0.015, ge=0.001, le=0.10)
    max_daily_trades: int = Field(default=20, ge=1)
    max_asset_concentration_pct: float = Field(default=0.25, ge=0.05, le=1.00)
    kelly_fraction: float = Field(default=0.5, ge=0.1, le=1.0)
    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    atr_period: int = Field(default=14, ge=2)


class StrategyConfig(BaseModel):
    name: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Root configuration container uniting broker, risk, and alert settings."""

    broker: BrokerConfig
    risk: RiskConfig
    telegram: TelegramConfig
    allocated_capital: float = Field(default=100000.0, gt=0.0)

    @classmethod
    def load_from_env(cls, env_path: str = ".env") -> "AppConfig":
        """Parses .env file, coerces types, and runs validation checks."""
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(
                f"No {env_path} file found. Falling back to environment defaults."
            )

        trading_mode_str = os.getenv("TRADING_MODE", "PAPER").upper()
        broker_type_str = os.getenv("BROKER_TYPE", "ALPACA").upper()

        broker_cfg = BrokerConfig(
            broker_type=BrokerType(broker_type_str),
            trading_mode=TradingMode(trading_mode_str),
            api_key=os.getenv("API_KEY", ""),
            api_secret=os.getenv("API_SECRET", ""),
            passphrase=os.getenv("API_PASSPHRASE") or None,
            host=os.getenv("BROKER_HOST", "127.0.0.1"),
            port=int(os.getenv("BROKER_PORT", "7497")),
        )

        # Enforce API Key presence for Live mode
        if broker_cfg.trading_mode == TradingMode.LIVE:
            if not broker_cfg.api_key or not broker_cfg.api_secret:
                raise ValueError(
                    "CRITICAL: LIVE trading mode requires valid API_KEY and API_SECRET in .env"
                )

        telegram_cfg = TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
            chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
        )

        capital_str = os.getenv("ALLOCATED_CAPITAL", "100000.0")

        app_config = cls(
            broker=broker_cfg,
            risk=RiskConfig(),
            telegram=telegram_cfg,
            allocated_capital=float(capital_str),
        )

        logger.info(
            f"Configuration initialized successfully: [{app_config.broker.broker_type.value} | "
            f"{app_config.broker.trading_mode.value} | Capital: ${app_config.allocated_capital:,.2f}]"
        )
        return app_config


# Global config factory helper
def get_config() -> AppConfig:
    return AppConfig.load_from_env()


if __name__ == "__main__":
    # Test loader directly
    config = get_config()
    print("Broker:", config.broker.broker_type)
    print("Mode:", config.broker.trading_mode)
    print("Capital:", config.allocated_capital)
    print("Telegram Enabled:", config.telegram.is_enabled)
