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