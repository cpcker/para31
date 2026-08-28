import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from config import RiskConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """Production Risk Management Module handling position sizing, daily turnover caps,
    leftover cash sweeping, margin buffers, and 24-hour peak equity drawdown tracking.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_executions: Dict[str, float] = {}
        self.peak_equity: float = 0.0

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates Average True Range (ATR) over high, low, and close columns."""
        if df.empty or len(df) < period:
            return 0.0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return float(atr) if not np.isnan(atr) else 0.0

    def calculate_atr_position_size(
        self,
        capital: float,
        current_price: float,
        atr: float,
        available_cash: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Calculates position quantity and Stop Loss distance with strict margin safeguards."""
        if atr <= 0 or current_price <= 0 or capital <= 0:
            return 0.0, 0.0

        sl_distance = atr * getattr(self.config, "atr_multiplier", 1.5)
        risk_per_trade_pct = getattr(self.config, "risk_per_trade_pct", 0.02)
        risk_amount = capital * risk_per_trade_pct

        # Initial risk-based position calculation
        target_notional = (risk_amount / sl_distance) * current_price

        # 1. HARD CAP: Never allow a single trade to exceed 18% of total account equity
        max_allowed_notional = capital * 0.18
        if target_notional > max_allowed_notional:
            target_notional = max_allowed_notional

        # 2. MARGIN BUFFER: Leave 15% free cash room for Binance SL margin reservation & fees
        if available_cash is not None and available_cash > 0:
            sweepable_cash = available_cash * 0.85
            if target_notional > sweepable_cash:
                target_notional = sweepable_cash

        # 3. MINIMUM NOTIONAL GUARDRAIL: Require at least $10.00 USDT per position
        if target_notional < 10.0:
            return 0.0, sl_distance

        units = target_notional / current_price
        return units, sl_distance

    def validate_trade(
        self, symbol: str, trade_notional_value: float, current_equity: float
    ) -> Tuple[bool, str]:
        """Enforces trade validation rules and daily volume limits."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        accumulated_turnover = self.daily_executions.get(today_str, 0.0)

        max_turnover_limit = current_equity * getattr(self.config, "max_daily_turnover_mult", 5.0)

        if (accumulated_turnover + trade_notional_value) > max_turnover_limit:
            reason = (
                f"24h Volume Cap Exceeded! Current: ${accumulated_turnover:,.2f} + "
                f"Trade: ${trade_notional_value:,.2f} > Max Limit: ${max_turnover_limit:,.2f}"
            )
            return False, reason

        return True, "Trade Approved"

    def record_execution(self, symbol: str, trade_notional_value: float) -> None:
        """Tracks daily execution volume for turnover reporting."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        self.daily_executions[today_str] = (
            self.daily_executions.get(today_str, 0.0) + trade_notional_value
        )

    def check_24h_drawdown_kill_switch(self, current_equity: float) -> bool:
        """Monitors total portfolio drawdown relative to 24h peak equity."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            return False

        if self.peak_equity <= 0:
            return False

        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
        max_dd_cap = getattr(self.config, "max_drawdown_pct", 0.15)

        if drawdown_pct >= max_dd_cap:
            return True

        return False