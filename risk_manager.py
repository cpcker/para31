import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple
import pandas as pd
import ta

logger = logging.getLogger("TradingEngine")


class RiskManager:
    """Production Risk Manager: handles ATR dynamic volatility sizing, 
    trade validation filters, daily exposure caps, and 24h drawdown kill-switch.
    """

    def __init__(self, config):
        self.config = config
        self.daily_executions: Dict[str, float] = {}
        self.peak_equity: float = 0.0
        self.last_equity_reset: datetime = datetime.utcnow()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates Average True Range (ATR) to size dynamic volatility stop-losses."""
        if df.empty or len(df) < period:
            return 0.0

        try:
            atr_series = ta.volatility.AverageTrueRange(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=period,
            ).average_true_range()

            atr_val = atr_series.iloc[-1]
            return float(atr_val) if not pd.isna(atr_val) else 0.0
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0

    def calculate_atr_position_size(
        self, capital: float, current_price: float, atr: float
    ) -> Tuple[float, float]:
        """Calculates position units and stop distance based on account equity and market volatility.
        Ensures trades respect Binance's ~$10 minimum notional requirement.
        """
        if current_price <= 0.0 or capital <= 0.0:
            return 0.0, 0.0

        # Default fallback stop loss if ATR is invalid
        stop_loss_distance = (atr * 2.0) if atr > 0 else (current_price * 0.02)

        # Risk allocation per trade (e.g., 2% of total equity)
        risk_per_trade_pct = getattr(self.config, "risk_per_trade_pct", 0.02)
        risk_amount = capital * risk_per_trade_pct

        # Position units derived from ATR risk distance
        units = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0.0

        # Cap maximum position size at 25% of account capital per trade
        max_allowed_units = (capital * 0.25) / current_price
        units = min(units, max_allowed_units)

        # Enforce Binance Spot ~$10 minimum notional order filter
        order_value = units * current_price
        if order_value < 10.50:
            units = 10.50 / current_price

        # Absolute cap: never order more than available capital
        if (units * current_price) > capital:
            units = capital / current_price

        return units, stop_loss_distance

    def validate_trade(
        self, symbol: str, order_value: float, equity: float
    ) -> Tuple[bool, str]:
        """Validates trade execution against capital exposure limits."""
        if equity <= 0:
            return False, "Equity is zero or negative."

        if order_value > equity * 0.95:
            return False, f"Order value (${order_value:.2f}) exceeds max safe capital allocation."

        # Check total daily execution cap to prevent runaway trade looping
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily_spent = self.daily_executions.get(today, 0.0)
        max_daily_limit = equity * 3.0  # Max 300% account turnover per day

        if daily_spent + order_value > max_daily_limit:
            return False, f"Daily turnover limit exceeded (${daily_spent:.2f} spent today)."

        return True, "Trade approved."

    def record_execution(self, symbol: str, order_value: float) -> None:
        """Tracks daily trade turnover to prevent overtrading."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        self.daily_executions[today] = self.daily_executions.get(today, 0.0) + order_value
        logger.info(f"Recorded trade execution for {symbol}: ${order_value:.2f}")

    def check_24h_drawdown_kill_switch(self, current_equity: float) -> bool:
        """Circuit breaker: Halts all trading if account equity drops past max allowed drawdown."""
        now = datetime.utcnow()

        # Reset peak equity tracking window every 24 hours
        if now - self.last_equity_reset > timedelta(hours=24) or current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.last_equity_reset = now

        if self.peak_equity <= 0:
            return False

        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
        max_allowed_dd = getattr(self.config, "max_drawdown_pct", 0.15)  # 15% max drawdown

        if drawdown_pct >= max_allowed_dd:
            logger.critical(
                f"🚨 CIRCUIT BREAKER TRIGGERED! 24h Drawdown: {drawdown_pct:.2%} "
                f"(Peak: ${self.peak_equity:.2f}, Current: ${current_equity:.2f})"
            )
            return True

        return False
