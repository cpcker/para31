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