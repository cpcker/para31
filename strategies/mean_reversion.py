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