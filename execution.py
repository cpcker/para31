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