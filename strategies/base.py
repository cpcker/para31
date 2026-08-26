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