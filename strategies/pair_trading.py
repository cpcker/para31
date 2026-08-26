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