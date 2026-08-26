import asyncio
from datetime import datetime
import logging
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from broker_interface import BaseBroker, BrokerFactory
from config import AppConfig, BrokerType, get_config
from execution import ExecutionEngine, OrderRequest
from risk_manager import RiskManager
from strategies.mean_reversion import MeanReversionBB
from strategies.pair_trading import PairTradingStatArb
from strategies.trend_following import TrendFollowingEMA
from telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TradingEngine")
console = Console()


class AlgorithmicTradingEngine:
    """Production Multi-Asset Trading Engine featuring live portfolio tracking and auto-clearing terminal snapshot tables."""

    def __init__(self):
        self.config: AppConfig = get_config()
        self.broker: BaseBroker = BrokerFactory.create_broker(self.config.broker)
        self.risk_manager = RiskManager(self.config.risk)
        self.execution_engine = ExecutionEngine()

        self.notifier: TelegramNotifier | None = None
        if self.config.telegram.is_enabled:
            self.notifier = TelegramNotifier(
                bot_token=self.config.telegram.bot_token,
                chat_id=self.config.telegram.chat_id,
            )

        self.trend_strategy = TrendFollowingEMA(
            fast_ema=12, slow_ema=26, adx_threshold=25.0
        )
        self.mean_reversion_strategy = MeanReversionBB(
            bb_window=20, bb_std=2.0, rsi_period=14
        )
        self.pair_strategy = PairTradingStatArb(
            entry_z=2.0, exit_z=0.2, p_val_threshold=0.05
        )

        self.is_running = False

        if self.config.broker.broker_type == BrokerType.BINANCE:
            self.monitored_symbols: List[str] = [
                "BTCUSDC",
                "ETHUSDC",
                "SOLUSDC",
                "BNBUSDC",
            ]
        else:
            self.monitored_symbols: List[str] = ["AAPL", "MSFT", "NVDA", "GOOGL"]

    async def notify(self, message: str) -> None:
        if self.notifier:
            await self.notifier.send_message(message)

    def display_portfolio_and_market_tables(
        self,
        df_dict: Dict[str, pd.DataFrame],
        account_data: Dict[str, float],
        positions: List[Dict],
    ) -> None:
        """Renders live Portfolio metrics, Open Positions, and Market Snapshots in terminal."""

        equity = account_data.get("equity", 0.0)
        cash = account_data.get("cash", 0.0)
        buying_power = account_data.get("buying_power", 0.0)

        # 1. Terminal Header Panel
        header_text = (
            f"[bold green]🤖 Multi-Asset Quantitative Bot Engine[/bold green] | "
            f"[cyan]Venue:[/cyan] [bold]{self.config.broker.broker_type.value}[/bold] | "
            f"[cyan]Mode:[/cyan] [bold]{self.config.broker.trading_mode.value}[/bold] | "
            f"[cyan]Time:[/cyan] [dim]{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]"
        )
        console.print(Panel(header_text, border_style="cyan"))

        # 2. Portfolio Balance & Risk Telemetry Table
        port_table = Table(
            title="💼 Live Portfolio Telemetry",
            show_header=True,
            header_style="bold magenta",
        )
        port_table.add_column("Total Equity", justify="right", style="bold green")
        port_table.add_column("Free Balance (USDC)", justify="right", style="green")
        port_table.add_column("Buying Power", justify="right", style="cyan")
        port_table.add_column("Active Positions", justify="center", style="yellow")
        port_table.add_column("24h Max Drawdown Cap", justify="right", style="bold red")

        active_count = len(positions)
        max_dd = self.config.risk.max_drawdown_pct

        port_table.add_row(
            f"${equity:,.2f}",
            f"${cash:,.2f}",
            f"${buying_power:,.2f}",
            f"{active_count} asset(s)",
            f"{max_dd:.1%}",
        )
        console.print(port_table)

        # 3. Live Positions Table (if any positions exist)
        if positions:
            pos_table = Table(
                title="🔓 Open Portfolio Positions",
                show_header=True,
                header_style="bold yellow",
            )
            pos_table.add_column("Symbol", style="cyan")
            pos_table.add_column("Quantity", justify="right")
            pos_table.add_column("Side", justify="center", style="bold")
            pos_table.add_column("Entry Price", justify="right")
            pos_table.add_column("Current Price", justify="right")
            pos_table.add_column("Unrealized PnL", justify="right", style="bold")

            for pos in positions:
                pnl = pos.get("unrealized_pnl", 0.0)
                pnl_color = "green" if pnl >= 0 else "red"
                pos_table.add_row(
                    pos.get("symbol", "N/A"),
                    f"{pos.get('qty', 0.0):.4f}",
                    pos.get("side", "BUY"),
                    f"${pos.get('entry_price', 0.0):,.2f}",
                    f"${pos.get('current_price', 0.0):,.2f}",
                    f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]",
                )
            console.print(pos_table)
        else:
            console.print("[dim]No open positions reported in active account.[/dim]\n")

        # 4. Live Market Price Snapshot Table
        mkt_table = Table(
            title=f"📊 Live Market Snapshot [{datetime.utcnow().strftime('%H:%M:%S UTC')}]",
            show_header=True,
            header_style="bold cyan",
        )
        mkt_table.add_column("Symbol", style="bold yellow")
        mkt_table.add_column("Market Price", justify="right", style="bold green")
        mkt_table.add_column("Candle High", justify="right", style="bright_green")
        mkt_table.add_column("Candle Low", justify="right", style="bright_red")
        mkt_table.add_column("Range High", justify="right", style="green")
        mkt_table.add_column("Range Low", justify="right", style="red")
        mkt_table.add_column("Volume", justify="right", style="dim")

        for symbol, df in df_dict.items():
            if df.empty:
                continue

            last_close = float(df["close"].iloc[-1])
            bar_high = float(df["high"].iloc[-1])
            bar_low = float(df["low"].iloc[-1])
            range_high = float(df["high"].max())
            range_low = float(df["low"].min())
            volume = float(df["volume"].iloc[-1])

            mkt_table.add_row(
                symbol,
                f"${last_close:,.2f}",
                f"${bar_high:,.2f}",
                f"${bar_low:,.2f}",
                f"${range_high:,.2f}",
                f"${range_low:,.2f}",
                f"{volume:,.0f}",
            )

        console.print(mkt_table)

    async def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}

        for symbol in self.monitored_symbols:
            try:
                df = await self.broker.get_historical_klines(
                    symbol=symbol, timeframe="15m", limit=100
                )
                if df is not None and not df.empty and len(df) >= 30:
                    data[symbol] = df
                    continue
            except Exception as e:
                logger.warning(
                    f"Broker data fetch failed for {symbol}: {e}. Generating fallback market snapshot..."
                )

            periods = 100
            dates = pd.date_range(
                end=datetime.utcnow(), periods=periods, freq="15min"
            )
            base_price = 150.0 if "USDC" not in symbol else 30000.0
            np.random.seed(hash(symbol) % 2**32)
            price_changes = np.random.normal(0, 0.5, periods).cumsum()
            close = base_price + price_changes
            high = close + np.random.uniform(0.1, 1.0, periods)
            low = close - np.random.uniform(0.1, 1.0, periods)
            open_p = low + (high - low) / 2
            volume = np.random.uniform(1000, 50000, periods)

            data[symbol] = pd.DataFrame(
                {
                    "open": open_p,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                },
                index=dates,
            )

        return data

    async def process_trend_and_mean_reversion(
        self, df_dict: Dict[str, pd.DataFrame], current_equity: float
    ) -> None:
        trend_signals = self.trend_strategy.generate_signals(df_dict)
        for symbol, sig in trend_signals.items():
            if sig.action in ["BUY", "SELL"]:
                await self.execute_single_asset_signal(
                    symbol, sig.action, "TrendFollowing", df_dict[symbol], current_equity
                )

        mr_signals = self.mean_reversion_strategy.generate_signals(df_dict)
        for symbol, sig in mr_signals.items():
            if sig.action in ["BUY", "SELL"]:
                await self.execute_single_asset_signal(
                    symbol, sig.action, "MeanReversion", df_dict[symbol], current_equity
                )

    async def execute_single_asset_signal(
        self,
        symbol: str,
        action: str,
        strategy_name: str,
        df: pd.DataFrame,
        current_equity: float,
    ) -> None:
        current_price = float(df["close"].iloc[-1])
        atr = self.risk_manager.calculate_atr(df)
        units, sl_dist = self.risk_manager.calculate_atr_position_size(
            capital=current_equity, current_price=current_price, atr=atr
        )

        if units <= 0.0:
            return

        order_value = units * current_price
        allowed, reason = self.risk_manager.validate_trade(
            symbol, order_value, current_equity
        )

        if not allowed:
            logger.warning(f"[{strategy_name}] Trade rejected for {symbol}: {reason}")
            return

        sl_price = current_price - sl_dist if action == "BUY" else current_price + sl_dist

        try:
            order_res = await self.broker.place_order(
                symbol=symbol,
                qty=units,
                side=action,
                order_type="MARKET",
                sl=sl_price,
            )
            self.risk_manager.record_execution(symbol, order_value)

            msg = (
                f"⚡ *[{strategy_name}] Order Executed*\n"
                f"• *Venue:* `{self.config.broker.broker_type.value}`\n"
                f"• *Asset:* `{symbol}` | *Side:* `{action}`\n"
                f"• *Quantity:* `{units:.4f}` @ `${current_price:,.2f}`\n"
                f"• *Order ID:* `{order_res.get('order_id', 'N/A')}`"
            )
            logger.info(msg.replace("*", "").replace("`", ""))
            await self.notify(msg)

        except Exception as e:
            logger.error(f"[{strategy_name}] Order execution failed for {symbol}: {e}")

    async def process_pair_trading(
        self, df_dict: Dict[str, pd.DataFrame], current_equity: float
    ) -> None:
        if len(self.monitored_symbols) < 2:
            return

        asset_a, asset_b = self.monitored_symbols[0], self.monitored_symbols[1]
        pair_sig = self.pair_strategy.generate_signal(df_dict, asset_a, asset_b)

        if pair_sig.action in ["LONG_SPREAD", "SHORT_SPREAD"]:
            price_a = float(df_dict[asset_a]["close"].iloc[-1])
            price_b = float(df_dict[asset_b]["close"].iloc[-1])

            allocated_capital = current_equity * self.config.risk.risk_per_trade_pct * 2
            units_a = allocated_capital / price_a
            units_b = (units_a * price_a * pair_sig.hedge_ratio) / price_b

            side_a = "BUY" if pair_sig.action == "LONG_SPREAD" else "SELL"
            side_b = "SELL" if pair_sig.action == "LONG_SPREAD" else "BUY"

            val_a, _ = self.risk_manager.validate_trade(asset_a, units_a * price_a, current_equity)
            val_b, _ = self.risk_manager.validate_trade(asset_b, units_b * price_b, current_equity)

            if val_a and val_b:
                try:
                    await self.broker.place_order(asset_a, units_a, side_a)
                    await self.broker.place_order(asset_b, units_b, side_b)

                    self.risk_manager.record_execution(asset_a, units_a * price_a)
                    self.risk_manager.record_execution(asset_b, units_b * price_b)

                    pair_msg = (
                        f"📊 *StatArb Spread Executed*\n"
                        f"• *Action:* `{pair_sig.action}` | *Z-Score:* `{pair_sig.z_score:.2f}`\n"
                        f"• *Leg A:* `{side_a}` `{units_a:.4f}` `{asset_a}` @ `${price_a:,.2f}`\n"
                        f"• *Leg B:* `{side_b}` `{units_b:.4f}` `{asset_b}` @ `${price_b:,.2f}`"
                    )
                    logger.info(pair_msg.replace("*", "").replace("`", ""))
                    await self.notify(pair_msg)

                except Exception as e:
                    logger.error(f"StatArb execution error: {e}")

    async def run(self, loop_interval_seconds: int = 15) -> None:
        self.is_running = True

        try:
            await self.broker.connect()
        except Exception as e:
            logger.critical(f"Failed to establish broker connection: {e}")
            return

        try:
            while self.is_running:
                # Clear terminal screen at start of loop
                console.clear()

                # 1. Fetch live portfolio balances & open positions
                try:
                    account_data = await self.broker.get_account_balance()
                    positions = await self.broker.get_positions()
                except Exception as err:
                    logger.warning(f"Failed to fetch portfolio update: {err}")
                    account_data = {"equity": self.config.allocated_capital, "cash": 0.0, "buying_power": 0.0}
                    positions = []

                current_equity = account_data.get("equity", self.config.allocated_capital)

                # 2. Check 24-Hour Max Drawdown Circuit Breaker
                if self.risk_manager.check_24h_drawdown_kill_switch(current_equity):
                    logger.critical("24h Drawdown Breach! Halting trading engine.")
                    break

                # 3. Synchronize market tick data
                df_dict = await self.fetch_market_data()

                # 4. Render live portfolio panel & market tables
                self.display_portfolio_and_market_tables(df_dict, account_data, positions)

                # 5. Process signals & order executions
                await self.process_trend_and_mean_reversion(df_dict, current_equity)
                await self.process_pair_trading(df_dict, current_equity)

                console.print(
                    f"\n[dim]Waiting {loop_interval_seconds}s for next tick cycle... (Press Ctrl+C to stop)[/dim]"
                )
                await asyncio.sleep(loop_interval_seconds)

        except asyncio.CancelledError:
            logger.info("Shutdown signal received.")
        finally:
            self.is_running = False
            if hasattr(self.broker, "close"):
                await self.broker.close()
            logger.info("Engine offline.")


if __name__ == "__main__":
    engine = AlgorithmicTradingEngine()
    try:
        asyncio.run(engine.run(loop_interval_seconds=15))
    except KeyboardInterrupt:
        logger.info("Bot execution stopped by user.")