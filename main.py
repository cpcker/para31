import asyncio
from datetime import datetime
import logging
import os
import sys
from typing import Dict, List, Set, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from broker_interface import BaseBroker, BrokerFactory
from config import AppConfig, BrokerType, MarketType, get_config
from execution import ExecutionEngine, OrderRequest
from risk_manager import RiskManager
from strategies.mean_reversion import MeanReversionBB
from strategies.pair_trading import PairTradingStatArb
from strategies.trend_following import TrendFollowingEMA
from telegram_notifier import TelegramNotifier

# Local timezone (EEST / UTC+3)
LOCAL_TZ = ZoneInfo("Europe/Athens")

# Force UTF-8 encoding on standard output for terminal emoji rendering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Dual Logger Setup
logger = logging.getLogger("TradingEngine")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("trading_bot.log", encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(file_formatter)
    logger.addHandler(stream_handler)

console = Console()


class AlgorithmicTradingEngine:
    """Production Multi-Asset Trading Engine supporting Spot & 1x Futures
    with position guardrails, leftover cash sweeping, single-fire attempt notifications,
    and Telegram telemetry.
    """

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

        # FINE-TUNED STRATEGY PARAMETERS
        self.trend_strategy = TrendFollowingEMA(
            fast_ema=9,
            slow_ema=21,
            adx_threshold=30.0
        )
        self.mean_reversion_strategy = MeanReversionBB(
            bb_window=20,
            bb_std=2.5,
            rsi_period=14
        )
        self.pair_strategy = PairTradingStatArb(
            entry_z=2.5,
            exit_z=0.1,
            p_val_threshold=0.01
        )

        self.is_running = False
        self.last_hourly_report_time: float = 0.0
        self.stoppage_reason: str = "Normal Shutdown / Manual Stop"

        # Track active attempt notifications to prevent phone spam on every tick
        self.attempted_signals: Dict[str, str] = {}

        if self.config.broker.broker_type == BrokerType.BINANCE:
            self.monitored_symbols: List[str] = self.config.broker.trading_pairs
        else:
            self.monitored_symbols: List[str] = ["AAPL", "MSFT", "NVDA", "GOOGL"]

    async def notify(self, message: str) -> None:
        if self.notifier:
            await self.notifier.send_message(message)

    async def send_hourly_portfolio_report(
        self, account_data: Dict[str, float], positions: List[Dict], df_dict: Dict[str, pd.DataFrame]
    ) -> None:
        """Sends an automated status report to Telegram every 60 minutes with TP/SL levels."""
        now_str = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        equity = account_data.get("equity", 0.0)
        cash = account_data.get("cash", 0.0)
        mkt_label = self.config.broker.market_type.value

        holdings_summary = ""
        if positions:
            for p in positions:
                sym = p.get("symbol", "N/A")
                qty = float(p.get("qty", 0.0))
                side = p.get("side", "HOLD")
                entry_price = float(p.get("entry_price", 0.0))
                curr_price = float(p.get("current_price", 0.0))
                val = qty * curr_price

                # Calculate TP/SL values for telemetry display
                tp_str, sl_str = "N/A", "N/A"
                df_s = df_dict.get(sym)
                if df_s is not None and not df_s.empty and entry_price > 0:
                    atr = self.risk_manager.calculate_atr(df_s)
                    _, sl_dist = self.risk_manager.calculate_atr_position_size(
                        capital=equity, current_price=entry_price, atr=atr
                    )
                    sl_dist = sl_dist if sl_dist > 0 else entry_price * 0.02

                    if side == "BUY":
                        sl_val = entry_price - sl_dist
                        tp_val = entry_price + (sl_dist * 1.5)
                    else:
                        sl_val = entry_price + sl_dist
                        tp_val = entry_price - (sl_dist * 1.5)
                    
                    tp_str = f"${tp_val:,.2f}"
                    sl_str = f"${sl_val:,.2f}"

                holdings_summary += (
                    f"  • *{sym}* ({side})\n"
                    f"    Qty: `{qty:.4f}` (~${val:,.2f})\n"
                    f"    Entry: `${entry_price:,.2f}` | Mark: `${curr_price:,.2f}`\n"
                    f"    🎯 TP: `{tp_str}` | 🛡️ SL: `{sl_str}`\n\n"
                )
        else:
            holdings_summary = "  • _No active positions held (100% Cash/Margin)_\n\n"

        today_str = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
        daily_turnover = self.risk_manager.daily_executions.get(today_str, 0.0)

        report_msg = (
            f"📈 *HOURLY TELEMETRY REPORT [{mkt_label}]*\n"
            f"⏰ *Time:* `{now_str}`\n\n"
            f"💰 *Total Account Equity:* `${equity:,.2f}`\n"
            f"💵 *Available Free Cash:* `${cash:,.2f}`\n"
            f"📊 *24h Volume Turnover:* `${daily_turnover:,.2f}`\n\n"
            f"📦 *Active Wallet Positions:*\n{holdings_summary}"
            f"🟢 *Status:* Running normally on `{self.config.broker.broker_type.value}` ({mkt_label})."
        )

        logger.info(f"Dispatching hourly status report for ${equity:,.2f} account equity.")
        await self.notify(report_msg)

    def display_portfolio_and_market_tables(
        self,
        df_dict: Dict[str, pd.DataFrame],
        account_data: Dict[str, float],
        positions: List[Dict],
    ) -> None:
        equity = account_data.get("equity", 0.0)
        cash = account_data.get("cash", 0.0)
        buying_power = account_data.get("buying_power", 0.0)

        now_str = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        mkt_label = self.config.broker.market_type.value

        header_text = (
            f"[bold green]🤖 Multi-Asset Quantitative Bot Engine[/bold green] | "
            f"[cyan]Venue:[/cyan] [bold]{self.config.broker.broker_type.value} ({mkt_label})[/bold] | "
            f"[cyan]Mode:[/cyan] [bold]{self.config.broker.trading_mode.value}[/bold] | "
            f"[cyan]Time:[/cyan] [dim]{now_str}[/dim]"
        )
        console.print(Panel(header_text, border_style="cyan"))

        port_table = Table(
            title="💼 Live Portfolio Telemetry",
            show_header=True,
            header_style="bold magenta",
        )
        port_table.add_column("Total Equity", justify="right", style="bold green")
        port_table.add_column("Free Cash / Margin", justify="right", style="green")
        port_table.add_column("Buying Power", justify="right", style="cyan")
        port_table.add_column("Active Holdings", justify="center", style="yellow")
        port_table.add_column("24h Max Drawdown Cap", justify="right", style="bold red")

        active_count = len(positions)
        max_dd = getattr(self.config.risk, "max_drawdown_pct", 0.15)

        port_table.add_row(
            f"${equity:,.2f}",
            f"${cash:,.2f}",
            f"${buying_power:,.2f}",
            f"{active_count} position(s)",
            f"{max_dd:.1%}",
        )
        console.print(port_table)

        if positions:
            pos_table = Table(
                title=f"🔓 Open Portfolio Positions [{mkt_label}]",
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

        mkt_table = Table(
            title=f"📊 Live Market Snapshot [{datetime.now(LOCAL_TZ).strftime('%H:%M:%S %Z')}]",
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
                end=datetime.now(LOCAL_TZ), periods=periods, freq="15min"
            )
            base_price = 150.0 if "USDC" not in symbol and "USDT" not in symbol else 30000.0
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

    async def process_software_stop_loss_guardrail(
        self, positions: List[Dict], df_dict: Dict[str, pd.DataFrame], current_equity: float
    ) -> None:
        """Backup safeguard: Market closes any position that breaches ATR Stop Loss limits."""
        for pos in positions:
            sym = pos.get("symbol")
            side = pos.get("side")
            curr_p = float(pos.get("current_price", 0.0))
            entry_p = float(pos.get("entry_price", 0.0))
            qty = float(pos.get("qty", 0.0))

            if qty > 0 and entry_p > 0 and curr_p > 0:
                df_s = df_dict.get(sym)
                if df_s is not None and not df_s.empty:
                    atr = self.risk_manager.calculate_atr(df_s)
                    _, sl_dist = self.risk_manager.calculate_atr_position_size(
                        capital=current_equity, current_price=entry_p, atr=atr
                    )
                    sl_dist = sl_dist if sl_dist > 0 else entry_p * 0.02

                    stop_triggered = False
                    if side == "BUY" and curr_p <= (entry_p - sl_dist):
                        stop_triggered = True
                    elif side == "SELL" and curr_p >= (entry_p + sl_dist):
                        stop_triggered = True

                    if stop_triggered:
                        close_side = "SELL" if side == "BUY" else "BUY"
                        logger.warning(
                            f"🚨 [SOFTWARE STOP LOSS TRIGGERED] {sym} @ ${curr_p:,.2f}! Market closing position..."
                        )
                        try:
                            await self.broker.place_order(sym, qty, close_side)
                            self.attempted_signals.pop(sym, None)  # Reset signal cache
                            await self.notify(
                                f"🚨 *[SOFTWARE STOP LOSS EXECUTED]*\n"
                                f"• *Asset:* `{sym}`\n"
                                f"• *Side:* `{side}`\n"
                                f"• *Entry:* `${entry_p:,.2f}`\n"
                                f"• *Exit:* `${curr_p:,.2f}`"
                            )
                        except Exception as close_err:
                            logger.error(f"Failed software stop loss close for {sym}: {close_err}")

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
        base_asset = symbol.replace("USDC", "").replace("USDT", "").replace("EUR", "")
        is_futures = self.config.broker.market_type == MarketType.FUTURES

        positions = await self.broker.get_positions()
        held_pos = next((p for p in positions if p.get("symbol") in [symbol, base_asset]), None)
        held_qty = float(held_pos["qty"]) if held_pos else 0.0
        held_side = held_pos.get("side", "") if held_pos else ""

        # DUAL-DIRECTION GUARDRAILS:
        if is_futures:
            if action == "BUY" and held_qty > 0 and held_side == "BUY":
                logger.info(
                    f"[{action}] Strategy: {strategy_name} | Asset: {symbol} | SKIPPED (Already holding LONG position)"
                )
                return
            if action == "SELL" and held_qty > 0 and held_side == "SELL":
                logger.info(
                    f"[{action}] Strategy: {strategy_name} | Asset: {symbol} | SKIPPED (Already holding SHORT position)"
                )
                return
        else:
            if action == "SELL" and held_qty <= 0:
                logger.info(
                    f"[{action}] Strategy: {strategy_name} | Asset: {symbol} | SKIPPED (No spot balance held)"
                )
                return

        # Fetch live free cash for leftover sweeping calculation
        account_data = await self.broker.get_account_balance()
        available_cash = account_data.get("cash", 0.0)

        atr = self.risk_manager.calculate_atr(df)
        units, sl_dist = self.risk_manager.calculate_atr_position_size(
            capital=current_equity,
            current_price=current_price,
            atr=atr,
            available_cash=available_cash,
        )

        if action == "BUY":
            sl_price = current_price - sl_dist if sl_dist > 0 else current_price * 0.98
            tp_price = current_price + (sl_dist * 1.5) if sl_dist > 0 else current_price * 1.03
        else:
            sl_price = current_price + sl_dist if sl_dist > 0 else current_price * 1.02
            tp_price = current_price - (sl_dist * 1.5) if sl_dist > 0 else current_price * 0.97

        if units <= 0.0:
            return

        order_value = units * current_price

        # SINGLE-FIRE ATTEMPT NOTIFICATION LOGIC
        # Log to terminal/file on EVERY tick
        logger.info(
            f"🚀 [ATTEMPTING {action}] Strategy: {strategy_name} | Asset: {symbol} | "
            f"Qty: {units:.4f} | Entry: ${current_price:,.2f} | TP: ${tp_price:,.2f} | SL: ${sl_price:,.2f}"
        )

        # Send to Telegram ONLY IF this specific action hasn't been notified yet for this symbol
        if self.attempted_signals.get(symbol) != action:
            self.attempted_signals[symbol] = action
            attempt_msg = (
                f"🚀 *[ATTEMPTING {action}]*\n"
                f"• *Venue:* `{self.config.broker.broker_type.value} ({self.config.broker.market_type.value})`\n"
                f"• *Strategy:* `{strategy_name}`\n"
                f"• *Asset:* `{symbol}`\n"
                f"• *Quantity:* `{units:.4f}`\n"
                f"• *Entry Price:* `${current_price:,.2f}`\n"
                f"• *Take Profit:* `${tp_price:,.2f}`\n"
                f"• *Stop Loss:* `${sl_price:,.2f}`"
            )
            await self.notify(attempt_msg)

        allowed, reason = self.risk_manager.validate_trade(
            symbol, order_value, current_equity
        )

        if not allowed:
            logger.warning(f"[{strategy_name}] Trade rejected for {symbol}: {reason}")
            return

        try:
            order_res = await self.broker.place_order(
                symbol=symbol,
                qty=units,
                side=action,
                order_type="MARKET",
                sl=sl_price,
                tp=tp_price,
            )
            self.risk_manager.record_execution(symbol, order_value)

            # Clear attempt record on successful entry
            self.attempted_signals.pop(symbol, None)

            exec_log = (
                f"✅ [{action} EXECUTED] Strategy: {strategy_name} | Asset: {symbol} | "
                f"Qty: {units:.4f} | Entry: ${current_price:,.2f} | Order ID: {order_res.get('order_id', 'N/A')}"
            )
            logger.info(exec_log)

            success_msg = (
                f"✅ *[{action} EXECUTED]*\n"
                f"• *Asset:* `{symbol}`\n"
                f"• *Side:* `{action}`\n"
                f"• *Quantity:* `{units:.4f}`\n"
                f"• *Entry Price:* `${current_price:,.2f}`\n"
                f"• *Take Profit:* `${tp_price:,.2f}`\n"
                f"• *Stop Loss:* `${sl_price:,.2f}`\n"
                f"• *Order ID:* `{order_res.get('order_id', 'N/A')}`"
            )
            await self.notify(success_msg)

        except Exception as e:
            logger.error(f"❌ [{action} FAILED] Strategy: {strategy_name} | Asset: {symbol} | Error: {e}")

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

            risk_pct = getattr(self.config.risk, "risk_per_trade_pct", 0.02)
            allocated_capital = current_equity * risk_pct * 2
            units_a = allocated_capital / price_a
            units_b = (units_a * price_a * pair_sig.hedge_ratio) / price_b

            side_a = "BUY" if pair_sig.action == "LONG_SPREAD" else "SELL"
            side_b = "SELL" if pair_sig.action == "LONG_SPREAD" else "BUY"

            logger.info(
                f"🚀 [ATTEMPTING PAIR SPREAD] Action: {pair_sig.action} | Z-Score: {pair_sig.z_score:.2f} | "
                f"Leg A ({asset_a}): {side_a} {units_a:.4f} @ ${price_a:,.2f} | "
                f"Leg B ({asset_b}): {side_b} {units_b:.4f} @ ${price_b:,.2f}"
            )

            val_a, _ = self.risk_manager.validate_trade(asset_a, units_a * price_a, current_equity)
            val_b, _ = self.risk_manager.validate_trade(asset_b, units_b * price_b, current_equity)

            if val_a and val_b:
                try:
                    await self.broker.place_order(asset_a, units_a, side_a)
                    await self.broker.place_order(asset_b, units_b, side_b)

                    self.risk_manager.record_execution(asset_a, units_a * price_a)
                    self.risk_manager.record_execution(asset_b, units_b * price_b)

                    pair_log = (
                        f"✅ [PAIR SPREAD EXECUTED] Action: {pair_sig.action} | Z-Score: {pair_sig.z_score:.2f} | "
                        f"Leg A ({asset_a}): {side_a} {units_a:.4f} @ ${price_a:,.2f} | "
                        f"Leg B ({asset_b}): {side_b} {units_b:.4f} @ ${price_b:,.2f}"
                    )
                    logger.info(pair_log)

                    pair_msg = (
                        f"📊 *StatArb Spread Executed*\n"
                        f"• *Action:* `{pair_sig.action}` | *Z-Score:* `{pair_sig.z_score:.2f}`\n"
                        f"• *Leg A:* `{side_a}` `{units_a:.4f}` `{asset_a}` @ `${price_a:,.2f}`\n"
                        f"• *Leg B:* `{side_b}` `{units_b:.4f}` `{asset_b}` @ `${price_b:,.2f}`"
                    )
                    await self.notify(pair_msg)

                except Exception as e:
                    logger.error(f"❌ [PAIR SPREAD FAILED] Error: {e}")

    async def run(self, loop_interval_seconds: int = 15) -> None:
        self.is_running = True
        self.stoppage_reason = "Normal Execution Completed"

        try:
            await self.broker.connect()
        except Exception as e:
            self.stoppage_reason = f"Broker Connection Failed: {e}"
            logger.critical(f"Failed to establish broker connection: {e}")
            return

        start_time_str = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        mkt_label = self.config.broker.market_type.value
        start_msg = (
            f"🟢 *[TRADING ENGINE STARTED]*\n"
            f"• *Venue:* `{self.config.broker.broker_type.value}`\n"
            f"• *Market Mode:* `{mkt_label}`\n"
            f"• *Trading Mode:* `{self.config.broker.trading_mode.value}`\n"
            f"• *Monitored Assets:* `{', '.join(self.monitored_symbols)}`\n"
            f"• *Time:* `{start_time_str}`\n"
            f"• *Status:* Engine initialized and monitoring tick signals."
        )
        logger.info("Bot engine initialized. Dispatching start message to Telegram.")
        await self.notify(start_msg)

        try:
            while self.is_running:
                console.clear()

                try:
                    account_data = await self.broker.get_account_balance()
                    positions = await self.broker.get_positions()
                except Exception as err:
                    logger.warning(f"Failed to fetch portfolio update: {err}")
                    account_data = {"equity": self.config.allocated_capital, "cash": 0.0, "buying_power": 0.0}
                    positions = []

                current_equity = account_data.get("equity", self.config.allocated_capital)

                if self.risk_manager.check_24h_drawdown_kill_switch(current_equity):
                    peak = self.risk_manager.peak_equity
                    dd_pct = ((peak - current_equity) / peak) if peak > 0 else 0.0
                    self.stoppage_reason = (
                        f"🚨 24h Drawdown Limit Breached!\n"
                        f"• *Current Equity:* `${current_equity:,.2f}`\n"
                        f"• *24h Peak Equity:* `${peak:,.2f}`\n"
                        f"• *Drawdown:* `{dd_pct:.2%}` (Threshold: `{self.config.risk.max_drawdown_pct:.1%}`)"
                    )
                    logger.critical(f"24h Drawdown Breach! {self.stoppage_reason}")
                    break

                df_dict = await self.fetch_market_data()

                self.display_portfolio_and_market_tables(df_dict, account_data, positions)

                # Software-level risk backup check
                await self.process_software_stop_loss_guardrail(positions, df_dict, current_equity)

                now_ts = asyncio.get_event_loop().time()
                if self.last_hourly_report_time == 0.0 or (now_ts - self.last_hourly_report_time) >= 3600:
                    await self.send_hourly_portfolio_report(account_data, positions, df_dict)
                    self.last_hourly_report_time = now_ts

                await self.process_trend_and_mean_reversion(df_dict, current_equity)
                await self.process_pair_trading(df_dict, current_equity)

                console.print(
                    f"\n[dim]Waiting {loop_interval_seconds}s for next tick cycle... (Press Ctrl+C to stop)[/dim]"
                )
                await asyncio.sleep(loop_interval_seconds)

        except asyncio.CancelledError:
            self.stoppage_reason = "Process Cancelled / Terminated"
            logger.info("Shutdown signal received.")
        except Exception as general_err:
            self.stoppage_reason = f"Runtime Error Exception: `{general_err}`"
            logger.error(f"Runtime error: {general_err}")
        finally:
            self.is_running = False

            stop_time_str = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
            stop_msg = (
                f"🔴 *[TRADING ENGINE STOPPED]*\n"
                f"• *Time:* `{stop_time_str}`\n"
                f"• *Reason for Stoppage:*\n{self.stoppage_reason}"
            )
            logger.info(f"Bot shutting down. Reason: {self.stoppage_reason}")
            await self.notify(stop_msg)

            if hasattr(self.broker, "close"):
                await self.broker.close()
            logger.info("Engine offline.")


if __name__ == "__main__":
    engine = AlgorithmicTradingEngine()
    try:
        asyncio.run(engine.run(loop_interval_seconds=15))
    except KeyboardInterrupt:
        engine.stoppage_reason = "Manual User Interruption (Ctrl + C)"
        logger.info("Bot execution stopped manually by user.")