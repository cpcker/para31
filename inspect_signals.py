import asyncio
import logging
import numpy as np
import pandas as pd
import ta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import get_config, BrokerType
from broker_interface import BrokerFactory

console = Console()
logging.basicConfig(level=logging.ERROR)


async def inspect_market_state():
    config = get_config()
    broker = BrokerFactory.create_broker(config.broker)
    
    try:
        await broker.connect()

        # Dynamic symbol selection (USDC pairs for Binance EU compliance)
        if config.broker.broker_type == BrokerType.BINANCE:
            symbols = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "BNBUSDC"]
        else:
            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL"]

        console.clear()
        
        # 1. Terminal Header Panel
        header_text = (
            f"[bold yellow]🔍 Live Market & Portfolio Inspection[/bold yellow] | "
            f"[cyan]Venue:[/cyan] [bold]{config.broker.broker_type.value}[/bold] | "
            f"[cyan]Mode:[/cyan] [bold]{config.broker.trading_mode.value}[/bold]"
        )
        console.print(Panel(header_text, border_style="yellow"))

        # 2. Fetch Live Portfolio & Positions Telemetry
        try:
            account_data = await broker.get_account_balance()
            positions = await broker.get_positions()
        except Exception as err:
            console.print(f"[red]Error fetching portfolio balance: {err}[/red]")
            account_data = {"equity": 0.0, "cash": 0.0, "buying_power": 0.0}
            positions = []

        equity = account_data.get("equity", 0.0)
        cash = account_data.get("cash", 0.0)
        buying_power = account_data.get("buying_power", 0.0)

        port_table = Table(
            title="💼 Live Portfolio Telemetry",
            show_header=True,
            header_style="bold magenta",
        )
        port_table.add_column("Total Equity", justify="right", style="bold green")
        port_table.add_column("Free Balance (USDC)", justify="right", style="green")
        port_table.add_column("Buying Power", justify="right", style="cyan")
        port_table.add_column("Active Positions", justify="center", style="yellow")

        port_table.add_row(
            f"${equity:,.2f}",
            f"${cash:,.2f}",
            f"${buying_power:,.2f}",
            f"{len(positions)} position(s)",
        )
        console.print(port_table)

        # 3. Live Positions Table (if any exist)
        if positions:
            pos_table = Table(
                title="🔓 Open Positions in Portfolio",
                show_header=True,
                header_style="bold yellow",
            )
            pos_table.add_column("Symbol", style="cyan")
            pos_table.add_column("Quantity", justify="right")
            pos_table.add_column("Side", justify="center", style="bold")
            pos_table.add_column("Entry Price", justify="right")
            pos_table.add_column("Current Price", justify="right")

            for pos in positions:
                pos_table.add_row(
                    pos.get("symbol", "N/A"),
                    f"{pos.get('qty', 0.0):.4f}",
                    pos.get("side", "BUY"),
                    f"${pos.get('entry_price', 0.0):,.2f}",
                    f"${pos.get('current_price', 0.0):,.2f}",
                )
            console.print(pos_table)
        else:
            console.print("[dim]No open positions reported in active account.[/dim]\n")

        # 4. Fetch Market Kline Data
        df_dict = {}
        for symbol in symbols:
            try:
                df = await broker.get_historical_klines(
                    symbol=symbol, timeframe="15m", limit=100
                )
                if df is not None and not df.empty and len(df) >= 30:
                    df_dict[symbol] = df
            except Exception as e:
                console.print(f"[red]Error fetching {symbol}: {e}[/red]")

        if not df_dict:
            console.print("[red]No market data received from active broker.[/red]")
            return

        # 5. Technical Indicators Table
        table = Table(
            title="📊 Live Strategy Telemetry & Technical Indicators",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Symbol", style="cyan")
        table.add_column("Price", justify="right")
        table.add_column("EMA 12/26", justify="center")
        table.add_column("ADX (14)", justify="right")
        table.add_column("RSI (14)", justify="right")
        table.add_column("BB Lower / Upper", justify="center")
        table.add_column("Vol Ratio", justify="right")
        table.add_column("Signal State", style="bold")

        for symbol, df in df_dict.items():
            close = df["close"]
            price = close.iloc[-1]

            # Trend Indicators
            fast_ema = ta.trend.ema_indicator(close, window=12).iloc[-1]
            slow_ema = ta.trend.ema_indicator(close, window=26).iloc[-1]
            adx = ta.trend.adx(df["high"], df["low"], close, window=14).iloc[-1]

            # Mean Reversion Indicators
            rsi = ta.momentum.rsi(close, window=14).iloc[-1]
            bb_band = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
            bb_lower = bb_band.bollinger_lband().iloc[-1]
            bb_upper = bb_band.bollinger_hband().iloc[-1]

            vol_ma = df["volume"].rolling(window=20).mean().iloc[-1]
            curr_vol = df["volume"].iloc[-1]
            vol_ratio = curr_vol / vol_ma if vol_ma > 0 else 1.0

            ema_state = "FAST>SLOW" if fast_ema > slow_ema else "FAST<SLOW"

            trend_signal = "NEUTRAL"
            if adx > 25.0:
                trend_signal = "BUY" if fast_ema > slow_ema else "SELL"

            mr_signal = "NEUTRAL"
            if vol_ratio > 1.5:
                if price < bb_lower and rsi < 30:
                    mr_signal = "BUY"
                elif price > bb_upper and rsi > 70:
                    mr_signal = "SELL"

            signal_str = f"Trend: {trend_signal} | MR: {mr_signal}"

            table.add_row(
                symbol,
                f"${price:,.2f}",
                f"{fast_ema:.2f} / {slow_ema:.2f} ({ema_state})",
                f"{adx:.1f} {'🔥' if adx > 25 else '😴'}",
                f"{rsi:.1f}",
                f"${bb_lower:.2f} / ${bb_upper:.2f}",
                f"{vol_ratio:.2f}x",
                f"[green]{signal_str}[/green]"
                if ("BUY" in signal_str or "SELL" in signal_str)
                else f"[dim]{signal_str}[/dim]",
            )

        console.print(table)

        # 6. StatArb Pair Trading Inspection Table
        if len(df_dict) >= 2:
            keys = list(df_dict.keys())
            sym_a, sym_b = keys[0], keys[1]
            df_a, df_b = df_dict[sym_a], df_dict[sym_b]
            aligned = pd.concat(
                [df_a["close"], df_b["close"]], axis=1, keys=[sym_a, sym_b]
            ).dropna()

            if len(aligned) >= 60:
                score, p_val, _ = coint(aligned[sym_a], aligned[sym_b])
                x = sm.add_constant(aligned[sym_b])
                model = sm.OLS(aligned[sym_a], x).fit()
                hedge_ratio = float(model.params.iloc[1])

                spread = aligned[sym_a] - (hedge_ratio * aligned[sym_b])
                mean_sp = spread.rolling(30).mean().iloc[-1]
                std_sp = spread.rolling(30).std().iloc[-1]
                z_score = (spread.iloc[-1] - mean_sp) / std_sp if std_sp > 0 else 0.0

                pair_table = Table(
                    title=f"\nStatistical Arbitrage Pair Inspection [{sym_a} / {sym_b}]",
                    show_header=True,
                    header_style="bold green",
                )
                pair_table.add_column("Metric", style="cyan")
                pair_table.add_column("Value", justify="right")
                pair_table.add_column("Entry Threshold Criteria", justify="left")

                pair_table.add_row(
                    "Engle-Granger p-value",
                    f"{p_val:.4f}",
                    f"{'< 0.05 (Valid Cointegration)' if p_val < 0.05 else '>= 0.05 (Not Cointegrated)'}",
                )
                pair_table.add_row("Dynamic Hedge Ratio (Beta)", f"{hedge_ratio:.4f}", "N/A")
                pair_table.add_row(
                    "Current Spread Z-Score",
                    f"{z_score:.2f}",
                    "Long: Z < -2.0 | Short: Z > +2.0",
                )

                state_desc = "NO_SIGNAL"
                if z_score < -2.0:
                    state_desc = "[bold green]LONG_SPREAD[/bold green]"
                elif z_score > 2.0:
                    state_desc = "[bold red]SHORT_SPREAD[/bold red]"
                elif abs(z_score) <= 0.2:
                    state_desc = "[yellow]CLOSE_SPREAD[/yellow]"

                pair_table.add_row("Pair Signal Action", state_desc, "")
                console.print(pair_table)

    finally:
        # Gracefully shut down connector
        if hasattr(broker, "close"):
            await broker.close()


if __name__ == "__main__":
    asyncio.run(inspect_market_state())