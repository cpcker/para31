from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from config import BrokerConfig, BrokerType, MarketType, TradingMode

logger = logging.getLogger(__name__)


class BaseBroker(ABC):
    """Abstract base broker interface normalizing multi-venue actions."""

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def get_account_balance(self) -> Dict[str, float]:
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_historical_klines(
        self, symbol: str, timeframe: str = "15m", limit: int = 100
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "MARKET",
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return symbol.replace("-", "").replace("/", "").upper()


class AlpacaBroker(BaseBroker):
    """Alpaca US Equities & Crypto Broker Implementation."""

    TIMEFRAME_MAP = {
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
    }

    def __init__(self, config: BrokerConfig):
        self.config = config
        is_paper = config.trading_mode == TradingMode.PAPER
        self.trading_client = TradingClient(
            api_key=config.api_key, secret_key=config.api_secret, paper=is_paper
        )
        self.data_client = StockHistoricalDataClient(
            api_key=config.api_key, secret_key=config.api_secret
        )

    async def connect(self) -> None:
        logger.info(f"Connected to Alpaca [{self.config.trading_mode.value}]")

    async def get_account_balance(self) -> Dict[str, float]:
        account = await asyncio.to_thread(self.trading_client.get_account)
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }

    async def get_positions(self) -> List[Dict[str, Any]]:
        if not self.client:
            return []

        if self.is_futures:
            try:
                pos_info = await self.client.futures_position_information()
                positions = []
                for pos in pos_info:
                    amt = float(pos["positionAmt"])
                    mark_price = float(pos["markPrice"])
                    
                    # 🧹 DUST FILTER: Ignore micro-positions worth less than $2.00 USDT
                    if amt != 0 and (abs(amt) * mark_price) > 2.0:
                        positions.append({
                            "symbol": pos["symbol"],
                            "qty": abs(amt),
                            "side": "BUY" if amt > 0 else "SELL",
                            "entry_price": float(pos["entryPrice"]),
                            "current_price": mark_price,
                            "unrealized_pnl": float(pos["unRealizedProfit"]),
                        })
                return positions
            except Exception as e:
                logger.error(f"Error fetching futures positions: {e}")
                return []
        else:
            acc = await self.client.get_account()
            price_map = await self._get_all_asset_prices()

            positions = []
            for b in acc["balances"]:
                asset = b["asset"]
                free = float(b["free"])
                locked = float(b["locked"])
                total = free + locked

                if total > 0:
                    pair_usdc = f"{asset}USDC"
                    pair_usdt = f"{asset}USDT"
                    pair_eur = f"{asset}EUR"

                    if asset in ["USDC", "USDT"]:
                        curr_price = 1.0
                    elif asset == "EUR":
                        curr_price = price_map.get("EURUSDT", 1.08)
                    elif pair_usdc in price_map:
                        curr_price = price_map[pair_usdc]
                    elif pair_usdt in price_map:
                        curr_price = price_map[pair_usdt]
                    elif pair_eur in price_map:
                        curr_price = price_map[pair_eur]
                    else:
                        curr_price = 0.0

                    # 🧹 DUST FILTER: Ignore spot wallets worth less than $2.00 USDT
                    if (total * curr_price) > 2.0:
                        positions.append(
                            {
                                "symbol": asset,
                                "qty": total,
                                "side": "HOLD",
                                "entry_price": curr_price,
                                "current_price": curr_price,
                                "unrealized_pnl": 0.0,
                            }
                        )

            return positions

    async def get_historical_klines(
        self, symbol: str, timeframe: str = "15m", limit: int = 100
    ) -> pd.DataFrame:
        norm = self.normalize_symbol(symbol)
        tf = self.TIMEFRAME_MAP.get(timeframe, TimeFrame(15, TimeFrameUnit.Minute))

        minutes_per_bar = 15
        if "m" in timeframe:
            minutes_per_bar = int(timeframe.replace("m", ""))
        elif "h" in timeframe:
            minutes_per_bar = int(timeframe.replace("h", "")) * 60
        elif "d" in timeframe:
            minutes_per_bar = 1440

        start_time = datetime.now(timezone.utc) - timedelta(
            minutes=minutes_per_bar * (limit + 20)
        )

        req = StockBarsRequest(
            symbol_or_symbols=norm, timeframe=tf, start=start_time, limit=limit
        )

        bars = await asyncio.to_thread(self.data_client.get_stock_bars, req)
        df = bars.df

        if df.empty:
            logger.warning(f"No historical bar data returned for symbol {symbol}")
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(norm, level="symbol")

        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        return df[["open", "high", "low", "close", "volume"]]

    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "MARKET",
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict[str, Any]:
        norm = self.normalize_symbol(symbol)
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        tp_req = TakeProfitRequest(limit_price=tp) if tp else None
        sl_req = StopLossRequest(stop_price=sl) if sl else None

        if order_type.upper() == "MARKET":
            order_data = MarketOrderRequest(
                symbol=norm,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                take_profit=tp_req,
                stop_loss=sl_req,
            )
        else:
            order_data = LimitOrderRequest(
                symbol=norm,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                take_profit=tp_req,
                stop_loss=sl_req,
            )

        order = await asyncio.to_thread(self.trading_client.submit_order, order_data)
        logger.info(
            f"Alpaca Order Submitted: {order.side.value.upper()} {order.qty} {order.symbol} [ID: {order.id}]"
        )
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "status": order.status.value,
            "qty": float(order.qty),
            "side": order.side.value.upper(),
        }

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await asyncio.to_thread(self.trading_client.cancel_order_by_id, order_id)
            logger.info(f"Successfully cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False


class BinanceBroker(BaseBroker):
    """Unified Binance Broker supporting Spot & Futures dynamically from config."""

    def __init__(self, config: BrokerConfig):
        self.config = config
        self.client = None
        self.is_futures = config.market_type == MarketType.FUTURES
        self.leverage = getattr(config, "futures_leverage", 1)

        raw_key = str(config.api_key or "")
        raw_secret = str(config.api_secret or "")

        self.api_key = re.sub(r"[^a-zA-Z0-9]", "", raw_key)
        self.api_secret = re.sub(r"[^a-zA-Z0-9]", "", raw_secret)

    async def connect(self) -> None:
        try:
            from binance import AsyncClient as BinanceAsyncClient
        except ImportError:
            raise ImportError(
                "python-binance library is required. Install via `pip install python-binance`."
            )

        testnet = self.config.trading_mode == TradingMode.PAPER
        self.client = await BinanceAsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=testnet,
        )

        mode_label = f"Futures ({self.leverage}x Leverage)" if self.is_futures else "Spot"
        logger.info(f"Connected to Binance {mode_label} [{'Testnet' if testnet else 'Live'}]")

        if self.is_futures:
            try:
                await self.client.futures_change_position_mode(dualSidePosition=False)
            except Exception:
                pass

            for sym in self.config.trading_pairs:
                try:
                    await self.client.futures_change_leverage(symbol=sym, leverage=self.leverage)
                    await self.client.futures_change_margin_type(symbol=sym, marginType="CROSSED")
                except Exception as e:
                    if "-4046" in str(e) or "No need to change margin type" in str(e):
                        pass
                    else:
                        logger.warning(f"Could not initialize futures leverage for {sym}: {e}")

    async def _get_all_asset_prices(self) -> Dict[str, float]:
        if not self.client:
            return {}
        try:
            if self.is_futures:
                tickers = await self.client.futures_symbol_ticker()
            else:
                tickers = await self.client.get_symbol_ticker()
            return {t["symbol"]: float(t["price"]) for t in tickers}
        except Exception:
            return {}

    async def get_account_balance(self) -> Dict[str, float]:
        if not self.client:
            return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0}

        if self.is_futures:
            try:
                acc = await self.client.futures_account()
                total_margin = float(acc.get("totalMarginBalance", 0.0))
                available_balance = float(acc.get("availableBalance", 0.0))
                return {
                    "equity": total_margin,
                    "cash": available_balance,
                    "buying_power": available_balance * self.leverage,
                }
            except Exception as e:
                logger.error(f"Error fetching futures account balance: {e}")
                return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0}
        else:
            acc = await self.client.get_account()
            price_map = await self._get_all_asset_prices()

            total_equity = 0.0
            free_fiat_stable = 0.0

            for b in acc["balances"]:
                asset = b["asset"]
                free = float(b["free"])
                locked = float(b["locked"])
                total = free + locked

                if total <= 0:
                    continue

                if asset in ["USDC", "EUR", "USDT", "BUSD"]:
                    free_fiat_stable += free

                if asset in ["USDT", "USDC"]:
                    asset_val = total
                elif asset == "EUR":
                    eur_usdt = price_map.get("EURUSDT", 1.08)
                    asset_val = total * eur_usdt
                else:
                    pair_usdt = f"{asset}USDT"
                    pair_usdc = f"{asset}USDC"
                    pair_eur = f"{asset}EUR"

                    if pair_usdt in price_map:
                        asset_val = total * price_map[pair_usdt]
                    elif pair_usdc in price_map:
                        asset_val = total * price_map[pair_usdc]
                    elif pair_eur in price_map:
                        asset_val = total * price_map[pair_eur] * price_map.get("EURUSDT", 1.08)
                    else:
                        asset_val = 0.0

                total_equity += asset_val

            return {
                "equity": total_equity,
                "cash": free_fiat_stable,
                "buying_power": free_fiat_stable,
            }

    async def get_positions(self) -> List[Dict[str, Any]]:
        if not self.client:
            return []

        if self.is_futures:
            try:
                pos_info = await self.client.futures_position_information()
                positions = []
                for pos in pos_info:
                    amt = float(pos["positionAmt"])
                    if amt != 0:
                        positions.append({
                            "symbol": pos["symbol"],
                            "qty": abs(amt),
                            "side": "BUY" if amt > 0 else "SELL",
                            "entry_price": float(pos["entryPrice"]),
                            "current_price": float(pos["markPrice"]),
                            "unrealized_pnl": float(pos["unRealizedProfit"]),
                        })
                return positions
            except Exception as e:
                logger.error(f"Error fetching futures positions: {e}")
                return []
        else:
            acc = await self.client.get_account()
            price_map = await self._get_all_asset_prices()

            positions = []
            for b in acc["balances"]:
                asset = b["asset"]
                free = float(b["free"])
                locked = float(b["locked"])
                total = free + locked

                if total > 0:
                    pair_usdc = f"{asset}USDC"
                    pair_usdt = f"{asset}USDT"
                    pair_eur = f"{asset}EUR"

                    if asset in ["USDC", "USDT"]:
                        curr_price = 1.0
                    elif asset == "EUR":
                        curr_price = price_map.get("EURUSDT", 1.08)
                    elif pair_usdc in price_map:
                        curr_price = price_map[pair_usdc]
                    elif pair_usdt in price_map:
                        curr_price = price_map[pair_usdt]
                    elif pair_eur in price_map:
                        curr_price = price_map[pair_eur]
                    else:
                        curr_price = 0.0

                    positions.append(
                        {
                            "symbol": asset,
                            "qty": total,
                            "side": "HOLD",
                            "entry_price": curr_price,
                            "current_price": curr_price,
                            "unrealized_pnl": 0.0,
                        }
                    )

            return positions

    async def get_historical_klines(
        self, symbol: str, timeframe: str = "15m", limit: int = 100
    ) -> pd.DataFrame:
        if not self.client:
            return pd.DataFrame()
        norm = self.normalize_symbol(symbol)

        if self.is_futures:
            klines = await self.client.futures_klines(
                symbol=norm, interval=timeframe, limit=limit
            )
        else:
            klines = await self.client.get_klines(
                symbol=norm, interval=timeframe, limit=limit
            )

        cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(klines, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["open", "high", "low", "close", "volume"]]

    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "MARKET",
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("Binance client not connected.")

        norm = self.normalize_symbol(symbol)

        # Quantity precision formatting per Binance symbol specifications
        if "BTC" in norm:
            formatted_qty = f"{qty:.3f}"
        elif "ETH" in norm:
            formatted_qty = f"{qty:.3f}"
        else:
            formatted_qty = f"{qty:.2f}"

        # Safety check: ensure formatted quantity is non-zero
        if float(formatted_qty) <= 0:
            raise ValueError(f"Calculated quantity {qty} formatted to {formatted_qty} (must be > 0)")

        order_side = side.upper()
        exit_side = "SELL" if order_side == "BUY" else "BUY"

        if self.is_futures:
            # 1. Primary Market Entry Order
            res = await self.client.futures_create_order(
                symbol=norm,
                side=order_side,
                type=order_type.upper(),
                quantity=formatted_qty,
            )
            order_id = str(res.get("orderId", "N/A"))

            price_precision = 2

            # 2. Attach Stop Loss Order
            if sl and sl > 0:
                try:
                    sl_str = f"{round(sl, price_precision):.{price_precision}f}"
                    sl_res = await self.client.futures_create_order(
                        symbol=norm,
                        side=exit_side,
                        type="STOP_MARKET",
                        stopPrice=sl_str,
                        closePosition=True,
                        workingType="MARK_PRICE",
                    )
                    logger.info(f"🛡️ Native Futures Stop Loss set for {norm} @ ${sl_str} (ID: {sl_res.get('orderId')})")
                except Exception as sl_err:
                    logger.error(f"❌ Failed to set native Futures Stop Loss for {norm}: {sl_err}")

            # 3. Attach Take Profit Order
            if tp and tp > 0:
                try:
                    tp_str = f"{round(tp, price_precision):.{price_precision}f}"
                    tp_res = await self.client.futures_create_order(
                        symbol=norm,
                        side=exit_side,
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=tp_str,
                        closePosition=True,
                        workingType="MARK_PRICE",
                    )
                    logger.info(f"🎯 Native Futures Take Profit set for {norm} @ ${tp_str} (ID: {tp_res.get('orderId')})")
                except Exception as tp_err:
                    logger.error(f"❌ Failed to set native Futures Take Profit for {norm}: {tp_err}")

            return {
                "order_id": order_id,
                "symbol": res.get("symbol", norm),
                "status": res.get("status", "SUBMITTED"),
            }
        else:
            res = await self.client.create_order(
                symbol=norm,
                side=order_side,
                type=order_type.upper(),
                quantity=formatted_qty,
            )
            return {
                "order_id": str(res["orderId"]),
                "symbol": res["symbol"],
                "status": res["status"],
            }

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def close(self) -> None:
        if self.client:
            await self.client.close_connection()


class IBKRBroker(BaseBroker):
    """Interactive Brokers TWS/Gateway Broker Implementation."""

    def __init__(self, config: BrokerConfig):
        self.config = config
        self.ib = None

    async def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError(
                "ib_insync library is required for Interactive Brokers. Install via `pip install ib_insync`."
            )

        self.ib = IB()
        await self.ib.connectAsync(
            host=self.config.host, port=self.config.port, clientId=1
        )
        logger.info(
            f"Connected to IBKR TWS/Gateway at {self.config.host}:{self.config.port}"
        )

    async def get_account_balance(self) -> Dict[str, float]:
        if not self.ib:
            return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0}
        summary = self.ib.accountSummary()
        values = {
            item.tag: float(item.value)
            for item in summary
            if item.tag in ["NetLiquidation", "TotalCashValue", "BuyingPower"]
        }
        return {
            "equity": values.get("NetLiquidation", 0.0),
            "cash": values.get("TotalCashValue", 0.0),
            "buying_power": values.get("BuyingPower", 0.0),
        }

    async def get_positions(self) -> List[Dict[str, Any]]:
        if not self.ib:
            return []
        positions = self.ib.positions()
        return [
            {
                "symbol": pos.contract.symbol,
                "qty": float(pos.position),
                "side": "BUY" if pos.position > 0 else "SELL",
                "entry_price": float(pos.avgCost),
                "current_price": 0.0,
            }
            for pos in positions
        ]

    async def get_historical_klines(
        self, symbol: str, timeframe: str = "15m", limit: int = 100
    ) -> pd.DataFrame:
        if not self.ib:
            return pd.DataFrame()
        from ib_insync import Stock

        contract = Stock(self.normalize_symbol(symbol), "SMART", "USD")
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="2 D",
            barSizeSetting="15 mins",
            whatToShow="TRADES",
            useRTH=True,
        )
        df = pd.DataFrame(
            [
                {
                    "timestamp": b.date,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ]
        )
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df

    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "MARKET",
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.ib:
            raise RuntimeError("IBKR broker not connected.")
        from ib_insync import MarketOrder, Stock

        contract = Stock(self.normalize_symbol(symbol), "SMART", "USD")
        order = MarketOrder(side.upper(), qty)
        trade = self.ib.placeOrder(contract, order)
        return {
            "order_id": str(trade.order.orderId),
            "symbol": symbol,
            "status": trade.orderStatus.status,
        }

    async def cancel_order(self, order_id: str) -> bool:
        return True


class BrokerFactory:
    """Factory class to dynamically instantiate the configured broker."""

    @staticmethod
    def create_broker(config: BrokerConfig) -> BaseBroker:
        if config.broker_type == BrokerType.ALPACA:
            return AlpacaBroker(config)
        elif config.broker_type == BrokerType.BINANCE:
            return BinanceBroker(config)
        elif config.broker_type == BrokerType.IBKR:
            return IBKRBroker(config)
        else:
            raise ValueError(f"Unsupported broker type: {config.broker_type}")