import asyncio
from config import get_config
from broker_interface import AlpacaBroker

async def test_connection():
    config = get_config()
    broker = AlpacaBroker(config.broker)
    await broker.connect()

    # 1. Test account balance fetch
    balance = await broker.get_account_balance()
    print(f"✓ Authentication Successful!")
    print(f"  Equity: ${balance['equity']:,.2f}")
    print(f"  Buying Power: ${balance['buying_power']:,.2f}")

    # 2. Test historical market data fetch
    print("\nFetching AAPL 15m historical bars...")
    df = await broker.get_historical_klines("AAPL", timeframe="15m", limit=5)
    print("✓ Historical Data Received:")
    print(df[["open", "high", "low", "close", "volume"]])

if __name__ == "__main__":
    asyncio.run(test_connection())