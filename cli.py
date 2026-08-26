from pathlib import Path
from dotenv import set_key
from rich.console import Console
from rich.prompt import Confirm, Prompt

console = Console()


def save_env_var(env_path: Path, key: str, value: str) -> None:
    """Writes or updates a single key-value pair in the .env file."""
    set_key(str(env_path), key, value, quote_mode="always")


def launch_cli() -> None:
    console.print(
        "\n[bold green]=== Multi-Asset Quantitative Bot Setup ===[/bold green]\n"
    )

    # 1. Trading Execution Mode & Venue Selection
    mode = Prompt.ask(
        "Select Execution Mode", choices=["PAPER", "LIVE"], default="PAPER"
    )
    broker = Prompt.ask(
        "Select Broker Venue",
        choices=["ALPACA", "BINANCE", "IBKR"],
        default="ALPACA",
    )

    # 2. Broker API Credentials (Masked Input)
    console.print("\n[bold yellow]--- API Credentials ---[/bold yellow]")
    api_key = Prompt.ask("Enter API Key / Public Key", password=True)
    api_secret = Prompt.ask("Enter API Secret Key", password=True)

    passphrase = ""
    if broker.upper() in ["BINANCE", "IBKR"]:
        if Confirm.ask("Does your API key require a passphrase?", default=False):
            passphrase = Prompt.ask("Enter API Passphrase", password=True)

    # 3. Portfolio Allocation & Risk Parameters
    console.print("\n[bold yellow]--- Account Allocation ---[/bold yellow]")
    capital = Prompt.ask("Allocated Capital ($)", default="100000.0")

    # 4. Asynchronous Telegram Notifications
    console.print("\n[bold yellow]--- Telegram Alerts ---[/bold yellow]")
    telegram_token = ""
    telegram_chat_id = ""

    if Confirm.ask("Enable instant Telegram alerts for fills & drawdowns?", default=False):
        telegram_token = Prompt.ask("Telegram Bot Token", password=True)
        telegram_chat_id = Prompt.ask("Telegram Chat ID")

    # 5. Save Configuration to local .env
    env_path = Path(".env")
    if not env_path.exists():
        env_path.touch()

    save_env_var(env_path, "TRADING_MODE", mode.upper())
    save_env_var(env_path, "BROKER_TYPE", broker.upper())
    save_env_var(env_path, "API_KEY", api_key)
    save_env_var(env_path, "API_SECRET", api_secret)
    save_env_var(env_path, "API_PASSPHRASE", passphrase)
    save_env_var(env_path, "ALLOCATED_CAPITAL", capital)
    save_env_var(env_path, "TELEGRAM_BOT_TOKEN", telegram_token)
    save_env_var(env_path, "TELEGRAM_CHAT_ID", telegram_chat_id)

    console.print(
        f"\n[bold green]✓ Configuration saved to {env_path.resolve()}[/bold green]"
    )


if __name__ == "__main__":
    launch_cli()
