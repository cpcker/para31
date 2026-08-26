import asyncio
from config import get_config
from telegram_notifier import TelegramNotifier

async def test_telegram_alert():
    config = get_config()
    if not config.telegram.is_enabled:
        print("❌ Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in .env")
        return

    notifier = TelegramNotifier(
        bot_token=config.telegram.bot_token,
        chat_id=config.telegram.chat_id
    )

    test_msg = (
        "🟢 *Telegram Alert Engine Connected Successfully!*\n"
        "• *Venue:* `BINANCE` (LIVE)\n"
        "• *Status:* Ready to receive real-time trade fills & kill-switch alerts."
    )

    print("Sending test message to Telegram...")
    success = await notifier.send_message(test_msg)
    if success:
        print("✓ Test notification delivered to your phone!")
    else:
        print("❌ Failed to deliver notification. Check your token and chat ID.")

if __name__ == "__main__":
    asyncio.run(test_telegram_alert())