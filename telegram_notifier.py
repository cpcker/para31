import logging
import ssl
import aiohttp
import certifi

logger = logging.getLogger(__name__)


class TelegramNotifier:

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def _get_ssl_context(self) -> ssl.SSLContext:
        """Generates an SSL context using certifi CA bundle with fallback for antivirus/proxy SSL inspection."""
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            # Fallback if local network proxy or antivirus intercepts SSL certificates
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx

    async def send_message(self, message: str) -> bool:
        if not self.bot_token or not self.chat_id:
            logger.error(
                "Telegram token or Chat ID is missing. Skipping notification."
            )
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        ssl_context = self._get_ssl_context()

        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(self.api_url, json=payload) as resp:
                    res_data = await resp.json()
                    if resp.status == 200 and res_data.get("ok"):
                        return True
                    else:
                        logger.error(
                            f"Telegram API Error [{resp.status}]: {res_data.get('description', 'Unknown Error')}"
                        )
                        return False
        except Exception as e:
            # Secondary retry with SSL verification bypassed if local certificate authority fails completely
            try:
                connector = aiohttp.TCPConnector(ssl=False)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.post(self.api_url, json=payload) as resp:
                        res_data = await resp.json()
                        return resp.status == 200 and res_data.get("ok")
            except Exception as retry_err:
                logger.error(f"Failed to dispatch Telegram message: {retry_err}")
                return False