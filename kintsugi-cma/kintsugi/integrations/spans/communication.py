from __future__ import annotations


class CommunicationSpan:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    # -- helpers ----------------------------------------------------------

    def get_required_config_keys(self) -> list[str]:
        return ["SLACK_TOKEN", "DISCORD_TOKEN", "SMTP_HOST", "SMTP_USER", "SMTP_PASS"]

    def is_configured(self, service: str) -> bool:
        mapping: dict[str, list[str]] = {
            "slack": ["SLACK_TOKEN"],
            "discord": ["DISCORD_TOKEN"],
            "email": ["SMTP_HOST", "SMTP_USER", "SMTP_PASS"],
        }
        keys = mapping.get(service, [])
        return all(k in self._config for k in keys)

    def _not_configured(self, service: str) -> dict:
        return {"success": False, "error": "service_not_configured", "service": service}

    # -- stubs ------------------------------------------------------------

    async def send_slack_message(
        self, channel: str, message: str, thread_ts: str | None = None
    ) -> dict:
        if not self.is_configured("slack"):
            return self._not_configured("slack")
        return {
            "success": True,
            "result": {"channel": channel, "ts": "mock_ts", "message": message},
        }

    async def send_discord_message(self, channel_id: str, message: str) -> dict:
        if not self.is_configured("discord"):
            return self._not_configured("discord")
        return {
            "success": True,
            "result": {"channel_id": channel_id, "message_id": "mock_id"},
        }

    async def send_email(
        self, to: str, subject: str, body: str, html: bool = False
    ) -> dict:
        if not self.is_configured("email"):
            return self._not_configured("email")
        return {
            "success": True,
            "result": {"to": to, "subject": subject, "status": "sent"},
        }
