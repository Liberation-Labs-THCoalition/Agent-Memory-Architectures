from __future__ import annotations


class OperationalSpan:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    def get_required_config_keys(self) -> list[str]:
        return ["GITHUB_TOKEN", "GOOGLE_DRIVE_CREDENTIALS"]

    def is_configured(self, service: str) -> bool:
        mapping: dict[str, list[str]] = {
            "github": ["GITHUB_TOKEN"],
            "google_drive": ["GOOGLE_DRIVE_CREDENTIALS"],
        }
        keys = mapping.get(service, [])
        return all(k in self._config for k in keys)

    def _not_configured(self, service: str) -> dict:
        return {"success": False, "error": "service_not_configured", "service": service}

    async def create_github_issue(
        self, repo: str, title: str, body: str = "", labels: list[str] | None = None
    ) -> dict:
        if not self.is_configured("github"):
            return self._not_configured("github")
        return {
            "success": True,
            "result": {
                "issue_number": 1,
                "repo": repo,
                "title": title,
                "url": f"https://github.com/{repo}/issues/1",
            },
        }

    async def list_github_issues(self, repo: str, state: str = "open") -> dict:
        if not self.is_configured("github"):
            return self._not_configured("github")
        return {
            "success": True,
            "result": {"repo": repo, "state": state, "issues": []},
        }

    async def upload_to_drive(
        self, filename: str, content: bytes, folder_id: str | None = None
    ) -> dict:
        if not self.is_configured("google_drive"):
            return self._not_configured("google_drive")
        return {
            "success": True,
            "result": {"file_id": "mock_file_id", "filename": filename},
        }
