from __future__ import annotations


class ProjectManagementSpan:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    def get_required_config_keys(self) -> list[str]:
        return ["ASANA_TOKEN", "JIRA_TOKEN", "JIRA_URL"]

    def is_configured(self, service: str) -> bool:
        mapping: dict[str, list[str]] = {
            "asana": ["ASANA_TOKEN"],
            "jira": ["JIRA_TOKEN", "JIRA_URL"],
        }
        keys = mapping.get(service, [])
        return all(k in self._config for k in keys)

    def _not_configured(self, service: str) -> dict:
        return {"success": False, "error": "service_not_configured", "service": service}

    async def create_task(
        self,
        title: str,
        description: str = "",
        assignee: str | None = None,
        due_date: str | None = None,
        platform: str = "asana",
    ) -> dict:
        if not self.is_configured(platform):
            return self._not_configured(platform)
        return {
            "success": True,
            "result": {
                "task_id": "mock_task_id",
                "title": title,
                "platform": platform,
            },
        }

    async def list_tasks(
        self, project_id: str, status: str = "active", platform: str = "asana"
    ) -> dict:
        if not self.is_configured(platform):
            return self._not_configured(platform)
        return {
            "success": True,
            "result": {"project_id": project_id, "tasks": [], "status": status},
        }

    async def update_task(
        self, task_id: str, updates: dict, platform: str = "asana"
    ) -> dict:
        if not self.is_configured(platform):
            return self._not_configured(platform)
        return {
            "success": True,
            "result": {"task_id": task_id, "updated_fields": list(updates.keys())},
        }
