from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    handler: Callable  # async
    requires_consent: bool = False
    consent_category: str = "general"


class MCPRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    async def execute_tool(self, name: str, arguments: dict) -> dict:
        tool = self.get_tool(name)
        if tool is None:
            return {"success": False, "result": None, "error": f"Tool '{name}' not found"}
        try:
            result = await tool.handler(**arguments)
            return {"success": True, "result": result, "error": None}
        except Exception as exc:
            return {"success": False, "result": None, "error": str(exc)}


class MCPHost:
    def __init__(self, registry: MCPRegistry | None = None) -> None:
        self.registry = registry or MCPRegistry()

    async def handle_request(self, method: str, params: dict) -> dict:
        if method == "tools/list":
            tools = self.registry.list_tools()
            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    }
                    for t in tools
                ]
            }
        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            return await self.registry.execute_tool(name, arguments)
        return {"error": f"Unknown method: {method}"}

    async def health_check(self) -> dict:
        return {
            "status": "ok",
            "tool_count": len(self.registry.list_tools()),
        }
