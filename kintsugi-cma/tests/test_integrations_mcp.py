"""Tests for kintsugi.integrations.mcp_host module."""

from __future__ import annotations

import pytest

from kintsugi.integrations.mcp_host import MCPHost, MCPRegistry, ToolDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _echo_handler(**kwargs):
    return kwargs


async def _failing_handler(**kwargs):
    raise RuntimeError("boom")


def _make_tool(name: str = "echo", handler=None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object"},
        handler=handler or _echo_handler,
    )


# ---------------------------------------------------------------------------
# MCPRegistry
# ---------------------------------------------------------------------------

class TestMCPRegistry:
    def test_register_and_get(self):
        reg = MCPRegistry()
        tool = _make_tool("t1")
        reg.register_tool(tool)
        assert reg.get_tool("t1") is tool

    def test_get_unknown_returns_none(self):
        reg = MCPRegistry()
        assert reg.get_tool("nope") is None

    def test_unregister(self):
        reg = MCPRegistry()
        reg.register_tool(_make_tool("t1"))
        reg.unregister_tool("t1")
        assert reg.get_tool("t1") is None

    def test_unregister_unknown_no_error(self):
        MCPRegistry().unregister_tool("nope")

    def test_list_tools(self):
        reg = MCPRegistry()
        reg.register_tool(_make_tool("a"))
        reg.register_tool(_make_tool("b"))
        names = {t.name for t in reg.list_tools()}
        assert names == {"a", "b"}

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        reg = MCPRegistry()
        reg.register_tool(_make_tool("echo"))
        result = await reg.execute_tool("echo", {"x": 1})
        assert result["success"] is True
        assert result["result"] == {"x": 1}
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self):
        reg = MCPRegistry()
        reg.register_tool(_make_tool("fail", handler=_failing_handler))
        result = await reg.execute_tool("fail", {})
        assert result["success"] is False
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = MCPRegistry()
        result = await reg.execute_tool("missing", {})
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_tool_definition_defaults(self):
        t = _make_tool()
        assert t.requires_consent is False
        assert t.consent_category == "general"


# ---------------------------------------------------------------------------
# MCPHost
# ---------------------------------------------------------------------------

class TestMCPHost:
    @pytest.mark.asyncio
    async def test_handle_tools_list(self):
        reg = MCPRegistry()
        reg.register_tool(_make_tool("a"))
        host = MCPHost(reg)
        resp = await host.handle_request("tools/list", {})
        assert len(resp["tools"]) == 1
        assert resp["tools"][0]["name"] == "a"
        assert "inputSchema" in resp["tools"][0]

    @pytest.mark.asyncio
    async def test_handle_tools_call(self):
        reg = MCPRegistry()
        reg.register_tool(_make_tool("echo"))
        host = MCPHost(reg)
        resp = await host.handle_request("tools/call", {"name": "echo", "arguments": {"k": "v"}})
        assert resp["success"] is True
        assert resp["result"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        host = MCPHost()
        resp = await host.handle_request("foo/bar", {})
        assert "error" in resp
        assert "Unknown method" in resp["error"]

    @pytest.mark.asyncio
    async def test_health_check(self):
        reg = MCPRegistry()
        reg.register_tool(_make_tool("x"))
        host = MCPHost(reg)
        h = await host.health_check()
        assert h["status"] == "ok"
        assert h["tool_count"] == 1

    @pytest.mark.asyncio
    async def test_default_registry(self):
        host = MCPHost()
        assert isinstance(host.registry, MCPRegistry)
        h = await host.health_check()
        assert h["tool_count"] == 0
