"""Tests for kintsugi.integrations.spans modules."""

from __future__ import annotations

import pytest

from kintsugi.integrations.spans.communication import CommunicationSpan
from kintsugi.integrations.spans.project_mgmt import ProjectManagementSpan
from kintsugi.integrations.spans.data import DataSpan
from kintsugi.integrations.spans.operational import OperationalSpan


# ---------------------------------------------------------------------------
# CommunicationSpan
# ---------------------------------------------------------------------------

class TestCommunicationSpan:
    def test_is_configured_true(self):
        cfg = {"SLACK_TOKEN": "x", "DISCORD_TOKEN": "x", "SMTP_HOST": "h", "SMTP_USER": "u", "SMTP_PASS": "p"}
        span = CommunicationSpan(cfg)
        assert span.is_configured("slack") is True
        assert span.is_configured("discord") is True
        assert span.is_configured("email") is True

    def test_is_configured_false(self):
        span = CommunicationSpan()
        assert span.is_configured("slack") is False
        assert span.is_configured("discord") is False
        assert span.is_configured("email") is False

    def test_is_configured_unknown_service(self):
        assert CommunicationSpan().is_configured("teams") is True  # empty keys list -> all()

    def test_get_required_config_keys(self):
        keys = CommunicationSpan().get_required_config_keys()
        assert "SLACK_TOKEN" in keys

    @pytest.mark.asyncio
    async def test_send_slack_configured(self):
        span = CommunicationSpan({"SLACK_TOKEN": "t"})
        r = await span.send_slack_message("#gen", "hi")
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_send_slack_unconfigured(self):
        r = await CommunicationSpan().send_slack_message("#gen", "hi")
        assert r["success"] is False
        assert r["error"] == "service_not_configured"

    @pytest.mark.asyncio
    async def test_send_discord_configured(self):
        span = CommunicationSpan({"DISCORD_TOKEN": "t"})
        r = await span.send_discord_message("123", "hi")
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_send_discord_unconfigured(self):
        r = await CommunicationSpan().send_discord_message("123", "hi")
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_send_email_configured(self):
        span = CommunicationSpan({"SMTP_HOST": "h", "SMTP_USER": "u", "SMTP_PASS": "p"})
        r = await span.send_email("a@b.com", "subj", "body")
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_send_email_unconfigured(self):
        r = await CommunicationSpan().send_email("a@b.com", "subj", "body")
        assert r["success"] is False


# ---------------------------------------------------------------------------
# ProjectManagementSpan
# ---------------------------------------------------------------------------

class TestProjectManagementSpan:
    def test_is_configured(self):
        span = ProjectManagementSpan({"ASANA_TOKEN": "a"})
        assert span.is_configured("asana") is True
        assert span.is_configured("jira") is False

    def test_is_configured_jira(self):
        span = ProjectManagementSpan({"JIRA_TOKEN": "t", "JIRA_URL": "u"})
        assert span.is_configured("jira") is True

    @pytest.mark.asyncio
    async def test_create_task_configured(self):
        span = ProjectManagementSpan({"ASANA_TOKEN": "a"})
        r = await span.create_task("Do thing")
        assert r["success"] is True
        assert r["result"]["title"] == "Do thing"

    @pytest.mark.asyncio
    async def test_create_task_unconfigured(self):
        r = await ProjectManagementSpan().create_task("x")
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_list_tasks_configured(self):
        span = ProjectManagementSpan({"ASANA_TOKEN": "a"})
        r = await span.list_tasks("proj1")
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_list_tasks_unconfigured(self):
        r = await ProjectManagementSpan().list_tasks("proj1")
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_update_task_configured(self):
        span = ProjectManagementSpan({"ASANA_TOKEN": "a"})
        r = await span.update_task("t1", {"status": "done"})
        assert r["success"] is True
        assert "status" in r["result"]["updated_fields"]

    @pytest.mark.asyncio
    async def test_update_task_unconfigured(self):
        r = await ProjectManagementSpan().update_task("t1", {"status": "done"})
        assert r["success"] is False

    def test_get_required_config_keys(self):
        keys = ProjectManagementSpan().get_required_config_keys()
        assert "JIRA_TOKEN" in keys


# ---------------------------------------------------------------------------
# DataSpan
# ---------------------------------------------------------------------------

class TestDataSpan:
    @pytest.mark.parametrize("q", ["SELECT * FROM t", "select count(*) from t"])
    def test_validate_query_safe(self, q):
        assert DataSpan()._validate_query(q) is True

    @pytest.mark.parametrize("kw", ["DROP", "DELETE", "TRUNCATE", "ALTER", "drop", "Delete"])
    def test_validate_query_dangerous(self, kw):
        assert DataSpan()._validate_query(f"{kw} TABLE foo") is False

    @pytest.mark.asyncio
    async def test_query_data_valid(self):
        r = await DataSpan().query_data("SELECT 1")
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_query_data_invalid(self):
        r = await DataSpan().query_data("DROP TABLE x")
        assert r["success"] is False
        assert r["error"] == "query_rejected"

    @pytest.mark.asyncio
    async def test_export_data_valid(self):
        r = await DataSpan().export_data("csv", "SELECT 1")
        assert r["success"] is True
        assert r["result"]["format"] == "csv"

    @pytest.mark.asyncio
    async def test_export_data_invalid_query(self):
        r = await DataSpan().export_data("json", "DELETE FROM t")
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_export_data_unsupported_format(self):
        r = await DataSpan().export_data("parquet", "SELECT 1")
        assert r["success"] is False
        assert r["error"] == "unsupported_format"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fmt", ["csv", "json", "xlsx"])
    async def test_export_data_supported_formats(self, fmt):
        r = await DataSpan().export_data(fmt, "SELECT 1")
        assert r["success"] is True


# ---------------------------------------------------------------------------
# OperationalSpan
# ---------------------------------------------------------------------------

class TestOperationalSpan:
    def test_is_configured(self):
        span = OperationalSpan({"GITHUB_TOKEN": "t"})
        assert span.is_configured("github") is True
        assert span.is_configured("google_drive") is False

    @pytest.mark.asyncio
    async def test_create_github_issue_configured(self):
        span = OperationalSpan({"GITHUB_TOKEN": "t"})
        r = await span.create_github_issue("org/repo", "title", "body")
        assert r["success"] is True
        assert "url" in r["result"]

    @pytest.mark.asyncio
    async def test_create_github_issue_unconfigured(self):
        r = await OperationalSpan().create_github_issue("org/repo", "title")
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_list_github_issues_configured(self):
        span = OperationalSpan({"GITHUB_TOKEN": "t"})
        r = await span.list_github_issues("org/repo")
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_list_github_issues_unconfigured(self):
        r = await OperationalSpan().list_github_issues("org/repo")
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_upload_to_drive_configured(self):
        span = OperationalSpan({"GOOGLE_DRIVE_CREDENTIALS": "c"})
        r = await span.upload_to_drive("f.txt", b"data")
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_upload_to_drive_unconfigured(self):
        r = await OperationalSpan().upload_to_drive("f.txt", b"data")
        assert r["success"] is False

    def test_get_required_config_keys(self):
        keys = OperationalSpan().get_required_config_keys()
        assert "GITHUB_TOKEN" in keys
        assert "GOOGLE_DRIVE_CREDENTIALS" in keys
