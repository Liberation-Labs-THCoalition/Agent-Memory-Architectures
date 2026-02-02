"""Tests for kintsugi.config.values_loader."""

import json
import pytest
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from kintsugi.config.values_loader import (
    load_values,
    load_from_template,
    merge_with_defaults,
    save_values,
    _deep_merge,
    FileWatcher,
)
from kintsugi.config.values_schema import OrganizationValues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL = {
    "organization": {"name": "Test", "mission": "Testing"},
}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# load_values
# ---------------------------------------------------------------------------

class TestLoadValues:
    def test_load_valid(self, tmp_path):
        p = tmp_path / "v.json"
        _write_json(p, _MINIMAL)
        v = load_values(p)
        assert v.organization.name == "Test"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_values(tmp_path / "nope.json")

    def test_invalid_json_schema(self, tmp_path):
        p = tmp_path / "v.json"
        _write_json(p, {"bad": "data"})
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            load_values(p)


# ---------------------------------------------------------------------------
# load_from_template
# ---------------------------------------------------------------------------

class TestLoadFromTemplate:
    @pytest.mark.parametrize("org_type", ["mutual_aid", "nonprofit_501c3", "cooperative", "advocacy"])
    def test_all_templates(self, org_type):
        v = load_from_template(org_type)
        assert isinstance(v, OrganizationValues)
        assert v.organization.type == org_type

    def test_invalid_org_type(self):
        with pytest.raises(ValueError, match="No template"):
            load_from_template("forprofit")


# ---------------------------------------------------------------------------
# merge_with_defaults
# ---------------------------------------------------------------------------

class TestMergeWithDefaults:
    def test_override_name(self):
        v = merge_with_defaults(
            {"organization": {"name": "My Org"}},
            "mutual_aid",
        )
        assert v.organization.name == "My Org"
        # mission comes from template
        assert len(v.organization.mission) > 0

    def test_override_shield(self):
        v = merge_with_defaults(
            {"shield": {"budget_per_session": 99.0}},
            "cooperative",
        )
        assert v.shield.budget_per_session == 99.0


# ---------------------------------------------------------------------------
# save_values
# ---------------------------------------------------------------------------

class TestSaveValues:
    def test_roundtrip(self, tmp_path):
        v = OrganizationValues(**_MINIMAL)
        p = tmp_path / "out.json"
        save_values(v, p)
        assert p.exists()
        v2 = load_values(p)
        assert v2.organization.name == "Test"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "v.json"
        v = OrganizationValues(**_MINIMAL)
        save_values(v, p)
        assert p.exists()

    def test_atomic_no_tmp_left(self, tmp_path):
        p = tmp_path / "v.json"
        save_values(OrganizationValues(**_MINIMAL), p)
        assert not p.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_simple(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested(self):
        base = {"x": {"a": 1, "b": 2}}
        over = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, over) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_list_replaced(self):
        assert _deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}

    def test_empty_override(self):
        assert _deep_merge({"a": 1}, {}) == {"a": 1}


# ---------------------------------------------------------------------------
# FileWatcher
# ---------------------------------------------------------------------------

class TestFileWatcher:
    def test_polling_fallback(self, tmp_path):
        """When watchdog is unavailable, polling fallback fires callback."""
        p = tmp_path / "v.json"
        p.write_text("{}")
        calls = []

        def cb(path):
            calls.append(path)

        with patch("kintsugi.config.values_loader.FileWatcher._start_watchdog", side_effect=ImportError):
            fw = FileWatcher(p, cb, poll_interval=0.1)
            fw.start()
            time.sleep(0.15)
            # modify file to trigger callback
            p.write_text('{"changed": true}')
            time.sleep(0.3)
            fw.stop()

        assert len(calls) >= 1
        assert calls[0] == p.resolve()

    def test_stop_without_start(self):
        fw = FileWatcher("/tmp/nonexistent", lambda p: None)
        fw.stop()  # should not raise

    def test_double_start_is_noop(self, tmp_path):
        p = tmp_path / "v.json"
        p.write_text("{}")
        with patch("kintsugi.config.values_loader.FileWatcher._start_watchdog", side_effect=ImportError):
            fw = FileWatcher(p, lambda p: None, poll_interval=0.1)
            fw.start()
            fw.start()  # second call is noop
            fw.stop()

    def test_polling_handles_missing_file(self, tmp_path):
        """Polling should not crash if file doesn't exist initially."""
        p = tmp_path / "missing.json"
        calls = []

        with patch("kintsugi.config.values_loader.FileWatcher._start_watchdog", side_effect=ImportError):
            fw = FileWatcher(p, lambda path: calls.append(path), poll_interval=0.1)
            fw.start()
            time.sleep(0.15)
            # create file
            p.write_text("{}")
            time.sleep(0.3)
            fw.stop()

        assert len(calls) >= 1

    def test_watchdog_backend(self, tmp_path):
        """If watchdog import succeeds, observer is set."""
        p = tmp_path / "v.json"
        p.write_text("{}")
        mock_observer = MagicMock()

        with patch("kintsugi.config.values_loader.FileWatcher._start_watchdog") as mock_wd:
            fw = FileWatcher(p, lambda p: None)
            fw.start()
            mock_wd.assert_called_once()
            fw.stop()
