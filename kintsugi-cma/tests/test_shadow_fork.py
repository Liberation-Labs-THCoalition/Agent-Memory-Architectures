"""Tests for kintsugi.kintsugi_engine.shadow_fork (Phase 3, Stream 3A)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from kintsugi.kintsugi_engine.shadow_fork import (
    ShadowConfig,
    ShadowFork,
    ShadowState,
    ShadowStatus,
)


# ---------------------------------------------------------------------------
# ShadowConfig defaults and custom values
# ---------------------------------------------------------------------------

class TestShadowConfig:
    def test_defaults(self):
        cfg = ShadowConfig()
        assert cfg.modification == {}
        assert cfg.timeout_seconds == 300
        assert cfg.max_memory_mb == 512
        assert cfg.mock_tool_responses == {}

    def test_custom(self):
        cfg = ShadowConfig(
            modification={"k": "v"},
            timeout_seconds=60,
            max_memory_mb=1024,
            mock_tool_responses={"tool1": "resp1"},
        )
        assert cfg.modification == {"k": "v"}
        assert cfg.timeout_seconds == 60
        assert cfg.max_memory_mb == 1024
        assert cfg.mock_tool_responses == {"tool1": "resp1"}


# ---------------------------------------------------------------------------
# ShadowStatus enum
# ---------------------------------------------------------------------------

class TestShadowStatus:
    def test_values(self):
        assert ShadowStatus.RUNNING == "RUNNING"
        assert ShadowStatus.COMPLETED == "COMPLETED"
        assert ShadowStatus.TIMEOUT == "TIMEOUT"
        assert ShadowStatus.ERROR == "ERROR"

    def test_is_str_enum(self):
        assert isinstance(ShadowStatus.RUNNING, str)


# ---------------------------------------------------------------------------
# ShadowFork
# ---------------------------------------------------------------------------

def _make_fork(modification=None, timeout=300, mock_tools=None):
    primary = {"model": "gpt-4", "temp": 0.7}
    scfg = ShadowConfig(
        modification=modification or {},
        timeout_seconds=timeout,
        mock_tool_responses=mock_tools or {},
    )
    return ShadowFork(primary, scfg)


class TestFork:
    def test_creates_unique_ids(self):
        sf = _make_fork()
        id1 = sf.fork()
        id2 = sf.fork()
        assert id1 != id2
        assert id1.startswith("shadow-")

    def test_stores_state(self):
        sf = _make_fork()
        sid = sf.fork()
        state = sf.get_state(sid)
        assert isinstance(state, ShadowState)
        assert state.status == ShadowStatus.RUNNING

    def test_merges_modification(self):
        sf = _make_fork(modification={"temp": 0.9, "new_key": True})
        sid = sf.fork()
        assert sf._shadow_configs[sid]["temp"] == 0.9
        assert sf._shadow_configs[sid]["model"] == "gpt-4"
        assert sf._shadow_configs[sid]["new_key"] is True

    def test_deep_copies_primary(self):
        primary = {"nested": {"a": 1}}
        scfg = ShadowConfig()
        sf = ShadowFork(primary, scfg)
        sid = sf.fork()
        # Mutating original should not affect shadow
        primary["nested"]["a"] = 999
        assert sf._shadow_configs[sid]["nested"]["a"] == 1


class TestExecuteTurn:
    def test_records_outputs(self):
        sf = _make_fork(mock_tools={"search": "result1"})
        sid = sf.fork()
        result = sf.execute_turn(sid, "hello")
        assert result["shadow_id"] == sid
        assert result["output"]["input"] == "hello"
        assert result["output"]["mock_tool_results"] == {"search": "result1"}

    def test_records_tool_calls(self):
        sf = _make_fork(mock_tools={"search": "r1", "write": "r2"})
        sid = sf.fork()
        result = sf.execute_turn(sid, "msg")
        assert len(result["tool_calls"]) == 2
        tools = {tc["tool"] for tc in result["tool_calls"]}
        assert tools == {"search", "write"}
        for tc in result["tool_calls"]:
            assert tc["intercepted"] is True

    def test_non_running_raises(self):
        sf = _make_fork()
        sid = sf.fork()
        sf.terminate(sid)
        with pytest.raises(RuntimeError, match="not running"):
            sf.execute_turn(sid, "msg")

    def test_accumulates_state(self):
        sf = _make_fork()
        sid = sf.fork()
        sf.execute_turn(sid, "a")
        sf.execute_turn(sid, "b")
        state = sf.get_state(sid)
        assert len(state.outputs) == 2


class TestGetState:
    def test_returns_state(self):
        sf = _make_fork()
        sid = sf.fork()
        assert sf.get_state(sid).status == ShadowStatus.RUNNING

    def test_unknown_raises_key_error(self):
        sf = _make_fork()
        with pytest.raises(KeyError, match="Unknown shadow"):
            sf.get_state("nonexistent")


class TestTerminate:
    def test_marks_completed(self):
        sf = _make_fork(timeout=300)
        sid = sf.fork()
        state = sf.terminate(sid)
        assert state.status == ShadowStatus.COMPLETED

    def test_marks_timeout(self):
        sf = _make_fork(timeout=0.0)
        sid = sf.fork()
        # tiny sleep to exceed 0-second timeout
        time.sleep(0.01)
        state = sf.terminate(sid)
        assert state.status == ShadowStatus.TIMEOUT

    def test_elapsed_set(self):
        sf = _make_fork()
        sid = sf.fork()
        state = sf.terminate(sid)
        assert state.elapsed_seconds > 0


class TestDistributeInput:
    def test_returns_primary_and_shadow(self):
        sf = _make_fork(modification={"temp": 0.1})
        primary, shadow = sf.distribute_input("question?")
        assert primary["agent"] == "primary"
        assert primary["input"] == "question?"
        assert shadow["agent"] == "shadow"
        assert shadow["input"] == "question?"
        assert "shadow_id" in shadow

    def test_creates_fork(self):
        sf = _make_fork()
        assert len(sf._shadows) == 0
        sf.distribute_input("msg")
        assert len(sf._shadows) == 1


class TestCheckResourceLimits:
    def test_within_limits(self):
        sf = _make_fork(timeout=300)
        sid = sf.fork()
        assert sf._check_resource_limits(sid) is True

    def test_exceeded(self):
        sf = _make_fork(timeout=0.0)
        sid = sf.fork()
        time.sleep(0.01)
        assert sf._check_resource_limits(sid) is False

    def test_execute_turn_timeout(self):
        sf = _make_fork(timeout=0.0)
        sid = sf.fork()
        time.sleep(0.01)
        with pytest.raises(RuntimeError, match="exceeded resource limits"):
            sf.execute_turn(sid, "msg")
        assert sf.get_state(sid).status == ShadowStatus.TIMEOUT


class TestMultipleShadows:
    def test_manage_simultaneously(self):
        sf = _make_fork(mock_tools={"t": "r"})
        ids = [sf.fork() for _ in range(5)]
        for sid in ids:
            sf.execute_turn(sid, "input")
        for sid in ids:
            state = sf.terminate(sid)
            assert state.status == ShadowStatus.COMPLETED
            assert len(state.outputs) == 1
