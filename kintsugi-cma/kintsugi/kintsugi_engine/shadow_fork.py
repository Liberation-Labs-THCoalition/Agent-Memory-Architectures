"""Shadow fork execution for safe modification testing.

A shadow fork runs a parallel agent with proposed modifications in an
isolated environment.  Tool calls are intercepted and answered with
mock/cached responses so the shadow never touches real resources.
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ShadowStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class ShadowConfig:
    """Configuration for a shadow fork execution."""

    modification: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300
    max_memory_mb: int = 512
    mock_tool_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowState:
    """Tracks the runtime state of a shadow fork."""

    outputs: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0
    elapsed_seconds: float = 0.0
    status: ShadowStatus = ShadowStatus.RUNNING


# ---------------------------------------------------------------------------
# ShadowFork
# ---------------------------------------------------------------------------

class ShadowFork:
    """Creates and manages isolated shadow agent executions.

    The shadow receives the same inputs as the primary agent but runs
    with a modified configuration and a mock tool layer that intercepts
    all tool calls.
    """

    def __init__(self, primary_config: dict, shadow_config: ShadowConfig) -> None:
        self._primary_config = copy.deepcopy(primary_config)
        self._shadow_config = shadow_config
        self._shadows: Dict[str, ShadowState] = {}
        self._shadow_configs: Dict[str, dict] = {}

    # -- public API ---------------------------------------------------------

    def fork(self) -> str:
        """Create a new shadow fork and return its unique ID."""
        shadow_id = f"shadow-{uuid.uuid4().hex[:12]}"
        # Build shadow config by merging modification into primary
        merged = copy.deepcopy(self._primary_config)
        merged.update(self._shadow_config.modification)
        self._shadow_configs[shadow_id] = merged

        state = ShadowState(start_time=time.monotonic())
        self._shadows[shadow_id] = state
        logger.info("Forked shadow %s", shadow_id)
        return shadow_id

    def execute_turn(self, shadow_id: str, input_message: str) -> dict:
        """Process *input_message* through the shadow's mock tool layer.

        Returns a structured result with shadow output and intercepted
        tool calls.  No real LLM or tool execution occurs -- this is a
        simulation scaffold.
        """
        state = self._get_state_or_raise(shadow_id)
        if state.status != ShadowStatus.RUNNING:
            raise RuntimeError(f"Shadow {shadow_id} is not running (status={state.status})")

        if not self._check_resource_limits(shadow_id):
            state.status = ShadowStatus.TIMEOUT
            state.elapsed_seconds = time.monotonic() - state.start_time
            raise RuntimeError(f"Shadow {shadow_id} exceeded resource limits")

        # Simulate processing: record output and any mock tool calls
        tool_calls: List[dict] = []
        mock_responses = self._shadow_config.mock_tool_responses
        for tool_name, mock_response in mock_responses.items():
            call_record = {
                "tool": tool_name,
                "input": input_message,
                "response": mock_response,
                "intercepted": True,
            }
            tool_calls.append(call_record)

        output = {
            "shadow_id": shadow_id,
            "input": input_message,
            "config": self._shadow_configs.get(shadow_id, {}),
            "mock_tool_results": {k: v for k, v in mock_responses.items()},
        }

        state.outputs.append(output)
        state.tool_calls.extend(tool_calls)
        state.elapsed_seconds = time.monotonic() - state.start_time

        result = {
            "shadow_id": shadow_id,
            "output": output,
            "tool_calls": tool_calls,
        }
        logger.debug("Shadow %s turn completed: %d tool calls", shadow_id, len(tool_calls))
        return result

    def get_state(self, shadow_id: str) -> ShadowState:
        """Return the current state of *shadow_id*."""
        return self._get_state_or_raise(shadow_id)

    def terminate(self, shadow_id: str) -> ShadowState:
        """Terminate a shadow fork and return its final state."""
        state = self._get_state_or_raise(shadow_id)
        state.elapsed_seconds = time.monotonic() - state.start_time
        if state.status == ShadowStatus.RUNNING:
            if state.elapsed_seconds > self._shadow_config.timeout_seconds:
                state.status = ShadowStatus.TIMEOUT
            else:
                state.status = ShadowStatus.COMPLETED
        logger.info("Terminated shadow %s with status %s", shadow_id, state.status)
        return state

    def distribute_input(self, input_message: str) -> tuple[dict, dict]:
        """Structure parallel dispatch of the same input to primary and shadow.

        Returns ``(primary_result, shadow_result)`` as dispatch descriptors.
        Does NOT call any LLM -- just structures the dispatch.
        """
        shadow_id = self.fork()
        primary_result = {
            "agent": "primary",
            "config": copy.deepcopy(self._primary_config),
            "input": input_message,
        }
        shadow_result = {
            "agent": "shadow",
            "shadow_id": shadow_id,
            "config": self._shadow_configs[shadow_id],
            "input": input_message,
        }
        return primary_result, shadow_result

    # -- internal -----------------------------------------------------------

    def _check_resource_limits(self, shadow_id: str) -> bool:
        """Return True if the shadow is within resource limits."""
        state = self._get_state_or_raise(shadow_id)
        elapsed = time.monotonic() - state.start_time
        if elapsed > self._shadow_config.timeout_seconds:
            logger.warning("Shadow %s timed out (%.1fs)", shadow_id, elapsed)
            return False
        return True

    def _get_state_or_raise(self, shadow_id: str) -> ShadowState:
        if shadow_id not in self._shadows:
            raise KeyError(f"Unknown shadow ID: {shadow_id}")
        return self._shadows[shadow_id]
