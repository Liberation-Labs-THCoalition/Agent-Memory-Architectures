"""Shadow sandbox for pre-execution of untrusted code.

Creates isolated temporary directories and runs code snippets in a
subprocess with enforced timeouts.  Designed as a pre-flight check so
that destructive or long-running code never touches the host environment.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class SandboxContext:
    """Handle to an active sandbox environment."""

    id: str
    path: str
    created_at: datetime


@dataclass(frozen=True)
class SandboxResult:
    """Captured output of a sandboxed execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    execution_time_ms: float


class ShadowSandbox:
    """Manages ephemeral sandbox directories and subprocess execution."""

    def __init__(self, base_dir: str | None = None) -> None:
        """
        Args:
            base_dir: Optional root for sandbox temp dirs. Defaults to the
                      system temp directory.
        """
        self._base_dir = base_dir
        self._sandboxes: Dict[str, SandboxContext] = {}

    def create_sandbox(self) -> SandboxContext:
        """Create a new isolated temporary directory.

        Returns:
            SandboxContext with a unique id and filesystem path.
        """
        sandbox_id = uuid.uuid4().hex[:16]
        path = tempfile.mkdtemp(prefix=f"kintsugi_sandbox_{sandbox_id}_", dir=self._base_dir)
        ctx = SandboxContext(
            id=sandbox_id,
            path=path,
            created_at=datetime.now(timezone.utc),
        )
        self._sandboxes[sandbox_id] = ctx
        return ctx

    def execute_in_sandbox(
        self,
        code: str,
        timeout: int = 30,
        sandbox_id: str | None = None,
    ) -> SandboxResult:
        """Run *code* as a Python script inside a sandbox directory.

        If *sandbox_id* is None a fresh sandbox is created (and cleaned up
        after execution).  If provided, the existing sandbox is reused and
        the caller is responsible for cleanup.

        Args:
            code:       Python source code to execute.
            timeout:    Maximum wall-clock seconds before the process is killed.
            sandbox_id: Optional existing sandbox to reuse.

        Returns:
            SandboxResult capturing stdout, stderr, exit code, timing, and
            whether the process was killed due to timeout.
        """
        auto_cleanup = sandbox_id is None
        if sandbox_id is None:
            ctx = self.create_sandbox()
        else:
            ctx = self._sandboxes.get(sandbox_id)  # type: ignore[assignment]
            if ctx is None:
                return SandboxResult(
                    stdout="",
                    stderr=f"Sandbox '{sandbox_id}' not found.",
                    exit_code=-1,
                    timed_out=False,
                    execution_time_ms=0.0,
                )

        script_path = Path(ctx.path) / "script.py"
        script_path.write_text(code, encoding="utf-8")

        timed_out = False
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["python3", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=ctx.path,
                env={"PATH": "/usr/bin:/bin", "HOME": ctx.path},
            )
            stdout = proc.stdout
            stderr = proc.stderr
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout = ""
            stderr = f"Execution timed out after {timeout}s."
            exit_code = -1
        except Exception as exc:
            stdout = ""
            stderr = str(exc)
            exit_code = -1
        elapsed_ms = (time.monotonic() - start) * 1000.0

        if auto_cleanup:
            self.cleanup(ctx.id)

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def cleanup(self, sandbox_id: str) -> None:
        """Remove the sandbox directory and deregister it."""
        ctx = self._sandboxes.pop(sandbox_id, None)
        if ctx is not None:
            shutil.rmtree(ctx.path, ignore_errors=True)

    def cleanup_all(self) -> None:
        """Remove every sandbox managed by this instance."""
        for sid in list(self._sandboxes):
            self.cleanup(sid)
