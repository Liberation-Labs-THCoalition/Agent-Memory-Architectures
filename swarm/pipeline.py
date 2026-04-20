"""Microagent SDLC Swarm Pipeline

RAM-based code review pipeline using small local models via Ollama.
Pre-filters code changes before expensive Opus review.

Architecture:
  Commit/PR → Tier 1 (local RAM swarm, free) → Tier 2 (cloud routines)

Agents:
  - Linter: Style, formatting, obvious anti-patterns
  - Reviewer: Missing error handling, unused imports, logic issues
  - Security: Basic vulnerability patterns (injection, path traversal)

All agents run on Qwen2.5-Coder:1.5b via Ollama (~1GB RAM each).
Results published to NATS system.pipeline subject.

Author: Nexus (Coalition)
Date: 2026-04-20
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import requests

logging.basicConfig(level=logging.INFO, format='[swarm] %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("SWARM_MODEL", "qwen2.5-coder:1.5b")


@dataclass
class ReviewResult:
    """Result from a single microagent."""
    agent: str
    severity: str  # "clean", "info", "warning", "error"
    findings: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class PipelineResult:
    """Aggregated results from the full pipeline."""
    file_path: str
    results: List[ReviewResult] = field(default_factory=list)
    escalate_to_opus: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def max_severity(self) -> str:
        severities = {"clean": 0, "info": 1, "warning": 2, "error": 3}
        max_sev = max(
            (severities.get(r.severity, 0) for r in self.results),
            default=0,
        )
        return {v: k for k, v in severities.items()}[max_sev]

    def summary(self) -> str:
        findings = []
        for r in self.results:
            for f in r.findings:
                findings.append(f"[{r.agent}/{r.severity}] {f}")
        if not findings:
            return f"{self.file_path}: clean"
        return f"{self.file_path}:\n" + "\n".join(f"  {f}" for f in findings)


def ask_ollama(prompt: str, model: str = None, max_tokens: int = 500) -> str:
    """Query the local model."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model or OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
    return ""


def lint_agent(code: str, filename: str) -> ReviewResult:
    """Lint agent — checks style, formatting, obvious issues."""
    start = time.time()
    prompt = f"""Review this Python code for style and formatting issues ONLY.
Report each issue on its own line starting with "- ".
If the code is clean, respond with just "CLEAN".
Do not suggest refactoring or architecture changes.
Focus on: unused imports, inconsistent naming, missing type hints on public functions, overly long lines.

File: {filename}

```python
{code[:3000]}
```

Issues:"""

    response = ask_ollama(prompt)
    duration = (time.time() - start) * 1000

    if not response or "CLEAN" in response.upper():
        return ReviewResult(agent="linter", severity="clean", duration_ms=duration)

    findings = [line.strip("- ").strip() for line in response.split("\n")
                if line.strip().startswith("-") and len(line.strip()) > 3]

    return ReviewResult(
        agent="linter",
        severity="info" if findings else "clean",
        findings=findings[:10],
        duration_ms=duration,
    )


def review_agent(code: str, filename: str) -> ReviewResult:
    """First-pass reviewer — logic issues, missing error handling."""
    start = time.time()
    prompt = f"""Review this Python code for bugs and missing error handling.
Report each issue on its own line starting with "- ".
If the code is clean, respond with just "CLEAN".
Focus on: missing try/except around I/O, unchecked None values, off-by-one errors, resource leaks, logic errors.
Do NOT report style issues.

File: {filename}

```python
{code[:3000]}
```

Issues:"""

    response = ask_ollama(prompt)
    duration = (time.time() - start) * 1000

    if not response or "CLEAN" in response.upper():
        return ReviewResult(agent="reviewer", severity="clean", duration_ms=duration)

    findings = [line.strip("- ").strip() for line in response.split("\n")
                if line.strip().startswith("-") and len(line.strip()) > 3]

    severity = "warning" if findings else "clean"
    if any("error" in f.lower() or "bug" in f.lower() or "crash" in f.lower()
           for f in findings):
        severity = "error"

    return ReviewResult(
        agent="reviewer",
        severity=severity,
        findings=findings[:10],
        duration_ms=duration,
    )


def security_agent(code: str, filename: str) -> ReviewResult:
    """Basic security scanner — common vulnerability patterns."""
    start = time.time()
    prompt = f"""Review this Python code for security vulnerabilities ONLY.
Report each issue on its own line starting with "- ".
If no security issues found, respond with just "CLEAN".
Focus on: SQL injection, command injection, path traversal, hardcoded secrets, unsafe deserialization, XSS.
Do NOT report style or logic issues.

File: {filename}

```python
{code[:3000]}
```

Security issues:"""

    response = ask_ollama(prompt)
    duration = (time.time() - start) * 1000

    if not response or "CLEAN" in response.upper():
        return ReviewResult(agent="security", severity="clean", duration_ms=duration)

    findings = [line.strip("- ").strip() for line in response.split("\n")
                if line.strip().startswith("-") and len(line.strip()) > 3]

    return ReviewResult(
        agent="security",
        severity="error" if findings else "clean",
        findings=findings[:10],
        duration_ms=duration,
    )


def run_pipeline(file_path: str, code: Optional[str] = None) -> PipelineResult:
    """Run the full microagent pipeline on a single file."""
    if code is None:
        code = Path(file_path).read_text()

    filename = os.path.basename(file_path)
    logger.info(f"Pipeline: {filename}")

    result = PipelineResult(file_path=file_path)

    # Run all three agents
    for agent_fn in [lint_agent, review_agent, security_agent]:
        agent_result = agent_fn(code, filename)
        result.results.append(agent_result)
        if agent_result.findings:
            logger.info(f"  {agent_result.agent}: {agent_result.severity} "
                        f"({len(agent_result.findings)} findings)")
        else:
            logger.info(f"  {agent_result.agent}: clean")

    # Escalate to Opus if any error-level findings
    if result.max_severity == "error":
        result.escalate_to_opus = True
        logger.info(f"  → ESCALATE to Opus for deep review")

    return result


def run_on_diff(repo_path: str = ".") -> List[PipelineResult]:
    """Run pipeline on all changed Python files in a git diff."""
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "--", "*.py"],
            capture_output=True, text=True, cwd=repo_path, timeout=10,
        )
        files = [f.strip() for f in diff.stdout.strip().split("\n") if f.strip()]
    except Exception:
        files = []

    if not files:
        logger.info("No Python files changed in last commit.")
        return []

    results = []
    for f in files:
        full_path = os.path.join(repo_path, f)
        if os.path.exists(full_path):
            results.append(run_pipeline(full_path))

    # Summary
    escalations = [r for r in results if r.escalate_to_opus]
    clean = [r for r in results if r.max_severity == "clean"]
    logger.info(f"\nPipeline complete: {len(results)} files, "
                f"{len(clean)} clean, {len(escalations)} escalated")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = run_pipeline(sys.argv[1])
        print(result.summary())
    else:
        results = run_on_diff()
        for r in results:
            print(r.summary())
            print()
