"""Swarm Service — NATS-triggered microagent pipeline.

Subscribes to system.pipeline on NATS. When a message arrives with
a file path or repo, runs the lint/review/security pipeline and
publishes results back to NATS + records in swarm memory.

Can also be triggered manually: python swarm_service.py <file_path>

Author: Nexus (Coalition)
Date: 2026-04-20
"""

import asyncio
import json
import logging
import os
import sys
import time

import requests

logging.basicConfig(level=logging.INFO, format='[swarm-svc] %(message)s')
logger = logging.getLogger(__name__)

# Add swarm dir to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import run_pipeline, run_on_diff
from swarm_memory import SwarmMemory

NATS_URL = os.environ.get("NATS_URL", "nats://127.0.0.1:4222")
NATS_USER = os.environ.get("NATS_USER", "nexus")
NATS_PASS = os.environ.get("NATS_PASS", "op-nats-changeme")
DISCORD_TOKEN_FILE = "/home/admin/.discord_bot_token"
DISCORD_CHANNEL = "1488699407762329652"

memory = SwarmMemory()


def notify_discord(message: str):
    """Post swarm results to Discord."""
    try:
        token = open(DISCORD_TOKEN_FILE).read().strip()
        requests.post(
            f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages",
            headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
            json={"content": message[:2000]},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Discord notify error: {e}")


def process_file(file_path: str) -> dict:
    """Run pipeline on a file and record results in swarm memory."""
    start = time.time()
    result = run_pipeline(file_path)
    duration = (time.time() - start) * 1000

    # Record each finding in swarm memory
    for agent_result in result.results:
        for finding in agent_result.findings:
            # Check if there's a known solution
            solution = memory.find_solution(finding, agent_result.agent)

            finding_id = memory.record(
                agent=agent_result.agent,
                file_path=file_path,
                finding=finding,
                severity=agent_result.severity,
            )

            # Auto-detect likely roasts (hallucinated vulnerabilities)
            roast_keywords = [
                "sql injection" if "sql" not in open(file_path).read().lower() else None,
                "xss" if "html" not in open(file_path).read().lower() else None,
            ]
            if any(k and k in finding.lower() for k in roast_keywords if k):
                memory.mark_as_roast(finding_id)
                logger.info(f"  🌵 Roast detected: {finding[:60]}")

    # Record pipeline run
    total_findings = sum(len(r.findings) for r in result.results)
    memory.record_pipeline_run(
        files=1,
        findings=total_findings,
        escalated=1 if result.escalate_to_opus else 0,
        clean=1 if result.max_severity == "clean" else 0,
        duration_ms=duration,
    )

    return {
        "file": file_path,
        "severity": result.max_severity,
        "escalate": result.escalate_to_opus,
        "findings": total_findings,
        "duration_ms": duration,
        "summary": result.summary(),
    }


async def nats_listener():
    """Listen on NATS for pipeline requests."""
    try:
        import nats
    except ImportError:
        logger.error("nats-py not installed")
        return

    nc = await nats.connect(NATS_URL, user=NATS_USER, password=NATS_PASS)
    js = nc.jetstream()

    # Ensure stream exists
    try:
        await js.add_stream(name="SYSTEM_EVENTS", subjects=["system.>"])
    except:
        pass

    logger.info("Swarm service listening on system.pipeline")

    async def handler(msg):
        try:
            data = json.loads(msg.data)
            file_path = data.get("file_path")
            repo_path = data.get("repo_path")

            if file_path and os.path.exists(file_path):
                logger.info(f"Pipeline request: {file_path}")
                result = process_file(file_path)

                # Publish result back to NATS
                await js.publish("system.pipeline.results", json.dumps({
                    "from": "swarm",
                    "type": "pipeline_result",
                    **result,
                    "timestamp": time.time(),
                }).encode())

                # Notify Discord if escalation needed
                if result["escalate"]:
                    notify_discord(
                        f"**[Swarm]** Escalating `{os.path.basename(file_path)}` "
                        f"to Opus — {result['findings']} findings, "
                        f"severity: {result['severity']}"
                    )

            elif repo_path:
                logger.info(f"Diff pipeline request: {repo_path}")
                results = run_on_diff(repo_path)
                for r in results:
                    result = process_file(r.file_path)

            await msg.ack()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await msg.ack()

    await js.subscribe("system.pipeline", durable="swarm-pipeline", cb=handler)

    # Keep running
    while True:
        await asyncio.sleep(60)


def main():
    """Entry point — either NATS listener or manual file review."""
    if len(sys.argv) > 1:
        # Manual mode
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            result = process_file(file_path)
            print(result["summary"])
            print(f"\nStats: {json.dumps(memory.stats(), indent=2)}")
        else:
            print(f"File not found: {file_path}")
    else:
        # NATS listener mode
        asyncio.run(nats_listener())


if __name__ == "__main__":
    main()
