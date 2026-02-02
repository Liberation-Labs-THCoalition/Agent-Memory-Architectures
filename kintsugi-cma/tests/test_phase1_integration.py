"""Kintsugi Phase 1 integration smoke test."""
from datetime import datetime, timezone

# --- Imports ---
from kintsugi.security.intent_capsule import sign_capsule, verify_capsule
from kintsugi.security.shield import Shield, ShieldConfig
from kintsugi.security.monitor import SecurityMonitor
from kintsugi.security.pii import PIIRedactor
from kintsugi.memory.cma_stage1 import Turn, segment_dialogue
from kintsugi.memory.significance import MemoryLayer, compute_layer
from kintsugi.memory.spaced import fib_interval
from kintsugi.config.values_loader import load_from_template

print("All imports OK")

# --- Intent Capsule ---
capsule = sign_capsule("test-goal", {"tools": ["search"]}, "org-1", "secret")
assert verify_capsule(capsule, "secret")
print("Capsule sign/verify OK")

# --- Memory layers + fibonacci ---
assert compute_layer(1) == MemoryLayer.PERMANENT
assert compute_layer(9) == MemoryLayer.VOLATILE
assert fib_interval(5) == 8
print("Significance + spaced OK")

# --- CMA segmentation ---
now = datetime.now(timezone.utc)
turns = [Turn(role="user", content=f"message {i}", timestamp=now) for i in range(25)]
windows = segment_dialogue(turns)
assert len(windows) > 0
print(f"CMA: {len(windows)} windows from 25 turns")

# --- Shield ---
cfg = ShieldConfig.from_dict({
    "budget_session_limit": 10.0,
    "budget_daily_limit": 100.0,
    "egress_allowlist": ["example.com"],
    "rate_limits": {},
    "circuit_breaker_threshold": 5,
})
shield = Shield(cfg)
v1 = shield.check_action("search", cost=1.0, url="https://example.com/api", tool="search")
assert v1.decision.value == "ALLOW", f"Expected ALLOW, got {v1}"
v2 = shield.check_action("search", cost=1.0, url="https://evil.com/api", tool="search")
assert v2.decision.value == "BLOCK", f"Expected BLOCK, got {v2}"
print("Shield allow/block OK")

# --- PII ---
redactor = PIIRedactor()
result = redactor.redact("Call 555-123-4567 or email test@example.com")
assert "555-123-4567" not in result.redacted_text
assert "test@example.com" not in result.redacted_text
print("PII redaction OK")

# --- Monitor ---
monitor = SecurityMonitor()
dangerous = monitor.check_command("rm -rf /")
assert dangerous.verdict.value != "ALLOW", f"Expected detection, got {dangerous}"
safe = monitor.check_command("ls -la")
assert safe.verdict.value == "ALLOW", f"Expected allow, got {safe}"
print("Monitor detect/pass OK")

# --- VALUES template ---
vals = load_from_template("mutual_aid")
assert vals.organization.name != ""
print(f"VALUES template: {vals.organization.name}")

print("\n=== ALL PHASE 1 INTEGRATION TESTS PASSED ===")
