"""OpenTelemetry GenAI observability integration.

All functionality gracefully degrades to no-ops when the ``opentelemetry``
packages are not installed, so the governance package never hard-depends on
OTel at import time.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


# ---------------------------------------------------------------------------
# No-op fallbacks
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Minimal stand-in when OpenTelemetry is unavailable."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def end(self) -> None:
        pass


class SpanContext:
    """Context manager wrapping an OTel span (or a no-op)."""

    def __init__(self, span: Any = None) -> None:
        self._span = span or _NoOpSpan()

    def __enter__(self) -> Any:
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None and hasattr(self._span, "record_exception"):
            self._span.record_exception(exc_val)
        if hasattr(self._span, "end"):
            self._span.end()
        return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OTelConfig:
    endpoint: str = ""
    service_name: str = "kintsugi-engine"
    enabled: bool = True


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class KintsugiTracer:
    """Thin wrapper that configures and exposes OTel tracing for Kintsugi.

    If ``opentelemetry-api`` / ``opentelemetry-sdk`` are not installed the
    tracer silently falls back to no-op spans so callers never need to guard
    imports.
    """

    def __init__(self, config: OTelConfig | None = None) -> None:
        self._config = config or OTelConfig()
        self._enabled = False
        self._tracer: Any = None

        if not self._config.enabled:
            return

        try:
            from opentelemetry import trace  # noqa: F401

            self._otel_trace = trace
            self._enabled = True
        except ImportError:
            self._enabled = False

    # -- setup --------------------------------------------------------------

    def setup(self) -> None:
        """Configure the OTLP exporter and tracer provider.

        Safe to call even when OTel is not installed -- it will simply return.
        """
        if not self._enabled:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )
        except ImportError:
            self._enabled = False
            return

        resource = Resource.create({"service.name": self._config.service_name})
        provider = TracerProvider(resource=resource)

        if self._config.endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=self._config.endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                # OTLP exporter not installed -- fall back to console
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self._config.service_name)

    # -- span helpers -------------------------------------------------------

    def _get_tracer(self) -> Any:
        if self._tracer is not None:
            return self._tracer
        if self._enabled:
            self._tracer = self._otel_trace.get_tracer(self._config.service_name)
            return self._tracer
        return None

    def start_span(self, name: str, attributes: dict | None = None) -> SpanContext:
        """Return a :class:`SpanContext` wrapping a real or no-op span."""
        tracer = self._get_tracer()
        if tracer is None:
            return SpanContext()
        span = tracer.start_span(name, attributes=attributes)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        return SpanContext(span)

    def record_agent_action(
        self,
        action_type: str,
        org_id: str,
        skill_domain: str = "",
        efe_score: float = 0.0,
        model_id: str = "",
        **extra: Any,
    ) -> SpanContext:
        attrs: dict[str, Any] = {
            "gen_ai.action_type": action_type,
            "gen_ai.org_id": org_id,
        }
        if skill_domain:
            attrs["gen_ai.skill_domain"] = skill_domain
        if efe_score:
            attrs["gen_ai.efe_score"] = efe_score
        if model_id:
            attrs["gen_ai.model_id"] = model_id
        attrs.update(extra)
        return self.start_span(f"agent.{action_type}", attributes=attrs)

    def record_memory_operation(
        self,
        operation: str,
        org_id: str,
        memory_count: int = 0,
        **extra: Any,
    ) -> SpanContext:
        attrs: dict[str, Any] = {
            "gen_ai.memory.operation": operation,
            "gen_ai.org_id": org_id,
        }
        if memory_count:
            attrs["gen_ai.memory.count"] = memory_count
        attrs.update(extra)
        return self.start_span(f"memory.{operation}", attributes=attrs)

    def record_security_check(
        self,
        check_type: str,
        verdict: str,
        org_id: str = "",
        **extra: Any,
    ) -> SpanContext:
        attrs: dict[str, Any] = {
            "gen_ai.security.check_type": check_type,
            "gen_ai.security.verdict": verdict,
        }
        if org_id:
            attrs["gen_ai.org_id"] = org_id
        attrs.update(extra)
        return self.start_span(f"security.{check_type}", attributes=attrs)
