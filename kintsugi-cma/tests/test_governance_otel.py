"""Tests for kintsugi.governance.otel module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kintsugi.governance.otel import (
    KintsugiTracer,
    OTelConfig,
    SpanContext,
    _NoOpSpan,
)


# ---------------------------------------------------------------------------
# OTelConfig
# ---------------------------------------------------------------------------

class TestOTelConfig:
    def test_defaults(self):
        cfg = OTelConfig()
        assert cfg.endpoint == ""
        assert cfg.service_name == "kintsugi-engine"
        assert cfg.enabled is True

    def test_custom(self):
        cfg = OTelConfig(endpoint="http://localhost:4317", service_name="test", enabled=False)
        assert cfg.endpoint == "http://localhost:4317"
        assert cfg.service_name == "test"
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# _NoOpSpan
# ---------------------------------------------------------------------------

class TestNoOpSpan:
    def test_set_attribute(self):
        span = _NoOpSpan()
        span.set_attribute("key", "value")  # should not raise

    def test_set_status(self):
        span = _NoOpSpan()
        span.set_status("OK", "fine")

    def test_record_exception(self):
        span = _NoOpSpan()
        span.record_exception(RuntimeError("boom"))

    def test_end(self):
        span = _NoOpSpan()
        span.end()


# ---------------------------------------------------------------------------
# SpanContext
# ---------------------------------------------------------------------------

class TestSpanContext:
    def test_context_manager_no_exception(self):
        span = _NoOpSpan()
        with SpanContext(span) as s:
            assert s is span

    def test_context_manager_with_exception(self):
        mock_span = MagicMock()
        with pytest.raises(ValueError):
            with SpanContext(mock_span) as s:
                raise ValueError("test error")
        mock_span.record_exception.assert_called_once()
        mock_span.end.assert_called_once()

    def test_context_manager_default_noop(self):
        ctx = SpanContext()
        with ctx as s:
            assert isinstance(s, _NoOpSpan)

    def test_end_called_on_normal_exit(self):
        mock_span = MagicMock()
        with SpanContext(mock_span):
            pass
        mock_span.end.assert_called_once()
        mock_span.record_exception.assert_not_called()


# ---------------------------------------------------------------------------
# KintsugiTracer (OTel NOT installed)
# ---------------------------------------------------------------------------

class TestKintsugiTracerNoOtel:
    def test_init_disabled_config(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        assert tracer._enabled is False

    def test_init_otel_not_installed(self):
        # Default path - opentelemetry likely not installed in test env
        tracer = KintsugiTracer()
        # Either enabled=False (not installed) or True (installed) - both ok
        # The key thing: start_span should always work
        ctx = tracer.start_span("test")
        assert isinstance(ctx, SpanContext)

    def test_start_span_returns_span_context(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        ctx = tracer.start_span("test.span", attributes={"k": "v"})
        assert isinstance(ctx, SpanContext)
        with ctx as s:
            assert isinstance(s, _NoOpSpan)

    def test_record_agent_action(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        ctx = tracer.record_agent_action(
            action_type="plan",
            org_id="org1",
            skill_domain="coding",
            efe_score=0.5,
            model_id="gpt-4",
            custom_key="val",
        )
        assert isinstance(ctx, SpanContext)

    def test_record_agent_action_minimal(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        ctx = tracer.record_agent_action(action_type="plan", org_id="org1")
        assert isinstance(ctx, SpanContext)

    def test_record_memory_operation(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        ctx = tracer.record_memory_operation(
            operation="store", org_id="org1", memory_count=5, extra_key="x"
        )
        assert isinstance(ctx, SpanContext)

    def test_record_memory_operation_minimal(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        ctx = tracer.record_memory_operation(operation="retrieve", org_id="org1")
        assert isinstance(ctx, SpanContext)

    def test_record_security_check(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        ctx = tracer.record_security_check(
            check_type="auth", verdict="pass", org_id="org1", detail="ok"
        )
        assert isinstance(ctx, SpanContext)

    def test_record_security_check_minimal(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        ctx = tracer.record_security_check(check_type="auth", verdict="pass")
        assert isinstance(ctx, SpanContext)

    def test_setup_noop_when_disabled(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        tracer.setup()  # should not raise
        assert tracer._tracer is None

    def test_get_tracer_returns_none_when_disabled(self):
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        assert tracer._get_tracer() is None


# ---------------------------------------------------------------------------
# KintsugiTracer with mocked OTel
# ---------------------------------------------------------------------------

class TestKintsugiTracerWithMockedOtel:
    def _make_tracer_with_mock(self):
        """Create a KintsugiTracer that thinks OTel is available."""
        tracer = KintsugiTracer(OTelConfig(enabled=False))
        # Manually wire up as if otel was found
        tracer._enabled = True
        mock_otel_trace = MagicMock()
        mock_otel_tracer = MagicMock()
        mock_otel_trace.get_tracer.return_value = mock_otel_tracer
        tracer._otel_trace = mock_otel_trace
        return tracer, mock_otel_tracer

    def test_start_span_with_attributes(self):
        tracer, mock_tracer = self._make_tracer_with_mock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        ctx = tracer.start_span("test.op", attributes={"a": 1})
        assert isinstance(ctx, SpanContext)
        mock_tracer.start_span.assert_called_once_with("test.op", attributes={"a": 1})
        mock_span.set_attribute.assert_called_with("a", 1)

    def test_start_span_no_attributes(self):
        tracer, mock_tracer = self._make_tracer_with_mock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        ctx = tracer.start_span("test.op")
        mock_tracer.start_span.assert_called_once_with("test.op", attributes=None)

    def test_get_tracer_caches(self):
        tracer, mock_tracer = self._make_tracer_with_mock()
        t1 = tracer._get_tracer()
        t2 = tracer._get_tracer()
        assert t1 is t2
        # get_tracer called only once
        tracer._otel_trace.get_tracer.assert_called_once()
