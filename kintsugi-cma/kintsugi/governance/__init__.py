"""Kintsugi Governance Layer -- Phase 2.

Consensus gating and observability for agent actions.
"""

from kintsugi.governance.consensus import (
    ConsensusPriority,
    ConsentCategory,
    ConsentItem,
    ConsentStatus,
    ConsensusConfig,
    ConsensusGate,
)
from kintsugi.governance.otel import (
    KintsugiTracer,
    OTelConfig,
    SpanContext,
)

__all__ = [
    # consensus
    "ConsensusPriority",
    "ConsentCategory",
    "ConsentItem",
    "ConsentStatus",
    "ConsensusConfig",
    "ConsensusGate",
    # otel
    "KintsugiTracer",
    "OTelConfig",
    "SpanContext",
]
