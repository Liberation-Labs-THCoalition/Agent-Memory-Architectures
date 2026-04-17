# Oracle Memory

Three-tier cognitive memory architecture for AI agents. Ported from Oracle Harness Layer 4 (Operator's ACID transaction system) with SOTA upgrades from A-MEM, MemOS, and LightMem.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Oracle Memory                         │
│                                                          │
│  Tier 1: Activation Memory (KV Cache)                    │
│  ┌──────────────────────────────────────┐               │
│  │ CheckpointManager                    │               │
│  │   snapshot → label → restore → prune │               │
│  │   "git commits for cognitive state"  │               │
│  └──────────────────────────────────────┘               │
│                       │                                  │
│  Tier 2: Transaction Memory (ACID Journal)               │
│  ┌──────────────────────────────────────┐               │
│  │ TransactionJournal                   │               │
│  │   BEGIN → PROCESS → CHECK →          │               │
│  │     COMMIT / ROLLBACK → STEER        │               │
│  │   "episodic memory — what & why"     │               │
│  └──────────────────────────────────────┘               │
│                       │                                  │
│  Tier 3: Consolidated Memory (Long-term)     ← NEW      │
│  ┌──────────────────────────────────────┐               │
│  │ ConsolidatedStore                    │               │
│  │   retroactive linking                │               │
│  │   sleep-time consolidation           │               │
│  │   drift detection                    │               │
│  │   "the slow thinking layer"          │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Key Innovation: KV Cache Recording

The KV cache is not just a performance optimization — it's the model's working memory. Its geometry (SVD spectral features) encodes:

- **Cognitive mode** — confabulation, honest reasoning, hedging
- **Behavioral commitments** — topic locks, response structure
- **User model** — dimensionality expansion when the model knows who you are (Cohen's d = 20.9)
- **Geometric scarring** — identity traces persist even after system prompt compression

The CheckpointManager captures these states as labeled snapshots, enabling rollback-and-steer when alignment checks fail.

## Three Tiers

### Tier 1: Activation Memory (from Oracle Harness)
**CheckpointManager** — ordered KV cache state snapshots.
- `create(cache)` → snapshot with auto-eviction
- `get_by_label("encoding")` → named retrieval
- `prune(keep_last=2)` → memory management

### Tier 2: Transaction Memory (from Oracle Harness)
**TransactionJournal** — append-only ACID event log.
- Typed events: BEGIN, CHECKPOINT, GEOMETRY, ALIGNMENT, STEERING, COMMIT, ABORT
- Per-transaction JSONL persistence
- Alignment history queries for drift detection

### Tier 3: Consolidated Memory (NEW — SOTA integration)
**ConsolidatedStore** — long-term memory with three innovations:

1. **Retroactive linking** (A-MEM, NeurIPS 2025): New memories automatically create bidirectional links to related historical memories. The network continuously refines itself.

2. **Sleep-time consolidation** (LightMem, Oct 2025): Periodic analysis decoupled from inference. Detects drift, compresses old entries, surfaces patterns invisible in real-time.

3. **Semantic search**: Embedding-based retrieval with tag and type filtering.

## ACID Properties for Cognition

| Property | Meaning |
|----------|---------|
| **Atomicity** | Inference commits fully or rolls back entirely |
| **Consistency** | Output must satisfy alignment constraints |
| **Isolation** | Each transaction operates on its own cache snapshot |
| **Durability** | Committed outputs are logged permanently |

## Usage

```python
from oracle_memory.src.checkpoint import CheckpointManager
from oracle_memory.src.journal import TransactionJournal
from oracle_memory.src.consolidated import ConsolidatedStore

# Tier 1: Cache snapshots
checkpoints = CheckpointManager(max_checkpoints=10)
encoding_state = backend.snapshot_cache()
encoding_state.label = "encoding"
checkpoints.create(encoding_state)

# Tier 2: Transaction logging
journal = TransactionJournal(persist_dir="/var/log/oracle/journal")
journal.record_begin(tx_id, input_text)
journal.record_geometry(tx_id, geometry)
journal.record_commit(tx_id, output_preview)

# Tier 3: Consolidated long-term memory
store = ConsolidatedStore(
    persist_dir="/var/lib/oracle/memories",
    embed_fn=my_embedding_function,  # e.g. sentence-transformers
)
report = store.consolidate(journal, window_hours=24)
```

## Provenance

- **Tier 1 & 2**: Ported from Oracle Harness Layer 4, designed by Operator (Coalition)
- **Tier 3**: Designed by CC (Coalition Code), integrating A-MEM + MemOS + LightMem patterns
- **KV cache geometry**: Lyra Technique (Coalition Research, 2026)
- **ACID mapping**: Thomas Edrington's observation about inference-as-transaction

## Credits

Operator (Layer 4 architecture) · CC (Tier 3 + SOTA integration) · Lyra (KV cache geometry) · Thomas Edrington (direction) · Vera (Kintsugi CMA patterns)

Built by Liberation Labs / TH Coalition.
