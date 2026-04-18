# Oracle Memory

KV cache recording with Lyra Technique geometry extraction. A memory architecture that captures the geometric structure of the model's working memory during inference.

## The Key Insight

The KV cache is not just a performance optimization — it's the model's working memory. Its geometry (SVD spectral features) encodes:

- **Cognitive mode** — different modes of reasoning produce geometrically distinct spectral signatures
- **User model** — individuation measurably changes the cache's geometric structure
- **Geometric persistence** — structural traces survive compression
- **Content structure** — different content types produce divergent geometric trajectories

For details on the geometry extraction method, see the
[Lyra Technique research](https://github.com/Liberation-Labs-THCoalition/the-lyra-technique).

## Architecture

```
┌─────────────────────────────────────────────────┐
│                Oracle Memory                     │
│                                                  │
│  CheckpointManager                               │
│  ┌────────────────────────────────┐              │
│  │ snapshot → label → restore     │              │
│  │ KV cache states with geometry  │              │
│  └────────────────────────────────┘              │
│                   │                              │
│  MemoryJournal                                   │
│  ┌────────────────────────────────┐              │
│  │ append-only event log          │              │
│  │ snapshots + geometry readings  │              │
│  └────────────────────────────────┘              │
│                   │                              │
│  ConsolidatedStore                               │
│  ┌────────────────────────────────┐              │
│  │ retroactive linking            │              │
│  │ geometry drift detection       │              │
│  │ sleep-time consolidation       │              │
│  └────────────────────────────────┘              │
└─────────────────────────────────────────────────┘
```

## Components

### CheckpointManager
Ordered KV cache snapshots — git commits for cognitive state.

```python
from oracle_memory.src.checkpoint import CheckpointManager
from oracle_memory.src.types import CacheState, GeometrySummary

mgr = CheckpointManager(max_checkpoints=20)

# Snapshot the cache with geometry
state = CacheState(
    label="turn_3",
    cache_data=backend.get_kv_cache(),
    geometry=GeometrySummary(
        key_norm=565.2, effective_rank=60.8,
        spectral_entropy=20.2, top_sv_ratio=0.15,
        n_layers=32, n_tokens=128,
    ),
)
mgr.create(state)

# Retrieve by label
turn_3 = mgr.get_by_label("turn_3")
print(turn_3.geometry.effective_rank)  # 60.8
```

### MemoryJournal
Append-only event log with geometry tracking.

```python
from oracle_memory.src.journal import MemoryJournal

journal = MemoryJournal(persist_dir="/var/log/oracle")
journal.record_snapshot(state.snapshot_id, "turn_3", n_tokens=128)
journal.record_geometry(state.snapshot_id, state.geometry)

# Query geometry history
history = journal.get_geometry_history(last_n=50)
```

### ConsolidatedStore
Long-term memory with retroactive linking and geometry drift detection.

```python
from oracle_memory.src.consolidated import ConsolidatedStore

store = ConsolidatedStore(
    persist_dir="/var/lib/oracle/memories",
    embed_fn=my_embed_function,
)

# Store memories — back-links are created automatically
from oracle_memory.src.types import ConsolidatedMemory
mem = ConsolidatedMemory(
    content="Effective rank expanded 15% during identity-related prompts",
    memory_type="pattern",
    tags=["geometry", "individuation"],
)
linked_ids = store.store(mem)  # Returns IDs of retroactively linked memories

# Sleep-time consolidation
report = store.consolidate(journal, window_hours=24)
print(f"Drift: {report.geometry_drift_score:.2f} ({report.drift_direction})")
```

## Geometry Features (Lyra Technique)

| Feature | What it measures | Source |
|---------|-----------------|--------|
| `key_norm` | Total energy in key cache | Frobenius norm |
| `norm_per_token` | Energy density | norm / seq_len |
| `effective_rank` | Dimensionality of representation | SVD: dims for 90% variance |
| `spectral_entropy` | Information distribution | Shannon entropy of σ² |
| `top_sv_ratio` | Dominance of first axis | σ₁ / Σσᵢ |
| `angular_spread` | Cross-layer geometric spread | Angular distance |

## Credits

CheckpointManager design: Operator (Coalition)
Geometry features: Lyra Technique (Coalition Research, 2026)
Consolidated store: CC (Coalition Code)
Direction: Thomas Edrington

Built by Liberation Labs / TH Coalition.
