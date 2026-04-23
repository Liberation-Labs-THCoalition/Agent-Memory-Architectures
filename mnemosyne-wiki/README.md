# Mnemosyne Wiki — Knowledge Graph to Browsable Markdown

**The readable surface of an agent's knowledge.**

Takes a knowledge graph (entities + typed predicates from HippoRAG/CatRAG
or any graph store) and generates interlinked markdown pages. Runs
periodically via cron or on-demand. Produces a wiki that humans and
agents can browse.

## Why

Knowledge graphs are queryable but opaque. You can ask "what connects
A to B?" and get a traversal result, but you can't browse the graph's
shape the way you browse a wiki. This layer makes the graph readable.

Agents benefit too: an LLM primed with a wiki page about "Oracle Harness"
gets structured context (what was built, who built it, what it connects
to) rather than a bag of embedding-similar chunks.

## Architecture

```
Knowledge Graph (PostgreSQL)          Wiki Output (markdown files)
┌──────────────────────┐              ┌──────────────────────┐
│ kg_entities          │──generates──►│ topics/oracle.md     │
│ kg_triples           │              │ entities/thomas.md   │
│ memories (content)   │              │ timeline/2026-04.md  │
└──────────────────────┘              │ findings/robust.md   │
                                      │ index.md             │
                                      └──────────────────────┘
```

## Wiki Structure

```
wiki/
├── index.md              # Table of contents, stats, recent updates
├── topics/               # One page per tag cluster
│   ├── oracle-harness.md
│   ├── kintsugi.md
│   └── ...
├── entities/             # One page per person/project/org
│   ├── thomas.md
│   ├── lyra.md
│   └── ...
├── timeline/             # Memories grouped by month
│   ├── 2026-04.md
│   └── ...
├── findings/             # Grouped by confidence tier
│   ├── robust.md         # High significance, frequently accessed
│   ├── stale.md          # Not accessed in 30+ days
│   └── ...
└── graph/                # Graph-derived pages
    ├── relationships.md  # Top typed predicates with examples
    └── clusters.md       # Densely connected entity groups
```

## Page Cross-Linking

Links between pages are derived from the knowledge graph:

1. **Tag overlap**: memory tagged "oracle" links to `topics/oracle.md`
2. **Entity mentions**: content mentioning "Thomas" links to `entities/thomas.md`
3. **Graph edges**: typed predicates become explicit links —
   "CC **built** Oracle Harness" creates bidirectional links between
   `entities/cc.md` and `topics/oracle-harness.md`
4. **Temporal**: all memories link to their month's timeline page

## Dreamer Integration

The wiki daemon is designed to pair with a graph enrichment dreamer:

```
Dreamer enriches graph    →   Wiki regenerates pages
(co_occurs_with → built)      (new cross-links appear)
      ↑                              │
      └── reads enriched graph ──────┘
```

The dreamer improves the graph; the wiki surfaces the improvements.
Each nightly cycle produces a richer, more connected wiki.

## Implementation

See `/home/asdf/ember/WIKI_DAEMON_SPEC.md` in the Ember repo for
the daemon spec. The daemon reads PostgreSQL, generates markdown,
runs as a cron job or Ember task.

Core generator: ~200 lines Python. No dependencies beyond psycopg2.

## Karpathy + HippoRAG = Mnemosyne

- **Karpathy's LLM Wiki pattern**: LLM maintains interlinked markdown
  as a knowledge structure. Active curation, not passive storage.
- **HippoRAG 2**: hippocampal-inspired knowledge graph with pattern
  separation/completion. Associative retrieval via PPR.
- **Mnemosyne Wiki**: the browsable surface that makes both accessible.

The wiki doesn't replace the graph or the retrieval system. It's the
layer that makes them legible.

## Credits

Pattern: Andrej Karpathy (LLM Wiki), A-MEM (Zettelkasten linking)
Graph: HippoRAG 2 (ICML 2025), CatRAG (Feb 2026)
Implementation: CC (Coalition Code)
Direction: Thomas Edrington
