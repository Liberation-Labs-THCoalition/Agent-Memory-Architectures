"""MCP server exposing memory tools for Claude Desktop / Cowork / Dispatch.

Run with: python -m dispatch_memory.server
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from .consolidation import Consolidator
from .entities import EntityExtractor
from .models import (
    BootstrapPayload, Memory, MemoryStatus, MemoryType, TTLClass,
)
from .storage import EmbeddingCache, NotionStore

logger = logging.getLogger(__name__)


class MemoryService:
    """Core memory service orchestrating Notion, embeddings, entities, and consolidation."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        notion_cfg = self.config["notion"]
        self.notion = NotionStore(
            token=notion_cfg["token"],
            database_ids=notion_cfg["databases"],
        )

        storage_cfg = self.config["storage"]
        self.cache = EmbeddingCache(
            db_path=storage_cfg["sqlite_path"],
            model_name=storage_cfg["embedding_model"],
            embedding_dim=storage_cfg["embedding_dim"],
        )

        entity_cfg = self.config.get("entities", {})
        self.extractor = EntityExtractor(
            model_name=entity_cfg.get("spacy_model", "en_core_web_sm")
        )

        sig_cfg = self.config.get("significance", {})
        self.consolidator = Consolidator(
            archive_threshold=sig_cfg.get("archive_threshold", 0.2),
            forget_threshold=sig_cfg.get("forget_threshold", 0.1),
        )

        self.significance_defaults = sig_cfg.get("defaults", {})
        self.half_lives = sig_cfg.get("half_lives", {})

    # ── Phase 1: Core Operations ──────────────────────────────────────

    async def memory_store(
        self,
        content: str,
        memory_type: str = "fact",
        tags: Optional[list[str]] = None,
        significance: Optional[float] = None,
        ttl_class: str = "medium",
    ) -> dict:
        """Create a memory: extract entities, embed, store in Notion + local cache."""
        mtype = MemoryType(memory_type)

        # Determine significance
        if significance is None:
            significance = self.significance_defaults.get(memory_type, 0.5)

        # Extract entities
        entities = self.extractor.extract(content)

        # Build memory
        memory = Memory(
            content=content,
            memory_type=mtype,
            tags=tags or [],
            entities=entities,
            significance=significance,
            ttl_class=TTLClass(ttl_class),
        )

        # Store in Notion
        memory = await self.notion.store(memory)

        # Index locally
        self.cache.index_memory(
            memory_id=memory.id,
            content=content,
            notion_page_id=memory.notion_page_id,
            content_hash=memory.content_hash,
            memory_type=memory_type,
            significance=significance,
            tags=tags or [],
            entities=[e.name for e in entities],
        )

        return {
            "id": memory.id,
            "notion_page_id": memory.notion_page_id,
            "entities_extracted": len(entities),
            "entity_names": [e.name for e in entities],
            "significance": significance,
            "ttl_class": ttl_class,
        }

    async def memory_search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_significance: Optional[float] = None,
    ) -> list[dict]:
        """Semantic search via local embedding cache."""
        results = self.cache.search(
            query=query,
            limit=limit,
            memory_type=memory_type,
            min_significance=min_significance,
        )

        # Hydrate from Notion for top results
        hydrated = []
        for r in results:
            if r.get("notion_page_id"):
                memory = await self.notion.retrieve(r["notion_page_id"])
                if memory:
                    hydrated.append({
                        "content": memory.content,
                        "memory_type": memory.memory_type.value,
                        "tags": memory.tags,
                        "entities": memory.entity_names,
                        "significance": memory.significance,
                        "similarity_score": r["score"],
                        "notion_page_id": r["notion_page_id"],
                    })
            else:
                hydrated.append(r)

        return hydrated

    async def memory_retrieve(
        self,
        notion_page_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        database: str = "projects",
        limit: int = 20,
    ) -> list[dict]:
        """Fetch memories by ID or type from Notion."""
        if notion_page_id:
            memory = await self.notion.retrieve(notion_page_id)
            if memory:
                return [memory.model_dump(exclude={"embedding"})]
            return []

        mtype = MemoryType(memory_type) if memory_type else None
        memories = await self.notion.query_database(
            database=database,
            memory_type=mtype,
            limit=limit,
        )
        return [m.model_dump(exclude={"embedding"}) for m in memories]

    async def memory_update(
        self,
        notion_page_id: str,
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        significance: Optional[float] = None,
    ) -> dict:
        """Update a memory's content or metadata."""
        memory = await self.notion.retrieve(notion_page_id)
        if not memory:
            return {"error": f"Memory not found: {notion_page_id}"}

        if content:
            memory.content = content
            memory.entities = self.extractor.extract(content)
        if tags is not None:
            memory.tags = tags
        if significance is not None:
            memory.significance = significance

        memory = await self.notion.update(memory)

        # Re-index locally
        if content:
            self.cache.index_memory(
                memory_id=memory.id,
                content=memory.content,
                notion_page_id=memory.notion_page_id,
                content_hash=memory.content_hash,
                memory_type=memory.memory_type.value,
                significance=memory.significance,
                tags=memory.tags,
                entities=[e.name for e in memory.entities],
            )

        return {"updated": True, "notion_page_id": notion_page_id}

    async def memory_delete(self, notion_page_id: str) -> dict:
        """Delete a memory."""
        memory = await self.notion.retrieve(notion_page_id)
        if memory:
            await self.notion.delete(memory)
            self.cache.remove(memory.id)
            return {"deleted": True}
        return {"error": "Memory not found"}

    async def memory_consolidate(self, mode: str = "full") -> dict:
        """Run dream consolidation cycle."""
        # Pull all active memories from Notion
        memories = await self.notion.query_all_active()

        # Run consolidation
        summary = self.consolidator.consolidate(memories, mode=mode)

        # Apply archival actions to Notion
        decay_result = self.consolidator.run_decay_pass(memories)
        for memory in decay_result["archive"]:
            await self.notion.archive(memory)
            self.cache.update_status(memory.id, "archived")

        for memory in decay_result["forget"]:
            self.cache.update_status(memory.id, "forgotten")

        return summary

    # ── Phase 2: Enhanced Operations ──────────────────────────────────

    async def memory_recall(
        self,
        memory_type: Optional[str] = None,
        entities: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        min_significance: Optional[float] = None,
        query: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Compound query across type, entities, significance, and semantic search."""
        results = []

        # If semantic query provided, start with embedding search
        if query:
            results = await self.memory_search(
                query=query,
                limit=limit * 2,  # Over-fetch to allow filtering
                memory_type=memory_type,
                min_significance=min_significance,
            )

        # Also query Notion for structured filters
        for db_name in ["projects", "resources", "inbox"]:
            mtype = MemoryType(memory_type) if memory_type else None
            notion_results = await self.notion.query_database(
                database=db_name,
                memory_type=mtype,
                tags=tags,
                min_significance=min_significance,
                limit=limit,
            )

            for memory in notion_results:
                # Filter by entity if requested
                if entities:
                    memory_entity_names = {e.name_lower for e in memory.entities}
                    if not any(e.lower() in memory_entity_names for e in entities):
                        continue

                entry = memory.model_dump(exclude={"embedding"})
                # Avoid duplicates
                if not any(
                    r.get("notion_page_id") == memory.notion_page_id
                    for r in results
                ):
                    results.append(entry)

        return results[:limit]

    async def memory_refresh(self, notion_page_id: str) -> dict:
        """Bump access metrics without modifying content."""
        memory = await self.notion.retrieve(notion_page_id)
        if not memory:
            return {"error": "Memory not found"}

        memory.refresh()
        await self.notion.update(memory)
        return {"refreshed": True, "access_count": memory.access_count}

    async def memory_compress(self) -> dict:
        """Run significance-aware decay pass."""
        memories = await self.notion.query_all_active()
        result = self.consolidator.run_decay_pass(memories)

        # Apply state changes
        for memory in result["archive"]:
            await self.notion.archive(memory)
            self.cache.update_status(memory.id, "archived")

        for memory in result["forget"]:
            self.cache.update_status(memory.id, "forgotten")

        return {
            "active": len(result["active"]),
            "archived": len(result["archive"]),
            "forgotten": len(result["forget"]),
        }

    async def memory_status(self) -> dict:
        """Return memory stats across all databases."""
        notion_stats = await self.notion.get_stats()
        cache_stats = self.cache.get_stats()
        return {
            "notion": notion_stats,
            "local_cache": cache_stats,
        }

    async def memory_bootstrap(self) -> dict:
        """Return curated context payload for session initialization."""
        boot_cfg = self.config.get("bootstrap", {})
        threshold = boot_cfg.get("significance_threshold", 0.8)
        max_memories = boot_cfg.get("max_memories", 50)

        payload = BootstrapPayload()

        # Standing instructions
        if boot_cfg.get("include_standing_instructions", True):
            instructions = await self.notion.query_database(
                database="resources",
                min_significance=threshold,
                limit=max_memories,
            )
            payload.standing_instructions = [
                m for m in instructions
                if m.memory_type == MemoryType.STANDING_INSTRUCTION
                or m.significance >= 0.9
            ]

        # Active projects
        if boot_cfg.get("include_active_projects", True):
            projects = await self.notion.query_database(
                database="projects",
                limit=max_memories,
            )
            payload.active_projects = projects

        # Recent high-significance
        if boot_cfg.get("include_recent_high_significance", True):
            recent = await self.notion.query_all_active(
                min_significance=threshold,
                limit=max_memories,
            )
            payload.recent_high_significance = recent

        # Stats
        payload.stats = self.cache.get_stats()

        return payload.model_dump(exclude={"standing_instructions": {"__all__": {"embedding"}},
                                           "active_projects": {"__all__": {"embedding"}},
                                           "recent_high_significance": {"__all__": {"embedding"}}})


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Entry point for running the MCP server."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Dispatch Notion Memory MCP server starting...")
    # MCP server setup will go here once we wire up the transport layer
    # For now, this validates the service can be instantiated
    try:
        service = MemoryService()
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise


if __name__ == "__main__":
    main()
