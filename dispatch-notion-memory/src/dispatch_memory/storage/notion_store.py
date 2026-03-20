"""Notion storage backend — primary store and human interface.

Maps PARA databases to memory lifecycle:
  Inbox     → new unprocessed memories
  Projects  → active project context
  Resources → reference material, facts
  Archive   → decayed/compressed memories
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from notion_client import AsyncClient

from ..models import (
    Entity, EntityType, Memory, MemoryStatus, MemoryType, TTLClass,
)

logger = logging.getLogger(__name__)

# Mapping from Notion Category to MemoryType
CATEGORY_TO_TYPE = {
    "Project": MemoryType.PROJECT,
    "Area": MemoryType.WORKFLOW,
    "Resource": MemoryType.FACT,
    "Archive": MemoryType.FACT,
    "Inbox": MemoryType.FACT,
}

# Mapping from MemoryType to target PARA database
TYPE_TO_DATABASE = {
    MemoryType.PROJECT: "projects",
    MemoryType.WORKFLOW: "projects",  # Areas live in projects db for now
    MemoryType.STANDING_INSTRUCTION: "resources",
    MemoryType.PREFERENCE: "resources",
    MemoryType.FACT: "resources",
    MemoryType.PERSON: "resources",
    MemoryType.DECISION: "projects",
    MemoryType.INTERACTION: "projects",
}


class NotionStore:
    """Read/write memories to Notion PARA databases."""

    def __init__(self, token: str, database_ids: dict[str, str]):
        """
        Args:
            token: Notion integration token
            database_ids: Mapping of PARA names to database IDs
                          e.g. {"inbox": "abc123", "projects": "def456", ...}
        """
        self.client = AsyncClient(auth=token)
        self.db_ids = database_ids

    async def store(self, memory: Memory) -> Memory:
        """Create a new memory as a Notion page in the appropriate database."""
        target_db = self._resolve_database(memory)
        db_id = self.db_ids.get(target_db)
        if not db_id:
            raise ValueError(f"No database configured for '{target_db}'")

        properties = self._memory_to_properties(memory)

        page = await self.client.pages.create(
            parent={"database_id": db_id},
            properties=properties,
        )

        memory.notion_page_id = page["id"]
        memory.notion_database = target_db
        logger.info(f"Stored memory {memory.id} in Notion/{target_db} as {page['id']}")
        return memory

    async def retrieve(self, notion_page_id: str) -> Optional[Memory]:
        """Fetch a single memory by its Notion page ID."""
        try:
            page = await self.client.pages.retrieve(page_id=notion_page_id)
            return self._page_to_memory(page)
        except Exception as e:
            logger.error(f"Failed to retrieve page {notion_page_id}: {e}")
            return None

    async def query_database(
        self,
        database: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        min_significance: Optional[float] = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Query a PARA database with optional filters."""
        db_id = self.db_ids.get(database)
        if not db_id:
            return []

        filter_conditions = []

        if memory_type and memory_type.value:
            filter_conditions.append({
                "property": "Category",
                "select": {"equals": self._type_to_category(memory_type)},
            })

        if tags:
            for tag in tags:
                filter_conditions.append({
                    "property": "Tags",
                    "multi_select": {"contains": tag},
                })

        if min_significance is not None:
            filter_conditions.append({
                "property": "Significance",
                "number": {"greater_than_or_equal_to": min_significance},
            })

        query_params = {"database_id": db_id, "page_size": min(limit, 100)}
        if filter_conditions:
            if len(filter_conditions) == 1:
                query_params["filter"] = filter_conditions[0]
            else:
                query_params["filter"] = {"and": filter_conditions}

        results = await self.client.databases.query(**query_params)
        return [self._page_to_memory(page) for page in results["results"]]

    async def query_all_active(
        self,
        min_significance: Optional[float] = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Query all active PARA databases (not archive)."""
        memories = []
        for db_name in ["inbox", "projects", "resources"]:
            if db_name in self.db_ids:
                batch = await self.query_database(
                    db_name,
                    min_significance=min_significance,
                    limit=limit,
                )
                memories.extend(batch)
        return memories[:limit]

    async def update(self, memory: Memory) -> Memory:
        """Update a memory's Notion page properties."""
        if not memory.notion_page_id:
            raise ValueError("Memory has no Notion page ID — cannot update")

        memory.touch()
        properties = self._memory_to_properties(memory)

        await self.client.pages.update(
            page_id=memory.notion_page_id,
            properties=properties,
        )
        logger.info(f"Updated memory {memory.id} in Notion")
        return memory

    async def archive(self, memory: Memory) -> Memory:
        """Move a memory to the Archive database.

        Creates a new page in Archive with Original Category preserved,
        then archives the original page.
        """
        archive_id = self.db_ids.get("archive")
        if not archive_id:
            raise ValueError("No archive database configured")

        memory.status = MemoryStatus.ARCHIVED
        properties = self._memory_to_properties(memory)

        # Add archive-specific fields
        properties["Archived Date"] = {
            "date": {"start": datetime.utcnow().isoformat()},
        }
        if memory.notion_database:
            properties["Original Category"] = {
                "select": {"name": memory.notion_database.title()},
            }

        # Create in archive
        page = await self.client.pages.create(
            parent={"database_id": archive_id},
            properties=properties,
        )

        # Archive the original
        if memory.notion_page_id:
            await self.client.pages.update(
                page_id=memory.notion_page_id,
                archived=True,
            )

        memory.notion_page_id = page["id"]
        memory.notion_database = "archive"
        logger.info(f"Archived memory {memory.id}")
        return memory

    async def delete(self, memory: Memory) -> None:
        """Soft-delete by archiving the Notion page."""
        if memory.notion_page_id:
            await self.client.pages.update(
                page_id=memory.notion_page_id,
                archived=True,
            )
            logger.info(f"Deleted memory {memory.id} from Notion")

    async def get_stats(self) -> dict:
        """Get memory counts across all databases."""
        stats = {}
        for db_name, db_id in self.db_ids.items():
            try:
                results = await self.client.databases.query(
                    database_id=db_id, page_size=1
                )
                # Notion doesn't give total count easily; we'd need to paginate
                # For now, just indicate the database exists
                stats[db_name] = {"configured": True}
            except Exception as e:
                stats[db_name] = {"configured": False, "error": str(e)}
        return stats

    def _resolve_database(self, memory: Memory) -> str:
        """Determine which PARA database a memory should live in."""
        if memory.status == MemoryStatus.ARCHIVED:
            return "archive"
        return TYPE_TO_DATABASE.get(memory.memory_type, "resources")

    def _type_to_category(self, memory_type: MemoryType) -> str:
        """Map MemoryType back to Notion Category select value."""
        reverse = {v: k for k, v in CATEGORY_TO_TYPE.items()}
        return reverse.get(memory_type, "Resource")

    def _memory_to_properties(self, memory: Memory) -> dict:
        """Convert a Memory model to Notion page properties."""
        properties = {
            "Name": {"title": [{"text": {"content": memory.content[:100]}}]},
            "Content": {"rich_text": [{"text": {"content": memory.content}}]},
            "Tags": {
                "multi_select": [{"name": tag} for tag in memory.tags[:10]],
            },
            "Confidence": {"number": memory.quality_score},
        }

        # Add Significance if the database supports it
        properties["Significance"] = {"number": memory.significance}

        # Add Entities as multi-select if present
        if memory.entities:
            properties["Entities"] = {
                "multi_select": [
                    {"name": e.name[:100]} for e in memory.entities[:10]
                ],
            }

        # Source
        properties["Source"] = {"select": {"name": "Text Command"}}

        # Captured At
        properties["Captured At"] = {
            "date": {"start": memory.created_at.isoformat()},
        }

        return properties

    def _page_to_memory(self, page: dict) -> Memory:
        """Convert a Notion page to a Memory model."""
        props = page.get("properties", {})

        # Extract content
        content = ""
        content_prop = props.get("Content", {})
        if content_prop.get("rich_text"):
            content = content_prop["rich_text"][0].get("text", {}).get("content", "")

        # Extract title as fallback
        if not content:
            name_prop = props.get("Name", {})
            if name_prop.get("title"):
                content = name_prop["title"][0].get("text", {}).get("content", "")

        # Extract tags
        tags = []
        tags_prop = props.get("Tags", {})
        if tags_prop.get("multi_select"):
            tags = [t["name"] for t in tags_prop["multi_select"]]

        # Extract entities
        entities = []
        entities_prop = props.get("Entities", {})
        if entities_prop.get("multi_select"):
            entities = [
                Entity(name=e["name"], entity_type=EntityType.UNKNOWN)
                for e in entities_prop["multi_select"]
            ]

        # Extract significance
        significance = 0.5
        sig_prop = props.get("Significance", {})
        if sig_prop.get("number") is not None:
            significance = sig_prop["number"]

        # Extract quality/confidence
        quality_score = 0.0
        conf_prop = props.get("Confidence", {})
        if conf_prop.get("number") is not None:
            quality_score = conf_prop["number"]

        # Extract category → memory_type
        memory_type = MemoryType.FACT
        cat_prop = props.get("Category", {})
        if cat_prop.get("select"):
            cat_name = cat_prop["select"].get("name", "")
            memory_type = CATEGORY_TO_TYPE.get(cat_name, MemoryType.FACT)

        # Timestamps
        created_at = datetime.utcnow()
        cap_prop = props.get("Captured At", {})
        if cap_prop.get("date") and cap_prop["date"].get("start"):
            try:
                created_at = datetime.fromisoformat(
                    cap_prop["date"]["start"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return Memory(
            content=content,
            memory_type=memory_type,
            tags=tags,
            entities=entities,
            significance=significance,
            quality_score=quality_score,
            created_at=created_at,
            notion_page_id=page["id"],
        )
