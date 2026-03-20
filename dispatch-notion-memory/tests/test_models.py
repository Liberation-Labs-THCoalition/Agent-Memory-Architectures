"""Tests for memory data models."""

import pytest
from dispatch_memory.models import (
    Entity, EntityType, Memory, MemoryStatus, MemoryType, TTLClass,
)


def test_memory_defaults():
    mem = Memory(content="Test memory")
    assert mem.memory_type == MemoryType.FACT
    assert mem.ttl_class == TTLClass.MEDIUM
    assert mem.status == MemoryStatus.ACTIVE
    assert mem.significance == 0.5
    assert mem.access_count == 0
    assert mem.content_hash != ""
    assert len(mem.id) > 0


def test_memory_content_hash():
    mem1 = Memory(content="Hello world")
    mem2 = Memory(content="Hello world")
    mem3 = Memory(content="Different content")
    assert mem1.content_hash == mem2.content_hash
    assert mem1.content_hash != mem3.content_hash


def test_memory_refresh():
    mem = Memory(content="Test")
    original_count = mem.access_count
    mem.refresh()
    assert mem.access_count == original_count + 1


def test_entity_name_lower():
    entity = Entity(name="Thomas Jefferson", entity_type=EntityType.PERSON)
    assert entity.name_lower == "thomas jefferson"


def test_entity_names_computed():
    mem = Memory(
        content="Test",
        entities=[
            Entity(name="Alice", entity_type=EntityType.PERSON),
            Entity(name="Acme Corp", entity_type=EntityType.ORG),
        ],
    )
    assert mem.entity_names == ["Alice", "Acme Corp"]


def test_memory_significance_bounds():
    mem = Memory(content="Test", significance=0.0)
    assert mem.significance == 0.0

    mem = Memory(content="Test", significance=1.0)
    assert mem.significance == 1.0

    with pytest.raises(Exception):
        Memory(content="Test", significance=1.5)
