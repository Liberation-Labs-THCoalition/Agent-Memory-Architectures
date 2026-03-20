"""Entity extraction via spaCy NER.

Extracts named entities from memory content and maps spaCy labels
to the EntityType enum. Entities are stored within each memory record,
ready to be promoted to knowledge graph nodes in Phase 3.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..models import Entity, EntityType

logger = logging.getLogger(__name__)

# Map spaCy NER labels to our EntityType enum
SPACY_TO_ENTITY_TYPE = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORG,
    "GPE": EntityType.LOCATION,      # Geopolitical entity → Location
    "LOC": EntityType.LOCATION,
    "FAC": EntityType.LOCATION,       # Facility → Location
    "EVENT": EntityType.EVENT,
    "DATE": EntityType.DATE,
    "TIME": EntityType.DATE,
    "PRODUCT": EntityType.TOOL,
    "WORK_OF_ART": EntityType.CONCEPT,
    "NORP": EntityType.CONCEPT,       # Nationalities, religious groups
    "LAW": EntityType.CONCEPT,
}


class EntityExtractor:
    """Extract named entities from text using spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Load the spaCy model.

        Args:
            model_name: spaCy model to use. en_core_web_sm is ~12MB
                        and handles basic NER well. Use en_core_web_md (~40MB)
                        for better accuracy.
        """
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            )
            self.nlp = None

    def extract(self, text: str, context_window: int = 50) -> list[Entity]:
        """Extract entities from text.

        Args:
            text: The memory content to extract entities from.
            context_window: Number of characters around the entity to capture
                            as mention_context.

        Returns:
            Deduplicated list of Entity objects.
        """
        if not self.nlp:
            return []

        doc = self.nlp(text)
        seen: dict[str, Entity] = {}

        for ent in doc.ents:
            # Skip very short or numeric-only entities
            if len(ent.text.strip()) < 2:
                continue
            if ent.text.strip().replace(".", "").replace(",", "").isdigit():
                continue

            entity_type = SPACY_TO_ENTITY_TYPE.get(ent.label_, EntityType.UNKNOWN)
            name = ent.text.strip()
            name_lower = name.lower()

            # Extract surrounding context
            start = max(0, ent.start_char - context_window)
            end = min(len(text), ent.end_char + context_window)
            mention_context = text[start:end].strip()

            # Dedup by lowercase name, keeping the most specific type
            if name_lower in seen:
                existing = seen[name_lower]
                if existing.entity_type == EntityType.UNKNOWN and entity_type != EntityType.UNKNOWN:
                    existing.entity_type = entity_type
                # Extend context if different mention
                if mention_context not in existing.mention_context:
                    existing.mention_context = (
                        existing.mention_context + " | " + mention_context
                    )[:500]
            else:
                seen[name_lower] = Entity(
                    name=name,
                    name_lower=name_lower,
                    entity_type=entity_type,
                    mention_context=mention_context,
                )

        return list(seen.values())

    def extract_with_custom_entities(
        self,
        text: str,
        custom_entities: Optional[list[tuple[str, str]]] = None,
    ) -> list[Entity]:
        """Extract entities with additional custom patterns.

        Useful for domain-specific entities that spaCy doesn't catch
        (project names, internal tools, etc.).

        Args:
            text: Memory content.
            custom_entities: List of (name, entity_type) tuples to look for.
        """
        entities = self.extract(text)

        if custom_entities:
            text_lower = text.lower()
            for name, etype in custom_entities:
                if name.lower() in text_lower and not any(
                    e.name_lower == name.lower() for e in entities
                ):
                    try:
                        entity_type = EntityType(etype)
                    except ValueError:
                        entity_type = EntityType.UNKNOWN

                    idx = text_lower.index(name.lower())
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(name) + 50)

                    entities.append(Entity(
                        name=name,
                        name_lower=name.lower(),
                        entity_type=entity_type,
                        mention_context=text[start:end].strip(),
                    ))

        return entities
