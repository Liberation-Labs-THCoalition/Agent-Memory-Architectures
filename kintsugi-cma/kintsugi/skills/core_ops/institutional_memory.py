"""
Institutional Memory Skill Chip for Kintsugi CMA.

This chip queries and maintains organizational knowledge using the Cognitive
Memory Architecture (CMA) and temporal memory systems. It serves as the
organization's collective memory, preserving decisions, policies, and
historical context.

Key capabilities:
- Search organizational memory using semantic queries
- Retrieve decision history with context
- Look up policies and procedures
- Identify knowledge gaps
- Maintain temporal context for decisions

Example:
    chip = InstitutionalMemoryChip()
    request = SkillRequest(
        intent="knowledge_search",
        entities={"query": "volunteer screening policy"}
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from kintsugi.skills import (
    BaseSkillChip,
    EFEWeights,
    SkillCapability,
    SkillContext,
    SkillDomain,
    SkillRequest,
    SkillResponse,
    register_chip,
)


class MemoryType(str, Enum):
    """Types of organizational memory."""
    DECISION = "decision"
    POLICY = "policy"
    PROCEDURE = "procedure"
    KNOWLEDGE = "knowledge"
    HISTORY = "history"
    CONTACT = "contact"
    LESSON_LEARNED = "lesson_learned"


class MemoryStatus(str, Enum):
    """Status of a memory record."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DRAFT = "draft"


@dataclass
class MemoryRecord:
    """Represents a record in institutional memory.

    Attributes:
        id: Unique identifier
        memory_type: Type of memory record
        title: Record title
        content: Full content
        summary: Brief summary
        created_at: When record was created
        updated_at: Last update time
        created_by: Who created the record
        tags: Searchable tags
        status: Current status
        superseded_by: ID of record that supersedes this one
        context: Decision context or background
        related_records: IDs of related records
    """
    id: str
    memory_type: MemoryType
    title: str
    content: str
    summary: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    created_by: str = ""
    tags: list[str] = field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE
    superseded_by: str | None = None
    context: str = ""
    related_records: list[str] = field(default_factory=list)
    embedding_vector: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "tags": self.tags,
            "status": self.status.value,
            "superseded_by": self.superseded_by,
            "context": self.context,
            "related_records": self.related_records,
        }


@dataclass
class SearchResult:
    """Result from a memory search.

    Attributes:
        record: The matched memory record
        relevance_score: Semantic similarity score
        matched_terms: Terms that matched the query
        snippet: Relevant text snippet
    """
    record: MemoryRecord
    relevance_score: float
    matched_terms: list[str] = field(default_factory=list)
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "record": self.record.to_dict(),
            "relevance_score": self.relevance_score,
            "matched_terms": self.matched_terms,
            "snippet": self.snippet,
        }


class InstitutionalMemoryChip(BaseSkillChip):
    """Query and maintain organizational knowledge using CMA and temporal memory.

    This chip serves as the organization's collective memory, enabling
    staff to search for historical decisions, policies, procedures,
    and organizational knowledge. It uses semantic search and maintains
    temporal context for all records.

    Intents handled:
        - knowledge_search: Search memory using natural language
        - history_query: Query historical decisions or events
        - policy_lookup: Find specific policies or procedures
        - decision_context: Get context for past decisions
        - gap_identify: Identify knowledge gaps

    Consensus actions:
        - archive_record: Requires approval to archive important records
        - delete_memory: Requires approval to delete any record

    Example:
        chip = InstitutionalMemoryChip()
        request = SkillRequest(
            intent="policy_lookup",
            entities={"policy_name": "expense reimbursement"}
        )
        response = await chip.handle(request, context)
    """

    name = "institutional_memory"
    description = "Query and maintain organizational knowledge using CMA and temporal memory"
    version = "1.0.0"
    domain = SkillDomain.OPERATIONS

    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.20,
        resource_efficiency=0.20,
        transparency=0.20,
        equity=0.15,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
    ]

    consensus_actions = ["archive_record", "delete_memory"]
    required_spans = ["cma_query", "temporal_log", "embedding_search"]

    SUPPORTED_INTENTS = {
        "knowledge_search": "_handle_knowledge_search",
        "history_query": "_handle_history_query",
        "policy_lookup": "_handle_policy_lookup",
        "decision_context": "_handle_decision_context",
        "gap_identify": "_handle_gap_identify",
    }

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context with org, user, BDI state

        Returns:
            SkillResponse with memory data or search results
        """
        handler_name = self.SUPPORTED_INTENTS.get(request.intent)

        if handler_name is None:
            return SkillResponse(
                content=f"Unknown intent '{request.intent}' for institutional_memory chip.",
                success=False,
                data={"supported_intents": list(self.SUPPORTED_INTENTS.keys())},
            )

        handler = getattr(self, handler_name)
        return await handler(request, context)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract operations-relevant BDI context.

        Filters BDI state for beliefs about organizational knowledge,
        documentation status, and information needs.
        """
        ops_types = {"knowledge_state", "documentation_status", "policy_current", "training_status"}

        filtered_beliefs = [
            b for b in beliefs
            if b.get("type") in ops_types or b.get("domain") == "operations"
        ]

        filtered_desires = [
            d for d in desires
            if d.get("type") in {"documentation_goal", "knowledge_sharing", "training_completion"}
        ]

        return {
            "beliefs": filtered_beliefs,
            "desires": filtered_desires,
            "intentions": intentions,
        }

    async def _handle_knowledge_search(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Search organizational memory using natural language.

        Supported entities:
            - query: Natural language search query
            - memory_types: Filter by types (decision, policy, etc.)
            - date_range: Filter by date range
            - tags: Filter by tags
            - limit: Maximum results to return
        """
        entities = request.entities
        query = entities.get("query", "")

        if not query:
            return SkillResponse(
                content="Please provide a search query.",
                success=False,
            )

        results = await self.search_memory(
            query=query,
            org_id=context.org_id,
            memory_types=entities.get("memory_types"),
            date_range=entities.get("date_range"),
            tags=entities.get("tags"),
            limit=entities.get("limit", 10),
        )

        if not results:
            return SkillResponse(
                content=f"No results found for '{query}'.",
                success=True,
                data={"results": [], "query": query},
                suggestions=[
                    "Try different keywords",
                    "Broaden your search",
                    "Check for related topics",
                ],
            )

        content_lines = [f"Found {len(results)} results for '{query}':\n"]
        for i, r in enumerate(results[:5], 1):
            score_pct = r.relevance_score * 100
            content_lines.append(
                f"{i}. **{r.record.title}** ({r.record.memory_type.value}) - {score_pct:.0f}% match\n"
                f"   {r.snippet or r.record.summary[:100]}..."
            )

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data={"results": [r.to_dict() for r in results], "query": query},
            suggestions=["View full record?", "Find related records?"],
        )

    async def _handle_history_query(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Query historical decisions or events.

        Supported entities:
            - topic: Topic or subject to query
            - date_range: Historical period
            - include_context: Include decision context
            - chronological: Sort chronologically
        """
        entities = request.entities
        topic = entities.get("topic", "")
        date_range = entities.get("date_range", "all")
        include_context = entities.get("include_context", True)

        history = await self.get_decision_history(
            topic=topic,
            org_id=context.org_id,
            date_range=date_range,
        )

        if not history:
            return SkillResponse(
                content=f"No historical records found for '{topic}'.",
                success=True,
                data={"history": []},
            )

        content_lines = [f"Decision History: {topic}\n"]
        for record in history:
            date_str = record.created_at.strftime("%Y-%m-%d")
            content_lines.append(
                f"**{date_str}** - {record.title}\n"
                f"   {record.summary}"
            )
            if include_context and record.context:
                content_lines.append(f"   Context: {record.context[:100]}...")

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data={"history": [r.to_dict() for r in history]},
        )

    async def _handle_policy_lookup(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Look up specific policies or procedures.

        Supported entities:
            - policy_name: Name or topic of policy
            - include_procedures: Include related procedures
            - current_only: Only return current (non-superseded) policies
        """
        entities = request.entities
        policy_name = entities.get("policy_name", "")
        include_procedures = entities.get("include_procedures", True)
        current_only = entities.get("current_only", True)

        if not policy_name:
            return SkillResponse(
                content="Please specify a policy name or topic to look up.",
                success=False,
            )

        policy = await self.find_policy(
            policy_name=policy_name,
            org_id=context.org_id,
            current_only=current_only,
        )

        if not policy:
            return SkillResponse(
                content=f"No policy found for '{policy_name}'.",
                success=True,
                data={"policy": None},
                suggestions=["Search broader terms?", "View all policies?"],
            )

        content = f"""**{policy.title}**
Status: {policy.status.value.upper()}
Last Updated: {policy.updated_at.strftime('%Y-%m-%d') if policy.updated_at else policy.created_at.strftime('%Y-%m-%d')}

{policy.content}
"""

        if policy.related_records:
            content += f"\nRelated Records: {len(policy.related_records)}"

        return SkillResponse(
            content=content,
            success=True,
            data={"policy": policy.to_dict()},
            suggestions=["View related procedures?", "See version history?"],
        )

    async def _handle_decision_context(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Get context for past decisions.

        Supported entities:
            - decision_id: Specific decision ID
            - decision_topic: Topic of decision
            - include_alternatives: Include alternatives considered
            - include_outcomes: Include known outcomes
        """
        entities = request.entities
        decision_id = entities.get("decision_id")
        decision_topic = entities.get("decision_topic")

        if not decision_id and not decision_topic:
            return SkillResponse(
                content="Please provide a decision ID or topic.",
                success=False,
            )

        # Find the decision record
        if decision_id:
            record = await self._get_record_by_id(decision_id)
        else:
            # Search for decision by topic
            results = await self.search_memory(
                query=decision_topic,
                org_id=context.org_id,
                memory_types=[MemoryType.DECISION],
                limit=1,
            )
            record = results[0].record if results else None

        if not record:
            return SkillResponse(
                content="Decision not found.",
                success=False,
            )

        content = f"""**Decision Context: {record.title}**

**Date**: {record.created_at.strftime('%Y-%m-%d')}
**Decided By**: {record.created_by or 'Unknown'}

**Summary**
{record.summary}

**Context**
{record.context or 'No additional context recorded.'}

**Full Decision**
{record.content}
"""

        return SkillResponse(
            content=content,
            success=True,
            data={
                "decision": record.to_dict(),
                "related_records": record.related_records,
            },
            suggestions=["View related decisions?", "See how this affected later decisions?"],
        )

    async def _handle_gap_identify(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Identify knowledge gaps in organizational memory.

        Supported entities:
            - domain: Domain to analyze (operations, finance, etc.)
            - comparison_orgs: Compare against similar orgs (if available)
            - focus_areas: Specific areas to check
        """
        entities = request.entities
        domain = entities.get("domain", "all")
        focus_areas = entities.get("focus_areas", [])

        gaps = await self.identify_knowledge_gaps(
            org_id=context.org_id,
            domain=domain,
            focus_areas=focus_areas,
        )

        if not gaps:
            return SkillResponse(
                content="No significant knowledge gaps identified.",
                success=True,
                data={"gaps": []},
            )

        content_lines = ["Knowledge Gaps Identified:\n"]
        for gap in gaps:
            priority = gap.get("priority", "medium")
            content_lines.append(
                f"- **{gap['area']}** [{priority.upper()}]\n"
                f"  {gap['description']}\n"
                f"  Recommendation: {gap['recommendation']}"
            )

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data={"gaps": gaps},
            suggestions=["Create documentation for top gap?", "Schedule knowledge capture session?"],
        )

    # Core implementation methods

    async def search_memory(
        self,
        query: str,
        org_id: str,
        memory_types: list[MemoryType] | None = None,
        date_range: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search organizational memory using semantic search.

        Uses embedding-based search to find relevant records,
        with optional filtering by type, date, and tags.

        Args:
            query: Natural language search query
            org_id: Organization identifier
            memory_types: Filter by memory types
            date_range: Filter by date range
            tags: Filter by tags
            limit: Maximum results to return

        Returns:
            List of SearchResult objects ranked by relevance
        """
        # Get all records (in production, would use vector database)
        all_records = await self._get_all_records(org_id)

        # Filter by type if specified
        if memory_types:
            all_records = [r for r in all_records if r.memory_type in memory_types]

        # Filter by tags if specified
        if tags:
            all_records = [r for r in all_records if any(t in r.tags for t in tags)]

        # Compute relevance scores (simplified keyword matching)
        query_terms = query.lower().split()
        results = []

        for record in all_records:
            searchable_text = f"{record.title} {record.content} {record.summary}".lower()

            # Count matching terms
            matches = sum(1 for term in query_terms if term in searchable_text)
            if matches > 0:
                relevance = matches / len(query_terms)

                # Create snippet
                snippet = self._extract_snippet(record.content, query_terms)

                results.append(SearchResult(
                    record=record,
                    relevance_score=relevance,
                    matched_terms=[t for t in query_terms if t in searchable_text],
                    snippet=snippet,
                ))

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        return results[:limit]

    async def get_decision_history(
        self,
        topic: str,
        org_id: str,
        date_range: str = "all",
    ) -> list[MemoryRecord]:
        """Retrieve decision history for a topic.

        Args:
            topic: Topic or subject to query
            org_id: Organization identifier
            date_range: Historical period to search

        Returns:
            List of decision records in chronological order
        """
        all_records = await self._get_all_records(org_id)

        # Filter to decisions
        decisions = [r for r in all_records if r.memory_type == MemoryType.DECISION]

        # Filter by topic
        topic_lower = topic.lower()
        relevant = [
            d for d in decisions
            if topic_lower in d.title.lower() or topic_lower in d.content.lower()
        ]

        # Sort chronologically
        relevant.sort(key=lambda r: r.created_at)

        return relevant

    async def find_policy(
        self,
        policy_name: str,
        org_id: str,
        current_only: bool = True,
    ) -> MemoryRecord | None:
        """Find a specific policy or procedure.

        Args:
            policy_name: Name or topic of policy
            org_id: Organization identifier
            current_only: Only return non-superseded policies

        Returns:
            Policy record if found, None otherwise
        """
        all_records = await self._get_all_records(org_id)

        # Filter to policies and procedures
        policies = [
            r for r in all_records
            if r.memory_type in (MemoryType.POLICY, MemoryType.PROCEDURE)
        ]

        # Filter by status if current_only
        if current_only:
            policies = [p for p in policies if p.status == MemoryStatus.ACTIVE]

        # Search by name
        policy_lower = policy_name.lower()
        for p in policies:
            if policy_lower in p.title.lower():
                return p

        # Try content search
        for p in policies:
            if policy_lower in p.content.lower():
                return p

        return None

    async def identify_knowledge_gaps(
        self,
        org_id: str,
        domain: str = "all",
        focus_areas: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Identify areas where organizational knowledge is lacking.

        Args:
            org_id: Organization identifier
            domain: Domain to analyze
            focus_areas: Specific areas to check

        Returns:
            List of identified gaps with recommendations
        """
        # Standard areas that should be documented
        required_documentation = {
            "operations": [
                "volunteer_screening",
                "emergency_procedures",
                "data_privacy",
                "complaints_handling",
            ],
            "finance": [
                "expense_reimbursement",
                "procurement",
                "financial_controls",
                "audit_procedures",
            ],
            "governance": [
                "board_policies",
                "conflict_of_interest",
                "executive_compensation",
                "whistleblower",
            ],
            "programs": [
                "program_evaluation",
                "participant_intake",
                "case_management",
                "outcomes_measurement",
            ],
        }

        all_records = await self._get_all_records(org_id)
        documented_topics = set()
        for r in all_records:
            documented_topics.update(r.tags)
            documented_topics.add(r.title.lower().replace(" ", "_"))

        gaps = []
        areas_to_check = required_documentation if domain == "all" else {domain: required_documentation.get(domain, [])}

        for area_domain, required in areas_to_check.items():
            for topic in required:
                if topic not in documented_topics:
                    gaps.append({
                        "area": topic.replace("_", " ").title(),
                        "domain": area_domain,
                        "description": f"No documentation found for {topic.replace('_', ' ')}",
                        "priority": "high" if topic in ["emergency_procedures", "data_privacy", "financial_controls"] else "medium",
                        "recommendation": f"Create a {topic.replace('_', ' ')} policy document",
                    })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: priority_order.get(g["priority"], 1))

        return gaps

    # Private helper methods

    async def _get_all_records(self, org_id: str) -> list[MemoryRecord]:
        """Fetch all memory records for an organization."""
        # Simulated data
        now = datetime.now(timezone.utc)
        return [
            MemoryRecord(
                id="mem_001",
                memory_type=MemoryType.POLICY,
                title="Volunteer Screening Policy",
                content="All volunteers must undergo background checks before working with vulnerable populations...",
                summary="Policy requiring background checks for volunteers",
                created_at=now,
                tags=["volunteer", "screening", "background_check"],
                status=MemoryStatus.ACTIVE,
            ),
            MemoryRecord(
                id="mem_002",
                memory_type=MemoryType.DECISION,
                title="Decision: Expand Youth Program",
                content="The board decided to expand the youth tutoring program to three additional locations...",
                summary="Expansion of youth program to new locations",
                created_at=now,
                created_by="Board of Directors",
                tags=["youth_program", "expansion", "board_decision"],
                context="Driven by increased demand and successful outcomes at existing locations",
                status=MemoryStatus.ACTIVE,
            ),
            MemoryRecord(
                id="mem_003",
                memory_type=MemoryType.PROCEDURE,
                title="Expense Reimbursement Procedure",
                content="Employees may request reimbursement for approved business expenses by submitting receipts within 30 days...",
                summary="Steps for requesting expense reimbursement",
                created_at=now,
                tags=["expense", "reimbursement", "finance"],
                status=MemoryStatus.ACTIVE,
            ),
            MemoryRecord(
                id="mem_004",
                memory_type=MemoryType.LESSON_LEARNED,
                title="Virtual Event Lessons Learned",
                content="Our first virtual fundraising gala had technical difficulties. Key learnings: always have backup streaming...",
                summary="Lessons from first virtual fundraising event",
                created_at=now,
                tags=["virtual_event", "fundraising", "lessons_learned"],
                status=MemoryStatus.ACTIVE,
            ),
        ]

    async def _get_record_by_id(self, record_id: str) -> MemoryRecord | None:
        """Fetch a specific record by ID."""
        records = await self._get_all_records("")
        for r in records:
            if r.id == record_id:
                return r
        return None

    def _extract_snippet(self, content: str, query_terms: list[str]) -> str:
        """Extract a relevant snippet from content."""
        content_lower = content.lower()

        # Find first matching term position
        best_pos = len(content)
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < best_pos:
                best_pos = pos

        # Extract surrounding text
        start = max(0, best_pos - 50)
        end = min(len(content), best_pos + 150)

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet
