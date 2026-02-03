"""
Grant Hunter Skill Chip for Kintsugi CMA.

This chip searches and matches grant opportunities from Grants.gov, Candid,
and other foundation directories. It helps nonprofit staff discover funding
opportunities aligned with their organization's mission and programs.

Key capabilities:
- Search grants by criteria (focus area, amount, deadline, etc.)
- Check eligibility against organization profile
- Track and alert on upcoming deadlines
- Generate draft letters of intent (LOI)

Example:
    chip = GrantHunterChip()
    request = SkillRequest(
        intent="grant_search",
        entities={"focus_area": "youth education", "amount_min": 10000}
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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


@dataclass
class GrantOpportunity:
    """Represents a grant opportunity from any source.

    Attributes:
        id: Unique identifier for the grant
        title: Grant program title
        funder: Name of the funding organization
        amount_min: Minimum grant amount (if range)
        amount_max: Maximum grant amount (if range)
        deadline: Application deadline
        focus_areas: List of focus areas the grant supports
        eligibility: Eligibility requirements text
        source: Data source (grants_gov, candid, foundation_directory)
        url: Link to grant details
        match_score: Computed match score against org profile (0-100)
    """
    id: str
    title: str
    funder: str
    amount_min: int
    amount_max: int
    deadline: datetime | None
    focus_areas: list[str] = field(default_factory=list)
    eligibility: str = ""
    source: str = "unknown"
    url: str = ""
    match_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "title": self.title,
            "funder": self.funder,
            "amount_min": self.amount_min,
            "amount_max": self.amount_max,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "focus_areas": self.focus_areas,
            "eligibility": self.eligibility,
            "source": self.source,
            "url": self.url,
            "match_score": self.match_score,
        }


class GrantHunterChip(BaseSkillChip):
    """Search and match grant opportunities from Grants.gov, Candid, and other sources.

    This chip helps nonprofit organizations discover funding opportunities
    by searching multiple grant databases, checking eligibility requirements,
    tracking deadlines, and drafting initial application materials.

    Intents handled:
        - grant_search: Search for grants matching criteria
        - grant_match: Score grants against org profile
        - grant_deadline: Get upcoming deadlines
        - grant_eligibility: Check eligibility for specific grant
        - grant_report: Generate grant pipeline report

    Consensus actions:
        - submit_application: Requires approval before submitting
        - commit_match_funds: Requires approval for matching fund commitments

    Example:
        chip = GrantHunterChip()
        request = SkillRequest(
            intent="grant_search",
            entities={"focus_area": "education", "amount_min": 25000}
        )
        response = await chip.handle(request, context)
        # Returns matching grants ranked by relevance
    """

    name = "grant_hunter"
    description = "Search and match grant opportunities from Grants.gov, Candid, and other sources"
    version = "1.0.0"
    domain = SkillDomain.FUNDRAISING

    efe_weights = EFEWeights(
        mission_alignment=0.35,
        stakeholder_benefit=0.25,
        resource_efficiency=0.15,
        transparency=0.15,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.EXTERNAL_API,
        SkillCapability.GENERATE_REPORTS,
    ]

    consensus_actions = ["submit_application", "commit_match_funds"]
    required_spans = ["grants_gov_api", "candid_api", "foundation_directory"]

    # Intent routing map
    SUPPORTED_INTENTS = {
        "grant_search": "_handle_search",
        "grant_match": "_handle_match",
        "grant_deadline": "_handle_deadline",
        "grant_eligibility": "_handle_eligibility",
        "grant_report": "_handle_report",
    }

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context with org, user, BDI state

        Returns:
            SkillResponse with grant information or error message
        """
        handler_name = self.SUPPORTED_INTENTS.get(request.intent)

        if handler_name is None:
            return SkillResponse(
                content=f"Unknown intent '{request.intent}' for grant_hunter chip.",
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
        """Extract fundraising-relevant BDI context.

        Filters BDI state to return only beliefs about funding status,
        desires related to fundraising goals, and relevant intentions.
        """
        fundraising_types = {"funding_status", "budget", "grant_pipeline", "funder_relationship"}

        filtered_beliefs = [
            b for b in beliefs
            if b.get("type") in fundraising_types or b.get("domain") == "fundraising"
        ]

        filtered_desires = [
            d for d in desires
            if d.get("type") in {"funding_goal", "grant_target", "revenue_target"}
        ]

        return {
            "beliefs": filtered_beliefs,
            "desires": filtered_desires,
            "intentions": intentions,
        }

    async def _handle_search(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Search for grants matching specified criteria.

        Supported entities:
            - focus_area: Program focus area (e.g., "education", "health")
            - amount_min: Minimum grant amount
            - amount_max: Maximum grant amount
            - deadline_after: Only grants with deadlines after this date
            - funder_type: Type of funder (foundation, government, corporate)
            - geographic_focus: Geographic area served
        """
        entities = request.entities

        # Extract search parameters
        focus_area = entities.get("focus_area", "")
        amount_min = entities.get("amount_min", 0)
        amount_max = entities.get("amount_max", float("inf"))
        deadline_after = entities.get("deadline_after")
        funder_type = entities.get("funder_type")
        geo_focus = entities.get("geographic_focus")

        # Search grants from all sources
        grants = await self.search_grants(
            focus_area=focus_area,
            amount_min=amount_min,
            amount_max=amount_max,
            deadline_after=deadline_after,
            funder_type=funder_type,
            geographic_focus=geo_focus,
            org_id=context.org_id,
        )

        if not grants:
            return SkillResponse(
                content="No grants found matching your criteria. Try broadening your search.",
                success=True,
                data={"grants": [], "total": 0},
                suggestions=[
                    "Try removing some filters",
                    "Search for related focus areas",
                    "Check back weekly for new opportunities",
                ],
            )

        # Format response
        grant_summaries = []
        for g in grants[:10]:  # Top 10 results
            deadline_str = g.deadline.strftime("%Y-%m-%d") if g.deadline else "Rolling"
            grant_summaries.append(
                f"- **{g.title}** ({g.funder})\n"
                f"  Amount: ${g.amount_min:,}-${g.amount_max:,} | Deadline: {deadline_str}\n"
                f"  Match Score: {g.match_score:.0f}%"
            )

        content = f"Found {len(grants)} grants matching your criteria:\n\n" + "\n\n".join(grant_summaries)

        return SkillResponse(
            content=content,
            success=True,
            data={
                "grants": [g.to_dict() for g in grants],
                "total": len(grants),
                "search_criteria": entities,
            },
            suggestions=[
                "Would you like me to check eligibility for any of these?",
                "Should I generate a draft LOI for the top match?",
            ],
        )

    async def _handle_match(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Score grants against organization profile for best matches."""
        grant_ids = request.entities.get("grant_ids", [])

        if not grant_ids:
            # If no specific grants, get top matches from recent search
            grants = await self.search_grants(org_id=context.org_id)
        else:
            grants = await self._fetch_grants_by_ids(grant_ids)

        # Score each grant
        scored_grants = []
        for grant in grants:
            score = await self._compute_match_score(grant, context)
            grant.match_score = score
            scored_grants.append(grant)

        # Sort by match score
        scored_grants.sort(key=lambda g: g.match_score, reverse=True)

        top_matches = scored_grants[:5]
        content_lines = ["Top grant matches for your organization:\n"]
        for i, g in enumerate(top_matches, 1):
            content_lines.append(
                f"{i}. **{g.title}** - {g.match_score:.0f}% match\n"
                f"   Funder: {g.funder} | Amount: ${g.amount_min:,}-${g.amount_max:,}"
            )

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data={"matched_grants": [g.to_dict() for g in scored_grants]},
            suggestions=["View detailed eligibility for top match?"],
        )

    async def _handle_deadline(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Get upcoming grant deadlines."""
        days_ahead = request.entities.get("days_ahead", 30)
        tracked_only = request.entities.get("tracked_only", False)

        deadlines = await self.get_deadlines(
            org_id=context.org_id,
            days_ahead=days_ahead,
            tracked_only=tracked_only,
        )

        if not deadlines:
            return SkillResponse(
                content=f"No grant deadlines in the next {days_ahead} days.",
                success=True,
                data={"deadlines": []},
            )

        content_lines = [f"Upcoming grant deadlines (next {days_ahead} days):\n"]
        for grant, days_until in deadlines:
            urgency = "URGENT" if days_until <= 7 else ""
            content_lines.append(
                f"- {urgency} **{grant.title}** - {days_until} days ({grant.deadline.strftime('%Y-%m-%d')})"
            )

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data={
                "deadlines": [
                    {"grant": g.to_dict(), "days_until": d} for g, d in deadlines
                ]
            },
        )

    async def _handle_eligibility(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Check organization eligibility for a specific grant."""
        grant_id = request.entities.get("grant_id")

        if not grant_id:
            return SkillResponse(
                content="Please specify a grant ID to check eligibility.",
                success=False,
            )

        eligibility = await self.check_eligibility(grant_id, context.org_id)

        status = "ELIGIBLE" if eligibility["is_eligible"] else "NOT ELIGIBLE"
        content_lines = [f"Eligibility Status: **{status}**\n"]

        if eligibility["met_criteria"]:
            content_lines.append("Met criteria:")
            for c in eligibility["met_criteria"]:
                content_lines.append(f"  + {c}")

        if eligibility["unmet_criteria"]:
            content_lines.append("\nUnmet criteria:")
            for c in eligibility["unmet_criteria"]:
                content_lines.append(f"  - {c}")

        if eligibility["notes"]:
            content_lines.append(f"\nNotes: {eligibility['notes']}")

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data=eligibility,
            suggestions=["Generate draft LOI?"] if eligibility["is_eligible"] else [],
        )

    async def _handle_report(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Generate grant pipeline report."""
        report_type = request.entities.get("report_type", "summary")
        date_range = request.entities.get("date_range", "ytd")

        # Gather pipeline data
        pipeline = await self._get_pipeline_data(context.org_id, date_range)

        content = f"""Grant Pipeline Report ({date_range.upper()})

**Summary**
- Active Applications: {pipeline['active']}
- Pending Decisions: {pipeline['pending']}
- Won This Period: {pipeline['won']} (${pipeline['won_amount']:,})
- Lost This Period: {pipeline['lost']}
- Success Rate: {pipeline['success_rate']:.1f}%

**Upcoming Deadlines**
- Next 7 days: {pipeline['deadlines_7d']}
- Next 30 days: {pipeline['deadlines_30d']}

**Top Prospects**
{chr(10).join(f"- {p['title']} ({p['funder']}): ${p['amount']:,}" for p in pipeline['top_prospects'][:3])}
"""

        return SkillResponse(
            content=content,
            success=True,
            data=pipeline,
            suggestions=["Export full report to PDF?", "Set up weekly digest?"],
        )

    # Core implementation methods

    async def search_grants(
        self,
        focus_area: str = "",
        amount_min: int = 0,
        amount_max: float = float("inf"),
        deadline_after: str | None = None,
        funder_type: str | None = None,
        geographic_focus: str | None = None,
        org_id: str = "",
    ) -> list[GrantOpportunity]:
        """Search grants across all configured sources.

        This method queries Grants.gov, Candid, and foundation directories
        to find matching opportunities. Results are deduplicated and
        scored against the organization's profile.

        Args:
            focus_area: Primary focus area to search
            amount_min: Minimum grant amount
            amount_max: Maximum grant amount
            deadline_after: Only grants with deadlines after this date
            funder_type: Filter by funder type
            geographic_focus: Filter by geographic service area
            org_id: Organization ID for profile matching

        Returns:
            List of GrantOpportunity objects sorted by match score
        """
        all_grants: list[GrantOpportunity] = []

        # Query each data source (in production, these would be actual API calls)
        grants_gov_results = await self._query_grants_gov(
            focus_area, amount_min, amount_max, deadline_after
        )
        all_grants.extend(grants_gov_results)

        candid_results = await self._query_candid(
            focus_area, funder_type, geographic_focus
        )
        all_grants.extend(candid_results)

        foundation_results = await self._query_foundation_directory(
            focus_area, amount_min, amount_max
        )
        all_grants.extend(foundation_results)

        # Deduplicate by title/funder combination
        seen = set()
        unique_grants = []
        for g in all_grants:
            key = (g.title.lower(), g.funder.lower())
            if key not in seen:
                seen.add(key)
                unique_grants.append(g)

        # Filter by criteria
        filtered = []
        now = datetime.now(timezone.utc)
        deadline_cutoff = None
        if deadline_after:
            deadline_cutoff = datetime.fromisoformat(deadline_after)

        for g in unique_grants:
            if g.amount_max < amount_min:
                continue
            if g.amount_min > amount_max:
                continue
            if deadline_cutoff and g.deadline and g.deadline < deadline_cutoff:
                continue
            if g.deadline and g.deadline < now:
                continue  # Skip past deadlines
            filtered.append(g)

        # Score against org profile
        for g in filtered:
            g.match_score = await self._compute_match_score(g, org_id=org_id)

        # Sort by match score
        filtered.sort(key=lambda x: x.match_score, reverse=True)

        return filtered

    async def check_eligibility(
        self, grant_id: str, org_id: str
    ) -> dict[str, Any]:
        """Check organization eligibility for a specific grant.

        Args:
            grant_id: The grant identifier
            org_id: Organization identifier

        Returns:
            Dictionary with eligibility status and criteria details
        """
        # In production, this would fetch grant requirements and org profile
        # and perform detailed eligibility checking

        return {
            "grant_id": grant_id,
            "org_id": org_id,
            "is_eligible": True,
            "met_criteria": [
                "501(c)(3) status",
                "Operating budget within range",
                "Geographic service area match",
            ],
            "unmet_criteria": [],
            "notes": "Organization appears to meet all stated eligibility requirements.",
            "confidence": 0.85,
        }

    async def get_deadlines(
        self,
        org_id: str,
        days_ahead: int = 30,
        tracked_only: bool = False,
    ) -> list[tuple[GrantOpportunity, int]]:
        """Get upcoming grant deadlines.

        Args:
            org_id: Organization identifier
            days_ahead: Number of days to look ahead
            tracked_only: If True, only return tracked/saved grants

        Returns:
            List of (grant, days_until_deadline) tuples sorted by deadline
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)

        # Get grants with upcoming deadlines
        all_grants = await self.search_grants(org_id=org_id)

        upcoming = []
        for g in all_grants:
            if g.deadline and now <= g.deadline <= cutoff:
                days_until = (g.deadline - now).days
                upcoming.append((g, days_until))

        # Sort by deadline (soonest first)
        upcoming.sort(key=lambda x: x[1])

        return upcoming

    async def generate_loi_draft(
        self,
        grant_id: str,
        org_id: str,
        context: SkillContext,
    ) -> dict[str, Any]:
        """Generate a draft Letter of Intent for a grant.

        Args:
            grant_id: The grant identifier
            org_id: Organization identifier
            context: Skill context for BDI state

        Returns:
            Dictionary with LOI draft content and metadata
        """
        # Fetch grant and org details
        grants = await self._fetch_grants_by_ids([grant_id])
        if not grants:
            return {"error": "Grant not found", "success": False}

        grant = grants[0]

        # In production, this would use LLM to generate contextual LOI
        loi_template = f"""
LETTER OF INTENT

Date: {datetime.now(timezone.utc).strftime('%B %d, %Y')}

To: {grant.funder}
Re: {grant.title}

Dear Grants Committee,

[Organization Name] respectfully submits this Letter of Intent for the {grant.title} program.

**Organization Overview**
[Brief description of organization, mission, and track record]

**Proposed Project**
[Description of proposed project aligned with grant focus areas: {', '.join(grant.focus_areas)}]

**Request Amount**
We are requesting ${grant.amount_min:,} to ${grant.amount_max:,} to support this initiative.

**Expected Outcomes**
[Measurable outcomes and impact]

**Contact Information**
[Primary contact details]

We look forward to the opportunity to discuss this proposal further.

Respectfully,
[Executive Director Name]
[Organization Name]
"""

        return {
            "success": True,
            "grant_id": grant_id,
            "draft": loi_template.strip(),
            "word_count": len(loi_template.split()),
            "placeholders": [
                "[Organization Name]",
                "[Brief description of organization, mission, and track record]",
                "[Description of proposed project aligned with grant focus areas]",
                "[Measurable outcomes and impact]",
                "[Primary contact details]",
                "[Executive Director Name]",
            ],
        }

    # Private helper methods

    async def _query_grants_gov(
        self,
        focus_area: str,
        amount_min: int,
        amount_max: float,
        deadline_after: str | None,
    ) -> list[GrantOpportunity]:
        """Query Grants.gov API for federal grants."""
        # Simulated response - in production, would call actual API
        return [
            GrantOpportunity(
                id="grants_gov_001",
                title="Community Development Block Grant",
                funder="HUD",
                amount_min=50000,
                amount_max=500000,
                deadline=datetime.now(timezone.utc) + timedelta(days=45),
                focus_areas=["community development", "housing", "economic development"],
                source="grants_gov",
                url="https://grants.gov/example/001",
            ),
        ]

    async def _query_candid(
        self,
        focus_area: str,
        funder_type: str | None,
        geographic_focus: str | None,
    ) -> list[GrantOpportunity]:
        """Query Candid Foundation Directory."""
        return [
            GrantOpportunity(
                id="candid_001",
                title="Youth Education Initiative Grant",
                funder="Example Foundation",
                amount_min=10000,
                amount_max=50000,
                deadline=datetime.now(timezone.utc) + timedelta(days=60),
                focus_areas=["education", "youth development"],
                source="candid",
                url="https://candid.org/example/001",
            ),
        ]

    async def _query_foundation_directory(
        self,
        focus_area: str,
        amount_min: int,
        amount_max: float,
    ) -> list[GrantOpportunity]:
        """Query local foundation directory."""
        return [
            GrantOpportunity(
                id="fd_001",
                title="Local Community Support Grant",
                funder="Community Foundation",
                amount_min=5000,
                amount_max=25000,
                deadline=datetime.now(timezone.utc) + timedelta(days=30),
                focus_areas=["community", "general support"],
                source="foundation_directory",
                url="https://communityfoundation.org/grants",
            ),
        ]

    async def _fetch_grants_by_ids(
        self, grant_ids: list[str]
    ) -> list[GrantOpportunity]:
        """Fetch specific grants by their IDs."""
        # In production, would query data sources by ID
        all_grants = await self.search_grants()
        return [g for g in all_grants if g.id in grant_ids]

    async def _compute_match_score(
        self,
        grant: GrantOpportunity,
        context: SkillContext | None = None,
        org_id: str = "",
    ) -> float:
        """Compute match score between grant and organization.

        Considers:
        - Focus area alignment
        - Grant amount vs org budget
        - Geographic fit
        - Past success with funder
        - Timeline feasibility
        """
        score = 50.0  # Base score

        # Focus area matching (simplified)
        if grant.focus_areas:
            score += 20.0  # Would compare against org focus areas

        # Amount feasibility
        if 10000 <= grant.amount_min <= 100000:
            score += 15.0  # Typical nonprofit sweet spot

        # Deadline feasibility
        if grant.deadline:
            days_until = (grant.deadline - datetime.now(timezone.utc)).days
            if days_until > 30:
                score += 10.0  # Adequate time to apply
            elif days_until > 14:
                score += 5.0  # Tight but possible

        return min(score, 100.0)

    async def _get_pipeline_data(
        self, org_id: str, date_range: str
    ) -> dict[str, Any]:
        """Get grant pipeline statistics."""
        return {
            "active": 5,
            "pending": 3,
            "won": 2,
            "won_amount": 75000,
            "lost": 1,
            "success_rate": 66.7,
            "deadlines_7d": 1,
            "deadlines_30d": 4,
            "top_prospects": [
                {"title": "Education Grant", "funder": "Foundation A", "amount": 50000},
                {"title": "Health Initiative", "funder": "Foundation B", "amount": 30000},
                {"title": "Community Support", "funder": "Foundation C", "amount": 25000},
            ],
        }
