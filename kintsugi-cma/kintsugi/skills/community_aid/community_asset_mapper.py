"""
Community Asset Mapper Skill Chip for Kintsugi CMA.

Maps and inventories local community resources, skills, and assets to enable
effective community organizing and resource coordination.

This chip enables asset-based community development by:
- Cataloging physical assets (spaces, equipment, vehicles)
- Inventorying human assets (skills, expertise, time)
- Mapping organizational assets (services, programs)
- Identifying gaps and opportunities for community resilience

Example usage:
    from kintsugi.skills.community_aid import CommunityAssetMapperChip
    from kintsugi.skills import SkillRequest, SkillContext, register_chip

    # Register the chip
    chip = CommunityAssetMapperChip()
    register_chip(chip)

    # Add a community asset
    request = SkillRequest(
        intent="asset_add",
        entities={
            "asset_type": "space",
            "name": "Community Center Main Hall",
            "description": "Large hall with capacity for 200 people",
            "address": "123 Main St",
            "categories": ["meeting_space", "event_venue"],
            "availability": "weekdays 9am-5pm"
        }
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from kintsugi.skills import (
    BaseSkillChip,
    EFEWeights,
    SkillCapability,
    SkillContext,
    SkillDomain,
    SkillRequest,
    SkillResponse,
)


class AssetType(str, Enum):
    """Types of community assets."""
    SPACE = "space"
    EQUIPMENT = "equipment"
    VEHICLE = "vehicle"
    SKILL = "skill"
    SERVICE = "service"
    ORGANIZATION = "organization"
    PROGRAM = "program"
    FUNDING = "funding"
    NETWORK = "network"
    OTHER = "other"


class AssetCategory(str, Enum):
    """Categories for asset classification."""
    MEETING_SPACE = "meeting_space"
    EVENT_VENUE = "event_venue"
    KITCHEN = "kitchen"
    STORAGE = "storage"
    OFFICE = "office"
    WORKSHOP = "workshop"
    OUTDOOR = "outdoor"
    TRANSPORTATION = "transportation"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    LEGAL = "legal"
    CHILDCARE = "childcare"
    FOOD = "food"
    HOUSING = "housing"
    EMPLOYMENT = "employment"
    ARTS_CULTURE = "arts_culture"
    RECREATION = "recreation"
    OTHER = "other"


class AssetStatus(str, Enum):
    """Status of assets in the system."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNDER_REVIEW = "under_review"
    ARCHIVED = "archived"


@dataclass
class GeoLocation:
    """Geographic coordinates for mapping."""
    latitude: float
    longitude: float
    accuracy_meters: float = 100.0


@dataclass
class CommunityAsset:
    """Represents a community asset."""
    id: str
    asset_type: AssetType
    name: str
    description: str
    address: str
    geo_location: GeoLocation | None
    categories: list[AssetCategory]
    owner_id: str
    owner_name: str
    contact_info: dict[str, str]
    availability: str
    capacity: str | None
    access_requirements: list[str]
    created_at: datetime
    updated_at: datetime
    status: AssetStatus
    verified: bool = False
    usage_count: int = 0
    ratings: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillAsset:
    """Represents a skill/expertise asset from a community member."""
    id: str
    member_id: str
    member_name: str
    skills: list[str]
    expertise_level: str  # beginner, intermediate, expert
    categories: list[AssetCategory]
    availability_hours: int  # Hours per week
    can_teach: bool
    certifications: list[str]
    contact_info: dict[str, str]
    created_at: datetime
    status: AssetStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GapAnalysis:
    """Represents a gap analysis result."""
    id: str
    conducted_at: datetime
    area_analyzed: str
    gaps_identified: list[dict[str, Any]]
    recommendations: list[str]
    priority_ranking: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class CommunityAssetMapperChip(BaseSkillChip):
    """Map and inventory local community resources with geocoding.

    This chip supports asset-based community development through:
    1. Asset registration and categorization
    2. Geographic mapping with filtering
    3. Skill and expertise inventory
    4. Gap analysis to identify unmet needs
    5. Partner sharing with consent

    Key feature: Geocoded asset mapping enables location-based
    searches and visual map generation.
    """

    name = "community_asset_mapper"
    description = "Map and inventory local community resources, skills, and assets"
    version = "1.0.0"
    domain = SkillDomain.COMMUNITY

    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.25,
        resource_efficiency=0.20,
        transparency=0.15,
        equity=0.15,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.EXTERNAL_API,
    ]

    consensus_actions = ["publish_asset_map", "share_with_partner"]

    required_spans = [
        "geocoding_api",
        "asset_database",
        "community_survey",
    ]

    # Simulated storage
    _assets: dict[str, CommunityAsset] = {}
    _skills: dict[str, SkillAsset] = {}
    _gap_analyses: dict[str, GapAnalysis] = {}

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request containing intent and entities
            context: Execution context with org/user info and BDI state

        Returns:
            SkillResponse with operation result
        """
        intent_handlers = {
            "asset_add": self._handle_asset_add,
            "asset_search": self._handle_asset_search,
            "asset_map": self._handle_asset_map,
            "skill_inventory": self._handle_skill_inventory,
            "gap_analysis": self._handle_gap_analysis,
        }

        handler = intent_handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}",
                success=False,
                data={"error": "unknown_intent", "valid_intents": list(intent_handlers.keys())},
            )

        return await handler(request, context)

    async def _handle_asset_add(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle adding a new community asset.

        Entities expected:
            asset_type: Type of asset (space, equipment, etc.)
            name: Asset name
            description: Description
            address: Physical address
            categories: List of categories
            availability: Availability description
            capacity: Capacity info (optional)
            contact_email: Contact email
            contact_phone: Contact phone
        """
        asset = await self.add_asset(
            asset_type=request.entities.get("asset_type", "other"),
            name=request.entities.get("name", ""),
            description=request.entities.get("description", ""),
            address=request.entities.get("address", ""),
            categories=request.entities.get("categories", ["other"]),
            owner_id=context.user_id,
            owner_name=request.entities.get("owner_name", ""),
            contact_email=request.entities.get("contact_email", ""),
            contact_phone=request.entities.get("contact_phone", ""),
            availability=request.entities.get("availability", ""),
            capacity=request.entities.get("capacity"),
        )

        return SkillResponse(
            content=f"Asset '{asset.name}' added successfully (ID: {asset.id[:8]}...).\n"
                    f"Type: {asset.asset_type.value}\n"
                    f"Categories: {', '.join(c.value for c in asset.categories)}\n"
                    f"Status: {asset.status.value} (pending verification)",
            success=True,
            data={
                "asset_id": asset.id,
                "asset_type": asset.asset_type.value,
                "name": asset.name,
                "categories": [c.value for c in asset.categories],
                "geo_location": {
                    "lat": asset.geo_location.latitude,
                    "lng": asset.geo_location.longitude,
                } if asset.geo_location else None,
            },
            suggestions=[
                "Search for similar assets with 'search community assets'",
                "View asset map with 'show asset map'",
            ],
        )

    async def _handle_asset_search(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle searching for community assets.

        Entities expected:
            query: Search query text
            asset_type: Filter by type (optional)
            categories: Filter by categories (optional)
            location: Near this address (optional)
            radius_km: Search radius (default: 10)
        """
        results = await self.search_assets(
            query=request.entities.get("query", ""),
            asset_type=request.entities.get("asset_type"),
            categories=request.entities.get("categories"),
            location=request.entities.get("location"),
            radius_km=request.entities.get("radius_km", 10.0),
        )

        if not results:
            return SkillResponse(
                content="No assets found matching your criteria.",
                success=True,
                data={"assets": [], "count": 0},
                suggestions=[
                    "Try broadening your search",
                    "Add an asset with 'add community asset'",
                ],
            )

        summary_lines = [f"Found {len(results)} asset(s):"]
        for asset in results[:10]:  # Top 10
            summary_lines.append(
                f"\n- {asset.name} ({asset.asset_type.value})\n"
                f"  {asset.address}\n"
                f"  Categories: {', '.join(c.value for c in asset.categories[:3])}"
            )

        return SkillResponse(
            content="\n".join(summary_lines),
            success=True,
            data={
                "assets": [self._asset_to_dict(a) for a in results],
                "count": len(results),
            },
        )

    async def _handle_asset_map(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle generating asset map.

        Entities expected:
            center_address: Center of map
            radius_km: Map radius (default: 10)
            categories: Filter by categories (optional)
            format: Output format (geojson, summary, list)
            share_publicly: Whether to publish publicly (requires consensus)
        """
        if request.entities.get("share_publicly", False):
            return SkillResponse(
                content="Publishing asset map publicly requires approval.",
                success=True,
                requires_consensus=True,
                consensus_action="publish_asset_map",
                data={"pending_action": "publish_asset_map"},
            )

        map_data = await self.generate_map(
            center_address=request.entities.get("center_address"),
            radius_km=request.entities.get("radius_km", 10.0),
            categories=request.entities.get("categories"),
            output_format=request.entities.get("format", "summary"),
        )

        return SkillResponse(
            content=f"Asset map generated with {map_data['total_assets']} assets "
                    f"across {map_data['categories_count']} categories.",
            success=True,
            data=map_data,
            suggestions=[
                "Filter by category with 'show map for [category]'",
                "Share with partners with 'share asset map'",
            ],
        )

    async def _handle_skill_inventory(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle skill inventory operations.

        Entities expected:
            action: add, search, or list
            skills: List of skills (for add)
            expertise_level: Expertise level (for add)
            can_teach: Whether they can teach (for add)
            availability_hours: Hours per week (for add)
            search_skills: Skills to search for (for search)
        """
        action = request.entities.get("action", "list")

        if action == "add":
            skill_asset = await self.inventory_skills(
                member_id=context.user_id,
                member_name=request.entities.get("member_name", ""),
                skills=request.entities.get("skills", []),
                expertise_level=request.entities.get("expertise_level", "intermediate"),
                categories=request.entities.get("categories", ["other"]),
                availability_hours=request.entities.get("availability_hours", 5),
                can_teach=request.entities.get("can_teach", False),
                certifications=request.entities.get("certifications", []),
                contact_email=request.entities.get("contact_email", ""),
            )

            return SkillResponse(
                content=f"Skills registered: {', '.join(skill_asset.skills)}\n"
                        f"Expertise: {skill_asset.expertise_level}\n"
                        f"Available: {skill_asset.availability_hours} hours/week",
                success=True,
                data={"skill_asset_id": skill_asset.id, "skills": skill_asset.skills},
            )

        elif action == "search":
            search_skills = request.entities.get("search_skills", [])
            matching = [
                s for s in self._skills.values()
                if s.status == AssetStatus.ACTIVE
                and any(skill.lower() in [sk.lower() for sk in s.skills] for skill in search_skills)
            ]

            return SkillResponse(
                content=f"Found {len(matching)} community member(s) with those skills.",
                success=True,
                data={
                    "skills_found": [
                        {
                            "id": s.id,
                            "skills": s.skills,
                            "expertise": s.expertise_level,
                            "can_teach": s.can_teach,
                            "availability": f"{s.availability_hours} hrs/week",
                        }
                        for s in matching
                    ]
                },
            )

        else:  # list
            all_skills = {}
            for skill_asset in self._skills.values():
                if skill_asset.status == AssetStatus.ACTIVE:
                    for skill in skill_asset.skills:
                        if skill not in all_skills:
                            all_skills[skill] = {"count": 0, "can_teach": 0}
                        all_skills[skill]["count"] += 1
                        if skill_asset.can_teach:
                            all_skills[skill]["can_teach"] += 1

            return SkillResponse(
                content=f"Community has {len(all_skills)} unique skills across "
                        f"{len(self._skills)} members.",
                success=True,
                data={"skills_inventory": all_skills},
            )

    async def _handle_gap_analysis(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle gap analysis for community assets.

        Entities expected:
            area: Area/neighborhood to analyze
            categories: Categories to analyze (optional)
            compare_to: Reference area or standard (optional)
        """
        analysis = await self.identify_gaps(
            area=request.entities.get("area", "community-wide"),
            categories=request.entities.get("categories"),
            compare_to=request.entities.get("compare_to"),
        )

        summary_lines = [
            f"Gap Analysis for {analysis.area_analyzed}",
            f"Conducted: {analysis.conducted_at.strftime('%Y-%m-%d')}",
            "",
            "Gaps Identified:",
        ]

        for gap in analysis.gaps_identified[:5]:
            summary_lines.append(f"  - {gap['description']} (Priority: {gap['priority']})")

        summary_lines.append("\nTop Recommendations:")
        for rec in analysis.recommendations[:3]:
            summary_lines.append(f"  - {rec}")

        return SkillResponse(
            content="\n".join(summary_lines),
            success=True,
            data={
                "analysis_id": analysis.id,
                "gaps": analysis.gaps_identified,
                "recommendations": analysis.recommendations,
                "priority_ranking": analysis.priority_ranking,
            },
            suggestions=[
                "Address top gap with 'create asset recruitment plan'",
                "Share analysis with 'share gap analysis with partners'",
            ],
        )

    # Core business logic methods

    async def add_asset(
        self,
        asset_type: str,
        name: str,
        description: str,
        address: str,
        categories: list[str],
        owner_id: str,
        owner_name: str,
        contact_email: str = "",
        contact_phone: str = "",
        availability: str = "",
        capacity: str | None = None,
    ) -> CommunityAsset:
        """Add a new community asset.

        Args:
            asset_type: Type of asset
            name: Asset name
            description: Description
            address: Physical address
            categories: Asset categories
            owner_id: ID of asset owner
            owner_name: Name of owner
            contact_email: Contact email
            contact_phone: Contact phone
            availability: Availability description
            capacity: Capacity information

        Returns:
            The created CommunityAsset
        """
        now = datetime.now(timezone.utc)

        # Geocode address (simulated)
        geo = await self._geocode_address(address)

        asset = CommunityAsset(
            id=str(uuid4()),
            asset_type=AssetType(asset_type),
            name=name,
            description=description,
            address=address,
            geo_location=geo,
            categories=[AssetCategory(c) for c in categories],
            owner_id=owner_id,
            owner_name=owner_name,
            contact_info={"email": contact_email, "phone": contact_phone},
            availability=availability,
            capacity=capacity,
            access_requirements=[],
            created_at=now,
            updated_at=now,
            status=AssetStatus.UNDER_REVIEW,
        )

        self._assets[asset.id] = asset
        return asset

    async def search_assets(
        self,
        query: str = "",
        asset_type: str | None = None,
        categories: list[str] | None = None,
        location: str | None = None,
        radius_km: float = 10.0,
    ) -> list[CommunityAsset]:
        """Search for community assets.

        Args:
            query: Text search query
            asset_type: Filter by type
            categories: Filter by categories
            location: Center location for radius search
            radius_km: Search radius

        Returns:
            List of matching assets
        """
        results = []

        for asset in self._assets.values():
            if asset.status not in [AssetStatus.ACTIVE, AssetStatus.UNDER_REVIEW]:
                continue

            # Type filter
            if asset_type and asset.asset_type.value != asset_type:
                continue

            # Category filter
            if categories:
                asset_cats = {c.value for c in asset.categories}
                if not any(c in asset_cats for c in categories):
                    continue

            # Text search
            if query:
                query_lower = query.lower()
                searchable = f"{asset.name} {asset.description} {asset.address}".lower()
                if query_lower not in searchable:
                    continue

            # Location filter (simplified - would use real geodistance in production)
            if location and asset.geo_location:
                # Simulated - accept all for now
                pass

            results.append(asset)

        # Sort by usage/ratings
        results.sort(key=lambda a: (a.usage_count, sum(a.ratings) / len(a.ratings) if a.ratings else 0), reverse=True)

        return results

    async def generate_map(
        self,
        center_address: str | None = None,
        radius_km: float = 10.0,
        categories: list[str] | None = None,
        output_format: str = "summary",
    ) -> dict[str, Any]:
        """Generate asset map data.

        Args:
            center_address: Center of map
            radius_km: Map radius
            categories: Filter by categories
            output_format: Output format

        Returns:
            Map data dictionary
        """
        # Get relevant assets
        assets = await self.search_assets(
            categories=categories,
            location=center_address,
            radius_km=radius_km,
        )

        # Build category breakdown
        category_breakdown = {}
        for asset in assets:
            for cat in asset.categories:
                if cat.value not in category_breakdown:
                    category_breakdown[cat.value] = []
                category_breakdown[cat.value].append(asset.id)

        # Build GeoJSON features if requested
        geojson_features = []
        if output_format == "geojson":
            for asset in assets:
                if asset.geo_location:
                    geojson_features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                asset.geo_location.longitude,
                                asset.geo_location.latitude,
                            ],
                        },
                        "properties": {
                            "id": asset.id,
                            "name": asset.name,
                            "type": asset.asset_type.value,
                            "categories": [c.value for c in asset.categories],
                        },
                    })

        return {
            "total_assets": len(assets),
            "categories_count": len(category_breakdown),
            "category_breakdown": category_breakdown,
            "center": center_address,
            "radius_km": radius_km,
            "format": output_format,
            "geojson": {
                "type": "FeatureCollection",
                "features": geojson_features,
            } if output_format == "geojson" else None,
            "assets": [self._asset_to_dict(a) for a in assets] if output_format == "list" else None,
        }

    async def inventory_skills(
        self,
        member_id: str,
        member_name: str,
        skills: list[str],
        expertise_level: str,
        categories: list[str],
        availability_hours: int,
        can_teach: bool,
        certifications: list[str],
        contact_email: str = "",
    ) -> SkillAsset:
        """Inventory member skills.

        Args:
            member_id: Member ID
            member_name: Member name
            skills: List of skills
            expertise_level: Expertise level
            categories: Skill categories
            availability_hours: Hours per week
            can_teach: Whether they can teach
            certifications: Certifications held
            contact_email: Contact email

        Returns:
            The created SkillAsset
        """
        skill_asset = SkillAsset(
            id=str(uuid4()),
            member_id=member_id,
            member_name=member_name,
            skills=skills,
            expertise_level=expertise_level,
            categories=[AssetCategory(c) for c in categories],
            availability_hours=availability_hours,
            can_teach=can_teach,
            certifications=certifications,
            contact_info={"email": contact_email},
            created_at=datetime.now(timezone.utc),
            status=AssetStatus.ACTIVE,
        )

        self._skills[skill_asset.id] = skill_asset
        return skill_asset

    async def identify_gaps(
        self,
        area: str,
        categories: list[str] | None = None,
        compare_to: str | None = None,
    ) -> GapAnalysis:
        """Identify gaps in community assets.

        Args:
            area: Area to analyze
            categories: Categories to focus on
            compare_to: Reference for comparison

        Returns:
            GapAnalysis result
        """
        # Analyze current assets
        assets = list(self._assets.values())
        skills = list(self._skills.values())

        # Category coverage
        covered_categories = set()
        for asset in assets:
            for cat in asset.categories:
                covered_categories.add(cat.value)

        # Find gaps
        all_categories = [c.value for c in AssetCategory]
        missing_categories = [c for c in all_categories if c not in covered_categories]

        # Build gap analysis
        gaps = []
        for missing in missing_categories[:10]:  # Top 10 gaps
            gaps.append({
                "category": missing,
                "description": f"No {missing.replace('_', ' ')} assets in {area}",
                "priority": "high" if missing in ["healthcare", "food", "housing"] else "medium",
                "estimated_need": "unknown",
            })

        # Add skill gaps
        skill_categories = set()
        for skill_asset in skills:
            for cat in skill_asset.categories:
                skill_categories.add(cat.value)

        for cat in ["healthcare", "legal", "education"]:
            if cat not in skill_categories:
                gaps.append({
                    "category": cat,
                    "description": f"No community members with {cat} skills registered",
                    "priority": "high",
                    "type": "skill_gap",
                })

        # Generate recommendations
        recommendations = []
        for gap in gaps[:5]:
            recommendations.append(
                f"Recruit or identify {gap['category']} resources through community outreach"
            )

        analysis = GapAnalysis(
            id=str(uuid4()),
            conducted_at=datetime.now(timezone.utc),
            area_analyzed=area,
            gaps_identified=gaps,
            recommendations=recommendations,
            priority_ranking=[g["category"] for g in sorted(gaps, key=lambda x: x["priority"])],
        )

        self._gap_analyses[analysis.id] = analysis
        return analysis

    # Helper methods

    async def _geocode_address(self, address: str) -> GeoLocation | None:
        """Geocode an address (simulated).

        In production, this would call a real geocoding API.
        """
        if not address:
            return None

        # Simulated geocoding - return mock coordinates
        import hashlib
        # Use address hash for consistent pseudo-random coordinates
        addr_hash = int(hashlib.md5(address.encode()).hexdigest()[:8], 16)

        # Generate coordinates in reasonable range (US-ish)
        lat = 30.0 + (addr_hash % 1000) / 100.0  # 30-40
        lng = -120.0 + (addr_hash % 500) / 100.0  # -120 to -115

        return GeoLocation(latitude=lat, longitude=lng)

    def _asset_to_dict(self, asset: CommunityAsset) -> dict[str, Any]:
        """Convert asset to dictionary."""
        return {
            "id": asset.id,
            "type": asset.asset_type.value,
            "name": asset.name,
            "description": asset.description,
            "address": asset.address,
            "categories": [c.value for c in asset.categories],
            "availability": asset.availability,
            "capacity": asset.capacity,
            "status": asset.status.value,
            "verified": asset.verified,
            "geo": {
                "lat": asset.geo_location.latitude,
                "lng": asset.geo_location.longitude,
            } if asset.geo_location else None,
        }

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for community asset mapping domain.

        Returns beliefs about community resources, desires for community
        resilience, and intentions related to asset development.
        """
        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in ["community", "resources", "assets"]
                or b.get("type") in ["asset_availability", "skill_inventory", "gap_status"]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in ["community_resilience", "resource_access", "skill_development"]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") == "asset_mapping"
                or i.get("action") in ["map_assets", "inventory_skills", "identify_gaps"]
            ],
        }
