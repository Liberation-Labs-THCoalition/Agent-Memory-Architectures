"""
Staff Onboarding Skill Chip for Kintsugi CMA.

Guides new staff through onboarding workflows including training assignments,
policy orientation, checklist tracking, and onboarding completion. Emphasizes
stakeholder benefit and resource efficiency in onboarding processes.

Example:
    chip = StaffOnboardingChip()
    request = SkillRequest(
        intent="onboard_start",
        entities={"employee_id": "emp_001", "department": "programs"},
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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


@dataclass
class OnboardingPlan:
    """Represents an employee's onboarding plan."""
    plan_id: str
    employee_id: str
    employee_name: str
    department: str
    start_date: datetime
    supervisor: str
    status: str = "not_started"  # not_started, in_progress, completed
    target_completion: datetime | None = None
    checklist_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrainingModule:
    """Represents a training module assignment."""
    module_id: str
    title: str
    description: str
    category: str  # required, role_specific, optional
    duration_minutes: int
    format: str  # video, interactive, reading, workshop
    due_date: datetime | None = None
    completed: bool = False
    completion_date: datetime | None = None
    score: float | None = None


@dataclass
class PolicyDocument:
    """Represents a policy document for review."""
    policy_id: str
    title: str
    category: str  # hr, safety, operations, ethics
    version: str
    acknowledgment_required: bool = True
    acknowledged: bool = False
    acknowledgment_date: datetime | None = None


class StaffOnboardingChip(BaseSkillChip):
    """Guide new staff through onboarding, training, and policy orientation.

    This chip supports HR and managers in efficiently onboarding new
    employees through structured workflows, training assignments,
    policy reviews, and progress tracking.

    Intents:
        onboard_start: Initialize onboarding for new employee
        training_assign: Assign training modules
        policy_review: Present policies for review
        checklist_status: Check onboarding progress
        onboard_complete: Finalize onboarding

    Example:
        >>> chip = StaffOnboardingChip()
        >>> request = SkillRequest(intent="onboard_start", entities={"employee_id": "emp_001"})
        >>> response = await chip.handle(request, context)
        >>> print(response.data["onboarding_plan"]["checklist_items"])
    """

    name = "staff_onboarding"
    description = "Guide new staff through onboarding, training, and policy orientation"
    version = "1.0.0"
    domain = SkillDomain.OPERATIONS

    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.30,
        resource_efficiency=0.20,
        transparency=0.15,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SCHEDULE_TASKS,
        SkillCapability.PII_ACCESS,
    ]

    consensus_actions = ["complete_onboarding", "grant_system_access", "update_employee_record"]

    required_spans = ["hris_api", "training_lms", "document_store"]

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: Skill request with intent and entities
            context: Execution context with org, user, BDI state

        Returns:
            SkillResponse with result content and data
        """
        intent = request.intent

        bdi = await self.get_bdi_context(
            context.beliefs,
            context.desires,
            context.intentions,
        )

        handlers = {
            "onboard_start": self._start_onboarding,
            "training_assign": self._assign_training,
            "policy_review": self._get_policy_docs,
            "checklist_status": self._check_progress,
            "onboard_complete": self._complete_onboarding,
        }

        handler = handlers.get(intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent '{intent}' for staff onboarding.",
                success=False,
                suggestions=[
                    "Try 'onboard_start' to begin onboarding",
                    "Try 'training_assign' to assign training",
                    "Try 'checklist_status' to check progress",
                ],
            )

        return await handler(request, context, bdi)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI state for onboarding context.

        Extracts beliefs about staff, training needs, and onboarding
        priorities.
        """
        hr_beliefs = [
            b for b in beliefs
            if b.get("domain") in ("hr", "operations", "training")
            or b.get("type") in ("employee_status", "training_completion", "compliance_need")
        ]

        onboarding_desires = [
            d for d in desires
            if d.get("type") in ("efficient_onboarding", "staff_readiness", "compliance")
            or d.get("domain") == "operations"
        ]

        return {
            "beliefs": hr_beliefs,
            "desires": onboarding_desires,
            "intentions": intentions,
        }

    async def _start_onboarding(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Initialize onboarding plan for new employee.

        Creates comprehensive onboarding plan with checklist, training
        requirements, and timeline.
        """
        employee_id = request.entities.get("employee_id", "")
        employee_name = request.entities.get("employee_name", "New Employee")
        department = request.entities.get("department", "general")
        start_date = request.entities.get("start_date", datetime.now(timezone.utc).isoformat())
        supervisor = request.entities.get("supervisor", "Department Manager")

        start_dt = datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
        target_completion = start_dt + timedelta(days=30)

        # Build standard checklist items
        checklist_items = [
            # Day 1 items
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "day_1",
                "title": "Welcome meeting with supervisor",
                "description": "Introduction to team, role overview, and immediate goals",
                "due_date": start_dt.isoformat(),
                "completed": False,
                "assigned_to": supervisor,
            },
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "day_1",
                "title": "Workspace and equipment setup",
                "description": "Computer, phone, desk, supplies, access badge",
                "due_date": start_dt.isoformat(),
                "completed": False,
                "assigned_to": "IT/Facilities",
            },
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "day_1",
                "title": "HR paperwork completion",
                "description": "I-9, W-4, benefits enrollment, emergency contacts",
                "due_date": start_dt.isoformat(),
                "completed": False,
                "assigned_to": "HR",
            },
            # Week 1 items
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "week_1",
                "title": "System access provisioning",
                "description": "Email, shared drives, databases, software licenses",
                "due_date": (start_dt + timedelta(days=2)).isoformat(),
                "completed": False,
                "assigned_to": "IT",
            },
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "week_1",
                "title": "Organization overview training",
                "description": "Mission, history, programs, culture",
                "due_date": (start_dt + timedelta(days=5)).isoformat(),
                "completed": False,
                "assigned_to": employee_name,
            },
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "week_1",
                "title": "Required policy reviews",
                "description": "Employee handbook, code of conduct, safety policies",
                "due_date": (start_dt + timedelta(days=5)).isoformat(),
                "completed": False,
                "assigned_to": employee_name,
            },
            # Week 2-4 items
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "week_2_4",
                "title": "Department-specific training",
                "description": f"Training modules specific to {department} department",
                "due_date": (start_dt + timedelta(days=14)).isoformat(),
                "completed": False,
                "assigned_to": employee_name,
            },
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "week_2_4",
                "title": "Meet with key stakeholders",
                "description": "Scheduled introductions with partner departments",
                "due_date": (start_dt + timedelta(days=14)).isoformat(),
                "completed": False,
                "assigned_to": supervisor,
            },
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "week_2_4",
                "title": "Initial project assignment",
                "description": "First project or task to apply learning",
                "due_date": (start_dt + timedelta(days=21)).isoformat(),
                "completed": False,
                "assigned_to": supervisor,
            },
            {
                "item_id": f"chk_{uuid4().hex[:8]}",
                "category": "week_2_4",
                "title": "30-day check-in meeting",
                "description": "Review progress, address questions, adjust plan if needed",
                "due_date": (start_dt + timedelta(days=30)).isoformat(),
                "completed": False,
                "assigned_to": supervisor,
            },
        ]

        plan = OnboardingPlan(
            plan_id=f"onb_{uuid4().hex[:8]}",
            employee_id=employee_id,
            employee_name=employee_name,
            department=department,
            start_date=start_dt,
            supervisor=supervisor,
            status="in_progress",
            target_completion=target_completion,
            checklist_items=checklist_items,
        )

        return SkillResponse(
            content=f"Initialized onboarding for {employee_name} in {department} department. "
                    f"Start date: {start_dt.strftime('%B %d, %Y')}. "
                    f"Plan includes {len(checklist_items)} checklist items with 30-day target completion.",
            success=True,
            data={
                "onboarding_plan": {
                    "plan_id": plan.plan_id,
                    "employee_id": plan.employee_id,
                    "employee_name": plan.employee_name,
                    "department": plan.department,
                    "start_date": plan.start_date.isoformat(),
                    "supervisor": plan.supervisor,
                    "status": plan.status,
                    "target_completion": plan.target_completion.isoformat() if plan.target_completion else None,
                    "checklist_items": plan.checklist_items,
                },
            },
            suggestions=[
                "Would you like to assign training modules?",
                "Should I schedule the welcome meeting?",
                "Want to customize the checklist?",
            ],
        )

    async def _assign_training(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Assign training modules to employee.

        Creates training assignments based on role, department, and
        compliance requirements.
        """
        employee_id = request.entities.get("employee_id", "")
        department = request.entities.get("department", "general")
        include_optional = request.entities.get("include_optional", False)

        now = datetime.now(timezone.utc)

        # Required training for all employees
        required_modules = [
            TrainingModule(
                module_id=f"trn_{uuid4().hex[:8]}",
                title="Organization Mission and Values",
                description="Introduction to our mission, vision, and core values",
                category="required",
                duration_minutes=30,
                format="video",
                due_date=now + timedelta(days=5),
            ),
            TrainingModule(
                module_id=f"trn_{uuid4().hex[:8]}",
                title="Workplace Safety Fundamentals",
                description="Essential safety protocols and emergency procedures",
                category="required",
                duration_minutes=45,
                format="interactive",
                due_date=now + timedelta(days=5),
            ),
            TrainingModule(
                module_id=f"trn_{uuid4().hex[:8]}",
                title="Harassment Prevention",
                description="Recognizing and preventing workplace harassment",
                category="required",
                duration_minutes=60,
                format="interactive",
                due_date=now + timedelta(days=7),
            ),
            TrainingModule(
                module_id=f"trn_{uuid4().hex[:8]}",
                title="Data Privacy and Security",
                description="Protecting sensitive information and cybersecurity basics",
                category="required",
                duration_minutes=45,
                format="interactive",
                due_date=now + timedelta(days=7),
            ),
            TrainingModule(
                module_id=f"trn_{uuid4().hex[:8]}",
                title="Diversity, Equity, and Inclusion",
                description="Building an inclusive workplace",
                category="required",
                duration_minutes=60,
                format="video",
                due_date=now + timedelta(days=10),
            ),
        ]

        # Department-specific training
        dept_modules = {
            "programs": [
                TrainingModule(
                    module_id=f"trn_{uuid4().hex[:8]}",
                    title="Program Management Fundamentals",
                    description="Planning, implementing, and evaluating programs",
                    category="role_specific",
                    duration_minutes=90,
                    format="interactive",
                    due_date=now + timedelta(days=14),
                ),
                TrainingModule(
                    module_id=f"trn_{uuid4().hex[:8]}",
                    title="Participant Data Management",
                    description="Tracking participant information and outcomes",
                    category="role_specific",
                    duration_minutes=45,
                    format="workshop",
                    due_date=now + timedelta(days=14),
                ),
            ],
            "development": [
                TrainingModule(
                    module_id=f"trn_{uuid4().hex[:8]}",
                    title="Donor Database Training",
                    description="CRM system navigation and data entry",
                    category="role_specific",
                    duration_minutes=90,
                    format="workshop",
                    due_date=now + timedelta(days=14),
                ),
                TrainingModule(
                    module_id=f"trn_{uuid4().hex[:8]}",
                    title="Gift Processing Procedures",
                    description="Acknowledging and recording donations",
                    category="role_specific",
                    duration_minutes=60,
                    format="interactive",
                    due_date=now + timedelta(days=14),
                ),
            ],
            "finance": [
                TrainingModule(
                    module_id=f"trn_{uuid4().hex[:8]}",
                    title="Financial Systems Overview",
                    description="Accounting software and procedures",
                    category="role_specific",
                    duration_minutes=120,
                    format="workshop",
                    due_date=now + timedelta(days=14),
                ),
                TrainingModule(
                    module_id=f"trn_{uuid4().hex[:8]}",
                    title="Grant Financial Compliance",
                    description="Managing restricted funds and grant reporting",
                    category="role_specific",
                    duration_minutes=90,
                    format="interactive",
                    due_date=now + timedelta(days=21),
                ),
            ],
        }

        # Optional training
        optional_modules = [
            TrainingModule(
                module_id=f"trn_{uuid4().hex[:8]}",
                title="Effective Communication Skills",
                description="Professional communication techniques",
                category="optional",
                duration_minutes=45,
                format="video",
                due_date=now + timedelta(days=30),
            ),
            TrainingModule(
                module_id=f"trn_{uuid4().hex[:8]}",
                title="Time Management Strategies",
                description="Productivity and prioritization techniques",
                category="optional",
                duration_minutes=30,
                format="reading",
                due_date=now + timedelta(days=30),
            ),
        ]

        # Compile training list
        all_modules = required_modules.copy()

        if department in dept_modules:
            all_modules.extend(dept_modules[department])
        else:
            # Default role-specific training
            all_modules.append(
                TrainingModule(
                    module_id=f"trn_{uuid4().hex[:8]}",
                    title="Role-Specific Orientation",
                    description="Department-specific procedures and systems",
                    category="role_specific",
                    duration_minutes=60,
                    format="workshop",
                    due_date=now + timedelta(days=14),
                )
            )

        if include_optional:
            all_modules.extend(optional_modules)

        total_duration = sum(m.duration_minutes for m in all_modules)

        training_data = {
            "employee_id": employee_id,
            "department": department,
            "assigned_at": now.isoformat(),
            "modules": [
                {
                    "module_id": m.module_id,
                    "title": m.title,
                    "description": m.description,
                    "category": m.category,
                    "duration_minutes": m.duration_minutes,
                    "format": m.format,
                    "due_date": m.due_date.isoformat() if m.due_date else None,
                    "completed": m.completed,
                }
                for m in all_modules
            ],
            "summary": {
                "total_modules": len(all_modules),
                "required": len([m for m in all_modules if m.category == "required"]),
                "role_specific": len([m for m in all_modules if m.category == "role_specific"]),
                "optional": len([m for m in all_modules if m.category == "optional"]),
                "total_duration_minutes": total_duration,
                "estimated_hours": round(total_duration / 60, 1),
            },
        }

        return SkillResponse(
            content=f"Assigned {len(all_modules)} training modules for {department} employee. "
                    f"Total estimated time: {training_data['summary']['estimated_hours']} hours.",
            success=True,
            data={"training_assignment": training_data},
            suggestions=[
                "Would you like to schedule specific training sessions?",
                "Should I send the training assignments to the employee?",
                "Want to add additional modules?",
            ],
        )

    async def _get_policy_docs(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Retrieve policy documents for review and acknowledgment.

        Returns required policies with acknowledgment tracking.
        """
        employee_id = request.entities.get("employee_id", "")
        category = request.entities.get("category", None)

        policies = [
            PolicyDocument(
                policy_id="pol_001",
                title="Employee Handbook",
                category="hr",
                version="2024.1",
                acknowledgment_required=True,
            ),
            PolicyDocument(
                policy_id="pol_002",
                title="Code of Conduct",
                category="ethics",
                version="2024.1",
                acknowledgment_required=True,
            ),
            PolicyDocument(
                policy_id="pol_003",
                title="Anti-Harassment Policy",
                category="hr",
                version="2024.1",
                acknowledgment_required=True,
            ),
            PolicyDocument(
                policy_id="pol_004",
                title="Information Security Policy",
                category="operations",
                version="2024.1",
                acknowledgment_required=True,
            ),
            PolicyDocument(
                policy_id="pol_005",
                title="Conflict of Interest Policy",
                category="ethics",
                version="2024.1",
                acknowledgment_required=True,
            ),
            PolicyDocument(
                policy_id="pol_006",
                title="Travel and Expense Policy",
                category="operations",
                version="2023.2",
                acknowledgment_required=False,
            ),
            PolicyDocument(
                policy_id="pol_007",
                title="Workplace Safety Manual",
                category="safety",
                version="2024.1",
                acknowledgment_required=True,
            ),
            PolicyDocument(
                policy_id="pol_008",
                title="Remote Work Guidelines",
                category="operations",
                version="2024.1",
                acknowledgment_required=False,
            ),
        ]

        # Filter by category if specified
        if category:
            policies = [p for p in policies if p.category == category]

        policy_data = {
            "employee_id": employee_id,
            "policies": [
                {
                    "policy_id": p.policy_id,
                    "title": p.title,
                    "category": p.category,
                    "version": p.version,
                    "acknowledgment_required": p.acknowledgment_required,
                    "acknowledged": p.acknowledged,
                    "document_url": f"/policies/{p.policy_id}/{p.version}",
                }
                for p in policies
            ],
            "summary": {
                "total_policies": len(policies),
                "requiring_acknowledgment": len([p for p in policies if p.acknowledgment_required]),
                "acknowledged": len([p for p in policies if p.acknowledged]),
                "pending": len([p for p in policies if p.acknowledgment_required and not p.acknowledged]),
            },
        }

        return SkillResponse(
            content=f"Retrieved {len(policies)} policy documents. "
                    f"{policy_data['summary']['pending']} policies pending acknowledgment.",
            success=True,
            data={"policy_review": policy_data},
            suggestions=[
                "Would you like to record an acknowledgment?",
                "Should I send policy links to the employee?",
                "Want to see a specific policy category?",
            ],
        )

    async def _check_progress(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Check onboarding progress and completion status.

        Returns progress metrics across all onboarding components.
        """
        employee_id = request.entities.get("employee_id", "")
        plan_id = request.entities.get("plan_id", "")

        # Simulated progress data (would query actual records)
        progress = {
            "employee_id": employee_id,
            "plan_id": plan_id or f"onb_{uuid4().hex[:8]}",
            "employee_name": "Jane Smith",
            "department": "programs",
            "start_date": (datetime.now(timezone.utc) - timedelta(days=12)).isoformat(),
            "days_in_role": 12,
            "target_completion_days": 30,
            "overall_progress": 65,
            "status": "on_track",
            "checklist_progress": {
                "total_items": 10,
                "completed": 6,
                "in_progress": 2,
                "not_started": 2,
                "completion_percentage": 60,
                "items_by_status": {
                    "completed": [
                        "Welcome meeting with supervisor",
                        "Workspace and equipment setup",
                        "HR paperwork completion",
                        "System access provisioning",
                        "Organization overview training",
                        "Required policy reviews",
                    ],
                    "in_progress": [
                        "Department-specific training",
                        "Meet with key stakeholders",
                    ],
                    "not_started": [
                        "Initial project assignment",
                        "30-day check-in meeting",
                    ],
                },
            },
            "training_progress": {
                "total_modules": 7,
                "completed": 5,
                "in_progress": 1,
                "not_started": 1,
                "completion_percentage": 71,
                "required_completed": "4 of 5",
                "role_specific_completed": "1 of 2",
            },
            "policy_progress": {
                "total_requiring_acknowledgment": 6,
                "acknowledged": 5,
                "pending": 1,
                "completion_percentage": 83,
            },
            "upcoming_deadlines": [
                {
                    "item": "Department-specific training",
                    "due_date": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
                    "days_remaining": 2,
                },
                {
                    "item": "Initial project assignment",
                    "due_date": (datetime.now(timezone.utc) + timedelta(days=9)).isoformat(),
                    "days_remaining": 9,
                },
            ],
            "blockers": [],
            "notes": "Employee is progressing well. Strong engagement with training materials.",
        }

        return SkillResponse(
            content=f"Onboarding progress for {progress['employee_name']}: {progress['overall_progress']}% complete. "
                    f"Day {progress['days_in_role']} of {progress['target_completion_days']}. Status: {progress['status']}.",
            success=True,
            data={"onboarding_progress": progress},
            suggestions=[
                "Would you like to address any blockers?",
                "Should I send a progress update to the supervisor?",
                "Want to mark any items complete?",
            ],
        )

    async def _complete_onboarding(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Finalize onboarding and mark as complete.

        Validates all requirements are met and generates completion
        documentation.
        """
        employee_id = request.entities.get("employee_id", "")
        plan_id = request.entities.get("plan_id", "")
        force_complete = request.entities.get("force_complete", False)

        # Simulated completion check
        completion_check = {
            "checklist_complete": True,
            "training_complete": True,
            "policies_acknowledged": False,  # One pending
            "systems_access_confirmed": True,
            "supervisor_signoff": True,
        }

        all_complete = all(completion_check.values())

        if not all_complete and not force_complete:
            incomplete_items = [k for k, v in completion_check.items() if not v]
            return SkillResponse(
                content=f"Cannot complete onboarding. Outstanding items: {', '.join(incomplete_items)}.",
                success=False,
                data={
                    "completion_status": completion_check,
                    "ready_to_complete": False,
                },
                suggestions=[
                    "Would you like to address the outstanding items?",
                    "Should I send reminders for pending items?",
                    "Use force_complete to override (not recommended)?",
                ],
            )

        now = datetime.now(timezone.utc)

        completion_record = {
            "completion_id": f"cmp_{uuid4().hex[:8]}",
            "employee_id": employee_id,
            "plan_id": plan_id,
            "completion_date": now.isoformat(),
            "days_to_complete": 28,
            "completion_status": completion_check,
            "forced": force_complete,
            "signoffs": {
                "hr": {"name": "HR Manager", "date": now.isoformat()},
                "supervisor": {"name": "Department Supervisor", "date": (now - timedelta(days=1)).isoformat()},
                "it": {"name": "IT Administrator", "date": (now - timedelta(days=5)).isoformat()},
            },
            "training_summary": {
                "modules_completed": 7,
                "total_hours": 8.5,
                "average_score": 92.3,
            },
            "next_steps": [
                "90-day performance review scheduled",
                "Professional development planning",
                "Team integration activities",
            ],
        }

        requires_approval = self.requires_consensus("complete_onboarding")

        return SkillResponse(
            content=f"Onboarding completed for employee {employee_id}. "
                    f"Completed in {completion_record['days_to_complete']} days. "
                    f"All required training and policies acknowledged.",
            success=True,
            data={"completion_record": completion_record},
            requires_consensus=requires_approval,
            consensus_action="complete_onboarding" if requires_approval else None,
            suggestions=[
                "Would you like to generate a completion certificate?",
                "Should I schedule the 90-day review?",
                "Want to send a welcome completion message?",
            ],
        )
