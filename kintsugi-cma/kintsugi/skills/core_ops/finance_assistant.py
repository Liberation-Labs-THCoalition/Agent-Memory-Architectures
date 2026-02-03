"""
Finance Assistant Skill Chip for Kintsugi CMA.

This chip handles financial management, budget tracking, and accounting
integration for nonprofit organizations. It connects with popular accounting
systems like QuickBooks, Xero, and Wave to provide real-time financial insights.

Key capabilities:
- Check budget status and available funds
- Track expenses against budget categories
- Generate financial reports and summaries
- Calculate variance between budget and actuals
- Create invoices and track payments

Example:
    chip = FinanceAssistantChip()
    request = SkillRequest(
        intent="budget_check",
        entities={"category": "programs", "fiscal_year": "2024"}
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
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


class BudgetCategory(str, Enum):
    """Standard nonprofit budget categories."""
    PROGRAMS = "programs"
    PERSONNEL = "personnel"
    OPERATIONS = "operations"
    FUNDRAISING = "fundraising"
    ADMINISTRATIVE = "administrative"
    CAPITAL = "capital"


class TransactionType(str, Enum):
    """Types of financial transactions."""
    EXPENSE = "expense"
    INCOME = "income"
    TRANSFER = "transfer"
    ADJUSTMENT = "adjustment"


class PaymentStatus(str, Enum):
    """Status of invoices and payments."""
    PENDING = "pending"
    PARTIAL = "partial"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


@dataclass
class BudgetLine:
    """Represents a budget line item.

    Attributes:
        id: Unique identifier
        category: Budget category
        name: Line item name
        budgeted: Budgeted amount
        spent: Amount spent to date
        committed: Amount committed but not yet spent
        remaining: Calculated remaining amount
        fiscal_year: Fiscal year this budget applies to
    """
    id: str
    category: BudgetCategory
    name: str
    budgeted: Decimal
    spent: Decimal = Decimal("0")
    committed: Decimal = Decimal("0")
    fiscal_year: str = ""

    @property
    def remaining(self) -> Decimal:
        """Calculate remaining budget."""
        return self.budgeted - self.spent - self.committed

    @property
    def utilization_pct(self) -> float:
        """Calculate budget utilization percentage."""
        if self.budgeted == 0:
            return 0.0
        return float((self.spent + self.committed) / self.budgeted * 100)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "category": self.category.value,
            "name": self.name,
            "budgeted": float(self.budgeted),
            "spent": float(self.spent),
            "committed": float(self.committed),
            "remaining": float(self.remaining),
            "utilization_pct": self.utilization_pct,
            "fiscal_year": self.fiscal_year,
        }


@dataclass
class Transaction:
    """Represents a financial transaction.

    Attributes:
        id: Transaction identifier
        date: Transaction date
        amount: Transaction amount
        transaction_type: Type of transaction
        category: Budget category
        description: Transaction description
        vendor: Vendor or payee name
        account_code: Chart of accounts code
        approved_by: Who approved the transaction
    """
    id: str
    date: datetime
    amount: Decimal
    transaction_type: TransactionType
    category: BudgetCategory
    description: str = ""
    vendor: str = ""
    account_code: str = ""
    approved_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "amount": float(self.amount),
            "transaction_type": self.transaction_type.value,
            "category": self.category.value,
            "description": self.description,
            "vendor": self.vendor,
            "account_code": self.account_code,
            "approved_by": self.approved_by,
        }


@dataclass
class Invoice:
    """Represents an invoice.

    Attributes:
        id: Invoice identifier
        invoice_number: External invoice number
        client_name: Client or donor name
        amount: Invoice amount
        due_date: Payment due date
        status: Payment status
        items: Line items on invoice
        created_date: When invoice was created
    """
    id: str
    invoice_number: str
    client_name: str
    amount: Decimal
    due_date: datetime
    status: PaymentStatus = PaymentStatus.PENDING
    items: list[dict[str, Any]] = field(default_factory=list)
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "invoice_number": self.invoice_number,
            "client_name": self.client_name,
            "amount": float(self.amount),
            "due_date": self.due_date.isoformat(),
            "status": self.status.value,
            "items": self.items,
            "created_date": self.created_date.isoformat(),
        }


class FinanceAssistantChip(BaseSkillChip):
    """Financial management, budget tracking, and accounting integration.

    This chip provides comprehensive financial management capabilities
    for nonprofit organizations, integrating with popular accounting
    systems and providing real-time budget insights.

    Intents handled:
        - budget_check: Check budget status for categories
        - expense_report: Generate expense reports
        - invoice_create: Create new invoices
        - financial_summary: Generate financial summaries
        - variance_analysis: Analyze budget vs actual variance

    Consensus actions:
        - approve_expense: Requires approval for expenses over threshold
        - transfer_funds: Requires approval for fund transfers
        - create_invoice: Requires approval for new invoices
        - modify_budget: Requires approval for budget modifications

    Example:
        chip = FinanceAssistantChip()
        request = SkillRequest(
            intent="budget_check",
            entities={"category": "programs"}
        )
        response = await chip.handle(request, context)
    """

    name = "finance_assistant"
    description = "Financial management, budget tracking, and accounting integration"
    version = "1.0.0"
    domain = SkillDomain.FINANCE

    efe_weights = EFEWeights(
        mission_alignment=0.15,
        stakeholder_benefit=0.20,
        resource_efficiency=0.30,
        transparency=0.25,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.FINANCIAL_OPERATIONS,
        SkillCapability.EXTERNAL_API,
        SkillCapability.GENERATE_REPORTS,
    ]

    consensus_actions = ["approve_expense", "transfer_funds", "create_invoice", "modify_budget"]
    required_spans = ["quickbooks_api", "xero_api", "wave_api", "plaid_api"]

    # Expense approval threshold
    EXPENSE_APPROVAL_THRESHOLD = Decimal("1000.00")

    SUPPORTED_INTENTS = {
        "budget_check": "_handle_budget_check",
        "expense_report": "_handle_expense_report",
        "invoice_create": "_handle_invoice_create",
        "financial_summary": "_handle_financial_summary",
        "variance_analysis": "_handle_variance_analysis",
    }

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context with org, user, BDI state

        Returns:
            SkillResponse with financial data or confirmation
        """
        handler_name = self.SUPPORTED_INTENTS.get(request.intent)

        if handler_name is None:
            return SkillResponse(
                content=f"Unknown intent '{request.intent}' for finance_assistant chip.",
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
        """Extract finance-relevant BDI context.

        Filters BDI state for beliefs about financial status,
        budget constraints, and fiscal goals.
        """
        finance_types = {"budget_status", "cash_flow", "expense_level", "funding_status"}

        filtered_beliefs = [
            b for b in beliefs
            if b.get("type") in finance_types or b.get("domain") == "finance"
        ]

        filtered_desires = [
            d for d in desires
            if d.get("type") in {"budget_target", "savings_goal", "reserve_goal"}
        ]

        return {
            "beliefs": filtered_beliefs,
            "desires": filtered_desires,
            "intentions": intentions,
        }

    async def _handle_budget_check(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Check budget status for specified categories.

        Supported entities:
            - category: Budget category to check (or "all")
            - fiscal_year: Fiscal year to check
            - include_committed: Whether to include committed funds
        """
        entities = request.entities
        category = entities.get("category", "all")
        fiscal_year = entities.get("fiscal_year", self._current_fiscal_year())
        include_committed = entities.get("include_committed", True)

        budget_data = await self.check_budget(
            org_id=context.org_id,
            category=category,
            fiscal_year=fiscal_year,
        )

        if category == "all":
            # Summary of all categories
            content_lines = [f"Budget Status - FY{fiscal_year}\n"]
            total_budgeted = Decimal("0")
            total_spent = Decimal("0")
            total_remaining = Decimal("0")

            for line in budget_data["lines"]:
                total_budgeted += Decimal(str(line["budgeted"]))
                total_spent += Decimal(str(line["spent"]))
                total_remaining += Decimal(str(line["remaining"]))

                status_icon = self._get_status_icon(line["utilization_pct"])
                content_lines.append(
                    f"- **{line['name']}**: ${line['spent']:,.2f} / ${line['budgeted']:,.2f} "
                    f"({line['utilization_pct']:.1f}%) {status_icon}"
                )

            content_lines.append(f"\n**Total**: ${float(total_spent):,.2f} / ${float(total_budgeted):,.2f}")
            content_lines.append(f"**Remaining**: ${float(total_remaining):,.2f}")

        else:
            # Specific category
            lines = [l for l in budget_data["lines"] if l["category"] == category]
            if not lines:
                return SkillResponse(
                    content=f"No budget data found for category '{category}'.",
                    success=False,
                )

            content_lines = [f"Budget Status - {category.title()} - FY{fiscal_year}\n"]
            for line in lines:
                content_lines.append(
                    f"**{line['name']}**\n"
                    f"  Budgeted: ${line['budgeted']:,.2f}\n"
                    f"  Spent: ${line['spent']:,.2f}\n"
                    f"  Committed: ${line.get('committed', 0):,.2f}\n"
                    f"  Remaining: ${line['remaining']:,.2f}\n"
                    f"  Utilization: {line['utilization_pct']:.1f}%"
                )

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data=budget_data,
            suggestions=["View expense details?", "Generate variance report?"],
        )

    async def _handle_expense_report(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Generate expense reports.

        Supported entities:
            - date_range: Period to report on
            - category: Filter by category
            - vendor: Filter by vendor
            - min_amount: Minimum transaction amount
            - group_by: Grouping (category, vendor, month)
        """
        entities = request.entities
        date_range = entities.get("date_range", "this_month")
        category = entities.get("category")
        vendor = entities.get("vendor")
        group_by = entities.get("group_by", "category")

        transactions = await self.track_expense(
            org_id=context.org_id,
            date_range=date_range,
            category=category,
            vendor=vendor,
        )

        if not transactions:
            return SkillResponse(
                content=f"No expenses found for the specified period.",
                success=True,
                data={"transactions": [], "total": 0},
            )

        # Group transactions
        grouped = self._group_transactions(transactions, group_by)

        # Build report
        total = sum(t.amount for t in transactions)
        content_lines = [f"Expense Report ({date_range})\n"]
        content_lines.append(f"**Total Expenses**: ${float(total):,.2f}\n")

        for group_name, group_txns in grouped.items():
            group_total = sum(t.amount for t in group_txns)
            content_lines.append(f"**{group_name}**: ${float(group_total):,.2f}")
            for t in group_txns[:3]:  # Top 3 per group
                content_lines.append(f"  - {t.description}: ${float(t.amount):,.2f}")

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data={
                "transactions": [t.to_dict() for t in transactions],
                "total": float(total),
                "grouped": {k: [t.to_dict() for t in v] for k, v in grouped.items()},
            },
            suggestions=["Export to CSV?", "Compare to previous period?"],
        )

    async def _handle_invoice_create(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Create a new invoice.

        Supported entities:
            - client_name: Name of client/donor
            - amount: Invoice total
            - due_date: Payment due date
            - items: Line items list
            - description: Invoice description
        """
        entities = request.entities

        required = ["client_name", "amount"]
        missing = [f for f in required if not entities.get(f)]
        if missing:
            return SkillResponse(
                content=f"Missing required fields: {', '.join(missing)}",
                success=False,
            )

        # Create invoice requires consensus
        return SkillResponse(
            content=f"Ready to create invoice for {entities['client_name']} "
                    f"totaling ${entities['amount']:,.2f}. Please confirm.",
            success=True,
            requires_consensus=True,
            consensus_action="create_invoice",
            data={
                "client_name": entities["client_name"],
                "amount": entities["amount"],
                "due_date": entities.get("due_date"),
                "items": entities.get("items", []),
                "description": entities.get("description", ""),
            },
        )

    async def _handle_financial_summary(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Generate financial summary report.

        Supported entities:
            - report_type: Type (monthly, quarterly, annual)
            - fiscal_year: Fiscal year
            - include_projections: Include projections
        """
        entities = request.entities
        report_type = entities.get("report_type", "monthly")
        fiscal_year = entities.get("fiscal_year", self._current_fiscal_year())

        summary = await self.generate_financial_report(
            org_id=context.org_id,
            report_type=report_type,
            fiscal_year=fiscal_year,
        )

        content = f"""Financial Summary - {report_type.title()} - FY{fiscal_year}

**Revenue**
- Total Revenue: ${summary['revenue']['total']:,.2f}
- Grants: ${summary['revenue']['grants']:,.2f}
- Donations: ${summary['revenue']['donations']:,.2f}
- Program Fees: ${summary['revenue']['program_fees']:,.2f}

**Expenses**
- Total Expenses: ${summary['expenses']['total']:,.2f}
- Programs: ${summary['expenses']['programs']:,.2f}
- Personnel: ${summary['expenses']['personnel']:,.2f}
- Operations: ${summary['expenses']['operations']:,.2f}

**Net Position**
- Net Income: ${summary['net_income']:,.2f}
- Cash on Hand: ${summary['cash_on_hand']:,.2f}
- Months of Runway: {summary['runway_months']:.1f}
"""

        return SkillResponse(
            content=content,
            success=True,
            data=summary,
            suggestions=["View detailed breakdown?", "Compare to last period?", "Export to PDF?"],
        )

    async def _handle_variance_analysis(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Analyze budget vs actual variance.

        Supported entities:
            - category: Budget category to analyze
            - fiscal_year: Fiscal year
            - threshold_pct: Highlight variances above this percentage
        """
        entities = request.entities
        category = entities.get("category", "all")
        fiscal_year = entities.get("fiscal_year", self._current_fiscal_year())
        threshold_pct = entities.get("threshold_pct", 10.0)

        variance_data = await self.calculate_variance(
            org_id=context.org_id,
            category=category,
            fiscal_year=fiscal_year,
        )

        content_lines = [f"Budget Variance Analysis - FY{fiscal_year}\n"]
        content_lines.append(f"(Flagging variances > {threshold_pct}%)\n")

        for item in variance_data["items"]:
            variance_pct = item["variance_pct"]
            flag = "***" if abs(variance_pct) > threshold_pct else ""

            direction = "over" if variance_pct > 0 else "under"
            content_lines.append(
                f"- **{item['name']}**: {direction} by ${abs(item['variance_amount']):,.2f} "
                f"({variance_pct:+.1f}%) {flag}"
            )

        # Summary
        total_variance = sum(item["variance_amount"] for item in variance_data["items"])
        content_lines.append(f"\n**Net Variance**: ${total_variance:,.2f}")

        if total_variance > 0:
            content_lines.append("Overall trending OVER budget")
        else:
            content_lines.append("Overall trending UNDER budget")

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data=variance_data,
            suggestions=["Drill into specific category?", "View historical trends?"],
        )

    # Core implementation methods

    async def check_budget(
        self,
        org_id: str,
        category: str = "all",
        fiscal_year: str = "",
    ) -> dict[str, Any]:
        """Check budget status for specified categories.

        Args:
            org_id: Organization identifier
            category: Category to check or "all"
            fiscal_year: Fiscal year to check

        Returns:
            Dictionary with budget lines and summary
        """
        budget_lines = await self._get_budget_lines(org_id, fiscal_year)

        if category != "all":
            try:
                cat_enum = BudgetCategory(category)
                budget_lines = [l for l in budget_lines if l.category == cat_enum]
            except ValueError:
                pass

        return {
            "org_id": org_id,
            "fiscal_year": fiscal_year or self._current_fiscal_year(),
            "lines": [l.to_dict() for l in budget_lines],
            "total_budgeted": sum(float(l.budgeted) for l in budget_lines),
            "total_spent": sum(float(l.spent) for l in budget_lines),
            "total_remaining": sum(float(l.remaining) for l in budget_lines),
        }

    async def track_expense(
        self,
        org_id: str,
        date_range: str = "this_month",
        category: str | None = None,
        vendor: str | None = None,
    ) -> list[Transaction]:
        """Track expenses within a date range.

        Args:
            org_id: Organization identifier
            date_range: Period to track
            category: Filter by category
            vendor: Filter by vendor

        Returns:
            List of transactions
        """
        transactions = await self._get_transactions(org_id, date_range)

        # Filter by category
        if category:
            try:
                cat_enum = BudgetCategory(category)
                transactions = [t for t in transactions if t.category == cat_enum]
            except ValueError:
                pass

        # Filter by vendor
        if vendor:
            vendor_lower = vendor.lower()
            transactions = [t for t in transactions if vendor_lower in t.vendor.lower()]

        # Only expenses
        transactions = [t for t in transactions if t.transaction_type == TransactionType.EXPENSE]

        return transactions

    async def generate_financial_report(
        self,
        org_id: str,
        report_type: str = "monthly",
        fiscal_year: str = "",
    ) -> dict[str, Any]:
        """Generate a financial summary report.

        Args:
            org_id: Organization identifier
            report_type: Type of report (monthly, quarterly, annual)
            fiscal_year: Fiscal year for the report

        Returns:
            Dictionary with financial summary data
        """
        # Simulated financial data
        return {
            "org_id": org_id,
            "report_type": report_type,
            "fiscal_year": fiscal_year or self._current_fiscal_year(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "revenue": {
                "total": 250000.00,
                "grants": 150000.00,
                "donations": 75000.00,
                "program_fees": 25000.00,
            },
            "expenses": {
                "total": 200000.00,
                "programs": 120000.00,
                "personnel": 50000.00,
                "operations": 20000.00,
                "fundraising": 10000.00,
            },
            "net_income": 50000.00,
            "cash_on_hand": 125000.00,
            "runway_months": 7.5,
        }

    async def calculate_variance(
        self,
        org_id: str,
        category: str = "all",
        fiscal_year: str = "",
    ) -> dict[str, Any]:
        """Calculate variance between budget and actuals.

        Args:
            org_id: Organization identifier
            category: Category to analyze or "all"
            fiscal_year: Fiscal year

        Returns:
            Dictionary with variance analysis
        """
        budget_data = await self.check_budget(org_id, category, fiscal_year)

        variance_items = []
        for line in budget_data["lines"]:
            budgeted = Decimal(str(line["budgeted"]))
            spent = Decimal(str(line["spent"]))
            variance_amount = spent - budgeted
            variance_pct = float((variance_amount / budgeted * 100)) if budgeted else 0

            variance_items.append({
                "name": line["name"],
                "category": line["category"],
                "budgeted": float(budgeted),
                "actual": float(spent),
                "variance_amount": float(variance_amount),
                "variance_pct": variance_pct,
            })

        return {
            "org_id": org_id,
            "fiscal_year": fiscal_year or self._current_fiscal_year(),
            "items": variance_items,
            "total_variance": sum(item["variance_amount"] for item in variance_items),
        }

    # Private helper methods

    async def _get_budget_lines(
        self, org_id: str, fiscal_year: str
    ) -> list[BudgetLine]:
        """Fetch budget lines from accounting system."""
        # Simulated data
        return [
            BudgetLine(
                id="bl_001",
                category=BudgetCategory.PROGRAMS,
                name="Program Expenses",
                budgeted=Decimal("120000"),
                spent=Decimal("95000"),
                committed=Decimal("10000"),
                fiscal_year=fiscal_year or self._current_fiscal_year(),
            ),
            BudgetLine(
                id="bl_002",
                category=BudgetCategory.PERSONNEL,
                name="Staff Salaries",
                budgeted=Decimal("180000"),
                spent=Decimal("135000"),
                committed=Decimal("45000"),
                fiscal_year=fiscal_year or self._current_fiscal_year(),
            ),
            BudgetLine(
                id="bl_003",
                category=BudgetCategory.OPERATIONS,
                name="Office & Utilities",
                budgeted=Decimal("24000"),
                spent=Decimal("18500"),
                committed=Decimal("2000"),
                fiscal_year=fiscal_year or self._current_fiscal_year(),
            ),
            BudgetLine(
                id="bl_004",
                category=BudgetCategory.FUNDRAISING,
                name="Fundraising",
                budgeted=Decimal("15000"),
                spent=Decimal("12000"),
                committed=Decimal("0"),
                fiscal_year=fiscal_year or self._current_fiscal_year(),
            ),
        ]

    async def _get_transactions(
        self, org_id: str, date_range: str
    ) -> list[Transaction]:
        """Fetch transactions from accounting system."""
        # Simulated data
        now = datetime.now(timezone.utc)
        return [
            Transaction(
                id="txn_001",
                date=now,
                amount=Decimal("5000"),
                transaction_type=TransactionType.EXPENSE,
                category=BudgetCategory.PROGRAMS,
                description="Program supplies",
                vendor="Office Supply Co",
                account_code="5100",
            ),
            Transaction(
                id="txn_002",
                date=now,
                amount=Decimal("2500"),
                transaction_type=TransactionType.EXPENSE,
                category=BudgetCategory.OPERATIONS,
                description="Monthly rent",
                vendor="Property Management LLC",
                account_code="5200",
            ),
            Transaction(
                id="txn_003",
                date=now,
                amount=Decimal("15000"),
                transaction_type=TransactionType.EXPENSE,
                category=BudgetCategory.PERSONNEL,
                description="Payroll",
                vendor="Payroll Provider",
                account_code="5000",
            ),
        ]

    def _group_transactions(
        self, transactions: list[Transaction], group_by: str
    ) -> dict[str, list[Transaction]]:
        """Group transactions by specified field."""
        grouped: dict[str, list[Transaction]] = {}

        for t in transactions:
            if group_by == "category":
                key = t.category.value.title()
            elif group_by == "vendor":
                key = t.vendor or "Unknown"
            elif group_by == "month":
                key = t.date.strftime("%Y-%m")
            else:
                key = "All"

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(t)

        return grouped

    def _current_fiscal_year(self) -> str:
        """Get current fiscal year (assuming July-June FY)."""
        now = datetime.now(timezone.utc)
        if now.month >= 7:
            return str(now.year + 1)
        return str(now.year)

    def _get_status_icon(self, utilization_pct: float) -> str:
        """Get status indicator based on utilization."""
        if utilization_pct < 50:
            return "[low]"
        elif utilization_pct < 80:
            return "[ok]"
        elif utilization_pct < 95:
            return "[watch]"
        else:
            return "[critical]"
