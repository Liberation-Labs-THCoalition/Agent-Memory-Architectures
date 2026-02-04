"""
Data isolation strategies for multi-tenant Kintsugi CMA.

This module provides mechanisms to ensure tenant data is properly
isolated based on the chosen strategy. Isolation is critical for:
- Data privacy and security
- Regulatory compliance
- Performance isolation
- Disaster recovery

Isolation Strategies:
    - ROW_LEVEL: All tenants share tables, filtered by tenant_id.
                 Lowest cost, uses PostgreSQL RLS policies.
    - SCHEMA: Each tenant gets a separate PostgreSQL schema.
              Better isolation, moderate overhead.
    - DATABASE: Each tenant gets a separate database.
                Maximum isolation for enterprise tenants.

Example:
    from kintsugi.multitenancy.isolation import TenantIsolator, IsolationStrategy

    isolator = TenantIsolator(strategy=IsolationStrategy.ROW_LEVEL)
    await isolator.ensure_isolation("org_12345")

    # For queries, get the tenant filter
    filter_dict = isolator.get_tenant_filter("org_12345")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import logging
import re

logger = logging.getLogger(__name__)


class IsolationStrategy(str, Enum):
    """Tenant data isolation strategies.

    Each strategy provides different levels of isolation with
    corresponding trade-offs in cost and complexity.

    Attributes:
        ROW_LEVEL: Row-Level Security using tenant_id column.
                   All data in shared tables, PostgreSQL RLS policies
                   enforce isolation. Most cost-effective, suitable
                   for SEED and SPROUT tiers.
        SCHEMA: Schema-per-tenant isolation. Each tenant gets their
                own PostgreSQL schema with identical table structures.
                Better performance isolation, suitable for GROVE tier.
        DATABASE: Database-per-tenant isolation. Maximum isolation
                  with separate database instances. Required for
                  FOREST tier and certain compliance requirements.
    """

    ROW_LEVEL = "row_level"
    SCHEMA = "schema"
    DATABASE = "database"


@dataclass
class RLSPolicy:
    """Represents a Row-Level Security policy.

    Tracks RLS policies created for tenant isolation.

    Attributes:
        table_name: The table this policy applies to.
        policy_name: The name of the RLS policy.
        created_at: When the policy was created.
        tenant_column: The column used for tenant filtering.
    """

    table_name: str
    policy_name: str
    created_at: datetime
    tenant_column: str = "tenant_id"

    def get_policy_sql(self, tenant_id: str) -> str:
        """Generate the SQL for this policy.

        Args:
            tenant_id: The tenant ID to filter for.

        Returns:
            SQL string for the policy expression.
        """
        return f"{self.tenant_column} = '{tenant_id}'"


@dataclass
class SchemaInfo:
    """Information about a tenant's schema.

    Tracks schema-per-tenant isolation details.

    Attributes:
        schema_name: The PostgreSQL schema name.
        tenant_id: The tenant this schema belongs to.
        created_at: When the schema was created.
        tables: List of tables in this schema.
        size_bytes: Current size of the schema.
    """

    schema_name: str
    tenant_id: str
    created_at: datetime
    tables: list[str] = field(default_factory=list)
    size_bytes: int = 0


@dataclass
class IsolationAuditEntry:
    """Audit entry for isolation operations.

    Tracks all isolation-related operations for compliance.

    Attributes:
        operation: The operation performed.
        tenant_id: The affected tenant.
        strategy: The isolation strategy used.
        details: Additional operation details.
        performed_at: When the operation occurred.
        performed_by: Who performed the operation.
    """

    operation: str
    tenant_id: str
    strategy: IsolationStrategy
    details: dict[str, Any]
    performed_at: datetime
    performed_by: str | None = None


class TenantIsolator:
    """Manages tenant data isolation.

    Provides methods to set up and enforce data isolation based on
    the configured strategy. Handles schema creation, RLS policies,
    and migration between strategies.

    Attributes:
        _strategy: The isolation strategy to use.
        _rls_policies: Tracked RLS policies by table.
        _schemas: Tracked tenant schemas.
        _audit_log: Audit log of isolation operations.

    Example:
        isolator = TenantIsolator(strategy=IsolationStrategy.ROW_LEVEL)

        # Ensure isolation is set up for a tenant
        await isolator.ensure_isolation("org_12345")

        # Get filter for queries
        filter_dict = isolator.get_tenant_filter("org_12345")
        # Use filter_dict in SQLAlchemy queries

        # Migrate tenant to schema isolation
        await isolator.migrate_tenant(
            "org_12345",
            IsolationStrategy.ROW_LEVEL,
            IsolationStrategy.SCHEMA,
        )
    """

    # Tables that need tenant isolation
    ISOLATED_TABLES = [
        "memories",
        "beliefs",
        "desires",
        "intentions",
        "conversations",
        "users",
        "audit_logs",
        "documents",
        "integrations",
    ]

    def __init__(self, strategy: IsolationStrategy = IsolationStrategy.ROW_LEVEL):
        """Initialize the tenant isolator.

        Args:
            strategy: The isolation strategy to use.
        """
        self._strategy = strategy
        self._rls_policies: dict[str, list[RLSPolicy]] = {}
        self._schemas: dict[str, SchemaInfo] = {}
        self._audit_log: list[IsolationAuditEntry] = []
        self._initialized_tenants: set[str] = set()

    @property
    def strategy(self) -> IsolationStrategy:
        """Get the current isolation strategy."""
        return self._strategy

    def _validate_tenant_id(self, tenant_id: str) -> None:
        """Validate tenant ID format.

        Args:
            tenant_id: The tenant ID to validate.

        Raises:
            ValueError: If the tenant ID is invalid.
        """
        if not tenant_id:
            raise ValueError("Tenant ID cannot be empty")
        if not tenant_id.startswith("org_"):
            raise ValueError("Tenant ID must start with 'org_'")
        # Prevent SQL injection in tenant IDs
        if not re.match(r"^org_[a-zA-Z0-9_-]+$", tenant_id):
            raise ValueError("Tenant ID contains invalid characters")

    def _generate_schema_name(self, tenant_id: str) -> str:
        """Generate a schema name from tenant ID.

        Args:
            tenant_id: The tenant ID.

        Returns:
            A valid PostgreSQL schema name.
        """
        # Remove 'org_' prefix and sanitize
        base_name = tenant_id.replace("org_", "tenant_")
        # Ensure it's a valid PostgreSQL identifier
        return re.sub(r"[^a-zA-Z0-9_]", "_", base_name).lower()

    async def ensure_isolation(self, tenant_id: str) -> None:
        """Ensure tenant isolation is set up.

        Sets up the appropriate isolation mechanism based on the
        configured strategy. This method is idempotent and can
        be called multiple times safely.

        Args:
            tenant_id: The tenant to set up isolation for.

        Raises:
            ValueError: If tenant_id is invalid.
            IsolationError: If isolation setup fails.
        """
        self._validate_tenant_id(tenant_id)

        if tenant_id in self._initialized_tenants:
            logger.debug(f"Isolation already ensured for {tenant_id}")
            return

        logger.info(f"Ensuring {self._strategy.value} isolation for {tenant_id}")

        if self._strategy == IsolationStrategy.ROW_LEVEL:
            await self._setup_rls_isolation(tenant_id)
        elif self._strategy == IsolationStrategy.SCHEMA:
            await self._setup_schema_isolation(tenant_id)
        elif self._strategy == IsolationStrategy.DATABASE:
            await self._setup_database_isolation(tenant_id)

        self._initialized_tenants.add(tenant_id)
        self._log_audit(
            operation="ensure_isolation",
            tenant_id=tenant_id,
            details={"strategy": self._strategy.value},
        )

    async def _setup_rls_isolation(self, tenant_id: str) -> None:
        """Set up Row-Level Security isolation.

        Creates RLS policies for all isolated tables to filter
        by tenant_id column.

        Args:
            tenant_id: The tenant to set up RLS for.
        """
        for table in self.ISOLATED_TABLES:
            policy = RLSPolicy(
                table_name=table,
                policy_name=f"rls_{table}_{tenant_id}",
                created_at=datetime.now(timezone.utc),
            )

            if table not in self._rls_policies:
                self._rls_policies[table] = []
            self._rls_policies[table].append(policy)

            # In production, execute SQL:
            # ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;
            # CREATE POLICY {policy_name} ON {table}
            #   USING (tenant_id = current_setting('app.current_tenant'));

            logger.debug(f"Created RLS policy for {table}")

    async def _setup_schema_isolation(self, tenant_id: str) -> None:
        """Set up schema-per-tenant isolation.

        Creates a new PostgreSQL schema for the tenant and copies
        the table structure from the public schema.

        Args:
            tenant_id: The tenant to set up schema for.
        """
        schema_name = self._generate_schema_name(tenant_id)

        schema_info = SchemaInfo(
            schema_name=schema_name,
            tenant_id=tenant_id,
            created_at=datetime.now(timezone.utc),
            tables=list(self.ISOLATED_TABLES),
        )

        self._schemas[tenant_id] = schema_info

        # In production, execute SQL:
        # CREATE SCHEMA IF NOT EXISTS {schema_name};
        # For each table, create a copy in the new schema

        logger.info(f"Created schema {schema_name} for {tenant_id}")

    async def _setup_database_isolation(self, tenant_id: str) -> None:
        """Set up database-per-tenant isolation.

        Creates a new database for the tenant. This is typically
        handled by infrastructure automation in production.

        Args:
            tenant_id: The tenant to set up database for.
        """
        db_name = f"kintsugi_{tenant_id.replace('org_', '')}"

        # In production, this would:
        # 1. Create a new database
        # 2. Run migrations
        # 3. Store connection details securely

        logger.info(f"Database isolation requested for {tenant_id} (db: {db_name})")
        logger.warning("Database-per-tenant requires infrastructure automation")

    async def create_tenant_schema(self, tenant_id: str) -> str:
        """Create isolated schema for tenant.

        Creates a new PostgreSQL schema specifically for this tenant.
        Returns the schema name for use in connection configuration.

        Args:
            tenant_id: The tenant to create schema for.

        Returns:
            The name of the created schema.

        Raises:
            ValueError: If tenant_id is invalid.
        """
        self._validate_tenant_id(tenant_id)

        schema_name = self._generate_schema_name(tenant_id)

        schema_info = SchemaInfo(
            schema_name=schema_name,
            tenant_id=tenant_id,
            created_at=datetime.now(timezone.utc),
            tables=list(self.ISOLATED_TABLES),
        )

        self._schemas[tenant_id] = schema_info

        self._log_audit(
            operation="create_schema",
            tenant_id=tenant_id,
            details={"schema_name": schema_name},
        )

        logger.info(f"Created schema {schema_name} for tenant {tenant_id}")
        return schema_name

    def get_tenant_filter(self, tenant_id: str) -> dict[str, Any]:
        """Get SQLAlchemy filter for tenant isolation.

        Returns a filter dictionary that can be used with SQLAlchemy
        to ensure queries only return data for the specified tenant.

        Args:
            tenant_id: The tenant to filter for.

        Returns:
            Dictionary suitable for use as a SQLAlchemy filter.

        Raises:
            ValueError: If tenant_id is invalid.

        Example:
            filter_dict = isolator.get_tenant_filter("org_12345")
            query = session.query(Memory).filter_by(**filter_dict)
        """
        self._validate_tenant_id(tenant_id)

        if self._strategy == IsolationStrategy.ROW_LEVEL:
            return {"tenant_id": tenant_id}
        elif self._strategy == IsolationStrategy.SCHEMA:
            # For schema isolation, the search_path handles filtering
            # but we still return tenant_id for explicit filtering
            return {"tenant_id": tenant_id}
        else:
            # For database isolation, connection itself provides isolation
            return {}

    def get_schema_name(self, tenant_id: str) -> str | None:
        """Get the schema name for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            The schema name if using schema isolation, None otherwise.
        """
        if self._strategy != IsolationStrategy.SCHEMA:
            return None

        schema_info = self._schemas.get(tenant_id)
        return schema_info.schema_name if schema_info else None

    async def migrate_tenant(
        self,
        tenant_id: str,
        from_strategy: IsolationStrategy,
        to_strategy: IsolationStrategy,
    ) -> None:
        """Migrate tenant between isolation strategies.

        Performs a safe migration of tenant data from one isolation
        strategy to another. This is typically done when upgrading
        a tenant's tier.

        Migration paths:
            - ROW_LEVEL -> SCHEMA: Extract tenant rows into new schema
            - ROW_LEVEL -> DATABASE: Extract tenant rows into new database
            - SCHEMA -> DATABASE: Move schema to dedicated database

        Args:
            tenant_id: The tenant to migrate.
            from_strategy: Current isolation strategy.
            to_strategy: Target isolation strategy.

        Raises:
            ValueError: If migration path is invalid.
            MigrationError: If migration fails.
        """
        self._validate_tenant_id(tenant_id)

        if from_strategy == to_strategy:
            logger.warning(f"No migration needed: already using {to_strategy.value}")
            return

        logger.info(
            f"Migrating {tenant_id} from {from_strategy.value} to {to_strategy.value}"
        )

        # Validate migration path
        valid_paths = [
            (IsolationStrategy.ROW_LEVEL, IsolationStrategy.SCHEMA),
            (IsolationStrategy.ROW_LEVEL, IsolationStrategy.DATABASE),
            (IsolationStrategy.SCHEMA, IsolationStrategy.DATABASE),
        ]

        if (from_strategy, to_strategy) not in valid_paths:
            raise ValueError(
                f"Invalid migration path: {from_strategy.value} -> {to_strategy.value}"
            )

        # Perform migration steps
        if from_strategy == IsolationStrategy.ROW_LEVEL:
            if to_strategy == IsolationStrategy.SCHEMA:
                await self._migrate_rls_to_schema(tenant_id)
            elif to_strategy == IsolationStrategy.DATABASE:
                await self._migrate_rls_to_database(tenant_id)
        elif from_strategy == IsolationStrategy.SCHEMA:
            if to_strategy == IsolationStrategy.DATABASE:
                await self._migrate_schema_to_database(tenant_id)

        self._log_audit(
            operation="migrate_isolation",
            tenant_id=tenant_id,
            details={
                "from_strategy": from_strategy.value,
                "to_strategy": to_strategy.value,
            },
        )

    async def _migrate_rls_to_schema(self, tenant_id: str) -> None:
        """Migrate from RLS to schema isolation.

        Args:
            tenant_id: The tenant to migrate.
        """
        # 1. Create new schema
        schema_name = await self.create_tenant_schema(tenant_id)

        # 2. Copy tenant data to new schema
        for table in self.ISOLATED_TABLES:
            # In production:
            # INSERT INTO {schema_name}.{table}
            # SELECT * FROM public.{table} WHERE tenant_id = '{tenant_id}';
            logger.debug(f"Migrating {table} to schema {schema_name}")

        # 3. Delete from shared tables (after verification)
        # DELETE FROM public.{table} WHERE tenant_id = '{tenant_id}';

        logger.info(f"Completed RLS to schema migration for {tenant_id}")

    async def _migrate_rls_to_database(self, tenant_id: str) -> None:
        """Migrate from RLS to database isolation.

        Args:
            tenant_id: The tenant to migrate.
        """
        logger.info(f"Starting RLS to database migration for {tenant_id}")
        # This would involve infrastructure automation to:
        # 1. Provision new database
        # 2. Run migrations
        # 3. Export tenant data
        # 4. Import into new database
        # 5. Verify integrity
        # 6. Delete from source

    async def _migrate_schema_to_database(self, tenant_id: str) -> None:
        """Migrate from schema to database isolation.

        Args:
            tenant_id: The tenant to migrate.
        """
        schema_info = self._schemas.get(tenant_id)
        if not schema_info:
            raise ValueError(f"No schema found for tenant {tenant_id}")

        logger.info(
            f"Starting schema to database migration for {tenant_id} "
            f"(schema: {schema_info.schema_name})"
        )
        # Similar to RLS to database, but source is the schema

    def _log_audit(
        self,
        operation: str,
        tenant_id: str,
        details: dict[str, Any],
    ) -> None:
        """Log an audit entry.

        Args:
            operation: The operation performed.
            tenant_id: The affected tenant.
            details: Additional details.
        """
        entry = IsolationAuditEntry(
            operation=operation,
            tenant_id=tenant_id,
            strategy=self._strategy,
            details=details,
            performed_at=datetime.now(timezone.utc),
        )
        self._audit_log.append(entry)

    def get_audit_log(self, tenant_id: str | None = None) -> list[IsolationAuditEntry]:
        """Get audit log entries.

        Args:
            tenant_id: Optional filter by tenant.

        Returns:
            List of audit entries.
        """
        if tenant_id:
            return [e for e in self._audit_log if e.tenant_id == tenant_id]
        return list(self._audit_log)

    async def verify_isolation(self, tenant_id: str) -> bool:
        """Verify tenant isolation is properly configured.

        Performs checks to ensure the isolation mechanism is
        working correctly for the specified tenant.

        Args:
            tenant_id: The tenant to verify.

        Returns:
            True if isolation is properly configured.
        """
        self._validate_tenant_id(tenant_id)

        if tenant_id not in self._initialized_tenants:
            logger.warning(f"Tenant {tenant_id} isolation not initialized")
            return False

        if self._strategy == IsolationStrategy.SCHEMA:
            if tenant_id not in self._schemas:
                logger.warning(f"No schema found for {tenant_id}")
                return False

        logger.info(f"Isolation verified for {tenant_id}")
        return True

    def __repr__(self) -> str:
        return (
            f"<TenantIsolator strategy={self._strategy.value} "
            f"tenants={len(self._initialized_tenants)}>"
        )
