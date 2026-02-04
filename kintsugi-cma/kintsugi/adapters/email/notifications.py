"""
Email notification system for Kintsugi CMA.

This module provides scheduled notification capabilities for grant
deadlines, report deliveries, and other time-sensitive communications.
It integrates with the EmailAdapter to send notifications at appropriate
times.

Features:
    - Grant deadline reminders with configurable lead times
    - Report delivery via email with attachments
    - Scheduled notification queue
    - Recurring reminder support
    - Deadline tracking and listing

Example:
    manager = NotificationManager(email_adapter)

    # Send immediate reminder
    notification = GrantDeadlineNotification(
        grant_name="Community Impact Grant",
        deadline=datetime(2024, 3, 15),
        days_remaining=7,
        amount=50000.00
    )
    await manager.send_grant_reminder(notification, ["team@nonprofit.org"])

    # Schedule for later
    manager.schedule_reminder(
        notification,
        send_at=datetime.now() + timedelta(days=1),
        recipients=["director@nonprofit.org"]
    )
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .adapter import EmailAdapter
from .templates import TemplateRenderer, EmailTemplate

from ..shared.base import AdapterResponse

logger = logging.getLogger(__name__)


@dataclass
class GrantDeadlineNotification:
    """
    Represents a grant deadline notification.

    Used for sending reminders about upcoming grant deadlines,
    with all relevant information about the grant.

    Attributes:
        grant_name: Name of the grant opportunity
        deadline: Deadline date and time
        days_remaining: Days until deadline (computed or provided)
        grant_url: Optional URL to grant application
        amount: Grant amount (if known)
        notes: Additional notes or instructions
        funder_name: Name of the funding organization
        grant_id: Internal grant identifier
        requirements: List of remaining requirements
        priority: Notification priority (1=urgent, 5=low)
        metadata: Additional grant-specific data

    Example:
        notification = GrantDeadlineNotification(
            grant_name="Environmental Justice Grant",
            deadline=datetime(2024, 6, 30, 17, 0),
            days_remaining=14,
            grant_url="https://foundation.org/apply",
            amount=75000.00,
            notes="Requires budget narrative",
            priority=2
        )
    """

    grant_name: str
    deadline: datetime
    days_remaining: int
    grant_url: str | None = None
    amount: float | None = None
    notes: str | None = None
    funder_name: str | None = None
    grant_id: str | None = None
    requirements: list[str] = field(default_factory=list)
    priority: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and process notification data."""
        if self.priority < 1 or self.priority > 5:
            raise ValueError("priority must be between 1 and 5")

        # Ensure deadline is timezone-aware
        if self.deadline.tzinfo is None:
            self.deadline = self.deadline.replace(tzinfo=timezone.utc)

    @property
    def is_urgent(self) -> bool:
        """Check if notification is urgent (7 days or less)."""
        return self.days_remaining <= 7

    @property
    def is_overdue(self) -> bool:
        """Check if deadline has passed."""
        return datetime.now(timezone.utc) > self.deadline

    @property
    def formatted_amount(self) -> str | None:
        """Get formatted currency amount."""
        if self.amount is None:
            return None
        return f"${self.amount:,.2f}"

    @property
    def formatted_deadline(self) -> str:
        """Get human-readable deadline string."""
        return self.deadline.strftime("%B %d, %Y at %I:%M %p %Z")

    def to_template_vars(self) -> dict[str, Any]:
        """
        Convert to template variable dictionary.

        Returns:
            Dictionary suitable for template rendering
        """
        return {
            "grant_name": self.grant_name,
            "deadline": self.formatted_deadline,
            "days_remaining": self.days_remaining,
            "grant_url": self.grant_url or "",
            "amount": self.formatted_amount or "Not specified",
            "notes": self.notes or "",
            "funder_name": self.funder_name or "Unknown",
            "requirements": "\n".join(f"- {r}" for r in self.requirements),
            "priority": self.priority,
            "is_urgent": self.is_urgent,
        }


@dataclass
class ReportDelivery:
    """
    Represents a report to be delivered via email.

    Used for sending generated reports such as grant status summaries,
    financial reports, or deadline calendars.

    Attributes:
        report_type: Type of report (e.g., "grant_status", "financial")
        report_title: Title for the email
        recipients: List of recipient email addresses
        attachment_path: Path to report file attachment
        body_text: Email body content
        cc_recipients: CC recipients
        attachment_name: Override filename for attachment
        include_summary: Include summary in email body
        send_immediately: Whether to send immediately or queue
        metadata: Additional report metadata

    Example:
        delivery = ReportDelivery(
            report_type="grant_status",
            report_title="Q4 Grant Status Report",
            recipients=["board@nonprofit.org"],
            attachment_path="/reports/q4_status.pdf",
            body_text="Please find the Q4 grant status report attached."
        )
    """

    report_type: str
    report_title: str
    recipients: list[str]
    attachment_path: str | None = None
    body_text: str = ""
    cc_recipients: list[str] = field(default_factory=list)
    attachment_name: str | None = None
    include_summary: bool = True
    send_immediately: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate delivery configuration."""
        if not self.recipients:
            raise ValueError("At least one recipient is required")

        if not self.report_title:
            raise ValueError("report_title is required")

    @property
    def has_attachment(self) -> bool:
        """Check if delivery includes an attachment."""
        return self.attachment_path is not None

    def get_attachment_name(self) -> str | None:
        """
        Get the filename for the attachment.

        Returns:
            Filename to use for attachment, or None if no attachment
        """
        if not self.attachment_path:
            return None

        if self.attachment_name:
            return self.attachment_name

        return Path(self.attachment_path).name


@dataclass
class ScheduledNotification:
    """
    A notification scheduled for future delivery.

    Internal class used by NotificationManager to track
    scheduled notifications.

    Attributes:
        id: Unique identifier for this scheduled notification
        send_at: When to send the notification
        notification: The notification or delivery object
        recipients: List of recipient email addresses
        status: Current status (pending, sent, failed, cancelled)
        created_at: When this was scheduled
        sent_at: When it was actually sent (if sent)
        error: Error message if failed
        recurring: Whether this is a recurring notification
        recurrence_interval: Interval for recurring notifications
    """

    id: str
    send_at: datetime
    notification: GrantDeadlineNotification | ReportDelivery
    recipients: list[str]
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: datetime | None = None
    error: str | None = None
    recurring: bool = False
    recurrence_interval: timedelta | None = None

    @property
    def is_due(self) -> bool:
        """Check if notification is due to be sent."""
        return (
            self.status == "pending" and
            datetime.now(timezone.utc) >= self.send_at
        )

    @property
    def is_grant_reminder(self) -> bool:
        """Check if this is a grant deadline notification."""
        return isinstance(self.notification, GrantDeadlineNotification)


class NotificationManager:
    """
    Manages scheduled email notifications.

    This class handles the scheduling, queuing, and sending of
    email notifications including grant deadline reminders and
    report deliveries.

    Attributes:
        adapter: Email adapter for sending notifications

    Thread Safety:
        This class is NOT thread-safe. Use appropriate locking
        for concurrent access.

    Example:
        manager = NotificationManager(adapter)

        # Send immediate notification
        await manager.send_grant_reminder(notification, recipients)

        # Schedule for later
        schedule_id = manager.schedule_reminder(
            notification,
            send_at=datetime.now() + timedelta(days=1),
            recipients=recipients
        )

        # Start background processor
        await manager.start_scheduler()
    """

    def __init__(
        self,
        adapter: EmailAdapter,
        template_renderer: TemplateRenderer | None = None
    ):
        """
        Initialize the notification manager.

        Args:
            adapter: Email adapter for sending
            template_renderer: Optional custom template renderer
        """
        self._adapter = adapter
        self._renderer = template_renderer or TemplateRenderer()
        self._scheduled: dict[str, ScheduledNotification] = {}
        self._running = False
        self._scheduler_task: asyncio.Task | None = None

        # Callbacks
        self._on_sent: Callable[[ScheduledNotification], None] | None = None
        self._on_failed: Callable[[ScheduledNotification, Exception], None] | None = None

        logger.info("NotificationManager initialized")

    @property
    def scheduled_count(self) -> int:
        """Get count of pending scheduled notifications."""
        return sum(
            1 for n in self._scheduled.values()
            if n.status == "pending"
        )

    async def send_grant_reminder(
        self,
        notification: GrantDeadlineNotification,
        recipients: list[str],
        template_name: str = "grant_reminder"
    ) -> str:
        """
        Send a grant deadline reminder immediately.

        Args:
            notification: Grant deadline notification details
            recipients: List of recipient email addresses
            template_name: Template to use for formatting

        Returns:
            Message ID of sent email

        Raises:
            ValueError: If notification is invalid
            SendError: If sending fails
        """
        if notification.is_overdue:
            logger.warning(
                "Sending reminder for overdue grant: %s",
                notification.grant_name
            )

        # Render template
        template_vars = notification.to_template_vars()
        subject, body_text, body_html = self._renderer.render(
            template_name,
            **template_vars
        )

        # Build response
        response = AdapterResponse(
            content=body_text,
            metadata={
                "subject": subject,
                "html_body": body_html,
                "notification_type": "grant_reminder",
                "grant_name": notification.grant_name,
            }
        )

        # Send to all recipients
        message_ids = []
        for recipient in recipients:
            try:
                msg_id = await self._adapter.send_email(
                    to=recipient,
                    response=response,
                    subject=subject
                )
                message_ids.append(msg_id)
                logger.info(
                    "Sent grant reminder to %s for %s",
                    recipient,
                    notification.grant_name
                )
            except Exception as e:
                logger.error(
                    "Failed to send reminder to %s: %s",
                    recipient, e
                )

        # Return first message ID (or generate one if all failed)
        return message_ids[0] if message_ids else f"<failed-{uuid.uuid4()}@kintsugi>"

    async def send_report(self, delivery: ReportDelivery) -> str:
        """
        Send a report via email.

        Args:
            delivery: Report delivery configuration

        Returns:
            Message ID of sent email

        Raises:
            ValueError: If delivery is invalid
            SendError: If sending fails
        """
        # Load attachment if present
        attachments = []
        if delivery.has_attachment:
            try:
                attachment_path = Path(delivery.attachment_path)
                with open(attachment_path, "rb") as f:
                    content = f.read()

                attachments.append({
                    "filename": delivery.get_attachment_name(),
                    "content": content,
                    "content_type": self._guess_mime_type(attachment_path.suffix)
                })
            except Exception as e:
                logger.error("Failed to load attachment: %s", e)
                raise ValueError(f"Cannot load attachment: {e}") from e

        # Render template if available, otherwise use provided body
        if delivery.body_text:
            body_text = delivery.body_text
            subject = delivery.report_title
        else:
            template_vars = {
                "report_type": delivery.report_type,
                "report_title": delivery.report_title,
                "has_attachment": delivery.has_attachment,
            }
            template_vars.update(delivery.metadata)

            subject, body_text, _ = self._renderer.render(
                "report_delivery",
                **template_vars
            )

        # Build response
        response = AdapterResponse(
            content=body_text,
            attachments=attachments,
            metadata={
                "subject": subject,
                "notification_type": "report_delivery",
                "report_type": delivery.report_type,
            }
        )

        # Send to all recipients
        message_ids = []
        for recipient in delivery.recipients:
            try:
                msg_id = await self._adapter.send_email(
                    to=recipient,
                    response=response,
                    subject=subject,
                    cc=delivery.cc_recipients if recipient == delivery.recipients[0] else None
                )
                message_ids.append(msg_id)
                logger.info(
                    "Sent report '%s' to %s",
                    delivery.report_title,
                    recipient
                )
            except Exception as e:
                logger.error(
                    "Failed to send report to %s: %s",
                    recipient, e
                )

        return message_ids[0] if message_ids else f"<failed-{uuid.uuid4()}@kintsugi>"

    def schedule_reminder(
        self,
        notification: GrantDeadlineNotification,
        send_at: datetime,
        recipients: list[str],
        recurring: bool = False,
        recurrence_interval: timedelta | None = None
    ) -> str:
        """
        Schedule a grant reminder for future delivery.

        Args:
            notification: Grant deadline notification
            send_at: When to send the reminder
            recipients: List of recipient email addresses
            recurring: Whether to repeat the notification
            recurrence_interval: Interval for recurring notifications

        Returns:
            Unique ID for the scheduled notification
        """
        # Ensure timezone-aware
        if send_at.tzinfo is None:
            send_at = send_at.replace(tzinfo=timezone.utc)

        schedule_id = str(uuid.uuid4())

        scheduled = ScheduledNotification(
            id=schedule_id,
            send_at=send_at,
            notification=notification,
            recipients=recipients,
            recurring=recurring,
            recurrence_interval=recurrence_interval
        )

        self._scheduled[schedule_id] = scheduled

        logger.info(
            "Scheduled reminder %s for %s at %s",
            schedule_id,
            notification.grant_name,
            send_at.isoformat()
        )

        return schedule_id

    def schedule_report(
        self,
        delivery: ReportDelivery,
        send_at: datetime
    ) -> str:
        """
        Schedule a report for future delivery.

        Args:
            delivery: Report delivery configuration
            send_at: When to send the report

        Returns:
            Unique ID for the scheduled delivery
        """
        if send_at.tzinfo is None:
            send_at = send_at.replace(tzinfo=timezone.utc)

        schedule_id = str(uuid.uuid4())

        scheduled = ScheduledNotification(
            id=schedule_id,
            send_at=send_at,
            notification=delivery,
            recipients=delivery.recipients
        )

        self._scheduled[schedule_id] = scheduled

        logger.info(
            "Scheduled report '%s' for %s",
            delivery.report_title,
            send_at.isoformat()
        )

        return schedule_id

    def cancel_scheduled(self, schedule_id: str) -> bool:
        """
        Cancel a scheduled notification.

        Args:
            schedule_id: ID of the scheduled notification

        Returns:
            True if cancelled, False if not found or already sent
        """
        scheduled = self._scheduled.get(schedule_id)
        if not scheduled:
            return False

        if scheduled.status != "pending":
            return False

        scheduled.status = "cancelled"
        logger.info("Cancelled scheduled notification %s", schedule_id)
        return True

    async def check_and_send_scheduled(self) -> int:
        """
        Check for due notifications and send them.

        Returns:
            Number of notifications sent
        """
        sent_count = 0

        for schedule_id, scheduled in list(self._scheduled.items()):
            if not scheduled.is_due:
                continue

            try:
                if scheduled.is_grant_reminder:
                    await self.send_grant_reminder(
                        scheduled.notification,
                        scheduled.recipients
                    )
                else:
                    await self.send_report(scheduled.notification)

                scheduled.status = "sent"
                scheduled.sent_at = datetime.now(timezone.utc)
                sent_count += 1

                if self._on_sent:
                    self._on_sent(scheduled)

                # Handle recurring notifications
                if scheduled.recurring and scheduled.recurrence_interval:
                    new_send_at = scheduled.send_at + scheduled.recurrence_interval
                    self.schedule_reminder(
                        scheduled.notification,
                        new_send_at,
                        scheduled.recipients,
                        recurring=True,
                        recurrence_interval=scheduled.recurrence_interval
                    )

            except Exception as e:
                scheduled.status = "failed"
                scheduled.error = str(e)
                logger.error(
                    "Failed to send scheduled notification %s: %s",
                    schedule_id, e
                )

                if self._on_failed:
                    self._on_failed(scheduled, e)

        return sent_count

    def get_upcoming_deadlines(
        self,
        days: int = 30
    ) -> list[GrantDeadlineNotification]:
        """
        Get upcoming grant deadlines from scheduled notifications.

        Args:
            days: Number of days to look ahead

        Returns:
            List of upcoming grant deadline notifications
        """
        cutoff = datetime.now(timezone.utc) + timedelta(days=days)

        deadlines = []
        for scheduled in self._scheduled.values():
            if scheduled.status != "pending":
                continue

            if not scheduled.is_grant_reminder:
                continue

            notification = scheduled.notification
            if notification.deadline <= cutoff:
                deadlines.append(notification)

        # Sort by deadline
        deadlines.sort(key=lambda n: n.deadline)
        return deadlines

    def get_scheduled(self, schedule_id: str) -> ScheduledNotification | None:
        """Get a scheduled notification by ID."""
        return self._scheduled.get(schedule_id)

    def list_scheduled(
        self,
        status: str | None = None
    ) -> list[ScheduledNotification]:
        """
        List scheduled notifications.

        Args:
            status: Optional status filter

        Returns:
            List of scheduled notifications
        """
        notifications = list(self._scheduled.values())

        if status:
            notifications = [n for n in notifications if n.status == status]

        # Sort by send_at
        notifications.sort(key=lambda n: n.send_at)
        return notifications

    async def start_scheduler(
        self,
        check_interval_seconds: int = 60
    ) -> None:
        """
        Start the background scheduler for processing notifications.

        Args:
            check_interval_seconds: How often to check for due notifications
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True

        async def scheduler_loop():
            while self._running:
                try:
                    sent = await self.check_and_send_scheduled()
                    if sent > 0:
                        logger.info("Sent %d scheduled notifications", sent)
                except Exception as e:
                    logger.error("Scheduler error: %s", e)

                await asyncio.sleep(check_interval_seconds)

        self._scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info(
            "Started notification scheduler (interval: %ds)",
            check_interval_seconds
        )

    async def stop_scheduler(self) -> None:
        """Stop the background scheduler."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
            logger.info("Stopped notification scheduler")

    def set_callbacks(
        self,
        on_sent: Callable[[ScheduledNotification], None] | None = None,
        on_failed: Callable[[ScheduledNotification, Exception], None] | None = None
    ) -> None:
        """
        Set callback functions for notification events.

        Args:
            on_sent: Called when a notification is successfully sent
            on_failed: Called when a notification fails to send
        """
        self._on_sent = on_sent
        self._on_failed = on_failed

    def _guess_mime_type(self, extension: str) -> str:
        """Guess MIME type from file extension."""
        mime_types = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }
        return mime_types.get(extension.lower(), "application/octet-stream")

    def __repr__(self) -> str:
        return (
            f"<NotificationManager scheduled={self.scheduled_count} "
            f"running={self._running}>"
        )
