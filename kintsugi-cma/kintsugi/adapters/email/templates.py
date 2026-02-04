"""
Email templates for Kintsugi CMA.

This module provides a template system for generating consistent,
professional email content. Templates support variable substitution
and can be customized per organization.

Features:
    - Default templates for common notification types
    - Variable substitution using Python string.Template
    - HTML and plain text versions
    - Custom template registration
    - Template validation

Template Variables:
    Templates use $variable_name syntax for substitution.
    Missing variables are replaced with empty strings by default.

Example:
    renderer = TemplateRenderer()

    # Render a grant reminder
    subject, body_text, body_html = renderer.render(
        "grant_reminder",
        grant_name="Community Impact Grant",
        deadline="March 15, 2024",
        days_remaining=7
    )

    # Add custom template
    renderer.add_template(EmailTemplate(
        name="custom_welcome",
        subject_template="Welcome to $org_name",
        body_text_template="Hello $name, welcome aboard!"
    ))
"""

import logging
from dataclasses import dataclass, field
from string import Template
from typing import Any

logger = logging.getLogger(__name__)


class TemplateError(Exception):
    """Error in template rendering."""
    pass


class TemplateNotFoundError(TemplateError):
    """Template not found."""
    pass


class TemplateValidationError(TemplateError):
    """Template failed validation."""
    pass


@dataclass
class EmailTemplate:
    """
    Represents an email template with subject and body.

    Templates support variable substitution using Python's
    string.Template syntax ($variable or ${variable}).

    Attributes:
        name: Unique template identifier
        subject_template: Template for email subject line
        body_text_template: Plain text body template
        body_html_template: Optional HTML body template
        description: Human-readable description
        required_vars: List of required variable names
        default_vars: Default values for optional variables
        category: Template category for organization

    Example:
        template = EmailTemplate(
            name="deadline_reminder",
            subject_template="Reminder: $grant_name deadline in $days days",
            body_text_template='''
            Dear Team,

            This is a reminder that $grant_name has a deadline
            coming up on $deadline.

            Best regards,
            Kintsugi
            ''',
            required_vars=["grant_name", "deadline", "days"]
        )
    """

    name: str
    subject_template: str
    body_text_template: str
    body_html_template: str | None = None
    description: str = ""
    required_vars: list[str] = field(default_factory=list)
    default_vars: dict[str, str] = field(default_factory=dict)
    category: str = "general"

    def __post_init__(self) -> None:
        """Validate template after initialization."""
        if not self.name:
            raise TemplateValidationError("Template name is required")

        if not self.subject_template:
            raise TemplateValidationError("subject_template is required")

        if not self.body_text_template:
            raise TemplateValidationError("body_text_template is required")

    def validate_vars(self, provided_vars: dict[str, Any]) -> list[str]:
        """
        Check if all required variables are provided.

        Args:
            provided_vars: Dictionary of provided variable values

        Returns:
            List of missing variable names
        """
        missing = []
        for var in self.required_vars:
            if var not in provided_vars and var not in self.default_vars:
                missing.append(var)
        return missing

    def get_all_vars(self) -> set[str]:
        """
        Extract all variable names from templates.

        Returns:
            Set of variable names used in templates
        """
        vars_found = set()

        for template_str in [
            self.subject_template,
            self.body_text_template,
            self.body_html_template or ""
        ]:
            # Find $var and ${var} patterns
            template = Template(template_str)
            # Use Template's pattern to find identifiers
            for match in template.pattern.finditer(template_str):
                name = match.group("named") or match.group("braced")
                if name:
                    vars_found.add(name)

        return vars_found


# =============================================================================
# Default Templates
# =============================================================================

GRANT_REMINDER_TEMPLATE = EmailTemplate(
    name="grant_reminder",
    subject_template="Grant Deadline Reminder: $grant_name - $days_remaining days",
    body_text_template='''Grant Deadline Reminder

Grant: $grant_name
Deadline: $deadline
Days Remaining: $days_remaining

$notes

Please ensure all required materials are submitted before the deadline.

If you have any questions about this grant opportunity, please reply to this email.

---
Sent by Kintsugi CMA
Your Grant Management Assistant''',
    body_html_template='''<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #4A90A4; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .deadline-box { background-color: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }
        .urgent { background-color: #f8d7da; border-color: #f5c6cb; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
        .btn { display: inline-block; padding: 10px 20px; background-color: #4A90A4; color: white; text-decoration: none; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Grant Deadline Reminder</h1>
        </div>
        <div class="content">
            <h2>$grant_name</h2>
            <div class="deadline-box">
                <strong>Deadline:</strong> $deadline<br>
                <strong>Days Remaining:</strong> $days_remaining
            </div>
            <p>$notes</p>
            <p>Please ensure all required materials are submitted before the deadline.</p>
            <p>If you have any questions about this grant opportunity, please reply to this email.</p>
        </div>
        <div class="footer">
            <p>Sent by Kintsugi CMA - Your Grant Management Assistant</p>
        </div>
    </div>
</body>
</html>''',
    description="Reminder for upcoming grant deadlines",
    required_vars=["grant_name", "deadline", "days_remaining"],
    default_vars={"notes": ""},
    category="grants"
)


PAIRING_CODE_TEMPLATE = EmailTemplate(
    name="pairing_code",
    subject_template="Your Kintsugi Pairing Code: $code",
    body_text_template='''Kintsugi CMA Pairing Code

Your pairing code is: $code

Use this code to connect your $platform account with your organization.

This code will expire in $expiration_minutes minutes.

IMPORTANT: Do not share this code with anyone. If you did not request
this code, please ignore this email.

---
Sent by Kintsugi CMA
Your Grant Management Assistant''',
    body_html_template='''<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #4A90A4; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .code-box { background-color: #e9ecef; border: 2px dashed #6c757d; padding: 20px; margin: 20px 0; text-align: center; font-size: 32px; font-family: monospace; letter-spacing: 8px; }
        .warning { background-color: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Kintsugi CMA Pairing Code</h1>
        </div>
        <div class="content">
            <p>Your pairing code is:</p>
            <div class="code-box">$code</div>
            <p>Use this code to connect your <strong>$platform</strong> account with your organization.</p>
            <p>This code will expire in <strong>$expiration_minutes minutes</strong>.</p>
            <div class="warning">
                <strong>IMPORTANT:</strong> Do not share this code with anyone. If you did not request this code, please ignore this email.
            </div>
        </div>
        <div class="footer">
            <p>Sent by Kintsugi CMA - Your Grant Management Assistant</p>
        </div>
    </div>
</body>
</html>''',
    description="Pairing code for account linking",
    required_vars=["code", "platform", "expiration_minutes"],
    category="system"
)


REPORT_DELIVERY_TEMPLATE = EmailTemplate(
    name="report_delivery",
    subject_template="$report_title",
    body_text_template='''$report_title

Please find the attached report for your review.

Report Type: $report_type
Generated: $generated_at

$summary

If you have any questions about this report, please reply to this email.

---
Sent by Kintsugi CMA
Your Grant Management Assistant''',
    body_html_template='''<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #4A90A4; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .info-box { background-color: #e9ecef; padding: 15px; margin: 15px 0; border-radius: 4px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>$report_title</h1>
        </div>
        <div class="content">
            <p>Please find the attached report for your review.</p>
            <div class="info-box">
                <strong>Report Type:</strong> $report_type<br>
                <strong>Generated:</strong> $generated_at
            </div>
            <p>$summary</p>
            <p>If you have any questions about this report, please reply to this email.</p>
        </div>
        <div class="footer">
            <p>Sent by Kintsugi CMA - Your Grant Management Assistant</p>
        </div>
    </div>
</body>
</html>''',
    description="Report delivery notification",
    required_vars=["report_title", "report_type"],
    default_vars={"generated_at": "", "summary": ""},
    category="reports"
)


WELCOME_TEMPLATE = EmailTemplate(
    name="welcome",
    subject_template="Welcome to Kintsugi CMA, $org_name!",
    body_text_template='''Welcome to Kintsugi CMA!

Hello $contact_name,

Your organization, $org_name, has been set up with Kintsugi CMA,
your intelligent grant management assistant.

Getting Started:
1. Connect your team's chat platform (Slack, Discord, or webchat)
2. Start asking questions about grants and deadlines
3. Set up notifications for important dates

Need help? Just reply to this email or visit our documentation.

Best regards,
The Kintsugi Team

---
Sent by Kintsugi CMA
Your Grant Management Assistant''',
    description="Welcome email for new organizations",
    required_vars=["org_name", "contact_name"],
    category="onboarding"
)


DEADLINE_URGENT_TEMPLATE = EmailTemplate(
    name="deadline_urgent",
    subject_template="URGENT: $grant_name deadline in $days_remaining days!",
    body_text_template='''URGENT DEADLINE ALERT

Grant: $grant_name
Deadline: $deadline
Days Remaining: $days_remaining

This is an urgent reminder that the deadline for $grant_name is
approaching very soon.

Required Actions:
$requirements

Please take immediate action to ensure all materials are submitted
before the deadline.

$notes

---
Sent by Kintsugi CMA
Your Grant Management Assistant''',
    body_html_template='''<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #dc3545; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .deadline-box { background-color: #f8d7da; border: 2px solid #dc3545; padding: 15px; margin: 15px 0; border-radius: 4px; }
        .requirements { background-color: #fff; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>URGENT DEADLINE ALERT</h1>
        </div>
        <div class="content">
            <h2>$grant_name</h2>
            <div class="deadline-box">
                <strong>Deadline:</strong> $deadline<br>
                <strong>Days Remaining:</strong> $days_remaining
            </div>
            <p>This is an urgent reminder that the deadline for <strong>$grant_name</strong> is approaching very soon.</p>
            <div class="requirements">
                <strong>Required Actions:</strong><br>
                $requirements
            </div>
            <p>$notes</p>
        </div>
        <div class="footer">
            <p>Sent by Kintsugi CMA - Your Grant Management Assistant</p>
        </div>
    </div>
</body>
</html>''',
    description="Urgent deadline alert (7 days or less)",
    required_vars=["grant_name", "deadline", "days_remaining"],
    default_vars={"requirements": "", "notes": ""},
    category="grants"
)


GRANT_APPROVED_TEMPLATE = EmailTemplate(
    name="grant_approved",
    subject_template="Congratulations! $grant_name has been approved",
    body_text_template='''Grant Approval Notification

Great news! Your application for $grant_name has been approved!

Grant Details:
- Grant Name: $grant_name
- Funder: $funder_name
- Award Amount: $amount
- Award Date: $award_date

Next Steps:
$next_steps

Congratulations on this achievement!

---
Sent by Kintsugi CMA
Your Grant Management Assistant''',
    description="Notification of grant approval",
    required_vars=["grant_name", "funder_name", "amount", "award_date"],
    default_vars={"next_steps": ""},
    category="grants"
)


# Collect all default templates
DEFAULT_TEMPLATES = {
    GRANT_REMINDER_TEMPLATE.name: GRANT_REMINDER_TEMPLATE,
    PAIRING_CODE_TEMPLATE.name: PAIRING_CODE_TEMPLATE,
    REPORT_DELIVERY_TEMPLATE.name: REPORT_DELIVERY_TEMPLATE,
    WELCOME_TEMPLATE.name: WELCOME_TEMPLATE,
    DEADLINE_URGENT_TEMPLATE.name: DEADLINE_URGENT_TEMPLATE,
    GRANT_APPROVED_TEMPLATE.name: GRANT_APPROVED_TEMPLATE,
}


class SafeTemplate(Template):
    """
    Template subclass that handles missing variables gracefully.

    Instead of raising KeyError for missing variables, replaces
    them with empty strings or a specified default.
    """

    def safe_substitute_with_default(
        self,
        mapping: dict[str, Any],
        default: str = ""
    ) -> str:
        """
        Substitute variables, using default for missing values.

        Args:
            mapping: Dictionary of variable values
            default: Default value for missing variables

        Returns:
            Substituted string
        """
        # Convert all values to strings
        string_mapping = {
            k: str(v) if v is not None else default
            for k, v in mapping.items()
        }

        # Use safe_substitute which replaces missing with $var
        result = self.safe_substitute(string_mapping)

        # Replace any remaining $var patterns with default
        import re
        result = re.sub(r'\$\{?\w+\}?', default, result)

        return result


class TemplateRenderer:
    """
    Render email templates with variable substitution.

    This class manages a collection of email templates and provides
    methods for rendering them with provided variable values.

    Attributes:
        templates: Dictionary of available templates

    Thread Safety:
        Template rendering is thread-safe. Template modification
        (add/remove) is NOT thread-safe.

    Example:
        renderer = TemplateRenderer()

        # Render default template
        subject, body, html = renderer.render(
            "grant_reminder",
            grant_name="Community Grant",
            deadline="March 15, 2024",
            days_remaining=7
        )

        # Add custom template
        renderer.add_template(custom_template)

        # List available templates
        for name in renderer.list_templates():
            print(name)
    """

    def __init__(
        self,
        custom_templates: dict[str, EmailTemplate] | None = None
    ):
        """
        Initialize the template renderer.

        Args:
            custom_templates: Optional dictionary of custom templates
                to add to defaults
        """
        # Start with default templates
        self._templates = dict(DEFAULT_TEMPLATES)

        # Add custom templates
        if custom_templates:
            self._templates.update(custom_templates)

        logger.debug(
            "TemplateRenderer initialized with %d templates",
            len(self._templates)
        )

    def render(
        self,
        template_name: str,
        strict: bool = False,
        **kwargs: Any
    ) -> tuple[str, str, str | None]:
        """
        Render a template with provided variables.

        Args:
            template_name: Name of the template to render
            strict: If True, raise error for missing required variables
            **kwargs: Variable values to substitute

        Returns:
            Tuple of (subject, body_text, body_html or None)

        Raises:
            TemplateNotFoundError: If template doesn't exist
            TemplateValidationError: If strict and required vars missing
        """
        template = self._templates.get(template_name)
        if not template:
            raise TemplateNotFoundError(f"Template '{template_name}' not found")

        # Merge defaults with provided values
        variables = dict(template.default_vars)
        variables.update(kwargs)

        # Check required variables if strict
        if strict:
            missing = template.validate_vars(variables)
            if missing:
                raise TemplateValidationError(
                    f"Missing required variables: {', '.join(missing)}"
                )

        # Render subject
        subject_tmpl = SafeTemplate(template.subject_template)
        subject = subject_tmpl.safe_substitute_with_default(variables)

        # Render body text
        body_tmpl = SafeTemplate(template.body_text_template)
        body_text = body_tmpl.safe_substitute_with_default(variables)

        # Render HTML if available
        body_html = None
        if template.body_html_template:
            html_tmpl = SafeTemplate(template.body_html_template)
            body_html = html_tmpl.safe_substitute_with_default(variables)

        return subject, body_text.strip(), body_html

    def add_template(self, template: EmailTemplate) -> None:
        """
        Add or replace a template.

        Args:
            template: Template to add

        Raises:
            TemplateValidationError: If template is invalid
        """
        if not isinstance(template, EmailTemplate):
            raise TemplateValidationError(
                "template must be an EmailTemplate instance"
            )

        self._templates[template.name] = template
        logger.info("Added template: %s", template.name)

    def remove_template(self, name: str) -> bool:
        """
        Remove a template by name.

        Args:
            name: Template name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._templates:
            del self._templates[name]
            logger.info("Removed template: %s", name)
            return True
        return False

    def get_template(self, name: str) -> EmailTemplate | None:
        """
        Get a template by name.

        Args:
            name: Template name

        Returns:
            Template if found, None otherwise
        """
        return self._templates.get(name)

    def list_templates(self, category: str | None = None) -> list[str]:
        """
        List available template names.

        Args:
            category: Optional category filter

        Returns:
            List of template names
        """
        if category:
            return [
                name for name, tmpl in self._templates.items()
                if tmpl.category == category
            ]
        return list(self._templates.keys())

    def list_categories(self) -> list[str]:
        """
        List available template categories.

        Returns:
            List of unique category names
        """
        categories = set(t.category for t in self._templates.values())
        return sorted(categories)

    def validate_template(
        self,
        template: EmailTemplate,
        test_vars: dict[str, Any] | None = None
    ) -> list[str]:
        """
        Validate a template can be rendered.

        Args:
            template: Template to validate
            test_vars: Optional test variables to use

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            # Check structure
            if not template.subject_template:
                errors.append("Missing subject_template")

            if not template.body_text_template:
                errors.append("Missing body_text_template")

            # Try to render with test vars
            if test_vars:
                missing = template.validate_vars(test_vars)
                if missing:
                    errors.append(f"Missing required vars: {missing}")

                # Try actual render
                try:
                    self.render(template.name, strict=True, **test_vars)
                except Exception as e:
                    errors.append(f"Render error: {e}")

        except Exception as e:
            errors.append(f"Validation error: {e}")

        return errors

    def __repr__(self) -> str:
        return f"<TemplateRenderer templates={len(self._templates)}>"
