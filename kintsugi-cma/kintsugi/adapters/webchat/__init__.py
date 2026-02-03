"""WebChat adapter for Kintsugi CMA.

This module provides an embeddable chat widget adapter for integrating
Kintsugi's cognitive memory architecture into websites and web applications.
The adapter supports WebSocket-based real-time communication, customizable
themes, and flexible embedding options.

Example usage:

    from kintsugi.adapters.webchat import (
        WebChatConfig,
        WebChatHandler,
        WebChatMessageType,
        WebChatSession,
        WidgetConfigGenerator,
        WidgetTheme,
        WidgetPosition,
    )

    # Create configuration
    config = WebChatConfig(
        org_id="your-org-uuid",
        widget_title="Customer Support",
        primary_color="#007AFF",
    )

    # Initialize handler
    handler = WebChatHandler(config)

    # Create a session
    session = handler.create_session(
        org_id=config.org_id,
        user_identifier="user@example.com",
    )

    # Generate embed code
    generator = WidgetConfigGenerator(
        base_url="https://api.kintsugi.ai",
        org_id=config.org_id,
    )
    embed_code = generator.generate_embed_code(config)
"""

from kintsugi.adapters.webchat.config import WebChatConfig
from kintsugi.adapters.webchat.handler import (
    WebChatHandler,
    WebChatMessageType,
    WebChatSession,
)
from kintsugi.adapters.webchat.routes import router
from kintsugi.adapters.webchat.static import (
    WIDGET_VERSION,
    get_css_link_tag,
    get_css_with_integrity,
    get_embed_script_tag,
    get_js_with_integrity,
    get_sri_hash,
    get_widget_css,
    get_widget_loader_js,
)
from kintsugi.adapters.webchat.widget import (
    WidgetConfigGenerator,
    WidgetPosition,
    WidgetTheme,
)

__all__ = [
    # Configuration
    "WebChatConfig",
    # Handler and session
    "WebChatHandler",
    "WebChatMessageType",
    "WebChatSession",
    # Widget generation
    "WidgetConfigGenerator",
    "WidgetPosition",
    "WidgetTheme",
    # Routes
    "router",
    # Static assets
    "WIDGET_VERSION",
    "get_widget_css",
    "get_widget_loader_js",
    "get_sri_hash",
    "get_css_with_integrity",
    "get_js_with_integrity",
    "get_embed_script_tag",
    "get_css_link_tag",
]
