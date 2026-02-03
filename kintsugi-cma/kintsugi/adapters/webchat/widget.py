"""Widget configuration generator for embeddable WebChat.

This module provides utilities for generating embeddable widget configurations,
including JavaScript snippets, iframe URLs, and JSON configuration for custom
integrations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urljoin

from kintsugi.adapters.webchat.config import WebChatConfig


@dataclass
class WidgetTheme:
    """Visual theme configuration for the chat widget.

    Defines the colors, fonts, and visual styling for the embeddable
    chat widget. All color values should be in hex format.

    Attributes:
        primary_color: Main accent color for buttons, headers, and highlights.
        secondary_color: Secondary accent color for hover states and borders.
        text_color: Primary text color for messages and labels.
        background_color: Widget background color.
        font_family: CSS font-family string for widget text.
        border_radius: CSS border-radius for widget container and elements.
        shadow: CSS box-shadow for widget container.
    """

    primary_color: str = "#9B59B6"
    secondary_color: str = "#8E44AD"
    text_color: str = "#333333"
    background_color: str = "#FFFFFF"
    font_family: str = "system-ui, -apple-system, sans-serif"
    border_radius: str = "12px"
    shadow: str = "0 4px 12px rgba(0,0,0,0.15)"

    def to_css_variables(self) -> dict[str, str]:
        """Convert theme to CSS custom properties.

        Returns:
            Dictionary mapping CSS variable names to values.
        """
        return {
            "--kintsugi-primary": self.primary_color,
            "--kintsugi-secondary": self.secondary_color,
            "--kintsugi-text": self.text_color,
            "--kintsugi-bg": self.background_color,
            "--kintsugi-font": self.font_family,
            "--kintsugi-radius": self.border_radius,
            "--kintsugi-shadow": self.shadow,
        }

    def to_dict(self) -> dict[str, str]:
        """Convert theme to dictionary representation.

        Returns:
            Dictionary with all theme properties.
        """
        return {
            "primaryColor": self.primary_color,
            "secondaryColor": self.secondary_color,
            "textColor": self.text_color,
            "backgroundColor": self.background_color,
            "fontFamily": self.font_family,
            "borderRadius": self.border_radius,
            "shadow": self.shadow,
        }


@dataclass
class WidgetPosition:
    """Positioning configuration for the chat widget.

    Controls where the widget appears on the page and how it's positioned.

    Attributes:
        position: Preset position or "custom" for manual positioning.
        bottom: CSS bottom offset from viewport.
        right: CSS right offset from viewport.
        left: CSS left offset from viewport (used for left positions).
    """

    position: str = "bottom-right"
    bottom: str = "20px"
    right: str = "20px"
    left: str | None = None

    def __post_init__(self) -> None:
        """Validate and adjust position settings."""
        valid_positions = {"bottom-right", "bottom-left", "custom"}
        if self.position not in valid_positions:
            raise ValueError(
                f"Invalid position '{self.position}'. "
                f"Must be one of: {', '.join(valid_positions)}"
            )

        # Adjust left/right based on position preset
        if self.position == "bottom-left":
            self.left = self.left or "20px"
            self.right = None
        elif self.position == "bottom-right":
            self.right = self.right or "20px"
            self.left = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert position to dictionary representation.

        Returns:
            Dictionary with position properties.
        """
        return {
            "position": self.position,
            "bottom": self.bottom,
            "right": self.right,
            "left": self.left,
        }

    def to_css_style(self) -> str:
        """Generate CSS style string for positioning.

        Returns:
            CSS style string for the widget container.
        """
        styles = [
            "position: fixed",
            f"bottom: {self.bottom}",
        ]

        if self.right:
            styles.append(f"right: {self.right}")
        if self.left:
            styles.append(f"left: {self.left}")

        return "; ".join(styles) + ";"


class WidgetConfigGenerator:
    """Generates embeddable widget configuration and code snippets.

    Provides methods to generate HTML/JavaScript embed snippets, JSON
    configuration, and URLs for integrating the chat widget into web pages.

    Attributes:
        _base_url: Base URL for the Kintsugi API.
        _org_id: Organization ID for this widget.
    """

    def __init__(self, base_url: str, org_id: str) -> None:
        """Initialize the widget configuration generator.

        Args:
            base_url: Base URL for the Kintsugi API (e.g., "https://api.kintsugi.ai").
            org_id: Organization ID this widget belongs to.
        """
        self._base_url = base_url.rstrip("/")
        self._org_id = org_id

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self._base_url

    @property
    def org_id(self) -> str:
        """Get the organization ID."""
        return self._org_id

    def generate_embed_code(
        self,
        config: WebChatConfig,
        theme: WidgetTheme | None = None,
        position: WidgetPosition | None = None,
    ) -> str:
        """Generate HTML/JavaScript embed snippet for the widget.

        Creates a self-contained script tag that loads and initializes
        the chat widget with the provided configuration.

        Args:
            config: WebChat configuration for the widget.
            theme: Optional theme configuration. Uses defaults if not provided.
            position: Optional position configuration. Uses defaults if not provided.

        Returns:
            HTML string containing the script tag for embedding.
        """
        theme = theme or WidgetTheme(primary_color=config.primary_color)
        position = position or WidgetPosition()

        config_json = self.generate_config_json(config, theme, position)
        config_str = json.dumps(config_json, indent=2)

        loader_url = urljoin(self._base_url, "/webchat/static/loader.js")

        return f'''<!-- Kintsugi WebChat Widget -->
<script>
  (function() {{
    var config = {config_str};

    var script = document.createElement('script');
    script.src = '{loader_url}';
    script.async = true;
    script.onload = function() {{
      if (window.KintsugiChat) {{
        window.KintsugiChat.init(config);
      }}
    }};
    document.head.appendChild(script);
  }})();
</script>
<!-- End Kintsugi WebChat Widget -->'''

    def generate_config_json(
        self,
        config: WebChatConfig,
        theme: WidgetTheme | None = None,
        position: WidgetPosition | None = None,
    ) -> dict[str, Any]:
        """Generate JSON configuration for custom integrations.

        Creates a complete configuration object that can be used by
        custom frontend implementations or the standard widget loader.

        Args:
            config: WebChat configuration for the widget.
            theme: Optional theme configuration. Uses defaults if not provided.
            position: Optional position configuration. Uses defaults if not provided.

        Returns:
            Dictionary containing all widget configuration.
        """
        theme = theme or WidgetTheme(primary_color=config.primary_color)
        position = position or WidgetPosition()

        return {
            "orgId": config.org_id,
            "baseUrl": self._base_url,
            "websocketUrl": self.get_websocket_url(),
            "sessionUrl": urljoin(self._base_url, "/webchat/session"),
            "widget": {
                "title": config.widget_title,
                "subtitle": config.widget_subtitle,
                "showPoweredBy": config.show_powered_by,
            },
            "theme": theme.to_dict(),
            "position": position.to_dict(),
            "limits": {
                "maxMessageLength": config.max_message_length,
                "rateLimitPerMinute": config.rate_limit_messages_per_minute,
            },
            "auth": {
                "required": config.require_auth,
            },
        }

    def get_websocket_url(self) -> str:
        """Get the WebSocket endpoint URL.

        Constructs the WebSocket URL based on the base URL, converting
        http(s) to ws(s) as appropriate.

        Returns:
            WebSocket URL string for the chat connection.
        """
        ws_base = self._base_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        return f"{ws_base}/ws/webchat/{self._org_id}"

    def get_iframe_url(self, config: WebChatConfig) -> str:
        """Get URL for iframe embedding.

        Generates a URL that can be used in an iframe for organizations
        that prefer iframe-based embedding over script injection.

        Args:
            config: WebChat configuration for the widget.

        Returns:
            URL string for iframe src attribute.
        """
        params = {
            "org_id": config.org_id,
            "title": config.widget_title,
            "primary_color": config.primary_color,
            "show_powered_by": str(config.show_powered_by).lower(),
        }

        if config.widget_subtitle:
            params["subtitle"] = config.widget_subtitle

        query_string = urlencode(params)
        return f"{self._base_url}/webchat/embed/{self._org_id}?{query_string}"

    def generate_react_component_code(
        self,
        config: WebChatConfig,
        theme: WidgetTheme | None = None,
    ) -> str:
        """Generate React component integration code.

        Creates example code for integrating the chat widget in a React
        application using the npm package.

        Args:
            config: WebChat configuration for the widget.
            theme: Optional theme configuration.

        Returns:
            String containing React/TypeScript code example.
        """
        theme = theme or WidgetTheme(primary_color=config.primary_color)

        return f'''import {{ KintsugiChat }} from '@kintsugi/webchat-react';

function App() {{
  return (
    <KintsugiChat
      orgId="{config.org_id}"
      baseUrl="{self._base_url}"
      title="{config.widget_title}"
      subtitle={json.dumps(config.widget_subtitle)}
      theme={{{{
        primaryColor: "{theme.primary_color}",
        secondaryColor: "{theme.secondary_color}",
        textColor: "{theme.text_color}",
        backgroundColor: "{theme.background_color}",
      }}}}
      showPoweredBy={{{str(config.show_powered_by).lower()}}}
    />
  );
}}'''

    def generate_iframe_embed_code(
        self,
        config: WebChatConfig,
        width: str = "400px",
        height: str = "600px",
    ) -> str:
        """Generate iframe embed code.

        Creates an iframe HTML snippet for simple embedding without
        JavaScript.

        Args:
            config: WebChat configuration for the widget.
            width: CSS width for the iframe.
            height: CSS height for the iframe.

        Returns:
            HTML string containing the iframe element.
        """
        iframe_url = self.get_iframe_url(config)

        return f'''<!-- Kintsugi WebChat Widget (iframe) -->
<iframe
  src="{iframe_url}"
  width="{width}"
  height="{height}"
  style="border: none; position: fixed; bottom: 20px; right: 20px; z-index: 9999;"
  allow="microphone"
  title="{config.widget_title}"
></iframe>
<!-- End Kintsugi WebChat Widget -->'''
