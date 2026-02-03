"""Static asset helpers for WebChat widget.

This module provides utilities for serving static assets including CSS
and JavaScript for the embeddable chat widget, along with security
features like SRI hash generation.
"""

from __future__ import annotations

import base64
import hashlib

WIDGET_VERSION = "1.0.0"
"""Current version of the WebChat widget."""


def get_widget_css() -> str:
    """Return minified widget CSS.

    Provides the core CSS styles for the chat widget including:
    - Widget container and positioning
    - Message bubbles for user and agent
    - Input area and send button
    - Typing indicators
    - Responsive adjustments

    Returns:
        Minified CSS string for widget styling.
    """
    return """/* Kintsugi WebChat Widget v""" + WIDGET_VERSION + """ */
:root {
  --kintsugi-primary: #9B59B6;
  --kintsugi-secondary: #8E44AD;
  --kintsugi-text: #333333;
  --kintsugi-bg: #FFFFFF;
  --kintsugi-font: system-ui, -apple-system, sans-serif;
  --kintsugi-radius: 12px;
  --kintsugi-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.kintsugi-chat-widget {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 9999;
  font-family: var(--kintsugi-font);
}

.kintsugi-chat-button {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: var(--kintsugi-primary);
  border: none;
  cursor: pointer;
  box-shadow: var(--kintsugi-shadow);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.2s, background 0.2s;
}

.kintsugi-chat-button:hover {
  transform: scale(1.05);
  background: var(--kintsugi-secondary);
}

.kintsugi-chat-button svg {
  width: 28px;
  height: 28px;
  fill: white;
}

.kintsugi-chat-container {
  display: none;
  width: 380px;
  height: 520px;
  background: var(--kintsugi-bg);
  border-radius: var(--kintsugi-radius);
  box-shadow: var(--kintsugi-shadow);
  flex-direction: column;
  overflow: hidden;
}

.kintsugi-chat-container.open {
  display: flex;
}

.kintsugi-chat-header {
  background: var(--kintsugi-primary);
  color: white;
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.kintsugi-chat-header-title {
  font-size: 16px;
  font-weight: 600;
  margin: 0;
}

.kintsugi-chat-header-subtitle {
  font-size: 12px;
  opacity: 0.9;
  margin: 4px 0 0 0;
}

.kintsugi-chat-close {
  background: none;
  border: none;
  color: white;
  cursor: pointer;
  padding: 4px;
  opacity: 0.8;
  transition: opacity 0.2s;
}

.kintsugi-chat-close:hover {
  opacity: 1;
}

.kintsugi-chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.kintsugi-message {
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 16px;
  font-size: 14px;
  line-height: 1.4;
  word-wrap: break-word;
}

.kintsugi-message.user {
  align-self: flex-end;
  background: var(--kintsugi-primary);
  color: white;
  border-bottom-right-radius: 4px;
}

.kintsugi-message.agent {
  align-self: flex-start;
  background: #F0F0F0;
  color: var(--kintsugi-text);
  border-bottom-left-radius: 4px;
}

.kintsugi-typing-indicator {
  display: flex;
  gap: 4px;
  padding: 10px 14px;
  background: #F0F0F0;
  border-radius: 16px;
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}

.kintsugi-typing-dot {
  width: 8px;
  height: 8px;
  background: #999;
  border-radius: 50%;
  animation: kintsugi-typing 1.4s infinite ease-in-out;
}

.kintsugi-typing-dot:nth-child(2) {
  animation-delay: 0.2s;
}

.kintsugi-typing-dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes kintsugi-typing {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-6px); }
}

.kintsugi-chat-input-area {
  display: flex;
  padding: 12px;
  border-top: 1px solid #E0E0E0;
  gap: 8px;
}

.kintsugi-chat-input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid #E0E0E0;
  border-radius: 20px;
  font-size: 14px;
  font-family: inherit;
  outline: none;
  transition: border-color 0.2s;
}

.kintsugi-chat-input:focus {
  border-color: var(--kintsugi-primary);
}

.kintsugi-chat-send {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--kintsugi-primary);
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.kintsugi-chat-send:hover {
  background: var(--kintsugi-secondary);
}

.kintsugi-chat-send:disabled {
  background: #CCC;
  cursor: not-allowed;
}

.kintsugi-chat-send svg {
  width: 18px;
  height: 18px;
  fill: white;
}

.kintsugi-powered-by {
  text-align: center;
  padding: 8px;
  font-size: 11px;
  color: #999;
}

.kintsugi-powered-by a {
  color: var(--kintsugi-primary);
  text-decoration: none;
}

@media (max-width: 480px) {
  .kintsugi-chat-container {
    width: 100%;
    height: 100%;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 0;
  }
}
"""


def get_widget_loader_js() -> str:
    """Return widget loader JavaScript.

    Provides the JavaScript that:
    - Creates the widget DOM structure
    - Establishes WebSocket connection
    - Handles user interactions
    - Manages message sending and receiving

    Returns:
        JavaScript string for widget initialization.
    """
    return """/* Kintsugi WebChat Widget Loader v""" + WIDGET_VERSION + """ */
(function(window, document) {
  'use strict';

  var KintsugiChat = {
    version: '""" + WIDGET_VERSION + """',
    config: null,
    session: null,
    ws: null,
    isOpen: false,
    elements: {},
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,

    init: function(config) {
      this.config = config;
      this.createWidget();
      this.bindEvents();
      this.createSession();
    },

    createWidget: function() {
      var container = document.createElement('div');
      container.className = 'kintsugi-chat-widget';
      container.innerHTML = this.getWidgetHTML();
      document.body.appendChild(container);

      this.elements = {
        container: container,
        button: container.querySelector('.kintsugi-chat-button'),
        chatContainer: container.querySelector('.kintsugi-chat-container'),
        closeBtn: container.querySelector('.kintsugi-chat-close'),
        messages: container.querySelector('.kintsugi-chat-messages'),
        input: container.querySelector('.kintsugi-chat-input'),
        sendBtn: container.querySelector('.kintsugi-chat-send')
      };

      this.applyTheme();
    },

    getWidgetHTML: function() {
      var cfg = this.config.widget || {};
      return [
        '<button class="kintsugi-chat-button" aria-label="Open chat">',
        '  <svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/></svg>',
        '</button>',
        '<div class="kintsugi-chat-container">',
        '  <div class="kintsugi-chat-header">',
        '    <div>',
        '      <h3 class="kintsugi-chat-header-title">' + (cfg.title || 'Chat with us') + '</h3>',
        cfg.subtitle ? '      <p class="kintsugi-chat-header-subtitle">' + cfg.subtitle + '</p>' : '',
        '    </div>',
        '    <button class="kintsugi-chat-close" aria-label="Close chat">',
        '      <svg width="20" height="20" viewBox="0 0 24 24"><path fill="currentColor" d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>',
        '    </button>',
        '  </div>',
        '  <div class="kintsugi-chat-messages"></div>',
        '  <div class="kintsugi-chat-input-area">',
        '    <input type="text" class="kintsugi-chat-input" placeholder="Type a message..." maxlength="' + (this.config.limits?.maxMessageLength || 4000) + '">',
        '    <button class="kintsugi-chat-send" disabled aria-label="Send message">',
        '      <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>',
        '    </button>',
        '  </div>',
        cfg.showPoweredBy !== false ? '  <div class="kintsugi-powered-by">Powered by <a href="https://kintsugi.ai" target="_blank">Kintsugi</a></div>' : '',
        '</div>'
      ].join('\\n');
    },

    applyTheme: function() {
      var theme = this.config.theme || {};
      var root = this.elements.container;
      if (theme.primaryColor) root.style.setProperty('--kintsugi-primary', theme.primaryColor);
      if (theme.secondaryColor) root.style.setProperty('--kintsugi-secondary', theme.secondaryColor);
      if (theme.textColor) root.style.setProperty('--kintsugi-text', theme.textColor);
      if (theme.backgroundColor) root.style.setProperty('--kintsugi-bg', theme.backgroundColor);
      if (theme.fontFamily) root.style.setProperty('--kintsugi-font', theme.fontFamily);
      if (theme.borderRadius) root.style.setProperty('--kintsugi-radius', theme.borderRadius);
      if (theme.shadow) root.style.setProperty('--kintsugi-shadow', theme.shadow);

      var pos = this.config.position || {};
      if (pos.bottom) this.elements.container.style.bottom = pos.bottom;
      if (pos.right) this.elements.container.style.right = pos.right;
      if (pos.left) {
        this.elements.container.style.left = pos.left;
        this.elements.container.style.right = 'auto';
      }
    },

    bindEvents: function() {
      var self = this;

      this.elements.button.addEventListener('click', function() {
        self.toggleChat();
      });

      this.elements.closeBtn.addEventListener('click', function() {
        self.toggleChat(false);
      });

      this.elements.input.addEventListener('input', function() {
        self.elements.sendBtn.disabled = !this.value.trim();
      });

      this.elements.input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && this.value.trim()) {
          e.preventDefault();
          self.sendMessage();
        }
      });

      this.elements.sendBtn.addEventListener('click', function() {
        self.sendMessage();
      });
    },

    toggleChat: function(open) {
      this.isOpen = open !== undefined ? open : !this.isOpen;
      this.elements.chatContainer.classList.toggle('open', this.isOpen);
      this.elements.button.style.display = this.isOpen ? 'none' : 'flex';

      if (this.isOpen) {
        this.elements.input.focus();
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
          this.connectWebSocket();
        }
      }
    },

    createSession: function() {
      var self = this;
      fetch(this.config.sessionUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ org_id: this.config.orgId })
      })
      .then(function(res) { return res.json(); })
      .then(function(data) {
        self.session = data;
      })
      .catch(function(err) {
        console.error('Kintsugi: Failed to create session', err);
      });
    },

    connectWebSocket: function() {
      if (!this.session) {
        setTimeout(this.connectWebSocket.bind(this), 500);
        return;
      }

      var wsUrl = this.config.websocketUrl + '?session_id=' + this.session.session_id;
      this.ws = new WebSocket(wsUrl);

      var self = this;

      this.ws.onopen = function() {
        self.reconnectAttempts = 0;
      };

      this.ws.onmessage = function(event) {
        var data = JSON.parse(event.data);
        self.handleMessage(data);
      };

      this.ws.onclose = function() {
        if (self.isOpen && self.reconnectAttempts < self.maxReconnectAttempts) {
          self.reconnectAttempts++;
          setTimeout(function() { self.connectWebSocket(); }, 1000 * self.reconnectAttempts);
        }
      };

      this.ws.onerror = function(err) {
        console.error('Kintsugi: WebSocket error', err);
      };
    },

    handleMessage: function(data) {
      switch (data.type) {
        case 'agent_response':
          this.hideTypingIndicator();
          this.addMessage(data.content, 'agent');
          break;
        case 'agent_typing':
          this.showTypingIndicator();
          break;
        case 'error':
          console.error('Kintsugi: Server error', data.error);
          this.addMessage('Sorry, an error occurred. Please try again.', 'agent');
          break;
      }
    },

    sendMessage: function() {
      var content = this.elements.input.value.trim();
      if (!content || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

      this.addMessage(content, 'user');
      this.ws.send(JSON.stringify({ type: 'message', content: content }));
      this.elements.input.value = '';
      this.elements.sendBtn.disabled = true;
    },

    addMessage: function(content, role) {
      var msg = document.createElement('div');
      msg.className = 'kintsugi-message ' + role;
      msg.textContent = content;
      this.elements.messages.appendChild(msg);
      this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    },

    showTypingIndicator: function() {
      if (this.elements.messages.querySelector('.kintsugi-typing-indicator')) return;
      var indicator = document.createElement('div');
      indicator.className = 'kintsugi-typing-indicator';
      indicator.innerHTML = '<span class="kintsugi-typing-dot"></span><span class="kintsugi-typing-dot"></span><span class="kintsugi-typing-dot"></span>';
      this.elements.messages.appendChild(indicator);
      this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    },

    hideTypingIndicator: function() {
      var indicator = this.elements.messages.querySelector('.kintsugi-typing-indicator');
      if (indicator) indicator.remove();
    }
  };

  window.KintsugiChat = KintsugiChat;

})(window, document);
"""


def get_sri_hash(content: str) -> str:
    """Generate SRI (Subresource Integrity) hash for content.

    Creates a SHA-384 hash of the content encoded in base64 format,
    suitable for use in integrity attributes of script and link tags.

    Args:
        content: The content to hash (CSS or JavaScript).

    Returns:
        SRI hash string in format "sha384-{base64_hash}".

    Example:
        >>> css = get_widget_css()
        >>> integrity = get_sri_hash(css)
        >>> print(integrity)
        sha384-abc123...
    """
    content_bytes = content.encode("utf-8")
    hash_bytes = hashlib.sha384(content_bytes).digest()
    hash_b64 = base64.b64encode(hash_bytes).decode("utf-8")
    return f"sha384-{hash_b64}"


def get_css_with_integrity() -> tuple[str, str]:
    """Get widget CSS with its integrity hash.

    Returns:
        Tuple of (css_content, integrity_hash).
    """
    css = get_widget_css()
    return css, get_sri_hash(css)


def get_js_with_integrity() -> tuple[str, str]:
    """Get widget loader JS with its integrity hash.

    Returns:
        Tuple of (js_content, integrity_hash).
    """
    js = get_widget_loader_js()
    return js, get_sri_hash(js)


def get_embed_script_tag(base_url: str, use_sri: bool = True) -> str:
    """Generate a script tag for loading the widget.

    Args:
        base_url: Base URL where the widget assets are hosted.
        use_sri: Whether to include integrity hash (recommended for security).

    Returns:
        HTML script tag string.
    """
    loader_url = f"{base_url.rstrip('/')}/webchat/static/loader.js"

    if use_sri:
        js, integrity = get_js_with_integrity()
        return (
            f'<script src="{loader_url}" '
            f'integrity="{integrity}" '
            f'crossorigin="anonymous" async></script>'
        )

    return f'<script src="{loader_url}" async></script>'


def get_css_link_tag(base_url: str, use_sri: bool = True) -> str:
    """Generate a link tag for loading the widget CSS.

    Args:
        base_url: Base URL where the widget assets are hosted.
        use_sri: Whether to include integrity hash (recommended for security).

    Returns:
        HTML link tag string.
    """
    css_url = f"{base_url.rstrip('/')}/webchat/static/widget.css"

    if use_sri:
        css, integrity = get_css_with_integrity()
        return (
            f'<link rel="stylesheet" href="{css_url}" '
            f'integrity="{integrity}" '
            f'crossorigin="anonymous">'
        )

    return f'<link rel="stylesheet" href="{css_url}">'
