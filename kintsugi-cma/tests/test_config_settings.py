"""Tests for kintsugi.config.settings."""

import os
import pytest
from unittest.mock import patch


class TestSettings:
    """Test the Settings pydantic-settings class."""

    def _make(self, **kwargs):
        """Create a Settings instance with env isolation."""
        # Avoid reading real env vars / .env by overriding
        from kintsugi.config.settings import Settings
        return Settings(**kwargs)

    # -- defaults --

    def test_defaults(self):
        s = self._make()
        assert s.DEPLOYMENT_TIER == "sprout"
        assert s.EMBEDDING_MODE == "local"
        assert s.EMBEDDING_MODEL == "all-mpnet-base-v2"
        assert s.ANTHROPIC_API_KEY == ""
        assert s.OPENAI_API_KEY == ""
        assert s.KINTSUGI_SHADOW_ENABLED is False
        assert s.SHIELD_BUDGET_PER_SESSION == 5.0
        assert s.SHIELD_BUDGET_PER_DAY == 50.0
        assert s.OTEL_EXPORTER_ENDPOINT == ""
        assert s.REDIS_URL == "redis://localhost:6379/0"
        assert s.SECRET_KEY == "CHANGE-ME-in-production"
        assert "http://localhost:3000" in s.CORS_ORIGINS
        assert "postgresql+asyncpg://" in s.DATABASE_URL

    # -- _auto_shadow validator --

    def test_grove_enables_shadow(self):
        s = self._make(DEPLOYMENT_TIER="grove")
        assert s.KINTSUGI_SHADOW_ENABLED is True

    def test_sprout_does_not_enable_shadow(self):
        s = self._make(DEPLOYMENT_TIER="sprout")
        assert s.KINTSUGI_SHADOW_ENABLED is False

    def test_seed_does_not_enable_shadow(self):
        s = self._make(DEPLOYMENT_TIER="seed")
        assert s.KINTSUGI_SHADOW_ENABLED is False

    def test_grove_shadow_override_still_true(self):
        # Even if explicitly set False, grove forces True
        s = self._make(DEPLOYMENT_TIER="grove", KINTSUGI_SHADOW_ENABLED=False)
        assert s.KINTSUGI_SHADOW_ENABLED is True

    # -- _fix_pg_scheme validator --

    def test_fix_pg_scheme_adds_asyncpg(self):
        s = self._make(DATABASE_URL="postgresql://user:pass@host/db")
        assert s.DATABASE_URL == "postgresql+asyncpg://user:pass@host/db"

    def test_already_asyncpg_untouched(self):
        url = "postgresql+asyncpg://user:pass@host/db"
        s = self._make(DATABASE_URL=url)
        assert s.DATABASE_URL == url

    def test_non_postgresql_untouched(self):
        url = "sqlite:///test.db"
        s = self._make(DATABASE_URL=url)
        assert s.DATABASE_URL == url

    # -- kwarg overrides --

    def test_override_via_kwargs(self):
        s = self._make(
            EMBEDDING_MODE="api",
            EMBEDDING_MODEL="text-embedding-3-small",
            ANTHROPIC_API_KEY="sk-test",
            SHIELD_BUDGET_PER_SESSION=10.0,
        )
        assert s.EMBEDDING_MODE == "api"
        assert s.EMBEDDING_MODEL == "text-embedding-3-small"
        assert s.ANTHROPIC_API_KEY == "sk-test"
        assert s.SHIELD_BUDGET_PER_SESSION == 10.0

    # -- model_routing default --

    def test_model_routing_default(self):
        s = self._make()
        assert "haiku" in s.MODEL_ROUTING
        assert "sonnet" in s.MODEL_ROUTING
        assert "opus" in s.MODEL_ROUTING

    # -- invalid tier --

    def test_invalid_tier_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._make(DEPLOYMENT_TIER="invalid")

    def test_invalid_embedding_mode_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._make(EMBEDDING_MODE="gpu")
