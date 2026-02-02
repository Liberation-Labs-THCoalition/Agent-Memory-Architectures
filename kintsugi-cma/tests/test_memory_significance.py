"""Tests for kintsugi.memory.significance module."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kintsugi.memory.significance import (
    MemoryLayer,
    ReapResult,
    compute_expiration,
    compute_layer,
    ExpiredMemoryReaper,
)


# ---------------------------------------------------------------------------
# MemoryLayer enum
# ---------------------------------------------------------------------------


class TestMemoryLayer:
    def test_values(self):
        assert MemoryLayer.PERMANENT.value == "PERMANENT"
        assert MemoryLayer.VOLATILE.value == "VOLATILE"

    def test_is_str(self):
        assert isinstance(MemoryLayer.CORE, str)

    def test_all_members(self):
        assert len(MemoryLayer) == 5


# ---------------------------------------------------------------------------
# compute_layer
# ---------------------------------------------------------------------------


class TestComputeLayer:
    @pytest.mark.parametrize("sig,expected", [
        (1, MemoryLayer.PERMANENT),
        (2, MemoryLayer.PERMANENT),
        (3, MemoryLayer.CORE),
        (4, MemoryLayer.CORE),
        (5, MemoryLayer.IMPORTANT),
        (6, MemoryLayer.IMPORTANT),
        (7, MemoryLayer.STANDARD),
        (8, MemoryLayer.STANDARD),
        (9, MemoryLayer.VOLATILE),
        (10, MemoryLayer.VOLATILE),
    ])
    def test_all_ranges(self, sig, expected):
        assert compute_layer(sig) == expected

    @pytest.mark.parametrize("sig", [0, -1, 11, 100])
    def test_out_of_range(self, sig):
        with pytest.raises(ValueError, match="significance must be 1-10"):
            compute_layer(sig)


# ---------------------------------------------------------------------------
# compute_expiration
# ---------------------------------------------------------------------------


class TestComputeExpiration:
    def test_permanent_returns_none(self):
        t = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert compute_expiration(1, t) is None
        assert compute_expiration(2, t) is None

    @pytest.mark.parametrize("sig,days", [
        (3, 730), (4, 730),
        (5, 365), (6, 365),
        (7, 90), (8, 90),
        (9, 30), (10, 30),
    ])
    def test_ttls(self, sig, days):
        t = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = compute_expiration(sig, t)
        assert result == t + timedelta(days=days)

    def test_invalid_sig(self):
        with pytest.raises(ValueError):
            compute_expiration(0, datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# ExpiredMemoryReaper — patch select/update to avoid SQLAlchemy validation
# ---------------------------------------------------------------------------


class TestExpiredMemoryReaper:
    def _make_mock_mu(self):
        """Create a mock MemoryUnit that works with patched select/update."""
        mu = MagicMock()
        mu.org_id = "org-1"
        mu.expires_at = MagicMock()
        mu.expires_at.isnot.return_value = True
        mu.expires_at.__le__ = MagicMock(return_value=True)
        mu.id = MagicMock()
        mu.id.in_.return_value = True
        return mu

    @pytest.mark.asyncio
    async def test_reap_no_expired(self):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_mu = self._make_mock_mu()
        mock_select = MagicMock(return_value=MagicMock())
        mock_update = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"kintsugi.models.base": MagicMock(MemoryUnit=mock_mu), "kintsugi.models": MagicMock()}), \
             patch("kintsugi.memory.significance.select", mock_select), \
             patch("kintsugi.memory.significance.update", mock_update):
            reaper = ExpiredMemoryReaper()
            result = await reaper.reap("org-1", mock_session)

        assert isinstance(result, ReapResult)
        assert result.checked == 0
        assert result.archived == 0
        assert result.errors == 0

    @pytest.mark.asyncio
    async def test_reap_with_expired(self):
        row1 = MagicMock(id="m1")
        row2 = MagicMock(id="m2")

        mock_select_result = MagicMock()
        mock_select_result.scalars.return_value.all.return_value = [row1, row2]
        mock_update_result = MagicMock()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_select_result, mock_update_result])

        mock_mu = self._make_mock_mu()
        mock_select = MagicMock(return_value=MagicMock())
        mock_update = MagicMock(return_value=MagicMock(return_value=MagicMock()))

        with patch.dict(sys.modules, {"kintsugi.models.base": MagicMock(MemoryUnit=mock_mu), "kintsugi.models": MagicMock()}), \
             patch("kintsugi.memory.significance.select", mock_select), \
             patch("kintsugi.memory.significance.update", mock_update):
            reaper = ExpiredMemoryReaper()
            result = await reaper.reap("org-1", mock_session)

        assert result.checked == 2
        assert result.expired == 2
        assert result.archived == 2
        assert result.errors == 0

    @pytest.mark.asyncio
    async def test_reap_with_error(self):
        row1 = MagicMock(id="m1")

        mock_select_result = MagicMock()
        mock_select_result.scalars.return_value.all.return_value = [row1]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_select_result, Exception("db error")])

        mock_mu = self._make_mock_mu()
        mock_select = MagicMock(return_value=MagicMock())
        mock_update = MagicMock(return_value=MagicMock(return_value=MagicMock()))

        with patch.dict(sys.modules, {"kintsugi.models.base": MagicMock(MemoryUnit=mock_mu), "kintsugi.models": MagicMock()}), \
             patch("kintsugi.memory.significance.select", mock_select), \
             patch("kintsugi.memory.significance.update", mock_update):
            reaper = ExpiredMemoryReaper()
            result = await reaper.reap("org-1", mock_session)

        assert result.errors == 1
        assert result.archived == 0


class TestReapResult:
    def test_dataclass(self):
        r = ReapResult(checked=5, expired=3, archived=2, errors=1)
        assert r.checked == 5
        assert r.expired == 3
