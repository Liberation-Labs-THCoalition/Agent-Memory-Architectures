from __future__ import annotations

import re


class DataSpan:
    _DANGEROUS_PATTERNS = re.compile(
        r"\b(DROP|DELETE|TRUNCATE|ALTER)\b", re.IGNORECASE
    )

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    def _validate_query(self, query: str) -> bool:
        return not self._DANGEROUS_PATTERNS.search(query)

    async def query_data(self, query: str, params: dict | None = None) -> dict:
        if not self._validate_query(query):
            return {"success": False, "error": "query_rejected", "reason": "dangerous pattern detected"}
        return {
            "success": True,
            "result": {"rows": [], "query": query, "params": params},
        }

    async def export_data(
        self, format: str, query: str, params: dict | None = None
    ) -> dict:
        if format not in ("csv", "json", "xlsx"):
            return {"success": False, "error": "unsupported_format", "format": format}
        if not self._validate_query(query):
            return {"success": False, "error": "query_rejected", "reason": "dangerous pattern detected"}
        return {
            "success": True,
            "result": {"format": format, "url": f"mock://export.{format}", "query": query},
        }
