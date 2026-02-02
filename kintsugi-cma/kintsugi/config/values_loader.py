"""Load, validate, merge, save, and watch VALUES.json files.

This module is the primary interface for reading and writing organizational
value documents.  It supports:

* Loading from a path or from built-in templates
* Deep-merging a partial dict on top of template defaults
* Atomic save with pretty-printed JSON
* File watching (watchdog with polling fallback)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .values_schema import OrganizationValues

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_VALID_ORG_TYPES = {"mutual_aid", "nonprofit_501c3", "cooperative", "advocacy"}


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_values(path: str | Path) -> OrganizationValues:
    """Load and validate a VALUES.json file.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the content does not conform to the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VALUES.json not found at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return OrganizationValues.model_validate(raw)


def load_from_template(org_type: str) -> OrganizationValues:
    """Load a built-in template by organization type.

    Raises:
        ValueError: If *org_type* has no matching template.
    """
    template_path = _TEMPLATES_DIR / f"{org_type}.json"
    if not template_path.exists():
        raise ValueError(
            f"No template for org_type={org_type!r}. "
            f"Available: {sorted(_VALID_ORG_TYPES)}"
        )
    return load_values(template_path)


def merge_with_defaults(partial: dict[str, Any], org_type: str) -> OrganizationValues:
    """Deep-merge *partial* config on top of a template's defaults.

    Keys present in *partial* override the template; keys absent in *partial*
    are filled from the template.  Lists are replaced wholesale (not appended).
    """
    template = load_from_template(org_type)
    base = template.model_dump(mode="json")
    merged = _deep_merge(base, partial)
    return OrganizationValues.model_validate(merged)


def save_values(values: OrganizationValues, path: str | Path) -> None:
    """Atomically write a validated VALUES.json.

    Writes to a temporary file first, then renames, so readers never see a
    partial write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    data = values.model_dump(mode="json")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    logger.info("Saved VALUES.json to %s", path)


# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

class FileWatcher:
    """Watch a file for modifications and invoke a callback.

    Uses *watchdog* if available, otherwise falls back to polling.

    Parameters:
        path: File to watch.
        callback: Called with the Path when the file is modified.
        poll_interval: Seconds between polls when using the fallback.
    """

    def __init__(
        self,
        path: str | Path,
        callback: Callable[[Path], Any],
        poll_interval: float = 2.0,
    ) -> None:
        self.path = Path(path).resolve()
        self.callback = callback
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer: Any = None  # watchdog observer, if available

    # -- public API ----------------------------------------------------------

    def start(self) -> None:
        """Begin watching (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            self._start_watchdog()
            logger.info("FileWatcher: using watchdog for %s", self.path)
        except ImportError:
            self._start_polling()
            logger.info("FileWatcher: using polling fallback for %s", self.path)

    def stop(self) -> None:
        """Stop watching and clean up."""
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval * 2)
            self._thread = None

    # -- watchdog backend ----------------------------------------------------

    def _start_watchdog(self) -> None:
        from watchdog.events import FileSystemEventHandler  # type: ignore[import-untyped]
        from watchdog.observers import Observer  # type: ignore[import-untyped]

        watcher = self  # closure

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event: Any) -> None:
                if Path(event.src_path).resolve() == watcher.path:
                    watcher.callback(watcher.path)

        self._observer = Observer()
        self._observer.schedule(_Handler(), str(self.path.parent), recursive=False)
        self._observer.daemon = True
        self._observer.start()

    # -- polling fallback ----------------------------------------------------

    def _start_polling(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        last_mtime: float = 0.0
        if self.path.exists():
            last_mtime = self.path.stat().st_mtime
        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval)
            if self._stop_event.is_set():
                break
            try:
                if self.path.exists():
                    mtime = self.path.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        self.callback(self.path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*. Lists are replaced, not appended."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
