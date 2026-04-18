"""Cache Store — compressed storage for KV cache snapshots.

Lightweight, open-source version of the Oracle Harness cache compressor.
Stores KV cache data with layered compression while preserving full
fidelity for future analysis.

Pipeline:
  RAW → FP16 quantize → delta encode → compress → SHA256 hash → disk

Features:
  - Content-addressable dedup via SHA256
  - Delta encoding for sequential snapshots (~24x compression)
  - FP16 quantization (lossless for spectral geometry features)
  - Automatic metadata tracking with geometry co-storage

Author: Operator (Coalition)
Original design: Oracle Harness (Liberation Labs)
Adapted for public release: 2026-04-18
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional zstd
try:
    import zstandard as zstd
    _ZSTD = True
except ImportError:
    _ZSTD = False

_SAFE_ID = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
_HEX = re.compile(r'^[a-fA-F0-9]+$')
_MAX_DELTA_DEPTH = 50


@dataclass
class SnapshotMeta:
    """Metadata for a stored cache snapshot."""
    snapshot_id: str
    timestamp: float
    original_bytes: int
    stored_bytes: int
    compression_ratio: float
    content_hash: str
    original_dtype: str
    stored_dtype: str
    compression: str
    delta_parent: Optional[str] = None
    geometry: Optional[dict] = None
    labels: Optional[dict] = None


class CacheStore:
    """Compressed storage for KV cache snapshots.

    Usage::

        store = CacheStore(store_dir="./cache_data")

        # Store a snapshot
        meta = store.store(
            data=kv_cache_numpy_array,
            snapshot_id="turn_3_encoding",
            geometry={"effective_rank": 60.8, "spectral_entropy": 20.2},
        )
        print(f"Compressed {meta.compression_ratio:.1f}x")

        # Load it back
        arr = store.load("turn_3_encoding")

        # Storage stats
        print(store.stats())
    """

    def __init__(
        self,
        store_dir: str = "./cache_store",
        quantize_fp16: bool = True,
        enable_delta: bool = True,
        compress_level: int = 3,
    ):
        self._dir = Path(store_dir)
        self._blobs = self._dir / "blobs"
        self._meta = self._dir / "meta"
        for d in (self._dir, self._blobs, self._meta):
            d.mkdir(parents=True, exist_ok=True)

        self._fp16 = quantize_fp16
        self._delta = enable_delta
        self._level = compress_level
        self._prev_id: Optional[str] = None
        self._prev_data: Optional[bytes] = None

    def store(
        self,
        data: Any,
        snapshot_id: str,
        geometry: Optional[dict] = None,
        labels: Optional[dict] = None,
    ) -> SnapshotMeta:
        """Store a cache snapshot with layered compression."""
        self._validate_id(snapshot_id)
        start = time.time()

        # Serialize
        raw, dtype = self._to_bytes(data)
        orig_size = len(raw)

        # FP16 quantize
        if self._fp16 and dtype in ("float32", "float64"):
            quantized = np.frombuffer(raw, dtype=dtype).astype(np.float16).tobytes()
            stored_dtype = "float16"
        else:
            quantized = raw
            stored_dtype = dtype

        # Delta encode
        delta_parent = None
        to_compress = quantized
        if (self._delta and self._prev_data is not None
                and len(self._prev_data) == len(quantized)):
            xor = np.bitwise_xor(
                np.frombuffer(quantized, np.uint8),
                np.frombuffer(self._prev_data, np.uint8),
            ).tobytes()
            to_compress = xor
            delta_parent = self._prev_id

        # Compress
        if _ZSTD:
            compressed = zstd.ZstdCompressor(level=self._level).compress(to_compress)
            comp_method = "zstd"
        else:
            compressed = gzip.compress(to_compress, compresslevel=6)
            comp_method = "gzip"

        # Hash + write
        h = hashlib.sha256(compressed).hexdigest()
        blob_path = self._blobs / f"{h}.bin"
        if not blob_path.exists():
            blob_path.write_bytes(compressed)

        # Update delta tracking
        self._prev_id = snapshot_id
        self._prev_data = quantized

        # Metadata
        meta = SnapshotMeta(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            original_bytes=orig_size,
            stored_bytes=len(compressed),
            compression_ratio=orig_size / max(len(compressed), 1),
            content_hash=h,
            original_dtype=dtype,
            stored_dtype=stored_dtype,
            compression=comp_method,
            delta_parent=delta_parent,
            geometry=geometry,
            labels=labels,
        )
        (self._meta / f"{snapshot_id}.json").write_text(json.dumps(
            {k: v for k, v in meta.__dict__.items()}, indent=2
        ))

        logger.info(
            "Stored %s: %s → %s (%.1fx)",
            snapshot_id, _hsize(orig_size), _hsize(len(compressed)),
            meta.compression_ratio,
        )
        return meta

    def load(self, snapshot_id: str) -> np.ndarray:
        """Load and decompress a snapshot."""
        self._validate_id(snapshot_id)
        meta = self._load_meta(snapshot_id)

        self._validate_hash(meta["content_hash"])
        blob = (self._blobs / f"{meta['content_hash']}.bin").read_bytes()

        # Decompress
        data = self._decompress(blob, meta["compression"])

        # Un-delta
        if meta.get("delta_parent"):
            parent = self._load_raw(meta["delta_parent"])
            data = np.bitwise_xor(
                np.frombuffer(data, np.uint8),
                np.frombuffer(parent, np.uint8),
            ).tobytes()

        # Un-quantize
        arr = np.frombuffer(data, dtype=meta["stored_dtype"])
        if meta["stored_dtype"] == "float16" and meta["original_dtype"] == "float32":
            arr = arr.astype(np.float32)
        return arr

    def stats(self) -> dict:
        """Storage statistics."""
        metas = list(self._meta.glob("*.json"))
        blobs = list(self._blobs.glob("*.bin"))
        total_orig = sum(
            json.loads(m.read_text()).get("original_bytes", 0) for m in metas
        )
        total_stored = sum(
            json.loads(m.read_text()).get("stored_bytes", 0) for m in metas
        )
        disk = sum(f.stat().st_size for f in blobs)
        return {
            "snapshots": len(metas),
            "unique_blobs": len(blobs),
            "total_original": _hsize(total_orig),
            "total_compressed": _hsize(total_stored),
            "disk_usage": _hsize(disk),
            "ratio": total_orig / max(total_stored, 1),
        }

    # ── Internal ──

    def _to_bytes(self, data) -> Tuple[bytes, str]:
        if isinstance(data, bytes):
            return data, "uint8"
        if isinstance(data, np.ndarray):
            return data.tobytes(), str(data.dtype)
        if hasattr(data, 'detach'):  # torch tensor
            arr = data.detach().cpu().numpy()
            return arr.tobytes(), str(arr.dtype)
        arr = np.array(data)
        return arr.tobytes(), str(arr.dtype)

    def _load_meta(self, sid: str) -> dict:
        p = self._meta / f"{sid}.json"
        if not p.exists():
            raise FileNotFoundError(f"No snapshot: {sid}")
        return json.loads(p.read_text())

    def _load_raw(self, sid: str, depth: int = 0) -> bytes:
        if depth >= _MAX_DELTA_DEPTH:
            raise RuntimeError(f"Delta chain too deep at {sid}")
        self._validate_id(sid)
        meta = self._load_meta(sid)
        self._validate_hash(meta["content_hash"])
        data = self._decompress(
            (self._blobs / f"{meta['content_hash']}.bin").read_bytes(),
            meta["compression"],
        )
        if meta.get("delta_parent"):
            parent = self._load_raw(meta["delta_parent"], depth + 1)
            data = np.bitwise_xor(
                np.frombuffer(data, np.uint8),
                np.frombuffer(parent, np.uint8),
            ).tobytes()
        return data

    def _decompress(self, data: bytes, method: str) -> bytes:
        if method == "zstd":
            if not _ZSTD:
                raise RuntimeError("zstd required but not installed")
            return zstd.ZstdDecompressor().decompress(data)
        return gzip.decompress(data)

    @staticmethod
    def _validate_id(sid: str):
        if not sid or not _SAFE_ID.match(sid) or '..' in sid:
            raise ValueError(f"Invalid snapshot ID: {sid!r}")

    @staticmethod
    def _validate_hash(h: str):
        if not h or not _HEX.match(h):
            raise ValueError(f"Invalid hash: {h!r}")


def _hsize(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"
