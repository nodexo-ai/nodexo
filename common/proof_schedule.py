"""Shared proof schedule derivation."""
from __future__ import annotations

import hashlib
import struct


BEACON_SLOT_PREFIX = b"NODEXO_BEACON_SLOT_V1"
JITTER_PREFIX = b"NODEXO_JITTER_V1"

# The beacon slot is network-wide, not per executor. Default range 0..2 blocks
# keeps the proof inside the 15-block cycle while making the actual seed block
# unknowable until the epoch has started.
DEFAULT_BEACON_MAX_OFFSET_BLOCKS = 3

# Per-executor jitter is deliberately tiny. It avoids exact same-second
# submissions without giving one physical GPU enough room to serialize several
# claimed executors.
DEFAULT_JITTER_SECONDS = 3


def _normalize_hash(value: bytes | str) -> bytes:
    if isinstance(value, bytes):
        return value
    return bytes.fromhex(str(value).removeprefix("0x"))


def derive_beacon_offset_blocks(
    epoch_start_hash: bytes | str,
    max_offset_blocks: int = DEFAULT_BEACON_MAX_OFFSET_BLOCKS,
) -> int:
    max_offset_blocks = max(1, int(max_offset_blocks))
    h = hashlib.sha256(BEACON_SLOT_PREFIX + _normalize_hash(epoch_start_hash)).digest()
    return struct.unpack("<H", h[:2])[0] % max_offset_blocks


def beacon_block_for_epoch(
    epoch_id: int,
    epoch_interval: int,
    epoch_start_hash: bytes | str,
    max_offset_blocks: int = DEFAULT_BEACON_MAX_OFFSET_BLOCKS,
) -> int:
    return (
        int(epoch_id) * int(epoch_interval)
        + derive_beacon_offset_blocks(epoch_start_hash, max_offset_blocks)
    )


def derive_jitter_seconds(
    beacon: bytes | str,
    executor_id: str,
    max_jitter_seconds: int = DEFAULT_JITTER_SECONDS,
) -> int:
    max_jitter_seconds = max(1, int(max_jitter_seconds))
    payload = JITTER_PREFIX + _normalize_hash(beacon) + bytes.fromhex(executor_id)
    h = hashlib.sha256(payload).digest()
    return struct.unpack("<H", h[:2])[0] % max_jitter_seconds
