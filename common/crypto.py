"""
SR25519 signature utilities for proof submission authentication.

Executors sign proof payloads with their miner's hotkey. Validators verify
the signature against the hotkey registered on-chain for that executor.

Signing flow (executor side):
  1. Canonical JSON of payload → SHA256 digest
  2. SR25519 sign(digest) with miner hotkey
  3. Include signature + hotkey SS58 in request headers

Verification flow (validator side):
  1. Extract signature + hotkey from headers
  2. Recompute SHA256 digest of the body
  3. Verify SR25519 signature against the hotkey
  4. Check the hotkey matches the executor's registered miner on-chain

Reference: see protocol design notes (build_signature_header)
Reference: see protocol design notes (verify)
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from typing import Optional

import bittensor as bt

# Header names for signed requests
HEADER_SIGNATURE = "X-Nodexo-Signature"
HEADER_HOTKEY = "X-Nodexo-Hotkey"
HEADER_TIMESTAMP = "X-Nodexo-Timestamp"
HEADER_NONCE = "X-Nodexo-Nonce"

# Max age for signed requests (prevents replay)
MAX_REQUEST_AGE = 120  # seconds
_NONCE_CACHE_CAP = 50000


def sign_payload(payload_bytes: bytes, hotkey_seed: bytes) -> dict[str, str]:
    """Sign a payload and return headers to include in the request.

    Args:
        payload_bytes: The raw request body (JSON bytes).
        hotkey_seed: The miner's hotkey seed (32 bytes).

    Returns:
        Dict of headers to add to the HTTP request.
    """
    from substrateinterface import Keypair

    keypair = Keypair.create_from_seed(hotkey_seed.hex())
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)

    # Digest: SHA256(timestamp + nonce + body)
    digest = hashlib.sha256(
        timestamp.encode() + nonce.encode() + payload_bytes
    ).digest()

    signature = keypair.sign(digest)

    return {
        HEADER_SIGNATURE: signature.hex(),
        HEADER_HOTKEY: keypair.ss58_address,
        HEADER_TIMESTAMP: timestamp,
        HEADER_NONCE: nonce,
    }


def monitor_binding_body(executor_id: str, monitor_hotkey_ss58: str) -> bytes:
    """Canonical body signed by the miner hotkey to authorize a monitor key.

    The monitor signs each report with its own sidecar key. On first bind the
    validator also requires this owner signature, proving the monitor key was
    delegated by the hotkey already bound to the executor.
    """
    return json.dumps(
        {
            "executor_id": str(executor_id).removeprefix("0x"),
            "monitor_hotkey": str(monitor_hotkey_ss58),
            "purpose": "nodexo-monitor-binding-v1",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def verify_signature(
    body_bytes: bytes,
    sig_hex: str,
    hotkey_ss58: str,
    timestamp_str: str,
    nonce: str,
    max_age: int | None = MAX_REQUEST_AGE,
) -> bool:
    """Verify an SR25519 signature on a request body.

    Args:
        body_bytes: The raw request body.
        sig_hex: Hex-encoded SR25519 signature.
        hotkey_ss58: SS58-encoded hotkey address of the signer.
        timestamp_str: Unix timestamp string from the request.
        nonce: Nonce string from the request.

    Returns:
        True if signature is valid and request is fresh.
    """
    from substrateinterface import Keypair

    # Check freshness for ordinary request signatures. Long-lived
    # delegation signatures can pass max_age=None when the signed body
    # itself is intentionally reusable and tightly scoped.
    try:
        ts = int(timestamp_str)
        age = abs(time.time() - ts)
        if max_age is not None and age > max_age:
            bt.logging.warning(f"Signature too old: age={age:.0f}s (max={max_age}s)")
            return False
    except (ValueError, TypeError):
        return False

    # Recompute digest
    digest = hashlib.sha256(
        timestamp_str.encode() + nonce.encode() + body_bytes
    ).digest()

    # Verify
    try:
        signature = bytes.fromhex(sig_hex)
        keypair = Keypair(ss58_address=hotkey_ss58)
        return keypair.verify(digest, signature)
    except Exception as e:
        bt.logging.warning(f"Signature verification failed: {e}")
        return False


class NonceReplayCache:
    """In-memory replay cache for signed HTTP requests."""

    def __init__(self, max_age: int = MAX_REQUEST_AGE, cap: int = _NONCE_CACHE_CAP):
        self.max_age = max_age
        self.cap = cap
        self._seen: dict[tuple[str, str, str, str], float] = {}
        self._lock = threading.Lock()

    def check_and_record(
        self,
        namespace: str,
        hotkey_ss58: str,
        nonce: str,
        body_bytes: bytes,
    ) -> bool:
        """Return False if this exact signed request was already accepted."""
        now = time.time()
        cutoff = now - self.max_age
        body_hash = hashlib.sha256(body_bytes).hexdigest()
        key = (namespace, hotkey_ss58, nonce, body_hash)
        with self._lock:
            if len(self._seen) >= self.cap:
                self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
            if key in self._seen:
                return False
            self._seen[key] = now
            return True


_nonce_replay_cache = NonceReplayCache()


def check_and_record_nonce(
    namespace: str,
    hotkey_ss58: str,
    nonce: str,
    body_bytes: bytes,
) -> bool:
    """Return False if this exact signed request was already accepted."""
    return _nonce_replay_cache.check_and_record(
        namespace, hotkey_ss58, nonce, body_bytes,
    )


def extract_signature_headers(headers: dict) -> Optional[tuple[str, str, str, str]]:
    """Extract signature headers from a request.

    Returns (sig_hex, hotkey_ss58, timestamp, nonce) or None if missing.
    """
    sig = headers.get(HEADER_SIGNATURE) or headers.get(HEADER_SIGNATURE.lower())
    hotkey = headers.get(HEADER_HOTKEY) or headers.get(HEADER_HOTKEY.lower())
    timestamp = headers.get(HEADER_TIMESTAMP) or headers.get(HEADER_TIMESTAMP.lower())
    nonce = headers.get(HEADER_NONCE) or headers.get(HEADER_NONCE.lower())

    if not all([sig, hotkey, timestamp, nonce]):
        return None
    return sig, hotkey, timestamp, nonce
