"""
SR25519-signed-validator auth for sensitive miner endpoints.

Lifecycle endpoints (POST /containers, DELETE /containers/{name}, GET
/containers) are RCE-equivalent: an unauthenticated caller can spin
arbitrary Docker containers on the miner host. Gate them behind a
hotkey signature from a known validator.

Allowlist sources (union):
  1. env NODEXO_ALLOWED_VALIDATOR_HOTKEYS — comma-separated SS58 list.
     Read on every request so operators can rotate keys without a restart.
  2. ValidatorRegistry chain reads via uid → metagraph hotkey lookup,
     refreshed every CHAIN_REFRESH_INTERVAL seconds.

Signing: common.crypto.sign_payload / verify_signature (SR25519 over
timestamp + nonce + body, 120-second freshness window).
"""
from __future__ import annotations

import os
import threading
import time

import bittensor as bt
from fastapi import HTTPException, Request

from common.crypto import (
    extract_signature_headers, verify_signature, MAX_REQUEST_AGE,
)


CHAIN_REFRESH_INTERVAL = 300.0  # 5 min between on-chain allowlist refreshes
CHAIN_REFRESH_FAILURE_BACKOFF_CAP = 1800.0  # 30 min max retry delay on RPC outage

# Replay window — the signature freshness check in verify_signature
# accepts any request whose timestamp is within MAX_REQUEST_AGE
# seconds, but it does NOT remember which nonces it has already
# accepted. A network attacker who captures one signed POST /containers
# request can re-submit it within that window and spin a second
# container under the validator's identity. The nonce cache below
# closes that hole — we reject any (hotkey, nonce) pair we've seen
# inside MAX_REQUEST_AGE.
#
# Cache size cap: a misbehaving validator can flood new nonces, but
# we bound the dict to avoid unbounded memory growth. At 5000 entries
# and ~80 bytes per (str hotkey + str nonce + float ts) tuple, the
# upper-bound footprint is ~400 KB — tiny relative to a python
# process and well over what any legit traffic pattern needs in 120s.
_NONCE_CACHE_CAP = 5000


class ValidatorAuth:
    """Verifies SR25519-signed requests from registered validators.

    Construct once in lifespan, then use as a FastAPI dependency:
        auth = ValidatorAuth(validator_registry=vali_reg, rpc=rpc)
        @app.post("/containers", dependencies=[Depends(auth)])
    """

    def __init__(self, validator_registry=None, rpc=None):
        self._validator_registry = validator_registry
        self._rpc = rpc
        self._chain_allowed: set[str] = set()
        self._last_refresh: float = 0.0
        self._next_refresh_after: float = 0.0
        self._refresh_failures: int = 0
        self._lock = threading.Lock()
        # Replay-protection cache. Key: (hotkey_ss58, nonce). Value:
        # the time we accepted that signature. Entries older than
        # MAX_REQUEST_AGE can never be replayed (verify_signature will
        # reject them as stale) so we drop them lazily.
        self._seen_nonces: dict[tuple[str, str], float] = {}
        self._nonce_lock = threading.Lock()

    def _record_nonce(self, hotkey: str, nonce: str) -> bool:
        """Return True if this (hotkey, nonce) is fresh; False if it's
        already been used inside the freshness window. Side effect:
        records the nonce on True.

        Caller has already verified the signature + freshness; this is
        the last gate before we declare the request valid.
        """
        now = time.time()
        key = (hotkey, nonce)
        with self._nonce_lock:
            # Lazy purge — drop expired entries when we touch the dict.
            # Bound the cost: only scan if it's getting full. The
            # MAX_REQUEST_AGE bound means even unscanned entries time out
            # by signature freshness, so the cache can never falsely
            # reject a legit request even if purging is delayed.
            if len(self._seen_nonces) >= _NONCE_CACHE_CAP:
                cutoff = now - MAX_REQUEST_AGE
                self._seen_nonces = {
                    k: v for k, v in self._seen_nonces.items() if v > cutoff
                }
            if key in self._seen_nonces:
                return False
            self._seen_nonces[key] = now
            return True

    def _env_allowed(self) -> set[str]:
        raw = os.environ.get("NODEXO_ALLOWED_VALIDATOR_HOTKEYS", "")
        return {h.strip() for h in raw.split(",") if h.strip()}

    def _runtime_allowed(self) -> set[str]:
        try:
            from common.subnet_runtime_config import get_subnet_runtime_config
            return {
                h.strip()
                for h in get_subnet_runtime_config().network.rental_validator_hotkeys
                if h.strip()
            }
        except Exception:
            return set()

    def _strict_mode(self) -> bool:
        """Strict mode disables the chain-allowlist entirely: only the
        explicit env list is trusted. Use this when you want to whitelist
        ONLY your own validator(s) and refuse to serve any peer validator
        that happens to be on the ValidatorRegistry. Recommended for
        production single-validator deployments.

        Default off so a fresh miner with no env config still works on
        a populated ValidatorRegistry.
        """
        return os.environ.get("NODEXO_STRICT_ALLOWLIST", "0") == "1"

    def _maybe_refresh_chain(self) -> None:
        if not self._validator_registry or not self._rpc:
            return
        now = time.time()
        with self._lock:
            if now < self._next_refresh_after:
                return
            if now - self._last_refresh < CHAIN_REFRESH_INTERVAL:
                return
            # Claim the refresh slot before doing blocking chain I/O. Without
            # this, concurrent control requests can all notice the stale cache
            # and stampede the same RPC endpoint.
            self._next_refresh_after = now + CHAIN_REFRESH_INTERVAL
        try:
            valis = self._validator_registry.get_active_validators()
            uids = [v.uid for v in valis if getattr(v, "is_active", True)]
            mg = self._rpc.get_metagraph(False)
            hk_list = getattr(mg, "hotkeys", [])
            chain_set = {hk_list[u] for u in uids if 0 <= u < len(hk_list)}
            env_set = self._env_allowed() | self._runtime_allowed()
            # Drift detection: chain hotkeys not in env mean a peer
            # validator could (in non-strict mode) provision containers
            # on this miner. Surface once per refresh so the operator
            # can decide to switch to strict mode or extend the env list.
            drift = chain_set - env_set
            if env_set and drift:
                bt.logging.warning(
                    f"ValidatorAuth: chain allowlist has {len(drift)} hotkey(s) "
                    f"not in NODEXO_ALLOWED_VALIDATOR_HOTKEYS env. "
                    f"In non-strict mode these CAN provision containers. "
                    f"Set NODEXO_STRICT_ALLOWLIST=1 to refuse them. "
                    f"Sample: {next(iter(drift))[:12]}..."
                )
            with self._lock:
                self._chain_allowed = chain_set
                self._last_refresh = now
                self._next_refresh_after = now + CHAIN_REFRESH_INTERVAL
                self._refresh_failures = 0
            bt.logging.debug(
                f"ValidatorAuth: chain allowlist refreshed ({len(chain_set)} hotkeys)"
            )
        except Exception as e:
            with self._lock:
                self._last_refresh = now
                self._refresh_failures += 1
                backoff = min(
                    CHAIN_REFRESH_FAILURE_BACKOFF_CAP,
                    CHAIN_REFRESH_INTERVAL * (2 ** max(0, self._refresh_failures - 1)),
                )
                self._next_refresh_after = now + backoff
            bt.logging.debug(f"ValidatorAuth: chain refresh failed: {e}")

    def allowed(self) -> set[str]:
        if self._strict_mode():
            # Strict: ignore chain entirely. Operator env and subnet runtime
            # config are the only trusted rental-controller sources.
            return self._env_allowed() | self._runtime_allowed()
        self._maybe_refresh_chain()
        with self._lock:
            chain = set(self._chain_allowed)
        return chain | self._env_allowed() | self._runtime_allowed()

    async def __call__(self, request: Request):
        allowed = self.allowed()
        if not allowed:
            # No allowlist configured at all — refuse rather than open.
            # Operators MUST set NODEXO_ALLOWED_VALIDATOR_HOTKEYS or supply
            # a populated ValidatorRegistry for sensitive routes to work.
            raise HTTPException(
                503,
                "Miner auth not configured: set NODEXO_ALLOWED_VALIDATOR_HOTKEYS "
                "or run with a ValidatorRegistry-bound chain config.",
            )

        headers = dict(request.headers)
        extracted = extract_signature_headers(headers)
        if extracted is None:
            raise HTTPException(401, "Missing signature headers")
        sig_hex, hotkey_ss58, timestamp, nonce = extracted

        if hotkey_ss58 not in allowed:
            bt.logging.warning(
                f"auth: rejecting hotkey {hotkey_ss58[:12]}… "
                f"(not in allowlist of {len(allowed)})"
            )
            raise HTTPException(403, "Validator hotkey not allowed")

        # Reading the body here caches it on the request object, so the
        # downstream handler's `await request.json()` returns the same
        # bytes without trying to re-consume the receive stream.
        body = await request.body()
        if not verify_signature(body, sig_hex, hotkey_ss58, timestamp, nonce):
            raise HTTPException(401, "Invalid signature")

        # Replay protection — even a valid signature can't be re-used
        # inside the freshness window. See _record_nonce comment.
        if not self._record_nonce(hotkey_ss58, nonce):
            bt.logging.warning(
                f"auth: replay rejected for hotkey {hotkey_ss58[:12]}… "
                f"nonce={nonce[:12]}…"
            )
            raise HTTPException(401, "Nonce reuse (replay)")

        request.state.validator_hotkey = hotkey_ss58
