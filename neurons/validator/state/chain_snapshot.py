"""Validator-local snapshot of ComputeRegistry state, refreshed on a schedule.

All read APIs (/marketplace/tiers, /executors, /rent's candidate selection,
verification's executor lookup) should serve from this snapshot, not hit
the chain directly. Direct RPC per request creates rate-limit cascades
and ties API responsiveness to chain health.

Refresh policy:
  - Every REFRESH_INTERVAL seconds, pull all executors via
    registry_client.get_all_active_executors(). On production registries this
    uses the active-list batched view, so refresh cost scales with active pages
    instead of stale/dead historical rows.
  - On failure: keep serving the previous snapshot (stale-OK), record the
    error, retry next tick. Snapshot age is exposed so callers can decide.
  - Eager invalidation: rent.py calls invalidate_executor(executor_id)
    after markRented/markAvailable so the snapshot updates without
    waiting for the next tick.

Write paths (markRented / markAvailable / register*) still hit the chain
directly — they're tx submissions, not reads, and need block-anchored
confirmation.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import bittensor as bt

from common.types import ExecutorSpec


REFRESH_INTERVAL = 30.0   # seconds between full chain pulls. Bumped from 15
                          # to ease pressure on the public testnet RPC; with
                          # eager invalidation on markRented/markAvailable
                          # the snapshot stays accurate for the renter UX
                          # even at 30s.
BACKOFF_CAP_S = 600.0     # ceiling for the 429-aware backoff (10 min).
                          # Doubling sequence: 30 → 60 → 120 → 240 → 480 → 600.
                          # On first successful refresh the backoff resets
                          # to REFRESH_INTERVAL. Stops the validator from
                          # hammering a rate-limited RPC and turning a
                          # transient 429 into a sustained outage; production
                          # validators with their own subtensor node won't
                          # hit this in practice.
STALE_WARN_AFTER = 120.0  # warn if snapshot hasn't refreshed this long
SNAPSHOT_PAGE_SIZE = int(os.environ.get("CHAIN_SNAPSHOT_PAGE_SIZE", "500"))
INDEX_STALE_AFTER = float(os.environ.get("CHAIN_SNAPSHOT_INDEX_STALE_AFTER_S", "90"))
INDEX_MAX_BLOCK_LAG = int(os.environ.get("CHAIN_SNAPSHOT_INDEX_MAX_BLOCK_LAG", "30"))
INDEX_ALLOW_STALE_ON_RPC_FAILURE = (
    os.environ.get("CHAIN_SNAPSHOT_STALE_INDEX_ON_RPC_FAILURE", "1") == "1"
)


class ChainSnapshot:
    """In-memory snapshot of all on-chain executors, refreshed by a loop.

    Thread-safe reads via a single RLock. The writer is the refresh loop;
    occasional eager invalidations from rent.py also hold the lock briefly.
    """

    def __init__(self, registry_client, *, db=None, registry_cursor_key: str = ""):
        self._registry = registry_client
        self._db = db
        self._registry_cursor_key = registry_cursor_key
        self._prefer_index = os.environ.get("CHAIN_SNAPSHOT_USE_REGISTRY_INDEX", "1") == "1"
        self._executors: dict[str, ExecutorSpec] = {}
        self._last_updated: float = 0.0
        self._last_error: Optional[str] = None
        self._lock = threading.RLock()
        self._task: Optional[asyncio.Task] = None

    # ── Reads ─────────────────────────────────────────────────────

    def all_executors(self) -> list[ExecutorSpec]:
        """All known executors, regardless of state."""
        with self._lock:
            return list(self._executors.values())

    def available_executors(self) -> list[ExecutorSpec]:
        """Available = is_active AND lease not expired AND not currently rented.

        This is the renter-facing 'rentable now' filter.
        """
        now = int(time.time())
        with self._lock:
            filtered = []
            dropped = []
            for e in self._executors.values():
                if e.is_active and not e.is_rented and e.expires_at > now:
                    filtered.append(e)
                else:
                    dropped.append(
                        f"{e.executor_id[:16]} active={e.is_active} "
                        f"rented={e.is_rented} expires={e.expires_at} now={now}"
                    )
        bt.logging.debug(
            f"available_executors: have={len(self._executors)} kept={len(filtered)} "
            f"dropped_sample={dropped[:10]} dropped_count={len(dropped)}"
        )
        return filtered

    def get(self, executor_id: str) -> Optional[ExecutorSpec]:
        with self._lock:
            return self._executors.get(executor_id)

    def age_seconds(self) -> float | None:
        """Seconds since the snapshot last refreshed, or None if it never has.

        Returns None (not float('inf')) so callers can JSON-encode safely:
        float('inf') isn't JSON-compliant and serializing it triggers an
        HTTP 500 in FastAPI's default encoder.
        """
        if not self._last_updated:
            return None
        return time.time() - self._last_updated

    def is_fresh(self) -> bool:
        age = self.age_seconds()
        return age is not None and age < STALE_WARN_AFTER

    # ── Eager invalidation hook ──────────────────────────────────

    def invalidate_executor(self, executor_id: str) -> None:
        """Re-fetch one executor's state from chain immediately (used after
        markRented / markAvailable so the snapshot doesn't lag 15s).
        """
        if not self._registry:
            return
        try:
            spec = self._registry.get_executor_info(bytes.fromhex(executor_id))
        except Exception as e:
            bt.logging.debug(f"invalidate_executor({executor_id[:16]}): {e}")
            return
        with self._lock:
            if spec:
                self._executors[executor_id] = spec
            else:
                self._executors.pop(executor_id, None)
        if spec and self._db is not None:
            try:
                from common.db import upsert_registry_executor_state
                block_number_fn = getattr(self._registry, "block_number", None)
                block_number = int(
                    block_number_fn()
                    if callable(block_number_fn)
                    else getattr(self._registry.w3.eth, "block_number", 0) or 0
                )
                upsert_registry_executor_state(
                    self._db,
                    spec,
                    block_number=block_number,
                    log_index=0,
                    event_kind="direct",
                )
            except Exception as e:
                bt.logging.debug(
                    f"invalidate_executor({executor_id[:16]}) index update failed: {e}"
                )

    # ── Background refresh loop ──────────────────────────────────

    async def run(self):
        """Refresh loop with 429-aware backoff. Cancel via task.cancel()
        at shutdown.

        Cadence:
          * Healthy path: refresh every REFRESH_INTERVAL.
          * 429-rate-limited: sleep `_backoff_s`, doubling each
            consecutive failure up to BACKOFF_CAP_S. On first success
            the backoff resets to REFRESH_INTERVAL. Stops the validator
            from hammering a rate-limited public RPC and turning a
            transient 429 into a 20-minute outage.
          * Non-429 errors (e.g. transient TCP) follow the same
            backoff because they tend to come in clusters too.
        """
        bt.logging.info(
            f"ChainSnapshot starting (refresh every {REFRESH_INTERVAL:.0f}s)"
        )
        # Eager first pull so the first API request after startup has data.
        ok = await self._refresh_once()
        # Backoff state: starts at REFRESH_INTERVAL on success path.
        backoff = REFRESH_INTERVAL if ok else min(REFRESH_INTERVAL * 2, BACKOFF_CAP_S)
        while True:
            await asyncio.sleep(backoff)
            ok = await self._refresh_once()
            if ok:
                backoff = REFRESH_INTERVAL
            else:
                backoff = min(backoff * 2, BACKOFF_CAP_S)

    async def _refresh_once(self) -> bool:
        """Returns True iff the refresh landed cleanly. False on any
        upstream failure (caller adjusts retry cadence)."""
        indexed = self._load_indexed_executors(allow_stale=False)
        if indexed is not None:
            execs, source_ts = indexed
            with self._lock:
                self._executors = {e.executor_id: e for e in execs}
                self._last_updated = source_ts or time.time()
                self._last_error = None
            bt.logging.debug(
                f"ChainSnapshot refreshed from registry index: count={len(execs)}"
            )
            return True

        try:
            execs = await asyncio.to_thread(
                self._registry.get_all_active_executors,
                SNAPSHOT_PAGE_SIZE,
            )
            total = getattr(self._registry, "last_executor_total", len(execs))
        except Exception as e:
            stale_index = (
                self._load_indexed_executors(allow_stale=True)
                if INDEX_ALLOW_STALE_ON_RPC_FAILURE else None
            )
            if stale_index is not None:
                execs, source_ts = stale_index
                with self._lock:
                    self._executors = {ex.executor_id: ex for ex in execs}
                    self._last_updated = source_ts or self._last_updated
                    self._last_error = str(e)
                bt.logging.warning(
                    f"ChainSnapshot live refresh failed; serving stale registry "
                    f"index age={(time.time() - (source_ts or 0)):.0f}s "
                    f"count={len(execs)}"
                )
                return False
            with self._lock:
                self._last_error = str(e)
            msg = str(e)
            is_rate_limited = ("429" in msg) or ("Too Many Requests" in msg)
            level = bt.logging.warning if is_rate_limited else bt.logging.warning
            tag = "429 rate-limited" if is_rate_limited else type(e).__name__
            level(
                f"ChainSnapshot refresh failed ({tag}); "
                f"serving snapshot from {(self.age_seconds() or 0):.0f}s ago; "
                f"backing off"
            )
            return False
        with self._lock:
            self._executors = {e.executor_id: e for e in execs}
            self._last_updated = time.time()
            self._last_error = None
        bt.logging.info(
            f"ChainSnapshot refreshed: count={total}, "
            f"active-and-unexpired={len(execs)}, "
            f"ids={[e.executor_id[:16] for e in execs]}"
        )
        return True

    def _load_indexed_executors(self, *, allow_stale: bool) -> tuple[list[ExecutorSpec], float] | None:
        if not self._prefer_index or self._db is None or not self._registry_cursor_key:
            return None
        try:
            from common.db import (
                get_chain_cursor_status,
                get_registry_executor_specs,
                registry_executor_state_count,
            )
            status = get_chain_cursor_status(self._db, self._registry_cursor_key)
            updated = status.get("updated_at")
            if not updated:
                return None
            dt = datetime.fromisoformat(str(updated))
            if dt.tzinfo is None:
                ts = dt.replace(tzinfo=timezone.utc).timestamp()
            else:
                ts = dt.timestamp()
            if not allow_stale and time.time() - ts > INDEX_STALE_AFTER:
                return None
            cursor_block = int(status.get("block_number") or 0)
            if not allow_stale and INDEX_MAX_BLOCK_LAG > 0:
                try:
                    confirmations = int(os.environ.get(
                        "REGISTRY_EVENT_INDEX_CONFIRMATIONS",
                        os.environ.get("RENTAL_EVENT_INDEX_CONFIRMATIONS", "6"),
                    ))
                    block_number_fn = getattr(self._registry, "block_number", None)
                    latest_raw = (
                        block_number_fn()
                        if callable(block_number_fn)
                        else int(self._registry.w3.eth.block_number)
                    )
                    latest = int(latest_raw) - max(0, confirmations)
                    if cursor_block < latest - INDEX_MAX_BLOCK_LAG:
                        return None
                except Exception:
                    if cursor_block <= 0:
                        return None
            if registry_executor_state_count(self._db) <= 0:
                return None
            return get_registry_executor_specs(
                self._db,
                active_only=True,
                unexpired_only=True,
                include_removed=False,
            ), ts
        except Exception as e:
            bt.logging.debug(f"ChainSnapshot registry index unavailable: {e}")
            return None

    def _load_indexed_executors_if_fresh(self) -> list[ExecutorSpec] | None:
        indexed = self._load_indexed_executors(allow_stale=False)
        return indexed[0] if indexed is not None else None
