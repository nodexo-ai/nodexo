"""Validator-local index of rental windows over time.

Lets verify_recipe answer "was this executor rented at block N?" without
doing a chain history query (which Substrate doesn't natively support).
Indexes this validator's own markRented / markAvailable calls plus
ComputeRegistry RentalStarted/RentalEnded events reconstructed by the
background event indexer. Event-indexed peer rentals are proof-mode context
only: they must not be returned as locally-owned rentals to terminate.

Without this, the race is real: miner produces a HEAVY proof at block N,
validator markRenteds at N+1, validator verifies at N+5 with
ctx.is_rented=True (current chain state) → expects MICRO N → mismatch →
falsely rejects an honest proof.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class RentalWindow:
    rental_id: str
    executor_id: str
    start_block: int                # block of the markRented tx
    end_block: int | None = None    # block of the markAvailable tx, None=active
    owned: bool = True              # True only for rentals created by this validator


class RentalState:
    """Thread-safe in-memory index of rentals.

    Two indexes for two query shapes:
      - executor_id → [window, …] in start-time order for `was_rented_at`
      - rental_id → executor_id for fast end_block updates
    """

    def __init__(self, db=None):
        self._by_executor: dict[str, list[RentalWindow]] = {}
        self._by_rental: dict[str, str] = {}
        self._lock = threading.Lock()
        # DB-backed (audit C-7): without persistence, validator
        # restart loses the entire block-anchored timeline and
        # `was_rented_at()` falls through to live chain state,
        # re-opening the cycle-window race this index exists to fix.
        self._db = db

    def load_from_db(self) -> None:
        """Hydrate the in-memory index from RentalWindowRow at startup.
        Safe to call multiple times; idempotent on rental_id."""
        if self._db is None:
            return
        try:
            from common.db import get_all_rental_windows, get_owned_rental_ids
            rows = get_all_rental_windows(self._db)
            owned_ids = get_owned_rental_ids(self._db)
        except Exception:
            return
        with self._lock:
            for r in rows:
                rid = r["rental_id"]
                if rid in self._by_rental:
                    continue
                eid = r["executor_id"]
                start_b = int(r["start_block"] or 0)
                end_b = r["end_block"] if r["end_block"] > 0 else None
                if end_b is not None and end_b < start_b:
                    end_b = start_b
                w = RentalWindow(
                    rental_id=rid, executor_id=eid,
                    start_block=start_b, end_block=end_b,
                    owned=(rid in owned_ids),
                )
                self._by_executor.setdefault(eid, []).append(w)
                self._by_rental[rid] = eid

    def _persist_open(self, eid: str, rid: str, block: int) -> None:
        if self._db is None:
            return
        try:
            from common.db import add_rental_window
            add_rental_window(self._db, rid, eid, block)
        except Exception:
            pass

    def _persist_close(self, rid: str, block: int) -> None:
        if self._db is None:
            return
        try:
            from common.db import close_rental_window
            close_rental_window(self._db, rid, block)
        except Exception:
            pass

    def record_rented(self, executor_id: str, rental_id: str, block: int,
                      owned: bool = True) -> None:
        should_persist = False
        with self._lock:
            existing_eid = self._by_rental.get(rental_id)
            if existing_eid:
                for w in self._by_executor.get(existing_eid, []):
                    if w.rental_id == rental_id:
                        if not w.start_block and block:
                            w.start_block = block
                        w.owned = w.owned or owned
                        return
            w = RentalWindow(rental_id=rental_id, executor_id=executor_id,
                             start_block=block, end_block=None, owned=owned)
            self._by_executor.setdefault(executor_id, []).append(w)
            self._by_rental[rental_id] = executor_id
            should_persist = True
        if should_persist:
            self._persist_open(executor_id, rental_id, block)

    def record_chain_rented(self, executor_id: str, rental_id: str,
                            block: int) -> None:
        """Record a RentalStarted event from chain logs.

        If we already have an open local window for the executor, do not add a
        synthetic peer window. ComputeRegistry enforces one active rental per
        executor, so the local window is the same chain lock and carries the
        real rental_id needed by our API.
        """
        with self._lock:
            for w in self._by_executor.get(executor_id, []):
                if w.end_block is None:
                    return
                if w.start_block == block:
                    return
        self.record_rented(executor_id, rental_id, block, owned=False)

    def record_available(self, rental_id: str, block: int) -> None:
        with self._lock:
            eid = self._by_rental.get(rental_id)
            if not eid:
                return
            for w in self._by_executor.get(eid, []):
                if w.rental_id == rental_id and w.end_block is None:
                    if block < w.start_block:
                        return
                    w.end_block = block
        self._persist_close(rental_id, block)

    def record_available_for_executor(self, executor_id: str, block: int) -> None:
        """Close open windows for an executor from a RentalEnded chain event."""
        closed: list[str] = []
        with self._lock:
            for w in self._by_executor.get(executor_id, []):
                if w.end_block is None and w.start_block <= block:
                    w.end_block = block
                    closed.append(w.rental_id)
        for rid in closed:
            self._persist_close(rid, block)

    def was_rented_at(self, executor_id: str, block: int) -> bool:
        """Was this executor under rent at block `block`?

        True when there exists a window with start_block <= block < end_block
        (or end_block is None, meaning still active).
        """
        with self._lock:
            for w in self._by_executor.get(executor_id, []):
                if w.start_block <= block and (w.end_block is None or w.end_block > block):
                    return True
        return False

    def ended_within(self, executor_id: str, block: int, grace_blocks: int) -> bool:
        """Did a rental for this executor end within `grace_blocks` of `block`?

        Used only for proof-mode transition handling. If a rental ends at the
        beacon boundary, the miner may still see the pre-end chain state from
        its RPC and submit one micro proof while the validator already sees the
        executor as free. That proof should be neutral, not a hard failure.
        """
        if block <= 0 or grace_blocks < 0:
            return False
        with self._lock:
            for w in self._by_executor.get(executor_id, []):
                if w.end_block is None:
                    continue
                if w.end_block <= block <= w.end_block + grace_blocks:
                    return True
        return False

    def started_within_after(self, executor_id: str, block: int, grace_blocks: int) -> bool:
        """Did a rental for this executor start shortly after `block`?

        Used only for proof-mode transition handling. If a rental starts just
        after the beacon block, a miner may observe the new chain state before
        generating the recipe and submit a micro proof while the validator's
        beacon-anchored context still says free. That proof is neutral, not a
        heavy/free pass and not a miner failure.
        """
        if block <= 0 or grace_blocks < 0:
            return False
        with self._lock:
            for w in self._by_executor.get(executor_id, []):
                if block < w.start_block <= block + grace_blocks:
                    return True
        return False

    def has_active_rental(self, executor_id: str, owned_only: bool = True) -> str | None:
        with self._lock:
            for w in reversed(self._by_executor.get(executor_id, [])):
                if w.end_block is None and (not owned_only or w.owned):
                    return w.rental_id
        return None

    def has_history(self, executor_id: str) -> bool:
        """Has this validator ever rented this executor? Used to decide whether
        to trust the local block-anchored answer or fall back to live chain.
        """
        with self._lock:
            return bool(self._by_executor.get(executor_id))

    def all_active(self) -> list[RentalWindow]:
        with self._lock:
            return [w for ws in self._by_executor.values() for w in ws if w.end_block is None]
