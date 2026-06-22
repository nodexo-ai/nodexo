"""ComputeRegistry event indexer.

Builds two durable validator-local views from chain logs:

  * registry_executor_state — current on-chain executor inventory for fast
    browse/rent snapshots without polling the whole registry on every tick.
  * rental_windows — block-anchored rental timeline used by proof verification
    to decide whether a beacon block was free/heavy or rented/micro mode.

The chain remains authoritative. This index is a cached projection with a
paginated reconciliation fallback so missed logs, old deployments, or ABI drift
do not permanently poison renter-facing state.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time

import bittensor as bt

from common.types import ExecutorSpec


class ComputeRegistryEventIndexer:
    def __init__(
        self,
        registry_client,
        db,
        rental_state,
        *,
        chain_id: int = 0,
        deploy_block: int = 0,
        interval_s: float = 10.0,
        batch_blocks: int = 2_000,
        confirmations: int = 6,
        fallback_backfill_blocks: int = 100_000,
        reconcile_interval_s: float = 300.0,
        reconcile_page_size: int = 500,
    ):
        self.registry = registry_client
        self.db = db
        self.rental_state = rental_state
        self.chain_id = int(chain_id or 0)
        self.deploy_block = int(deploy_block or 0)
        self.interval_s = float(interval_s)
        self.batch_blocks = max(1, int(batch_blocks))
        self.confirmations = max(0, int(confirmations))
        self.fallback_backfill_blocks = max(1, int(fallback_backfill_blocks))
        self.reconcile_interval_s = max(0.0, float(reconcile_interval_s))
        self.reconcile_page_size = max(1, int(reconcile_page_size))
        self._last_reconcile = 0.0
        addr = str(getattr(registry_client.contract, "address", "")).lower()
        self.cursor_key = self.cursor_key_for(self.chain_id, addr)

    def _failure_backoff_s(self, failures: int) -> float:
        max_delay_s = max(
            self.interval_s,
            float(os.environ.get("REGISTRY_EVENT_INDEXER_FAILURE_BACKOFF_MAX_S", "300")),
        )
        base_s = max(0.1, self.interval_s)
        exponent = min(max(1, int(failures)), 8)
        return min(max_delay_s, base_s * (2 ** exponent))

    @staticmethod
    def cursor_key_for(chain_id: int, contract_address: str) -> str:
        return f"compute_registry_events:{int(chain_id or 0)}:{str(contract_address).lower()}"

    def _initial_cursor(self, latest: int) -> int:
        from common.db import get_chain_cursor

        cursor = get_chain_cursor(self.db, self.cursor_key)
        if cursor > 0:
            return cursor

        start = int(os.environ.get("COMPUTE_REGISTRY_DEPLOY_BLOCK", "0") or "0")
        if start <= 0:
            start = self.deploy_block
        if start <= 0:
            start = max(0, latest - self.fallback_backfill_blocks)
            bt.logging.warning(
                "ComputeRegistryEventIndexer: no compute_registry_deploy_block configured; "
                f"backfilling last {self.fallback_backfill_blocks} blocks from {start}. "
                "Set COMPUTE_REGISTRY_DEPLOY_BLOCK for full production history."
            )
        return max(0, start - 1)

    @staticmethod
    def _synthetic_rental_id(event: dict) -> str:
        raw = (
            f"{event.get('executor_id','')}:{event.get('block_number',0)}:"
            f"{event.get('log_index',0)}:{event.get('tx_hash','')}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def _spec_from_event(event: dict) -> ExecutorSpec:
        return ExecutorSpec(
            executor_id=str(event.get("executor_id", "")).removeprefix("0x"),
            miner_address=str(event.get("miner", "")),
            endpoint=str(event.get("endpoint", "")),
            gpu_model_hash=str(event.get("gpu_model_hash", "")).removeprefix("0x"),
            gpu_count=int(event.get("gpu_count") or 0),
            vram_mb=int(event.get("vram_mb") or 0),
            price_per_gpu_hour=int(event.get("price_per_gpu_hour") or 0),
            expires_at=int(event.get("expires_at") or 0),
            is_active=bool(event.get("is_active")),
            is_rented=bool(event.get("is_rented")),
            miner_uid=0,
            miner_registered=False,
            uid_owner_match=False,
        )

    def _fetch_current_spec(self, executor_id: str) -> ExecutorSpec | None:
        try:
            return self.registry.get_executor_info(bytes.fromhex(executor_id))
        except Exception as e:
            bt.logging.debug(
                f"registry event current-state fetch failed for {executor_id[:16]}: {e}"
            )
            return None

    def _apply_event(self, event: dict) -> None:
        from common.db import (
            mark_registry_executor_inactive,
            mark_registry_executor_rental_state,
            upsert_registry_executor_state,
        )

        executor_id = str(event.get("executor_id", "")).removeprefix("0x")
        if len(executor_id) != 64:
            return
        block = int(event.get("block_number") or 0)
        log_index = int(event.get("log_index") or 0)
        if block <= 0:
            return

        kind = str(event.get("kind") or "")
        if kind == "executor_state":
            # ExecutorStateChanged logs intentionally carry the cheap rental /
            # inventory fields, but not UID/EVM ownership metadata. Fetch the
            # current view before upserting so validator proof auth does not
            # interpret missing metadata as "owner EVM is not registered".
            spec = self._fetch_current_spec(executor_id)
            upsert_registry_executor_state(
                self.db,
                spec or self._spec_from_event(event),
                block_number=block,
                log_index=log_index,
                event_kind=kind,
            )
            return

        if event.get("needs_info"):
            spec = self._fetch_current_spec(executor_id)
            if spec is not None:
                upsert_registry_executor_state(
                    self.db,
                    spec,
                    block_number=block,
                    log_index=log_index,
                    event_kind=kind,
                )
            return

        if kind == "rental_started":
            mark_registry_executor_rental_state(
                self.db,
                executor_id,
                is_rented=True,
                block_number=block,
                log_index=log_index,
                event_kind=kind,
            )
            self.rental_state.record_chain_rented(
                executor_id,
                self._synthetic_rental_id(event),
                block,
            )
        elif kind == "rental_ended":
            mark_registry_executor_rental_state(
                self.db,
                executor_id,
                is_rented=False,
                block_number=block,
                log_index=log_index,
                event_kind=kind,
            )
            self.rental_state.record_available_for_executor(executor_id, block)
        elif kind == "executor_deactivated":
            mark_registry_executor_inactive(
                self.db,
                executor_id,
                removed=False,
                block_number=block,
                log_index=log_index,
                event_kind=kind,
            )
        elif kind == "executor_deregistered":
            mark_registry_executor_inactive(
                self.db,
                executor_id,
                removed=True,
                block_number=block,
                log_index=log_index,
                event_kind=kind,
            )

    def _latest_block(self) -> int:
        block_number_fn = getattr(self.registry, "block_number", None)
        if callable(block_number_fn):
            return int(block_number_fn())
        return int(getattr(self.registry.w3.eth, "block_number", 0) or 0)

    def sync_once(self, max_batches: int = 3) -> int:
        """Index up to `max_batches` chunks. Returns events applied."""
        from common.db import set_chain_cursor

        latest = self._latest_block() - self.confirmations
        if latest <= 0:
            return 0

        cursor = self._initial_cursor(latest)
        if cursor >= latest:
            return 0

        applied = 0
        for _ in range(max(1, int(max_batches))):
            from_block = cursor + 1
            if from_block > latest:
                break
            to_block = min(latest, from_block + self.batch_blocks - 1)
            events = self.registry.get_registry_events(from_block, to_block)
            for event in events:
                self._apply_event(event)
                applied += 1
            cursor = to_block
            set_chain_cursor(self.db, self.cursor_key, cursor)
            if cursor >= latest:
                break
        if applied:
            bt.logging.info(
                f"ComputeRegistryEventIndexer: applied {applied} event(s), cursor={cursor}"
            )
        return applied

    def reconcile_active_once(self) -> int:
        """Seed/correct current inventory from the active-list paginated view."""
        from common.db import (
            mark_registry_reconcile_absent_inactive,
            upsert_registry_executor_state,
        )

        execs = self.registry.get_all_active_executors(self.reconcile_page_size)
        active_ids = {e.executor_id for e in execs}
        for spec in execs:
            upsert_registry_executor_state(
                self.db,
                spec,
                event_kind="reconcile",
            )
        missing = mark_registry_reconcile_absent_inactive(
            self.db,
            active_ids,
        )
        self._last_reconcile = time.time()
        bt.logging.info(
            f"ComputeRegistryEventIndexer reconciled active inventory: "
            f"active={len(active_ids)} inactive_missing={missing}"
        )
        return len(active_ids)

    async def run(self) -> None:
        bt.logging.info(
            f"ComputeRegistryEventIndexer starting (interval={self.interval_s:.0f}s, "
            f"batch={self.batch_blocks} blocks, reconcile={self.reconcile_interval_s:.0f}s)"
        )
        max_batches = int(os.environ.get("REGISTRY_EVENT_INDEXER_MAX_BATCHES", "5"))
        if os.environ.get("REGISTRY_EVENT_INDEXER_RECONCILE_ON_START", "1") == "1":
            try:
                await asyncio.to_thread(self.reconcile_active_once)
            except Exception as e:
                bt.logging.warning(f"ComputeRegistryEventIndexer startup reconcile failed: {e}")
        failures = 0
        while True:
            sleep_s = self.interval_s
            try:
                await asyncio.to_thread(self.sync_once, max_batches)
                if (
                    self.reconcile_interval_s > 0
                    and time.time() - self._last_reconcile >= self.reconcile_interval_s
                ):
                    await asyncio.to_thread(self.reconcile_active_once)
                failures = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                failures += 1
                sleep_s = self._failure_backoff_s(failures)
                bt.logging.warning(
                    f"ComputeRegistryEventIndexer sync failed; retrying in "
                    f"{sleep_s:.0f}s: {e}"
                )
            await asyncio.sleep(sleep_s)


# Backwards-compatible name for tests/importers that only cared about rental
# windows. The class now indexes the full registry plus rental events.
RentalEventIndexer = ComputeRegistryEventIndexer
