"""
Canary scheduler — background asyncio task that picks the next
executor to canary and runs the canary pipeline.

POLICY
======

For each known active executor E:
  - "last canaried at" persisted in DB (CanaryRecord table — added
    by `init_canary_table` lazily on first scheduler tick if absent).
  - If E has NO canary record → eligible immediately (new arrival
    or first scheduler run). Highest priority — verify before any
    real renter pays.
  - If E.last_canaried_at < (now - period) → eligible. Period is
    configurable (default 12h).
  - Eligible executors are run one at a time. A canary takes ~7-8
    minutes; running serially means at most ~5/hour. Plenty for a
    small fleet.

The scheduler also can be triggered on-demand by:
  - The validator's sybil scanner opening a soft signal → escalate
    to canary.
  - An admin HTTP endpoint.

CONCURRENCY
===========

ONE canary at a time. Multiple canaries running concurrently would
collide on the validator's gas allowance and on chain RPC limits.
"""
from __future__ import annotations

import asyncio
import os
import secrets
import time
from typing import Optional

import bittensor as bt

from common.subnet_runtime_config import get_subnet_runtime_config


# Canary cadence is a strict MEMORYLESS Poisson process — there is
# no floor and no ceiling. Every tick the scheduler runs an HMAC-keyed
# Bernoulli trial per executor with probability `tick / mean`. The
# core property: P(canary in next N seconds | last canary was X seconds
# ago) is INDEPENDENT of X. The miner can never say "I just got
# canaried, I have a safe window now" — the probability of the next
# canary is the same in the next minute whether the last one was 60s
# or 60h ago.
#
# Per-executor inter-canary distribution: Exponential(λ=1/mean).
#   Mean inter-canary = CANARY_MEAN_SECONDS (default 24h)
#   Per-tick probability ≈ 60s / 86400s ≈ 0.069%
#   1% of gaps will be < 14 minutes (deliberately — denies "safe window")
#   1% of gaps will be > 4.6 × mean (acceptable; we re-canary anyway)
#
# Serial guarantee: the scheduler holds an _inflight slot, so even if
# multiple executors roll-pass on one tick, only one canary runs at a
# time. Others just retry on subsequent ticks (still memoryless).
CANARY_MEAN_SECONDS_DEFAULT = 24 * 3600
SCHEDULER_TICK_SECONDS      = 60
# First-canary errors usually mean a transient SSH/container issue, not
# a vetted result. Retry quickly so a healthy miner is not hidden from
# rentals for the 24h Poisson mean after one transport failure.
CANARY_ERROR_RETRY_SECONDS  = 15 * 60

# Consecutive canary errors that flip the executor to BANNED. Without
# this, a miner could SABOTAGE pip-install / network egress so every
# canary completes with status=error (not fail), bypassing the
# canary_failed gate while keeping last_canary_status=pass from a
# prior run. After N errors in a row, treat as effective failure.
CANARY_ERROR_STREAK_THRESHOLD = 3


def _is_validator_transient_canary_result(result: dict) -> bool:
    """True when the canary did not reach the miner because validator-side
    chain/RPC control-plane work failed.

    These attempts are operationally important, but they are not miner canary
    errors. Counting them toward canary_error_streak hides honest hardware when
    the public EVM RPC is rate-limited.
    """
    if result.get("status") != "error":
        return False
    reason = str(result.get("reason") or "").lower()
    # markRented is validator-side chain control-plane work that happens
    # before the canary reaches the miner. Any failure here means the canary
    # never tested the executor, so it must not count as miner sabotage.
    if "markrented failed" in reason:
        return True
    if "runner crashed" not in reason:
        return False
    return any(
        token in reason
        for token in (
            "429",
            "too many requests",
            "rpc may be throttled",
            "receipt timed out",
            "timeout",
            "connection",
        )
    )


def _load_exclude_set() -> set[str]:
    """Read CANARY_EXCLUDE_EXECUTOR_IDS (comma-separated 64-char hex
    ids) from env. The scheduler skips any executor whose id is in
    this set. Use this to permanently exclude a known-bad host (e.g.
    a defective local card that shouldn't take any GPU load) without
    relying on chain deregistration alone.
    """
    raw = os.environ.get("CANARY_EXCLUDE_EXECUTOR_IDS", "")
    if not raw:
        return set()
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def _is_local_endpoint(endpoint: str) -> bool:
    """Endpoint loopback / private-net heuristic. Validator-local
    endpoints are almost always wrong for canary (they'd load the
    same box the validator runs on)."""
    if not endpoint:
        return True  # no endpoint to reach
    ep = endpoint.lower()
    return (
        "127.0.0.1" in ep
        or "localhost" in ep
        or "://0.0.0.0" in ep
    )


class CanaryScheduler:
    """Background canary scheduler. Subscribe via run() in lifespan."""

    def __init__(
        self,
        runner,                           # CanaryRunner
        get_active_executors,             # () -> list[ExecutorSpec]
        db,
        mean_seconds: int = CANARY_MEAN_SECONDS_DEFAULT,
        jitter_secret: Optional[bytes] = None,
    ):
        self._runner = runner
        self._get_active = get_active_executors
        self._db = db
        self._mean_s = max(mean_seconds, SCHEDULER_TICK_SECONDS * 2)
        # Validator-secret keys the per-(executor, tick) HMAC so the
        # miner can't predict their own next canary time without the
        # validator's hotkey seed. Optional: if absent, falls back to
        # os.urandom() at startup (still unpredictable to the miner;
        # just non-reproducible across validator restarts, which is
        # fine — restart re-randomises the schedule too).
        self._jitter_secret = jitter_secret or os.urandom(32)
        self._stop = False
        self._inflight: Optional[str] = None
        self._exclude = _load_exclude_set()
        if self._exclude:
            bt.logging.info(
                f"CanaryScheduler exclude list: "
                f"{[eid[:16] for eid in self._exclude]}"
            )
        bt.logging.info(
            f"CanaryScheduler cadence: memoryless Poisson, mean="
            f"{self._mean_s}s, per-tick p={SCHEDULER_TICK_SECONDS/self._mean_s:.5f}"
        )

    def stop(self) -> None:
        self._stop = True

    async def run(self) -> None:
        """Main loop: every SCHEDULER_TICK_SECONDS, pick the next
        canary target and run it. Serial — never more than one canary
        in flight at a time.
        """
        bt.logging.info(
            f"CanaryScheduler started (mean={self._mean_s}s, "
            f"tick={SCHEDULER_TICK_SECONDS}s)"
        )
        while not self._stop:
            try:
                target = self._pick_next_target()
                if target is None:
                    await asyncio.sleep(SCHEDULER_TICK_SECONDS)
                    continue
                self._inflight = target["executor_id"]
                # Persist `inflight_at` so a validator crash during the
                # ~8min run can be reconciled at next boot. The runner's
                # `_record_canary` clears this in its finally; on crash
                # we leave it stamped and reconcile_canary_inflight()
                # picks it up next startup.
                self._set_inflight_db(target["executor_id"], inflight=True)
                bt.logging.info(
                    f"CanaryScheduler: picking executor={target['executor_id'][:16]} "
                    f"({target.get('reason', 'periodic')})"
                )
                result: Optional[dict] = None
                try:
                    result = await self._runner.run_canary(
                        executor_id=target["executor_id"],
                        executor_endpoint=target["endpoint"],
                        gpu_model_name=target["gpu_model_name"],
                        gpu_count=target.get("gpu_count", 1),
                    )
                    bt.logging.info(
                        f"CanaryScheduler: executor={target['executor_id'][:16]} "
                        f"status={result['status']} reason={result.get('reason', '')[:80]}"
                    )
                except Exception as e:
                    # CRITICAL: if run_canary itself throws, we still
                    # MUST record an error outcome so consecutive_errors
                    # advances. Otherwise a miner who reliably crashes
                    # the runner (network fault, container resource
                    # limit, etc.) keeps last_status='pass' forever and
                    # bypasses both the canary_failed gate AND the
                    # canary_error_streak gate.
                    bt.logging.error(
                        f"CanaryScheduler: canary on {target['executor_id'][:16]} "
                        f"crashed: {type(e).__name__}: {e}"
                    )
                    result = {
                        "status": "error",
                        "executor_id": target["executor_id"],
                        "reason": f"runner crashed: {type(e).__name__}: {str(e)[:200]}",
                    }
                finally:
                    # Always persist outcome — pass, fail, or error.
                    # consecutive_errors increments on error, resets on pass/fail.
                    if result is not None:
                        try:
                            self._record_canary(target["executor_id"], result)
                        except Exception as e:
                            bt.logging.warning(
                                f"_record_canary post-canary write failed: {e}"
                            )
                    self._inflight = None
                # Brief pause before considering next target — gives
                # chain state time to settle from the markAvailable.
                await asyncio.sleep(30)
            except Exception as e:
                bt.logging.error(f"CanaryScheduler tick failed: {e}")
                await asyncio.sleep(SCHEDULER_TICK_SECONDS)

    def _pick_next_target(self) -> Optional[dict]:
        """Return the highest-priority eligible executor, or None.

        Priority:
          1. Never-canaried AND has at least one valid proof on record
             (new arrivals after first successful scoring → first canary
             unlocks /rent for them).
          2. Executors whose randomized next-canary time has elapsed,
             picked randomly from the overdue set (not deterministic
             oldest-first) so a miner can't predict ordering.
        """
        import hmac
        import hashlib
        import time as _t

        try:
            execs = self._get_active()
        except Exception as e:
            bt.logging.debug(f"CanaryScheduler get_active failed: {e}")
            return None
        if not execs:
            return None

        # Apply exclusions: rented, exclude-list, loopback endpoint
        candidates = []
        for e in execs:
            if getattr(e, "is_rented", False):
                continue
            if e.executor_id.lower() in self._exclude:
                bt.logging.debug(
                    f"CanaryScheduler: skipping {e.executor_id[:16]} "
                    f"(in exclude set)"
                )
                continue
            if _is_local_endpoint(e.endpoint):
                bt.logging.debug(
                    f"CanaryScheduler: skipping {e.executor_id[:16]} "
                    f"(local endpoint {e.endpoint})"
                )
                continue
            candidates.append(e)
        if not candidates:
            return None

        canary_state = self._fetch_canary_state([e.executor_id for e in candidates])
        now = _t.time()
        runtime_canary = get_subnet_runtime_config().canary
        mean_s = max(int(runtime_canary.mean_seconds), SCHEDULER_TICK_SECONDS * 2)
        error_retry_s = max(int(runtime_canary.error_retry_seconds), SCHEDULER_TICK_SECONDS)
        error_streak_threshold = max(int(runtime_canary.error_streak_threshold), 1)

        # ── Tier 1: not yet canary-passed AND already producing valid proofs ──
        # First canary unlocks /rent. A previous transport/runtime error is
        # not a pass and should retry quickly; otherwise one closed SSH
        # connection can hide a healthy proof-producing miner for ~24h.
        needs_unlock: list[tuple[object, str]] = []
        for e in candidates:
            from common.proof_timing import is_timing_model_calibrated
            from neurons.validator.api.routes.browse import _name_for_hash

            gpu_model = _name_for_hash(e.gpu_model_hash)
            gpu_count = max(1, int(getattr(e, "gpu_count", 1) or 1))
            if not is_timing_model_calibrated(gpu_model, gpu_count):
                bt.logging.debug(
                    f"CanaryScheduler: skipping {e.executor_id[:16]} "
                    f"(uncalibrated timing model {gpu_model} × {gpu_count})"
                )
                continue
            if not self._has_valid_proof(e.executor_id):
                continue
            st = canary_state.get(e.executor_id, {})
            last = float(st.get("last_ts", 0) or 0)
            status = str(st.get("status", "") or "")
            if last == 0 or status == "":
                needs_unlock.append((e, "new arrival (scored)"))
            elif (
                status in {"error", "deferred"}
                and now - last >= error_retry_s
                and int(st.get("consecutive_errors", 0) or 0) < error_streak_threshold
            ):
                needs_unlock.append((e, "retry after canary deferred/error"))
        if needs_unlock:
            chosen, reason = secrets.choice(needs_unlock)
            return self._build_target(chosen, reason=reason)

        # ── Tier 2: memoryless Poisson trial per executor ──
        # For each executor, draw an HMAC-keyed Bernoulli with
        # probability tick/mean. Fires if true.
        #
        # Memoryless: P(fire at tick N | not fired at tick N-1) is
        # SAME as P(fire at tick 1). The miner has no safe window
        # after a canary — the next canary could land on the next
        # tick (60s later) with probability tick/mean.
        #
        # Tick bucket as HMAC input means: same (validator-secret,
        # executor, tick) → same Bernoulli outcome. Within a tick the
        # decision is deterministic; across ticks the outcomes are
        # independent. Miner can't predict because they don't have
        # the secret.
        tick_id = int(now // SCHEDULER_TICK_SECONDS)
        p_fire = SCHEDULER_TICK_SECONDS / mean_s
        ready: list = []
        for e in candidates:
            last = float(canary_state.get(e.executor_id, {}).get("last_ts", 0) or 0)
            if last == 0:
                continue  # never-canaried handled above
            mac = hmac.new(
                self._jitter_secret,
                e.executor_id.encode() + tick_id.to_bytes(8, "little"),
                hashlib.sha256,
            ).digest()
            u = int.from_bytes(mac[:8], "little") / float(1 << 64)
            if u < p_fire:
                ready.append(e)

        if not ready:
            return None

        # Multiple executors can roll-fire on the same tick. Pick one
        # uniformly random; the others will roll again next tick (with
        # the same per-tick probability, still memoryless).
        chosen = secrets.choice(ready)
        return self._build_target(chosen, reason="poisson")

    def _has_valid_proof(self, executor_id: str) -> bool:
        """Has the validator already verified at least one proof from
        this executor? Used to gate the first-canary trigger so a
        miner that hasn't produced a single valid proof yet doesn't
        consume a canary slot."""
        if self._db is None:
            return False
        try:
            from common.db import ProofResult
            with self._db.session() as s:
                row = s.query(ProofResult).filter(
                    ProofResult.executor_id == executor_id,
                    ProofResult.valid == True,  # noqa: E712
                    ProofResult.tier != "skipped",
                    ProofResult.allocation_state != "allocated",
                ).first()
                return row is not None
        except Exception:
            return False

    def _build_target(self, exec_spec, reason: str) -> dict:
        # Resolve GPU model name from hash via the existing helper
        from neurons.validator.api.routes.browse import _name_for_hash
        return {
            "executor_id": exec_spec.executor_id,
            "endpoint": exec_spec.endpoint,
            "gpu_model_name": _name_for_hash(exec_spec.gpu_model_hash),
            "gpu_count": exec_spec.gpu_count,
            "reason": reason,
        }

    def _fetch_canary_state(self, executor_ids: list[str]) -> dict[str, dict]:
        """Read canary_records; returns executor_id → status/timestamp.
        Missing = never canaried.

        NOTE: We store rows with `datetime.utcnow()` (naive but UTC).
        Calling `.timestamp()` on a naive datetime treats it as LOCAL
        time, which is wrong by the host's TZ offset. Use calendar.timegm
        on utctimetuple() to convert correctly."""
        import calendar
        out: dict[str, dict] = {}
        if self._db is None:
            return out
        try:
            with self._db.session() as s:
                from common.db import CanaryRecord
                for row in s.query(CanaryRecord).filter(
                    CanaryRecord.executor_id.in_(executor_ids),
                ).all():
                    last_ts = 0.0
                    if row.last_canaried_at:
                        last_ts = calendar.timegm(
                            row.last_canaried_at.utctimetuple(),
                        )
                    out[row.executor_id] = {
                        "last_ts": last_ts,
                        "status": row.last_status or "",
                        "consecutive_errors": row.consecutive_errors or 0,
                    }
        except Exception as e:
            bt.logging.debug(f"_fetch_canary_state: {e}")
        return out

    def _record_canary(self, executor_id: str, result: dict) -> None:
        """Upsert the canary_records row with the latest outcome.

        Also maintains the consecutive_errors streak. If consecutive
        errors crosses CANARY_ERROR_STREAK_THRESHOLD, opens a HARD
        sybil flag (`canary_error_streak`) so the rental gate locks
        the executor out. This stops the sabotage-via-error exploit
        where a miner reliably breaks pip-install / network egress
        so the canary always errors instead of failing — keeping the
        last 'pass' record alive forever."""
        if self._db is None:
            return
        try:
            import json as _json
            from datetime import datetime
            from common.db import CanaryRecord, clear_open_sybil_flag, open_sybil_flag
            status = result.get("status", "error")
            validator_transient = _is_validator_transient_canary_result(result)
            with self._db.session() as s:
                row = s.query(CanaryRecord).filter_by(
                    executor_id=executor_id,
                ).first()
                if row is None:
                    recorded_status = "deferred" if validator_transient else status
                    row = CanaryRecord(
                        executor_id=executor_id,
                        last_canaried_at=datetime.utcnow(),
                        last_status=recorded_status,
                        last_summary=result.get("reason", "")[:500],
                        last_result_json=_json.dumps(result, default=str)[:8000],
                        consecutive_errors=(
                            0 if validator_transient else (1 if status == "error" else 0)
                        ),
                        inflight_at=None,
                    )
                    s.add(row)
                else:
                    recorded_status = (
                        row.last_status
                        if validator_transient and row.last_status == "pass"
                        else ("deferred" if validator_transient else status)
                    )
                    row.last_canaried_at = datetime.utcnow()
                    row.last_status = recorded_status
                    row.last_summary = result.get("reason", "")[:500]
                    row.last_result_json = _json.dumps(result, default=str)[:8000]
                    if validator_transient:
                        # Validator/RPC outage. Do not count as a miner canary
                        # error, and do not wipe an existing pass.
                        row.consecutive_errors = 0
                    elif status == "error":
                        row.consecutive_errors = (row.consecutive_errors or 0) + 1
                    else:
                        row.consecutive_errors = 0
                    # Cleanup ran (we're recording a terminal outcome) so
                    # the inflight stamp is now stale — clear it.
                    row.inflight_at = None
                # Sabotage detection: if a miner reliably breaks canaries
                # without letting them complete (error, not fail), open
                # a HARD flag so the rental gate locks them out.
                error_streak_threshold = max(
                    int(get_subnet_runtime_config().canary.error_streak_threshold),
                    1,
                )
                if (not validator_transient) and row.consecutive_errors >= error_streak_threshold:
                    try:
                        open_sybil_flag(
                            self._db, executor_id, "canary_error_streak",
                            {
                                "consecutive_errors": row.consecutive_errors,
                                "threshold": error_streak_threshold,
                                "last_summary": (result.get("reason", "") or "")[:300],
                            },
                        )
                    except Exception as e:
                        bt.logging.debug(f"canary_error_streak flag: {e}")
                elif row.last_status == "pass":
                    try:
                        clear_open_sybil_flag(
                            self._db, executor_id, "canary_error_streak",
                            cleared_by="canary_pass",
                        )
                    except Exception as e:
                        bt.logging.debug(f"clear canary_error_streak flag: {e}")
        except Exception as e:
            bt.logging.warning(f"_record_canary failed: {e}")

    def _set_inflight_db(self, executor_id: str, inflight: bool) -> None:
        """Stamp/clear the `inflight_at` column on CanaryRecord.

        Called right before run_canary so a validator crash mid-canary
        is recoverable. reconcile_canary_inflight() reads this at boot.
        """
        if self._db is None:
            return
        try:
            from datetime import datetime
            from common.db import CanaryRecord
            with self._db.session() as s:
                row = s.query(CanaryRecord).filter_by(
                    executor_id=executor_id,
                ).first()
                if row is None:
                    # First-canary case: create an initial row so we have
                    # somewhere to stamp inflight_at. The terminal
                    # _record_canary call will overwrite the fields.
                    row = CanaryRecord(
                        executor_id=executor_id,
                        last_canaried_at=datetime.utcnow(),
                        last_status="",
                        inflight_at=datetime.utcnow() if inflight else None,
                    )
                    s.add(row)
                else:
                    row.inflight_at = datetime.utcnow() if inflight else None
        except Exception as e:
            bt.logging.warning(f"_set_inflight_db failed: {e}")


# Cap per boot. A mass-crash scenario could leave many stamped rows;
# we don't want lifespan startup to block on N×seconds of chain RPC
# before serving requests. Anything over the cap is left for the next
# canary tick to handle naturally (the executor is_rented stays True
# until the 30-min lease expires; the canary scheduler will mark it
# available the next time it picks the executor). Validators that
# routinely exceed this should look into PG-backed jobs/queues.
RECONCILE_MAX_PER_BOOT = 20


async def reconcile_canary_inflight_async(db, registry_client) -> int:
    """Async, non-blocking variant of reconcile_canary_inflight.

    Each chain `mark_available` call is wrapped in asyncio.to_thread so
    the event loop keeps serving heartbeats/API requests while we settle
    orphaned canary rentals. Capped at RECONCILE_MAX_PER_BOOT entries per
    boot; remaining rows stay inflight so the next restart/tick can settle
    them instead of silently clearing DB state while chain stays rented.
    """
    import asyncio
    if db is None or registry_client is None:
        return 0
    try:
        from common.db import CanaryRecord
        with db.session() as s:
            rows = (
                s.query(CanaryRecord)
                .filter(CanaryRecord.inflight_at.isnot(None))
                .order_by(CanaryRecord.inflight_at.asc())
                .limit(RECONCILE_MAX_PER_BOOT)
                .all()
            )
            stale_eids = [r.executor_id for r in rows]

        reconciled = 0
        for eid in stale_eids:
            try:
                tx = await asyncio.to_thread(
                    registry_client.mark_available, bytes.fromhex(eid),
                )
                bt.logging.info(
                    f"reconcile_canary_inflight: markAvailable({eid[:16]}) "
                    f"tx={tx[:16] if tx else 'n/a'}"
                )
                reconciled += 1
            except ValueError as e:
                # bytes.fromhex on bad data — skip, don't crash the
                # whole reconcile. DB-compromise mitigation.
                bt.logging.warning(
                    f"reconcile_canary_inflight: bad executor_id={eid[:16]!r}: {e}"
                )
            except Exception as e:
                bt.logging.warning(
                    f"reconcile_canary_inflight: markAvailable({eid[:16]}) failed: {e}"
                )
            try:
                with db.session() as s:
                    row = s.query(CanaryRecord).filter_by(
                        executor_id=eid,
                    ).first()
                    if row is not None:
                        row.inflight_at = None
            except Exception:
                pass

        remaining = 0
        try:
            with db.session() as s:
                remaining = (
                    s.query(CanaryRecord)
                    .filter(CanaryRecord.inflight_at.isnot(None))
                    .count()
                )
        except Exception:
            remaining = 0

        if reconciled or remaining:
            bt.logging.info(
                f"reconcile_canary_inflight: reconciled={reconciled}, "
                f"remaining={remaining}"
            )
        return reconciled
    except Exception as e:
        bt.logging.warning(f"reconcile_canary_inflight error: {e}")
        return 0


def reconcile_canary_inflight(db, registry_client) -> int:
    """Sync wrapper for callers outside an async loop. Internal callers
    in `validator.py` lifespan should prefer `_async` so reconciliation
    doesn't block startup."""
    import asyncio
    return asyncio.run(reconcile_canary_inflight_async(db, registry_client))
