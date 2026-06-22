"""
Proof analyzer — rolling pass rates, timing delta tracking, status classification.

Runs as a background task, computes per-executor statistics that feed into scoring.

Reference: see protocol design notes
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import bittensor as bt

from common.types import ProofScore

# Status classification thresholds (in multiples of expected_period)
# Reference: see protocol design notes G_OK, G_STALE, G_OFFLINE
GRACE_OK = 2          # ok if verified within 2× period (default: 360s)
GRACE_STALE = 5       # stale if no verification within 5× period (900s)
GRACE_OFFLINE = 10    # offline if no activity for 10× period (1800s)
ANALYZER_INTERVAL = 60  # Run status computation every 60s

IGNORED_POLICY_FAILURE_PREFIXES = (
    "Allocation state mismatch:",
)

NEUTRAL_RESULT_TIERS = {"skipped"}


def _is_neutral_result(valid: bool, tier: str, reason: str) -> bool:
    """Return True for validator-side skips that are audit-only.

    These rows mean the validator could not verify locally because of its own
    capacity or chain availability. They must not count as a cryptographic pass
    and must not count as a miner failure.
    """
    return (tier or "").lower() in NEUTRAL_RESULT_TIERS


def _is_ignored_policy_failure(valid: bool, reason: str) -> bool:
    """Return True for historical verifier policy failures that should not score.

    These rows are kept in proof_results for auditability, but they came from
    validator-side rental-transition state rather than failed proof work.
    """
    if valid:
        return False
    reason = reason or ""
    return any(reason.startswith(prefix) for prefix in IGNORED_POLICY_FAILURE_PREFIXES)


@dataclass
class EpochResult:
    """Result of verifying one executor's proof for one epoch."""
    epoch_id: int
    executor_id: str
    valid: bool
    tier: str                    # light/spot/full
    delta_seconds: float = 0     # commit→recipe arrival delta
    compute_time: float = 0      # Self-reported GPU time
    timestamp: float = 0         # When verified
    reason: str = ""


class ProofAnalyzer:
    """Tracks proof history and computes per-executor statistics.

    Keeps an in-memory cache for fast lookups and persists results to the
    database for survival across restarts.
    """

    def __init__(self, expected_period_sec: float = 180, db=None):
        self.expected_period = expected_period_sec
        self._db = db  # common.db.Database instance (optional)
        self._results: dict[str, list[EpochResult]] = defaultdict(list)
        self._commit_times: dict[str, dict[int, float]] = defaultdict(dict)
        self._recipe_times: dict[str, dict[int, float]] = defaultdict(dict)
        # Per-GPU per-pass receipt times: (executor_id, epoch_id, gpu_idx, u) → recv_time
        self._receipt_times: dict[tuple[str, int, int, int], float] = {}
        # Last-known trust tier per executor — used to log INFO on transitions
        self._last_trust: dict[str, str] = {}
        # Rehydrate recent results from DB so validator restarts don't
        # wipe trust state. Without this, every restart drops every
        # executor back to "full" tier for 3 cycles (3-4s each instead
        # of ~17ms spot), and trust transitions log spuriously.
        if self._db is not None:
            self._rehydrate_from_db()

    def _reliability_miss_grace_slots(self) -> int:
        try:
            from common.subnet_runtime_config import get_subnet_runtime_config
            value = get_subnet_runtime_config().scoring.reliability_miss_grace_slots
            return max(0, int(value or 0))
        except Exception:
            return 20

    def _rehydrate_from_db(self) -> None:
        """Load the last 24h of proof_results into the in-memory cache
        so trust-tier classification survives restarts. Without this,
        len(self._results[eid]) starts at 0 each restart and the
        analyzer (analyzer.py:160) classifies the executor as "full"
        for its first 3 cycles every time."""
        import calendar
        try:
            from common.db import ProofResult
            with self._db.session() as s:
                rows = s.query(ProofResult).order_by(
                    ProofResult.verified_at.desc(),
                ).limit(5000).all()
                loaded = 0
                ignored = 0
                for r in rows:
                    if r.verified_at is None:
                        continue
                    if (
                        _is_neutral_result(bool(r.valid), r.tier or "", r.reason or "")
                        or _is_ignored_policy_failure(bool(r.valid), r.reason or "")
                    ):
                        ignored += 1
                        continue
                    ts = calendar.timegm(r.verified_at.utctimetuple())
                    self._results[r.executor_id].append(EpochResult(
                        epoch_id=r.epoch_id,
                        executor_id=r.executor_id,
                        valid=bool(r.valid),
                        tier=r.tier or "spot",
                        reason=r.reason or "",
                        delta_seconds=float(r.delta_seconds or 0),
                        compute_time=float(r.compute_time or 0),
                        timestamp=ts,
                    ))
                    loaded += 1
            # Sort each executor's list by epoch ascending for sane indexing.
            for eid in self._results:
                self._results[eid].sort(key=lambda r: r.epoch_id)
            # Seed _last_trust so the first cycle after restart doesn't
            # log a spurious "Trust transition: None → spot" entry.
            for eid in self._results:
                self._last_trust[eid] = self.get_score(eid).trust_level
            msg = (
                f"Analyzer rehydrated: {loaded} results across "
                f"{len(self._results)} executors"
            )
            if ignored:
                msg += f" ({ignored} policy-only failures ignored)"
            bt.logging.info(msg)
        except Exception as e:
            bt.logging.warning(f"Analyzer rehydrate failed: {e}")

    def record_receipt(self, executor_id: str, epoch_id: int, gpu_idx: int, u: int, recv_time: float):
        """Record when a per-GPU per-pass receipt arrived (for delta measurement).

        The validator's own clock — miner cannot influence this.
        Delta = recv_time(gpu, u=1) - recv_time(gpu, u=0) ≈ GPU compute time for pass_1.
        Network latency cancels because both receipts traverse the same path.
        """
        self._receipt_times[(executor_id, epoch_id, gpu_idx, u)] = recv_time

    def get_gpu_delta(self, executor_id: str, epoch_id: int, gpu_idx: int) -> float:
        """Get the validator-measured delta for one GPU.

        Returns seconds between pass_0 and pass_1 receipt arrivals, or -1 if missing.
        """
        t0 = self._receipt_times.get((executor_id, epoch_id, gpu_idx, 0), 0)
        t1 = self._receipt_times.get((executor_id, epoch_id, gpu_idx, 1), 0)
        if t0 > 0 and t1 > t0:
            return t1 - t0
        return -1.0

    def get_all_gpu_deltas(self, executor_id: str, epoch_id: int, gpu_count: int) -> list[float]:
        """Get deltas for all GPUs. Returns list of deltas (-1 if missing)."""
        return [self.get_gpu_delta(executor_id, epoch_id, i) for i in range(gpu_count)]

    def record_commitment(self, executor_id: str, epoch_id: int, recv_time: float):
        """Record when a commitment was received (for delta calculation)."""
        self._commit_times[executor_id][epoch_id] = recv_time

    def record_result(self, result: EpochResult):
        """Record a verification result."""
        if _is_neutral_result(result.valid, result.tier, result.reason):
            return
        if _is_ignored_policy_failure(result.valid, result.reason):
            return

        # Compute delta from per-GPU receipt timings (mean across GPUs).
        # The legacy commit/recipe-time path is dead — the miner never
        # called `broadcast_commitment`, so `_commit_times` was empty and
        # delta_seconds stayed at 0 forever. The per-pass receipt times
        # ARE populated (via POST /proofs/receipt) and give us the same
        # information at GPU granularity.
        if result.delta_seconds <= 0:
            # gpu_count is unknown to the analyzer here; iterate per-GPU
            # keys we have and take the mean of populated deltas.
            deltas: list[float] = []
            gpu_idx = 0
            while True:
                t0 = self._receipt_times.get(
                    (result.executor_id, result.epoch_id, gpu_idx, 0), 0
                )
                t1 = self._receipt_times.get(
                    (result.executor_id, result.epoch_id, gpu_idx, 1), 0
                )
                if t0 == 0 and t1 == 0:
                    break
                if t0 > 0 and t1 > t0:
                    deltas.append(t1 - t0)
                gpu_idx += 1
                if gpu_idx > 64:  # safety
                    break
            if deltas:
                result.delta_seconds = sum(deltas) / len(deltas)

        self._results[result.executor_id].append(result)

        # Persist to database
        if self._db:
            from common.db import store_proof_result
            store_proof_result(
                self._db, result.executor_id, result.epoch_id,
                valid=result.valid, tier=result.tier, reason=result.reason,
                delta=result.delta_seconds, compute_time=result.compute_time,
            )

        # Keep only last 1000 results per executor in memory
        if len(self._results[result.executor_id]) > 1000:
            self._results[result.executor_id] = self._results[result.executor_id][-500:]

        # Trust-tier transition log: fires once per change (full→spot, spot→light, etc.)
        new_trust = self.get_score(result.executor_id).trust_level
        prev_trust = self._last_trust.get(result.executor_id)
        if prev_trust is not None and prev_trust != new_trust:
            samples = len(self._results[result.executor_id])
            pr_24h = sum(1 for r in self._results[result.executor_id][-200:] if r.valid) / max(1, min(200, samples))
            bt.logging.info(
                f"Trust transition: executor={result.executor_id[:16]} "
                f"{prev_trust} → {new_trust} (samples={samples}, pr_24h={pr_24h:.2f})"
            )
        self._last_trust[result.executor_id] = new_trust

    def record_recipe_time(self, executor_id: str, epoch_id: int, recv_time: float):
        """Record when a recipe was received."""
        self._recipe_times[executor_id][epoch_id] = recv_time

    def get_score(self, executor_id: str) -> ProofScore:
        """Compute current proof score for an executor."""
        results = self._results.get(executor_id, [])
        now = time.time()

        # Rolling windows
        results_1h = [r for r in results if now - r.timestamp < 3600]
        results_24h = [r for r in results if now - r.timestamp < 86400]

        def pass_rate(rs):
            if not rs:
                return 0.0
            return sum(1 for r in rs if r.valid) / len(rs)

        miss_grace_slots = self._reliability_miss_grace_slots()

        def expected_pass_rate(rs, window_s: float):
            """Passes over expected proof opportunities in a rolling window.

            The denominator starts when the executor first has scored proof
            history in the window, so fresh miners are not penalized for time
            before they joined. Epoch gaps are used when available so slow or
            fast chain block times do not distort expected-slot counts.
            """
            if not rs:
                return 0.0
            epoch_ids = [int(r.epoch_id) for r in rs if int(r.epoch_id) > 0]
            if len(epoch_ids) == len(rs):
                expected = max(epoch_ids) - min(epoch_ids) + 1
            else:
                period = max(1.0, float(self.expected_period or 1.0))
                first_ts = min(r.timestamp for r in rs)
                window_start = max(now - window_s, first_ts)
                elapsed = max(0.0, now - window_start)
                outage = self._validator_outage_overlap(window_start, now)
                elapsed = max(0.0, elapsed - outage)
                import math
                expected = int(math.ceil(elapsed / period))
            expected = max(1, expected, len(rs))
            missing = max(0, expected - len(rs))
            if missing and miss_grace_slots:
                expected = max(len(rs), expected - min(missing, miss_grace_slots))
            passed = sum(1 for r in rs if r.valid)
            return min(1.0, passed / expected)

        def avg_delta(rs):
            deltas = [r.delta_seconds for r in rs if r.valid and r.delta_seconds > 0]
            return sum(deltas) / len(deltas) if deltas else 0.0

        def avg_compute(rs):
            times = [r.compute_time for r in rs if r.valid and r.compute_time > 0]
            return sum(times) / len(times) if times else 0.0

        # Status classification
        status, _reason = self._classify_status(executor_id, now)

        # Determine trust level for next verification
        pr_24h = expected_pass_rate(results_24h, 86400)
        # Trust ladder:
        #   - full: first 3 samples or pass_rate < 0.5 (cryptographic certainty)
        #   - spot: steady state — sumcheck + ~50 random PRF samples per block
        # `light` is intentionally not selected: it skips PRF verification entirely
        # and is only sound with an external fraud-proof mechanism, which we don't
        # have. Keeping the function in the verifier for future on-chain use.
        is_new = len(results) < 3
        if is_new or pr_24h < 0.5:
            trust_level = "full"
        else:
            trust_level = "spot"

        last_epoch = max((r.epoch_id for r in results), default=0)

        return ProofScore(
            executor_id=executor_id,
            pass_rate_1h=expected_pass_rate(results_1h, 3600),
            pass_rate_24h=pr_24h,
            pass_rate_all=pass_rate(results),
            avg_delta=avg_delta(results_24h),
            avg_compute_time=avg_compute(results_24h),
            status=status,
            last_verified_epoch=last_epoch,
            trust_level=trust_level,
        )

    def _classify_status(self, executor_id: str, now: float) -> tuple[str, str]:
        """Classify executor status and reason.

        Returns (status, reason) where status is one of:
          ok      — verified OK within ok_window
          fail    — recent verification failed
          stale   — no recent verification, some past activity
          offline — no activity for extended period
          waiting — too new to classify

        Reference: see protocol design notes:_compute_current_status
        """
        results = self._results.get(executor_id, [])
        if not results:
            return "offline", "No proof results recorded"

        latest = results[-1]
        outage_since_latest = self._validator_outage_overlap(latest.timestamp, now)
        age = max(0.0, now - latest.timestamp - outage_since_latest)

        ok_window = self.expected_period * GRACE_OK
        stale_window = self.expected_period * GRACE_STALE
        offline_window = self.expected_period * GRACE_OFFLINE

        # Also check receipt times for activity
        last_activity = latest.timestamp
        for key, t in self._receipt_times.items():
            if key[0] == executor_id and t > last_activity:
                last_activity = t
        activity_age = max(
            0.0,
            now - last_activity - self._validator_outage_overlap(last_activity, now),
        )

        if age < ok_window:
            if latest.valid:
                return "ok", ""
            return "fail", latest.reason or "Proof verification failed"

        if len(results) < 3:
            return "waiting", "Too few samples"

        if age < stale_window:
            return "stale", f"No verified proof for {age:.0f}s"

        if activity_age < offline_window:
            return "stale", f"Last activity {activity_age:.0f}s ago (receipts only)"

        return "offline", f"No activity for {activity_age:.0f}s"

    def _validator_outage_overlap(self, start_ts: float, end_ts: float) -> float:
        if self._db is None or start_ts <= 0 or end_ts <= start_ts:
            return 0.0
        try:
            from common.db import validator_outage_overlap_seconds
            return float(validator_outage_overlap_seconds(self._db, start_ts, end_ts))
        except Exception:
            return 0.0

    def get_all_scores(self) -> dict[str, ProofScore]:
        """Get scores for all tracked executors."""
        return {eid: self.get_score(eid) for eid in self._results}

    def sync_stats_to_db(self):
        """Compute status + scores for all executors and persist to DB.

        Called every ANALYZER_INTERVAL (60s) by the background loop.
        Reference: see protocol design notes analyzer_loop()
        """
        if not self._db:
            return

        from common.db import upsert_executor_stats
        now = time.time()

        for executor_id in list(self._results.keys()):
            score = self.get_score(executor_id)
            status, reason = self._classify_status(executor_id, now)

            results = self._results.get(executor_id, [])
            last_activity = max((r.timestamp for r in results), default=0)
            # Include receipt times in activity
            for key, t in self._receipt_times.items():
                if key[0] == executor_id and t > last_activity:
                    last_activity = t
            age_sec = now - last_activity if last_activity > 0 else 0

            # Rolling averages
            results_1h = [r for r in results if now - r.timestamp < 3600]
            results_24h = [r for r in results if now - r.timestamp < 86400]

            def _avg(rs, attr):
                vals = [getattr(r, attr) for r in rs if r.valid and getattr(r, attr, 0) > 0]
                return sum(vals) / len(vals) if vals else 0

            upsert_executor_stats(self._db, executor_id,
                pass_rate_1h=score.pass_rate_1h,
                pass_rate_24h=score.pass_rate_24h,
                pass_rate_all=score.pass_rate_all,
                avg_delta_1h=_avg(results_1h, 'delta_seconds'),
                avg_delta_24h=_avg(results_24h, 'delta_seconds'),
                samples_1h=len(results_1h),
                samples_24h=len(results_24h),
                samples_all=len(results),
                current_status=status,
                current_reason=reason,
                trust_level=score.trust_level,
                last_epoch_id=score.last_verified_epoch,
                age_sec=age_sec,
            )

        bt.logging.debug(f"Analyzer: synced {len(self._results)} executor stats to DB")

    def prune_old(self, max_age_seconds: float = 259200):
        """Remove results older than max_age (default 3 days).

        Also prunes orphan commit/recipe/receipt timestamp dicts. These grow
        when a miner posts a Phase A commitment/receipt but never the matching
        Phase B recipe — the entries would otherwise leak forever.

        Reference: see protocol design notes
        """
        now = time.time()
        cutoff = now - max_age_seconds

        # 1. Per-executor result lists
        for eid in list(self._results.keys()):
            self._results[eid] = [r for r in self._results[eid] if r.timestamp > cutoff]
            if not self._results[eid]:
                del self._results[eid]

        # 2. Per-executor commit/recipe time dicts (cleaned by recv-time)
        for time_dict in (self._commit_times, self._recipe_times):
            for eid in list(time_dict.keys()):
                time_dict[eid] = {ep: t for ep, t in time_dict[eid].items() if t > cutoff}
                if not time_dict[eid]:
                    del time_dict[eid]

        # 3. Per-(executor, epoch, gpu, pass) receipt times
        self._receipt_times = {
            k: t for k, t in self._receipt_times.items() if t > cutoff
        }
