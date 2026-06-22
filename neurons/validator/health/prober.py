"""Validator-side executor endpoint health probing.

Miner heartbeats are push-based and arrive roughly every 60s. They are good
for live metrics, but a miner that stops right after a heartbeat can otherwise
remain visible until the heartbeat window expires. This prober is the fast
pull-side liveness signal used to remove stopped/unreachable executors from
renter-facing inventory.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass

import aiohttp
import bittensor as bt


@dataclass
class EndpointHealth:
    status: str = "unknown"  # unknown | healthy | degraded | unhealthy
    reason: str = ""
    last_ok_at: float = 0.0
    last_probe_at: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def is_unhealthy(self) -> bool:
        return self.status == "unhealthy"


class EndpointHealthProber:
    def __init__(
        self,
        chain_snapshot,
        *,
        interval_s: float | None = None,
        loop_interval_s: float | None = None,
        unhealthy_interval_s: float | None = None,
        initial_spread_s: float | None = None,
        timeout_s: float | None = None,
        retries: int | None = None,
        concurrency: int | None = None,
        max_per_tick: int | None = None,
        fail_threshold: int | None = None,
        recover_threshold: int | None = None,
    ):
        self._chain_snapshot = chain_snapshot
        # Per-endpoint cadence. This is intentionally not "probe every
        # endpoint every loop"; on a production subnet the prober is a bounded
        # negative liveness signal, while signed heartbeats/proofs remain the
        # primary positive signal.
        self.interval_s = interval_s or float(os.environ.get(
            "ENDPOINT_HEALTH_PROBE_PERIOD_S",
            os.environ.get("ENDPOINT_HEALTH_INTERVAL_S", "60"),
        ))
        self.loop_interval_s = loop_interval_s or float(os.environ.get(
            "ENDPOINT_HEALTH_LOOP_INTERVAL_S", "5",
        ))
        self.unhealthy_interval_s = unhealthy_interval_s or float(os.environ.get(
            "ENDPOINT_HEALTH_UNHEALTHY_PERIOD_S", "30",
        ))
        self.initial_spread_s = initial_spread_s
        if self.initial_spread_s is None:
            self.initial_spread_s = float(os.environ.get(
                "ENDPOINT_HEALTH_INITIAL_SPREAD_S",
                "0",
            ))
        self.timeout_s = timeout_s or float(os.environ.get("ENDPOINT_HEALTH_TIMEOUT_S", "2.5"))
        self.retries = retries if retries is not None else int(os.environ.get("ENDPOINT_HEALTH_RETRIES", "1"))
        self.concurrency = concurrency or int(os.environ.get("ENDPOINT_HEALTH_CONCURRENCY", "64"))
        if max_per_tick is None:
            max_per_tick_env = os.environ.get("ENDPOINT_HEALTH_MAX_PER_TICK")
            if max_per_tick_env:
                max_per_tick = int(max_per_tick_env)
            else:
                max_rps = float(os.environ.get("ENDPOINT_HEALTH_MAX_RPS", "20"))
                max_per_tick = max(1, int(max_rps * max(1.0, self.loop_interval_s)))
        self.max_per_tick = max(1, int(max_per_tick))
        self.fail_threshold = fail_threshold or int(os.environ.get("ENDPOINT_HEALTH_FAIL_THRESHOLD", "2"))
        self.recover_threshold = recover_threshold or int(os.environ.get("ENDPOINT_HEALTH_RECOVER_THRESHOLD", "1"))
        self._states: dict[str, EndpointHealth] = {}
        self._next_probe_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def run(self):
        bt.logging.info(
            "EndpointHealthProber starting "
            f"(per-endpoint={self.interval_s:.0f}s loop={self.loop_interval_s:.1f}s "
            f"timeout={self.timeout_s:.1f}s retries={self.retries} "
            f"concurrency={self.concurrency} max_per_tick={self.max_per_tick})"
        )
        # Small startup jitter so a validator restart does not hit every miner
        # at exactly the same moment as chain snapshot/canary startup.
        await asyncio.sleep(random.uniform(1.0, min(5.0, self.loop_interval_s)))
        while True:
            try:
                await self.probe_once()
                await asyncio.sleep(self.loop_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.warning(f"EndpointHealthProber loop error: {e}")
                await asyncio.sleep(self.loop_interval_s)

    def state(self, executor_id: str) -> EndpointHealth | None:
        return self._states.get(executor_id)

    def is_unhealthy(self, executor_id: str) -> bool:
        st = self._states.get(executor_id)
        return bool(st and st.is_unhealthy)

    def reason(self, executor_id: str) -> str:
        st = self._states.get(executor_id)
        return st.reason if st else ""

    async def probe_once(self):
        execs = list(self._chain_snapshot.all_executors()) if self._chain_snapshot else []
        execs = [e for e in execs if getattr(e, "endpoint", "")]
        if not execs:
            async with self._lock:
                self._states.clear()
                self._next_probe_at.clear()
            return
        now = time.time()
        active_ids = {e.executor_id for e in execs}
        for e in execs:
            if e.executor_id not in self._next_probe_at:
                self._next_probe_at[e.executor_id] = now + random.uniform(
                    0.0, max(0.0, self.initial_spread_s),
                )

        due = [
            e for e in execs
            if self._next_probe_at.get(e.executor_id, 0.0) <= now
        ]
        if not due:
            async with self._lock:
                for eid in list(self._states.keys()):
                    if eid not in active_ids:
                        self._states.pop(eid, None)
                        self._next_probe_at.pop(eid, None)
            return
        due.sort(key=lambda e: self._next_probe_at.get(e.executor_id, 0.0))
        selected = due[:self.max_per_tick]

        sem = asyncio.Semaphore(max(1, self.concurrency))
        timeout = aiohttp.ClientTimeout(total=self.timeout_s, connect=min(1.0, self.timeout_s))
        connector = aiohttp.TCPConnector(
            limit=max(1, self.concurrency),
            limit_per_host=4,
            ttl_dns_cache=60,
            force_close=True,
        )
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            await asyncio.gather(
                *(self._probe_one(session, sem, e) for e in selected),
                return_exceptions=True,
            )

        async with self._lock:
            for eid in list(self._states.keys()):
                if eid not in active_ids:
                    self._states.pop(eid, None)
                    self._next_probe_at.pop(eid, None)

    async def _probe_one(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, spec):
        async with sem:
            ok = False
            reason = ""
            attempts = max(1, self.retries + 1)
            for attempt in range(1, attempts + 1):
                ok, reason = await self._try_health(session, spec)
                if ok:
                    break
                if attempt < attempts:
                    await asyncio.sleep(0.15 * attempt)
            await self._record(spec.executor_id, ok, reason)

    async def _try_health(self, session: aiohttp.ClientSession, spec) -> tuple[bool, str]:
        url = spec.endpoint.rstrip("/") + "/health"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return False, f"health_http_{resp.status}: {body[:120]}"
                data = await resp.json(content_type=None)
        except asyncio.TimeoutError:
            return False, "health_timeout"
        except Exception as e:
            return False, f"health_error: {type(e).__name__}"

        if str(data.get("status", "")).lower() != "ok":
            return False, f"health_status={data.get('status')}"
        miner_prefix = str(data.get("miner_id") or "")
        if miner_prefix and not spec.executor_id.startswith(miner_prefix):
            return False, "health_executor_mismatch"
        return True, ""

    async def _record(self, executor_id: str, ok: bool, reason: str):
        now = time.time()
        async with self._lock:
            st = self._states.get(executor_id) or EndpointHealth()
            st.last_probe_at = now
            if ok:
                st.last_ok_at = now
                st.consecutive_successes += 1
                st.consecutive_failures = 0
                st.reason = ""
                if st.consecutive_successes >= self.recover_threshold:
                    st.status = "healthy"
                elif st.status == "unhealthy":
                    st.status = "degraded"
            else:
                st.consecutive_failures += 1
                st.consecutive_successes = 0
                st.reason = reason
                st.status = (
                    "unhealthy"
                    if st.consecutive_failures >= self.fail_threshold
                    else "degraded"
                )
            self._states[executor_id] = st
            period = self.interval_s if ok and st.status == "healthy" else self.unhealthy_interval_s
            jitter = random.uniform(0.8, 1.2)
            self._next_probe_at[executor_id] = now + max(1.0, period * jitter)
