"""
Monitor daemon — runs in the monitor container, observes the GPU,
posts signed reports to the validator.

Run standalone (containerization pending):
    python -m neurons.monitor.monitor \\
        --executor-id <hex> \\
        --validator-url http://validator:8443 \\
        --epoch-interval 15 \\
        --chain-rpc <subtensor-ws>

See neurons/monitor/__init__.py for the full architecture / threat
model / protocol contract. This module is the working implementation
skeleton — polls NVML, computes expected proof schedule, builds
signed reports, posts them. Containerization (Dockerfile + pinned
image) is Phase 2.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import logging as _logging

from common.proof_schedule import (
    DEFAULT_BEACON_MAX_OFFSET_BLOCKS,
    DEFAULT_JITTER_SECONDS,
    derive_beacon_offset_blocks,
    derive_jitter_seconds,
)

# Stdlib logging only — the monitor intentionally has no bittensor
# dependency. Pulling bittensor in drags async-substrate-interface,
# which conflicts with py-scale-codec, which substrate-interface
# (the Keypair lib we DO need) depends on. Keep the dep graph
# narrow: pynvml + aiohttp + substrate-interface only.
_log = _logging.getLogger("nodexo.monitor")


class _BtShim:
    """Tiny adapter so the existing `bt.logging.info(...)` call sites
    keep working after we dropped the real bittensor import. Logs go
    to stdlib stderr at the level requested.
    """
    class _Sub:
        def info(self, msg): _log.info(msg)
        def warning(self, msg): _log.warning(msg)
        def error(self, msg): _log.error(msg)
        def debug(self, msg): _log.debug(msg)
        def enable_info(self):
            _logging.basicConfig(
                level=_logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
            )
    logging = _Sub()


bt = _BtShim()


# Local keypair persistence path inside the monitor container's data
# volume. Generated on first start, never extractable from outside
# the running process unless the operator mounts the volume (which
# they CAN do — TEE is what prevents this, not us).
KEYPAIR_PATH = Path(
    os.environ.get("NODEXO_MONITOR_KEYPAIR", "/data/monitor_keypair.json")
)

def _derive_jitter(beacon: bytes, executor_id: str, max_jitter: int) -> int:
    """Mirror of derive_jitter from proof_service. Kept identical so
    the monitor predicts the same micro-proof start the miner does.
    """
    return derive_jitter_seconds(beacon, executor_id, max_jitter)


def _ensure_keypair() -> tuple[str, bytes]:
    """Load existing monitor keypair, or generate a new one and persist
    to KEYPAIR_PATH. Returns (ss58_address, seed_bytes).

    The seed lives inside the monitor container's data volume. A miner
    with host root CAN read it — that's a TEE-only protection. What we
    DO get is: a miner can't switch monitor identities mid-stream
    without the validator noticing (the pubkey changes), and they
    can't sign as the monitor unless they actually run the monitor.
    """
    from substrateinterface import Keypair
    seed_hex = os.environ.get("NODEXO_MONITOR_SEED_HEX", "").strip()
    if seed_hex:
        try:
            seed = bytes.fromhex(seed_hex)
            if len(seed) != 32:
                raise ValueError("seed must be 32 bytes")
            kp = Keypair.create_from_seed(seed.hex())
            return kp.ss58_address, seed
        except Exception as e:
            bt.logging.error(f"NODEXO_MONITOR_SEED_HEX invalid: {e}")
            raise
    if KEYPAIR_PATH.exists():
        try:
            data = json.loads(KEYPAIR_PATH.read_text())
            seed = bytes.fromhex(data["seed"])
            kp = Keypair.create_from_seed(seed.hex())
            return kp.ss58_address, seed
        except Exception as e:
            bt.logging.warning(f"existing keypair unreadable: {e}; regenerating")
    KEYPAIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed = os.urandom(32)
    kp = Keypair.create_from_seed(seed.hex())
    KEYPAIR_PATH.write_text(json.dumps({
        "seed": seed.hex(),
        "ss58_address": kp.ss58_address,
        "created_ts": time.time(),
    }))
    # Restrict the file to the monitor user only.
    try:
        os.chmod(KEYPAIR_PATH, 0o600)
    except OSError:
        pass
    bt.logging.info(f"Monitor keypair generated; ss58={kp.ss58_address[:12]}…")
    return kp.ss58_address, seed


def _sample_nvml(gpu_index: int | None = None) -> dict:
    """NVML snapshot — live GPU state + per-process accounting.

    Returns the structured fields the validator's monitor-report
    consumer expects (see __init__.py protocol section).
    """
    out: dict = {
        "gpu_live": {},
        "gpus_live": [],
        "processes": [],
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            count = int(pynvml.nvmlDeviceGetCount())
            indices = [int(gpu_index)] if gpu_index is not None else list(range(count))
            live_rows: list[dict] = []
            total_used_mb = 0
            total_mem_mb = 0
            max_util = 0
            max_temp = 0
            total_power_w = 0.0
            for idx in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp_c = 0
                power_w = 0.0
                try:
                    temp_c = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU,
                    )
                except Exception:
                    pass
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    pass
                mem_used_mb = mem.used // (1024 * 1024)
                mem_total_mb = mem.total // (1024 * 1024)
                live_rows.append({
                    "index": idx,
                    "util_pct": int(util.gpu),
                    "mem_used_mb": mem_used_mb,
                    "mem_total_mb": mem_total_mb,
                    "temp_c": int(temp_c),
                    "power_w": round(power_w, 1),
                })
                total_used_mb += int(mem_used_mb)
                total_mem_mb += int(mem_total_mb)
                max_util = max(max_util, int(util.gpu))
                max_temp = max(max_temp, int(temp_c))
                total_power_w += float(power_w)

                # Per-process accounting — the ground-truth view from the
                # kernel driver. Each process touching the GPU shows up
                # with its allocated VRAM (and util on supported GPUs).
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    pid = p.pid
                    mem_mb = (p.usedGpuMemory or 0) // (1024 * 1024)
                    # NVML's `nvmlSystemGetProcessName` returns /proc/<pid>/comm
                    # (15-char executable basename — "python" for any python
                    # script). To get the actual command including the script
                    # name we read /proc/<pid>/cmdline. Requires the monitor
                    # container to run with --pid=host so it sees the host
                    # PIDs that NVML reports.
                    proc_name = ""
                    cmdline = ""
                    try:
                        proc_name = pynvml.nvmlSystemGetProcessName(pid).decode(
                            "utf-8", errors="replace",
                        )
                    except Exception:
                        pass
                    try:
                        with open(f"/proc/{pid}/cmdline", "rb") as f:
                            cmdline = f.read().replace(b"\x00", b" ").decode(
                                "utf-8", errors="replace",
                            ).strip()
                    except Exception:
                        # /proc/<pid> unreachable (likely --pid=host not set,
                        # or process exited). Fall back to comm name only.
                        pass
                    # cgroup membership — distinguishes host miner-owned
                    # processes from container-owned (renter or sybil
                    # containers). Reported to validator so cross-tier
                    # correlation can attribute compute to the right party.
                    cgroup = ""
                    try:
                        with open(f"/proc/{pid}/cgroup", "r") as f:
                            cgroup = f.read().strip()[:200]
                    except Exception:
                        pass
                    out["processes"].append({
                        "pid": int(pid),
                        "gpu_index": idx,
                        "name": proc_name,
                        "cmdline": cmdline,
                        "cgroup": cgroup,
                        "mem_mb": int(mem_mb),
                        "gpu_total_vram_mb": int(mem_total_mb),
                    })
            out["gpus_live"] = live_rows
            out["gpu_live"] = {
                "util_pct": max_util,
                "mem_used_mb": total_used_mb,
                "mem_total_mb": total_mem_mb,
                "temp_c": max_temp,
                "power_w": round(total_power_w, 1),
            }
            # GPU total VRAM goes alongside the gpu_live block so the
            # classifier can size the heavy-proof footprint range.
            out["gpu_total_vram_mb"] = total_mem_mb
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        out["nvml_error"] = str(e)
    return out


class ProofExpector:
    """Signature-based proof-shape classifier.

    Doesn't trust process names — the miner controls those and can
    rename their sybil's worker to anything ("inference.py",
    "user_workload"). Instead, classifies CUDA-context processes by:

      * VRAM footprint shape (proof ≈500MB micro, heavy proof can consume
        a large fraction of device memory depending on GPU/kernel workspace)
      * First-seen timestamp relative to the executor's expected
        jitter window (computed from beacon + executor_id, public
        knowledge — miner can't shift this)

    The defense logic:
      A real proof allocates ~500MB. A sybil running ANOTHER proof
      on the same physical GPU must also allocate ~500MB. The sybil
      has a different executor_id ⇒ different jitter ⇒ different
      start time. Monitor sees both processes, classifies one as
      expected (started during the monitored executor's jitter
      window) and the other as extra (started outside).

      If the sybil synchronizes their start time to match the
      monitored executor's jitter window: BOTH proofs run on the
      same GPU at the same time ⇒ resource contention ⇒ neither
      completes in the timing-model deadline ⇒ validator rejects
      one or both anyway.

      Either way, sybil loses. The classifier never needs to
      believe a process name.
    """

    # Micro-proof shape: validator-known constant (compute_micro_matrix_dim
    # → ~500MB on any GPU). Tolerate ±50% to absorb python overhead +
    # CUDA context bytes + measurement noise.
    PROOF_MICRO_VRAM_MIN_MB = 200
    PROOF_MICRO_VRAM_MAX_MB = 1500

    # Heavy-proof shape. The optimized kernels can transiently use most of
    # device memory on H100-class cards, while older A6000/A100 paths sit
    # lower. Keep this broad; passive proof witnessing is audit-only.
    PROOF_HEAVY_VRAM_FRAC_MIN = 0.25
    PROOF_HEAVY_VRAM_FRAC_MAX = 0.98

    # Jitter window: the proof MUST start shortly after [epoch_start +
    # jitter_seconds]. We accept a few extra seconds of slack so a slow start
    # doesn't false-positive, but keep the window narrow enough that jitter
    # remains an anti-sybil overlap mechanism rather than a serialization slot.
    JITTER_WINDOW_SLACK_S = 5.0
    PROOF_START_WINDOW_S = 10.0

    def __init__(self, executor_id: str, epoch_interval: int = 15,
                 max_jitter: int = DEFAULT_JITTER_SECONDS):
        self._executor_id = executor_id
        self._epoch = epoch_interval
        self._max_jitter = max_jitter

        # pid → process history. Lifetime peak is retained for diagnostics;
        # current-cycle peak is reset on every cycle transition so persistent
        # proof daemons can be witnessed by activity in THIS cycle without
        # letting an idle CUDA context count forever.
        self._proc_history: dict[int, dict] = {}

        # PIDs we've already classified this cycle.
        self._counted_expected: set[int] = set()
        self._counted_extra: set[int] = set()

        # Wall-clock bounds of the current jitter window. Set by
        # update_cycle() and consulted by classify().
        self._current_cycle_id: int = -1
        self._cycle_start_ts: Optional[float] = None
        self._jitter_window_start_ts: Optional[float] = None
        self._jitter_window_end_ts: Optional[float] = None

    def clear_context(self) -> None:
        """Disable behavioral proof classification until fresh chain context
        is available again.

        This is critical across validator restarts. If /chain/context is
        temporarily unavailable, keeping the previous cycle's window causes
        fresh honest proof PIDs from later cycles to be counted as "extra".
        """
        self._cycle_start_ts = None
        self._jitter_window_start_ts = None
        self._jitter_window_end_ts = None
        self._current_cycle_id = -1
        self._counted_expected.clear()
        self._counted_extra.clear()

    def update_cycle(self, cycle_id: int, beacon: bytes,
                     current_block: int, epoch_start_block: int,
                     beacon_block: Optional[int] = None,
                     block_number_at_ts: Optional[float] = None) -> None:
        """Recompute the wall-clock bounds of the current cycle's
        jitter window.

        block_number_at_ts: the wall-clock time when the chain
            observed `current_block`. Provided by the validator's
            /chain/context endpoint; using `time.time()` here instead
            of this value makes the calculation wrong by the cache
            age (up to CHAIN_CTX_TTL_S seconds), which then mis-
            classifies the legitimate proof as `extra`.
        """
        if not beacon:
            self._jitter_window_start_ts = None
            self._jitter_window_end_ts = None
            self._cycle_start_ts = None
            return

        jitter_seconds = _derive_jitter(beacon, self._executor_id, self._max_jitter)
        proof_beacon_block = int(beacon_block or epoch_start_block)
        seconds_since_beacon = (current_block - proof_beacon_block) * 12
        block_observed_at = block_number_at_ts if block_number_at_ts else time.time()
        beacon_ts = block_observed_at - seconds_since_beacon
        self._cycle_start_ts = beacon_ts - self.JITTER_WINDOW_SLACK_S
        self._jitter_window_start_ts = (
            beacon_ts + jitter_seconds - self.JITTER_WINDOW_SLACK_S
        )
        self._jitter_window_end_ts = (
            beacon_ts + jitter_seconds + self.PROOF_START_WINDOW_S
            + self.JITTER_WINDOW_SLACK_S
        )

        if cycle_id != self._current_cycle_id:
            self._counted_expected.clear()
            self._counted_extra.clear()
            for info in self._proc_history.values():
                info["cycle_peak_mem_mb"] = 0
            self._current_cycle_id = cycle_id

    def _is_proof_shape(self, mem_mb: int, total_vram_mb: int) -> bool:
        """True iff `mem_mb` matches either micro or heavy proof
        VRAM footprint. `total_vram_mb` is needed to bound the
        heavy range against the GPU's actual capacity.
        """
        if self.PROOF_MICRO_VRAM_MIN_MB <= mem_mb <= self.PROOF_MICRO_VRAM_MAX_MB:
            return True
        if total_vram_mb > 0:
            heavy_min = total_vram_mb * self.PROOF_HEAVY_VRAM_FRAC_MIN
            heavy_max = total_vram_mb * self.PROOF_HEAVY_VRAM_FRAC_MAX
            if heavy_min <= mem_mb <= heavy_max:
                return True
        return False

    def _is_heavy_proof_shape(self, mem_mb: int, total_vram_mb: int) -> bool:
        if total_vram_mb <= 0:
            return False
        heavy_min = total_vram_mb * self.PROOF_HEAVY_VRAM_FRAC_MIN
        heavy_max = total_vram_mb * self.PROOF_HEAVY_VRAM_FRAC_MAX
        return heavy_min <= mem_mb <= heavy_max

    def classify(self, observed_processes: list[dict],
                 total_vram_mb: int,
                 now: float,
                 expected_proof_slots: int = 1) -> dict:
        """Per-tick observation update. Returns the report fields
        that go in the next signed monitor report.

        Returns expected_proof_seen=None / extra_proofs_seen=None
        when chain context isn't available (no jitter window
        defined) — the safe "we can't measure" answer rather than
        a false-positive flood.
        """
        if self._jitter_window_start_ts is None:
            return {"expected_proof_seen": None, "extra_proofs_seen": None}

        # Lifecycle tracking. For each PID currently on the GPU:
        #   - Record first_seen_ts the moment it appears.
        #   - Track peak VRAM (so a process that allocates more later
        #     gets classified by its peak, not its start size).
        current_pids = {p["pid"]: p for p in observed_processes}
        for pid, p in current_pids.items():
            mem = int(p.get("mem_mb", 0))
            gpu_total = int(p.get("gpu_total_vram_mb") or total_vram_mb)
            if pid not in self._proc_history:
                self._proc_history[pid] = {
                    "first_seen_ts": now,
                    "peak_mem_mb": mem,
                    "cycle_peak_mem_mb": mem,
                    "gpu_total_vram_mb": gpu_total,
                }
            else:
                self._proc_history[pid]["peak_mem_mb"] = max(
                    self._proc_history[pid]["peak_mem_mb"], mem,
                )
                self._proc_history[pid]["cycle_peak_mem_mb"] = max(
                    int(self._proc_history[pid].get("cycle_peak_mem_mb") or 0),
                    mem,
                )
                self._proc_history[pid]["gpu_total_vram_mb"] = max(
                    int(self._proc_history[pid].get("gpu_total_vram_mb") or 0),
                    gpu_total,
                )
        # Prune exited PIDs (keep ~5 min in case we want to compute
        # lifetime stats; longer than that is wasted memory).
        for pid in list(self._proc_history):
            if pid not in current_pids:
                age = now - self._proc_history[pid]["first_seen_ts"]
                if age > 300:
                    del self._proc_history[pid]

        # Classify each PID with proof-shape VRAM that hasn't been classified
        # yet this cycle. Fresh one-shot workers count when born in this cycle.
        # Persistent proof daemons can count too, but only if their current
        # cycle peak reaches the heavy-proof VRAM band; an idle CUDA context
        # in the micro range is not enough.
        proof_slot_budget = max(1, int(expected_proof_slots or 1))
        expected = bool(self._counted_expected)
        extra = len(self._counted_extra)
        cycle_start = self._cycle_start_ts or 0
        for pid, info in self._proc_history.items():
            if pid in self._counted_expected or pid in self._counted_extra:
                continue
            proc = current_pids.get(pid) or {}
            proc_total_vram_mb = int(
                proc.get("gpu_total_vram_mb")
                or info.get("gpu_total_vram_mb")
                or total_vram_mb
            )
            cycle_peak_mem_mb = int(info.get("cycle_peak_mem_mb") or 0)
            first_ts = info["first_seen_ts"]
            if first_ts < cycle_start:
                cmdline = str(proc.get("cmdline") or "")
                is_worker_daemon = (
                    "--daemon" in cmdline
                    and (
                        "gpu_worker.py" in cmdline
                        or "neurons.miner.services.gpu_worker" in cmdline
                    )
                )
                if not is_worker_daemon:
                    continue
                if not self._is_heavy_proof_shape(cycle_peak_mem_mb, proc_total_vram_mb):
                    continue
            elif not self._is_proof_shape(cycle_peak_mem_mb, proc_total_vram_mb):
                continue
            # Witness rule: a fresh proof-shaped PID inside this cycle
            # is the witness, regardless of where in the cycle it
            # started. The earlier design also checked the per-cycle
            # jitter window to reject pre-computed-before-beacon
            # proofs, but that attack is already blocked by the
            # cryptographic check (proof correctness depends on the
            # beacon, which is the chain block hash and unpredictable).
            # Layering a wall-clock timing check on top added a
            # fragility — small variations in spawn / CUDA-init
            # latency would push the visible PID outside a 24-second
            # window and falsely flag honest miners. Just see the
            # proof or don't.
            if len(self._counted_expected) < proof_slot_budget:
                self._counted_expected.add(pid)
                expected = True
            else:
                # More fresh proof-shaped PIDs than this executor's GPU
                # count is the real sybil signal. Multi-GPU executors can
                # legitimately spawn one worker per GPU in the same cycle.
                self._counted_extra.add(pid)
                extra = len(self._counted_extra)

        return {
            "expected_proof_seen": expected,
            "extra_proofs_seen": extra,
        }


def _sign_body(body_bytes: bytes, seed: bytes) -> dict[str, str]:
    """Compute X-Monitor-* signature headers (mirror of common.crypto
    sign_payload but with monitor-specific header names so the
    validator can distinguish monitor reports from miner/proof
    submissions)."""
    from substrateinterface import Keypair
    kp = Keypair.create_from_seed(seed.hex())
    ts = str(int(time.time()))
    nonce = hashlib.sha256(
        f"{ts}{body_bytes[:32].hex()}".encode()
    ).hexdigest()[:16]
    digest = hashlib.sha256(ts.encode() + nonce.encode() + body_bytes).digest()
    sig = kp.sign(digest)
    return {
        "X-Monitor-Signature": sig.hex(),
        "X-Monitor-Hotkey": kp.ss58_address,
        "X-Monitor-Timestamp": ts,
        "X-Monitor-Nonce": nonce,
    }


def _owner_binding_headers() -> dict[str, str]:
    """One-time miner-hotkey delegation headers for first monitor bind."""
    out: dict[str, str] = {}
    mapping = {
        "NODEXO_MONITOR_OWNER_SIGNATURE": "X-Monitor-Owner-Signature",
        "NODEXO_MONITOR_OWNER_HOTKEY": "X-Monitor-Owner-Hotkey",
        "NODEXO_MONITOR_OWNER_TIMESTAMP": "X-Monitor-Owner-Timestamp",
        "NODEXO_MONITOR_OWNER_NONCE": "X-Monitor-Owner-Nonce",
    }
    for env_name, header in mapping.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            out[header] = value
    return out


async def _fetch_chain_context(session, validator_urls: list[str],
                                epoch_interval: int) -> dict | None:
    """Poll the validator's /chain/context endpoint. The validator owns
    the chain RPC; the monitor never speaks to the chain directly. This
    is intentional:
      - No bittensor / no substrate dep in the monitor image (avoids
        the scalecodec / cyscale conflict that broke us once already).
      - The miner can't lie to the monitor about block numbers, because
        the monitor never asks the miner — it asks the validator.
      - The validator can't lie to the monitor about block numbers in
        a way that helps the miner, because the validator is the same
        party that uses this data to verify proofs; lying would only
        hurt the validator's own attestation.

    Returns the parsed context dict or None on failure (caller treats
    failure as "no chain context for this tick").
    """
    for url in validator_urls:
        try:
            full = f"{url.rstrip('/')}/chain/context?epoch_interval={epoch_interval}"
            async with session.get(full, timeout=4) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            continue
    return None


async def run_monitor(executor_id: str, validator_urls: list[str],
                      epoch_interval: int = 15,
                      max_jitter: int = DEFAULT_JITTER_SECONDS,
                      max_beacon_offset_blocks: int = DEFAULT_BEACON_MAX_OFFSET_BLOCKS,
                      poll_hz: float = 5.0,
                      chain_context_refresh_s: float = 6.0,
                      report_interval_s: float = 30.0,
                      get_block_number=None,
                      get_block_hash=None) -> None:
    """Main loop. Polls NVML at `poll_hz`, accumulates per-cycle
    observations, posts a signed report every `report_interval_s` to
    each validator URL.

    Block-number / beacon callbacks let this run with a real chain or
    against a mock. When running in real production, the monitor
    container has its own websocket connection to the chain (kept
    simple here — pass in callbacks).
    """
    bt.logging.info(
        f"Monitor starting: executor={executor_id[:16]} "
        f"validators={len(validator_urls)} poll_hz={poll_hz}"
    )
    ss58, seed = _ensure_keypair()
    bt.logging.info(f"Monitor identity: ss58={ss58[:12]}…")

    expector = ProofExpector(
        executor_id, epoch_interval=epoch_interval, max_jitter=max_jitter,
    )
    last_report_ts = 0.0
    report_seq = 0
    last_chain_ctx_fetch_ts = 0.0
    chain_ctx_retry_after_ts = 0.0
    chain_ctx_failure_streak = 0
    cached_chain_ctx: dict | None = None
    chain_context_refresh_s = max(3.0, float(chain_context_refresh_s or 6.0))
    nvml_failure_streak = 0
    nvml_exit_failures = max(
        0,
        int(os.environ.get("NODEXO_MONITOR_NVML_EXIT_FAILURES", "15") or "15"),
    )

    import aiohttp
    timeout = aiohttp.ClientTimeout(total=10)
    # Long-lived session reused for chain-context polling and report POSTs.
    # Chain context is refreshed at a lower cadence than NVML sampling so
    # 1000 monitors do not become 5000 validator HTTP requests per second.
    session = aiohttp.ClientSession(timeout=timeout)

    poll_period_s = 1.0 / poll_hz
    accumulated_processes: list[dict] = []
    accumulated_live: dict = {}
    accumulated_gpus_live: list[dict] = []

    while True:
        try:
            tick_t0 = time.time()
            snap = _sample_nvml()
            accumulated_processes = snap.get("processes", [])
            accumulated_live = snap.get("gpu_live", {})
            accumulated_gpus_live = snap.get("gpus_live", [])
            nvml_error = str(snap.get("nvml_error") or "")
            if nvml_error:
                nvml_failure_streak += 1
                if nvml_exit_failures and nvml_failure_streak >= nvml_exit_failures:
                    bt.logging.error(
                        "Monitor NVML unavailable for "
                        f"{nvml_failure_streak} consecutive samples; exiting "
                        "so the miner/docker supervisor recreates the sidecar"
                    )
                    raise SystemExit(42)
            else:
                nvml_failure_streak = 0
            chain_context_ok = False

            # Pull chain context from validator (preferred path) or
            # the standalone callbacks (dev path). The classifier needs
            # cycle_id + beacon to compute the wall-clock jitter window.
            cycle_id = 0
            ctx = None
            if get_block_number is None:
                should_refresh = (
                    tick_t0 >= chain_ctx_retry_after_ts
                    and (
                        cached_chain_ctx is None
                        or tick_t0 - last_chain_ctx_fetch_ts >= chain_context_refresh_s
                    )
                )
                if should_refresh:
                    fresh_ctx = await _fetch_chain_context(
                        session, validator_urls, epoch_interval,
                    )
                    last_chain_ctx_fetch_ts = tick_t0
                    if fresh_ctx:
                        cached_chain_ctx = fresh_ctx
                        chain_ctx_failure_streak = 0
                        chain_ctx_retry_after_ts = 0.0
                    elif (
                        cached_chain_ctx is not None
                        and tick_t0 - float(cached_chain_ctx.get("block_number_at_ts") or 0)
                        > chain_context_refresh_s * 2
                    ):
                        cached_chain_ctx = None
                    if not fresh_ctx:
                        chain_ctx_failure_streak += 1
                        backoff_s = min(
                            60.0,
                            chain_context_refresh_s
                            * (2 ** min(chain_ctx_failure_streak - 1, 4)),
                        )
                        # Spread monitors after validator restarts so a cold
                        # cache does not get a synchronized retry burst.
                        chain_ctx_retry_after_ts = (
                            tick_t0 + backoff_s + random.uniform(0.0, 1.0)
                        )
                ctx = cached_chain_ctx
                if ctx and ctx.get("beacon_hex"):
                    expector.update_cycle(
                        int(ctx["cycle_id"]),
                        bytes.fromhex(ctx["beacon_hex"]),
                        int(ctx["block_number"]),
                        int(ctx["epoch_start_block"]),
                        beacon_block=int(ctx.get("beacon_block") or ctx["epoch_start_block"]),
                        block_number_at_ts=float(ctx.get("block_number_at_ts") or 0) or None,
                    )
                    cycle_id = int(ctx["cycle_id"])
                    chain_context_ok = True
                else:
                    expector.clear_context()
            elif get_block_hash is not None:
                try:
                    block_number = await get_block_number()
                    cycle_id = block_number // epoch_interval
                    epoch_start = cycle_id * epoch_interval
                    epoch_start_hash = await get_block_hash(epoch_start)
                    offset = derive_beacon_offset_blocks(
                        epoch_start_hash, max_beacon_offset_blocks,
                    )
                    beacon_block = epoch_start + offset
                    beacon = (
                        epoch_start_hash if offset == 0
                        else await get_block_hash(beacon_block)
                    )
                    if beacon:
                        expector.update_cycle(
                            cycle_id, beacon, block_number, epoch_start,
                            beacon_block=beacon_block,
                        )
                        chain_context_ok = True
                    else:
                        expector.clear_context()
                except Exception:
                    expector.clear_context()

            # Classify against the signature-based detector. Pass total
            # VRAM so heavy-proof shape can be bounded against the actual
            # GPU capacity (heavy ≈ 40% of total VRAM).
            total_vram = int(snap.get("gpu_total_vram_mb")
                             or accumulated_live.get("mem_total_mb", 0))
            if nvml_error:
                # NVML failure means the monitor is blind. Report health
                # clearly, but do NOT emit a negative proof witness sample:
                # the validator must treat this as monitor_unhealthy, not as
                # evidence that a miner hid an otherwise-valid proof process.
                expector.clear_context()
                classified = {
                    "expected_proof_seen": None,
                    "extra_proofs_seen": None,
                }
            else:
                classified = expector.classify(
                    accumulated_processes,
                    total_vram,
                    tick_t0,
                    expected_proof_slots=len(accumulated_gpus_live) or 1,
                )

            now = time.time()
            if now - last_report_ts >= report_interval_s:
                last_report_ts = now
                report_seq += 1

                # Identify rental containers visible — naming
                # convention is `nodexo-rental-<timestamp>`.
                rental_containers = sorted({
                    p.get("name", "").split("/")[-1]
                    for p in accumulated_processes
                    if "nodexo-rental" in (p.get("name") or "")
                })

                body = {
                    "executor_id": executor_id,
                    "monitor_pubkey": ss58,
                    "report_ts": now,
                    "report_seq": report_seq,
                    "cycle_id": cycle_id,
                    "monitor_health": {
                        "nvml_ok": not bool(nvml_error),
                        "nvml_error": nvml_error[:300],
                        "chain_context_ok": bool(chain_context_ok),
                    },
                    "nvml_error": nvml_error[:300],
                    "gpu_live": accumulated_live,
                    "gpus_live": accumulated_gpus_live,
                    "processes": accumulated_processes,
                    "expected_proof_seen": classified["expected_proof_seen"],
                    "extra_proofs_seen": classified["extra_proofs_seen"],
                    "rental_containers_visible": rental_containers,
                }
                body_bytes = json.dumps(
                    body, separators=(",", ":"), sort_keys=True,
                ).encode()
                headers = _sign_body(body_bytes, seed)
                headers.update(_owner_binding_headers())
                headers["Content-Type"] = "application/json"

                for vurl in validator_urls:
                    url = vurl.rstrip("/") + "/monitor/report"
                    try:
                        async with session.post(
                            url, data=body_bytes, headers=headers,
                        ) as resp:
                            if resp.status >= 400:
                                bt.logging.warning(
                                    f"monitor report → {url}: HTTP {resp.status}"
                                )
                    except Exception as e:
                        bt.logging.warning(f"monitor report → {url}: {e}")

                # extra_proofs_seen is None in telemetry-only mode (no
                # chain wiring). Treat None as "not measured" rather
                # than "zero" — we just don't log the warning when we
                # can't measure.
                extra = classified.get("extra_proofs_seen")
                if extra and extra > 0:
                    bt.logging.warning(
                        f"Monitor: extra_proofs_seen={extra} "
                        f"on cycle {cycle_id} — possible sybil signal"
                    )

            # Pace the loop
            elapsed = time.time() - tick_t0
            if elapsed < poll_period_s:
                await asyncio.sleep(poll_period_s - elapsed)

        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"Monitor loop error: {e}; retrying in 5s")
            await asyncio.sleep(5)


def main() -> None:
    """Entry point. Config from CLI args OR env vars (container path).

    Env vars (Dockerfile ENTRYPOINT consumes these directly):
        NODEXO_EXECUTOR_ID         hex executor_id to bind to
        NODEXO_VALIDATOR_URLS      comma-separated list of validator URLs
        NODEXO_EPOCH_INTERVAL      blocks per proof cycle (default 15)
        NODEXO_POLL_HZ             NVML polling rate (default 5)
        NODEXO_REPORT_INTERVAL_S   seconds between signed reports (default 30)
        NODEXO_SUBTENSOR_NETWORK   "test" / "finney" — enables behavioral signal
        NODEXO_SUBTENSOR_ENDPOINT  optional private/local RPC endpoint
        NODEXO_NETUID              subnet id (default 0)
    """
    parser = argparse.ArgumentParser(description="Nodexo Monitor daemon")
    parser.add_argument("--executor-id",
                        default=os.environ.get("NODEXO_EXECUTOR_ID", ""))
    parser.add_argument("--validator-url", action="append", default=[],
                        help="validator endpoint (repeatable; also accepts "
                             "comma-separated NODEXO_VALIDATOR_URLS env)")
    parser.add_argument("--epoch-interval", type=int,
                        default=int(os.environ.get("NODEXO_EPOCH_INTERVAL", "15")))
    parser.add_argument("--max-jitter", type=int,
                        default=int(os.environ.get(
                            "NODEXO_PROOF_MAX_JITTER_SECONDS",
                            str(DEFAULT_JITTER_SECONDS),
                        )))
    parser.add_argument("--max-beacon-offset-blocks", type=int,
                        default=int(os.environ.get(
                            "PROOF_BEACON_MAX_OFFSET_BLOCKS",
                            str(DEFAULT_BEACON_MAX_OFFSET_BLOCKS),
                        )))
    parser.add_argument("--poll-hz", type=float,
                        default=float(os.environ.get("NODEXO_POLL_HZ", "5.0")))
    parser.add_argument("--chain-context-refresh-s", type=float,
                        default=float(os.environ.get("NODEXO_CHAIN_CONTEXT_REFRESH_S", "6.0")))
    parser.add_argument("--report-interval-s", type=float,
                        default=float(os.environ.get("NODEXO_REPORT_INTERVAL_S", "30.0")))
    parser.add_argument("--subtensor-network",
                        default=os.environ.get("NODEXO_SUBTENSOR_NETWORK", ""))
    parser.add_argument("--subtensor-endpoint",
                        default=os.environ.get("NODEXO_SUBTENSOR_ENDPOINT", ""))
    parser.add_argument("--netuid", type=int,
                        default=int(os.environ.get("NODEXO_NETUID", "0")))
    args = parser.parse_args()

    # Fold NODEXO_VALIDATOR_URLS (CSV) into the validator-url list. Env
    # is the dominant config path under the Dockerfile entrypoint.
    env_urls = os.environ.get("NODEXO_VALIDATOR_URLS", "")
    if env_urls:
        args.validator_url = list(args.validator_url) + [
            u.strip() for u in env_urls.split(",") if u.strip()
        ]

    if not args.executor_id:
        print("error: --executor-id (or NODEXO_EXECUTOR_ID) required",
              file=sys.stderr)
        sys.exit(2)
    if not args.validator_url:
        print("error: at least one --validator-url (or NODEXO_VALIDATOR_URLS) required",
              file=sys.stderr)
        sys.exit(2)

    bt.logging.enable_info()

    # Production path: monitors get chain context from the validator's
    # cached /chain/context endpoint. That keeps chain RPC fanout bounded
    # by validator count instead of monitor count. Direct chain access is
    # kept as an explicit dev/debug escape hatch only.
    get_block_number = None
    get_block_hash = None
    direct_chain = os.environ.get("NODEXO_MONITOR_DIRECT_CHAIN", "0") == "1"
    if direct_chain and args.subtensor_network:
        try:
            # Lazy import — only pulled in when chain wiring is enabled.
            # If bittensor isn't installed in this environment (the
            # default monitor container deliberately ships without it
            # to avoid the scalecodec/cyscale conflict) we fall through
            # to telemetry-only mode. The full behavioral signal needs
            # bittensor; operators who want it use a "fat" image variant.
            from common.chain.rpc import SubtensorRPC
            import bittensor as _bt  # noqa: F401 — triggers ImportError early
            SubCls = getattr(_bt, "Subtensor", None) or _bt.subtensor
            sub = SubCls(network=args.subtensor_endpoint or args.subtensor_network)
            rpc = SubtensorRPC(
                network=args.subtensor_network,
                netuid=args.netuid,
                subtensor=sub,
                chain_endpoint=args.subtensor_endpoint,
            )
            async def _get_block_number():
                return await asyncio.to_thread(rpc.get_current_block)
            async def _get_block_hash(block_number):
                return await asyncio.to_thread(rpc.get_block_hash, block_number)
            get_block_number = _get_block_number
            get_block_hash = _get_block_hash
            bt.logging.info(
                f"Monitor: chain wired ({args.subtensor_network}, "
                f"netuid={args.netuid}) — behavioral signal enabled"
            )
        except Exception as e:
            bt.logging.warning(
                f"Monitor: direct chain wiring failed ({e}); "
                "using validator chain context endpoint"
            )
    else:
        bt.logging.info(
            "Monitor: using validator chain context endpoint for proof timing"
        )

    asyncio.run(run_monitor(
        executor_id=args.executor_id,
        validator_urls=args.validator_url,
        epoch_interval=args.epoch_interval,
        max_jitter=args.max_jitter,
        max_beacon_offset_blocks=args.max_beacon_offset_blocks,
        poll_hz=args.poll_hz,
        chain_context_refresh_s=args.chain_context_refresh_s,
        report_interval_s=args.report_interval_s,
        get_block_number=get_block_number,
        get_block_hash=get_block_hash,
    ))


if __name__ == "__main__":
    main()
