"""
Autonomous ZkGEMM proof generation service.

Runs as a background task on the executor. Watches block height, generates
proofs every epoch, and triggers broadcast to all validators.

Non-interactive protocol:
1. Derive seed from block hash (beacon) + executor_id
2. Compute pass_0 GEMM + Merkle commitment
3. Broadcast commitment (Phase A) to all validators
4. Compute chained pass_1 GEMM + Merkle commitment
5. Self-derive challenge indices from root_1
6. Generate sumcheck proofs for challenged blocks
7. Broadcast full recipe (Phase B) to all validators

Reference: see protocol design notes for prior-art context.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import struct
import sys
import time
from typing import Optional, Callable

import bittensor as bt

from common.config import compute_matrix_dim, compute_micro_matrix_dim, GPU_VRAM_GB, BUFFER_HEAVY
from common.proof_schedule import (
    DEFAULT_BEACON_MAX_OFFSET_BLOCKS,
    DEFAULT_JITTER_SECONDS,
    derive_beacon_offset_blocks,
    derive_jitter_seconds,
)
from common.types import (
    ProofCommitment, ProofRecipe, PassData, BlockProof, HardwareAttestation,
)

# Domain separation prefixes (must match validator verification)
SEED_PREFIX = b"NODEXO_SEED_V1"
CHAIN_PREFIX = b"NODEXO_CHAIN_V1"
CHALLENGE_PREFIX = b"NODEXO_CHALLENGE_V1"


def derive_seed(beacon: bytes, executor_id: str, gpu_index: int = 0) -> bytes:
    """Derive per-GPU seed from block hash beacon.

    seed = SHA256("NODEXO_SEED_V1" || beacon || executor_id_bytes || gpu_index_le32)[:8]

    Each GPU gets a UNIQUE seed because gpu_index is part of the hash input.
    This ensures each GPU must independently prove its existence.

    Each GPU's seed is keyed on its index so independent proofs are required per device.
    """
    payload = SEED_PREFIX + beacon + bytes.fromhex(executor_id) + struct.pack("<I", gpu_index)
    return hashlib.sha256(payload).digest()[:8]


def derive_jitter(beacon: bytes, executor_id: str, max_jitter: int) -> int:
    """Derive per-executor jitter in seconds.

    jitter = SHA256("NODEXO_JITTER_V1" || beacon || executor_id)[:2] mod max_jitter

    Keep this deliberately small. It is not a load-spreading mechanism; it
    only avoids exact same-second stampedes while preserving the anti-sybil
    property that two executor IDs backed by one GPU must overlap.
    """
    return derive_jitter_seconds(beacon, executor_id, max_jitter)


def beacon_from_chain_context(
    ctx: dict | None,
    epoch_id: int,
    epoch_interval: int,
) -> tuple[bytes, int, int] | None:
    """Return (beacon, beacon_block, offset) from validator chain context."""
    if not isinstance(ctx, dict):
        return None
    try:
        if int(ctx.get("cycle_id")) != int(epoch_id):
            return None
        epoch_start_block = int(epoch_id) * int(epoch_interval)
        if int(ctx.get("epoch_start_block", epoch_start_block)) != epoch_start_block:
            return None
        beacon_hex = str(ctx.get("beacon_hex") or "").strip().removeprefix("0x")
        if len(beacon_hex) != 64:
            return None
        beacon_block = int(ctx.get("beacon_block") or 0)
        offset = int(ctx.get("beacon_offset_blocks", beacon_block - epoch_start_block))
        if beacon_block != epoch_start_block + offset:
            return None
        if offset < 0 or offset >= int(epoch_interval):
            return None
        return bytes.fromhex(beacon_hex), beacon_block, offset
    except Exception:
        return None


def seconds_until_block(current_block: int, target_block: int,
                        block_time_s: float = 12.0,
                        wake_margin_s: float = 1.0,
                        max_sleep_s: float = 60.0) -> float:
    """Wall-clock sleep estimate to reduce chain polling.

    The proof loop still uses chain block hashes as the source of truth; this
    helper only decides how long to sleep before trying the next hash/read.
    Capping keeps the miner responsive if block production stalls.
    """
    blocks = int(target_block) - int(current_block)
    if blocks <= 0:
        return 0.0
    return min(float(max_sleep_s), max(0.0, blocks * float(block_time_s) - float(wake_margin_s)))


def estimate_beacon_age_seconds(
    current_block: int,
    beacon_block: int,
    block_time_s: float = 12.0,
    chain_ctx: dict | None = None,
    now_s: float | None = None,
) -> float:
    """Estimate elapsed wall time since the beacon block became available."""
    if isinstance(chain_ctx, dict):
        try:
            beacon_age_s = float(chain_ctx.get("beacon_age_s") or 0.0)
            if beacon_age_s > 0:
                received_at = float(chain_ctx.get("_received_at_ts") or 0.0)
                now = time.time() if now_s is None else float(now_s)
                local_elapsed_s = max(0.0, now - received_at) if received_at > 0 else 0.0
                return max(0.0, beacon_age_s + local_elapsed_s)
        except Exception:
            pass
        try:
            beacon_available_at = float(chain_ctx.get("beacon_available_at_ts") or 0.0)
            if beacon_available_at > 0:
                now = time.time() if now_s is None else float(now_s)
                return max(0.0, now - beacon_available_at)
        except Exception:
            pass
    age_s = max(
        0.0,
        (int(current_block) - int(beacon_block)) * float(block_time_s),
    )
    if isinstance(chain_ctx, dict):
        try:
            observed_at = float(chain_ctx.get("block_number_at_ts") or 0.0)
            if observed_at > 0:
                now = time.time() if now_s is None else float(now_s)
                age_s += min(float(block_time_s), max(0.0, now - observed_at))
        except Exception:
            pass
    return age_s


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def proof_worker_timeout_seconds(gpu_model: str, gpu_count: int) -> float:
    """Timeout for gpu_worker subprocesses, aligned with validator timing."""
    override = os.environ.get("PROOF_WORKER_TIMEOUT_SECONDS", "").strip()
    if override:
        return max(10.0, float(override))
    try:
        from common.proof_timing import (
            get_timing_model,
            compute_wall_clock_deadline,
        )
        model = get_timing_model(gpu_model)
        if model is not None:
            return max(
                60.0,
                compute_wall_clock_deadline(model, gpu_count) + 30.0,
            )
    except Exception as e:
        bt.logging.debug(f"worker timeout model unavailable: {e}")
    return 180.0


def free_proof_timing_budget_seconds(gpu_model: str, gpu_count: int) -> tuple[float, float]:
    """Return (expected_runtime_s, validator_wall_deadline_s) for free proofs.

    This is a local miner-side preflight guard only. The validator remains the
    source of truth; the miner uses the same timing table to skip cycles that
    are already too old to land inside the validator's beacon-relative window.
    """
    try:
        from common.proof_timing import (
            get_timing_model,
            compute_gpu_count_scaling,
            compute_wall_clock_deadline,
        )
        model = get_timing_model(gpu_model)
        if model is not None:
            if not getattr(model, "calibrated", False) and _env_bool("NODEXO_ALLOW_UNCALIBRATED_GPU"):
                expected = proof_worker_timeout_seconds(gpu_model, gpu_count)
                grace = _env_float("PROOF_UNCALIBRATED_STALE_GRACE_SECONDS", 120.0)
                return max(1.0, expected), max(1.0, expected + grace)
            scaling = compute_gpu_count_scaling(model, gpu_count)
            return (
                max(1.0, float(model.base2) * scaling),
                max(1.0, compute_wall_clock_deadline(model, gpu_count)),
            )
    except Exception as e:
        bt.logging.debug(f"free proof timing budget unavailable: {e}")
    return 60.0, 120.0


def free_proof_pass0_budget_seconds(
    gpu_model: str,
    gpu_count: int,
    jitter_s: float = 0.0,
) -> tuple[float, float]:
    """Return (expected_pass0_s, validator_pass0_deadline_s).

    The broad wall-clock deadline protects total recipe completion. The pass0
    receipt deadline is stricter: if the miner is already too late to get u=0
    witnessed, it must skip the cycle before broadcasting a bad timing sample.
    """
    try:
        from common.proof_timing import (
            compute_delta_deadline,
            compute_pass0_receipt_deadline,
            get_timing_model,
        )
        model = get_timing_model(gpu_model)
        if model is not None:
            if not getattr(model, "calibrated", False) and _env_bool("NODEXO_ALLOW_UNCALIBRATED_GPU"):
                expected, wall_deadline = free_proof_timing_budget_seconds(
                    gpu_model,
                    gpu_count,
                )
                return max(1.0, expected * 0.5), max(1.0, wall_deadline)
            return (
                max(1.0, compute_delta_deadline(model, gpu_count)),
                max(
                    1.0,
                    compute_pass0_receipt_deadline(
                        model,
                        gpu_count,
                        jitter_s=jitter_s,
                    ),
                ),
            )
    except Exception as e:
        bt.logging.debug(f"free proof pass0 budget unavailable: {e}")
    return 30.0, 60.0


def receipt_broadcast_outcome(outcome) -> tuple[bool, bool, int, int]:
    """Return (has_success, is_partial, success, total) for receipt broadcast."""
    if not isinstance(outcome, dict):
        return True, False, 0, 0
    total = int(outcome.get("total") or 0)
    success = int(outcome.get("success") or 0)
    if total <= 0 or success <= 0:
        return False, False, success, total
    return True, success < total, success, total


def _iter_gpu_worker_pids(executor_id: str) -> list[int]:
    """Return stale gpu_worker.py pids owned by this user.

    New workers include `--executor-id`; older workers did not, so the fallback
    also matches legacy one-shot gpu_worker processes. This is intentionally
    scoped to the current Unix uid and the unique gpu_worker entrypoint, never
    arbitrary Python or renter containers.
    """
    current_pid = os.getpid()
    current_uid = os.getuid()
    out: list[int] = []
    proc_root = "/proc"
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == current_pid:
            continue
        base = os.path.join(proc_root, entry)
        try:
            if os.stat(base).st_uid != current_uid:
                continue
            with open(os.path.join(base, "cmdline"), "rb") as f:
                raw = f.read()
        except OSError:
            continue
        if not raw:
            continue
        parts = [p.decode("utf-8", errors="replace") for p in raw.split(b"\0") if p]
        joined = " ".join(parts)
        if "gpu_worker.py" not in joined and "neurons.miner.services.gpu_worker" not in joined:
            continue
        if executor_id and "--executor-id" in parts:
            try:
                if parts[parts.index("--executor-id") + 1] != executor_id:
                    continue
            except (ValueError, IndexError):
                continue
        out.append(pid)
    return sorted(set(out))


def cleanup_stale_gpu_workers(executor_id: str, *, grace_s: float = 1.0) -> list[int]:
    pids = _iter_gpu_worker_pids(executor_id)
    if not pids:
        return []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            bt.logging.warning(f"Could not terminate stale gpu_worker pid={pid}: permission denied")
    deadline = time.time() + max(0.0, grace_s)
    remaining = list(pids)
    while remaining and time.time() < deadline:
        remaining = [pid for pid in remaining if os.path.exists(f"/proc/{pid}")]
        if remaining:
            time.sleep(0.05)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            bt.logging.warning(f"Could not kill stale gpu_worker pid={pid}: permission denied")
    return pids


def _worker_timing_summary(timings: dict | None) -> str:
    if not isinstance(timings, dict):
        return ""
    labels = (
        ("import_s", "import"),
        ("pass0_s", "pass0"),
        ("pass1_s", "pass1"),
        ("proof1_s", "proof1"),
        ("proof0_s", "proof0"),
        ("serialize_s", "serialize"),
        ("total_s", "total"),
    )
    parts: list[str] = []
    for key, label in labels:
        try:
            value = float(timings.get(key) or 0.0)
        except Exception:
            continue
        if value > 0:
            parts.append(f"{label}={value:.2f}s")
    return " ".join(parts)


def derive_chain_seed(seed: bytes, root_0: bytes) -> bytes:
    """Derive chained seed for pass_1.

    seed_1 = SHA256("NODEXO_CHAIN_V1" || seed || root_0)[:8]
    """
    payload = CHAIN_PREFIX + seed + root_0
    return hashlib.sha256(payload).digest()[:8]


def derive_challenge_indices(
    seed: bytes, root_1: bytes, num_challenges: int, total_blocks: int
) -> list[int]:
    """Derive deterministic challenge block indices from root_1.

    Challenges are derived from the chained root (pass_1), ensuring the
    executor must complete BOTH passes before knowing which blocks to prove.

    idx = SHA256("NODEXO_CHALLENGE_V1" || seed || root_1 || i)[:4] mod total_blocks
    """
    indices = []
    seen = set()
    counter = 0
    while len(indices) < num_challenges:
        payload = CHALLENGE_PREFIX + seed + root_1 + struct.pack("<I", counter)
        h = hashlib.sha256(payload).digest()
        idx = struct.unpack("<I", h[:4])[0] % total_blocks
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
        counter += 1
    return indices


class ProofService:
    """Autonomous proof generation loop running on the executor."""

    def __init__(
        self,
        executor_id: str,
        gpu_model: str,
        vram_mb: int,
        gpu_count: int,
        epoch_interval: int = 15,       # blocks per epoch
        max_jitter: int = DEFAULT_JITTER_SECONDS,
        max_beacon_offset_blocks: int = DEFAULT_BEACON_MAX_OFFSET_BLOCKS,
        num_challenges: int = 4,        # blocks to challenge per epoch
        num_spot_checks: int = 500,     # spot checks per block proof
        block_size: int = 256,
        get_block_hash: Optional[Callable] = None,  # RPC callback
        get_block_number: Optional[Callable] = None,
        get_chain_context: Optional[Callable] = None,
        is_rented: Optional[Callable] = None,        # Rental state callback
        should_run: Optional[Callable] = None,        # If returns False, skip cycle (e.g. monitor down)
        on_receipt: Optional[Callable] = None,        # Phase A: per-GPU per-pass receipt
        on_recipe: Optional[Callable] = None,         # Phase B: full recipe with all proofs
    ):
        self.executor_id = executor_id
        self.gpu_model = gpu_model
        self.vram_mb = vram_mb
        self.vram_gb = vram_mb / 1024
        self.gpu_count = gpu_count
        self.epoch_interval = epoch_interval
        self.max_jitter = max_jitter
        self.max_beacon_offset_blocks = max_beacon_offset_blocks
        self.num_challenges = num_challenges
        self.num_spot_checks = num_spot_checks
        self.block_size = block_size

        # Callbacks
        self._get_block_hash = get_block_hash
        self._get_block_number = get_block_number
        self._get_chain_context = get_chain_context
        self._is_rented = is_rented
        self._should_run = should_run
        self._on_receipt = on_receipt
        self._on_recipe = on_recipe

        self._last_proven_epoch = -1
        self._running = False
        self._busy = False
        self._receipt_tasks: set[asyncio.Task] = set()
        self._worker_daemons: dict[int, "asyncio.subprocess.Process"] = {}
        self._worker_daemon_stderr_tasks: dict[int, asyncio.Task] = {}

    def is_busy(self) -> bool:
        return self._busy

    def _worker_daemon_policy(self, rented: bool, mode_label: str) -> bool:
        raw = os.environ.get("PROOF_WORKER_DAEMON", "auto").strip().lower()
        if raw in {"0", "false", "no", "off", "oneshot", "one-shot"}:
            return False
        if raw in {"1", "true", "yes", "on", "always"}:
            return True
        # Auto mode uses persistent workers only for free heavy proofs. Rented
        # micro proofs keep the old one-shot path so a long-lived CUDA context
        # does not sit beside customer workloads.
        return (not rented) and mode_label == "heavy"

    def _worker_daemon_prewarm_enabled(self) -> bool:
        raw = os.environ.get("PROOF_WORKER_DAEMON_PREWARM", "").strip().lower()
        if raw:
            return raw in {"1", "true", "yes", "on"}
        return True

    def _worker_env(self, gpu_idx: int) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        env["PYTHONPATH"] = f"{project_root}:{os.path.join(project_root, 'zkgemm', 'cuda')}"
        return env

    async def _drain_worker_daemon_stderr(
        self,
        gpu_idx: int,
        proc: "asyncio.subprocess.Process",
    ) -> None:
        if proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    bt.logging.warning(f"  GPU[{gpu_idx}]: worker daemon stderr: {text[:300]}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            bt.logging.debug(f"worker daemon stderr drain failed gpu={gpu_idx}: {e}")

    async def _stop_worker_daemon(self, gpu_idx: int) -> None:
        proc = self._worker_daemons.pop(gpu_idx, None)
        task = self._worker_daemon_stderr_tasks.pop(gpu_idx, None)
        if proc is not None and proc.returncode is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(b'{"op":"exit"}\n')
                    await proc.stdin.drain()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except Exception:
                try:
                    if proc.pid:
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    await proc.wait()
                except Exception:
                    pass
        if task is not None:
            task.cancel()

    async def _stop_worker_daemons(self) -> None:
        for gpu_idx in list(self._worker_daemons):
            await self._stop_worker_daemon(gpu_idx)

    async def _ensure_worker_daemon(
        self,
        gpu_idx: int,
    ) -> "asyncio.subprocess.Process | None":
        proc = self._worker_daemons.get(gpu_idx)
        if (
            proc is not None
            and proc.returncode is None
            and proc.stdin is not None
            and proc.stdout is not None
        ):
            return proc
        if proc is not None:
            await self._stop_worker_daemon(gpu_idx)

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "gpu_worker.py"),
            "--daemon",
            "--gpu-idx", str(gpu_idx),
            "--executor-id", self.executor_id,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=self._worker_env(gpu_idx),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            if proc.stdout is None:
                await self._stop_worker_daemon(gpu_idx)
                return None
            timeout_s = _env_float("PROOF_WORKER_DAEMON_START_TIMEOUT_SECONDS", 45.0)
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_s)
            try:
                ready = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                ready = {}
            if proc.returncode is not None or ready.get("op") != "ready":
                try:
                    err = ""
                    if proc.stderr is not None:
                        err = (await asyncio.wait_for(proc.stderr.read(), timeout=0.2)).decode(
                            "utf-8",
                            errors="replace",
                        )[:300]
                except Exception:
                    err = ""
                bt.logging.warning(
                    f"  GPU[{gpu_idx}]: worker daemon failed to start; "
                    f"falling back to one-shot mode {err}".rstrip()
                )
                try:
                    if proc.pid:
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
                return None
            self._worker_daemons[gpu_idx] = proc
            self._worker_daemon_stderr_tasks[gpu_idx] = asyncio.create_task(
                self._drain_worker_daemon_stderr(gpu_idx, proc)
            )
            bt.logging.info(f"  GPU[{gpu_idx}]: worker daemon ready (pid={proc.pid})")
            return proc
        except Exception as e:
            bt.logging.warning(
                f"  GPU[{gpu_idx}]: worker daemon start failed; "
                f"falling back to one-shot mode: {e}"
            )
            return None

    async def _submit_worker_daemon_run(
        self,
        gpu_idx: int,
        proc: "asyncio.subprocess.Process",
        payload: dict,
    ) -> bool:
        if proc.stdin is None or proc.returncode is not None:
            return False
        try:
            proc.stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            )
            await proc.stdin.drain()
            return True
        except Exception as e:
            bt.logging.warning(f"  GPU[{gpu_idx}]: worker daemon submit failed: {e}")
            await self._stop_worker_daemon(gpu_idx)
            return False

    async def _wait_worker_daemon_done(
        self,
        gpu_idx: int,
        epoch_id: int,
    ) -> dict:
        proc = self._worker_daemons.get(gpu_idx)
        if proc is None or proc.stdout is None:
            return {"status": "error", "returncode": -1, "stderr": "worker daemon missing"}
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return {
                        "status": "error",
                        "returncode": proc.returncode if proc.returncode is not None else -1,
                        "stderr": "worker daemon exited",
                    }
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if msg.get("op") != "done":
                    continue
                if int(msg.get("epoch_id") or -1) != int(epoch_id):
                    bt.logging.warning(
                        f"  GPU[{gpu_idx}]: worker daemon returned stale epoch "
                        f"{msg.get('epoch_id')} while waiting for {epoch_id}"
                    )
                    continue
                return {
                    "status": "ok" if msg.get("status") == "ok" else "error",
                    "returncode": 0 if msg.get("status") == "ok" else 1,
                    "stderr": str(msg.get("error") or "")[:500],
                }
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return {"status": "error", "returncode": -1, "stderr": str(e)[:500]}

    async def run(self):
        """Main proof generation loop. Runs indefinitely."""
        self._running = True
        block_time_s = _env_float("PROOF_BLOCK_TIME_SECONDS", 12)
        loop_max_sleep_s = _env_float("PROOF_LOOP_MAX_SLEEP_SECONDS", 30)
        boundary_max_sleep_s = _env_float("PROOF_BOUNDARY_MAX_SLEEP_SECONDS", 2.0)
        beacon_max_sleep_s = _env_float("PROOF_BEACON_MAX_SLEEP_SECONDS", 2.0)
        chain_context_retry_s = _env_float("PROOF_CHAIN_CONTEXT_RETRY_SECONDS", 0.5)
        chain_context_required = self._get_chain_context is not None
        bt.logging.info(
            f"ProofService started: executor={self.executor_id[:16]} gpu={self.gpu_model} vram={self.vram_mb}MB epoch_interval={self.epoch_interval}"
        )
        if (
            self._worker_daemon_policy(False, "heavy")
            and self._worker_daemon_prewarm_enabled()
        ):
            try:
                rented_now = self._is_rented() if self._is_rented else False
            except Exception:
                rented_now = False
            if not rented_now:
                for gpu_idx in range(self.gpu_count):
                    await self._ensure_worker_daemon(gpu_idx)

        async def _get_hash_with_retry(block: int, attempts: int = 12,
                                       delay_s: float = 1.0) -> bytes | None:
            for attempt in range(max(1, attempts)):
                try:
                    h = await self._get_block_hash(block)
                    if h:
                        return h
                except Exception as e:
                    if attempt == 0:
                        bt.logging.debug(f"Block hash {block} not ready: {e}")
                await asyncio.sleep(delay_s)
            return None

        async def _get_validator_chain_context() -> dict | None:
            if self._get_chain_context is None:
                return None
            try:
                ctx = await self._get_chain_context()
                if not isinstance(ctx, dict):
                    return None
                out = dict(ctx)
                out["_received_at_ts"] = time.time()
                return out
            except Exception as e:
                bt.logging.debug(f"validator chain context unavailable: {e}")
                return None

        while self._running:
            try:
                # Pause condition — used today to skip cycles when the
                # sidecar monitor container is down. The miner's sybil
                # attestation surface is gone in that state; broadcasting
                # proofs would only inflate reliability scores against a
                # non-attestable executor. Sleep one cycle and retry.
                if self._should_run is not None and not self._should_run():
                    bt.logging.warning(
                        "ProofService: should_run=False (monitor likely down); "
                        "skipping cycle until it recovers"
                    )
                    await asyncio.sleep(self.epoch_interval * 12)
                    continue

                chain_ctx = await _get_validator_chain_context()
                if chain_ctx is None and chain_context_required:
                    # Validator context returns 503 until the beacon block is
                    # actually known. Do not fall back to local/public RPC here:
                    # public RPC can expose the hash one block late, which is
                    # enough to miss the pass_0 receipt window on tight models.
                    await asyncio.sleep(max(0.1, chain_context_retry_s))
                    continue
                if chain_ctx is not None:
                    try:
                        block_number = int(chain_ctx["block_number"])
                        current_epoch = int(chain_ctx["cycle_id"])
                    except Exception:
                        chain_ctx = None
                if chain_ctx is None:
                    block_number = await self._get_block_number()
                    current_epoch = block_number // self.epoch_interval
                if self._last_proven_epoch < 0:
                    epoch_id = current_epoch
                else:
                    # Always work on the next pending cycle. If RPC/hash reads
                    # stall near a boundary, jumping straight to
                    # current_epoch can silently skip a cycle and undercount
                    # reliability. Pending cycles that are already too old are
                    # explicitly skipped by the stale-deadline guard below.
                    epoch_id = self._last_proven_epoch + 1

                if epoch_id > current_epoch:
                    next_epoch_block = epoch_id * self.epoch_interval
                    sleep_s = seconds_until_block(
                        block_number, next_epoch_block,
                        block_time_s=block_time_s,
                        wake_margin_s=2.0,
                        max_sleep_s=min(loop_max_sleep_s, boundary_max_sleep_s),
                    )
                    await asyncio.sleep(max(0.5, sleep_s))
                    continue
                if epoch_id < current_epoch:
                    bt.logging.warning(
                        f"ProofService: processing pending cycle {epoch_id} while "
                        f"chain is at cycle {current_epoch}; last_proven="
                        f"{self._last_proven_epoch}"
                    )

                epoch_start_block = epoch_id * self.epoch_interval
                beacon_ctx = beacon_from_chain_context(
                    chain_ctx,
                    epoch_id,
                    self.epoch_interval,
                )
                if beacon_ctx is not None:
                    beacon, beacon_block, beacon_offset = beacon_ctx
                else:
                    epoch_start_hash = await _get_hash_with_retry(epoch_start_block)
                    if not epoch_start_hash:
                        try:
                            latest_block = await self._get_block_number()
                        except Exception:
                            latest_block = block_number
                        _, wall_deadline_s = free_proof_timing_budget_seconds(
                            self.gpu_model,
                            self.gpu_count,
                        )
                        stale_after_blocks = (
                            int(self.max_beacon_offset_blocks)
                            + max(2, int(wall_deadline_s / block_time_s) + 2)
                        )
                        if int(latest_block) - int(epoch_start_block) > stale_after_blocks:
                            self._last_proven_epoch = epoch_id
                            bt.logging.warning(
                                f"Cycle {epoch_id}: skipping stale proof because "
                                f"epoch_start_hash was unavailable after retries "
                                f"(epoch_start_block={epoch_start_block}, "
                                f"latest_block={latest_block})"
                            )
                            continue
                        bt.logging.debug(
                            f"Cycle {epoch_id}: epoch_start_hash not ready for "
                            f"block {epoch_start_block}; retrying"
                        )
                        await asyncio.sleep(3)
                        continue
                    beacon_offset = derive_beacon_offset_blocks(
                        epoch_start_hash, self.max_beacon_offset_blocks,
                    )
                    beacon_block = epoch_start_block + beacon_offset
                    sleep_s = seconds_until_block(
                        block_number, beacon_block,
                        block_time_s=block_time_s,
                        wake_margin_s=1.0,
                        max_sleep_s=min(30.0, beacon_max_sleep_s),
                    )
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    beacon = (
                        epoch_start_hash if beacon_offset == 0
                        else await _get_hash_with_retry(beacon_block)
                    )
                    if not beacon:
                        try:
                            latest_block = await self._get_block_number()
                        except Exception:
                            latest_block = block_number
                        _, wall_deadline_s = free_proof_timing_budget_seconds(
                            self.gpu_model,
                            self.gpu_count,
                        )
                        stale_after_blocks = (
                            int(self.max_beacon_offset_blocks)
                            + max(2, int(wall_deadline_s / block_time_s) + 2)
                        )
                        if int(latest_block) - int(beacon_block) > stale_after_blocks:
                            self._last_proven_epoch = epoch_id
                            bt.logging.warning(
                                f"Cycle {epoch_id}: skipping stale proof because "
                                f"beacon hash was unavailable after retries "
                                f"(beacon_block={beacon_block}, "
                                f"latest_block={latest_block})"
                            )
                            continue
                        bt.logging.debug(
                            f"Cycle {epoch_id}: beacon hash not ready for block "
                            f"{beacon_block}; retrying"
                        )
                        await asyncio.sleep(3)
                        continue

                # Decide proof mode at the beacon boundary, before jitter.
                # The validator checks rental state at the same beacon block;
                # reading live state after jitter can flip a miner into micro
                # mode inside a cycle the validator still treats as free.
                rented = self._is_rented() if self._is_rented else False

                # Derive a small per-executor jitter (same for all GPUs on
                # this executor). This must stay seconds-level, not block-level:
                # wide jitter lets one physical GPU serialize multiple claimed
                # executors inside the epoch.
                jitter = derive_jitter(beacon, self.executor_id, self.max_jitter)

                if not rented:
                    try:
                        if (
                            chain_ctx is not None
                            and int(chain_ctx.get("cycle_id") or -1) == int(epoch_id)
                        ):
                            current_block_for_age = int(chain_ctx.get("block_number") or block_number)
                        else:
                            current_block_for_age = await self._get_block_number()
                    except Exception:
                        current_block_for_age = block_number
                    beacon_age_s = estimate_beacon_age_seconds(
                        current_block_for_age,
                        beacon_block,
                        block_time_s=block_time_s,
                        chain_ctx=chain_ctx if (
                            chain_ctx is not None
                            and int(chain_ctx.get("cycle_id") or -1) == int(epoch_id)
                        ) else None,
                    )
                    expected_runtime_s, wall_deadline_s = free_proof_timing_budget_seconds(
                        self.gpu_model,
                        self.gpu_count,
                    )
                    pass0_expected_s, pass0_deadline_s = free_proof_pass0_budget_seconds(
                        self.gpu_model,
                        self.gpu_count,
                        jitter_s=float(jitter),
                    )
                    submit_safety_s = _env_float("PROOF_SUBMIT_SAFETY_SECONDS", 3.0)
                    pass0_safety_s = _env_float("PROOF_PASS0_SUBMIT_SAFETY_SECONDS", 1.0)
                    start_grace_s = _env_float("PROOF_START_GRACE_SECONDS", 3.0)
                    pass0_start_grace_s = _env_float("PROOF_PASS0_START_GRACE_SECONDS", 0.0)
                    projected_pass0_s = (
                        beacon_age_s
                        + float(jitter)
                        + pass0_expected_s
                        + pass0_safety_s
                    )
                    projected_arrival_s = (
                        beacon_age_s
                        + float(jitter)
                        + expected_runtime_s
                        + submit_safety_s
                    )
                    if projected_pass0_s > pass0_deadline_s + pass0_start_grace_s:
                        self._last_proven_epoch = epoch_id
                        bt.logging.warning(
                            f"Cycle {epoch_id} (beacon_block {beacon_block}): skipping stale "
                            f"free proof before pass_0; beacon_age≈{beacon_age_s:.1f}s "
                            f"jitter={jitter}s expected_pass0≈{pass0_expected_s:.1f}s "
                            f"safety={pass0_safety_s:.1f}s > pass0_deadline={pass0_deadline_s:.1f}s "
                            f"grace={pass0_start_grace_s:.1f}s"
                        )
                        continue
                    if projected_arrival_s > wall_deadline_s + start_grace_s:
                        self._last_proven_epoch = epoch_id
                        bt.logging.warning(
                            f"Cycle {epoch_id} (beacon_block {beacon_block}): skipping stale "
                            f"free proof; beacon_age≈{beacon_age_s:.1f}s jitter={jitter}s "
                            f"expected_runtime≈{expected_runtime_s:.1f}s safety={submit_safety_s:.1f}s "
                            f"> deadline={wall_deadline_s:.1f}s grace={start_grace_s:.1f}s"
                        )
                        continue

                # Wait for jitter.
                if jitter > 0:
                    await asyncio.sleep(jitter)

                # Heavy: ~40% VRAM, full proof (~50s on A6000).
                # Micro: fixed ~500 MB VRAM budget regardless of GPU
                # (~N=3328 on any size). Sized so the proof survives even
                # if the renter pegs VRAM to 99% — % of VRAM was too
                # fragile (renter at 99% → micro OOMs). Constants live
                # in common.config so the validator uses the same values.
                if rented:
                    n = compute_micro_matrix_dim(block_size=self.block_size)
                    mode_label = "micro"
                else:
                    n = compute_matrix_dim(self.vram_gb, buffer=BUFFER_HEAVY,
                                           block_size=self.block_size)
                    mode_label = "heavy"

                bt.logging.info(
                    f"Cycle {epoch_id} (beacon_block {beacon_block}): generating proofs "
                    f"for {self.gpu_count} GPU(s) (n={n}, mode={mode_label})"
                )
                self._busy = True

                # ── Per-GPU proof generation via isolated workers ──
                # Free heavy proofs reuse one long-lived worker per GPU to
                # remove import/CUDA-init overhead from the calibrated timing
                # path. Rented micro proofs use one-shot workers so no idle
                # proof daemon sits beside a customer workload.
                import tempfile
                import random

                run_id = f"epoch{epoch_id}_{int(time.time())}_{random.randrange(1<<16):04x}"
                run_dir = tempfile.mkdtemp(prefix=f"nodexo_proof_{self.executor_id[:12]}_{epoch_id}_")
                t_start = time.time()
                worker_timeout = proof_worker_timeout_seconds(
                    self.gpu_model,
                    self.gpu_count,
                )

                use_worker_daemon = self._worker_daemon_policy(rented, mode_label)
                if use_worker_daemon:
                    for gpu_idx in range(self.gpu_count):
                        if await self._ensure_worker_daemon(gpu_idx) is None:
                            use_worker_daemon = False
                            break
                if not use_worker_daemon:
                    await self._stop_worker_daemons()

                procs: dict[int, "asyncio.subprocess.Process"] = {}
                daemon_tasks: dict[int, asyncio.Task[dict]] = {}
                done_results: dict[int, dict] = {}
                if use_worker_daemon:
                    bt.logging.debug(
                        f"Cycle {epoch_id}: using persistent worker daemon(s)"
                    )
                else:
                    stale_pids = await asyncio.to_thread(
                        cleanup_stale_gpu_workers,
                        self.executor_id,
                    )
                    if stale_pids:
                        bt.logging.warning(
                            "Cleaned stale gpu_worker process(es) before proof cycle "
                            f"{epoch_id}: {stale_pids}"
                        )
                for gpu_idx in range(self.gpu_count):
                    seed = derive_seed(beacon, self.executor_id, gpu_idx)
                    if use_worker_daemon:
                        proc = self._worker_daemons.get(gpu_idx)
                        payload = {
                            "op": "run",
                            "seed_hex": seed.hex(),
                            "n": n,
                            "block_size": self.block_size,
                            "gpu_idx": gpu_idx,
                            "executor_id": self.executor_id,
                            "epoch_id": epoch_id,
                            "num_challenges": self.num_challenges,
                            "num_spot_checks": self.num_spot_checks,
                            "output_dir": run_dir,
                        }
                        if proc is not None and await self._submit_worker_daemon_run(
                            gpu_idx,
                            proc,
                            payload,
                        ):
                            daemon_tasks[gpu_idx] = asyncio.create_task(
                                self._wait_worker_daemon_done(gpu_idx, epoch_id)
                            )
                            bt.logging.debug(
                                f"  GPU[{gpu_idx}]: cycle {epoch_id} submitted to "
                                f"worker daemon (pid={proc.pid})"
                            )
                        else:
                            done_results[gpu_idx] = {
                                "status": "error",
                                "returncode": -1,
                                "stderr": "worker daemon unavailable",
                            }
                        continue

                    cmd = [
                        sys.executable,
                        os.path.join(os.path.dirname(__file__), "gpu_worker.py"),
                        "--seed-hex", seed.hex(),
                        "--n", str(n),
                        "--block-size", str(self.block_size),
                        "--gpu-idx", str(gpu_idx),
                        "--executor-id", self.executor_id,
                        "--epoch-id", str(epoch_id),
                        "--num-challenges", str(self.num_challenges),
                        "--num-spot-checks", str(self.num_spot_checks),
                        "--output-dir", run_dir,
                    ]
                    p = await asyncio.create_subprocess_exec(
                        *cmd, env=self._worker_env(gpu_idx),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,
                    )
                    procs[gpu_idx] = p
                    bt.logging.debug(
                        f"  GPU[{gpu_idx}]: cycle {epoch_id} subprocess spawned (pid={p.pid})"
                    )

                # Poll for per-GPU pass receipts while subprocesses
                # run, and detect completion via Process.returncode.
                # Subprocess exit is the natural cycle-completion
                # signal — no JSON-over-pipe protocol needed.
                receipts_scheduled: set[str] = set()
                receipt_tasks_by_path: dict[str, asyncio.Task[bool]] = {}
                receipt_results: dict[str, bool] = {}

                async def _broadcast_receipt_file(
                    fpath: str,
                    gpu_idx: int,
                    u: int,
                    receipt: dict,
                ) -> bool:
                    try:
                        outcome = await self._on_receipt(receipt)
                        # broadcast_service returns success/total. Unit tests
                        # and simple callbacks may return None; treat that as
                        # success for compatibility with local-only callbacks.
                        has_success, is_partial, success, total = receipt_broadcast_outcome(outcome)
                        if not has_success:
                            bt.logging.warning(
                                f"  Receipt broadcast incomplete: gpu={gpu_idx} "
                                f"u={u} success={success}/{total}"
                            )
                            return False
                        if is_partial:
                            bt.logging.warning(
                                f"  Receipt broadcast partial: gpu={gpu_idx} "
                                f"u={u} success={success}/{total}"
                            )
                        return True
                    except Exception as e:
                        bt.logging.warning(
                            f"  Receipt broadcast failed: gpu={gpu_idx} u={u}: {e}"
                        )
                        return False

                def _schedule_receipt_file(fpath: str, gpu_idx: int, u: int) -> bool:
                    try:
                        try:
                            receipt = json.loads(open(fpath).read())
                        except json.JSONDecodeError:
                            bt.logging.debug(
                                f"  Receipt not ready yet: gpu={gpu_idx} u={u}"
                            )
                            return False
                        receipt["executor_id"] = self.executor_id
                        receipt["epoch_id"] = epoch_id
                        from neurons.version import miner_version, miner_version_str
                        receipt["software_version"] = miner_version_str
                        receipt["software_version_int"] = miner_version
                        task = asyncio.create_task(
                            _broadcast_receipt_file(fpath, gpu_idx, u, receipt)
                        )
                        self._receipt_tasks.add(task)
                        receipt_tasks_by_path[fpath] = task
                        receipts_scheduled.add(fpath)

                        def _log_result(done: "asyncio.Task[bool]") -> None:
                            self._receipt_tasks.discard(done)
                            try:
                                receipt_results[fpath] = bool(done.result())
                            except Exception as e:
                                receipt_results[fpath] = False
                                bt.logging.warning(
                                    f"  Receipt broadcast task failed: gpu={gpu_idx} u={u}: {e}"
                                )

                        task.add_done_callback(_log_result)
                        return True
                    except Exception as e:
                        bt.logging.warning(
                            f"  Receipt scheduling failed: gpu={gpu_idx} u={u}: {e}"
                        )
                        return False

                async def _drain_receipt_tasks(timeout_s: float) -> None:
                    pending = [task for task in receipt_tasks_by_path.values() if not task.done()]
                    if pending:
                        await asyncio.wait(pending, timeout=max(0.0, timeout_s))
                    for path, task in receipt_tasks_by_path.items():
                        if not task.done() or path in receipt_results:
                            continue
                        try:
                            receipt_results[path] = bool(task.result())
                        except Exception:
                            receipt_results[path] = False

                deadline = time.time() + worker_timeout
                while (procs or daemon_tasks) and time.time() < deadline:
                    if self._on_receipt:
                        for gpu_idx in range(self.gpu_count):
                            for u in (0, 1):
                                fpath = os.path.join(run_dir, f"gpu_{gpu_idx}_pass{u}.json")
                                if fpath not in receipts_scheduled and os.path.exists(fpath):
                                    if _schedule_receipt_file(fpath, gpu_idx, u):
                                        bt.logging.debug(
                                            f"  Receipt scheduled: gpu={gpu_idx} u={u}"
                                        )
                    # Reap any finished daemon commands.
                    for gpu_idx in list(daemon_tasks.keys()):
                        task = daemon_tasks[gpu_idx]
                        if task.done():
                            try:
                                done_results[gpu_idx] = task.result()
                            except Exception as e:
                                done_results[gpu_idx] = {
                                    "status": "error",
                                    "returncode": -1,
                                    "stderr": str(e)[:500],
                                }
                            del daemon_tasks[gpu_idx]
                    # Reap any finished subprocesses.
                    for gpu_idx in list(procs.keys()):
                        p = procs[gpu_idx]
                        if p.returncode is not None:
                            stderr_bytes = b""
                            try:
                                _stdout_bytes, stderr_bytes = await p.communicate()
                            except Exception:
                                pass
                            done_results[gpu_idx] = {
                                "status": "ok" if p.returncode == 0 else "error",
                                "returncode": p.returncode,
                                "stderr": stderr_bytes.decode("utf-8", errors="replace")[:500],
                            }
                            del procs[gpu_idx]
                    if procs:
                        await asyncio.sleep(0.1)
                    elif daemon_tasks:
                        await asyncio.sleep(0.1)

                # Kill any stragglers past the deadline.
                for gpu_idx, task in list(daemon_tasks.items()):
                    bt.logging.error(
                        f"  GPU[{gpu_idx}]: worker daemon timed out at "
                        f"{worker_timeout:.0f}s, killing"
                    )
                    task.cancel()
                    await self._stop_worker_daemon(gpu_idx)
                    done_results[gpu_idx] = {
                        "status": "error",
                        "returncode": -1,
                        "stderr": f"worker daemon killed after {worker_timeout:.0f}s timeout",
                    }
                    del daemon_tasks[gpu_idx]
                for gpu_idx, p in list(procs.items()):
                    bt.logging.error(
                        f"  GPU[{gpu_idx}]: subprocess timed out at "
                        f"{worker_timeout:.0f}s, killing"
                    )
                    try:
                        if p.pid:
                            try:
                                os.killpg(p.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            except Exception:
                                p.kill()
                        else:
                            p.kill()
                        await p.wait()
                    except Exception:
                        pass
                    done_results[gpu_idx] = {
                        "status": "error", "returncode": -1,
                        "stderr": f"killed after {worker_timeout:.0f}s timeout",
                    }
                    del procs[gpu_idx]

                for gpu_idx, r in done_results.items():
                    if r["status"] != "ok":
                        bt.logging.error(
                            f"  GPU[{gpu_idx}]: cycle failed (rc={r['returncode']}): "
                            f"{r['stderr'][:200]}"
                        )

                # Send any remaining receipts not caught by polling
                if self._on_receipt:
                    for gpu_idx in range(self.gpu_count):
                        for u in (0, 1):
                            fpath = os.path.join(run_dir, f"gpu_{gpu_idx}_pass{u}.json")
                            if fpath not in receipts_scheduled and os.path.exists(fpath):
                                _schedule_receipt_file(fpath, gpu_idx, u)
                    if receipts_scheduled:
                        await _drain_receipt_tasks(
                            _env_float("PROOF_RECEIPT_DRAIN_SECONDS", 4.5)
                        )

                # Read all GPU results
                gpu_proofs = []
                all_ok = True
                for gpu_idx in range(self.gpu_count):
                    result_path = os.path.join(run_dir, f"gpu_{gpu_idx}.json")
                    if not os.path.exists(result_path):
                        bt.logging.error(f"  GPU[{gpu_idx}]: no output file")
                        all_ok = False
                        continue
                    gpu_result = json.loads(open(result_path).read())
                    if gpu_result.get("status") != "ok":
                        bt.logging.error(f"  GPU[{gpu_idx}]: error: {gpu_result.get('error', 'unknown')}")
                        all_ok = False
                        continue

                    gpu_proofs.append({
                        "gpu_index": gpu_idx,
                        "seed": bytes.fromhex(gpu_result["seed"]),
                        "ts_pass0_ns": gpu_result["ts_pass0_ns"],
                        "ts_pass1_ns": gpu_result["ts_pass1_ns"],
                        "T_commit_0": gpu_result["T_commit_0"],
                        "T_commit_1": gpu_result["T_commit_1"],
                        "pass_0": gpu_result["pass_0"],  # Already serialized by worker
                        "pass_1": gpu_result["pass_1"],
                    })
                    timing_summary = _worker_timing_summary(gpu_result.get("timings"))
                    if timing_summary:
                        try:
                            total_s = float((gpu_result.get("timings") or {}).get("total_s") or 0.0)
                        except Exception:
                            total_s = 0.0
                        slow_threshold_s = _env_float(
                            "PROOF_WORKER_TIMING_LOG_THRESHOLD_SECONDS",
                            30.0,
                        )
                        if (
                            _env_bool("PROOF_WORKER_TIMING_LOG", False)
                            or total_s >= slow_threshold_s
                        ):
                            bt.logging.info(
                                f"  GPU[{gpu_idx}]: worker timings {timing_summary}"
                            )
                    bt.logging.debug(
                        f"  GPU[{gpu_idx}]: ok, delta={(gpu_result['ts_pass1_ns'] - gpu_result['ts_pass0_ns']) / 1e9:.2f}s, root_0={gpu_result['root_0'][:16]}"
                    )

                compute_time = time.time() - t_start

                expected_receipts = self.gpu_count * 2 if self._on_receipt else 0
                expected_receipt_paths = [
                    os.path.join(run_dir, f"gpu_{gpu_idx}_pass{u}.json")
                    for gpu_idx in range(self.gpu_count)
                    for u in (0, 1)
                ] if self._on_receipt else []
                receipts_complete = (
                    not self._on_receipt
                    or len(receipts_scheduled) >= expected_receipts
                )
                receipts_delivered = (
                    not self._on_receipt
                    or all(receipt_results.get(path) is True for path in expected_receipt_paths)
                )
                if not all_ok or len(gpu_proofs) != self.gpu_count:
                    bt.logging.error(f"Cycle {epoch_id}: proof incomplete ({len(gpu_proofs)}/{self.gpu_count} GPUs)")
                    if use_worker_daemon:
                        await self._stop_worker_daemons()
                    else:
                        stale_pids = await asyncio.to_thread(
                            cleanup_stale_gpu_workers,
                            self.executor_id,
                        )
                        if stale_pids:
                            bt.logging.warning(
                                "Cleaned stale gpu_worker process(es) after incomplete "
                                f"cycle {epoch_id}: {stale_pids}"
                            )
                    # Don't broadcast incomplete proofs
                else:
                    if not receipts_complete:
                        bt.logging.warning(
                            f"Cycle {epoch_id}: continuing recipe broadcast although "
                            f"receipt files were incomplete "
                            f"({len(receipts_scheduled)}/{expected_receipts})"
                        )
                    elif not receipts_delivered:
                        delivered = sum(
                            1
                            for path in expected_receipt_paths
                            if receipt_results.get(path) is True
                        )
                        bt.logging.warning(
                            f"Cycle {epoch_id}: continuing recipe broadcast although "
                            f"receipt POST acknowledgements were incomplete "
                            f"({delivered}/{expected_receipts}); validators will "
                            f"score only receipts they recorded"
                        )

                    # Broadcast per-GPU commitments (Phase A already sent above via receipts)

                    # Broadcast recipe (Phase B) — contains all GPU proofs
                    if self._on_recipe:
                        from neurons.version import miner_version, miner_version_str
                        recipe = {
                            "epoch_id": epoch_id,
                            "executor_id": self.executor_id,
                            "software_version": miner_version_str,
                            "software_version_int": miner_version,
                            "matrix_dim": n,
                            "gpu_count": self.gpu_count,
                            "run_id": run_id,
                            "beacon_meta": {
                                "b_num": beacon_block,
                                "b_hash": beacon.hex(),
                                "anchor_block": epoch_start_block,
                                "beacon_offset_blocks": beacon_offset,
                            },
                            "allocation_state": "allocated" if rented else "unallocated",
                            "gpu_proofs": gpu_proofs,
                            "compute_time": compute_time,
                            "timestamp_start": t_start,
                            "timestamp_end": time.time(),
                        }
                        recipe_outcome = await self._on_recipe(recipe)
                        if isinstance(recipe_outcome, dict):
                            total = int(recipe_outcome.get("total") or 0)
                            success = int(recipe_outcome.get("success") or 0)
                            if total <= 0 or success < total:
                                bt.logging.warning(
                                    f"Cycle {epoch_id}: recipe broadcast incomplete "
                                    f"({success}/{total})"
                                )

                # Cleanup temp dir
                import shutil
                shutil.rmtree(run_dir, ignore_errors=True)
                self._busy = False

                self._last_proven_epoch = epoch_id
                # epoch_id = block_number // PROOF_EPOCH_BLOCKS — a 15-block proof
                # cycle, NOT a bittensor tempo. Logged with the start block for clarity.
                bt.logging.info(
                    f"Cycle {epoch_id} (beacon_block {beacon_block}): proof complete "
                    f"({compute_time:.1f}s, n={n}, {len(gpu_proofs)} GPUs, {self.num_challenges} challenges/gpu)"
                )

            except Exception as e:
                self._busy = False
                try:
                    await self._stop_worker_daemons()
                    stale_pids = await asyncio.to_thread(
                        cleanup_stale_gpu_workers,
                        self.executor_id,
                    )
                    if stale_pids:
                        bt.logging.warning(
                            "Cleaned stale gpu_worker process(es) after proof error "
                            f"{epoch_id}: {stale_pids}"
                        )
                except Exception as cleanup_error:
                    bt.logging.warning(f"Stale gpu_worker cleanup failed: {cleanup_error}")
                bt.logging.error(f"Proof generation error: {e}")
                await asyncio.sleep(12)

    def stop(self):
        self._running = False
        for task in list(self._worker_daemon_stderr_tasks.values()):
            task.cancel()
        self._worker_daemon_stderr_tasks.clear()
        for proc in list(self._worker_daemons.values()):
            if proc.returncode is not None:
                continue
            try:
                if proc.pid:
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._worker_daemons.clear()

    # NOTE: _compute_pass() and _prove_blocks() were removed.
    # Proof generation now happens in gpu_worker.py subprocesses.

    def _build_hw_attestation(self) -> HardwareAttestation:
        """Build hardware attestation from current system state."""
        from neurons.miner.services.hardware_service import (
            detect_gpus, detect_system, get_nvml_digest_compat, detect_sysbox,
        )
        from neurons.version import miner_version_str
        gpus = detect_gpus()
        sys_info = detect_system()

        return HardwareAttestation(
            gpu_model=gpus[0].name if gpus else "unknown",
            gpu_uuids=[g.uuid for g in gpus],
            gpu_count=len(gpus),
            vram_mb=gpus[0].vram_mb if gpus else 0,
            driver_version=gpus[0].driver_version if gpus else "unknown",
            nvml_digest=get_nvml_digest_compat(),
            sysbox_detected=detect_sysbox(),
            cpu_model=sys_info.cpu_model,
            ram_gb=sys_info.ram_gb,
            software_version=miner_version_str,
        )
