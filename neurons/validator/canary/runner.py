"""
Canary runner — executes the full VRAM-fill probe against a single
executor end-to-end.

PHASES
======

  T0  validator decides to canary executor E
  T1  generate fresh canary SSH keypair (rotated per canary)
  T2  generate cryptographic nonce
  T3  markRented(E) on-chain
  T4  wait T_drain ≈ 4 minutes — miner reads chain, switches to micro
      mode on next cycle (longest legitimate heavy proof ≈ 60 s on
      A6000, plus one cycle ≈ 180 s margin)
  T5  provision a container via the standard /containers endpoint
      using a stock public image (rotate among pytorch, cuda, etc)
  T6  open SSH connection using the canary keypair
  T7  pipe fill.py to `python3 -` via stdin → wait for JSON result
  T8  evaluate result (alloc, bandwidth, checksum) against expected
  T9  open sybil flag(s) if anything failed
  T10 terminate container (DELETE /containers/{name})
  T11 markAvailable(E) on-chain

The runner is idempotent on partial-failure paths: every cleanup
step is best-effort (logs but doesn't raise), so a botched canary
never leaves an executor stuck on chain.

USAGE
=====
    from neurons.validator.canary.runner import CanaryRunner
    runner = CanaryRunner(
        registry_client=registry,
        orchestrator=rental_orchestrator,
        db=db,
    )
    result = await runner.run_canary(
        executor_id=eid,
        executor_endpoint=endpoint,
        gpu_model_name="NVIDIA RTX A6000",
        gpu_count=1,
    )
    # result["status"] in {"pass", "fail", "error"}
    # result["flags"] is a list of (reason, evidence) tuples opened
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import bittensor as bt

from common.images import (
    canary_image_catalog,
    image_canary_compatible,
    image_runtime_reference,
)

# Canary images must already contain Python, torch, and numpy. The default
# pool is the required PyTorch image set from common.images. Docker runs use
# digest-pinned references via image_runtime_reference(), so mutable tag drift
# cannot change what honest miner code starts.
CANARY_IMAGES = canary_image_catalog()


def _image_has_torch_preinstalled(image: str) -> bool:
    """Heuristic: does this image already have python+torch+numpy?
    Used to skip the apt/pip steps in the SSH setup pipeline."""
    return image_canary_compatible(image)

# Drain delays per GPU model: how long after markRented before we
# start the fill. The miner needs at least one cycle to read chain
# state and transition heavy → micro. Add buffer for chain finality
# + cycle boundary. 240s = chain confirm + 1 cycle + heavy proof
# duration safety margin.
T_DRAIN_DEFAULT = 240
T_DRAIN_BY_GPU = {
    # Bigger GPUs do heavier proofs that take longer; more margin.
    "NVIDIA RTX A6000": 240,
    "NVIDIA GeForce RTX 4090": 200,
    "NVIDIA GeForce RTX 5090": 200,
    "NVIDIA H100 80GB HBM3": 220,
    "NVIDIA H100 NVL": 220,
    "NVIDIA A100 80GB PCIe": 240,
    "NVIDIA A100-SXM4-80GB": 240,
    "NVIDIA A100-SXM4-40GB": 220,
    "NVIDIA A100-PCIE-40GB": 220,
    "NVIDIA B200": 260,
    "NVIDIA B300 SXM6": 280,
}

# How long to hold the fill after measuring. Must span at least one
# proof cycle so any concurrent sybil heavy fails its next attempt.
HOLD_SECONDS_DEFAULT = 200

# Fill chunk size. 512 MB is a sweet spot: enough HBM transactions
# per kernel to amortize launch overhead, small enough that an OOM
# at the cap doesn't waste a full minute.
CHUNK_MB_DEFAULT = 512

# Micro headroom: VRAM we leave un-filled so the miner's micro proof
# still succeeds during the hold. The nominal micro target is 500 MB,
# but CUDA context + allocator overhead pushed an honest A6000 into OOM
# with only 1 GB free, so default to a wider operational margin.
MICRO_HEADROOM_MB = int(os.environ.get("CANARY_MICRO_HEADROOM_MB", "4096"))
CANARY_MIN_STORAGE_GB = int(os.environ.get(
    "CANARY_MIN_STORAGE_GB",
    os.environ.get("RENTAL_MIN_STORAGE_GB", "30"),
))
CANARY_STORAGE_WRITE_MB = int(os.environ.get("CANARY_STORAGE_WRITE_MB", "64"))
CANARY_FILL_SSH_ATTEMPTS = int(os.environ.get("CANARY_FILL_SSH_ATTEMPTS", "3"))
CANARY_FILL_SSH_RETRY_DELAY_S = float(os.environ.get("CANARY_FILL_SSH_RETRY_DELAY_S", "8"))

_RETRYABLE_SSH_TRANSPORT_MARKERS = (
    "no route to host",
    "network is unreachable",
    "connection refused",
    "connection timed out",
    "operation timed out",
    "connection reset",
    "connection closed",
    "kex_exchange_identification",
    "banner exchange",
)

# Per-GPU bandwidth floor (GB/s). Anything below = suspect rootkit
# faking allocations via host RAM. Conservative floors with margin
# for measurement noise. Real HBM peaks:
#   4090   1008 GB/s observed > 200 GB/s
#   A6000   768 GB/s observed > 80 GB/s (under contention)
#   H100  3350 GB/s
#   B200  8000 GB/s
# PCIe-tier (rootkit lying): 32-64 GB/s, sometimes lower
HBM_BANDWIDTH_FLOOR_GB_S = {
    "NVIDIA RTX A6000": 50.0,
    "NVIDIA GeForce RTX 4090": 80.0,
    "NVIDIA GeForce RTX 5090": 100.0,
    "NVIDIA H100 80GB HBM3": 200.0,
    "NVIDIA H100 NVL": 200.0,
    "NVIDIA A100 80GB PCIe": 150.0,
    "NVIDIA A100-SXM4-80GB": 200.0,
    "NVIDIA A100-SXM4-40GB": 200.0,
    "NVIDIA A100-PCIE-40GB": 150.0,
    "NVIDIA B200": 400.0,
    "NVIDIA B300 SXM6": 400.0,
}
HBM_BANDWIDTH_FLOOR_DEFAULT = 50.0

# Allow allocated_mb / target_mb >= this. Below → canary_vram_fail.
ALLOC_RATIO_THRESHOLD = 0.95


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _read_fill_script() -> str:
    """Return fill.py's source as a string for inline piping."""
    return open(os.path.join(_here(), "fill.py")).read()


def _storage_probe_command(write_mb: int) -> str:
    """Return a shell command that emits one JSON storage probe line."""
    write_mb = max(0, int(write_mb or 0))
    script = r'''
import json, os, tempfile
out = {"path": "/", "total_bytes": 0, "available_bytes": 0,
       "write_mb": WRITE_MB, "write_ok": False}
try:
    st = os.statvfs("/")
    out["total_bytes"] = int(st.f_frsize * st.f_blocks)
    out["available_bytes"] = int(st.f_frsize * st.f_bavail)
    if WRITE_MB <= 0:
        out["write_ok"] = True
    else:
        fd, tmp = tempfile.mkstemp(prefix=".nodexo-canary-", dir="/tmp")
        try:
            chunk = b"\0" * 1048576
            with os.fdopen(fd, "wb") as f:
                for _ in range(WRITE_MB):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            out["write_ok"] = True
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
except Exception as e:
    out["error"] = type(e).__name__ + ": " + str(e)[:200]
print(json.dumps(out, separators=(",", ":")))
'''.replace("WRITE_MB", str(write_mb))
    encoded = base64.b64encode(script.encode()).decode()
    return (
        "PY=$(command -v /opt/conda/bin/python3 || command -v python3 || true); "
        ": \"${PY:?no python3 found in container image}\"; "
        f"echo {encoded} | base64 -d | \"$PY\""
    )


def _is_retryable_ssh_transport_failure(
    returncode: int | None,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> bool:
    """Return true for SSH failures that can happen before fill.py runs."""
    if returncode not in (255,):
        return False
    text = (stderr + b"\n" + stdout).decode("utf-8", errors="replace").lower()
    return any(marker in text for marker in _RETRYABLE_SSH_TRANSPORT_MARKERS)


@dataclass
class CanaryConfig:
    """Per-run knobs. Default values are sensible for an A6000 cycle."""
    drain_seconds: int = T_DRAIN_DEFAULT
    hold_seconds: int = HOLD_SECONDS_DEFAULT
    chunk_mb: int = CHUNK_MB_DEFAULT
    micro_headroom_mb: int = MICRO_HEADROOM_MB
    chunk_overshoot_ok: bool = True
    image: Optional[str] = None
    container_ttl_seconds: int = 1200  # 20 min — covers drain + fill + hold + margin
    min_storage_gb: int = CANARY_MIN_STORAGE_GB
    storage_write_mb: int = CANARY_STORAGE_WRITE_MB
    fill_ssh_attempts: int = CANARY_FILL_SSH_ATTEMPTS
    fill_ssh_retry_delay_s: float = CANARY_FILL_SSH_RETRY_DELAY_S


class CanaryRunner:
    """Executes one canary against one executor end-to-end."""

    def __init__(self, registry_client, orchestrator, db):
        self._registry = registry_client
        self._orchestrator = orchestrator
        self._db = db

    async def run_canary(
        self,
        executor_id: str,
        executor_endpoint: str,
        gpu_model_name: str,
        gpu_count: int = 1,
        config: Optional[CanaryConfig] = None,
    ) -> dict:
        """Run one canary end-to-end against `executor_id`.

        Returns a dict like:
          {
            "status": "pass" | "fail" | "error",
            "executor_id": <hex>,
            "reason": "<short string>",
            "fill_result": <the parsed fill.py JSON or None>,
            "evaluation": <dict of per-device decisions>,
            "flags_opened": [(reason, evidence), ...],
            "phase_times": {"drain": float, "provision": float, "fill": float, "cleanup": float}
          }
        """
        cfg = config or CanaryConfig()
        result: dict = {
            "status": "error",
            "executor_id": executor_id,
            "reason": "",
            "fill_result": None,
            "storage_result": None,
            "evaluation": {},
            "flags_opened": [],
            "phase_times": {},
        }

        # Spec-derived target. We allocate up to spec_total - headroom,
        # so an honest miner with full VRAM available fills cleanly.
        # An under-reporting miner trips the heartbeat check before
        # the canary; we treat the canary's target as the spec, not
        # the miner-reported value — that way a sybil hiding behind
        # over-reported "used" still gets caught when the fill OOMs
        # below spec.
        from common.config import GPU_VRAM_GB
        spec_gb = GPU_VRAM_GB.get(gpu_model_name)
        if spec_gb is None:
            result["reason"] = f"Unknown GPU model: {gpu_model_name!r}"
            return result
        target_mb_per_gpu = (spec_gb * 1024) - cfg.micro_headroom_mb

        # Fresh SSH keypair for this canary — rotated, fingerprint
        # uncorrelated with prior canaries.
        keydir = tempfile.mkdtemp(prefix="canary_ssh_")
        keypath = os.path.join(keydir, "id_ed25519")
        try:
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", keypath],
                check=True, capture_output=True,
            )
            ssh_pub = open(f"{keypath}.pub").read().strip()
        except Exception as e:
            result["reason"] = f"ssh-keygen failed: {e}"
            return result

        # Cryptographic nonce — used to seed the fill's RNG and also
        # tag the result so it can't be replayed.
        nonce = secrets.token_hex(32)
        image = cfg.image or CANARY_IMAGES[secrets.randbelow(len(CANARY_IMAGES))]
        provision_image = image_runtime_reference(image)
        result["image"] = {
            "selected": image,
            "runtime": provision_image,
            "container_image": "",
            "container_image_id": "",
        }
        canary_rental_id: Optional[str] = f"canary-{executor_id[:8]}-{int(time.time())}"[:32]
        rental_block = 0
        requested_row_persisted = False
        if self._db:
            try:
                from common.db import store_rental
                store_rental(
                    self._db,
                    canary_rental_id,
                    executor_id,
                    executor_endpoint=executor_endpoint,
                    container_name="",
                    ssh_host="",
                    ssh_port=0,
                    ssh_user="",
                    client_request_id=canary_rental_id,
                    ttl_seconds=cfg.container_ttl_seconds,
                    gpu_model=gpu_model_name,
                    vram_mb=target_mb_per_gpu,
                    gpu_count=gpu_count,
                    price_per_gpu_hour_rao=0,
                    renter_pubkey_fp="canary",
                    start_block=0,
                    status="requested",
                )
                requested_row_persisted = True
            except Exception as e:
                bt.logging.warning(
                    f"canary[{executor_id[:8]}]: requested row write failed: {e}"
                )

        # PHASE 1: markRented
        t_phase = time.perf_counter()
        try:
            tx = await asyncio.to_thread(
                self._registry.mark_rented, bytes.fromhex(executor_id),
            )
            bt.logging.info(
                f"canary[{executor_id[:8]}]: markRented tx={tx[:16] if tx else 'n/a'}"
            )
        except Exception as e:
            result["reason"] = f"markRented failed: {e}"
            if requested_row_persisted and "submitted" not in str(e).lower():
                try:
                    from common.db import terminate_rental
                    terminate_rental(self._db, canary_rental_id, reason="orphan", end_block=0)
                except Exception:
                    pass
            self._cleanup_keydir(keydir)
            return result
        try:
            from neurons.validator.api.routes import rent as rent_route
            rental_block = await asyncio.to_thread(rent_route._block_of_tx, tx)
            rent_route.rental_state.record_rented(
                executor_id, canary_rental_id, rental_block,
            )
            if requested_row_persisted:
                from common.db import Rental
                with self._db.session() as s:
                    row = s.query(Rental).filter_by(rental_id=canary_rental_id).first()
                    if row is not None:
                        row.status = "active"
                        row.start_block = rental_block
        except Exception as e:
            bt.logging.warning(
                f"canary[{executor_id[:8]}]: rental state record failed: {e}"
            )
        result["phase_times"]["mark_rented"] = round(time.perf_counter() - t_phase, 2)

        # Track everything we need to undo on cleanup
        container_name: Optional[str] = None
        try:
            # PHASE 2: drain — wait for the miner to transition to micro
            t_phase = time.perf_counter()
            drain = cfg.drain_seconds
            bt.logging.info(f"canary[{executor_id[:8]}]: draining {drain}s for micro-mode transition")
            await asyncio.sleep(drain)
            result["phase_times"]["drain"] = round(time.perf_counter() - t_phase, 2)

            # PHASE 3: provision container
            t_phase = time.perf_counter()
            try:
                rental = await self._orchestrator.provision(
                    executor_endpoint=executor_endpoint,
                    ssh_pub_key=ssh_pub,
                    image=provision_image,
                    gpu_count=gpu_count,
                    storage_gb=max(1, int(cfg.min_storage_gb or 1)),
                    ttl_seconds=cfg.container_ttl_seconds,
                )
                container_name = rental.container_name
                result["image"]["container_image"] = rental.image
                result["image"]["container_image_id"] = rental.image_id
                if rental.image != provision_image or not rental.image_id:
                    result["reason"] = (
                        "provisioned container image identity was not confirmed"
                    )
                    bt.logging.warning(
                        f"canary[{executor_id[:8]}]: {result['reason']} "
                        f"selected={image} runtime={provision_image} "
                        f"miner_image={rental.image or '<missing>'} "
                        f"image_id={rental.image_id or '<missing>'}"
                    )
                    return result
                if canary_rental_id:
                    from neurons.validator.api.routes import rent as rent_route
                    rent_route._active_rentals[canary_rental_id] = {
                        "kind": "canary",
                        "executor_id": executor_id,
                        "executor_endpoint": executor_endpoint,
                        "container_name": container_name,
                        "image": image,
                        "provision_image": provision_image,
                        "container_image_id": rental.image_id,
                        "created_at": time.time(),
                        "ttl_seconds": cfg.container_ttl_seconds,
                        "start_block": rental_block,
                    }
                bt.logging.info(
                    f"canary[{executor_id[:8]}]: provisioned {rental.container_name} "
                    f"image={image} runtime={provision_image} "
                    f"image_id={rental.image_id[:20]} "
                    f"ssh={rental.ssh_host}:{rental.ssh_port}"
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=4)
                result["reason"] = f"provision failed: {type(e).__name__}: {e!r} | tb={tb[-400:]}"
                bt.logging.warning(
                    f"canary[{executor_id[:8]}]: provision exception: {type(e).__name__}: {e!r}"
                )
                return result
            result["phase_times"]["provision"] = round(time.perf_counter() - t_phase, 2)

            # The miner can prove a host port is bound locally, but only the
            # validator's network vantage point proves provider-firewall
            # reachability for renters. Probe before spending the full SSH
            # canary budget so operators get a direct port diagnosis.
            t_phase = time.perf_counter()
            from neurons.validator.rental.ports import probe_tcp_port
            port_reachable = False
            for _attempt in range(3):
                port_reachable = await asyncio.to_thread(
                    probe_tcp_port, rental.ssh_host, rental.ssh_port, 2.5,
                )
                if port_reachable:
                    break
                await asyncio.sleep(2)
            result["phase_times"]["port_probe"] = round(time.perf_counter() - t_phase, 2)
            if not port_reachable:
                result["reason"] = (
                    f"rental port unreachable from validator: "
                    f"{rental.ssh_host}:{rental.ssh_port}"
                )
                bt.logging.warning(
                    f"canary[{executor_id[:8]}]: {result['reason']}"
                )
                return result

            # PHASE 4a: storage sanity from inside the actual rental
            # container. This is the same SSH path renters use, so it catches
            # Docker data-root / container filesystem misconfiguration without
            # creating a separate canary lifecycle.
            t_phase = time.perf_counter()
            storage_cmd = _storage_probe_command(cfg.storage_write_mb)
            ssh_storage_cmd = [
                "ssh",
                "-i", keypath,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=30",
                "-p", str(rental.ssh_port),
                "--",
                f"{rental.ssh_user}@{rental.ssh_host}",
                storage_cmd,
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *ssh_storage_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=max(45, cfg.storage_write_mb + 30),
                )
                lines = stdout.decode("utf-8", errors="replace").strip().split("\n")
                json_line = next((ln for ln in reversed(lines) if ln.strip().startswith("{")), None)
                if json_line:
                    result["storage_result"] = json.loads(json_line)
                else:
                    result["storage_result"] = {
                        "error": (
                            f"no JSON in storage probe rc={proc.returncode} "
                            f"stderr={stderr[:200].decode(errors='replace')}"
                        )
                    }
            except asyncio.TimeoutError:
                result["storage_result"] = {"error": "storage probe timeout"}
            except Exception as e:
                result["storage_result"] = {"error": f"storage probe failed: {e}"}
            result["phase_times"]["storage_probe"] = round(time.perf_counter() - t_phase, 2)

            # PHASE 4b: ssh in, pipe fill.py to python3 -, capture JSON
            t_phase = time.perf_counter()
            fill_script = _read_fill_script()
            ssh_env = {
                "CANARY_NONCE": nonce,
                "CANARY_TARGET_MB": str(target_mb_per_gpu),
                "CANARY_HOLD_SECONDS": str(cfg.hold_seconds),
                "CANARY_CHUNK_MB": str(cfg.chunk_mb),
            }
            env_prefix = " ".join(f"{k}={v}" for k, v in ssh_env.items())
            # Two paths depending on the image:
            #   1. pre-installed (pytorch/pytorch:*-cuda*): torch + numpy
            #      already there; setup is just "pipe fill.py to python".
            #      ~90s total canary instead of ~6 min.
            #   2. legacy/empty (e.g. ubuntu:22.04): apt+pip install,
            #      slow and failure-prone but kept for rare images that
            #      do not ship torch.
            if _image_has_torch_preinstalled(image):
                # pytorch/pytorch:* images put python+torch in a conda
                # env at /opt/conda. The SSH login shell's PATH does
                # NOT include /opt/conda/bin (sshd inherits /etc/profile,
                # not the docker ENTRYPOINT env), so plain `python3`
                # resolves to system python without torch. We saw this
                # fail with `exit_code: 3, "torch not installed"`.
                # Fall back to /opt/conda/bin/python3 explicitly if
                # present; else regular python3 (in case a different
                # pre-installed-torch image lays the binary elsewhere).
                # `${PY:?...}` aborts loudly if neither path exists,
                # so the operator gets a clear diagnostic in stderr
                # instead of an empty-string-command-not-found.
                setup = (
                    "set -e; "
                    "PY=$(command -v /opt/conda/bin/python3 || command -v python3 || true); "
                    ": \"${PY:?no python3 found in container image}\"; "
                    f"{env_prefix} \"$PY\" -"
                )
            else:
                # Quiet apt/pip output to stderr so stdout stays JSON-only.
                # torch wheel includes CUDA runtime libs; the host's
                # nvidia driver is mounted by --gpus all. Pin cu121 —
                # widely compatible with driver 525+.
                setup = (
                    "set -e; "
                    "export DEBIAN_FRONTEND=noninteractive; "
                    "apt-get update -qq >&2 && "
                    "apt-get install -y -qq --no-install-recommends "
                    "  python3 python3-pip python3-venv ca-certificates "
                    "  >&2 && "
                    "python3 -m pip install -q --no-input numpy >&2 && "
                    "python3 -m pip install -q --no-input torch "
                    "  --index-url https://download.pytorch.org/whl/cu121 "
                    "  >&2 && "
                    f"{env_prefix} python3 -"
                )
            # ssh argv is also validated upstream in the orchestrator,
            # but we add the literal `--` separator here as defence in
            # depth so ssh stops option-parsing before the destination.
            # See orchestrator._validate_ssh_target for the threat
            # (miner-controlled ssh_user="-oProxyCommand=..." RCE).
            ssh_cmd = [
                "ssh",
                "-i", keypath,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=30",
                "-p", str(rental.ssh_port),
                "--",
                f"{rental.ssh_user}@{rental.ssh_host}",
                f"sh -c '{setup}'",
            ]
            # ssh_timeout budget. For the pre-installed image path
            # (default), apt/pip is gone and the budget collapses
            # to:
            #   60s ssh handshake + image cold-start (with cache warm)
            #   90s fill + scan + verify
            #   hold_seconds explicit hold
            #   30s margin
            # For the legacy ubuntu path we still need the 180s pip
            # download window.
            pip_budget = 0 if _image_has_torch_preinstalled(image) else 180
            ssh_timeout = 60 + pip_budget + 90 + cfg.hold_seconds + 30
            attempts = max(1, int(cfg.fill_ssh_attempts or 1))
            retry_delay = max(0.0, float(cfg.fill_ssh_retry_delay_s or 0.0))
            json_line = None
            stdout = b""
            stderr = b""
            returncode: int | None = None
            for attempt in range(1, attempts + 1):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *ssh_cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=fill_script.encode()),
                        timeout=ssh_timeout,
                    )
                    returncode = proc.returncode
                    bt.logging.debug(
                        f"canary[{executor_id[:8]}]: ssh attempt={attempt}/{attempts} "
                        f"rc={returncode} stdout_len={len(stdout)} stderr_len={len(stderr)}"
                    )
                except asyncio.TimeoutError:
                    result["reason"] = f"ssh+fill timeout after {ssh_timeout}s"
                    return result
                except Exception as e:
                    result["reason"] = f"ssh exec failed: {e}"
                    return result

                # Parse the LAST line of stdout as JSON — earlier lines
                # may be stock-image stdout (warnings, etc.). fill.py
                # prints exactly one JSON line.
                lines = stdout.decode("utf-8", errors="replace").strip().split("\n")
                json_line = next((ln for ln in reversed(lines) if ln.strip().startswith("{")), None)
                if json_line:
                    break
                if attempt < attempts and _is_retryable_ssh_transport_failure(returncode, stdout, stderr):
                    bt.logging.warning(
                        f"canary[{executor_id[:8]}]: ssh fill transport failure "
                        f"attempt={attempt}/{attempts}; retrying in {retry_delay:.1f}s "
                        f"stderr={stderr[:160].decode(errors='replace')!r}"
                    )
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)
                    continue
                result["reason"] = (
                    f"no JSON in stdout (rc={returncode}, "
                    f"stderr={stderr[:200].decode(errors='replace')})"
                )
                return result
            result["phase_times"]["fill"] = round(time.perf_counter() - t_phase, 2)
            result["fill_ssh_attempts"] = attempt
            try:
                fill_result = json.loads(json_line)
            except json.JSONDecodeError as e:
                result["reason"] = f"bad JSON from fill.py: {e}"
                return result
            result["fill_result"] = fill_result

            # PHASE 5: evaluate
            from neurons.validator.canary.evaluator import evaluate_fill_result
            evaluation = evaluate_fill_result(
                fill_result=fill_result,
                expected_nonce=nonce,
                target_mb_per_gpu=target_mb_per_gpu,
                gpu_model_name=gpu_model_name,
                chunk_mb=cfg.chunk_mb,
                storage_result=result.get("storage_result"),
                min_storage_gb=cfg.min_storage_gb,
            )
            result["evaluation"] = evaluation

            # PHASE 6: open flags
            from common.db import open_sybil_flag
            for reason, evidence in evaluation["flags"]:
                opened = open_sybil_flag(
                    self._db, executor_id, reason, evidence,
                )
                if opened:
                    result["flags_opened"].append((reason, evidence))
                    bt.logging.warning(
                        f"canary[{executor_id[:8]}]: opened {reason} "
                        f"evidence={evidence}"
                    )

            result["status"] = "pass" if not evaluation["flags"] else "fail"
            result["reason"] = evaluation.get("summary", "")
            return result

        finally:
            # PHASE 7: cleanup — always runs. Best-effort.
            t_phase = time.perf_counter()
            mark_available_ok = False
            available_block = 0
            if container_name:
                try:
                    await self._orchestrator.terminate(
                        executor_endpoint, container_name,
                    )
                    bt.logging.info(
                        f"canary[{executor_id[:8]}]: terminated container {container_name}"
                    )
                except Exception as e:
                    bt.logging.warning(
                        f"canary[{executor_id[:8]}]: container terminate failed: {e}"
                    )
            try:
                tx = await asyncio.to_thread(
                    self._registry.mark_available, bytes.fromhex(executor_id),
                )
                if canary_rental_id:
                    from neurons.validator.api.routes import rent as rent_route
                    available_block = await asyncio.to_thread(rent_route._block_of_tx, tx)
                    rent_route.rental_state.record_available(
                        canary_rental_id, available_block,
                    )
                mark_available_ok = True
                bt.logging.info(
                    f"canary[{executor_id[:8]}]: markAvailable tx={tx[:16] if tx else 'n/a'}"
                )
            except Exception as e:
                bt.logging.warning(
                    f"canary[{executor_id[:8]}]: markAvailable failed: {e}"
                )
            if canary_rental_id:
                try:
                    from neurons.validator.api.routes import rent as rent_route
                    rent_route._active_rentals.pop(canary_rental_id, None)
                    if requested_row_persisted:
                        if mark_available_ok:
                            from common.db import terminate_rental
                            terminate_rental(
                                self._db,
                                canary_rental_id,
                                reason="canary",
                                end_block=available_block,
                            )
                        else:
                            from common.db import mark_rental_release_pending
                            mark_rental_release_pending(
                                self._db,
                                canary_rental_id,
                                reason="canary",
                                error="canary markAvailable failed; retry pending",
                                effective_ttl_seconds=cfg.container_ttl_seconds,
                            )
                    if rent_route.chain_snapshot is not None:
                        rent_route.chain_snapshot.invalidate_executor(executor_id)
                except Exception:
                    pass
            self._cleanup_keydir(keydir)
            result["phase_times"]["cleanup"] = round(time.perf_counter() - t_phase, 2)

    def _cleanup_keydir(self, keydir: str) -> None:
        try:
            import shutil
            shutil.rmtree(keydir, ignore_errors=True)
        except Exception:
            pass
