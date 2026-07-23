"""
Hardware detection and monitoring service.

Uses pynvml for GPU detection and psutil for system metrics.
Uses pynvml for GPU detection and psutil for system metrics.
Prior-art context: protocol design notes.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import time
import urllib.request
import uuid as uuid_mod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bittensor as bt

from neurons.version import miner_version_str

IDENTITY_PATH = (
    Path(os.environ.get("NODEXO_DATA_DIR", "~/.nodexo")).expanduser().resolve()
    / "executor_identity.json"
)

_NETWORK_SPEED_LOCK = threading.Lock()
_NETWORK_SPEED_CACHE: dict[str, Any] = {}
_STORAGE_LOCK = threading.Lock()
_STORAGE_STATIC_CACHE: dict[str, Any] | None = None

_NETWORK_FS_TYPES = {
    "nfs", "nfs4", "cifs", "smb3", "fuse.sshfs", "ceph", "glusterfs",
    "lustre", "9p", "virtiofs",
}

# Optional GPU subset filter. Set at most ONE of these to run several miner
# instances on a single host, each owning a disjoint slice of the GPUs.
GPU_UUIDS_ENV = "NODEXO_GPU_UUIDS"
GPU_INDICES_ENV = "NODEXO_GPU_INDICES"


@dataclass
class GpuInfo:
    index: int
    name: str
    uuid: str
    vram_mb: int
    driver_version: str
    compute_capability: str


@dataclass
class SystemInfo:
    cpu_model: str
    cpu_cores: int
    ram_gb: int
    hostname: str
    os_version: str


def _normalize_gpu_uuid(raw: str) -> str:
    """Canonical form used to compare GPU UUIDs.

    NVML reports ``GPU-<hex>``; operators copy them around with or without
    that prefix and in either case. Compare on the bare lowercase hex.
    """
    token = raw.strip().lower()
    if token.startswith("gpu-"):
        token = token[len("gpu-"):]
    return token


def _parse_gpu_filter_env(
    env: Mapping[str, str] | None = None,
) -> tuple[list[str], list[int]]:
    """Read the GPU subset filter from the environment.

    Parsing is lenient: entries are whitespace-stripped and empty entries
    are dropped, so ``"0, 1,"`` means the same as ``"0,1"``.

    Args:
        env: mapping to read from; defaults to ``os.environ``.

    Returns:
        ``(uuids, indices)``. Either or both may be empty, meaning "no
        filter of that kind".

    Raises:
        ValueError: if an entry of ``NODEXO_GPU_INDICES`` is not an integer.
    """
    src = os.environ if env is None else env

    uuids = [tok.strip() for tok in src.get(GPU_UUIDS_ENV, "").split(",")]
    uuids = [tok for tok in uuids if tok]
    _reject_duplicates(uuids, GPU_UUIDS_ENV, _normalize_gpu_uuid)

    indices: list[int] = []
    for tok in src.get(GPU_INDICES_ENV, "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            indices.append(int(tok))
        except ValueError:
            raise ValueError(
                f"{GPU_INDICES_ENV} entry is not an integer: {tok!r}"
            ) from None
    _reject_duplicates(indices, GPU_INDICES_ENV)

    return uuids, indices


def _reject_duplicates(
    entries: list,
    label: str,
    key=lambda x: x,
) -> None:
    """Raise if a filter list names the same GPU twice.

    ``NODEXO_GPU_INDICES="0,0"`` is a plausible typo for ``"0,1"`` and would
    otherwise be silently deduped into a one-GPU executor. Everywhere else
    this filter fails closed on operator error; be consistent here.
    """
    seen = set()
    for entry in entries:
        token = key(entry)
        if token in seen:
            raise ValueError(f"{label} names the same GPU twice: {entry!r}")
        seen.add(token)


def select_gpu_subset(
    gpus: list[GpuInfo],
    *,
    uuids: list[str] | None = None,
    indices: list[int] | None = None,
    strict: bool = True,
) -> list[GpuInfo]:
    """Narrow a detected GPU list down to an operator-selected subset.

    This is what lets one physical host be split into several independently
    rentable executors: point each miner instance at a disjoint subset and
    ``get_or_create_executor_id()`` derives a distinct identity per instance.
    NVML ignores ``CUDA_VISIBLE_DEVICES``, so without this filter every
    instance on a host would enumerate every GPU and collide on one
    executor_id.

    Args:
        gpus: GPUs as enumerated by NVML, in physical index order.
        uuids: GPU UUIDs to keep. Matched case-insensitively, with or
            without the ``GPU-`` prefix.
        indices: physical NVML indices to keep.
        strict: when True (the default) a requested UUID/index that is not
            present is an error. When False the missing entries are dropped
            and whatever survives is returned — used for the periodic
            hardware refresh, where a device that has transiently vanished
            (Xid, driver reset, ECC retirement) must not turn into a
            broadcast claiming zero GPUs.

    Returns:
        The kept GPUs in their original NVML order, as a new list. Each
        ``GpuInfo.index`` keeps its PHYSICAL NVML value — subsets are never
        renumbered, because that index is the key into the per-physical-GPU
        MIG map assembled in :func:`build_hw_static`.

        Note that the physical index is NOT what pins proof workers today:
        ``ProofService`` receives only a GPU *count* and walks
        ``range(gpu_count)``, so proof slot k always runs on CUDA device k
        (see :func:`_assert_proof_slot_mapping_safe`).

    Raises:
        ValueError: if both filters are supplied, if a filter names the same
            GPU twice, if the selection ends up empty, or — when ``strict``
            — if any requested UUID or index is absent. This fails closed on
            purpose: registering a GPU set other than the one the operator
            asked for is exactly the sybil condition validators ban (see
            neurons/validator/sybil/scanner.py).
    """
    if uuids and indices:
        raise ValueError(
            f"Set only one of {GPU_UUIDS_ENV} and {GPU_INDICES_ENV}, not both"
        )
    if not uuids and not indices:
        return list(gpus)

    if uuids:
        available = {_normalize_gpu_uuid(g.uuid) for g in gpus}
        missing = [u for u in uuids if _normalize_gpu_uuid(u) not in available]
        if missing:
            message = (
                f"requested GPU UUID(s) not present on this host: "
                f"{', '.join(missing)} — detected: "
                f"{', '.join(g.uuid for g in gpus) or '(none)'}"
            )
            if strict:
                raise ValueError(message)
            bt.logging.warning(f"GPU subset refresh: {message}")
        wanted_uuids = {_normalize_gpu_uuid(u) for u in uuids}
        selected = [g for g in gpus if _normalize_gpu_uuid(g.uuid) in wanted_uuids]
    else:
        available_idx = {g.index for g in gpus}
        missing_idx = [i for i in indices if i not in available_idx]
        if missing_idx:
            message = (
                f"requested GPU index/indices not present on this host: "
                f"{', '.join(str(i) for i in missing_idx)} — detected: "
                f"{', '.join(str(g.index) for g in gpus) or '(none)'}"
            )
            if strict:
                raise ValueError(message)
            bt.logging.warning(f"GPU subset refresh: {message}")
        wanted_idx = set(indices)
        selected = [g for g in gpus if g.index in wanted_idx]

    if not selected:
        raise ValueError("GPU subset selection matched no GPU")

    return selected


def _assert_proof_slot_mapping_safe(selected: list[GpuInfo]) -> None:
    """Refuse a subset whose physical indices are not exactly ``0..n-1``.

    ``ProofService`` is constructed with a GPU *count*, not a device list
    (neurons/miner/miner.py, ``gpu_count=len(gpus)``). It then iterates
    ``for gpu_idx in range(self.gpu_count)`` and launches each worker with
    ``CUDA_VISIBLE_DEVICES=str(gpu_idx)`` (proof_service.py ``_worker_env``),
    an absolute overwrite the operator cannot pre-empt. So proof slot k runs
    on CUDA device k no matter which physical GPUs this executor selected.

    An executor holding NVML indices 4..7 would therefore prove on physical
    GPUs 0..3 — its neighbour's silicon. The proofs still *verify* (the
    validator re-derives the seed from the reported 0-based slot), but the
    wrong hardware did the work: the neighbour's rentals are disturbed and
    both executors trip the timing and canary gates.

    Until ProofService accepts the selected device list, only a subset that
    already starts at physical 0 and is contiguous is safe, so that the
    slot->device mapping is the identity. That permits "mine on the first N
    of my GPUs" but blocks the multi-instance split.
    """
    physical = [g.index for g in selected]
    if physical == list(range(len(physical))):
        return
    raise ValueError(
        "GPU subset {"
        + ",".join(str(i) for i in physical)
        + "} is not supported: proof workers are pinned by ordinal position "
        "(CUDA_VISIBLE_DEVICES=<slot>, slot in range(gpu_count)), not by "
        "physical index, so this executor would run its proofs on physical "
        "GPU(s) " + ",".join(str(i) for i in range(len(physical)))
        + " — hardware it does not own. Only a subset covering physical "
        "indices 0..n-1 is safe today. Splitting a host across several "
        "executors additionally needs ProofService to take the selected "
        "device list; see docs/multi-executor.md."
    )


def detect_gpus(strict: bool = True) -> list[GpuInfo]:
    """Detect all NVIDIA GPUs via pynvml.

    If ``NODEXO_GPU_UUIDS`` or ``NODEXO_GPU_INDICES`` is set, only that
    subset is returned (see :func:`select_gpu_subset`). With neither set the
    full device list is returned, exactly as before.

    Args:
        strict: startup semantics. True means a bad or unsatisfiable filter
            yields ``[]`` so the miner refuses to start, and the subset must
            pass :func:`_assert_proof_slot_mapping_safe`. False is for the
            periodic hardware refresh: the filter is applied best-effort and
            a device that has gone missing is dropped with a warning rather
            than collapsing the report to "this host has no GPUs".
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8")
            # Compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

            gpus.append(GpuInfo(
                index=i,
                name=name,
                uuid=uuid,
                vram_mb=mem_info.total // (1024 * 1024),
                driver_version=driver,
                compute_capability=f"{major}.{minor}",
            ))
        pynvml.nvmlShutdown()
    except Exception as e:
        bt.logging.error(f"GPU detection failed: {e}")
        return []

    try:
        want_uuids, want_indices = _parse_gpu_filter_env()
        selected = select_gpu_subset(
            gpus, uuids=want_uuids, indices=want_indices, strict=strict,
        )
        if strict and (want_uuids or want_indices):
            _assert_proof_slot_mapping_safe(selected)
    except ValueError as e:
        # Fail closed: better to report no GPU (the miner refuses to start)
        # than to register a set the operator did not ask for.
        bt.logging.error(f"GPU subset selection failed: {e}")
        return []

    if len(selected) != len(gpus):
        bt.logging.info(
            f"GPU subset: {len(selected)} of {len(gpus)} selected "
            f"(indices {','.join(str(g.index) for g in selected)})"
        )
    return selected


def detect_system() -> SystemInfo:
    """Detect CPU, RAM, and OS info."""
    import psutil
    cpu_model = platform.processor() or "unknown"
    # Try to get a better CPU model from /proc/cpuinfo on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    return SystemInfo(
        cpu_model=cpu_model,
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        ram_gb=psutil.virtual_memory().total // (1024**3),
        hostname=platform.node(),
        os_version=platform.platform(),
    )


def detect_mig_for_gpu(handle) -> dict:
    """Per-physical-GPU MIG (Multi-Instance GPU) state.

    MIG hard-partitions a datacenter GPU's silicon (SM clusters,
    L2 cache, memory controllers) into independent slices, each
    with its own MMU + UUID. Tenants on different slices cannot
    observe each other's compute or VRAM — the isolation is
    enforced by the GPU's MMU, not the kernel driver.

    For nodexo this matters in two ways:
      - Renter UX: a MIG-isolated executor offers cryptographic
        tenant isolation (no co-tenant can side-channel into the
        rental even on hardware-rootkit threat models).
      - Sybil defense: a miner cannot run a second "sybil" identity
        on the same MIG slice — the slice has a fixed UUID and
        physical resource budget. Cross-slice sybils would need
        TWO MIG instances exposed as TWO executors, which the
        validator can see.

    Returns:
      {
        "capable": bool,    # silicon supports MIG (A100/H100/H200/A30/Bxxx)
        "enabled": bool,    # MIG mode is currently active on this device
        "devices": [        # populated only when enabled
            {"index": int, "uuid": str, "vram_bytes": int},
            ...
        ],
      }

    Non-MIG GPUs (RTX 4090, RTX 5090, A6000, V100, …) return
    {"capable": False, "enabled": False, "devices": []}.
    """
    out = {"capable": False, "enabled": False, "devices": []}
    try:
        import pynvml
        current, _pending = pynvml.nvmlDeviceGetMigMode(handle)
        # The call succeeded → silicon is MIG-capable.
        out["capable"] = True
        enabled = (current == pynvml.NVML_DEVICE_MIG_ENABLE)
        out["enabled"] = enabled
        if enabled:
            try:
                max_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
            except Exception:
                max_count = 8  # A100 has up to 7 1g.5gb slices; cap defensive
            for i in range(max_count):
                try:
                    mig = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, i)
                    uuid = pynvml.nvmlDeviceGetUUID(mig)
                    if isinstance(uuid, bytes):
                        uuid = uuid.decode("utf-8")
                    mem = pynvml.nvmlDeviceGetMemoryInfo(mig)
                    out["devices"].append({
                        "index": i,
                        "uuid": uuid,
                        "vram_bytes": int(mem.total),
                    })
                except Exception:
                    # Slot not populated — common on partial MIG configs.
                    continue
    except Exception:
        # NVML_ERROR_NOT_SUPPORTED on non-MIG silicon, OK.
        return out
    return out


def detect_mig_summary(indices: list[int] | None = None) -> dict:
    """Top-level MIG summary across this executor's physical GPUs. Goes
    into `hw_static` so the validator can route MIG-isolated executors
    to a higher-tier marketplace listing without re-querying.

    Args:
        indices: physical NVML indices to report on. ``None`` (the default)
            means every GPU on the host. Pass this executor's own indices
            when a GPU subset filter is in play — ``capable_any`` and
            ``enabled_any`` drive marketplace tier selection, so computing
            them across a co-tenant's GPUs would advertise this executor
            into the isolated tier on someone else's MIG configuration.

    Returns:
      {
        "capable_any": bool,   # at least one reported GPU supports MIG
        "enabled_any": bool,   # at least one reported GPU has MIG active
        "per_gpu": [           # one entry per reported physical GPU
            {"index": 0, "capable": ..., "enabled": ..., "devices": [...]},
            ...
        ],
      }
    """
    out = {"capable_any": False, "enabled_any": False, "per_gpu": []}
    wanted = None if indices is None else set(indices)
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            if wanted is not None and i not in wanted:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = detect_mig_for_gpu(handle)
            info["index"] = i
            out["per_gpu"].append(info)
            if info["capable"]:
                out["capable_any"] = True
            if info["enabled"]:
                out["enabled_any"] = True
        pynvml.nvmlShutdown()
    except Exception as e:
        bt.logging.debug(f"detect_mig_summary failed: {e}")
    return out


def _env_selected_nvml_indices(pynvml, count: int) -> set[int] | None:
    """Resolve the configured GPU filter to a set of physical NVML indices.

    Returns ``None`` when no filter is configured or the configuration is
    unusable, meaning "report every GPU" — the pre-filter behaviour. The
    unfiltered path issues no extra NVML calls and gains no new failure
    mode. This is telemetry, not identity, so it degrades to best-effort
    rather than failing closed; :func:`detect_gpus` is the gate that stops a
    miner with a broken filter from starting at all.
    """
    try:
        want_uuids, want_indices = _parse_gpu_filter_env()
    except ValueError:
        return None
    if want_uuids and want_indices:
        return None
    if want_indices:
        return {i for i in want_indices if 0 <= i < count}
    if not want_uuids:
        return None

    wanted = {_normalize_gpu_uuid(u) for u in want_uuids}
    selected: set[int] = set()
    for i in range(count):
        try:
            raw_uuid = pynvml.nvmlDeviceGetUUID(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            )
        except Exception:
            continue
        if isinstance(raw_uuid, bytes):
            raw_uuid = raw_uuid.decode("utf-8")
        if _normalize_gpu_uuid(raw_uuid) in wanted:
            selected.add(i)
    return selected


def get_gpu_utilization() -> list[dict]:
    """Get current per-GPU utilization, memory, temp, and power.

    Scoped to this executor's GPU subset when ``NODEXO_GPU_UUIDS`` or
    ``NODEXO_GPU_INDICES`` is set; with neither set, every GPU on the host
    is reported exactly as before. Without that scoping a one-GPU executor
    on a split host would return eight rows here while `hardware_info`
    carries one, and a renter would be shown a co-tenant's workload.

    Used by:
      - Idle checks inside the miner.
      - The miner's `/hardware` endpoint, which the fleet dashboard
        AND the renter-side rental detail view both read. Keep the
        fields parallel to validator's hw_live so both renderers can
        format identical telemetry.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        selected = _env_selected_nvml_indices(pynvml, count)
        utils = []
        for i in range(count):
            if selected is not None and i not in selected:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
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
                # nvmlDeviceGetPowerUsage returns milliwatts.
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                pass
            utils.append({
                "index": i,
                "gpu_util_pct": util.gpu,
                "mem_util_pct": round(mem.used / mem.total * 100, 1) if mem.total > 0 else 0,
                "mem_used_mb": mem.used // (1024 * 1024),
                "mem_total_mb": mem.total // (1024 * 1024),
                "temp_c": temp_c,
                "power_w": round(power_w, 1),
            })
        pynvml.nvmlShutdown()
        return utils
    except Exception as e:
        bt.logging.error(f"GPU utilization check failed: {e}")
        return []


def _resolve_loaded_nvml_path() -> str | None:
    """Find the libnvidia-ml.so path actually mapped into THIS process.

    Reads /proc/self/maps and returns the first line referencing
    libnvidia-ml.so. Falls back to opening NVML in-process to force the
    library to be loaded if it isn't already.

    Why /proc/self/maps and not a hardcoded path list:
      - An LD_PRELOAD shim presents a different .so at process-load
        time, the hardcoded path on disk is the unmodified original,
        so hashing it gives a false-clean. Reading from /proc/self/maps
        tells us what the kernel actually has mapped into this PID.
      - A `mount --bind` overlay on the lib path inside our namespace
        only shows up correctly through /proc/self/maps.
      - An attacker who patches the library on disk in place would
        succeed against either approach; that's a layer-deeper attack
        and TEE is the only defense.
    """
    try:
        # Force NVML to load so it appears in /proc/self/maps even when
        # the caller hasn't touched pynvml yet.
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
        except Exception:
            pass
        with open("/proc/self/maps") as f:
            for line in f:
                # /proc/<pid>/maps lines end with the mapped file path; if
                # the line contains libnvidia-ml.so anywhere, grab it.
                if "libnvidia-ml.so" in line:
                    # path is the part after the last space.
                    parts = line.rstrip("\n").split(" ")
                    # The path component starts after the file offset etc.;
                    # the path is everything from the first '/' to end of line.
                    idx = line.find("/")
                    if idx != -1:
                        return line[idx:].strip()
    except Exception:
        pass
    return None


def get_nvml_digest() -> dict:
    """Compute MD5+SHA256 of the loaded libnvidia-ml.so, with path attest.

    Returns a dict so the validator can verify both the path the miner
    claims to have hashed AND the hash itself. Without the path, a
    miner could compute a sha of the *real* library while loading a
    shim — a static-path approach has this hole. Resolving the
    path via /proc/self/maps closes it for LD_PRELOAD and mount-bind
    attacks (per _resolve_loaded_nvml_path).

    The returned shape:
        {
          "path": "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.575.57.08",
          "md5":  "abcd...",
          "sha256": "ef01...",
          "resolution": "proc_maps" | "fallback_path_list",
        }

    Failures to resolve return {"path":"", "md5":"", "sha256":"",
    "resolution":"unknown"} — never "looks valid but is empty"; the
    validator's scanner uses presence-of-path as a proof-of-life.
    """
    resolved = _resolve_loaded_nvml_path()
    resolution = "proc_maps"
    if resolved is None or not os.path.exists(resolved):
        # Last-resort fallback — kept ONLY for environments where
        # /proc/self/maps isn't readable (container with --read-only
        # /proc, etc.). Defensive: known to be the weaker path.
        resolution = "fallback_path_list"
        for path in (
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1",
            "/usr/lib/libnvidia-ml.so.1",
        ):
            if os.path.exists(path):
                resolved = path
                break
    if not resolved or not os.path.exists(resolved):
        return {"path": "", "md5": "", "sha256": "", "resolution": "unknown"}

    try:
        data = Path(resolved).read_bytes()
        return {
            "path": resolved,
            "md5": hashlib.md5(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "resolution": resolution,
        }
    except Exception as e:
        bt.logging.warning(f"Failed to hash {resolved}: {e}")
        return {"path": resolved, "md5": "", "sha256": "",
                "resolution": resolution}


def get_nvml_digest_compat() -> str:
    """Legacy `MD5:SHA256` string form, for the heartbeat field that
    older validators key on. New code should consume the structured
    dict from get_nvml_digest().
    """
    d = get_nvml_digest()
    if d.get("md5") and d.get("sha256"):
        return f"{d['md5']}:{d['sha256']}"
    return "unknown"


def detect_sysbox() -> bool:
    """Detect if Sysbox runtime is available."""
    return os.path.exists("/usr/bin/sysbox-runc")


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _decode_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _nearest_existing_path(path: str) -> str:
    p = Path(path).expanduser()
    while not p.exists() and p != p.parent:
        p = p.parent
    return str(p if p.exists() else Path("/"))


def _docker_root_dir() -> str:
    configured = os.environ.get("NODEXO_DOCKER_ROOT_DIR") or os.environ.get("DOCKER_ROOT_DIR")
    if configured:
        return _nearest_existing_path(configured)
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        root = proc.stdout.strip()
        if proc.returncode == 0 and root:
            return _nearest_existing_path(root)
    except Exception:
        pass
    return _nearest_existing_path("/var/lib/docker")


def _mount_info_for_path(path: str) -> dict[str, Any]:
    path_real = os.path.realpath(_nearest_existing_path(path))
    best: dict[str, Any] = {
        "mountpoint": "/",
        "source": "",
        "fstype": "",
    }
    best_len = 0
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                if " - " not in line:
                    continue
                left, right = line.rstrip("\n").split(" - ", 1)
                left_parts = left.split()
                right_parts = right.split()
                if len(left_parts) < 5 or len(right_parts) < 3:
                    continue
                mountpoint = os.path.realpath(_decode_mount_field(left_parts[4]))
                if path_real == mountpoint or path_real.startswith(mountpoint.rstrip("/") + "/"):
                    if len(mountpoint) >= best_len:
                        best_len = len(mountpoint)
                        best = {
                            "mountpoint": mountpoint,
                            "source": _decode_mount_field(right_parts[1]),
                            "fstype": right_parts[0],
                        }
    except Exception:
        pass
    return best


def _flatten_lsblk_rows(rows: list[dict[str, Any]], out: dict[str, dict[str, Any]]) -> None:
    for row in rows or []:
        name = str(row.get("name") or row.get("kname") or "")
        kname = str(row.get("kname") or name)
        if name:
            out[name] = row
        if kname:
            out[kname] = row
        for child in row.get("children") or []:
            child.setdefault("pkname", kname or name)
        _flatten_lsblk_rows(row.get("children") or [], out)


def _lsblk_index() -> dict[str, dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "lsblk", "-b", "-J",
                "-o", "NAME,KNAME,PKNAME,TYPE,SIZE,ROTA,MOUNTPOINTS,FSTYPE,MODEL,SERIAL,TRAN",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        data = json.loads(proc.stdout)
        out: dict[str, dict[str, Any]] = {}
        _flatten_lsblk_rows(data.get("blockdevices") or [], out)
        return out
    except Exception:
        return {}


def _classify_storage(source: str, fstype: str, block: dict[str, Any] | None) -> str:
    fstype_l = (fstype or "").lower()
    source_l = (source or "").lower()
    if fstype_l in _NETWORK_FS_TYPES or ":" in source_l:
        return "network"
    if fstype_l in {"overlay", "aufs"}:
        return "overlay"
    if block:
        tran = str(block.get("tran") or "").lower()
        name = str(block.get("name") or block.get("kname") or source_l).lower()
        rota = block.get("rota")
        if tran == "nvme" or "nvme" in name:
            return "nvme"
        if rota is False or str(rota) == "0":
            return "ssd"
        if rota is True or str(rota) == "1":
            return "hdd"
    return fstype_l or "unknown"


def _block_device_info(source: str) -> dict[str, Any] | None:
    if not source or not source.startswith("/dev/"):
        return None
    index = _lsblk_index()
    if not index:
        return None
    name = os.path.basename(source)
    row = index.get(name)
    if row is None and name.startswith("mapper/"):
        row = index.get(name.split("/", 1)[1])
    if row is None:
        return None
    parent = index.get(str(row.get("pkname") or ""))

    def _int_value(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    out = {
        "name": row.get("name") or row.get("kname") or name,
        "type": row.get("type") or "",
        "size_bytes": _int_value(row.get("size")),
        "rota": row.get("rota"),
        "tran": row.get("tran") or "",
        "model": row.get("model") or "",
    }
    if parent:
        out["parent_name"] = parent.get("name") or parent.get("kname") or ""
        out["parent_type"] = parent.get("type") or ""
        out["parent_size_bytes"] = _int_value(parent.get("size"))
        out["parent_tran"] = parent.get("tran") or ""
    return out


def _storage_scope_for_path(path: str) -> dict[str, Any]:
    import psutil

    resolved = _nearest_existing_path(path)
    usage = psutil.disk_usage(resolved)
    mount = _mount_info_for_path(resolved)
    block = _block_device_info(str(mount.get("source") or ""))
    kind = _classify_storage(str(mount.get("source") or ""), str(mount.get("fstype") or ""), block)
    return {
        "path": path,
        "resolved_path": resolved,
        "mountpoint": mount.get("mountpoint") or "",
        "source": mount.get("source") or "",
        "fstype": mount.get("fstype") or "",
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "available_bytes": int(usage.free),
        "percent": float(usage.percent),
        "block_device": block,
        "kind": kind,
        "is_network_fs": kind == "network",
        "capacity_origin": "filesystem_mount",
    }


def detect_storage(force_refresh: bool = False) -> dict[str, Any]:
    """Detect rental-relevant storage without pretending root == usable disk.

    Docker's data root is the capacity that matters for rental containers.
    On hosted platforms this may be a partition, overlay mount, or network
    filesystem; the payload carries source/fstype metadata so UIs can label
    the number correctly.
    """
    global _STORAGE_STATIC_CACHE
    with _STORAGE_LOCK:
        if _STORAGE_STATIC_CACHE is not None and not force_refresh:
            return json.loads(json.dumps(_STORAGE_STATIC_CACHE))

    docker_path = _docker_root_dir()
    storage = {
        "reported_scope": "docker_storage_filesystem",
        "docker": _storage_scope_for_path(docker_path),
        "root": _storage_scope_for_path("/"),
    }
    with _STORAGE_LOCK:
        _STORAGE_STATIC_CACHE = storage
    return json.loads(json.dumps(storage))


def storage_live_snapshot() -> dict[str, Any]:
    storage = detect_storage()
    docker = storage.get("docker") or {}
    path = str(docker.get("resolved_path") or docker.get("path") or _docker_root_dir())
    live = _storage_scope_for_path(path)
    return {
        "reported_scope": "docker_storage_filesystem",
        "docker": {
            **{k: v for k, v in docker.items() if k not in {"used_bytes", "free_bytes", "percent"}},
            "used_bytes": live.get("used_bytes", 0),
            "free_bytes": live.get("free_bytes", 0),
            "available_bytes": live.get("available_bytes", live.get("free_bytes", 0)),
            "total_bytes": live.get("total_bytes", docker.get("total_bytes", 0)),
            "percent": live.get("percent", 0),
        },
    }


def get_cached_network_speedtest() -> dict[str, Any]:
    with _NETWORK_SPEED_LOCK:
        return dict(_NETWORK_SPEED_CACHE)


def run_network_speed_test() -> dict[str, Any]:
    """Run a miner-side Internet speed sample and cache the result.

    This is operator/renter telemetry, not a fraud-proof attestation: a miner
    can fake self-reported bandwidth. Validators should only score bandwidth
    after independent spot probes or canary measurements.
    """
    now = int(time.time())
    result: dict[str, Any] = {
        "source": "miner_self_test",
        "provider": os.environ.get("MINER_SPEEDTEST_PROVIDER", "cloudflare"),
        "measured_at": now,
        "download_mbps": None,
        "upload_mbps": None,
        "download_bytes": 0,
        "upload_bytes": 0,
        "download_ms": None,
        "upload_ms": None,
        "error": "",
    }
    if not _env_bool("MINER_NETWORK_SPEEDTEST_ENABLED", True):
        result["error"] = "disabled"
        with _NETWORK_SPEED_LOCK:
            _NETWORK_SPEED_CACHE.clear()
            _NETWORK_SPEED_CACHE.update(result)
        return dict(result)

    timeout = _env_int("MINER_SPEEDTEST_TIMEOUT_S", 20, minimum=3)
    download_bytes = _env_int("MINER_SPEEDTEST_DOWNLOAD_BYTES", 25_000_000, minimum=1_000_000)
    upload_bytes = _env_int("MINER_SPEEDTEST_UPLOAD_BYTES", 5_000_000, minimum=256_000)
    download_url = os.environ.get(
        "MINER_SPEEDTEST_DOWNLOAD_URL",
        f"https://speed.cloudflare.com/__down?bytes={download_bytes}",
    )
    upload_url = os.environ.get("MINER_SPEEDTEST_UPLOAD_URL", "https://speed.cloudflare.com/__up")
    headers = {"User-Agent": "nodexo-miner/0.1"}
    errors: list[str] = []

    try:
        req = urllib.request.Request(download_url, headers=headers)
        start = time.monotonic()
        count = 0
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
        elapsed = max(time.monotonic() - start, 0.001)
        result["download_bytes"] = count
        result["download_ms"] = round(elapsed * 1000, 1)
        result["download_mbps"] = round((count * 8) / elapsed / 1_000_000, 2)
    except Exception as e:
        errors.append(f"download: {type(e).__name__}")

    try:
        body = b"\0" * upload_bytes
        req = urllib.request.Request(upload_url, data=body, headers=headers, method="POST")
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(1024)
        elapsed = max(time.monotonic() - start, 0.001)
        result["upload_bytes"] = upload_bytes
        result["upload_ms"] = round(elapsed * 1000, 1)
        result["upload_mbps"] = round((upload_bytes * 8) / elapsed / 1_000_000, 2)
    except Exception as e:
        errors.append(f"upload: {type(e).__name__}")

    result["error"] = "; ".join(errors)
    result["measured_at"] = int(time.time())
    with _NETWORK_SPEED_LOCK:
        _NETWORK_SPEED_CACHE.clear()
        _NETWORK_SPEED_CACHE.update(result)
    if errors:
        bt.logging.debug(f"Network speed test completed with errors: {result['error']}")
    else:
        bt.logging.info(
            "Network speed test: "
            f"down={result['download_mbps']} Mbps up={result['upload_mbps']} Mbps"
        )
    return dict(result)


def get_or_create_executor_id(gpus: list[GpuInfo]) -> str:
    """Get or create a persistent executor ID.

    executor_id = SHA256(gpu_uuid_0 || gpu_uuid_1 || ... || system_uuid)

    Stored in ~/.nodexo/executor_identity.json.
    Changing GPUs changes the executor_id, requiring re-registration.

    Stable executor identity pattern (see protocol design notes).

    Raises:
        ValueError: if ``gpus`` is empty. detect_gpus() returns [] both when
            NVML fails and when a GPU subset filter is rejected; persisting
            an identity from that would overwrite a registered executor's
            executor_identity.json with an empty gpu_uuids list on nothing
            worse than a typo in NODEXO_GPU_UUIDS.
    """
    if not gpus:
        raise ValueError(
            "refusing to derive an executor identity from an empty GPU list — "
            "GPU detection returned nothing. Check the log for 'GPU detection "
            f"failed' or 'GPU subset selection failed', and verify "
            f"{GPU_UUIDS_ENV}/{GPU_INDICES_ENV} if you have set them."
        )

    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        IDENTITY_PATH.parent.chmod(0o700)
    except OSError:
        pass

    # Check if we have a stored identity
    if IDENTITY_PATH.exists():
        try:
            IDENTITY_PATH.chmod(0o600)
        except OSError:
            pass
        try:
            data = json.loads(IDENTITY_PATH.read_text())
            stored_gpu_uuids = data.get("gpu_uuids", [])
            current_gpu_uuids = [g.uuid for g in gpus]
            # If GPUs haven't changed, return stored ID
            if stored_gpu_uuids == current_gpu_uuids:
                return data["executor_id"]
            bt.logging.warning(
                f"GPU UUIDs changed (stored={stored_gpu_uuids}, current={current_gpu_uuids}) — generating new executor_id"
            )
        except Exception:
            pass

    # Generate new executor_id
    gpu_uuids = sorted(g.uuid for g in gpus)
    system_uuid = _get_system_uuid()
    payload = "||".join(gpu_uuids + [system_uuid])
    executor_id = hashlib.sha256(payload.encode()).hexdigest()

    # Persist
    tmp_path = IDENTITY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps({
        "executor_id": executor_id,
        "gpu_uuids": [g.uuid for g in gpus],
        "system_uuid": system_uuid,
        "created_at": __import__("time").time(),
    }, indent=2))
    tmp_path.chmod(0o600)
    tmp_path.replace(IDENTITY_PATH)

    bt.logging.info(f"Generated executor_id: {executor_id}")
    return executor_id


def _get_system_uuid() -> str:
    """Get a system-level UUID (DMI product UUID or fallback)."""
    try:
        return Path("/sys/class/dmi/id/product_uuid").read_text().strip()
    except Exception:
        pass
    try:
        return Path("/etc/machine-id").read_text().strip()
    except Exception:
        return str(uuid_mod.uuid4())


def detect_container_images() -> dict[str, Any]:
    """Report whether curated renter images are already cached locally.

    The miner deliberately reports only a configured catalog, not every
    Docker image on the box. Rental images can contain private repository
    names, and a busy host may have hundreds of layers. The validator only
    needs to know whether public warm presets can start without a pull.
    """
    from common.images import (
        image_digest_matches,
        image_expected_digest,
        image_runtime_reference,
        rental_image_catalog,
    )

    catalog = rental_image_catalog()
    cached: list[dict[str, Any]] = []
    missing: list[str] = []
    errors: dict[str, str] = {}

    for image in catalog:
        try:
            inspect_ref = image
            result = subprocess.run(
                ["docker", "image", "inspect", inspect_ref],
                shell=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                inspect_ref = image_runtime_reference(image)
                if inspect_ref != image:
                    result = subprocess.run(
                        ["docker", "image", "inspect", inspect_ref],
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
        except Exception as e:
            errors[image] = str(e)[:160]
            missing.append(image)
            continue
        if result.returncode != 0:
            missing.append(image)
            continue
        try:
            rows = json.loads(result.stdout or "[]")
            row = rows[0] if rows else {}
        except Exception as e:
            errors[image] = f"inspect_parse_failed: {e}"[:160]
            missing.append(image)
            continue
        repo_digests = row.get("RepoDigests") or []
        expected_digest = image_expected_digest(image)
        if expected_digest and not image_digest_matches(image, repo_digests):
            errors[image] = (
                "digest_mismatch: expected "
                f"{expected_digest}, got {repo_digests or ['<none>']}"
            )[:240]
            missing.append(image)
            continue
        cached.append({
            "image": image,
            "id": str(row.get("Id") or "")[:32],
            "repo_digests": repo_digests,
            "expected_digest": expected_digest or "",
            "digest_verified": bool(expected_digest),
            "size_bytes": int(row.get("Size") or 0),
        })

    return {
        "checked": catalog,
        "cached": cached,
        "missing": missing,
        "checked_at": int(time.time()),
        "errors": errors,
    }


def build_hw_static(gpus: list[GpuInfo] = None, sys_info: SystemInfo = None) -> dict:
    """Build static hardware fingerprint dict for heartbeat.

    Called once at startup (or when hardware changes).
    """
    if gpus is None:
        # strict=False: this runs on the periodic heartbeat refresh too, and
        # a device that has transiently vanished must not turn the broadcast
        # into a claim of zero GPUs. Startup uses the strict path.
        gpus = detect_gpus(strict=False)
    if sys_info is None:
        sys_info = detect_system()

    import psutil
    storage = detect_storage()
    docker_storage = storage.get("docker") or {}

    # Scoped to this executor's GPUs: mig_capable_any/mig_enabled_any below
    # drive validator-side marketplace tier filtering, and must not inherit
    # a co-tenant instance's MIG configuration on a split host.
    mig_summary = detect_mig_summary([g.index for g in gpus])
    # Build a per-GPU MIG info map keyed by index so the gpus list can
    # carry "mig" alongside the existing fields. Cheaper than nesting a
    # second lookup table on the validator side.
    mig_by_index = {entry["index"]: entry for entry in mig_summary["per_gpu"]}

    return {
        "system_uuid": _get_system_uuid(),
        "cpu": {
            "model": sys_info.cpu_model,
            "arch": platform.machine(),
            "cores_physical": psutil.cpu_count(logical=False) or 1,
            "cores_logical": psutil.cpu_count(logical=True) or 1,
            "freq_max_mhz": getattr(psutil.cpu_freq(), "max", 0) if psutil.cpu_freq() else 0,
        },
        "ram": {
            "total_bytes": psutil.virtual_memory().total,
        },
        "gpus": [
            {
                "name": g.name,
                "uuid": g.uuid,
                "vram_bytes": g.vram_mb * 1024 * 1024,
                "driver_version": g.driver_version,
                "compute_capability": g.compute_capability,
                "mig": mig_by_index.get(g.index, {
                    "capable": False, "enabled": False, "devices": [],
                }),
            }
            for g in gpus
        ],
        "disk": {
            "total_bytes": int(docker_storage.get("total_bytes") or 0),
            "type": docker_storage.get("kind") or "unknown",
            "scope": "docker_storage_filesystem",
        },
        "storage": storage,
        "sysbox_detected": detect_sysbox(),
        "nvml_digest": get_nvml_digest(),
        "container_images": detect_container_images(),
        # Top-level mirror of per-GPU MIG state for cheap validator-side
        # marketplace tier filtering ("isolated tier" = mig_enabled_any).
        "mig_capable_any": mig_summary["capable_any"],
        "mig_enabled_any": mig_summary["enabled_any"],
        "software_version": miner_version_str,
        "os": sys_info.os_version,
        "hostname": sys_info.hostname,
    }


def build_hw_live() -> dict:
    """Build live metrics dict for heartbeat.

    Called every heartbeat interval (~60s).
    """
    import psutil

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    storage = storage_live_snapshot()
    docker_storage = storage.get("docker") or {}
    net = psutil.net_io_counters()

    result = {
        "cpu_pct": psutil.cpu_percent(interval=0.1),
        "ram_used_bytes": mem.used,
        "ram_total_bytes": mem.total,
        "swap_used_bytes": swap.used,
        "swap_total_bytes": swap.total,
        "disk_used_bytes": int(docker_storage.get("used_bytes") or 0),
        "disk_total_bytes": int(docker_storage.get("total_bytes") or 0),
        "disk_free_bytes": int(docker_storage.get("available_bytes") or docker_storage.get("free_bytes") or 0),
        "net_rx_bytes": net.bytes_recv,
        "net_tx_bytes": net.bytes_sent,
        "storage": storage,
        "network_speedtest": get_cached_network_speedtest(),
    }

    # GPU metrics
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = 0
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
        except Exception:
            power = 0
        pynvml.nvmlShutdown()

        result["gpu_pct"] = util.gpu
        result["gpu_vram_used_bytes"] = gpu_mem.used
        result["gpu_vram_total_bytes"] = gpu_mem.total
        result["gpu_temp_c"] = temp
        result["gpu_power_w"] = power
    except Exception:
        pass

    return result
