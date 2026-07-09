"""
Nodexo Miner — the single daemon process for GPU providers.

Runs on the physical GPU machine. Handles everything:
1. Hardware detection and identity
2. On-chain registration (EVM + ComputeRegistry) — self-registers, no coordinator needed
3. Autonomous ZkGEMM proof generation every epoch
4. Signed proof broadcast to all validators
5. Docker/Sysbox container lifecycle for rentals
6. Heartbeat loop (lease renewal)
7. Serves API for validators (proof/rental) and operator CLI (health/status)

Usage:
  python -m neurons.miner.miner \\
    --wallet miner --hotkey default --subtensor-network test \\
    --endpoint http://YOUR_PUBLIC_IP:8091 --port 8091

For fleet management, use the nodexo CLI tool separately.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import bittensor as bt
from fastapi import Depends, FastAPI, HTTPException, Request

from common.proof_schedule import (
    DEFAULT_BEACON_MAX_OFFSET_BLOCKS,
    DEFAULT_JITTER_SECONDS,
)
from common.types import ValidatorDiscoveryResult
from neurons.miner.middleware.auth import ValidatorAuth

# Logging is handled exclusively by bt.logging (loguru-based).
# main() calls bt.logging.enable_info() before starting the server.

# ── Global state (initialized in lifespan) ─────────────────────
proof_service = None
broadcast_service = None
docker_service = None
hardware_info = None
miner_id = None  # executor_id (persisted identity of this GPU machine)
_rpc = None
_compute_registry = None
_validator_auth: ValidatorAuth | None = None
_evm_address: str = ""
_evm_private_key: str = ""


def _default_netuid_for_network(network: str) -> int:
    if network == "finney":
        return 106
    if network == "test":
        return 468
    return 0


def _resolve_chain_config_path(network: str, explicit_path: str = "") -> str:
    if explicit_path:
        return explicit_path
    root = Path(__file__).resolve().parents[2]
    if network == "finney":
        candidates = [root / "chain_config_mainnet.json", root / "chain_config.json"]
    elif network == "test":
        candidates = [root / "chain_config_testnet.json", root / "chain_config.json"]
    else:
        candidates = [root / "chain_config.json"]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float_value(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _call_or_value(value):
    return value() if callable(value) else value


def _split_host_port_value(value: str, fallback_port: int) -> tuple[str, int]:
    host = str(value or "").strip()
    port = int(fallback_port or 0)
    if not host:
        return host, port

    # Bittensor has exposed ip_str both as "/ipv4/host/tcp/port" and as
    # "/ipv4/host:port" while also setting axon.port. Strip the embedded port
    # for hostname/IPv4 values; leave raw IPv6 literals alone unless bracketed.
    should_parse = host.startswith("[") or host.count(":") == 1
    if should_parse:
        parsed = urlparse(f"//{host}")
        try:
            parsed_port = parsed.port
        except ValueError:
            parsed_port = None
        if parsed.hostname and parsed_port:
            host = parsed.hostname
            port = int(parsed_port)
    return host, port


def _parse_axon_host_port(axon) -> tuple[str, int]:
    raw_ip = _call_or_value(getattr(axon, "ip_str", "")) or getattr(axon, "ip", "")
    raw_port = int(getattr(axon, "port", 0) or 0)
    ip = str(raw_ip or "").strip()
    port = raw_port

    # Bittensor axons may expose ip_str as a multiaddr, e.g.
    # /ipv4/203.0.113.10/tcp/9443.
    if ip.startswith("/"):
        parts = [part for part in ip.strip("/").split("/") if part]
        for idx, part in enumerate(parts):
            if part in {"ip4", "ipv4", "ip6", "ipv6"} and idx + 1 < len(parts):
                ip = parts[idx + 1]
            elif part == "tcp" and idx + 1 < len(parts):
                try:
                    port = int(parts[idx + 1])
                except Exception:
                    pass
        ip, port = _split_host_port_value(ip, port)
    else:
        ip, port = _split_host_port_value(ip, port)
    return ip, port


def _axon_endpoint(axon, *, scheme: str = "http") -> str:
    serving = _call_or_value(getattr(axon, "is_serving", False))
    if not serving:
        return ""
    ip, port = _parse_axon_host_port(axon)
    ip = str(ip or "").strip()
    if not ip or port <= 0:
        return ""
    if ip in {"0.0.0.0", "::", "127.0.0.1", "localhost"}:
        return ""
    return f"{scheme}://{ip}:{port}"


def _validator_axon_endpoints_from_metagraph(
    mg,
    *,
    scheme: str = "http",
    exclude_uid: int | None = None,
) -> list:
    """Discover no-EVM validator endpoints from native Bittensor axon state.

    The miner does not choose these validators. They must be registered neurons
    with validator permit on this subnet and a serving axon endpoint on chain.
    """
    from common.types import ValidatorEndpoint

    hotkeys = list(getattr(mg, "hotkeys", []) or [])
    axons = list(getattr(mg, "axons", []) or [])
    permits = _validator_permits_from_metagraph(mg)
    count = min(len(hotkeys), len(axons), len(permits))
    endpoints = []
    for uid in range(count):
        try:
            if exclude_uid is not None and int(uid) == int(exclude_uid):
                continue
            if not bool(permits[uid]):
                continue
            endpoint = _axon_endpoint(axons[uid], scheme=scheme)
            if not endpoint:
                continue
            endpoints.append(ValidatorEndpoint(
                address=f"axon:{hotkeys[uid]}",
                proxy_endpoint=endpoint,
                uid=uid,
                is_active=True,
            ))
        except Exception:
            continue
    return endpoints


def _validator_permits_from_metagraph(mg) -> list:
    permits_obj = getattr(mg, "validator_permit", None)
    if permits_obj is None:
        permits_obj = getattr(mg, "validator_permits", [])
    if hasattr(permits_obj, "tolist"):
        return list(permits_obj.tolist())
    return list(permits_obj or [])


def _metagraph_validator_has_permit(mg, uid: int) -> bool:
    permits = _validator_permits_from_metagraph(mg)
    try:
        return bool(permits[int(uid)])
    except Exception:
        return False


def _validator_uid(value) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def _validator_registry_endpoints_with_current_permit(validators: list, mg) -> list:
    """Filter ValidatorRegistry rows against the current metagraph permit set.

    ValidatorRegistry registration is permit-gated at the time of registration,
    but a row can remain active after stake moves change validator permit. Miners
    re-check the current permit set before broadcasting proofs.
    """
    kept = []
    for validator in validators or []:
        try:
            uid = _validator_uid(getattr(validator, "uid", None))
            if not _metagraph_validator_has_permit(mg, uid):
                continue
            kept.append(validator)
        except Exception:
            continue
    return kept


def _merge_validator_endpoints(primary: list, extra: list) -> list:
    merged = []
    seen: set[str] = set()
    seen_uids: set[int] = set()
    for validator in list(primary or []) + list(extra or []):
        endpoint = _normalize_validator_proxy_endpoint(
            getattr(validator, "proxy_endpoint", "") or ""
        )
        if not endpoint:
            continue
        uid = _validator_uid(getattr(validator, "uid", None))
        if uid >= 0 and uid in seen_uids:
            continue
        key = endpoint.lower()
        if key in seen:
            continue
        seen.add(key)
        if uid >= 0:
            seen_uids.add(uid)
        validator.proxy_endpoint = endpoint
        merged.append(validator)
    return merged


def _normalize_validator_proxy_endpoint(endpoint: str) -> str:
    raw = (endpoint or "").strip().rstrip("/")
    if not raw:
        return ""

    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    scheme = parsed.scheme or "http"
    if scheme not in {"http", "https"}:
        return ""

    host = (parsed.hostname or "").strip()
    if not host:
        return ""

    try:
        port = parsed.port
    except ValueError:
        port = None
        # Recover the common bad registry/cache shape "host:port:port".
        # Anything more ambiguous is dropped rather than broadcast to a bad URL.
        netloc = (parsed.netloc or "").rsplit("@", 1)[-1]
        if not netloc.startswith("["):
            parts = netloc.rsplit(":", 2)
            if len(parts) == 3 and parts[1] == parts[2] and parts[1].isdigit():
                host = parts[0]
                port = int(parts[1])
    if not host:
        return ""

    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if port is not None:
        netloc = f"{netloc}:{int(port)}"
    return f"{scheme}://{netloc}{parsed.path or ''}"


def _validator_endpoint_role_probe_enabled() -> bool:
    return _env_bool("NODEXO_VALIDATOR_ENDPOINT_ROLE_PROBE", "1")


def _validator_endpoint_role_probe_timeout_s() -> float:
    return _env_float_value(
        "NODEXO_VALIDATOR_ENDPOINT_ROLE_PROBE_TIMEOUT_SECONDS",
        2.0,
        minimum=0.2,
    )


def _validator_endpoint_api_role(endpoint: str) -> str:
    """Classify a discovered endpoint as validator, non-validator, or unknown.

    Discovery keeps inconclusive EVM/configured endpoints because a validator
    can be restarting or briefly hidden behind a 502/connection-refused window;
    native axon endpoints must identify positively to avoid caching miner APIs.
    """
    if not _validator_endpoint_role_probe_enabled():
        return "validator"

    import urllib.error
    import urllib.request

    base = _normalize_validator_proxy_endpoint(endpoint)
    if not base:
        return "non-validator"

    timeout = _validator_endpoint_role_probe_timeout_s()
    saw_unknown = False
    for path in ("/health", "/version"):
        url = f"{base.rstrip('/')}{path}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "nodexo-miner-validator-discovery/1"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(4096).decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                continue
            service = str(data.get("service") or "").strip().lower()
            if service == "nodexo-validator":
                return "validator"
            if service == "nodexo-miner" or data.get("miner_id"):
                return "non-validator"
            saw_unknown = True
        except urllib.error.HTTPError as e:
            code = int(getattr(e, "code", 0) or 0)
            if code in {401, 403, 404}:
                # Health/version should be public on Nodexo validators. A
                # gated endpoint is not usable for proof broadcast discovery.
                return "non-validator"
            saw_unknown = True
            continue
        except TimeoutError:
            saw_unknown = True
            continue
        except OSError:
            saw_unknown = True
            continue
        except Exception:
            saw_unknown = True
            continue
    return "unknown" if saw_unknown else "non-validator"


def _validator_endpoint_is_validator_api(endpoint: str) -> bool:
    """Return True only for endpoints that identify as a Nodexo validator API."""
    return _validator_endpoint_api_role(endpoint) == "validator"


def _filter_validator_api_endpoints(validators: list) -> list:
    """Drop axon/registry rows that point at non-validator HTTP services."""
    if not _validator_endpoint_role_probe_enabled():
        return validators or []

    kept = []
    dropped = 0
    unknown = 0
    unknown_native = 0
    for validator in validators or []:
        endpoint = getattr(validator, "proxy_endpoint", "") or ""
        address = str(getattr(validator, "address", "") or "")
        role = _validator_endpoint_api_role(endpoint)
        if role == "validator":
            kept.append(validator)
        elif role == "unknown":
            if address.startswith("axon:"):
                unknown_native += 1
                bt.logging.warning(
                    f"Native validator endpoint role probe inconclusive; "
                    f"skipping until it identifies as validator: {endpoint}"
                )
            else:
                kept.append(validator)
                unknown += 1
                bt.logging.warning(
                    f"Validator endpoint role probe inconclusive; keeping endpoint "
                    f"for broadcast health tracking: {endpoint}"
                )
        else:
            dropped += 1
            bt.logging.warning(
                f"Dropped validator endpoint without Nodexo validator API: {endpoint}"
            )
    if unknown:
        bt.logging.warning(
            f"Validator discovery role probe kept {unknown} inconclusive endpoint(s)"
        )
    if unknown_native:
        bt.logging.warning(
            f"Validator discovery role probe skipped {unknown_native} "
            "inconclusive native endpoint(s)"
        )
    if dropped:
        bt.logging.warning(
            f"Validator discovery role probe dropped {dropped} endpoint(s)"
        )
    return ValidatorDiscoveryResult(
        kept,
        partial=unknown_native > 0,
        inconclusive_native_count=unknown_native,
    )


def _startup_chain_ready_timeout_s() -> float:
    # 0 means wait indefinitely. Public RPC rate limits should block miner
    # readiness, not let stale discovery or partial registration start proofs.
    return _env_float_value("NODEXO_MINER_CHAIN_READY_TIMEOUT_SECONDS", 0, minimum=0)


def _startup_chain_ready_retry_s() -> float:
    return _env_float_value("NODEXO_MINER_CHAIN_READY_RETRY_SECONDS", 5, minimum=0.1)


def _startup_chain_ready_max_retry_s() -> float:
    return _env_float_value("NODEXO_MINER_CHAIN_READY_MAX_RETRY_SECONDS", 120, minimum=1)


def _is_transient_startup_chain_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    transient_markers = (
        "429",
        "too many requests",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "502",
        "503",
        "504",
        "server disconnected",
        "connection reset",
        "connection refused",
        "eof",
        "no usable endpoints",
        "empty validator",
        "fresh validator discovery",
    )
    return any(marker in msg for marker in transient_markers)


def _run_startup_chain_step(label: str, fn):
    timeout_s = _startup_chain_ready_timeout_s()
    retry_s = _startup_chain_ready_retry_s()
    max_retry_s = _startup_chain_ready_max_retry_s()
    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    attempt = 0
    delay = retry_s

    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            msg_safe = str(e).encode("unicode_escape").decode()[:300]
            if not _is_transient_startup_chain_error(e):
                bt.logging.error(
                    f"Miner startup chain step failed permanently: "
                    f"{label}: {type(e).__name__}: {msg_safe}"
                )
                raise

            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{label} did not become ready within {timeout_s:.1f}s "
                    f"after {attempt} attempt(s): {msg_safe}"
                ) from e

            sleep_s = min(delay, max_retry_s)
            if deadline is not None:
                sleep_s = min(sleep_s, max(0.1, deadline - time.monotonic()))
            bt.logging.warning(
                f"Miner startup waiting for chain/RPC readiness: {label} "
                f"failed on attempt {attempt} ({type(e).__name__}: {msg_safe}); "
                f"retrying in {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)
            delay = min(max_retry_s, delay * 2)


def _startup_discover_validators(rpc, validator_registry, own_uid: int | None) -> list:
    validators = []
    mg = rpc.get_metagraph(False)

    native_validators = _validator_axon_endpoints_from_metagraph(
        mg,
        exclude_uid=own_uid,
    )
    if validator_registry:
        try:
            validators = validator_registry.get_active_validators(
                raise_on_error=not native_validators,
            )
        except Exception:
            if not native_validators:
                raise
            bt.logging.debug(
                "ValidatorRegistry startup discovery failed; using native "
                "validator endpoints from metagraph"
            )

    if validators:
        before = len(validators)
        validators = _validator_registry_endpoints_with_current_permit(validators, mg)
        dropped = before - len(validators)
        if dropped:
            bt.logging.warning(
                f"Dropped {dropped} ValidatorRegistry endpoint(s) without "
                "current validator permit during startup discovery"
            )

    merged = _filter_validator_api_endpoints(
        _merge_validator_endpoints(validators, native_validators)
    )
    if not merged:
        raise RuntimeError("fresh validator discovery returned no usable endpoints")
    bt.logging.info(
        f"Fresh startup validator discovery ready: {len(merged)} endpoint(s)"
    )
    return merged


def _validate_miner_endpoint(endpoint: str) -> None:
    """Reject executor URLs that validators/renters cannot reach in production."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"miner --endpoint must be an absolute http(s) URL, got {endpoint!r}")

    host = (parsed.hostname or "").strip().lower()
    allow_local = _env_bool("MINER_ALLOW_LOCAL_ENDPOINT")
    strict_public = _env_bool("MINER_STRICT_PUBLIC_ENDPOINT")

    local_name = host == "localhost" or host.endswith(".localhost")
    bad_bind_host = host in {"0.0.0.0", "::", ""}
    loopback_or_unspecified = False
    private_literal = False
    try:
        ip = ipaddress.ip_address(host)
        loopback_or_unspecified = ip.is_loopback or ip.is_unspecified
        private_literal = ip.is_private
    except ValueError:
        pass

    if (local_name or bad_bind_host or loopback_or_unspecified) and not allow_local:
        raise RuntimeError(
            "miner --endpoint advertises a local-only address. Set --endpoint "
            "to a validator-reachable URL, or set MINER_ALLOW_LOCAL_ENDPOINT=1 "
            "for local development."
        )
    if strict_public:
        if parsed.scheme != "https":
            raise RuntimeError("MINER_STRICT_PUBLIC_ENDPOINT=1 requires an https URL")
        if private_literal:
            raise RuntimeError(
                "MINER_STRICT_PUBLIC_ENDPOINT=1 rejects private/local IP literals"
            )


_GIB = 1 << 30
DEFAULT_MIN_RENTAL_STORAGE_GB = 30
DEFAULT_RENTAL_STORAGE_HEADROOM_GB = 10


def _env_int_value(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _request_int(
    body: dict,
    key: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int = 1_000_000,
) -> int:
    try:
        value = int(body.get(key, default))
    except (TypeError, ValueError):
        raise HTTPException(400, f"{key} must be an integer")
    if value < minimum or value > maximum:
        raise HTTPException(400, f"{key} must be between {minimum} and {maximum}")
    return value


def _request_bool(body: dict, key: str, default: bool) -> bool:
    value = body.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _docker_storage_bytes() -> tuple[int, int]:
    from neurons.miner.services.hardware_service import storage_live_snapshot

    storage = storage_live_snapshot()
    docker = storage.get("docker") or {}
    total = int(docker.get("total_bytes") or 0)
    available = int(docker.get("available_bytes") or docker.get("free_bytes") or 0)
    return total, available


def _docker_storage_capacity_gb() -> int:
    total, _available = _docker_storage_bytes()
    return int(total // _GIB)


def _docker_storage_available_gb() -> int:
    _total, available = _docker_storage_bytes()
    return int(available // _GIB)


def _min_rental_storage_gb() -> int:
    return _env_int_value(
        "MINER_MIN_RENTAL_STORAGE_GB",
        DEFAULT_MIN_RENTAL_STORAGE_GB,
        minimum=1,
    )


def _rental_storage_headroom_gb() -> int:
    return _env_int_value(
        "MINER_RENTAL_STORAGE_HEADROOM_GB",
        DEFAULT_RENTAL_STORAGE_HEADROOM_GB,
        minimum=0,
    )


def _rental_storage_reserve_gb() -> int:
    return _min_rental_storage_gb() + _rental_storage_headroom_gb()


def _proof_only_calibration_mode() -> bool:
    return _env_bool("NODEXO_PROOF_ONLY_CALIBRATION")


def _preflight_rental_resources() -> None:
    """Check host storage before registration."""
    min_storage_gb = _min_rental_storage_gb()
    storage_gb = _docker_storage_capacity_gb()
    if storage_gb < min_storage_gb:
        raise RuntimeError(
            "Docker storage filesystem is too small for rentals: "
            f"{storage_gb}GB total, require at least {min_storage_gb}GB. "
            "Move Docker data-root to a larger disk or set "
            "MINER_MIN_RENTAL_STORAGE_GB for dev-only runs."
        )
    available_gb = _docker_storage_available_gb()
    if available_gb < min_storage_gb:
        bt.logging.warning(
            "Docker storage has less usable free space than the minimum rental "
            f"profile: {available_gb}GB free, require {min_storage_gb}GB. "
            "Existing containers may continue, but new rentals are rejected "
            "until enough Docker storage is free."
        )


def _load_subnet_config_for_startup() -> None:
    """Load last-known-good subnet config and refresh once before registration."""
    try:
        from common.subnet_config_client import (
            fetch_and_apply_subnet_config,
            load_bundled_subnet_config,
            load_cached_subnet_config,
            subnet_config_url,
        )
        from common.proof_timing import timing_config_metadata

        cache_loaded = load_cached_subnet_config()
        if not cache_loaded:
            cache_loaded = load_bundled_subnet_config()
        url = subnet_config_url()
        if url:
            refreshed = asyncio.run(fetch_and_apply_subnet_config(url))
            if refreshed is None and not cache_loaded:
                bt.logging.warning(
                    "Subnet config fetch failed and no cache is available; "
                    "using built-in timing models."
                )
        elif not cache_loaded:
            bt.logging.info(
                "Subnet config URL not set; using built-in timing models. "
                "Set NODEXO_SUBNET_CONFIG_URL on production miners."
            )
        meta = timing_config_metadata()
        bt.logging.info(
            f"Timing config active: version={meta.get('version')} "
            f"models={meta.get('model_count')} source={meta.get('source')}"
        )
    except Exception as e:
        bt.logging.warning(f"Startup subnet config load failed: {e}")


def _preflight_gpu_timing_support(gpus: list) -> None:
    """Reject GPUs without reviewed proof timing before advertising capacity."""
    if not gpus:
        raise RuntimeError("No GPUs detected; cannot validate timing model support")

    from common.config import canonical_gpu_model_name
    from common.proof_timing import get_timing_model, is_timing_model_calibrated

    canonical_models = {
        canonical_gpu_model_name(getattr(gpu, "name", "") or "")
        for gpu in gpus
    }
    if len(canonical_models) != 1:
        raise RuntimeError(
            "Mixed GPU models in one executor are not supported for calibrated "
            f"proof timing: {', '.join(sorted(canonical_models))}"
        )

    gpu_model = next(iter(canonical_models))
    timing_model = get_timing_model(gpu_model)
    gpu_count = len(gpus)
    if is_timing_model_calibrated(gpu_model, gpu_count):
        bt.logging.info(
            f"GPU timing model calibrated: {gpu_model} × {gpu_count} "
            f"({timing_model.provenance or 'no provenance'})"
        )
        return

    message = (
        f"GPU model/count {gpu_model!r} × {gpu_count} has no calibrated timing model. "
        "This executor cannot be safely scored or rented until that count is "
        "measured and marked calibrated in subnet config."
    )
    if _env_bool("NODEXO_ALLOW_UNCALIBRATED_GPU"):
        bt.logging.warning(
            message
            + " Continuing because NODEXO_ALLOW_UNCALIBRATED_GPU=1; use only for calibration runs."
        )
        return
    raise RuntimeError(message)


def _docker_image_present(image: str) -> bool:
    import subprocess as _sp

    try:
        from common.images import image_runtime_reference

        inspect_ref = image
        result = _sp.run(
            ["docker", "image", "inspect", inspect_ref],
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            inspect_ref = image_runtime_reference(image)
            if inspect_ref != image:
                result = _sp.run(
                    ["docker", "image", "inspect", inspect_ref],
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
        if result.returncode != 0:
            return False
        rows = json.loads(result.stdout or "[]")
        row = rows[0] if rows else {}
        repo_digests = row.get("RepoDigests") or []
        from common.images import image_digest_matches, image_expected_digest

        expected = image_expected_digest(image)
        if expected and not image_digest_matches(image, repo_digests):
            bt.logging.warning(
                "Rental image digest mismatch; treating as not cached: "
                f"{image} expected={expected} got={repo_digests or ['<none>']}"
            )
            return False
        return True
    except Exception as e:
        bt.logging.warning(f"Could not inspect rental image {image}: {e}")
        return False


def _pull_docker_image(image: str, timeout_s: int | None = None) -> bool:
    import subprocess as _sp

    bt.logging.info(f"Pre-pulling rental image: {image}")
    if timeout_s is None:
        timeout_s = int(os.environ.get("MINER_IMAGE_PULL_TIMEOUT_S", "1800"))
    try:
        result = _sp.run(
            ["docker", "pull", image],
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as e:
        bt.logging.warning(f"Image pull failed for {image}: {e}")
        return False
    if result.returncode == 0:
        return True
    bt.logging.warning(
        f"Image pull failed for {image}: {(result.stderr or result.stdout)[:400]}"
    )
    return False


def _remove_docker_image(image: str) -> None:
    import subprocess as _sp

    try:
        _sp.run(
            ["docker", "image", "rm", image],
            shell=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as e:
        bt.logging.warning(f"Could not remove image after storage reserve breach: {image}: {e}")


def _prune_dangling_docker_images() -> None:
    import subprocess as _sp

    try:
        _sp.run(
            ["docker", "image", "prune", "-f"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as e:
        bt.logging.warning(f"Could not prune dangling Docker image layers: {e}")


def _cold_image_storage_reserve_gb(image: str, rental_storage_gb: int) -> int:
    from common.images import image_pull_reserve_gb

    return int(rental_storage_gb) + _rental_storage_headroom_gb() + image_pull_reserve_gb(image)


def _image_cold_pull_allowed(image: str) -> bool:
    if not _env_bool("MINER_ALLOW_COLD_IMAGE_PULL"):
        return False
    if _env_bool("MINER_ALLOW_CUSTOM_COLD_IMAGES"):
        return True
    from common.images import image_in_catalog

    return image_in_catalog(image)


def _preflight_rental_images(mode: str = "strict") -> None:
    """Ensure configured warm rental images are cached before registration."""
    if mode == "off":
        bt.logging.info("Rental image preflight disabled")
        return
    from common.images import required_rental_images

    required = required_rental_images()
    if not required:
        bt.logging.warning("Rental image preflight has an empty required image set")
        return

    missing = [image for image in required if not _docker_image_present(image)]
    if not missing:
        bt.logging.info(f"Rental image preflight passed ({len(required)} images cached)")
        return

    if mode == "pull":
        failed: list[str] = []
        reserve_gb = _rental_storage_reserve_gb()
        for image in missing:
            before_gb = _docker_storage_available_gb()
            if before_gb < reserve_gb:
                bt.logging.warning(
                    "Skipping warm image pull to preserve rental storage reserve: "
                    f"{image} ({before_gb}GB free, reserve {reserve_gb}GB)"
                )
                failed.append(image)
                continue
            if not _pull_docker_image(image):
                failed.append(image)
                continue
            after_gb = _docker_storage_available_gb()
            if after_gb < reserve_gb:
                bt.logging.warning(
                    "Pulled image would leave too little rental storage; removing "
                    f"{image} ({after_gb}GB free, reserve {reserve_gb}GB)"
                )
                _remove_docker_image(image)
                failed.append(image)
        if not failed:
            bt.logging.info(
                f"Rental image preflight passed after pulling {len(missing)} image(s)"
            )
            return
        missing = failed

    message = (
        "Required warm rental image(s) are not cached: "
        + ", ".join(missing)
        + ". Run scripts/setup_miner.sh or set NODEXO_REQUIRED_RENTAL_IMAGES "
        + "to the curated images this host should advertise."
    )
    if mode in {"pull", "strict"}:
        raise RuntimeError(message)
    bt.logging.warning(message)


def _preflight_optional_rental_images(mode: str = "warn") -> None:
    """Opportunistically cache non-required catalog images.

    Optional images should improve renter UX without consuming the storage
    profile the miner advertises. Each pull is attempted only when current
    Docker storage can cover the minimum rental profile, headroom, and the
    image-specific reserve estimate.
    """
    if mode == "off":
        bt.logging.info("Optional rental image warmup disabled")
        return
    from common.images import (
        image_pull_reserve_gb,
        rental_image_catalog,
        required_rental_images,
    )

    required = set(required_rental_images())
    optional = [image for image in rental_image_catalog() if image not in required]
    if not optional:
        return
    missing = [image for image in optional if not _docker_image_present(image)]
    if not missing:
        bt.logging.info(f"Optional rental image warmup passed ({len(optional)} images cached)")
        return
    if mode == "warn":
        bt.logging.warning(
            "Optional rental image(s) are not cached: " + ", ".join(missing)
        )
        return
    pulled = 0
    skipped: list[str] = []
    for image in missing:
        reserve_gb = _rental_storage_reserve_gb() + image_pull_reserve_gb(image)
        before_gb = _docker_storage_available_gb()
        if before_gb < reserve_gb:
            skipped.append(
                f"{image} ({before_gb}GB free, reserve {reserve_gb}GB)"
            )
            continue
        if not _pull_docker_image(image):
            skipped.append(image)
            continue
        after_gb = _docker_storage_available_gb()
        post_pull_reserve_gb = _rental_storage_reserve_gb()
        if after_gb < post_pull_reserve_gb:
            _remove_docker_image(image)
            _prune_dangling_docker_images()
            skipped.append(
                f"{image} ({after_gb}GB free after pull, reserve {post_pull_reserve_gb}GB)"
            )
            continue
        pulled += 1
    if pulled:
        bt.logging.info(f"Optional rental image warmup pulled {pulled} image(s)")
    if skipped:
        bt.logging.warning(
            "Optional rental image warmup skipped: " + "; ".join(skipped[:8])
        )


def _validated_container_resources(body: dict) -> dict[str, int]:
    """Validate a validator's container resource request against this host.

    Storage is currently a placement requirement and, where Docker supports it,
    a requested container quota. The default overlay2 driver usually cannot
    enforce per-container size, so we still reject hosts whose Docker storage
    filesystem is smaller than the requested rental profile.
    """
    min_storage_gb = _min_rental_storage_gb()
    storage_gb = _request_int(
        body,
        "storage_gb",
        min_storage_gb,
        minimum=min_storage_gb,
        maximum=100_000,
    )
    memory_gb = _request_int(body, "memory_gb", 16, minimum=1, maximum=100_000)
    cpu_count = _request_int(body, "cpu_count", 4, minimum=1, maximum=1024)

    docker_storage_gb = _docker_storage_available_gb()
    if docker_storage_gb < storage_gb:
        raise HTTPException(
            422,
            "requested storage exceeds this executor's Docker storage "
            f"free space ({storage_gb}GB requested, {docker_storage_gb}GB available)",
        )

    try:
        import psutil

        logical_cpu = int(psutil.cpu_count(logical=True) or 0)
        ram_gb = int(psutil.virtual_memory().total // _GIB)
    except Exception:
        logical_cpu = 0
        ram_gb = 0
    if logical_cpu and cpu_count > logical_cpu:
        raise HTTPException(
            422,
            f"requested CPU count exceeds host ({cpu_count} requested, {logical_cpu} available)",
        )
    if ram_gb and memory_gb > ram_gb:
        raise HTTPException(
            422,
            f"requested memory exceeds host ({memory_gb}GB requested, {ram_gb}GB available)",
        )

    return {
        "storage_gb": storage_gb,
        "memory_gb": memory_gb,
        "cpu_count": cpu_count,
    }


def _rental_ports_to_check(
    start: int,
    end: int,
    scope: str = "sample",
    samples: int = 3,
) -> list[int]:
    """Return the rental ports used by miner-side Internet reachability preflight."""
    start = int(start)
    end = int(end)
    if start < 1 or end > 65535 or end < start:
        raise RuntimeError(
            f"Invalid rental port range {start}-{end}; expected 1 <= start <= end <= 65535"
        )
    if scope == "full":
        return list(range(start, end + 1))

    count = end - start + 1
    samples = max(1, min(int(samples), count))
    if samples == 1:
        return [start]

    ports = {
        start + round(i * (count - 1) / (samples - 1))
        for i in range(samples)
    }
    return sorted(ports)


class _TemporaryPortListener:
    """Small TCP listener used only for startup Internet reachability checks."""

    def __init__(self, port: int):
        self.port = int(port)
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.listen(32)
            sock.settimeout(0.5)
        except OSError:
            try:
                sock.close()  # type: ignore[name-defined]
            except Exception:
                pass
            return False

        self._sock = sock
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return True

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Length: 2\r\n"
                        b"Connection: close\r\n\r\n"
                        b"ok"
                    )
                except OSError:
                    pass

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _portchecker_results(host: str, ports: list[int], timeout_s: float) -> dict[int, bool] | None:
    """Query portchecker.io. Returns None if the service did not answer cleanly."""
    import httpx

    try:
        resp = httpx.get(
            "https://portchecker.io/api/v1/query",
            params={"host": host, "ports": ",".join(str(p) for p in ports)},
            headers={"User-Agent": "nodexo-miner/0.1"},
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    out: dict[int, bool] = {}
    for item in data.get("ports") or []:
        try:
            port = int(item.get("port"))
        except (TypeError, ValueError):
            continue
        status = str(item.get("status") or "").lower()
        if status in {"open", "closed"}:
            out[port] = status == "open"
    return out or None


def _yougetsignal_result(host: str, port: int, timeout_s: float) -> bool | None:
    """Fallback single-port check. Returns None if the service did not answer."""
    import httpx

    try:
        resp = httpx.post(
            "https://ports.yougetsignal.com/check-port.php",
            data={"remoteAddress": host, "portNumber": str(port)},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            return None
        body = resp.text.lower()
    except Exception:
        return None

    if "is open" in body:
        return True
    if "is closed" in body:
        return False
    return None


def _internet_port_results(
    host: str,
    ports: list[int],
    timeout_s: float,
) -> tuple[dict[int, bool], list[str]]:
    """Return external TCP reachability results and the services that answered."""
    results: dict[int, bool] = {}
    services: list[str] = []

    bulk = _portchecker_results(host, ports, timeout_s)
    if bulk is not None:
        services.append("portchecker.io")
        results.update({p: bulk[p] for p in ports if p in bulk})

    # Avoid abusing fallback checkers for large full-range preflights. Full
    # mode is intended to use a bulk-capable service; sample mode gets a
    # second service for operator convenience.
    missing = [p for p in ports if p not in results]
    if missing and len(ports) <= 10:
        used = False
        for port in missing:
            res = _yougetsignal_result(host, port, timeout_s)
            if res is None:
                continue
            used = True
            results[port] = res
        if used:
            services.append("yougetsignal")

    return results, services


def _preflight_rental_port_range(
    endpoint: str,
    start: int,
    end: int,
    *,
    mode: str = "strict",
    scope: str = "sample",
    samples: int = 3,
    timeout_s: float = 8.0,
) -> None:
    """Check configured rental SSH ports from an Internet vantage point.

    The miner does this before registering/renewing the executor, so a bad
    provider firewall fails fast instead of advertising broken capacity.
    """
    if mode == "off":
        bt.logging.info("Rental port preflight disabled")
        return

    parsed = urlparse(endpoint)
    host = parsed.hostname or ""
    if not host:
        raise RuntimeError(f"Cannot parse host from miner endpoint {endpoint!r}")

    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_unspecified:
            if mode == "strict":
                raise RuntimeError(
                    "Rental port preflight requires a public endpoint host. "
                    "Use --rental-port-preflight off for local development."
                )
            bt.logging.warning("Skipping rental port preflight for local endpoint")
            return
    except ValueError:
        if host == "localhost" or host.endswith(".localhost"):
            if mode == "strict":
                raise RuntimeError(
                    "Rental port preflight requires a public endpoint host. "
                    "Use --rental-port-preflight off for local development."
                )
            bt.logging.warning("Skipping rental port preflight for local endpoint")
            return

    ports = _rental_ports_to_check(start, end, scope=scope, samples=samples)
    bt.logging.info(
        f"Rental port preflight: checking {len(ports)} port(s) on {host} "
        f"from the Internet ({ports[0]}-{ports[-1]}, scope={scope})"
    )

    listeners: list[_TemporaryPortListener] = []
    bound_ports: list[int] = []
    occupied_ports: list[int] = []
    try:
        for port in ports:
            listener = _TemporaryPortListener(port)
            if listener.start():
                listeners.append(listener)
                bound_ports.append(port)
            else:
                # A running rental/Docker mapping can already hold the port.
                # External reachability is still the truth, so keep checking it.
                occupied_ports.append(port)

        if bound_ports:
            time.sleep(0.5)
        if occupied_ports:
            bt.logging.info(
                "Rental port preflight: port(s) already bound locally; "
                f"checking only bindable port(s): {bound_ports or 'none'}"
            )

        if not bound_ports:
            bt.logging.warning(
                "Rental port preflight skipped external checks because all "
                f"sampled port(s) are already bound locally: {occupied_ports}"
            )
            return

        results, services = _internet_port_results(host, bound_ports, timeout_s)
    finally:
        for listener in listeners:
            listener.close()

    unknown = [p for p in bound_ports if p not in results]
    closed = [p for p in bound_ports if results.get(p) is False]
    if closed:
        service_text = ", ".join(services) if services else "none"
        message = (
            "Rental port preflight failed: "
            f"closed={closed} on host {host}. Services={service_text}. "
            "Open the configured rental range in the provider firewall/security "
            "group, or adjust --rental-port-start/--rental-port-end."
        )
        if mode == "strict":
            raise RuntimeError(message)
        bt.logging.warning(message)
        return

    if unknown:
        service_text = ", ".join(services) if services else "none"
        bt.logging.warning(
            "Rental port preflight inconclusive: "
            f"unchecked={unknown} on host {host}. Services={service_text}. "
            "Continuing because the external checker did not return a closed "
            "result; strict mode still fails explicitly closed ports."
        )
        return

    bt.logging.success(
        f"Rental port preflight passed: {host} reachable on {len(ports)} "
        f"tested rental port(s) via {', '.join(services)}"
    )

# True iff the sidecar monitor container is currently running. Flipped
# by _monitor_supervisor; gating reads by everyone that needs to know:
# - POST /containers refuses to provision new rentals when False
# - proof production does not read this flag
# Without this rental gate, a miner whose monitor died would still serve
# rentals the validator can't attest.
_monitor_alive: bool = False
_monitor_nvml_failures: dict[str, int] = {}


def _monitor_seed_for_executor(executor_id: str, hotkey_seed: bytes) -> bytes:
    return hashlib.sha256(
        b"nodexo-monitor-sidecar-v1|" + hotkey_seed + b"|" + executor_id.encode()
    ).digest()


def _monitor_owner_env(executor_id: str, hotkey_seed: bytes) -> tuple[str, dict[str, str]]:
    from substrateinterface import Keypair
    from common.crypto import monitor_binding_body, sign_payload

    monitor_seed = _monitor_seed_for_executor(executor_id, hotkey_seed)
    monitor_hotkey = Keypair.create_from_seed(monitor_seed.hex()).ss58_address
    binding_body = monitor_binding_body(executor_id, monitor_hotkey)
    signed = sign_payload(binding_body, hotkey_seed)
    return monitor_hotkey, {
        "NODEXO_MONITOR_SEED_HEX": monitor_seed.hex(),
        "NODEXO_MONITOR_OWNER_SIGNATURE": signed["X-Nodexo-Signature"],
        "NODEXO_MONITOR_OWNER_HOTKEY": signed["X-Nodexo-Hotkey"],
        "NODEXO_MONITOR_OWNER_TIMESTAMP": signed["X-Nodexo-Timestamp"],
        "NODEXO_MONITOR_OWNER_NONCE": signed["X-Nodexo-Nonce"],
    }


def _spawn_monitor_container(executor_id: str, validator_urls: list[str],
                              subtensor_network: str, netuid: int,
                              subtensor_endpoint: str,
                              image: str, hotkey_seed: bytes) -> bool:
    """Spawn (or noop if already running) the monitor container.

    Returns True if the container is running at exit. Idempotent: if
    a container with the expected name is already running, returns
    True without doing anything; otherwise it cleans up any stopped
    container with that name and starts a fresh one.

    Uses sysbox-runc when present (rootless, hardened) and falls back
    to default runtime otherwise (loud warning — operator should
    install sysbox in production).
    """
    import subprocess as _sp
    name = f"nodexo-monitor-{executor_id[:8]}"
    monitor_hotkey, owner_env = _monitor_owner_env(executor_id, hotkey_seed)

    # Rewrite 127.0.0.1 / localhost references in validator URLs to
    # host.docker.internal — inside the container's network namespace,
    # 127.0.0.1 is the CONTAINER's loopback, not the host's, so a
    # validator running on the same box via a reverse SSH tunnel (the
    # testnet topology) isn't reachable from 127.0.0.1 inside docker.
    # `--add-host=host.docker.internal:host-gateway` (docker 20.10+)
    # plus URL rewriting gets us to the host's services.
    rewritten_urls = []
    for u in validator_urls:
        u2 = u.replace("://127.0.0.1", "://host.docker.internal")
        u2 = u2.replace("://localhost", "://host.docker.internal")
        rewritten_urls.append(u2)
    # Net mode: --network=host when validators are loopback (testnet
    # reverse-SSH-tunnel topology — sshd's -R binds to host 127.0.0.1
    # which a bridged container can't reach). Drop sysbox-runc in that
    # mode; the two are mutually exclusive (sysbox mandates its own
    # net namespace). Production miners use reachable validator
    # endpoints + bridge + sysbox.
    use_host_net = any(
        ("://127.0.0.1" in u) or ("://localhost" in u)
        for u in validator_urls
    )
    monitor_validator_urls = ",".join(validator_urls if use_host_net else rewritten_urls)

    def _monitor_nvml_ready(timeout_s: float = 10.0) -> tuple[bool, str]:
        """Return True only when the sidecar can actually initialize NVML.

        Docker's `State.Running=true` is not enough: we have repeatedly seen
        containers keep running while NVML returns "Unknown Error", which makes
        the validator see valid proofs but no monitor-witnessed proof process.
        """
        deadline = time.time() + max(0.5, float(timeout_s))
        last = ""
        probe = (
            "import pynvml; "
            "pynvml.nvmlInit(); "
            "count=pynvml.nvmlDeviceGetCount(); "
            "assert count > 0, 'no NVIDIA devices visible'; "
            "pynvml.nvmlShutdown(); "
            "print(count)"
        )
        while True:
            try:
                r = _sp.run(
                    ["docker", "exec", name, "python3", "-c", probe],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    return True, ""
                last = (r.stderr or r.stdout or "").strip()
            except _sp.TimeoutExpired:
                last = "NVML probe timed out"
            except Exception as e:
                last = f"NVML probe failed: {e}"
            if time.time() >= deadline:
                return False, last[:300]
            time.sleep(1.0)

    try:
        # Already running?
        r = _sp.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and "true" in r.stdout.strip().lower():
            reasons: list[str] = []
            env = _sp.run(
                ["docker", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", name],
                capture_output=True, text=True, timeout=10,
            )
            env_lines = env.stdout.splitlines() if env.returncode == 0 else []
            if f"NODEXO_VALIDATOR_URLS={monitor_validator_urls}" not in env_lines:
                reasons.append("validator URLs changed")
            if "NVIDIA_DRIVER_CAPABILITIES=compute,utility" not in env_lines:
                reasons.append("NVIDIA capabilities changed")
            if f"NODEXO_MONITOR_SEED_HEX={owner_env['NODEXO_MONITOR_SEED_HEX']}" not in env_lines:
                reasons.append("monitor seed changed")
            if f"NODEXO_MONITOR_OWNER_HOTKEY={owner_env['NODEXO_MONITOR_OWNER_HOTKEY']}" not in env_lines:
                reasons.append("monitor owner changed")

            pid_mode = _sp.run(
                ["docker", "inspect", "-f", "{{.HostConfig.PidMode}}", name],
                capture_output=True, text=True, timeout=10,
            )
            if pid_mode.returncode != 0 or pid_mode.stdout.strip() != "host":
                reasons.append("PID mode changed")

            nvml_ok, nvml_detail = _monitor_nvml_ready(timeout_s=5.0)
            if not nvml_ok:
                max_failures = max(
                    1,
                    int(os.environ.get("NODEXO_MONITOR_NVML_RECREATE_FAILURES", "1") or "1"),
                )
                restart_on_nvml_loss = (
                    os.environ.get("NODEXO_MONITOR_RESTART_ON_NVML_LOSS", "1") == "1"
                )
                failures = _monitor_nvml_failures.get(name, 0) + 1
                _monitor_nvml_failures[name] = failures
                if restart_on_nvml_loss and failures >= max_failures:
                    reasons.append("monitor NVML unavailable")
                else:
                    restart_note = (
                        "restart disabled"
                        if not restart_on_nvml_loss
                        else f"below restart threshold {failures}/{max_failures}"
                    )
                    bt.logging.warning(
                        "Monitor NVML readiness probe failed; "
                        f"treating sidecar as down ({restart_note}): "
                        f"{nvml_detail}"
                    )
                    return False
            else:
                _monitor_nvml_failures[name] = 0

            net_mode = _sp.run(
                ["docker", "inspect", "-f", "{{.HostConfig.NetworkMode}}", name],
                capture_output=True, text=True, timeout=10,
            )
            if net_mode.returncode == 0:
                current_net = net_mode.stdout.strip()
                if use_host_net and current_net != "host":
                    reasons.append("network mode changed")
                elif not use_host_net and current_net == "host":
                    reasons.append("network mode changed")

            container_image = _sp.run(
                ["docker", "inspect", "-f", "{{.Image}}", name],
                capture_output=True, text=True, timeout=10,
            )
            desired_image = _sp.run(
                ["docker", "image", "inspect", "-f", "{{.Id}}", image],
                capture_output=True, text=True, timeout=10,
            )
            if (
                container_image.returncode == 0
                and desired_image.returncode == 0
                and container_image.stdout.strip()
                and desired_image.stdout.strip()
                and container_image.stdout.strip() != desired_image.stdout.strip()
            ):
                reasons.append("monitor image changed")

            if not reasons:
                return True
            bt.logging.info(
                "Recreating monitor container: " + ", ".join(sorted(set(reasons)))
            )
            _sp.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
        # Clean stale entry (Created/Exited).
        _sp.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
    except Exception as e:
        bt.logging.warning(f"_spawn_monitor_container inspect failed: {e}")

    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "--restart", "unless-stopped",
        "--gpus", "all",
        # CRITICAL: NVIDIA_DRIVER_CAPABILITIES must include `compute,utility`
        # for NVML to initialize inside the container. The default
        # `--gpus all` only requests the basic "gpu" capability which
        # mounts the device nodes but leaves NVML returning
        # "Failed to initialize NVML: Unknown Error". Without NVML the
        # monitor classifier can't list GPU processes → every cycle
        # reports `expected_proof_seen=False, extra_proofs_seen=0` and loses
        # useful monitor telemetry. The validator treats passive monitor
        # proof-witness misses as audit-only, but the sidecar should still
        # self-heal so operators get accurate live process data.
        # `compute` lets the container call CUDA driver APIs, `utility`
        # grants nvidia-smi / NVML query access.
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        # PID host share — required so /proc/<pid>/cmdline in the
        # container resolves the host PIDs that NVML returns from
        # nvmlDeviceGetComputeRunningProcesses. Without this the
        # behavioral signal (matching gpu_worker by command name)
        # is silently disabled — comm alone is only 15 chars and
        # reads "python" for any script.
        "--pid", "host",
        # Persistent volume for the monitor's per-monitor keypair so
        # rotating monitors don't churn through TOFU bindings on the
        # validator and trigger monitor_pubkey_rotation flags.
        "-v", f"nodexo_monitor_data_{executor_id[:8]}:/data",
        "-e", f"NODEXO_EXECUTOR_ID={executor_id}",
    ]
    for key, value in owner_env.items():
        cmd += ["-e", f"{key}={value}"]
    if use_host_net:
        cmd += ["--network", "host"]
        # In host-net mode 127.0.0.1 inside the container IS the host's
        # loopback, so use the original URLs unchanged.
        cmd += ["-e", f"NODEXO_VALIDATOR_URLS={','.join(validator_urls)}"]
    else:
        cmd += ["--add-host", "host.docker.internal:host-gateway"]
        cmd += ["-e", f"NODEXO_VALIDATOR_URLS={monitor_validator_urls}"]
    if subtensor_network:
        cmd += ["-e", f"NODEXO_SUBTENSOR_NETWORK={subtensor_network}"]
    if subtensor_endpoint:
        cmd += ["-e", f"NODEXO_SUBTENSOR_ENDPOINT={subtensor_endpoint}"]
    if netuid:
        cmd += ["-e", f"NODEXO_NETUID={netuid}"]
    cmd += [
        "-e",
        "NODEXO_PROOF_MAX_JITTER_SECONDS="
        f"{os.environ.get('NODEXO_PROOF_MAX_JITTER_SECONDS', str(DEFAULT_JITTER_SECONDS))}",
    ]
    cmd += [
        "-e",
        "PROOF_BEACON_MAX_OFFSET_BLOCKS="
        f"{os.environ.get('PROOF_BEACON_MAX_OFFSET_BLOCKS', str(DEFAULT_BEACON_MAX_OFFSET_BLOCKS))}",
    ]
    cmd += [
        "-e",
        "NODEXO_MONITOR_NVML_EXIT_FAILURES="
        f"{os.environ.get('NODEXO_MONITOR_NVML_EXIT_FAILURES', '15')}",
    ]
    if use_host_net:
        bt.logging.warning(
            "Monitor on --network=host (loopback validator URL detected). "
            "Production must use a public "
            "validator endpoint so host networking is not needed."
        )
    else:
        bt.logging.info(
            "Monitor container uses Docker's default runtime because host PID "
            "namespace access is required for NVML process attribution."
        )
    cmd.append(image)

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            bt.logging.error(
                f"docker run nodexo-monitor failed: rc={r.returncode} "
                f"stderr={r.stderr.strip()[:300]}"
            )
            return False
        nvml_ok, nvml_detail = _monitor_nvml_ready(timeout_s=15.0)
        if not nvml_ok:
            bt.logging.error(
                "docker run nodexo-monitor produced an NVML-blind sidecar; "
                f"stderr={nvml_detail}"
            )
            _sp.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
            return False
        _monitor_nvml_failures[name] = 0
        return True
    except Exception as e:
        bt.logging.error(f"_spawn_monitor_container exception: {e}")
        return False


async def _monitor_supervisor(executor_id: str, validator_urls: list[str],
                               subtensor_network: str, netuid: int,
                               subtensor_endpoint: str,
                               image: str, require: bool, hotkey_seed: bytes,
                               check_interval_s: float = 30.0) -> None:
    """Watchdog for the signed monitor sidecar.

    The monitor gates rental admission because rentals need an independent
    witness. Proof production is intentionally independent from monitor health:
    an honest miner should keep submitting proofs while the supervisor repairs
    a broken sidecar.
    """
    global _monitor_alive
    misses = 0
    while True:
        try:
            await asyncio.sleep(check_interval_s)
            ok = _spawn_monitor_container(
                executor_id, validator_urls, subtensor_network, netuid,
                subtensor_endpoint, image, hotkey_seed,
            )
            if ok:
                if not _monitor_alive:
                    bt.logging.info("Monitor recovered; resuming normal operation")
                _monitor_alive = True
                misses = 0
            else:
                misses += 1
                bt.logging.warning(
                    f"Monitor not running (miss #{misses}); will retry"
                )
                if misses >= 3:
                    _monitor_alive = False
                    if require:
                        bt.logging.error(
                            "Monitor down for ≥3 consecutive checks "
                            "(~90s). Refusing new rentals until the monitor "
                            "recovers; proof production continues."
                        )
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"_monitor_supervisor: {e}")


async def _require_validator(request: Request):
    """FastAPI dependency: validate SR25519 sig against the validator allowlist.

    Indirects through the module-level `_validator_auth` so lifespan can
    construct the auth after the chain client is ready, and routes don't
    need to know about lifespan order.
    """
    if _validator_auth is None:
        raise HTTPException(503, "Miner auth not initialized yet")
    await _validator_auth(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Miner startup/shutdown lifecycle."""
    global proof_service, broadcast_service, docker_service
    global hardware_info, miner_id, _rpc, _compute_registry
    global _evm_address, _evm_private_key

    from neurons.miner.services.hardware_service import (
        detect_gpus, detect_system, get_or_create_executor_id,
    )
    from neurons.miner.services.proof_service import ProofService
    from neurons.miner.services.broadcast_service import BroadcastService
    from neurons.miner.services.docker_service import DockerService

    # ── Config from env / CLI args stored on app ───────────────
    wallet_name = app.state.wallet_name
    hotkey_name = app.state.hotkey_name
    netuid = app.state.netuid
    chain_config_path = app.state.chain_config
    subtensor_network = app.state.subtensor_network
    subtensor_endpoint = getattr(app.state, "subtensor_endpoint", "")

    subnet_config_task = None
    try:
        from common.subnet_config_client import (
            load_bundled_subnet_config,
            load_cached_subnet_config,
            run_subnet_config_refresh_loop,
            subnet_config_url,
        )
        if not load_cached_subnet_config():
            load_bundled_subnet_config()
        if subnet_config_url():
            subnet_config_task = asyncio.create_task(run_subnet_config_refresh_loop())
    except Exception as e:
        bt.logging.warning(f"Subnet config refresh failed to start: {e}")

    # ── Hardware detection ─────────────────────────────────────
    gpus = detect_gpus()
    sys_info = detect_system()
    miner_id = get_or_create_executor_id(gpus)

    if not gpus:
        bt.logging.error("No GPUs detected! Miner cannot start.")
        sys.exit(1)

    hardware_info = {
        "miner_id": miner_id,
        "gpus": [{"name": g.name, "uuid": g.uuid, "vram_mb": g.vram_mb} for g in gpus],
        "system": {"cpu": sys_info.cpu_model, "cores": sys_info.cpu_cores, "ram_gb": sys_info.ram_gb},
    }
    bt.logging.info(
        f"Miner {miner_id[:16]}: {len(gpus)} GPU(s) — {gpus[0].name} ({gpus[0].vram_mb} MB VRAM)"
    )

    # ── Docker service ─────────────────────────────────────────
    # Pass the rental-port range through. Container SSH ports get bound here
    # and must be opened in the provider firewall once. Without a range,
    # Docker auto-assigns a random high port (only reachable for dev via
    # ProxyJump through the host — never for a real renter).
    _port_range = None
    _ps = getattr(app.state, "rental_port_start", 0)
    _pe = getattr(app.state, "rental_port_end", 0)
    if _ps and _pe and _pe >= _ps:
        _port_range = (_ps, _pe)
        bt.logging.info(f"Rental port range: {_ps}-{_pe} ({_pe-_ps+1} concurrent rentals max)")
    else:
        bt.logging.warning(
            "No --rental-port-start/--rental-port-end set; containers will get "
            "random Docker ports that real renters can't reach without host SSH."
        )
    docker_service = DockerService(port_range=_port_range)

    # ── Mock mode vs production mode ─────────────────────────
    mock_mode = app.state.mock

    _validator_registry = None
    _compute_registry = None
    hotkey_seed = None

    if mock_mode:
        # ── MOCK: skip all chain interaction ───────────────────
        bt.logging.info("MOCK MODE — skipping wallet, subtensor, EVM registration")
        bt.logging.info("Proofs use fake block hashes, broadcast to --validator-url if set")
    elif getattr(app.state, '_chain_ready', False):
        # ── PRODUCTION (pre-initialized in main() before uvicorn) ──
        _rpc = app.state._pre_rpc
        _compute_registry = getattr(app.state, '_pre_compute_registry', None)
        _validator_registry = getattr(app.state, '_pre_validator_registry', None)
        hotkey_seed = getattr(app.state, '_pre_hotkey_seed', None)
        _evm_address = getattr(app.state, '_pre_evm_address', "") or ""
        _evm_private_key = getattr(app.state, '_pre_evm_private_key', "") or ""
        bt.logging.info("Chain state loaded (pre-initialized)")
    else:
        bt.logging.error("Production mode requires pre-init. Use main() entry point.")
        sys.exit(1)

    # ── Validator discovery ────────────────────────────────────
    _mock_validator_url = app.state.validator_url  # For mock mode

    def discover_validators(*, raise_on_registry_error: bool = False):
        validators = []
        registry_failed = False
        # Mock mode: use --validator-url if provided
        if mock_mode and _mock_validator_url:
            from common.types import ValidatorEndpoint
            validators = [ValidatorEndpoint(
                address="mock_validator",
                proxy_endpoint=_mock_validator_url,
                uid=0, is_active=True,
            )]
        native_validators = []
        mg = None
        if not mock_mode and _rpc:
            try:
                mg = _rpc.get_metagraph(False)
                if validators:
                    before = len(validators)
                    validators = _validator_registry_endpoints_with_current_permit(
                        validators, mg,
                    )
                    dropped = before - len(validators)
                    if dropped:
                        bt.logging.warning(
                            f"Dropped {dropped} ValidatorRegistry endpoint(s) "
                            "without current validator permit"
                        )
                native_validators = _validator_axon_endpoints_from_metagraph(
                    mg,
                    exclude_uid=getattr(app.state, "_pre_uid", None),
                )
            except Exception as e:
                bt.logging.debug(f"validator permit discovery failed: {e}")
        if _validator_registry:
            try:
                validators = _validator_registry.get_active_validators(
                    raise_on_error=True,
                )
            except Exception as e:
                if raise_on_registry_error and not native_validators:
                    raise
                registry_failed = True
                bt.logging.debug(f"ValidatorRegistry discovery failed: {e}")
                validators = []
            if mg is not None and validators:
                before = len(validators)
                validators = _validator_registry_endpoints_with_current_permit(
                    validators, mg,
                )
                dropped = before - len(validators)
                if dropped:
                    bt.logging.warning(
                        f"Dropped {dropped} ValidatorRegistry endpoint(s) "
                        "without current validator permit"
                    )
            elif validators and not mock_mode:
                bt.logging.warning(
                    "Validator permit set unavailable; ignoring fresh "
                    "ValidatorRegistry endpoints for this refresh"
                )
                validators = []
        filtered = _filter_validator_api_endpoints(
                _merge_validator_endpoints(validators, native_validators)
        )
        return ValidatorDiscoveryResult(
            filtered,
            registry_failed=registry_failed,
            partial=bool(getattr(filtered, "partial", False)),
            inconclusive_native_count=int(
                getattr(filtered, "inconclusive_native_count", 0) or 0
            ),
        )

    # ── Signing ────────────────────────────────────────────────
    _sign_fn = None
    if hotkey_seed:
        def sign_payload_fn(payload_json: str) -> dict:
            from common.crypto import sign_payload
            return sign_payload(payload_json.encode(), hotkey_seed)
        _sign_fn = sign_payload_fn

    broadcast_service = BroadcastService(
        discover_validators=discover_validators,
        sign_payload=_sign_fn,
        endpoint=getattr(app.state, "endpoint", ""),
    )
    startup_validators = list(
        getattr(app.state, "_pre_startup_validators", None) or []
    )
    if startup_validators:
        broadcast_service.set_validator_cache(
            startup_validators,
            source="pre-startup discovery",
            save=True,
        )
    elif not mock_mode:
        await broadcast_service.refresh_until_ready(
            discover_validators=discover_validators,
        )

    # ── RPC callbacks for proof service ────────────────────────
    _mock_block = [0]  # Mutable counter for mock mode

    async def get_block_number():
        if mock_mode:
            import time as _t
            return int(_t.time() / 12)
        return await asyncio.to_thread(_rpc.get_current_block)

    async def get_block_hash(block_number):
        if mock_mode:
            import hashlib as _h
            return _h.sha256(str(block_number).encode()).digest()
        return await asyncio.to_thread(_rpc.get_block_hash, block_number)

    async def get_chain_context():
        validators = broadcast_service.current_validators() if broadcast_service else []
        if not validators:
            return None
        try:
            timeout_total = float(os.environ.get("NODEXO_CHAIN_CONTEXT_TIMEOUT_SECONDS", "2.5"))
        except Exception:
            timeout_total = 2.5
        try:
            connect_timeout = float(os.environ.get("NODEXO_CHAIN_CONTEXT_CONNECT_TIMEOUT_SECONDS", "0.75"))
        except Exception:
            connect_timeout = 0.75
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=timeout_total, connect=connect_timeout)

        async def fetch_one(session, endpoint: str):
            url = f"{endpoint}/chain/context?epoch_interval={int(os.environ.get('PROOF_EPOCH_BLOCKS', '15'))}"
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, dict) and data.get("beacon_hex"):
                            return data
            except Exception:
                return None
            return None

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for validator in validators:
                endpoint = str(getattr(validator, "proxy_endpoint", "") or "").rstrip("/")
                if not endpoint:
                    continue
                tasks.append(asyncio.create_task(fetch_one(session, endpoint)))
            if not tasks:
                return None
            try:
                for task in asyncio.as_completed(tasks):
                    data = await task
                    if data is not None:
                        for pending in tasks:
                            if pending is not task and not pending.done():
                                pending.cancel()
                        return data
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        return None

    rental_state_cache = {
        "is_rented": False,
        "updated_at": 0.0,
    }

    def is_rented():
        try:
            if docker_service and docker_service.has_active_containers():
                return True
        except Exception as e:
            bt.logging.debug(f"local rental container state unavailable: {e}")
        try:
            ttl_s = max(1.0, float(os.environ.get("NODEXO_RENTAL_STATE_CACHE_TTL_SECONDS", "45")))
        except Exception:
            ttl_s = 45.0
        if time.time() - float(rental_state_cache.get("updated_at") or 0.0) <= ttl_s:
            return bool(rental_state_cache.get("is_rented"))
        return False

    async def _rental_state_cache_loop():
        if _compute_registry is None:
            try:
                from common.subnet_runtime_config import get_subnet_runtime_config
                if get_subnet_runtime_config().allocation.read_mode == "chain":
                    return
            except Exception:
                return
        try:
            interval_s = max(1.0, float(os.environ.get("NODEXO_RENTAL_STATE_REFRESH_SECONDS", "15")))
        except Exception:
            interval_s = 15.0
        executor_id_bytes = bytes.fromhex(miner_id)
        while True:
            try:
                rented_now = None
                try:
                    from common.subnet_runtime_config import get_subnet_runtime_config
                    if get_subnet_runtime_config().allocation.read_mode != "chain":
                        from common.allocation_client import fetch_executor_status
                        status = await fetch_executor_status(miner_id, hotkey_seed)
                        rented_now = bool((status or {}).get("busy"))
                except Exception as e:
                    bt.logging.debug(f"allocation rental state refresh failed: {e}")
                if rented_now is None:
                    if _compute_registry is None:
                        await asyncio.sleep(interval_s)
                        continue
                    rented_now = await asyncio.to_thread(
                        _compute_registry.is_rented,
                        executor_id_bytes,
                    )
                rental_state_cache["is_rented"] = bool(rented_now)
                rental_state_cache["updated_at"] = time.time()
            except Exception as e:
                bt.logging.debug(f"rental state cache refresh failed: {e}")
            await asyncio.sleep(interval_s)


    def _should_run_proof() -> bool:
        """Keep proof production independent from monitor health.

        The monitor is the independent witness used for rental eligibility and
        sybil checks. If the sidecar breaks, the miner should keep submitting
        proofs so an honest operator does not lose proof recency while the
        supervisor repairs the monitor container.
        """
        return True

    from common.config import canonical_gpu_model_name
    proof_gpu_model = canonical_gpu_model_name(gpus[0].name if gpus else "unknown")

    proof_service = ProofService(
        executor_id=miner_id,
        gpu_model=proof_gpu_model,
        vram_mb=gpus[0].vram_mb,
        gpu_count=len(gpus),
        epoch_interval=int(os.environ.get("PROOF_EPOCH_BLOCKS", "15")),
        max_jitter=int(os.environ.get(
            "NODEXO_PROOF_MAX_JITTER_SECONDS", str(DEFAULT_JITTER_SECONDS),
        )),
        max_beacon_offset_blocks=int(os.environ.get(
            "PROOF_BEACON_MAX_OFFSET_BLOCKS", str(DEFAULT_BEACON_MAX_OFFSET_BLOCKS),
        )),
        get_block_hash=get_block_hash,
        get_block_number=get_block_number,
        get_chain_context=get_chain_context,
        is_rented=is_rented,
        should_run=_should_run_proof,
        on_receipt=broadcast_service.broadcast_receipt,
        on_recipe=broadcast_service.broadcast_recipe,
    )

    # ── Validator auth (gates /containers + sensitive endpoints) ──
    # Allowlist sources: env NODEXO_ALLOWED_VALIDATOR_HOTKEYS (CSV SS58s)
    # and on-chain ValidatorRegistry via metagraph hotkey lookup. Without
    # at least one entry, the gated endpoints 503 — the miner refuses to
    # serve container creation to an unauthenticated network.
    global _validator_auth
    _validator_auth = ValidatorAuth(
        validator_registry=_validator_registry, rpc=_rpc,
    )
    if not _validator_auth.allowed():
        bt.logging.warning(
            "No validator hotkeys in allowlist. Set NODEXO_ALLOWED_VALIDATOR_HOTKEYS "
            "or ensure ValidatorRegistry is reachable; /containers will 503."
        )
    else:
        bt.logging.info(
            f"Validator auth ready ({len(_validator_auth.allowed())} hotkey(s) allowed)"
        )

    # ── Monitor container — spawn now, supervise forever ──────
    # The monitor is the validator's signed witness of what's actually
    # happening on this GPU. Without it, the miner has no attestable
    # surface and can't be rentable. We require it to be running before
    # accepting rentals (POST /containers gates on _monitor_alive). Proof
    # production remains independent so monitor repair does not destroy proof
    # recency for an honest miner.
    #
    # Opt-out: NODEXO_REQUIRE_MONITOR=0 (dev/test only; production sites
    # MUST run with monitor required — this is the sybil defense path).
    global _monitor_alive
    _monitor_alive = False
    _validator_urls_for_monitor: list[str] = []
    try:
        for v in broadcast_service.current_validators() or []:
            if getattr(v, "proxy_endpoint", ""):
                _validator_urls_for_monitor.append(v.proxy_endpoint)
    except Exception:
        pass

    require_monitor = os.environ.get("NODEXO_REQUIRE_MONITOR", "1") == "1"
    monitor_image = os.environ.get("NODEXO_MONITOR_IMAGE", "nodexo-monitor:dev")
    monitor_task = None
    if mock_mode:
        bt.logging.warning(
            "MOCK MODE — skipping monitor container spawn. "
            "Monitor-based sybil defense is DISABLED for this run."
        )
    else:
        if not _validator_urls_for_monitor:
            msg = (
                "Monitor cannot start: no validator URLs discovered yet. "
                "The miner needs ValidatorRegistry to be populated."
            )
            if require_monitor:
                bt.logging.error(msg + " Refusing to start.")
                sys.exit(1)
            bt.logging.warning(msg + " Continuing (NODEXO_REQUIRE_MONITOR=0).")
        else:
            # First synchronous spawn — fail loud if it can't start so
            # the operator sees the problem at boot rather than at first
            # missing report 10 minutes later.
            ok = _spawn_monitor_container(
                executor_id=miner_id,
                validator_urls=_validator_urls_for_monitor,
                subtensor_network=subtensor_network,
                netuid=netuid,
                subtensor_endpoint=subtensor_endpoint,
                image=monitor_image,
                hotkey_seed=hotkey_seed,
            )
            if not ok:
                if require_monitor:
                    bt.logging.error(
                        f"Monitor container failed to start (image={monitor_image}). "
                        f"Build it locally with: "
                        f"  docker build -t nodexo-monitor:dev -f neurons/monitor/Dockerfile . "
                        f"Or set NODEXO_MONITOR_IMAGE to a pre-published digest. "
                        f"Refusing to start."
                    )
                    sys.exit(1)
                bt.logging.warning(
                    "Monitor spawn failed but NODEXO_REQUIRE_MONITOR=0; "
                    "continuing without sybil defense."
                )
            else:
                _monitor_alive = True
                bt.logging.info(
                    f"Monitor container running (image={monitor_image}, "
                    f"validators={len(_validator_urls_for_monitor)})"
                )
            # Supervise: respawn if it crashes, flip _monitor_alive
            # so other paths can refuse work when it's down.
            monitor_task = asyncio.create_task(_monitor_supervisor(
                executor_id=miner_id,
                validator_urls=_validator_urls_for_monitor,
                subtensor_network=subtensor_network,
                netuid=netuid,
                subtensor_endpoint=subtensor_endpoint,
                image=monitor_image,
                require=require_monitor,
                hotkey_seed=hotkey_seed,
            ))

    # ── Orphan reap (validator cross-check) ────────────────────
    # On every miner boot, ask the validator(s) "which rental
    # containers do you currently own on me?". Any local docker
    # container not in that union is a ghost (e.g., validator DB
    # was wiped, peer validator's rental ended while we were down)
    # and gets destroyed. Safe-by-design: if no validator is
    # reachable, do nothing — we'd rather leak a ghost than nuke
    # a real rental.
    if not mock_mode and _validator_urls_for_monitor:
        try:
            destroyed = docker_service.reap_orphans(
                executor_id=miner_id,
                validator_urls=_validator_urls_for_monitor,
                hotkey_seed=hotkey_seed,
            )
            if destroyed:
                bt.logging.info(f"reap_orphans: destroyed {destroyed} ghost container(s)")
        except Exception as e:
            bt.logging.warning(f"reap_orphans at startup: {e}")

    # Periodic reap task — runs every 5 min so a long-running miner
    # also reaps ghosts created mid-run by chain re-orgs / validator
    # crashes, not just at startup.
    async def _reap_loop():
        while True:
            try:
                await asyncio.sleep(300)
                if docker_service is None or not _validator_urls_for_monitor:
                    continue
                d = await asyncio.to_thread(
                    docker_service.reap_orphans,
                    miner_id, _validator_urls_for_monitor, hotkey_seed,
                )
                if d:
                    bt.logging.info(f"reap_orphans (periodic): destroyed {d}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.debug(f"reap_orphans loop: {e}")
    reap_task: Optional[asyncio.Task] = (
        asyncio.create_task(_reap_loop()) if not mock_mode else None
    )

    # ── Background tasks ───────────────────────────────────────
    proof_start_delay = 0.0
    if bool(getattr(app.state, "_pre_executor_fresh_registration", False)):
        try:
            proof_start_delay = max(
                0.0,
                float(os.environ.get("NODEXO_FRESH_REGISTRATION_PROOF_DELAY_S", "300")),
            )
        except Exception:
            proof_start_delay = 300.0
    if proof_start_delay > 0:
        bt.logging.info(
            "Fresh executor registration detected; delaying proof loop "
            f"{proof_start_delay:.0f}s so validators can index the chain event"
        )

    async def _run_proof_after_startup_delay():
        if proof_start_delay > 0:
            await asyncio.sleep(proof_start_delay)
        await proof_service.run()

    proof_task = asyncio.create_task(_run_proof_after_startup_delay())
    ttl_task = asyncio.create_task(_ttl_checker())
    heartbeat_task = asyncio.create_task(_heartbeat_loop(_compute_registry, miner_id))
    try:
        from common.subnet_runtime_config import get_subnet_runtime_config
        _allocation_state_enabled = (
            get_subnet_runtime_config().allocation.read_mode != "chain"
        )
    except Exception:
        _allocation_state_enabled = False
    rental_state_task: Optional[asyncio.Task] = (
        asyncio.create_task(_rental_state_cache_loop())
        if (_compute_registry is not None or _allocation_state_enabled) else None
    )
    vali_hb_task = asyncio.create_task(_validator_heartbeat_loop(interval=60))
    speedtest_task = asyncio.create_task(_network_speedtest_loop())
    _stats_uid = getattr(app.state, '_pre_uid', None)
    mg_log_task = asyncio.create_task(_metagraph_stats_loop(_rpc, _stats_uid, interval=60))
    auto_updater = None
    if getattr(app.state, "auto_update_enabled", False) and not mock_mode:
        from neurons.auto_update import AutoUpdater

        def _miner_busy() -> bool:
            try:
                has_rentals = bool(
                    docker_service and docker_service.has_active_containers()
                )
            except Exception:
                has_rentals = True
            return has_rentals or bool(
                proof_service and getattr(proof_service, "is_busy", lambda: False)()
            )

        auto_updater = AutoUpdater(
            role="miner",
            check_interval=int(getattr(app.state, "auto_update_interval", 1800)),
            restart_delay=int(getattr(app.state, "auto_update_restart_delay", 5)),
            busy_check=_miner_busy,
            jitter_seconds=int(getattr(app.state, "auto_update_jitter_seconds", 1800)),
            jitter_seed=(miner_id or "").encode("utf-8"),
        )
        auto_updater.start()

    bt.logging.info("Miner daemon started (proof loop + heartbeat + TTL checker)")
    yield

    # Shutdown
    if auto_updater is not None:
        auto_updater.stop()
    if subnet_config_task is not None:
        subnet_config_task.cancel()
    proof_service.stop()
    proof_task.cancel()
    ttl_task.cancel()
    heartbeat_task.cancel()
    if rental_state_task is not None:
        rental_state_task.cancel()
    vali_hb_task.cancel()
    speedtest_task.cancel()
    mg_log_task.cancel()
    if monitor_task is not None:
        monitor_task.cancel()
    if reap_task is not None:
        reap_task.cancel()
    await broadcast_service.close()
    bt.logging.info("Miner daemon stopped")


# ── Background tasks ───────────────────────────────────────────

def _ensure_executor_registry_state(
    registry,
    miner_id_hex: str,
    *,
    endpoint: str,
    gpu_model_hash: bytes,
    gpu_count: int,
    vram_mb: int,
    price_per_gpu_hour: int,
    force_renew: bool = False,
) -> str | None:
    """Keep this live miner's ComputeRegistry row active and current."""
    executor_id = bytes.fromhex(miner_id_hex)
    spec = registry.get_executor_info(executor_id)
    if spec is not None and not getattr(spec, "is_active", False):
        if getattr(spec, "is_rented", False):
            raise RuntimeError("executor is inactive while rented; refusing self-deregister")
        dereg_tx = registry.deregister_executor(executor_id)
        reg_tx = registry.register_executor(
            executor_id=executor_id,
            endpoint=endpoint,
            gpu_model_hash=gpu_model_hash,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            price_per_gpu_hour=price_per_gpu_hour,
        )
        return (
            "Executor deregistered/reactivated on ComputeRegistry: "
            f"deregister_tx={dereg_tx} register_tx={reg_tx}"
        )

    if spec is None:
        tx = registry.register_executor(
            executor_id=executor_id,
            endpoint=endpoint,
            gpu_model_hash=gpu_model_hash,
            gpu_count=gpu_count,
            vram_mb=vram_mb,
            price_per_gpu_hour=price_per_gpu_hour,
        )
        return f"Executor registered/reactivated on ComputeRegistry: tx={tx}"

    messages: list[str] = []
    if force_renew:
        tx = registry.renew_executor(executor_id)
        messages.append(f"Lease renewed: tx={tx}")

    if getattr(spec, "endpoint", "") != endpoint:
        tx = registry.update_endpoint(executor_id, endpoint)
        messages.append(f"Endpoint updated on ComputeRegistry: tx={tx}")

    return "; ".join(messages) if messages else None


async def _heartbeat_loop(registry, miner_id_hex: str):
    """Renew lease and recover if the on-chain executor was reported offline."""
    import time as _time

    reconcile_interval = float(os.environ.get("EXECUTOR_RECONCILE_INTERVAL_S", "300"))
    renew_interval = float(os.environ.get("EXECUTOR_LEASE_RENEW_INTERVAL_S", "43200"))
    last_renew_at = 0.0

    while True:
        try:
            await asyncio.sleep(reconcile_interval)
            if registry:
                now = _time.time()
                force_renew = now - last_renew_at >= renew_interval
                msg = await asyncio.to_thread(
                    _ensure_executor_registry_state,
                    registry,
                    miner_id_hex,
                    endpoint=getattr(app.state, "_pre_executor_endpoint", ""),
                    gpu_model_hash=getattr(app.state, "_pre_executor_gpu_model_hash", b""),
                    gpu_count=int(getattr(app.state, "_pre_executor_gpu_count", 1) or 1),
                    vram_mb=int(getattr(app.state, "_pre_executor_vram_mb", 0) or 0),
                    price_per_gpu_hour=int(getattr(app.state, "_pre_executor_price_per_gpu_hour", 0) or 0),
                    force_renew=force_renew,
                )
                if force_renew:
                    last_renew_at = now
                if msg:
                    bt.logging.info(msg)
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"Lease reconcile failed: {e}")
            await asyncio.sleep(60)


async def _validator_heartbeat_loop(interval: int = 60):
    """Push hardware heartbeat to all known validators every ~60s.

    Sends hw_static on startup and periodically, plus hw_live every interval,
    so validators can track liveness, resources, and image cache availability.
    Reference: see protocol design notes
    """
    from neurons.miner.services.hardware_service import build_hw_static, build_hw_live

    # Build static hardware info once, then refresh it periodically. The
    # payload includes bounded Docker-image cache status for curated renter
    # presets; that can change after setup or an operator pre-pulls images.
    hw_static = build_hw_static()
    hb_count = 0

    while True:
        try:
            await asyncio.sleep(interval)

            if not broadcast_service:
                continue

            hw_live = build_hw_live()

            send_static = hb_count < 3 or hb_count % 10 == 0
            if send_static and hb_count > 0:
                try:
                    hw_static = build_hw_static()
                except Exception as e:
                    bt.logging.warning(f"Failed to refresh static heartbeat payload: {e}")

            payload = {
                "executor_id": miner_id,
                "hw_static": hw_static if send_static else None,
                "hw_live": hw_live,
                "endpoint": getattr(broadcast_service, "endpoint", "")
                or getattr(app.state, "endpoint", ""),
            }
            from neurons.version import miner_version, miner_version_str
            payload["software_version"] = miner_version_str
            payload["software_version_int"] = miner_version

            await broadcast_service.broadcast_heartbeat(payload)
            hb_count += 1

        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.debug(f"Validator heartbeat error: {e}")


async def _network_speedtest_loop():
    """Refresh miner-side bandwidth telemetry without blocking heartbeat."""
    from neurons.miner.services.hardware_service import run_network_speed_test

    if os.environ.get("MINER_NETWORK_SPEEDTEST_ENABLED", "1").strip().lower() in {
        "0", "false", "no", "off",
    }:
        bt.logging.info("Network speed test disabled (MINER_NETWORK_SPEEDTEST_ENABLED=0)")
        return

    try:
        interval = int(os.environ.get("MINER_NETWORK_SPEEDTEST_INTERVAL_S", "900"))
    except ValueError:
        interval = 900
    try:
        delay = int(os.environ.get("MINER_NETWORK_SPEEDTEST_INITIAL_DELAY_S", "15"))
    except ValueError:
        delay = 15
    interval = max(300, interval)
    delay = max(0, delay)
    while True:
        try:
            await asyncio.sleep(delay)
            await asyncio.to_thread(run_network_speed_test)
            delay = interval
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.debug(f"Network speed test loop error: {e}")
            delay = min(interval, 300)


async def _ttl_checker():
    """Check for and terminate expired rental containers."""
    while True:
        try:
            if docker_service:
                docker_service.check_ttl_expiry()
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"TTL checker error: {e}")


async def _metagraph_stats_loop(rpc, uid: int | None, interval: float = 60.0):
    """Periodic compact on-chain position summary.

    Logs at INFO every `interval` seconds:
        Metagraph | block=N | UID X | incentive=… | emission=…α/tempo | trust=… | stake=…α

    Operators want to see their UID's standing without opening a CLI; this
    is the "where do I stand right now" heartbeat. RPC failures are silent
    (DEBUG) so a transient testnet 429 doesn't pollute the log.
    """
    if rpc is None or uid is None:
        return
    while True:
        try:
            await asyncio.sleep(interval)
            # Single call: get_metagraph returns cached metagraph (3-min TTL)
            # if it was refreshed recently. Calling get_current_block separately
            # would race against proof_service's chain calls on the same websocket.
            mg = await asyncio.to_thread(rpc.get_metagraph, False)
            block = int(mg.block) if hasattr(mg, "block") else 0

            def _g(attr):
                return float(getattr(mg, attr)[uid]) if hasattr(mg, attr) and uid < len(getattr(mg, attr)) else 0.0

            parts = (
                f"UID {uid}",
                f"incentive={_g('incentive'):.4f}",
                f"emission={_g('emission'):.2f}α/tempo",
                f"trust={_g('trust'):.2f}",
                f"stake={_g('stake'):.2f}α",
            )
            bt.logging.info(f"Metagraph | block={block} | {' | '.join(parts)}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.debug(f"Metagraph stats refresh failed: {e}")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(title="Nodexo Miner", version="0.1.0", lifespan=lifespan)


# ── Health / status endpoints (for CLI + validators) ───────────

@app.get("/health")
async def health():
    return {"status": "ok", "miner_id": miner_id[:16] if miner_id else None}

@app.get("/version")
async def version():
    from neurons.version import (
        miner_version,
        miner_version_str,
        spec_version,
        version_str,
    )
    return {
        "service": "nodexo-miner",
        "version": version_str,
        "spec_version": spec_version,
        "miner_version": miner_version,
        "miner_version_str": miner_version_str,
    }

@app.post("/identity/challenge")
async def identity_challenge(request: Request):
    """Prove this endpoint controls the registered executor EVM identity."""
    if not _evm_address or not _evm_private_key or not miner_id:
        raise HTTPException(501, "identity challenge unavailable")

    body = await request.json()
    nonce_hex = str(body.get("nonce") or "")
    executor_id = str(body.get("executor_id") or "")
    try:
        nonce = bytes.fromhex(nonce_hex)
        if executor_id.lower().removeprefix("0x") != miner_id.lower():
            raise HTTPException(400, "executor_id mismatch")
        from common.identity import sign_identity_challenge
        signature = sign_identity_challenge(
            nonce, _evm_address, miner_id, _evm_private_key,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"invalid identity challenge: {e}")

    return {
        "address": _evm_address,
        "executor_id": miner_id,
        "signature": signature,
    }


@app.get("/hardware")
async def get_hardware():
    from neurons.miner.services.hardware_service import get_gpu_utilization
    return {**hardware_info, "gpu_utilization": get_gpu_utilization()}

@app.get("/proof/status")
async def proof_status():
    if proof_service is None:
        raise HTTPException(503, "Proof service not initialized")
    return {
        "last_proven_epoch": proof_service._last_proven_epoch,
        "running": proof_service._running,
        "miner_id": miner_id,
    }

@app.get("/proof/latest")
async def proof_latest():
    return {"status": "not_implemented"}


# ── Container endpoints (for validator rental orchestration) ───
#
# Gated by `_require_validator` — only SR25519-signed requests from a
# hotkey in the configured allowlist may create/destroy containers.
# Without this gate any host on the network can spin arbitrary Docker
# containers on the miner box (RCE-equivalent).
#
# NOTE: container names are restricted to a safe character set inside
# docker_service.create_container — body["name"] is not blindly trusted
# even after the signature check, since a compromised validator key
# could still try to break out via shell metacharacters.

# Docker name rule: alnum, dash, underscore, dot. Anchored.
import re as _re
_CONTAINER_NAME_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_container_name(name: str) -> str:
    if not _CONTAINER_NAME_RE.match(name):
        raise HTTPException(400, "Invalid container name")
    return name


@app.post("/containers", dependencies=[Depends(_require_validator)])
async def create_container(request: Request):
    # Hard gate — without an attestable monitor we refuse to host
    # rentals. The validator's sybil scanner ALSO enforces this from
    # its side (monitor_silent audit flag); the
    # miner enforces locally too so a misconfigured operator sees the
    # problem immediately rather than via a slow reliability drop.
    if not _monitor_alive:
        raise HTTPException(
            503,
            "Monitor container not running. The miner refuses to provision "
            "rentals without an attestable witness. Check docker logs for "
            "nodexo-monitor-<executor_id[:8]> and the NODEXO_MONITOR_IMAGE env.",
        )
    body = await request.json()
    resources = _validated_container_resources(body)
    from neurons.miner.services.docker_service import ContainerConfig
    config = ContainerConfig(
        name=_validate_container_name(
            body.get("name") or f"nodexo-rental-{int(__import__('time').time())}"
        ),
        image=body.get("image", "ubuntu:22.04"),
        gpu_uuids=body.get("gpu_uuids"),
        cpu_count=resources["cpu_count"],
        memory_gb=resources["memory_gb"],
        storage_gb=resources["storage_gb"],
        ssh_pub_key=body.get("ssh_pub_key", ""),
        ports=body.get("ports"),
        expose_tcp_ports=_request_bool(body, "expose_tcp_ports", False),
        use_sysbox=body.get("use_sysbox", True),
        ttl_seconds=body.get("ttl_seconds", 0),
    )
    try:
        image_present = docker_service.image_present(config.image)
        if not image_present:
            require_cached = _request_bool(body, "require_image_cached", True)
            pull_image = _request_bool(body, "pull_image", False)
            if require_cached or not pull_image:
                raise HTTPException(
                    409,
                    f"Image is not ready on this executor: {config.image}.",
                )
            if not _image_cold_pull_allowed(config.image):
                raise HTTPException(
                    403,
                    f"Image is not enabled for cold start on this executor: {config.image}.",
                )
            reserve_gb = _cold_image_storage_reserve_gb(config.image, resources["storage_gb"])
            available_gb = _docker_storage_available_gb()
            if available_gb < reserve_gb:
                raise HTTPException(
                    422,
                    "not enough free Docker storage to prepare this image "
                    f"({available_gb}GB free, require {reserve_gb}GB before pull)",
                )
            pull_timeout = _request_int(
                body,
                "image_pull_timeout_s",
                _env_int_value("MINER_IMAGE_PULL_TIMEOUT_S", 1800, minimum=60),
                minimum=60,
                maximum=7200,
            )
            pulled = await asyncio.to_thread(_pull_docker_image, config.image, pull_timeout)
            if not pulled or not docker_service.image_present(config.image):
                await asyncio.to_thread(_prune_dangling_docker_images)
                raise HTTPException(
                    503,
                    f"Image could not be prepared on this executor: {config.image}.",
                )
            after_pull_gb = _docker_storage_available_gb()
            post_pull_reserve_gb = resources["storage_gb"] + _rental_storage_headroom_gb()
            if after_pull_gb < post_pull_reserve_gb:
                _remove_docker_image(config.image)
                _prune_dangling_docker_images()
                raise HTTPException(
                    422,
                    "prepared image would leave too little Docker storage for "
                    f"the rental ({after_pull_gb}GB free, require "
                    f"{post_pull_reserve_gb}GB)",
                )
            try:
                resources = _validated_container_resources(body)
            except HTTPException:
                _remove_docker_image(config.image)
                _prune_dangling_docker_images()
                raise
            config.cpu_count = resources["cpu_count"]
            config.memory_gb = resources["memory_gb"]
            config.storage_gb = resources["storage_gb"]
        info = docker_service.create_container(config)
        return {"container_id": info.container_id, "name": info.name,
                "ssh_port": info.ssh_port, "ssh_user": info.ssh_user,
                "tcp_ports": info.tcp_ports,
                "status": info.status, "image": info.image, "image_id": info.image_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Container creation failed: {e}")

@app.delete("/containers/{name}", dependencies=[Depends(_require_validator)])
async def destroy_container(name: str):
    _validate_container_name(name)
    if not docker_service.destroy_container(name):
        raise HTTPException(500, f"Failed to destroy container {name}")
    return {"status": "destroyed", "name": name}

@app.get("/containers", dependencies=[Depends(_require_validator)])
async def list_containers():
    return [{"container_id": c.container_id, "name": c.name, "ssh_port": c.ssh_port,
             "ssh_user": c.ssh_user, "status": c.status,
             "tcp_ports": c.tcp_ports,
             "created_at": c.created_at, "ttl_seconds": c.ttl_seconds,
             "image": c.image, "image_id": c.image_id}
            for c in docker_service.list_containers()]

@app.get("/ports", dependencies=[Depends(_require_validator)])
async def port_status():
    """Validator-visible rental SSH port pool.

    Renter UI uses this to explain where the assigned SSH port came from
    and whether the miner still has free rental ports.
    """
    return docker_service.port_status()


@app.post("/containers/{name}/ssh_keys", dependencies=[Depends(_require_validator)])
async def add_container_ssh_key(name: str, request: Request):
    """Append an SSH pubkey to a rental container's authorized_keys.

    Validator-only (signed). The key text is piped via stdin to
    `docker exec … tee -a` so renter-controlled content never reaches
    a shell.
    """
    _validate_container_name(name)
    body = await request.json()
    pubkey = body.get("ssh_pub_key", "")
    if not pubkey:
        raise HTTPException(400, "ssh_pub_key required")
    if not docker_service.add_ssh_key(name, pubkey):
        raise HTTPException(500, "Failed to add SSH key inside container")
    return {"status": "ok"}


@app.delete("/containers/{name}/ssh_keys", dependencies=[Depends(_require_validator)])
async def remove_container_ssh_key(name: str, request: Request):
    """Remove an SSH pubkey from a rental container's authorized_keys.

    Matches by key body (the AAAA… field). Idempotent: returns ok even
    if the key was already absent.
    """
    _validate_container_name(name)
    body = await request.json()
    pubkey = body.get("ssh_pub_key", "")
    if not pubkey:
        raise HTTPException(400, "ssh_pub_key required")
    if not docker_service.remove_ssh_key(name, pubkey):
        raise HTTPException(500, "Failed to remove SSH key inside container")
    return {"status": "ok"}


# /upload_ssh_key and /remove_ssh_key were a vestigial dead path: they
# wrote to the MINER HOST's ~/.ssh/authorized_keys (not into the rental
# container), and the CLI never invoked them with the right body field
# anyway. Container SSH keys are injected by docker_service._setup_ssh
# at create time. Routes deleted — host-level shells are not part of
# the rental contract and shouldn't be reachable from the network.


# ── Entry point ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nodexo Miner")
    parser.add_argument("--wallet", default="miner", help="Bittensor wallet name")
    parser.add_argument("--hotkey", default="default", help="Bittensor hotkey name")
    parser.add_argument("--netuid", type=int, default=0, help="Subnet UID")
    parser.add_argument("--chain-config", default="", help="Path to chain_config.json")
    parser.add_argument(
        "--subtensor-network",
        default=(
            os.environ.get("NODEXO_MINER_SUBTENSOR_NETWORK")
            or os.environ.get("NODEXO_SUBTENSOR_NETWORK")
            or os.environ.get("SUBTENSOR_NETWORK")
            or "finney"
        ),
        help="Semantic Bittensor network: test or finney",
    )
    parser.add_argument(
        "--subtensor-endpoint",
        default=(
            os.environ.get("NODEXO_MINER_SUBTENSOR_ENDPOINT")
            or os.environ.get("NODEXO_SUBTENSOR_ENDPOINT")
            or os.environ.get("SUBTENSOR_ENDPOINT")
            or ""
        ),
        help="Optional private/local subtensor RPC endpoint for the selected network",
    )
    parser.add_argument("--mock", action="store_true", help="Mock mode: skip chain, use fake block hashes")
    parser.add_argument("--validator-url", default="", help="Validator URL for proof broadcast (mock mode)")
    parser.add_argument("--port", type=int, default=8090, help="Miner daemon API port")
    parser.add_argument(
        "--bind-host",
        default=os.environ.get("MINER_BIND_HOST", "0.0.0.0"),
        help="Local interface for the miner API. Use 127.0.0.1 behind nginx.",
    )
    parser.add_argument("--endpoint", default="", help="Public endpoint URL (registered on-chain, e.g. https://mygpu.example.com:8090)")
    parser.add_argument("--rental-port-start", type=int, default=20000, help="Start of port range for rental containers")
    parser.add_argument("--rental-port-end", type=int, default=20100, help="End of port range for rental containers")
    parser.add_argument(
        "--rental-port-preflight",
        choices=("strict", "warn", "off"),
        default=os.environ.get("MINER_RENTAL_PORT_PREFLIGHT", "strict"),
        help=(
            "Internet reachability preflight for rental SSH ports before "
            "executor registration (strict, warn, off)"
        ),
    )
    parser.add_argument(
        "--rental-port-preflight-scope",
        choices=("sample", "full"),
        default=os.environ.get("MINER_RENTAL_PORT_PREFLIGHT_SCOPE", "sample"),
        help="Check sampled ports (default) or every port in the rental range",
    )
    parser.add_argument(
        "--rental-port-preflight-samples",
        type=int,
        default=int(os.environ.get("MINER_RENTAL_PORT_PREFLIGHT_SAMPLES", "3")),
        help="Number of evenly-spaced ports to check when scope=sample",
    )
    parser.add_argument(
        "--rental-port-preflight-timeout",
        type=float,
        default=float(os.environ.get("MINER_RENTAL_PORT_PREFLIGHT_TIMEOUT_S", "8")),
        help="Timeout per external port-check request in seconds",
    )
    parser.add_argument(
        "--rental-image-preflight",
        choices=("pull", "strict", "warn", "off"),
        default=os.environ.get("MINER_RENTAL_IMAGE_PREFLIGHT", "strict"),
        help=(
            "Warm image startup check before registration: strict-fail without "
            "pulling, pull missing required images and fail if any remain "
            "missing, warn, or off"
        ),
    )
    parser.add_argument(
        "--optional-image-preflight",
        choices=("pull", "warn", "off"),
        default=os.environ.get("MINER_OPTIONAL_IMAGE_PREFLIGHT", "warn"),
        help=(
            "Best-effort warmup for optional catalog images while preserving "
            "the minimum rental storage reserve"
        ),
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        default=_env_bool("MINER_AUTO_UPDATE") or _env_bool("AUTO_UPDATE"),
        help="Enable role-aware git auto-update when remote miner_version increases",
    )
    parser.add_argument(
        "--auto-update-interval",
        type=int,
        default=int(os.environ.get("MINER_AUTO_UPDATE_INTERVAL_S", os.environ.get("AUTO_UPDATE_INTERVAL_S", "1800"))),
        help="Seconds between auto-update checks",
    )
    parser.add_argument(
        "--auto-update-restart-delay",
        type=int,
        default=int(os.environ.get("MINER_AUTO_UPDATE_RESTART_DELAY_S", "5")),
        help="Seconds to wait after installing an update before restarting",
    )
    parser.add_argument(
        "--auto-update-jitter-seconds",
        type=int,
        default=int(os.environ.get("MINER_AUTO_UPDATE_JITTER_SECONDS", "1800")),
        help="Deterministic miner restart stagger window in seconds",
    )
    parser.add_argument("--price", type=int, default=0, help="Price per GPU hour in RAO")
    args = parser.parse_args()
    if not args.netuid:
        args.netuid = _default_netuid_for_network(args.subtensor_network)
    if not args.chain_config:
        args.chain_config = _resolve_chain_config_path(args.subtensor_network)
    if not args.mock and not args.netuid:
        bt.logging.error(
            "No netuid configured. Pass --netuid for custom networks or use "
            "--subtensor-network finney|test."
        )
        sys.exit(1)
    if not args.mock and not args.chain_config:
        bt.logging.error(
            "No chain config found. Expected chain_config_mainnet.json, "
            "chain_config_testnet.json, or chain_config.json in the repo root. "
            "Pass --chain-config only for custom deployments."
        )
        sys.exit(1)

    # Store config on app.state for lifespan access
    app.state.wallet_name = args.wallet
    app.state.hotkey_name = args.hotkey
    app.state.netuid = args.netuid
    app.state.chain_config = args.chain_config
    app.state.subtensor_network = args.subtensor_network
    app.state.subtensor_endpoint = args.subtensor_endpoint
    app.state.price_per_gpu_hour = args.price
    app.state.endpoint = args.endpoint or f"http://0.0.0.0:{args.port}"
    app.state.bind_host = args.bind_host
    app.state.rental_port_start = args.rental_port_start
    app.state.rental_port_end = args.rental_port_end
    app.state.mock = args.mock
    app.state.validator_url = args.validator_url
    app.state.auto_update_enabled = bool(args.auto_update)
    app.state.auto_update_interval = args.auto_update_interval
    app.state.auto_update_restart_delay = args.auto_update_restart_delay
    app.state.auto_update_jitter_seconds = args.auto_update_jitter_seconds

    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    bt.logging.enable_info()

    # ── Pre-init ALL blocking chain work before uvicorn starts ──
    # bittensor and web3 are synchronous; can't run in async lifespan.
    if not args.mock:
        desired_endpoint = args.endpoint or f"http://0.0.0.0:{args.port}"
        from neurons.miner.services.hardware_service import detect_gpus

        _validate_miner_endpoint(desired_endpoint)
        preflight_gpus = detect_gpus()
        if not preflight_gpus:
            bt.logging.error("No GPUs detected! Miner cannot start.")
            sys.exit(1)
        _load_subnet_config_for_startup()
        try:
            _preflight_gpu_timing_support(preflight_gpus)
        except RuntimeError as e:
            bt.logging.error(str(e))
            sys.exit(1)
        _preflight_rental_port_range(
            desired_endpoint,
            args.rental_port_start,
            args.rental_port_end,
            mode=args.rental_port_preflight,
            scope=args.rental_port_preflight_scope,
            samples=args.rental_port_preflight_samples,
            timeout_s=args.rental_port_preflight_timeout,
        )
        if _proof_only_calibration_mode():
            bt.logging.warning(
                "NODEXO_PROOF_ONLY_CALIBRATION=1: skipping rental storage and "
                "image readiness checks. This host can collect proof timing "
                "samples but must not be offered for rentals."
            )
        else:
            _preflight_rental_resources()
            _preflight_rental_images(args.rental_image_preflight)
            _preflight_optional_rental_images(args.optional_image_preflight)

        from common.chain.rpc import SubtensorRPC
        from common.chain.wallet import load_hotkey_seed, derive_evm_account

        SubCls = getattr(bt, "Subtensor", None) or bt.subtensor
        WalletCls = getattr(bt, "Wallet", None) or bt.wallet
        AxonCls = getattr(bt, "Axon", None) or getattr(bt, "axon", None)

        # Public RPC cooldowns should hold the miner in startup readiness, not
        # create a PM2 crash loop or let proof generation start with partial
        # chain context.
        def _connect_subtensor():
            target = args.subtensor_endpoint or args.subtensor_network
            if args.subtensor_endpoint:
                bt.logging.info(
                    f"Connecting to subtensor ({args.subtensor_network}, "
                    f"endpoint={args.subtensor_endpoint})..."
                )
            else:
                bt.logging.info(f"Connecting to subtensor ({args.subtensor_network})...")
            return SubCls(network=target)

        sub = _run_startup_chain_step("Subtensor connection", _connect_subtensor)
        rpc = SubtensorRPC(
            network=args.subtensor_network,
            netuid=args.netuid,
            subtensor=sub,
            chain_endpoint=args.subtensor_endpoint,
        )
        current_block = _run_startup_chain_step(
            "Subtensor current block",
            lambda: sub.get_current_block(),
        )
        bt.logging.info(f"Subtensor connected (block={current_block})")

        wallet = WalletCls(name=args.wallet, hotkey=args.hotkey)
        uid = _run_startup_chain_step(
            "subnet UID lookup",
            lambda: rpc.get_uid_for_hotkey(wallet.hotkey.ss58_address),
        )
        if uid is None:
            bt.logging.error(f"Hotkey {wallet.hotkey.ss58_address} not registered on subnet {args.netuid}")
            sys.exit(1)
        bt.logging.info(f"UID: {uid} (hotkey: {wallet.hotkey.ss58_address[:16]})")
        # Stash for the metagraph-stats logger (read in lifespan)
        app.state._pre_uid = uid
        app.state._pre_hotkey_ss58 = wallet.hotkey.ss58_address

        if AxonCls:
            try:
                parsed_endpoint = urlparse(desired_endpoint)
                host = (parsed_endpoint.hostname or "").strip()
                port = int(parsed_endpoint.port or args.port)
                ip = ipaddress.ip_address(host)
                if not (ip.is_loopback or ip.is_unspecified or ip.is_private):
                    rpc.subtensor.serve_axon(
                        axon=AxonCls(
                            wallet=wallet,
                            port=port,
                            ip="0.0.0.0",
                            external_ip=str(ip),
                            external_port=port,
                        ),
                        netuid=args.netuid,
                    )
                else:
                    bt.logging.debug("serve_axon skipped: miner endpoint IP is not public")
            except ValueError:
                bt.logging.debug("serve_axon skipped: miner endpoint host is not an IP")
            except Exception as e:
                bt.logging.warning(f"serve_axon: {e}")

        hotkey_seed = load_hotkey_seed(args.wallet, args.hotkey)
        evm_account = derive_evm_account(hotkey_seed)
        bt.logging.info(f"EVM address: {evm_account.address}")

        compute_reg = None
        vali_reg = None
        fresh_or_reactivated_registration = False
        startup_validators = []
        executor_id_hex = ""
        gpu_model_hash = b""
        gpu_count = len(preflight_gpus) or 1
        vram_mb = preflight_gpus[0].vram_mb if preflight_gpus else 0
        if args.chain_config and os.path.exists(args.chain_config):
            from common.config import ChainConfig
            from common.chain.compute_registry import ComputeRegistryClient
            from common.chain.validator_registry import ValidatorRegistryClient

            cfg = ChainConfig.from_json(
                args.chain_config,
                subtensor_network=args.subtensor_network,
                chain_endpoint=args.subtensor_endpoint,
            )
            compute_reg = ComputeRegistryClient(cfg, private_key=evm_account.key.hex())
            vali_reg = ValidatorRegistryClient(cfg)

            # EVM + executor registration with proper SR25519 proof
            from neurons.miner.services.hardware_service import get_or_create_executor_id
            from common.chain.wallet import sign_evm_registration
            from common.config import gpu_model_hash_bytes
            gpus = preflight_gpus
            executor_id_hex = get_or_create_executor_id(gpus)
            gpu_model_hash = gpu_model_hash_bytes(gpus[0].name if gpus else "unknown")
            gpu_count = len(gpus) or 1
            vram_mb = gpus[0].vram_mb if gpus else 0

            def _ensure_evm_registration():
                if not compute_reg.is_evm_registered(evm_account.address):
                    bt.logging.info(f"Registering EVM address for UID {uid}...")
                    sig_r, sig_s = sign_evm_registration(
                        hotkey_seed, evm_account.address, uid, args.netuid,
                        compute_reg.contract.address)
                    compute_reg.register_evm(uid, sig_r, sig_s)
                    bt.logging.info("EVM registered on ComputeRegistry")

            _run_startup_chain_step(
                "ComputeRegistry EVM registration",
                _ensure_evm_registration,
            )

            # Register executor (GPU machine) on ComputeRegistry.
            # On restart with already-registered executor, sync any launch-config
            # changes to the chain spec so the dashboard / discovery don't show
            # stale data. Today: endpoint. Future: price, vram, etc. via
            # updateExecutorSpecs.
            def _reconcile_executor_registry():
                return _ensure_executor_registry_state(
                    compute_reg,
                    executor_id_hex,
                    endpoint=desired_endpoint,
                    gpu_model_hash=gpu_model_hash,
                    gpu_count=gpu_count,
                    vram_mb=vram_mb,
                    price_per_gpu_hour=args.price,
                    force_renew=True,
                )

            msg = _run_startup_chain_step(
                "ComputeRegistry executor registration",
                _reconcile_executor_registry,
            )
            bt.logging.info(f"Executor registry reconciled: {executor_id_hex[:16]}")
            fresh_or_reactivated_registration = bool(
                msg and (
                    "registered/reactivated" in msg
                    or "deregistered/reactivated" in msg
                )
            )
            if msg:
                bt.logging.info(msg)

        startup_validators = _run_startup_chain_step(
            "fresh validator discovery",
            lambda: _startup_discover_validators(rpc, vali_reg, uid),
        )

        # Store pre-computed state for the lifespan
        app.state._chain_ready = True
        app.state._pre_rpc = rpc
        app.state._pre_compute_registry = compute_reg
        app.state._pre_validator_registry = vali_reg
        app.state._pre_hotkey_seed = hotkey_seed
        app.state._pre_evm_address = evm_account.address
        app.state._pre_evm_private_key = evm_account.key.hex()
        app.state._pre_executor_endpoint = desired_endpoint
        app.state._pre_executor_gpu_model_hash = gpu_model_hash
        app.state._pre_executor_gpu_count = gpu_count
        app.state._pre_executor_vram_mb = vram_mb
        app.state._pre_executor_price_per_gpu_hour = args.price
        app.state._pre_executor_fresh_registration = fresh_or_reactivated_registration
        app.state._pre_startup_validators = startup_validators

    import uvicorn
    # access_log=False suppresses per-request `INFO: 127.0.0.1 - "POST" 200 OK` spam.
    # log_level="warning" silences uvicorn's own startup chatter (we already log via bt.logging).
    uvicorn.run(
        app, host=args.bind_host, port=args.port,
        log_level="warning", access_log=False, loop="asyncio",
    )


if __name__ == "__main__":
    main()
