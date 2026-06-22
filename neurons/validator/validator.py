"""
Nodexo Validator — main daemon.

Responsibilities:
1. Register on EVM (ComputeRegistry + ValidatorRegistry)
2. Discover executors from ComputeRegistry on-chain
3. Receive proof commitments + recipes from executors (passive)
4. Verify proofs (light/spot/full via ProcessPoolExecutor)
5. Compute composite scores per executor
6. Set weights on Bittensor chain every tempo
7. Expose rental API for users

Proof ingestion + verification entrypoint (prior-art in protocol design notes).
"""
from __future__ import annotations

import asyncio
import argparse
import ipaddress
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def _set_default_validator_thread_env() -> None:
    """Keep native verifier libraries from starving the validator HTTP loop."""
    defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


_set_default_validator_thread_env()

import bittensor as bt
from fastapi import FastAPI, HTTPException, Request
from neurons.validator.proof.verifier import VerificationResult

# Logging is handled exclusively by bt.logging (loguru-based).
# __main__ calls bt.logging.enable_info() before starting the server.

# ── Global state ───────────────────────────────────────────────
analyzer = None
verification_queue: asyncio.Queue = None  # Initialized in lifespan
executor_scores = {}
verify_pool = None


class RollingBeaconCache:
    """Validator-local block hash cache used by proof verification.

    Proof verification must not block on chain RPC while miner receipts are
    arriving. A single background task keeps recent block hashes warm; the
    verification path reads from this cache and retries neutrally when the
    validator's chain view is unavailable.

    The timestamp stored beside each hash is the validator's best local
    availability estimate. For blocks observed at the live head that is the
    poll time; for backfilled blocks it is derived from the observed head time
    and block distance. A stalled cache must not mark an old beacon as becoming
    available only when the delayed refresh finally returns, because that turns
    honest receipts into false pre-beacon causality failures.
    """

    def __init__(
        self,
        rpc_ref,
        *,
        poll_s: float | None = None,
        max_blocks: int | None = None,
        backfill_blocks: int | None = None,
        max_fetch_per_refresh: int | None = None,
        refresh_timeout_s: float | None = None,
    ):
        self._rpc_ref = rpc_ref
        self._poll_s = (
            max(0.25, float(os.environ.get("NODEXO_BEACON_CACHE_POLL_S", "2")))
            if poll_s is None else max(0.25, float(poll_s))
        )
        self._max_blocks = (
            max(32, int(os.environ.get("NODEXO_BEACON_CACHE_MAX_BLOCKS", "512")))
            if max_blocks is None else max(32, int(max_blocks))
        )
        self._backfill_blocks = (
            max(4, int(os.environ.get("NODEXO_BEACON_CACHE_BACKFILL_BLOCKS", "48")))
            if backfill_blocks is None else max(4, int(backfill_blocks))
        )
        self._max_fetch_per_refresh = (
            max(4, int(os.environ.get("NODEXO_BEACON_CACHE_MAX_FETCH_PER_REFRESH", "64")))
            if max_fetch_per_refresh is None else max(4, int(max_fetch_per_refresh))
        )
        self._refresh_timeout_s = (
            max(1.0, float(os.environ.get("NODEXO_BEACON_CACHE_REFRESH_TIMEOUT_S", "20")))
            if refresh_timeout_s is None else max(1.0, float(refresh_timeout_s))
        )
        self._lock = threading.Lock()
        self._blocks: dict[int, tuple[bytes, float]] = {}
        self._head = 0
        self._head_at = 0.0
        self._last_error = ""
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nodexo-beacon-cache",
        )
        self._last_timeout_log_at = 0.0
        self._rpc_min_interval_s = self._default_rpc_min_interval_s(rpc_ref)
        self._last_rpc_call_at = 0.0
        self._block_seconds = max(
            1.0,
            float(os.environ.get("NODEXO_BEACON_CACHE_BLOCK_SECONDS", "12")),
        )

    @staticmethod
    def _default_rpc_min_interval_s(rpc_ref) -> float:
        configured = os.environ.get("NODEXO_BEACON_CACHE_RPC_MIN_INTERVAL_S")
        if configured is not None:
            return max(0.0, float(configured))
        network = str(getattr(rpc_ref, "network", "") or "").lower()
        if network in {"test", "finney"}:
            # Public Bittensor endpoints are rate-limited. Keep the validator's
            # beacon cache useful there without moving chain latency onto proof
            # ingress or spawning overlapping retry storms.
            return 1.05
        return 0.0

    def _throttle_rpc_call(self) -> None:
        if self._rpc_min_interval_s <= 0:
            return
        now = time.monotonic()
        wait_s = self._rpc_min_interval_s - (now - self._last_rpc_call_at)
        if wait_s > 0:
            time.sleep(wait_s)
        self._last_rpc_call_at = time.monotonic()

    def head(self, *, max_age_s: float | None = None) -> int:
        with self._lock:
            if max_age_s is not None and self._head_at:
                if time.time() - self._head_at > max_age_s:
                    return 0
            return int(self._head or 0)

    def head_state(self, *, max_age_s: float | None = None) -> tuple[int, float]:
        with self._lock:
            head = int(self._head or 0)
            head_at = float(self._head_at or 0.0)
            if max_age_s is not None and head_at:
                if time.time() - head_at > max_age_s:
                    return 0, 0.0
            return head, head_at

    def is_fresh(self, max_age_s: float | None = None) -> bool:
        max_age = (
            max(1.0, float(os.environ.get("NODEXO_BEACON_CACHE_MAX_AGE_S", "30")))
            if max_age_s is None else max(1.0, float(max_age_s))
        )
        return self.head(max_age_s=max_age) > 0

    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def context_for_epoch(
        self,
        epoch_id: int,
        epoch_interval: int,
        max_beacon_offset_blocks: int,
        current_head: int = 0,
    ) -> tuple[int, bytes, float, int] | None:
        from common.proof_schedule import derive_beacon_offset_blocks

        cycle_start = int(epoch_id) * int(epoch_interval)
        with self._lock:
            head = max(int(current_head or 0), int(self._head or 0))
            anchor_entry = self._blocks.get(cycle_start)
        if anchor_entry is None:
            return None

        anchor_hash, _anchor_ts = anchor_entry
        offset = derive_beacon_offset_blocks(anchor_hash, max_beacon_offset_blocks)
        beacon_block = cycle_start + offset
        with self._lock:
            beacon_entry = self._blocks.get(beacon_block)
        if beacon_entry is None:
            return None

        beacon_hash, beacon_available_at = beacon_entry
        return head, beacon_hash, beacon_available_at, int(beacon_block)

    def refresh_once(self) -> int:
        self._throttle_rpc_call()
        current = int(self._rpc_ref.get_current_block())
        observed_at = time.time()
        if current <= 0:
            return 0

        with self._lock:
            previous_head = int(self._head or 0)
            known_blocks = set(self._blocks)
            self._head = max(int(self._head or 0), current)
            self._head_at = observed_at

        if previous_head <= 0:
            start = max(0, current - self._backfill_blocks + 1)
        else:
            start = max(previous_head + 1, current - self._backfill_blocks + 1)
        start = max(start, current - self._max_fetch_per_refresh + 1)
        if start > current:
            start = current

        updates: dict[int, tuple[bytes, float]] = {}
        for block_number in range(start, current + 1):
            if block_number in known_blocks:
                continue
            self._throttle_rpc_call()
            block_hash = self._rpc_ref.get_block_hash(block_number)
            blocks_ago = max(0, int(current) - int(block_number))
            hash_available_at = observed_at - (blocks_ago * self._block_seconds)
            hash_available_at = min(float(hash_available_at), time.time())
            if not isinstance(block_hash, bytes):
                block_hash = bytes.fromhex(str(block_hash).removeprefix("0x"))
            block_entry = (block_hash, float(hash_available_at))
            updates[int(block_number)] = block_entry
            with self._lock:
                self._blocks[int(block_number)] = block_entry

        with self._lock:
            self._blocks.update(updates)
            self._head = max(int(self._head or 0), current)
            self._head_at = observed_at
            self._last_error = ""
            if len(self._blocks) > self._max_blocks:
                for old_block in sorted(self._blocks)[: len(self._blocks) - self._max_blocks]:
                    self._blocks.pop(old_block, None)
        return current

    async def run(self) -> None:
        bt.logging.info(
            f"Beacon cache starting (poll={self._poll_s:.1f}s, "
            f"backfill={self._backfill_blocks} blocks)"
        )
        loop = asyncio.get_running_loop()
        refresh_future = None
        try:
            while True:
                try:
                    if refresh_future is None:
                        refresh_future = loop.run_in_executor(
                            self._executor,
                            self.refresh_once,
                        )
                    head = await asyncio.wait_for(
                        asyncio.shield(refresh_future),
                        timeout=self._refresh_timeout_s,
                    )
                    refresh_future = None
                    if head:
                        bt.logging.debug(f"Beacon cache refreshed: head={head}")
                    await asyncio.sleep(self._poll_s)
                except asyncio.TimeoutError:
                    msg = (
                        f"refresh still running after "
                        f"{self._refresh_timeout_s:.1f}s"
                    )
                    with self._lock:
                        self._last_error = msg
                    now = time.time()
                    if now - self._last_timeout_log_at >= 60:
                        self._last_timeout_log_at = now
                        bt.logging.warning(f"Beacon cache refresh stalled: {msg}")
                    # Keep waiting for the same worker future. Starting another
                    # chain read would pile up unkillable public-RPC calls and
                    # starve proof ingress.
                    await asyncio.sleep(max(self._poll_s, 1.0))
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    refresh_future = None
                    detail = str(e).encode("unicode_escape").decode()
                    msg = f"{type(e).__name__}: {detail}"[:200]
                    with self._lock:
                        self._last_error = msg
                    bt.logging.warning(f"Beacon cache refresh failed: {msg}")
                    await asyncio.sleep(max(self._poll_s, 1.0))
        finally:
            if refresh_future is not None and not refresh_future.done():
                refresh_future.cancel()
            self._executor.shutdown(wait=False, cancel_futures=True)


def _verify_pool_init():
    _set_default_validator_thread_env()
    try:
        nice = int(os.environ.get("VERIFY_WORKER_NICE", "5"))
    except Exception:
        nice = 5
    if nice > 0:
        try:
            os.nice(nice)
        except Exception:
            pass
    # Pre-import heavy modules so first verify call doesn't pay the import cost.
    from zkgemm.verifier import ZkGemmBlockVerifier  # noqa: F401
    from neurons.validator.proof.verifier import verify_recipe  # noqa: F401


def _verification_worker_start_method() -> str:
    method = os.environ.get("VERIFY_WORKER_START_METHOD", "spawn").strip().lower()
    if method not in mp.get_all_start_methods():
        raise RuntimeError(
            f"VERIFY_WORKER_START_METHOD={method!r} is unsupported; "
            f"available={','.join(mp.get_all_start_methods())}"
        )
    return method


def _create_verify_pool(num_workers: int) -> ProcessPoolExecutor:
    # Use spawn by default so workers do not inherit validator RPC/websocket,
    # Postgres, or listener sockets. Forked workers can leave inherited chain
    # sockets in CLOSE_WAIT and stall proof verification after HTTP accepts.
    ctx = mp.get_context(_verification_worker_start_method())
    return ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_verify_pool_init,
        mp_context=ctx,
    )


def _default_verify_workers() -> int:
    cpu_count = max(1, int(os.cpu_count() or 1))
    reserve = max(1, int(os.environ.get("VERIFY_RESERVED_CPUS", "1")))
    return max(1, min(cpu_count - reserve, 8))


def _verify_max_inflight_default() -> int:
    workers = int(getattr(verify_pool, "_max_workers", 0) or 0)
    if workers <= 0:
        workers = _default_verify_workers()
    return max(1, min(workers, 32))


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _validator_no_evm_enabled(cli_value: bool = False) -> bool:
    return (
        bool(cli_value)
        or _env_bool("VALIDATOR_NO_EVM")
        or _env_bool("NODEXO_VALIDATOR_NO_EVM")
        or _env_bool("NODEXO_NO_EVM")
    )


def _validate_validator_db_url(db_url: str) -> None:
    """Fail public validator launchers that accidentally fell back to SQLite."""
    if not _env_bool("VALIDATOR_REQUIRE_POSTGRES"):
        return
    if db_url.strip().lower().startswith("sqlite:"):
        raise RuntimeError(
            "VALIDATOR_REQUIRE_POSTGRES=1 but DB_URL points at SQLite. "
            "Run scripts/setup_validator.sh to provision Postgres, set "
            "NODEXO_VALIDATOR_DB_URL/DB_URL to a Postgres URL, or explicitly "
            "disable the guard for local-only development."
        )


def _normalize_validator_discovery_mode(value: str, *, no_evm: bool) -> str:
    mode = (value or "auto").strip().lower()
    if mode == "auto":
        return "native" if no_evm else "evm"
    if mode not in {"evm", "native", "both", "off"}:
        raise ValueError("validator discovery mode must be auto, evm, native, both, or off")
    if no_evm and mode == "evm":
        raise ValueError("no-EVM validators cannot use evm discovery mode")
    return mode


def _metagraph_validator_permit(metagraph, uid: int) -> bool:
    permits = getattr(metagraph, "validator_permit", None)
    if permits is None:
        return False
    try:
        return bool(permits[int(uid)])
    except Exception:
        return False


def _is_transient_subtensor_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return True
    msg = str(error)
    lower = msg.lower()
    return (
        "429" in msg
        or "too many requests" in lower
        or "timed out" in lower
        or "temporarily unavailable" in lower
    )


async def _subtensor_call_with_retry(label: str, fn):
    attempts = max(1, int(os.environ.get("VALIDATOR_CHAIN_CALL_MAX_ATTEMPTS", "6")))
    delay = float(os.environ.get("VALIDATOR_CHAIN_CALL_INITIAL_BACKOFF_S", "5"))
    max_delay = float(os.environ.get("VALIDATOR_CHAIN_CALL_MAX_BACKOFF_S", "60"))
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt >= attempts or not _is_transient_subtensor_error(e):
                raise
            msg_safe = str(e).encode("unicode_escape").decode()[:200]
            bt.logging.warning(
                f"{label} transient subtensor failure "
                f"({attempt}/{attempts}); sleeping {delay:.0f}s: {msg_safe}"
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts")


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


def _public_endpoint_ip_port(endpoint: str, default_port: int) -> tuple[str, int] | None:
    parsed = urlparse(endpoint or "")
    host = (parsed.hostname or "").strip()
    port = int(parsed.port or default_port or 0)
    if not host or port <= 0:
        bt.logging.warning("native validator axon not served: VALIDATOR_ENDPOINT is missing host/port")
        return None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        bt.logging.info(
            "native validator axon not served: endpoint host is not an IP. "
            "Use ValidatorRegistry for domain/HTTPS validator URLs."
        )
        return None
    if ip.is_loopback or ip.is_unspecified or ip.is_private:
        bt.logging.warning(
            "native validator axon not served: endpoint IP is not publicly routable"
        )
        return None
    return str(ip), port


def _serve_native_validator_axon(rpc, axon_cls, wallet, netuid: int, endpoint: str, default_port: int) -> None:
    """Publish validator API host/port through native subtensor axon metadata.

    This is the discovery path for no-EVM validators. ValidatorRegistry remains
    the richer EVM path for HTTPS/domain endpoints.
    """
    public_endpoint = _public_endpoint_ip_port(endpoint, default_port)
    if not public_endpoint:
        return
    ip, port = public_endpoint

    try:
        from bittensor.core.extrinsics.serving import serve_extrinsic

        serve_extrinsic(
            subtensor=rpc.subtensor,
            wallet=wallet,
            ip=ip,
            port=port,
            protocol=4,
            netuid=netuid,
            raise_error=False,
        )
        bt.logging.info(f"Native validator axon served: {ip}:{port}")
        return
    except Exception as e:
        bt.logging.warning(
            f"native validator direct serve failed; falling back to Axon wrapper: {e}"
        )

    if axon_cls is None:
        return
    axon = axon_cls(
        wallet=wallet,
        port=default_port,
        ip="0.0.0.0",
        external_ip=ip,
        external_port=port,
    )
    rpc.subtensor.serve_axon(axon=axon, netuid=netuid)
    bt.logging.info(f"Native validator axon served: {ip}:{port}")


def _parse_runtime_args():
    """Parse optional daemon args while preserving env-based launchers."""
    parser = argparse.ArgumentParser(description="Nodexo Validator", add_help=True)
    parser.add_argument("--wallet", default=os.environ.get("WALLET", "validator"))
    parser.add_argument("--hotkey", default=os.environ.get("HOTKEY", "default"))
    parser.add_argument("--netuid", type=int, default=int(os.environ.get("NETUID", "0")))
    parser.add_argument("--chain-config", default=os.environ.get("CHAIN_CONFIG", ""))
    parser.add_argument(
        "--subtensor-network",
        default=(
            os.environ.get("NODEXO_VALIDATOR_SUBTENSOR_NETWORK")
            or os.environ.get("NODEXO_SUBTENSOR_NETWORK")
            or os.environ.get("SUBTENSOR_NETWORK")
            or "finney"
        ),
        help="Semantic Bittensor network: test or finney",
    )
    parser.add_argument(
        "--subtensor-endpoint",
        default=(
            os.environ.get("NODEXO_VALIDATOR_SUBTENSOR_ENDPOINT")
            or os.environ.get("NODEXO_SUBTENSOR_ENDPOINT")
            or os.environ.get("SUBTENSOR_ENDPOINT")
            or ""
        ),
        help="Optional private/local subtensor RPC endpoint for the selected network",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("VALIDATOR_PORT", "8443")))
    parser.add_argument("--bind-host", default=os.environ.get("VALIDATOR_BIND_HOST", "0.0.0.0"))
    parser.add_argument("--endpoint", default=os.environ.get("VALIDATOR_ENDPOINT", ""))
    parser.add_argument(
        "--discovery-mode",
        default=os.environ.get("NODEXO_VALIDATOR_DISCOVERY_MODE", "auto"),
        help=(
            "Validator discovery publication mode: auto, evm, native, both, or off. "
            "auto publishes ValidatorRegistry endpoint for EVM validators and native "
            "Bittensor axon for no-EVM validators."
        ),
    )
    parser.add_argument(
        "--no-evm",
        action="store_true",
        default=_validator_no_evm_enabled(),
        help=(
            "Run as a read/verify validator without EVM writes. This skips "
            "ValidatorRegistry/ComputeRegistry registration, reportOffline, "
            "canaries, and rental control. Weight setting and proof "
            "verification still run."
        ),
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        default=_env_bool("VALIDATOR_AUTO_UPDATE") or _env_bool("AUTO_UPDATE"),
        help="Enable role-aware git auto-update when remote validator_version increases",
    )
    parser.add_argument(
        "--auto-update-interval",
        type=int,
        default=int(os.environ.get("VALIDATOR_AUTO_UPDATE_INTERVAL_S", os.environ.get("AUTO_UPDATE_INTERVAL_S", "1800"))),
        help="Seconds between auto-update checks",
    )
    parser.add_argument(
        "--auto-update-restart-delay",
        type=int,
        default=int(os.environ.get("VALIDATOR_AUTO_UPDATE_RESTART_DELAY_S", "5")),
        help="Seconds to wait after installing an update before restarting",
    )
    args, _unknown = parser.parse_known_args()
    try:
        args.discovery_mode = _normalize_validator_discovery_mode(
            args.discovery_mode, no_evm=bool(args.no_evm),
        )
    except ValueError as e:
        parser.error(str(e))
    os.environ["WALLET"] = args.wallet
    os.environ["HOTKEY"] = args.hotkey
    if not args.netuid:
        args.netuid = _default_netuid_for_network(args.subtensor_network)
    if not args.chain_config:
        args.chain_config = _resolve_chain_config_path(args.subtensor_network)
    if not args.netuid:
        bt.logging.error(
            "No netuid configured. Pass --netuid for custom networks or use "
            "--subtensor-network finney|test."
        )
        sys.exit(1)
    if not args.chain_config and not _env_bool("VALIDATOR_ALLOW_NO_CHAIN_CONFIG"):
        bt.logging.error(
            "No chain config found. Expected chain_config_mainnet.json, "
            "chain_config_testnet.json, or chain_config.json in the repo root. "
            "Pass --chain-config only for custom deployments, or set "
            "VALIDATOR_ALLOW_NO_CHAIN_CONFIG=1 for local route tests."
        )
        sys.exit(1)

    os.environ["NETUID"] = str(args.netuid)
    os.environ["SUBTENSOR_NETWORK"] = args.subtensor_network
    if args.subtensor_endpoint:
        os.environ["SUBTENSOR_ENDPOINT"] = args.subtensor_endpoint
        os.environ["NODEXO_SUBTENSOR_ENDPOINT"] = args.subtensor_endpoint
    os.environ["VALIDATOR_PORT"] = str(args.port)
    os.environ["VALIDATOR_BIND_HOST"] = args.bind_host
    os.environ["NODEXO_VALIDATOR_DISCOVERY_MODE"] = args.discovery_mode
    if args.no_evm:
        os.environ["VALIDATOR_NO_EVM"] = "1"
    if args.chain_config:
        os.environ["CHAIN_CONFIG"] = args.chain_config
    if args.endpoint:
        os.environ["VALIDATOR_ENDPOINT"] = args.endpoint
    return args


def _validate_validator_endpoint(endpoint: str) -> None:
    """Reject validator URLs that miners cannot reach in production."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            f"VALIDATOR_ENDPOINT must be an absolute http(s) URL, got {endpoint!r}"
        )

    host = (parsed.hostname or "").strip().lower()
    allow_local = _env_bool("VALIDATOR_ALLOW_LOCAL_ENDPOINT")
    strict_public = _env_bool("VALIDATOR_STRICT_PUBLIC_ENDPOINT")

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
            "VALIDATOR_ENDPOINT advertises a local-only address. Set "
            "VALIDATOR_ENDPOINT/VALIDATOR_PUBLIC_URL to a miner-reachable URL, "
            "or set VALIDATOR_ALLOW_LOCAL_ENDPOINT=1 for local development."
        )

    if strict_public:
        if parsed.scheme != "https":
            raise RuntimeError("VALIDATOR_STRICT_PUBLIC_ENDPOINT=1 requires an https URL")
        if private_literal:
            raise RuntimeError(
                "VALIDATOR_STRICT_PUBLIC_ENDPOINT=1 rejects private/local IP literals"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validator startup/shutdown lifecycle."""
    global analyzer, verify_pool, verification_queue

    from neurons.validator.proof.analyzer import ProofAnalyzer
    from neurons.validator.api.routes import proofs, browse, rent

    # ── Database ───────────────────────────────────────────────
    from common.db import Database, default_db_url
    db_url = (
        os.environ.get("DB_URL")
        or os.environ.get("NODEXO_VALIDATOR_DB_URL")
        or default_db_url()
    )
    _validate_validator_db_url(db_url)
    db = Database(db_url)
    app.state.db = db
    app.state.validator_registry_client = None

    # ── Initialize analyzer ────────────────────────────────────
    epoch_seconds = int(os.environ.get("PROOF_EPOCH_BLOCKS", "15")) * 12
    analyzer = ProofAnalyzer(expected_period_sec=epoch_seconds, db=db)

    # ── Initialize queue ───────────────────────────────────────
    follower_mode = (
        _env_bool("VALIDATOR_FOLLOWER_MODE")
        or _env_bool("NODEXO_VALIDATOR_FOLLOWER_MODE")
    )
    no_evm_mode = _validator_no_evm_enabled()
    app.state.no_evm_mode = no_evm_mode
    if no_evm_mode:
        bt.logging.info(
            "Validator no-EVM mode enabled: proof verification and weight "
            "setting remain active; EVM registration, rental control, "
            "canaries, and offline reports are disabled."
        )
    if follower_mode:
        verification_queue = None
    else:
        queue_size = int(os.environ.get("VERIFY_QUEUE_SIZE", "2000"))
        verification_queue = asyncio.PriorityQueue(maxsize=queue_size)

    # Wire up API route dependencies
    proofs.analyzer = analyzer
    proofs.verification_queue = verification_queue
    proofs.follower_mode = follower_mode
    rent.evm_write_enabled = not no_evm_mode
    # proofs.registry_client is set below after registry client creation

    # ── Registry client (for browse/rent endpoints) ────────────
    chain_config_path = os.environ.get("CHAIN_CONFIG", "")
    chain_endpoint = os.environ.get("SUBTENSOR_ENDPOINT") or os.environ.get("NODEXO_SUBTENSOR_ENDPOINT", "")
    evm_key = None
    if chain_config_path and os.path.exists(chain_config_path):
        from common.config import ChainConfig
        from common.chain.compute_registry import ComputeRegistryClient
        chain_config = ChainConfig.from_json(
            chain_config_path,
            subtensor_network=os.environ.get("SUBTENSOR_NETWORK", "finney"),
            chain_endpoint=chain_endpoint,
        )
        # Derive the validator's EVM private key from the bittensor hotkey seed
        # (same derivation as the on-chain ValidatorRegistry registration above).
        # Required for write paths like markRented / markAvailable; reads work
        # either way. Falls back to VALIDATOR_EVM_KEY env var for dev/testing.
        if no_evm_mode:
            bt.logging.info("ComputeRegistry client initialized read-only (--no-evm)")
        else:
            evm_key = os.environ.get("VALIDATOR_EVM_KEY") or None
        if not evm_key and not no_evm_mode:
            try:
                from common.chain.wallet import load_hotkey_seed, derive_evm_account
                _wname = os.environ.get("WALLET", "validator")
                _hname = os.environ.get("HOTKEY", "default")
                evm_key = derive_evm_account(load_hotkey_seed(_wname, _hname)).key.hex()
            except Exception as e:
                bt.logging.warning(
                    f"Validator EVM key derivation failed: {e}; "
                    "rental/control EVM writes disabled"
                )
                rent.evm_write_enabled = False
        registry = ComputeRegistryClient(chain_config, private_key=evm_key)
        from neurons.validator.api.routes import heartbeat
        browse.registry_client = registry
        browse.db_instance = db
        rent.registry_client = registry
        proofs.registry_client = registry  # For registered-executor verification
        heartbeat.registry_client = registry
        heartbeat.db_instance = db
        rent.db = db  # For rental persistence
        # Hook DB into all routes that need persistence (audit C-7).
        try:
            from neurons.validator.api.routes import monitor as monitor_route
            monitor_route.db_instance = db
            monitor_route.load_monitor_pubkeys_from_db()
        except Exception as e:
            bt.logging.warning(f"monitor_route DB wire failed: {e}")
        try:
            proofs.db_instance = db
            proofs._load_executor_bindings_from_db()
        except Exception as e:
            bt.logging.warning(f"proofs route DB wire failed: {e}")
        # rental_state needs DB injection too; rebuild it with the db
        # and rehydrate the timeline from RentalWindowRow.
        try:
            from neurons.validator.rental.state import RentalState
            rent.rental_state = RentalState(db=db)
            rent.rental_state.load_from_db()
        except Exception as e:
            bt.logging.warning(f"rental_state DB wire failed: {e}")
        bt.logging.info(f"Registry client initialized (chain_config={chain_config_path})")
    else:
        from neurons.validator.api.routes import heartbeat, browse
        heartbeat.db_instance = db
        browse.db_instance = db  # /instances works without chain config
        bt.logging.warning("No CHAIN_CONFIG set — on-chain browse/rent unavailable, local /instances works")

    # ── Subtensor RPC + wallet (pre-initialized in __main__ before uvicorn) ──
    # These env reads moved ahead of the rental orchestrator construction
    # so the orchestrator's hotkey_seed load can use them. Previously a
    # forward-reference here logged "wallet_name referenced before
    # assignment" and the orchestrator went out signing-disabled, which
    # made all miner-bound requests 401 against the auth-enabled miner.
    subtensor_network = os.environ.get("SUBTENSOR_NETWORK", "finney")
    subtensor_endpoint = chain_endpoint
    netuid = int(os.environ.get("NETUID", "0"))
    wallet_name = os.environ.get("WALLET", "validator")
    hotkey_name = os.environ.get("HOTKEY", "default")

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

    # ── Rental orchestrator ────────────────────────────────────
    # Loaded with the validator's hotkey seed so every miner-bound
    # request (POST /containers, DELETE /containers/{name}) carries an
    # SR25519 signature. Without the seed the orchestrator sends
    # unsigned requests and any auth-enabled miner will 401.
    from neurons.validator.rental.orchestrator import RentalOrchestrator
    _orch_seed = None
    try:
        from common.chain.wallet import load_hotkey_seed
        _orch_seed = load_hotkey_seed(wallet_name, hotkey_name)
    except Exception as e:
        bt.logging.warning(f"Orchestrator hotkey_seed load failed: {e}")
    rent.rental_orchestrator = RentalOrchestrator(hotkey_seed=_orch_seed)

    if getattr(app, '_vali_pre_rpc', None):
        rpc = app._vali_pre_rpc
        wallet = app._vali_pre_wallet
        bt.logging.info(f"Subtensor loaded (pre-initialized, netuid={netuid})")
    else:
        # Fallback: init here (will block event loop — only for dev/testing)
        from common.chain.rpc import SubtensorRPC
        WalletCls = getattr(bt, "Wallet", None) or bt.wallet
        rpc = SubtensorRPC(
            network=subtensor_network,
            netuid=netuid,
            chain_endpoint=subtensor_endpoint,
        )
        wallet = WalletCls(name=wallet_name, hotkey=hotkey_name)
        bt.logging.warning("Subtensor initialized inline (may block event loop)")

    # ── EVM + validator registration ──────────────────────────
    # Two-step dance (mirrors the miner's ComputeRegistry flow):
    #   1. registerEvm(uid, sigR, sigS) — proves we own the hotkey for this UID
    #   2. register(proxy_endpoint)      — activates as a validator
    # `register` reverts hard if `evmRegistered[caller]` is false, so step 1
    # MUST happen first. Previously the validator skipped step 1 and the
    # register tx silently reverted (no receipt-status check), leaving the
    # validator invisible to miners' allowlist discovery.
    if no_evm_mode and chain_config_path and os.path.exists(chain_config_path):
        bt.logging.info(
            "Skipping ValidatorRegistry and ComputeRegistry EVM registration "
            "because VALIDATOR_NO_EVM/--no-evm is enabled"
        )
    elif chain_config_path and os.path.exists(chain_config_path):
        try:
            from common.chain.wallet import derive_evm_account, load_hotkey_seed, sign_evm_registration
            from common.chain.validator_registry import ValidatorRegistryClient

            hotkey_seed = load_hotkey_seed(wallet_name, hotkey_name)
            evm_account = derive_evm_account(hotkey_seed)

            vali_registry = ValidatorRegistryClient(chain_config, private_key=evm_account.key.hex())

            # Resolve UID/permit with lightweight subtensor calls. A full
            # metagraph fetch is heavier and is easier to rate-limit on testnet.
            from common.chain.rpc import SubtensorRPC as _RPC
            _rpc = app._vali_pre_rpc if getattr(app, '_vali_pre_rpc', None) else rpc
            wallet_ss58 = wallet.hotkey.ss58_address
            uid = await _subtensor_call_with_retry(
                "validator startup UID lookup",
                lambda: _rpc.subtensor.get_uid_for_hotkey_on_subnet(
                    wallet_ss58, netuid=int(netuid),
                ),
            )
            if uid is None:
                raise RuntimeError(
                    f"Validator hotkey {wallet_ss58[:12]}... is not registered "
                    f"on subnet {netuid}; EVM endpoint mode cannot start."
                )
            validator_permits = await _subtensor_call_with_retry(
                "validator startup permit lookup",
                lambda: _rpc.subtensor.get_subnet_validator_permits(netuid=int(netuid)),
            )
            try:
                has_permit = bool(validator_permits[int(uid)])
            except Exception:
                has_permit = False
            if not has_permit:
                raise RuntimeError(
                    f"UID {uid} does not have validator permit on subnet {netuid}. "
                    "Miner validator discovery is permit-gated for both "
                    "ValidatorRegistry and native axon paths; EVM write mode "
                    "cannot activate registry/control authority without permit. "
                    "Use VALIDATOR_NO_EVM=1 for local health/readiness checks "
                    "until the hotkey has permit."
                )
            else:
                # Step 1: ensure EVM is bound to this UID on ValidatorRegistry.
                # Each registry (Compute/Validator) has its own evmRegistered table -
                # being registered on one doesn't carry over to the other.
                try:
                    if not vali_registry.is_evm_registered_for_uid(evm_account.address, uid):
                        bt.logging.info(f"Registering EVM on ValidatorRegistry for UID {uid}...")
                        sig_r, sig_s = sign_evm_registration(
                            hotkey_seed, evm_account.address, uid, netuid,
                            vali_registry.contract.address,
                        )
                        vali_registry.register_evm(uid, sig_r, sig_s)
                        bt.logging.info("EVM registered on ValidatorRegistry")
                    else:
                        bt.logging.info(f"ValidatorRegistry EVM binding already current for UID {uid}")
                except Exception as e:
                    raise RuntimeError(f"ValidatorRegistry EVM registration failed: {e}") from e

                # Step 2: register endpoint (idempotent - also updates if changed).
                validator_port = int(os.environ.get("VALIDATOR_PORT", "8443"))
                proxy_endpoint = os.environ.get("VALIDATOR_ENDPOINT", f"http://127.0.0.1:{validator_port}")
                _validate_validator_endpoint(proxy_endpoint)
                try:
                    current_validator = vali_registry.get_validator(evm_account.address)
                    if (
                        current_validator.is_active
                        and int(current_validator.uid) == int(uid)
                        and current_validator.proxy_endpoint == proxy_endpoint
                    ):
                        bt.logging.info(f"ValidatorRegistry endpoint already current: {proxy_endpoint}")
                    elif current_validator.is_active and int(current_validator.uid) == int(uid):
                        vali_registry.update_endpoint(proxy_endpoint)
                        bt.logging.info(
                            "ValidatorRegistry endpoint updated: "
                            f"{current_validator.proxy_endpoint} -> {proxy_endpoint}"
                        )
                    else:
                        vali_registry.register(proxy_endpoint)
                        bt.logging.info(f"Registered on ValidatorRegistry: {proxy_endpoint}")
                    app.state.validator_registry_client = vali_registry
                except Exception as e:
                    raise RuntimeError(f"ValidatorRegistry endpoint registration failed: {e}") from e

                # Step 3: ALSO registerEvm on ComputeRegistry. markRented /
                # markAvailable / reportOffline all carry the onlyValidator
                # modifier which checks ComputeRegistry's *own* evmRegistered
                # table - being registered on ValidatorRegistry doesn't
                # carry over. Without this, the first rental's markRented
                # silently reverts ("Not EVM-registered" inside the modifier)
                # and the executor's is_rented flag stays false on chain.
                try:
                    from common.chain.compute_registry import ComputeRegistryClient
                    compute_reg = ComputeRegistryClient(chain_config, private_key=evm_account.key.hex())
                    if (
                        not compute_reg.is_evm_registered(evm_account.address)
                        or int(compute_reg.evm_to_uid(evm_account.address)) != int(uid)
                    ):
                        bt.logging.info(f"Registering EVM on ComputeRegistry for UID {uid}...")
                        sig_r, sig_s = sign_evm_registration(
                            hotkey_seed, evm_account.address, uid, netuid,
                            compute_reg.contract.address,
                        )
                        compute_reg.register_evm(uid, sig_r, sig_s)
                        bt.logging.info("EVM registered on ComputeRegistry (markRented now works)")
                    else:
                        bt.logging.info(f"ComputeRegistry EVM binding already current for UID {uid}")
                except Exception as e:
                    raise RuntimeError(f"ComputeRegistry EVM registration failed: {e}") from e
        except Exception as e:
            message = f"Validator chain registration failed: {e}"
            bt.logging.error(message)
            if _env_bool("VALIDATOR_FORCE_EXIT_ON_STARTUP_FAILURE", "1"):
                print(message, file=sys.stderr, flush=True)
                os._exit(1)
            bt.logging.warning(
                "Continuing after validator chain registration failure because "
                "VALIDATOR_FORCE_EXIT_ON_STARTUP_FAILURE=0. EVM discovery and "
                "rental control may be stale until the next successful restart "
                "or operator reconciliation."
            )

    # Store for weight setting
    app.state.rpc = rpc
    app.state.wallet = wallet
    app.state.netuid = netuid
    proofs.rpc_client = rpc
    proofs.verification_priority_salt = getattr(wallet.hotkey, "ss58_address", "") or wallet_name

    # Chain anchor for cycle-start-ts estimation (validator-external
    # latency signal). At lifespan startup we sample the current
    # block + wall-clock time; thereafter we extrapolate cycle starts
    # by `anchor_ts + (cycle_start_block - anchor_block) * 12`. Drift
    # accumulates slowly but is small over the rolling-30-cycle window
    # the sybil scanner uses; refresh on long-lived runs is fine to
    # add later if needed.
    try:
        anchor_block = int(rpc.get_current_block())
        app.state.chain_anchor = (anchor_block, time.time())
        bt.logging.info(
            f"Chain anchor: block={anchor_block} at {app.state.chain_anchor[1]:.0f}"
        )
    except Exception as e:
        bt.logging.warning(f"Chain anchor failed: {e}; arrival_latency disabled")
        app.state.chain_anchor = None

    beacon_cache_task = None
    app.state.beacon_cache = None
    if not follower_mode and os.environ.get("NODEXO_BEACON_CACHE_ENABLED", "1") == "1":
        beacon_cache = RollingBeaconCache(rpc)
        app.state.beacon_cache = beacon_cache
        beacon_cache_task = asyncio.create_task(beacon_cache.run())

    # ── Process pool for proof verification ────────────────────
    if follower_mode:
        verify_pool = None
        bt.logging.info("Follower validator mode enabled: local proof verification disabled")
    else:
        # Reserve CPU for HTTP ingress, chain/cache tasks, and Postgres. Proof
        # verification is CPU-bound and can spike at cycle boundaries; operators
        # with dedicated validator hardware can override via VERIFY_WORKERS.
        num_workers = int(os.environ.get("VERIFY_WORKERS") or _default_verify_workers())

        verify_pool = _create_verify_pool(num_workers)
        bt.logging.info(
            f"Verification pool: {num_workers} workers "
            f"(start_method={_verification_worker_start_method()}, "
            f"worker_nice={os.environ.get('VERIFY_WORKER_NICE', '5')}, "
            "native_threads=1, with pre-import)"
        )

    # Resolve the UID for the periodic metagraph stats line in the background.
    # Testnet RPC can stall here; keeping it out of startup lets health and
    # proof ingestion come online first.
    _stats_wallet_ss58 = wallet.hotkey.ss58_address if rpc and wallet else None
    metagraph_warmup_task = asyncio.create_task(_metagraph_hotkey_cache_warmup(rpc))

    # ── Background tasks ───────────────────────────────────────
    verify_task = None
    scoring_task = None
    follower_task = None
    liveness_task = None
    if follower_mode:
        from neurons.validator.follower import run_follower_state_loop

        def _apply_follower_scores(scores: dict):
            global executor_scores
            executor_scores = scores
            browse.scoring_data = scores
            rent.scoring_data = scores

        follower_task = asyncio.create_task(
            run_follower_state_loop(_apply_follower_scores)
        )
    else:
        verify_task = asyncio.create_task(_verification_loop())
        scoring_task = asyncio.create_task(_scoring_loop())
    prune_task = asyncio.create_task(_prune_loop())
    weight_task = asyncio.create_task(_weight_setting_loop(rpc, wallet, netuid))
    mg_log_task = asyncio.create_task(
        _metagraph_stats_loop(rpc, None, interval=180, wallet_ss58=_stats_wallet_ss58)
    )
    auto_updater = None
    if getattr(app.state, "auto_update_enabled", False):
        from neurons.auto_update import AutoUpdater

        def _validator_busy() -> bool:
            try:
                return bool(verification_queue and verification_queue.qsize() > 0)
            except Exception:
                return True

        auto_updater = AutoUpdater(
            role="validator",
            check_interval=int(getattr(app.state, "auto_update_interval", 1800)),
            restart_delay=int(getattr(app.state, "auto_update_restart_delay", 5)),
            busy_check=_validator_busy,
        )
        auto_updater.start()

    # ChainSnapshot — single chain poller that every read path uses. Without
    # this, /marketplace, /rent, /executors all hit chain on every request,
    # cascading 429s under load and tying API responsiveness to chain health.
    chain_snapshot_task = None
    rental_event_index_task = None
    ttl_sweep_task = None
    orphan_sweep_task = None
    rental_watchdog_task = None
    sybil_scan_task = None
    offline_report_task = None
    endpoint_health_task = None
    endpoint_health = None
    if browse.registry_client is not None and not follower_mode:
        from neurons.validator.state.chain_snapshot import ChainSnapshot
        # Give the snapshot its OWN registry client so its periodic
        # `get_all_active_executors` (~paginated 2-3 chain reads) doesn't
        # share Web3 session state with the high-throughput proof
        # verification path. Sharing a single Web3 HTTPProvider across
        # threads created connection-pool issues that silently returned
        # empty data — diagnosed during testnet rent E2E.
        try:
            from common.chain.compute_registry import ComputeRegistryClient
            from common.config import ChainConfig as _CC
            snap_registry = ComputeRegistryClient(
                _CC.from_json(
                    chain_config_path,
                    subtensor_network=subtensor_network,
                    chain_endpoint=subtensor_endpoint,
                ),
                private_key=evm_key,
            )
        except Exception as e:
            bt.logging.warning(
                f"snapshot dedicated registry init failed: {e}; "
                f"falling back to shared client"
            )
            snap_registry = browse.registry_client
        try:
            from common.config import ChainConfig as _CC
            from neurons.validator.rental.event_indexer import ComputeRegistryEventIndexer
            _snapshot_cfg = _CC.from_json(
                chain_config_path,
                subtensor_network=subtensor_network,
                chain_endpoint=subtensor_endpoint,
            )
            registry_cursor_key = ComputeRegistryEventIndexer.cursor_key_for(
                _snapshot_cfg.chain_id,
                getattr(snap_registry.contract, "address", ""),
            )
        except Exception:
            registry_cursor_key = ""
        snapshot = ChainSnapshot(
            snap_registry,
            db=db,
            registry_cursor_key=registry_cursor_key,
        )
        app.state.chain_snapshot = snapshot
        # Inject into routes that need it
        browse.chain_snapshot = snapshot
        rent.chain_snapshot = snapshot
        proofs.chain_snapshot = snapshot
        chain_snapshot_task = asyncio.create_task(snapshot.run())

        if os.environ.get("ENDPOINT_HEALTH_ENABLED", "1") == "1":
            try:
                from neurons.validator.health.prober import EndpointHealthProber
                endpoint_health = EndpointHealthProber(snapshot)
                browse.endpoint_health = endpoint_health
                endpoint_health_task = asyncio.create_task(endpoint_health.run())
            except Exception as e:
                bt.logging.warning(f"EndpointHealthProber failed to start: {e}")

        registry_index_enabled = (
            os.environ.get(
                "REGISTRY_EVENT_INDEXER_ENABLED",
                os.environ.get("RENTAL_EVENT_INDEXER_ENABLED", "1"),
            ) == "1"
        )
        if registry_index_enabled:
            try:
                from common.chain.compute_registry import ComputeRegistryClient
                from common.config import ChainConfig as _CC
                from neurons.validator.rental.event_indexer import ComputeRegistryEventIndexer

                index_cfg = _CC.from_json(
                    chain_config_path,
                    subtensor_network=subtensor_network,
                    chain_endpoint=subtensor_endpoint,
                )
                index_registry = ComputeRegistryClient(index_cfg, private_key=evm_key)
                indexer = ComputeRegistryEventIndexer(
                    index_registry,
                    db,
                    rent.rental_state,
                    chain_id=index_cfg.chain_id,
                    deploy_block=index_cfg.compute_registry_deploy_block,
                    interval_s=float(os.environ.get(
                        "REGISTRY_EVENT_INDEX_INTERVAL_S",
                        os.environ.get("RENTAL_EVENT_INDEX_INTERVAL_S", "10"),
                    )),
                    batch_blocks=int(os.environ.get(
                        "REGISTRY_EVENT_INDEX_BATCH_BLOCKS",
                        os.environ.get("RENTAL_EVENT_INDEX_BATCH_BLOCKS", "2000"),
                    )),
                    confirmations=int(os.environ.get(
                        "REGISTRY_EVENT_INDEX_CONFIRMATIONS",
                        os.environ.get("RENTAL_EVENT_INDEX_CONFIRMATIONS", "6"),
                    )),
                    reconcile_interval_s=float(os.environ.get(
                        "REGISTRY_EVENT_INDEX_RECONCILE_INTERVAL_S", "300",
                    )),
                    reconcile_page_size=int(os.environ.get(
                        "REGISTRY_EVENT_INDEX_RECONCILE_PAGE_SIZE",
                        os.environ.get("CHAIN_SNAPSHOT_PAGE_SIZE", "500"),
                    )),
                )
                rental_event_index_task = asyncio.create_task(indexer.run())
            except Exception as e:
                bt.logging.warning(f"ComputeRegistryEventIndexer failed to start: {e}")

        if rent.evm_write_enabled:
            # Reconcile in-memory rental state with DB + chain after restart.
            # Without this, an unterminated rental sticks the executor as
            # is_rented=True forever until manually fixed (we hit this multiple
            # times today). Reconciler runs once at boot; TTL sweeper handles
            # the steady-state.
            try:
                await rent.reconcile_on_startup()
            except Exception as e:
                bt.logging.warning(f"Rental reconcile failed: {e}")
            ttl_sweep_task = asyncio.create_task(rent.ttl_sweeper(interval_s=60))
            # Dropped from 300s → 60s so a stuck rented executor (orphan from a
            # partial-failure rent or a peer validator's stale lock) becomes
            # available again within a minute instead of up to five.
            orphan_sweep_task = asyncio.create_task(rent.orphan_sweeper(interval_s=60))
            if os.environ.get("RENTAL_CONTAINER_WATCHDOG_ENABLED", "1") == "1":
                rental_watchdog_task = asyncio.create_task(
                    rent.rental_container_watchdog(
                        interval_s=float(os.environ.get(
                            "RENTAL_CONTAINER_WATCHDOG_INTERVAL_S", "30"
                        )),
                    )
                )
        else:
            bt.logging.info(
                "Rental reconcile/watchdog/sweepers disabled because validator "
                "has no EVM write mode"
            )

        # Sybil scanner — correlates heartbeat hardware fingerprints
        # across executors and force-bans confirmed sybils. Runs every
        # 10 min; cheap (single DB scan). Catches the double-rent
        # attack via gpu_uuid + endpoint_ip + system_uuid collisions.
        from neurons.validator.sybil.scanner import sybil_scanner_loop
        from common.db import get_active_rentals as _get_active_rentals
        sybil_scan_task = asyncio.create_task(sybil_scanner_loop(
            db=db,
            get_active_rentals_callable=lambda: _get_active_rentals(db),
            terminate_callable=(
                rent._terminate_rental_internal if rent.evm_write_enabled else None
            ),
            chain_snapshot=snapshot,
            orchestrator=rent.rental_orchestrator,
        ))
        if rent.evm_write_enabled:
            offline_report_task = asyncio.create_task(_offline_report_loop(
                db=db,
                registry=snap_registry,
                chain_snapshot=snapshot,
                endpoint_health=endpoint_health,
            ))
        else:
            bt.logging.info("Offline report loop disabled because validator has no EVM write mode")

    # ── Canary scheduler: validator-initiated VRAM-fill probes ──
    # Optional. Disabled by default until we explicitly opt in via
    # env, so first-deploy validators don't immediately start
    # canary-renting every executor on the network.
    canary_task = None
    if (follower_mode or no_evm_mode) and os.environ.get("CANARY_ENABLED", "0") == "1":
        reason = "follower validator mode" if follower_mode else "no-EVM validator mode"
        bt.logging.warning(f"CANARY_ENABLED ignored in {reason}")
    if not follower_mode and not no_evm_mode and os.environ.get("CANARY_ENABLED", "0") == "1":
        try:
            from neurons.validator.canary.runner import CanaryRunner
            from neurons.validator.canary.scheduler import (
                CanaryScheduler, reconcile_canary_inflight_async,
            )
            # Reconcile any markRented orphans from a crashed canary,
            # but DON'T block startup on it — chain RPCs are seconds
            # each and a mass-crash could leave 10+ orphans; serving
            # /health and /marketplace matters more than instantly
            # releasing the orphans. The reconcile is capped per boot
            # (RECONCILE_MAX_PER_BOOT); anything over the cap settles
            # naturally as canaries tick.
            asyncio.create_task(
                reconcile_canary_inflight_async(db, rent.registry_client)
            )
            canary_runner = CanaryRunner(
                registry_client=rent.registry_client,
                orchestrator=rent.rental_orchestrator,
                db=db,
            )
            def _get_active_executors():
                if rent.chain_snapshot is not None:
                    return rent.chain_snapshot.available_executors()
                if rent.registry_client is None:
                    return []
                return rent.registry_client.get_all_active_executors()
            canary_mean_s = int(os.environ.get(
                "CANARY_MEAN_SECONDS", str(24 * 3600),
            ))
            # The validator's hotkey seed is unpredictable to miners
            # (it's the validator's secret). Use it to key the
            # per-(executor, tick) Bernoulli HMAC. Falls back to
            # os.urandom() inside the scheduler if seed isn't available
            # (still unpredictable to miners; just re-randomises on
            # validator restart, which is fine — restart resets the
            # schedule too).
            try:
                from common.chain.wallet import load_hotkey_seed
                _wname = os.environ.get("WALLET", "validator")
                _hname = os.environ.get("HOTKEY", "default")
                jitter_secret = load_hotkey_seed(_wname, _hname)
            except Exception:
                jitter_secret = None
            canary_scheduler = CanaryScheduler(
                runner=canary_runner,
                get_active_executors=_get_active_executors,
                db=db,
                mean_seconds=canary_mean_s,
                jitter_secret=jitter_secret,
            )
            canary_task = asyncio.create_task(canary_scheduler.run())
            bt.logging.info(
                f"CanaryScheduler enabled (mean={canary_mean_s}s)"
            )
        except Exception as e:
            bt.logging.warning(f"Canary scheduler failed to start: {e}")

    app.state.validator_ready_at = time.time()
    liveness_service_name = os.environ.get("VALIDATOR_LIVENESS_SERVICE", "primary")
    liveness_instance_id = (
        os.environ.get("VALIDATOR_INSTANCE_ID")
        or f"{getattr(os.uname(), 'nodename', 'validator')}:{os.getpid()}:{int(app.state.validator_ready_at)}"
    )
    try:
        from common.db import record_validator_startup
        info = await asyncio.to_thread(
            record_validator_startup,
            db,
            service_name=liveness_service_name,
            instance_id=liveness_instance_id,
            ready_ts=app.state.validator_ready_at,
            stale_after_s=float(os.environ.get("VALIDATOR_OUTAGE_MIN_GAP_S", "12")),
        )
        if info.get("created_outage"):
            bt.logging.warning(
                "Validator outage recorded: "
                f"service={liveness_service_name} duration={info.get('duration_s', 0):.1f}s"
            )
    except Exception as e:
        bt.logging.warning(f"Validator outage startup record failed: {e}")
    liveness_task = asyncio.create_task(
        _validator_liveness_loop(
            db,
            service_name=liveness_service_name,
            instance_id=liveness_instance_id,
            interval_s=float(os.environ.get("VALIDATOR_LIVENESS_INTERVAL_S", "5")),
        )
    )
    bt.logging.info("Validator daemon started")
    yield

    # Shutdown
    try:
        from common.db import record_validator_shutdown
        await asyncio.to_thread(
            record_validator_shutdown,
            db,
            service_name=liveness_service_name,
        )
    except Exception:
        pass
    validator_registry_client = getattr(app.state, "validator_registry_client", None)
    if (
        validator_registry_client is not None
        and _env_bool("VALIDATOR_DEACTIVATE_ON_SHUTDOWN", "1")
    ):
        timeout_s = float(os.environ.get("VALIDATOR_DEACTIVATE_SHUTDOWN_TIMEOUT_S", "45"))
        try:
            tx = await asyncio.wait_for(
                asyncio.to_thread(validator_registry_client.deactivate),
                timeout=timeout_s,
            )
            bt.logging.info(f"ValidatorRegistry deactivated on shutdown: tx={tx}")
        except asyncio.TimeoutError:
            bt.logging.warning(
                "ValidatorRegistry deactivate timed out during shutdown; "
                "miners will quarantine the stale endpoint if it remains advertised"
            )
        except Exception as e:
            bt.logging.warning(f"ValidatorRegistry deactivate on shutdown failed: {e}")
    if auto_updater is not None:
        auto_updater.stop()
    if liveness_task is not None:
        liveness_task.cancel()
    if subnet_config_task is not None:
        subnet_config_task.cancel()
    if beacon_cache_task is not None:
        beacon_cache_task.cancel()
    if verify_task is not None:
        verify_task.cancel()
    if scoring_task is not None:
        scoring_task.cancel()
    if follower_task is not None:
        follower_task.cancel()
    prune_task.cancel()
    weight_task.cancel()
    metagraph_warmup_task.cancel()
    mg_log_task.cancel()
    if chain_snapshot_task is not None:
        chain_snapshot_task.cancel()
    if rental_event_index_task is not None:
        rental_event_index_task.cancel()
    if ttl_sweep_task is not None:
        ttl_sweep_task.cancel()
    if orphan_sweep_task is not None:
        orphan_sweep_task.cancel()
    if rental_watchdog_task is not None:
        rental_watchdog_task.cancel()
    if sybil_scan_task is not None:
        sybil_scan_task.cancel()
    if offline_report_task is not None:
        offline_report_task.cancel()
    if endpoint_health_task is not None:
        endpoint_health_task.cancel()
    if canary_task is not None:
        canary_task.cancel()
    if verify_pool is not None:
        verify_pool.shutdown(wait=False)
    bt.logging.info("Validator daemon stopped")


app = FastAPI(title="Nodexo Validator", version="0.1.0", lifespan=lifespan)

# ── Mount API routes ───────────────────────────────────────────
from neurons.validator.api.routes.proofs import router as proofs_router
from neurons.validator.api.routes.browse import router as browse_router
from neurons.validator.api.routes.rent import router as rent_router
from neurons.validator.api.routes.heartbeat import router as heartbeat_router
from neurons.validator.api.routes.monitor import router as monitor_router

app.include_router(proofs_router)
app.include_router(browse_router)
app.include_router(rent_router)
app.include_router(heartbeat_router)
app.include_router(monitor_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "nodexo-validator",
        "no_evm": bool(getattr(app.state, "no_evm_mode", False)),
    }


@app.get("/subnet-config")
async def subnet_config():
    from common.subnet_config_client import read_cached_or_bundled_subnet_config

    data = read_cached_or_bundled_subnet_config()
    if data is None:
        raise HTTPException(status_code=503, detail="subnet config cache unavailable")
    return data


@app.get("/version")
async def version():
    from neurons.version import (
        spec_version,
        validator_version,
        validator_version_str,
        version_str,
    )
    return {
        "service": "nodexo-validator",
        "version": version_str,
        "spec_version": spec_version,
        "validator_version": validator_version,
        "validator_version_str": validator_version_str,
    }


@app.get("/scores")
async def get_scores(request: Request):
    """Get current executor scores (transparency endpoint)."""
    from neurons.validator.api.routes.rent import _is_admin_async
    if not await _is_admin_async(request):
        raise HTTPException(403, "internal validator auth required")
    if analyzer is None:
        return {"scores": {}}
    all_scores = analyzer.get_all_scores()
    return {
        "scores": {
            eid: {
                "pass_rate_1h": s.pass_rate_1h,
                "pass_rate_24h": s.pass_rate_24h,
                "avg_delta": s.avg_delta,
                "status": s.status,
                "trust_level": s.trust_level,
            }
            for eid, s in all_scores.items()
        }
    }


# ── Background loops ──────────────────────────────────────────

_verified_epochs: set[tuple[str, int]] = set()  # (executor_id, epoch_id) dedup


_verify_start_times: dict[tuple[str, int], float] = {}  # Track in-flight verifications
VERIFY_LAG_BLOCKS = 8     # Wait this many blocks past the proof block before verifying
VERIFY_TIMEOUT_SEC = 90   # Kill stuck verifications
_chain_ready_tasks: dict[tuple[int, int], asyncio.Task[int]] = {}
_beacon_task_async_lock: asyncio.Lock | None = None
_beacon_task_async_lock_loop: asyncio.AbstractEventLoop | None = None
_beacon_run_async_lock: asyncio.Lock | None = None
_beacon_run_async_lock_loop: asyncio.AbstractEventLoop | None = None
_beacon_lookup_cooldown_until = 0.0
_beacon_context_cache: dict[tuple[str, int, int, int, int], tuple[int, bytes, float, int]] = {}
_beacon_context_tasks: dict[tuple[str, int, int, int, int], asyncio.Task[tuple[int, bytes, float, int]]] = {}
_BEACON_CONTEXT_CACHE_MAX = 2048


def _verify_lag_blocks() -> int:
    return max(0, int(os.environ.get("VERIFY_LAG_BLOCKS", str(VERIFY_LAG_BLOCKS))))


def _chain_ready_wait_timeout_s(max_beacon_offset_blocks: int) -> float:
    configured = os.environ.get("VERIFY_CHAIN_WAIT_TIMEOUT_S")
    if configured:
        return max(1.0, float(configured))
    # A receipt can arrive before the selected beacon + verifier lag has
    # reached the local node. The default covers the nominal block span plus
    # enough slack for testnet/mainnet block jitter without tying up tasks
    # indefinitely during a stalled chain.
    nominal = (max(1, max_beacon_offset_blocks) + _verify_lag_blocks() + 10) * 12
    return float(max(240, nominal))


def _chain_ready_poll_s() -> float:
    return max(0.1, float(os.environ.get("VERIFY_CHAIN_WAIT_POLL_S", "6")))


def _chain_ready_near_miss_blocks() -> int:
    return max(0, int(os.environ.get("VERIFY_CHAIN_WAIT_NEAR_MISS_BLOCKS", "2")))


def _chain_ready_near_miss_grace_s() -> float:
    return max(0.0, float(os.environ.get("VERIFY_CHAIN_WAIT_NEAR_MISS_GRACE_S", "90")))


def _chain_head_call_timeout_s() -> float:
    return max(0.001, float(os.environ.get("VERIFY_CHAIN_HEAD_CALL_TIMEOUT_S", "10")))


def _beacon_lookup_timeout_s() -> float:
    return max(0.001, float(os.environ.get("VERIFY_BEACON_LOOKUP_TIMEOUT_S", "90")))


def _beacon_lookup_attempts() -> int:
    return max(1, int(os.environ.get("VERIFY_BEACON_LOOKUP_ATTEMPTS", "3")))


def _beacon_lookup_initial_backoff_s() -> float:
    return max(0.0, float(os.environ.get("VERIFY_BEACON_LOOKUP_INITIAL_BACKOFF_S", "2")))


def _beacon_lookup_max_backoff_s() -> float:
    return max(0.0, float(os.environ.get("VERIFY_BEACON_LOOKUP_MAX_BACKOFF_S", "8")))


def _beacon_lookup_failure_cooldown_s() -> float:
    return max(0.0, float(os.environ.get("VERIFY_BEACON_LOOKUP_FAILURE_COOLDOWN_S", "15")))


def _beacon_task_lock() -> asyncio.Lock:
    global _beacon_task_async_lock, _beacon_task_async_lock_loop
    loop = asyncio.get_running_loop()
    if _beacon_task_async_lock is None or _beacon_task_async_lock_loop is not loop:
        _beacon_task_async_lock = asyncio.Lock()
        _beacon_task_async_lock_loop = loop
        _beacon_context_tasks.clear()
    return _beacon_task_async_lock


def _beacon_run_lock() -> asyncio.Lock:
    global _beacon_run_async_lock, _beacon_run_async_lock_loop
    loop = asyncio.get_running_loop()
    if _beacon_run_async_lock is None or _beacon_run_async_lock_loop is not loop:
        _beacon_run_async_lock = asyncio.Lock()
        _beacon_run_async_lock_loop = loop
    return _beacon_run_async_lock


def _chain_unavailable_requeue_attempts() -> int:
    return max(0, int(os.environ.get("VERIFY_CHAIN_UNAVAILABLE_REQUEUE_ATTEMPTS", "4")))


def _chain_unavailable_requeue_initial_delay_s() -> float:
    return max(0.0, float(os.environ.get("VERIFY_CHAIN_UNAVAILABLE_REQUEUE_INITIAL_DELAY_S", "60")))


def _chain_unavailable_requeue_max_delay_s() -> float:
    return max(0.0, float(os.environ.get("VERIFY_CHAIN_UNAVAILABLE_REQUEUE_MAX_DELAY_S", "480")))


def _stale_verification_skip_after_epochs() -> int:
    return max(1, int(os.environ.get("VERIFY_STALE_SKIP_AFTER_EPOCHS", "2")))


def _fresh_current_block(network: str) -> int:
    subtensor_cls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
    fresh = subtensor_cls(network=network)
    try:
        return int(fresh.get_current_block())
    finally:
        closer = getattr(fresh, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def _beacon_context_cache_key(
    rpc_ref,
    epoch_id: int,
    epoch_interval: int,
    max_beacon_offset_blocks: int,
) -> tuple[str, int, tuple[str, int, int, int, int]]:
    network = str(getattr(rpc_ref, "network", "") or os.environ.get("SUBTENSOR_NETWORK", "finney"))
    netuid = int(getattr(rpc_ref, "netuid", 0) or os.environ.get("NETUID", "0"))
    return (
        network,
        netuid,
        (
            network,
            netuid,
            int(epoch_id),
            int(epoch_interval),
            int(max_beacon_offset_blocks),
        ),
    )


def _beacon_cache_result_for_head(
    cache_key: tuple[str, int, int, int, int],
    current_head: int,
) -> tuple[int, bytes, float, int] | None:
    cached = _beacon_context_cache.get(cache_key)
    if cached is None:
        return None
    cached_head, beacon, beacon_ts, beacon_block = cached
    return max(int(current_head), int(cached_head)), beacon, beacon_ts, beacon_block


def _lookup_beacon_context_sync(
    rpc_ref,
    epoch_id: int,
    epoch_interval: int,
    max_beacon_offset_blocks: int,
    current_head: int,
) -> tuple[int, bytes, float, int]:
    import time as _t

    from common.chain.rpc import SubtensorRPC
    from common.proof_schedule import derive_beacon_offset_blocks

    network, netuid, cache_key = _beacon_context_cache_key(
        rpc_ref, epoch_id, epoch_interval, max_beacon_offset_blocks,
    )
    cached = _beacon_cache_result_for_head(cache_key, current_head)
    if cached is not None:
        return cached

    attempts = _beacon_lookup_attempts()
    delay_s = _beacon_lookup_initial_backoff_s()
    max_delay_s = _beacon_lookup_max_backoff_s()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            # current_head was observed before this function is called. Chain
            # hash lookups can block for many seconds, so anchor timing before
            # slow hash reads. When possible, refresh the head in this same
            # timed lookup so retries do not reuse a stale head/timestamp pair.
            head_anchor_ts = _t.time()
            fresh = SubtensorRPC(network=network, netuid=netuid)
            try:
                observed_head = int(current_head)
                try:
                    refreshed_head = int(fresh.get_current_block())
                    observed_head = max(observed_head, refreshed_head)
                    head_anchor_ts = _t.time()
                except Exception as e:
                    bt.logging.debug(f"Beacon lookup head refresh failed; using ready head: {e}")
                cycle_start = int(epoch_id) * int(epoch_interval)
                anchor = fresh.get_block_hash(cycle_start)
                offset = derive_beacon_offset_blocks(anchor, max_beacon_offset_blocks)
                beacon_block = cycle_start + offset
                beacon = anchor if offset == 0 else fresh.get_block_hash(beacon_block)
                blocks_ago = observed_head - beacon_block
                beacon_ts = head_anchor_ts - blocks_ago * 12
            finally:
                closer = getattr(fresh, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
            result = (observed_head, beacon, beacon_ts, int(beacon_block))
            _beacon_context_cache[cache_key] = result
            if len(_beacon_context_cache) > _BEACON_CONTEXT_CACHE_MAX:
                for old_key in list(_beacon_context_cache)[: len(_beacon_context_cache) // 2]:
                    _beacon_context_cache.pop(old_key, None)
            return result
        except Exception as e:
            last_error = e
            if attempt >= attempts or not _is_transient_subtensor_error(e):
                raise
            msg_safe = str(e).encode("unicode_escape").decode()[:200]
            bt.logging.warning(
                f"Verification beacon lookup transient chain failure "
                f"({attempt}/{attempts}); sleeping {delay_s:.0f}s: {msg_safe}"
            )
            if delay_s > 0:
                _t.sleep(delay_s)
            delay_s = min(max(delay_s * 2, 0.0), max_delay_s)
    raise RuntimeError(f"beacon lookup failed after {attempts} attempts: {last_error}")


async def _run_beacon_context_lookup_with_timeout(
    rpc_ref,
    epoch_id: int,
    epoch_interval: int,
    max_beacon_offset_blocks: int,
    current_head: int,
) -> tuple[int, bytes, float, int]:
    global _beacon_lookup_cooldown_until
    async with _beacon_run_lock():
        cooldown_wait_s = _beacon_lookup_cooldown_until - time.time()
        if cooldown_wait_s > 0:
            bt.logging.warning(
                f"Verification beacon lookup cooling down for "
                f"{cooldown_wait_s:.0f}s after transient chain failure"
            )
            await asyncio.sleep(cooldown_wait_s)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _lookup_beacon_context_sync,
                    rpc_ref,
                    epoch_id,
                    epoch_interval,
                    max_beacon_offset_blocks,
                    current_head,
                ),
                timeout=_beacon_lookup_timeout_s(),
            )
        except Exception as e:
            if _is_transient_subtensor_error(e):
                _beacon_lookup_cooldown_until = (
                    time.time() + _beacon_lookup_failure_cooldown_s()
                )
            raise


async def _lookup_beacon_context_with_timeout(
    rpc_ref,
    epoch_id: int,
    epoch_interval: int,
    max_beacon_offset_blocks: int,
    current_head: int,
) -> tuple[int, bytes, float, int]:
    beacon_cache = getattr(app.state, "beacon_cache", None) if hasattr(app, "state") else None
    if beacon_cache is not None:
        cached_context = beacon_cache.context_for_epoch(
            epoch_id,
            epoch_interval,
            max_beacon_offset_blocks,
            current_head,
        )
        if cached_context is not None:
            _network, _netuid, cache_key = _beacon_context_cache_key(
                rpc_ref, epoch_id, epoch_interval, max_beacon_offset_blocks,
            )
            _beacon_context_cache[cache_key] = cached_context
            return cached_context
        detail = ""
        last_error = getattr(beacon_cache, "last_error", lambda: "")()
        if last_error:
            detail = f": {last_error}"
        raise asyncio.TimeoutError(
            f"beacon cache missing context for cycle={epoch_id}{detail}"
        )

    _network, _netuid, cache_key = _beacon_context_cache_key(
        rpc_ref, epoch_id, epoch_interval, max_beacon_offset_blocks,
    )
    cached = _beacon_cache_result_for_head(cache_key, current_head)
    if cached is not None:
        return cached

    async with _beacon_task_lock():
        cached = _beacon_cache_result_for_head(cache_key, current_head)
        if cached is not None:
            return cached
        task = _beacon_context_tasks.get(cache_key)
        if task is None:
            task = asyncio.create_task(
                _run_beacon_context_lookup_with_timeout(
                    rpc_ref,
                    epoch_id,
                    epoch_interval,
                    max_beacon_offset_blocks,
                    current_head,
                )
            )
            _beacon_context_tasks[cache_key] = task

            def _clear_done(done: asyncio.Task[tuple[int, bytes, float, int]], *, task_key=cache_key):
                if _beacon_context_tasks.get(task_key) is done:
                    _beacon_context_tasks.pop(task_key, None)

            task.add_done_callback(_clear_done)
        else:
            bt.logging.debug(
                f"Verification beacon lookup joined in-flight task "
                f"cycle={epoch_id}"
            )

    observed_head, beacon, beacon_ts, beacon_block = await task
    return max(int(current_head), int(observed_head)), beacon, beacon_ts, beacon_block


async def _read_current_block_with_timeout(rpc_ref) -> int:
    beacon_cache = getattr(app.state, "beacon_cache", None) if hasattr(app, "state") else None
    if beacon_cache is not None:
        cached_head = beacon_cache.head()
        if cached_head > 0:
            return int(cached_head)

    timeout_s = _chain_head_call_timeout_s()
    try:
        return int(
            await asyncio.wait_for(
                asyncio.to_thread(rpc_ref.get_current_block),
                timeout=timeout_s,
            )
        )
    except (asyncio.TimeoutError, TimeoutError):
        network = str(getattr(rpc_ref, "network", "") or "")
        if not network:
            raise
        return int(
            await asyncio.wait_for(
                asyncio.to_thread(_fresh_current_block, network),
                timeout=timeout_s,
            )
        )


async def _wait_for_chain_head(target_block: int, epoch_id: int, max_beacon_offset_blocks: int) -> int:
    rpc_ref = getattr(app.state, "rpc", None) if hasattr(app, "state") else None
    if rpc_ref is None:
        return 0

    timeout_s = _chain_ready_wait_timeout_s(max_beacon_offset_blocks)
    poll_s = _chain_ready_poll_s()
    deadline = time.monotonic() + timeout_s
    last_current = 0
    last_error = None
    near_miss_extended = False
    beacon_cache = getattr(app.state, "beacon_cache", None) if hasattr(app, "state") else None
    while True:
        try:
            if beacon_cache is not None:
                current = int(beacon_cache.head() or 0)
                if current <= 0:
                    last_error = RuntimeError("beacon cache has no chain head yet")
                else:
                    last_current = current
                    if current >= target_block:
                        return current
            else:
                current = await _read_current_block_with_timeout(rpc_ref)
                last_current = current
                if current >= target_block:
                    return current
        except Exception as e:
                last_error = e

        if time.monotonic() >= deadline:
            remaining = int(target_block) - int(last_current or 0)
            if (
                not near_miss_extended
                and last_current > 0
                and 0 < remaining <= _chain_ready_near_miss_blocks()
            ):
                extra_s = _chain_ready_near_miss_grace_s()
                if extra_s > 0:
                    near_miss_extended = True
                    deadline = time.monotonic() + extra_s
                    bt.logging.warning(
                        f"Chain readiness wait near target for cycle={epoch_id}: "
                        f"current={last_current} target={target_block}; "
                        f"extending up to {extra_s:.0f}s"
                    )
                    await asyncio.sleep(poll_s)
                    continue

            detail = f", last_error={last_error}" if last_error else ""
            bt.logging.warning(
                f"Chain readiness wait exceeded {timeout_s:.0f}s for cycle={epoch_id}: "
                f"current={last_current} target={target_block}{detail}"
            )
            return 0

        await asyncio.sleep(poll_s)


async def _cycle_chain_ready(epoch_id: int, epoch_interval: int, max_beacon_offset_blocks: int) -> int:
    cycle_start = epoch_id * epoch_interval
    latest_beacon_block = cycle_start + max(1, max_beacon_offset_blocks) - 1
    target_block = latest_beacon_block + _verify_lag_blocks()
    key = (epoch_id, target_block)

    task = _chain_ready_tasks.get(key)
    if task is None:
        task = asyncio.create_task(
            _wait_for_chain_head(target_block, epoch_id, max_beacon_offset_blocks)
        )
        _chain_ready_tasks[key] = task

        def _clear_done(done: asyncio.Task[int], *, task_key=key):
            if _chain_ready_tasks.get(task_key) is done:
                _chain_ready_tasks.pop(task_key, None)

        task.add_done_callback(_clear_done)
    return await task


def _is_timing_failure(reason: str) -> bool:
    return (reason or "").startswith("Timing:")


def _recipe_uses_micro_matrix(recipe_data: dict) -> bool:
    try:
        from common.config import compute_micro_matrix_dim
        bs = int(recipe_data.get("block_size") or 256)
        matrix_dim = int(recipe_data.get("matrix_dim") or 0)
        micro_n = compute_micro_matrix_dim(block_size=bs)
        return abs(matrix_dim - micro_n) <= bs
    except Exception:
        return False


def _rented_micro_wall_deadline_s() -> int:
    try:
        from common.subnet_runtime_config import get_subnet_runtime_config
        return int(
            os.environ.get(
                "RENTED_MICRO_WALL_DEADLINE_S",
                str(get_subnet_runtime_config().scoring.rented_micro_wall_deadline_s),
            )
        )
    except Exception:
        return 90


def _rented_micro_wall_timing_reason(
    recipe_data: dict,
    *,
    is_rented: bool,
    wall_clock_s: float,
    cycle_arrival_latency_s: float = 0.0,
    max_beacon_offset_blocks: int = 0,
) -> str | None:
    if not is_rented or not _recipe_uses_micro_matrix(recipe_data):
        return None
    deadline_s = _rented_micro_wall_deadline_s()
    if deadline_s <= 0:
        return None
    if wall_clock_s > 0:
        if wall_clock_s <= float(deadline_s):
            return None
        return (
            f"Timing: rented micro proof arrival {wall_clock_s:.2f}s after beacon "
            f"exceeds deadline {deadline_s:.2f}s"
        )
    if cycle_arrival_latency_s > 0:
        beacon_window_s = max(1, int(max_beacon_offset_blocks or 0)) * 12
        cycle_deadline_s = float(deadline_s + beacon_window_s)
        if cycle_arrival_latency_s <= cycle_deadline_s:
            return None
        return (
            f"Timing: rented micro proof arrival {cycle_arrival_latency_s:.2f}s after cycle start "
            f"exceeds fallback deadline {cycle_deadline_s:.2f}s"
        )
    return "Timing: rented micro proof arrival timestamp unavailable"


def _is_graceable_timing_outlier(reason: str) -> bool:
    reason = reason or ""
    return (
        reason.startswith("Timing: pass delta ")
        or reason.startswith("Timing: pass_0 receipt ")
        or reason.startswith("Timing: proof arrival ")
        or reason.startswith("Timing: GPU receipt spread ")
        or reason.startswith("Timing: only ")
    )


def _is_timing_grace_consumed_reason(reason: str) -> bool:
    reason = reason or ""
    if _is_graceable_timing_outlier(reason):
        return True
    prefix = "timing_outlier_graced: "
    return reason.startswith(prefix) and _is_graceable_timing_outlier(reason[len(prefix):])


def _count_timing_grace_events(rows) -> int:
    return sum(
        1
        for row in rows
        if _is_timing_grace_consumed_reason(getattr(row, "reason", "") or "")
    )


def _consecutive_timing_events(db, executor_id: str, limit: int = 10) -> int:
    """Count latest consecutive timing-only misses from persisted results."""
    try:
        from common.db import ProofResult
        with db.session() as s:
            rows = s.query(ProofResult).filter(
                ProofResult.executor_id == executor_id,
            ).order_by(
                ProofResult.verified_at.desc(),
                ProofResult.id.desc(),
            ).limit(limit).all()
            streak = 0
            for row in rows:
                if not _is_timing_grace_consumed_reason(row.reason or ""):
                    break
                streak += 1
            return streak
    except Exception as e:
        bt.logging.debug(f"Timing event streak lookup failed for {executor_id[:16]}: {e}")
        return 0


def _consecutive_timing_failures(db, executor_id: str, limit: int = 10) -> int:
    """Count consecutive timing misses for rented-proof auto-cancel gating.

    The persisted stream records isolated timing outliers as neutral skipped
    proof rows. Those rows still consume the consecutive timing allowance, so the
    auto-cancel path uses the same timing-event streak as grace enforcement.
    """
    return _consecutive_timing_events(db, executor_id, limit)


def _recent_timing_events(db, executor_id: str, window_s: int = 3600) -> int:
    """Count timing-only misses in a recent window.

    This supports bounded operational grace for otherwise-valid proofs that
    miss the calibrated timing deadline by a small amount. The miss is still
    audited in logs, but sporadic outliers should not hard-hide an honest GPU.
    """
    try:
        from datetime import datetime, timedelta
        from common.db import ProofResult

        cutoff = datetime.utcnow() - timedelta(seconds=max(1, int(window_s)))
        with db.session() as s:
            rows = s.query(ProofResult).filter(
                ProofResult.executor_id == executor_id,
                ProofResult.verified_at >= cutoff,
                ProofResult.reason.isnot(None),
            ).all()
            return _count_timing_grace_events(rows)
    except Exception as e:
        bt.logging.debug(f"Recent timing event lookup failed for {executor_id[:16]}: {e}")
        return 0


def _maybe_grace_timing_failure(result, executor_id: str) -> VerificationResult:
    """Convert an isolated timing failure into a neutral skipped result.

    The cryptographic proof already passed before timing enforcement changed
    the result to invalid. Allow a small number of non-consecutive timing
    outliers per window per executor; repeated consecutive misses remain hard
    failures and reduce reliability.
    """
    if not _is_graceable_timing_outlier(result.reason):
        return result
    db = getattr(app.state, "db", None)
    if db is None:
        return result
    try:
        from common.subnet_runtime_config import get_subnet_runtime_config
        scoring_cfg = get_subnet_runtime_config().scoring
        default_allowance = int(scoring_cfg.timing_grace_count)
        default_window_s = int(scoring_cfg.timing_grace_window_s)
        default_consecutive_allowance = int(scoring_cfg.timing_grace_consecutive_count)
    except Exception:
        default_allowance = 3
        default_window_s = 3600
        default_consecutive_allowance = 1
    try:
        allowance = int(os.environ.get("TIMING_FAILURE_GRACE_COUNT", str(default_allowance)))
    except Exception:
        allowance = default_allowance
    if allowance <= 0:
        return result
    try:
        window_s = int(os.environ.get("TIMING_FAILURE_GRACE_WINDOW_S", str(default_window_s)))
    except Exception:
        window_s = default_window_s
    try:
        consecutive_allowance = int(os.environ.get(
            "TIMING_FAILURE_GRACE_CONSECUTIVE_COUNT",
            str(default_consecutive_allowance),
        ))
    except Exception:
        consecutive_allowance = default_consecutive_allowance
    consecutive_allowance = max(1, consecutive_allowance)
    consecutive = _consecutive_timing_events(db, executor_id)
    if consecutive >= consecutive_allowance:
        bt.logging.warning(
            f"Timing grace exhausted by consecutive misses: executor={executor_id[:16]} "
            f"consecutive={consecutive}/{consecutive_allowance} reason={result.reason}"
        )
        return result
    recent = _recent_timing_events(db, executor_id, window_s)
    if recent >= allowance:
        bt.logging.warning(
            f"Timing grace exhausted by window: executor={executor_id[:16]} "
            f"recent={recent}/{allowance} window={window_s}s reason={result.reason}"
        )
        return result
    bt.logging.warning(
        f"Timing grace: executor={executor_id[:16]} recent={recent}/{allowance} "
        f"consecutive={consecutive}/{consecutive_allowance} "
        f"window={window_s}s reason={result.reason}"
    )
    return VerificationResult(
        True,
        "skipped",
        f"timing_outlier_graced: {result.reason}",
        result.verification_time_ms,
    )


def _missed_receipts_during_validator_startup(ctx) -> bool:
    """True when this validator likely missed receipt phase during restart.

    Recipes can arrive after a validator restart even if pass_0/pass_1 receipts
    were already emitted while the API was unavailable. That is validator-local
    data loss, not miner misbehavior, so it must not reduce miner score.
    """
    ready_at = float(getattr(app.state, "validator_ready_at", 0) or 0)
    if ready_at <= 0 or float(getattr(ctx, "beacon_timestamp", 0) or 0) <= 0:
        return False
    db = getattr(app.state, "db", None)
    if db is not None:
        try:
            from common.db import validator_outage_overlaps
            if validator_outage_overlaps(
                db,
                float(ctx.beacon_timestamp),
                max(ready_at, time.time()),
            ):
                return True
        except Exception:
            pass
    # Receipt phase starts soon after the beacon. If the validator became
    # ready after the beacon, it cannot safely require receipt evidence for
    # that cycle.
    return ready_at > float(ctx.beacon_timestamp)


def _is_rental_end_mode_transition(result, recipe_data: dict, ctx, executor_id: str) -> bool:
    """True for the free-vs-micro race immediately after a rental ends.

    This is a neutral skip condition, not a pass. Free executors still need
    heavy proofs; we only tolerate a micro proof when the local rental window
    shows the executor was released a few blocks before the proof beacon and
    the recipe itself says it was generated in allocated mode.
    """
    if result.valid or getattr(ctx, "is_rented", False):
        return False
    reason = result.reason or ""
    if "invalid for free executor" not in reason:
        return False
    if str(recipe_data.get("allocation_state") or "") != "allocated":
        return False

    try:
        from common.config import compute_micro_matrix_dim
        matrix_dim = int(recipe_data.get("matrix_dim") or 0)
        micro_n = compute_micro_matrix_dim()
        if abs(matrix_dim - micro_n) > 256:
            return False
    except Exception:
        return False

    try:
        from neurons.validator.api.routes import rent as rent_route
        grace_blocks = int(os.environ.get("RENTAL_MODE_TRANSITION_GRACE_BLOCKS", "4"))
        mode_block = int(getattr(ctx, "beacon_block", 0) or 0)
        if mode_block <= 0:
            mode_block = int(recipe_data.get("epoch_id") or 0) * int(getattr(ctx, "epoch_interval", 15) or 15)
        return rent_route.rental_state.ended_within(executor_id, mode_block, grace_blocks)
    except Exception as e:
        bt.logging.debug(f"rental end transition check failed for {executor_id[:16]}: {e}")
        return False


def _is_rental_start_mode_transition(result, recipe_data: dict, ctx, executor_id: str) -> bool:
    """True for the free-vs-micro race immediately after a rental starts.

    This is a neutral skip condition, not a pass. The validator remains strict
    for free executors: micro proofs are tolerated only when a local/event
    rental window starts within a small block grace after the proof beacon and
    the recipe says it was generated in allocated mode.
    """
    if result.valid or getattr(ctx, "is_rented", False):
        return False
    reason = result.reason or ""
    if "invalid for free executor" not in reason:
        return False
    if str(recipe_data.get("allocation_state") or "") != "allocated":
        return False

    try:
        from common.config import compute_micro_matrix_dim
        matrix_dim = int(recipe_data.get("matrix_dim") or 0)
        micro_n = compute_micro_matrix_dim()
        if abs(matrix_dim - micro_n) > 256:
            return False
    except Exception:
        return False

    try:
        from neurons.validator.api.routes import rent as rent_route
        grace_blocks = int(os.environ.get("RENTAL_MODE_TRANSITION_GRACE_BLOCKS", "4"))
        mode_block = int(getattr(ctx, "beacon_block", 0) or 0)
        if mode_block <= 0:
            mode_block = int(recipe_data.get("epoch_id") or 0) * int(getattr(ctx, "epoch_interval", 15) or 15)
        return rent_route.rental_state.started_within_after(
            executor_id, mode_block, grace_blocks,
        )
    except Exception as e:
        bt.logging.debug(f"rental start transition check failed for {executor_id[:16]}: {e}")
        return False


def _chain_unavailable_retry_count(item: dict, recipe_data: dict) -> int:
    return int(
        item.get("chain_unavailable_retries")
        or recipe_data.get("__validator_chain_unavailable_retries")
        or 0
    )


def _chain_unavailable_retry_delay_s(retry_index: int) -> float:
    initial = _chain_unavailable_requeue_initial_delay_s()
    max_delay = _chain_unavailable_requeue_max_delay_s()
    return min(max_delay, initial * (2 ** max(0, retry_index)))


def _schedule_chain_unavailable_retry(item: dict, reason: str) -> bool:
    """Requeue validator-side chain outages without asking the miner to resend.

    The proof ingress route rejects duplicate recipes, so temporary subtensor
    outages must be retried internally. This is only for validator-chain
    unavailability; cryptographic proof failures are still final.
    """
    if verification_queue is None:
        return False
    recipe_data = item.get("recipe") or {}
    executor_id = str(recipe_data.get("executor_id") or "")
    epoch_id = int(recipe_data.get("epoch_id") or 0)
    retries = _chain_unavailable_retry_count(item, recipe_data)
    max_retries = _chain_unavailable_requeue_attempts()
    if retries >= max_retries:
        return False

    delay_s = _chain_unavailable_retry_delay_s(retries)
    recv_time = float(item.get("recv_time") or time.time())
    retry_recipe = dict(recipe_data)
    retry_recipe["__validator_chain_unavailable_retries"] = retries + 1
    async def _delayed_requeue():
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        try:
            from neurons.validator.api.routes import proofs as proofs_route
            queued = proofs_route._enqueue_verification_or_skip(retry_recipe, recv_time)
            if queued:
                bt.logging.warning(
                    f"Requeued proof after validator chain unavailable: "
                    f"executor={executor_id[:16]} cycle={epoch_id} "
                    f"retry={retries + 1}/{max_retries} reason={reason}"
                )
            else:
                bt.logging.warning(
                    f"Could not requeue chain-unavailable proof due to "
                    f"validator capacity: executor={executor_id[:16]} cycle={epoch_id}"
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            bt.logging.warning(
                f"Chain-unavailable proof requeue failed: "
                f"executor={executor_id[:16]} cycle={epoch_id}: {e}"
            )

    asyncio.create_task(_delayed_requeue())
    bt.logging.warning(
        f"Scheduling proof retry after validator chain unavailable: "
        f"executor={executor_id[:16]} cycle={epoch_id} retry={retries + 1}/{max_retries} "
        f"in {delay_s:.0f}s reason={reason}"
    )
    return True


async def _verify_one_item(item: dict):
    """Process a single recipe — finalization wait, dispatch to pool, record result.

    Spawned as an asyncio.Task per recipe so the verification pool's workers
    actually run in parallel. All recipes from one cycle (which arrive within
    seconds of each other when the cycle boundary fires) are processed
    concurrently up to `verify_pool.max_workers`.
    """
    recipe_data = item["recipe"]
    recv_time = item["recv_time"]
    executor_id = recipe_data.get("executor_id", "")
    epoch_id = recipe_data.get("epoch_id", 0)
    chain_retry_count = _chain_unavailable_retry_count(item, recipe_data)
    bt.logging.info(
        f"Verification started: executor={executor_id[:16]} cycle={epoch_id}"
    )

    # ── CHAIN READINESS WAIT ────────────────────────────
    # Don't verify until the configured chain head has moved far enough beyond the
    # latest possible beacon block. This prevents verifying against a very
    # recent head while keeping the wait bounded if the node stalls. The wait
    # is shared per cycle, so a burst of miner receipts causes one chain poll
    # loop instead of one loop per executor.
    epoch_interval = int(os.environ.get("PROOF_EPOCH_BLOCKS", "15"))
    from common.proof_schedule import DEFAULT_BEACON_MAX_OFFSET_BLOCKS
    max_beacon_offset_blocks = int(os.environ.get(
        "PROOF_BEACON_MAX_OFFSET_BLOCKS",
        str(DEFAULT_BEACON_MAX_OFFSET_BLOCKS),
    ))
    current_head = await _cycle_chain_ready(
        int(epoch_id), epoch_interval, max_beacon_offset_blocks,
    )
    if not current_head:
        if _schedule_chain_unavailable_retry(item, "chain_readiness"):
            return
        from neurons.validator.api.routes import proofs as proofs_route
        proofs_route._record_validator_capacity_skip(
            recipe_data, recv_time, reason="validator_chain_unavailable",
        )
        bt.logging.warning(
            f"Chain readiness wait failed for cycle={epoch_id} — "
            "recorded validator_chain_unavailable skip"
        )
        return
    bt.logging.info(
        f"Verification chain ready: executor={executor_id[:16]} "
        f"cycle={epoch_id} head={current_head}"
    )

    # If our own queue was so backed up that the recipe is already outside the
    # verifier's freshness window, accept it as validator-capacity skipped.
    # Do not record a proof failure or auto-cancel a rental for our lag.
    if current_head:
        current_epoch = current_head // epoch_interval
        stale_after_epochs = _stale_verification_skip_after_epochs()
        if current_epoch > epoch_id + stale_after_epochs:
            from neurons.validator.api.routes import proofs as proofs_route
            skip_reason = (
                "validator_chain_unavailable"
                if chain_retry_count > 0
                else "validator_capacity_skip"
            )
            proofs_route._record_validator_capacity_skip(
                recipe_data,
                recv_time,
                reason=skip_reason,
            )
            bt.logging.warning(
                f"Skipping stale queued proof due to validator capacity: "
                f"executor={executor_id[:16]} recipe_cycle={epoch_id} "
                f"current_cycle={current_epoch} retry={chain_retry_count} "
                f"reason={skip_reason}"
            )
            return

    try:
        # ── VERIFICATION TIER ──────────────────────────────
        proof_score = analyzer.get_score(executor_id) if analyzer else None
        trust = proof_score.trust_level if proof_score else "full"

        # ── BUILD VERIFICATION CONTEXT ─────────────────────
        from neurons.validator.proof.verifier import verify_recipe, VerificationContext, VerificationResult
        from neurons.validator.api.routes import browse

        ctx = VerificationContext()
        ctx.executor_id = executor_id
        ctx.recv_time = recv_time
        ctx.num_challenges = int(os.environ.get("NUM_CHALLENGES", "4"))
        ctx.epoch_interval = int(os.environ.get("PROOF_EPOCH_BLOCKS", "15"))
        ctx.require_beacon = True
        ctx.require_registry = True

        # Beacon from chain (for seed verification)
        rpc_ref = getattr(app.state, 'rpc', None) if hasattr(app, 'state') else None
        if rpc_ref:
            bt.logging.info(
                f"Verification beacon lookup: executor={executor_id[:16]} cycle={epoch_id}"
            )
            try:
                cb, beacon, beacon_ts, beacon_block = await _lookup_beacon_context_with_timeout(
                    rpc_ref,
                    int(epoch_id),
                    ctx.epoch_interval,
                    max_beacon_offset_blocks,
                    int(current_head),
                )
            except (asyncio.TimeoutError, TimeoutError) as e:
                bt.logging.warning(
                    f"Chain lookup timed out for cycle={epoch_id}: {e}"
                )
                cb, beacon, beacon_ts, beacon_block = None, None, None, None
            except Exception as e:
                bt.logging.warning(f"Chain lookup failed for cycle={epoch_id}: {e}")
                cb, beacon, beacon_ts, beacon_block = None, None, None, None
            if not cb or not beacon or not beacon_block:
                if _schedule_chain_unavailable_retry(item, "beacon_lookup"):
                    return
                from neurons.validator.api.routes import proofs as proofs_route
                proofs_route._record_validator_capacity_skip(
                    recipe_data, recv_time, reason="validator_chain_unavailable",
                )
                bt.logging.warning(
                    f"Skipping proof due to validator chain lookup failure: "
                    f"executor={executor_id[:16]} cycle={epoch_id}"
                )
                return

            ctx.current_epoch = int(epoch_id) if chain_retry_count > 0 else cb // ctx.epoch_interval
            ctx.beacon = beacon
            ctx.beacon_timestamp = beacon_ts
            ctx.beacon_block = int(beacon_block or 0)
            bt.logging.debug(
                f"Beacon for cycle={epoch_id}: block={ctx.beacon_block} "
                f"hash={beacon[:8].hex()}"
            )
            bt.logging.info(
                f"Verification beacon ready: executor={executor_id[:16]} "
                f"cycle={epoch_id} block={ctx.beacon_block}"
            )
        else:
            from neurons.validator.api.routes import proofs as proofs_route
            proofs_route._record_validator_capacity_skip(
                recipe_data, recv_time, reason="validator_chain_unavailable",
            )
            bt.logging.warning(
                f"Skipping proof because validator RPC is unavailable: "
                f"executor={executor_id[:16]} cycle={epoch_id}"
            )
            return

        # ── DUPLICATE REJECTION ────────────────────────────
        dedup_key = (executor_id, epoch_id)
        if dedup_key in _verified_epochs:
            bt.logging.debug(f"Duplicate proof rejected: executor={executor_id[:16]} cycle={epoch_id}")
            return
        _verified_epochs.add(dedup_key)
        if len(_verified_epochs) > 10000:
            _verified_epochs.clear()

        # Registry data (for VRAM + GPU count validation). `is_rented` comes
        # from the block-anchored rental_state if possible — that answers
        # "was this executor rented at the proof beacon block?" rather than
        # "is it rented right now?". rental_state is populated from this
        # validator's own writes plus ComputeRegistry RentalStarted/RentalEnded
        # events, so peer-validator rentals are covered once the event indexer
        # has caught up.
        if browse.registry_client:
            try:
                # Use the proofs.py-side cache (60s TTL) so cycle-boundary
                # bursts don't hammer the chain RPC. With ~20 executors and
                # 5-block finalization waits we used to do ~40 chain calls/min
                # just for VRAM/is_rented lookup; cache cuts that to ~2.
                from neurons.validator.api.routes.proofs import _get_executor_info_cached
                spec = await _get_executor_info_cached(executor_id)
                if spec:
                    ctx.expected_vram_mb = spec.vram_mb
                    ctx.expected_gpu_count = spec.gpu_count
                    # Resolve gpu_model from the keccak/sha256 hash so the
                    # timing model lookup hits the right entry. Without this,
                    # every proof falls back to 'default' (10s base) which
                    # false-rejects A6000s (~20s expected) and any GPU slower
                    # than 10s — we caught this when an A6000 proof timed
                    # out at 35s vs the 18s 'default' deadline.
                    from neurons.validator.api.routes.browse import _name_for_hash
                    ctx.gpu_model_name = _name_for_hash(spec.gpu_model_hash)
                    # Block-anchored lookup: was rented at the beacon block?
                    # The miner decides proof mode when the beacon is known
                    # and before per-executor jitter, so using the cycle start
                    # would false-reject a rental that began inside the cycle
                    # before the beacon. If beacon_block is unavailable, fall
                    # back to the cycle start for older receipts/tests.
                    # Prefer the event-backed historical timeline. Only fall
                    # back to the live chain flag if the event indexer has not
                    # seen this executor yet.
                    from neurons.validator.api.routes import rent as rent_route
                    cycle_block = epoch_id * ctx.epoch_interval
                    if rent_route.rental_state.has_history(executor_id):
                        mode_block = ctx.beacon_block or cycle_block
                        ctx.is_rented = rent_route.rental_state.was_rented_at(
                            executor_id, mode_block
                        )
                    else:
                        ctx.is_rented = spec.is_rented
            except Exception:
                pass

        if ctx.require_registry and (
            int(ctx.expected_gpu_count or 0) <= 0
            or int(ctx.expected_vram_mb or 0) <= 0
        ):
            from neurons.validator.api.routes import proofs as proofs_route
            proofs_route._record_validator_capacity_skip(
                recipe_data,
                recv_time,
                reason="validator_registry_unavailable",
            )
            bt.logging.warning(
                f"Skipping proof until validator registry context is available: "
                f"executor={executor_id[:16]} cycle={epoch_id}"
            )
            return

        # ── DISPATCH TO PROCESS POOL ───────────────────────
        # The pool has N workers; multiple in-flight tasks can run concurrently.
        bt.logging.info(
            f"Verification dispatch: executor={executor_id[:16]} cycle={epoch_id} "
            f"gpu={ctx.gpu_model_name or 'unknown'} count={ctx.expected_gpu_count}"
        )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            verify_pool, verify_recipe, recipe_data, trust, ctx,
        )
        bt.logging.info(
            f"Verification returned: executor={executor_id[:16]} cycle={epoch_id} "
            f"valid={result.valid}"
        )

        external_pass_delta_s = 0.0
        wall_clock_s = 0.0
        arrival_latency_s = 0.0
        pass0_spread_s = 0.0
        pass1_spread_s = 0.0
        pass0_wall_s = 0.0
        pass1_wall_s = 0.0

        # ── TIMING MODEL ENFORCEMENT ───────────────────────
        if result.valid and analyzer and ctx.expected_gpu_count > 0:
            from common.proof_timing import (
                compute_delta_deadline, compute_delta_floor,
                compute_pass0_receipt_deadline,
                get_timing_model, is_timing_model_calibrated,
                compute_wall_clock_deadline,
                compute_receipt_spread_deadline,
            )
            from neurons.validator.api.routes.proofs import get_bound_pass_timing_stats

            gpu_model = ctx.gpu_model_name or "default"
            timing_model = get_timing_model(gpu_model)
            timing_calibrated = is_timing_model_calibrated(gpu_model, ctx.expected_gpu_count)
            calibration_mode = _env_bool("NODEXO_TIMING_CALIBRATION_MODE")

            timing_stats = get_bound_pass_timing_stats(recipe_data)
            min_pass_delta_s = float(timing_stats.get("min_delta") or 0.0)
            max_pass_delta_s = float(timing_stats.get("max_delta") or 0.0)
            external_pass_delta_s = max_pass_delta_s
            pass0_spread_s = float(timing_stats.get("pass0_spread") or 0.0)
            pass1_spread_s = float(timing_stats.get("pass1_spread") or 0.0)
            bound_receipt_count = int(timing_stats.get("count") or 0)
            max_pass0_recv_s = float(timing_stats.get("max_pass0_recv") or 0.0)
            max_pass1_recv_s = float(timing_stats.get("max_pass1_recv") or 0.0)
            if not timing_calibrated:
                detail = (
                    f"Timing: GPU model/count {gpu_model} × {ctx.expected_gpu_count} has no calibrated timing model"
                )
                if calibration_mode:
                    bt.logging.warning(
                        f"Timing calibration mode: accepting {executor_id[:16]} "
                        f"cycle={epoch_id} for uncalibrated model {gpu_model}; "
                        "model-derived deadlines skipped"
                    )
                else:
                    result = VerificationResult(
                        False,
                        result.tier,
                        detail,
                        result.verification_time_ms,
                    )
                    bt.logging.warning(
                        f"Timing reject: {executor_id[:16]} cycle={epoch_id} "
                        f"uncalibrated_model={gpu_model} count={ctx.expected_gpu_count}"
                    )

            if result.valid and timing_calibrated and max_pass_delta_s > 0:
                delta_deadline = compute_delta_deadline(timing_model, ctx.expected_gpu_count)
                if max_pass_delta_s > delta_deadline:
                    result = VerificationResult(
                        False, result.tier,
                        f"Timing: pass delta {max_pass_delta_s:.2f}s exceeds deadline {delta_deadline:.2f}s "
                        f"for {gpu_model} × {ctx.expected_gpu_count}",
                        result.verification_time_ms,
                    )
                    bt.logging.warning(
                        f"Timing reject: {executor_id[:16]} cycle={epoch_id} "
                        f"delta={max_pass_delta_s:.2f} deadline={delta_deadline:.2f}"
                    )

            if result.valid and timing_calibrated and not ctx.is_rented and min_pass_delta_s > 0:
                try:
                    from common.subnet_runtime_config import get_subnet_runtime_config
                    min_delta_fraction = (
                        get_subnet_runtime_config().scoring.timing_min_delta_fraction
                    )
                except Exception:
                    from common.proof_timing import DEFAULT_MIN_DELTA_FRACTION
                    min_delta_fraction = DEFAULT_MIN_DELTA_FRACTION
                min_delta_fraction = max(
                    float(min_delta_fraction or 0.0),
                    _env_float("NODEXO_TIMING_MIN_DELTA_FRACTION_MIN", 0.55),
                )
                delta_floor = compute_delta_floor(
                    timing_model, ctx.expected_gpu_count, min_delta_fraction,
                )
                if delta_floor > 0 and min_pass_delta_s < delta_floor:
                    result = VerificationResult(
                        False, result.tier,
                        f"Timing: pass delta {min_pass_delta_s:.2f}s below credible floor "
                        f"{delta_floor:.2f}s for {gpu_model} × {ctx.expected_gpu_count}",
                        result.verification_time_ms,
                    )
                    bt.logging.warning(
                        f"Timing reject: {executor_id[:16]} cycle={epoch_id} "
                        f"delta_floor={min_pass_delta_s:.2f}/{delta_floor:.2f}"
                    )

            # Free-capacity heavy proofs must also arrive close to the beacon.
            # Otherwise one physical GPU can serialize several claimed
            # executors and still keep each individual pass delta plausible.
            if ctx.beacon_timestamp > 0 and recv_time > 0:
                wall_clock_s = max(0.0, recv_time - ctx.beacon_timestamp)
            if ctx.beacon_timestamp > 0 and max_pass0_recv_s > 0:
                pass0_wall_s = max(0.0, max_pass0_recv_s - ctx.beacon_timestamp)
            if ctx.beacon_timestamp > 0 and max_pass1_recv_s > 0:
                pass1_wall_s = max(0.0, max_pass1_recv_s - ctx.beacon_timestamp)
            anchor = getattr(app.state, 'chain_anchor', None)
            if anchor is not None and recv_time:
                anchor_block, anchor_ts = anchor
                cycle_start_block = epoch_id * ctx.epoch_interval
                est_cycle_start_ts = anchor_ts + (cycle_start_block - anchor_block) * 12
                arrival_latency_s = max(0.0, recv_time - est_cycle_start_ts)

            rented_micro_timing_reason = _rented_micro_wall_timing_reason(
                recipe_data,
                is_rented=bool(ctx.is_rented),
                wall_clock_s=wall_clock_s,
                cycle_arrival_latency_s=arrival_latency_s,
                max_beacon_offset_blocks=max_beacon_offset_blocks,
            )
            if result.valid and rented_micro_timing_reason:
                result = VerificationResult(
                    False, result.tier,
                    rented_micro_timing_reason,
                    result.verification_time_ms,
                )
                bt.logging.warning(
                    f"Timing reject: rented micro executor={executor_id[:16]} "
                    f"cycle={epoch_id} wall={wall_clock_s:.2f} "
                    f"deadline={_rented_micro_wall_deadline_s():.2f}"
                )

            if result.valid and timing_calibrated and not ctx.is_rented and wall_clock_s > 0:
                wall_deadline = compute_wall_clock_deadline(
                    timing_model, ctx.expected_gpu_count,
                )
                if wall_clock_s > wall_deadline:
                    result = VerificationResult(
                        False, result.tier,
                        f"Timing: proof arrival {wall_clock_s:.2f}s after beacon exceeds deadline "
                        f"{wall_deadline:.2f}s for {gpu_model} × {ctx.expected_gpu_count}",
                        result.verification_time_ms,
                    )
                    bt.logging.warning(
                        f"Timing reject: {executor_id[:16]} cycle={epoch_id} "
                        f"wall={wall_clock_s:.2f} deadline={wall_deadline:.2f}"
                    )

            if (
                result.valid
                and timing_calibrated
                and not ctx.is_rented
                and pass0_wall_s > 0
            ):
                from common.proof_schedule import (
                    DEFAULT_JITTER_SECONDS,
                    derive_jitter_seconds,
                )
                max_jitter_s = int(os.environ.get(
                    "NODEXO_PROOF_MAX_JITTER_SECONDS",
                    str(DEFAULT_JITTER_SECONDS),
                ))
                try:
                    jitter_s = derive_jitter_seconds(
                        ctx.beacon, executor_id, max_jitter_s,
                    )
                except Exception:
                    jitter_s = max_jitter_s
                pass0_deadline = (
                    compute_pass0_receipt_deadline(
                        timing_model,
                        ctx.expected_gpu_count,
                        jitter_s=jitter_s,
                        grace_s=_env_float("NODEXO_TIMING_PASS0_RECEIPT_GRACE_S", 6.0),
                    )
                )
                if pass0_wall_s > pass0_deadline:
                    result = VerificationResult(
                        False, result.tier,
                        f"Timing: pass_0 receipt {pass0_wall_s:.2f}s after beacon exceeds deadline "
                        f"{pass0_deadline:.2f}s for {gpu_model} × {ctx.expected_gpu_count}",
                        result.verification_time_ms,
                    )
                    bt.logging.warning(
                        f"Timing reject: {executor_id[:16]} cycle={epoch_id} "
                        f"pass0_wall={pass0_wall_s:.2f} deadline={pass0_deadline:.2f}"
                    )

            # Free proofs require all per-GPU timing receipts to bind to the
            # final recipe. Otherwise a miner could omit receipts and bypass
            # the stricter pass-delta gate, leaving only the looser wall-clock
            # guard. Multi-GPU proofs also require the receipts to arrive as a
            # parallel batch.
            if result.valid and not ctx.is_rented:
                if bound_receipt_count < ctx.expected_gpu_count:
                    if _missed_receipts_during_validator_startup(ctx):
                        from neurons.validator.api.routes import proofs as proofs_route
                        proofs_route._record_validator_capacity_skip(
                            recipe_data, recv_time,
                            reason="validator_startup_missing_receipts",
                        )
                        bt.logging.warning(
                            f"Skipping proof due to validator startup missing receipts: "
                            f"executor={executor_id[:16]} cycle={epoch_id} "
                            f"receipts={bound_receipt_count}/{ctx.expected_gpu_count}"
                        )
                        return
                    result = VerificationResult(
                        False, result.tier,
                        f"Timing: only {bound_receipt_count}/{ctx.expected_gpu_count} GPU receipts bound to recipe",
                        result.verification_time_ms,
                    )
                    bt.logging.warning(
                        f"Timing reject: {executor_id[:16]} cycle={epoch_id} "
                        f"receipts={bound_receipt_count}/{ctx.expected_gpu_count}"
                    )
                elif timing_calibrated and ctx.expected_gpu_count > 1:
                    spread_deadline = compute_receipt_spread_deadline(
                        timing_model, ctx.expected_gpu_count,
                    )
                    spread_s = max(pass0_spread_s, pass1_spread_s)
                    if spread_s > spread_deadline:
                        result = VerificationResult(
                            False, result.tier,
                            f"Timing: GPU receipt spread {spread_s:.2f}s exceeds deadline "
                            f"{spread_deadline:.2f}s for {gpu_model} × {ctx.expected_gpu_count}",
                            result.verification_time_ms,
                        )
                        bt.logging.warning(
                            f"Timing reject: {executor_id[:16]} cycle={epoch_id} "
                            f"spread={spread_s:.2f} deadline={spread_deadline:.2f}"
                        )

        result = _maybe_grace_timing_failure(result, executor_id)

        if _is_rental_end_mode_transition(result, recipe_data, ctx, executor_id):
            from neurons.validator.api.routes import proofs as proofs_route
            proofs_route._record_validator_capacity_skip(
                recipe_data, recv_time, reason="rental_end_mode_transition_skip",
            )
            bt.logging.info(
                f"Skipping rental end mode transition proof: "
                f"executor={executor_id[:16]} cycle={epoch_id} "
                f"matrix_dim={recipe_data.get('matrix_dim')} "
                f"beacon_block={ctx.beacon_block}"
            )
            return

        if _is_rental_start_mode_transition(result, recipe_data, ctx, executor_id):
            from neurons.validator.api.routes import proofs as proofs_route
            proofs_route._record_validator_capacity_skip(
                recipe_data, recv_time, reason="rental_start_mode_transition_skip",
            )
            bt.logging.info(
                f"Skipping rental start mode transition proof: "
                f"executor={executor_id[:16]} cycle={epoch_id} "
                f"matrix_dim={recipe_data.get('matrix_dim')} "
                f"beacon_block={ctx.beacon_block}"
            )
            return

        # Record result in analyzer + DB
        if analyzer:
            from neurons.validator.proof.analyzer import EpochResult
            analyzer.record_result(EpochResult(
                epoch_id=epoch_id,
                executor_id=executor_id,
                valid=result.valid,
                tier=result.tier,
                compute_time=recipe_data.get("compute_time", 0),
                timestamp=time.time(),
                reason=result.reason,
                delta_seconds=external_pass_delta_s,
            ))

        timing_failure_streak = 0
        if hasattr(app.state, 'db') and app.state.db:
            from common.db import store_proof_result, update_executor_on_proof
            # Compute arrival_latency relative to estimated cycle start.
            # Externally-anchored signal — miner controls neither side
            # (recv_time = our wall-clock; cycle_start_ts derived from
            # chain block timestamp). Detects GPU contention without
            # trusting miner-reported compute_time.
            # Cleaner signal: delta between validator's recv_time for pass_0
            # and pass_1 receipts, accepted only when receipts match the final
            # recipe roots/T_commit values. Same network path → jitter cancels.
            store_proof_result(
                app.state.db, executor_id, epoch_id,
                valid=result.valid, tier=result.tier, reason=result.reason,
                delta=external_pass_delta_s,
                compute_time=recipe_data.get("compute_time", 0),
                verify_ms=result.verification_time_ms,
                recv_time=float(recv_time or 0),
                arrival_latency_s=arrival_latency_s,
                external_pass_delta_s=external_pass_delta_s,
                wall_clock_s=wall_clock_s,
                receipt_pass0_spread_s=pass0_spread_s,
                receipt_pass1_spread_s=pass1_spread_s,
                receipt_pass0_wall_s=pass0_wall_s,
                receipt_pass1_wall_s=pass1_wall_s,
                matrix_dim=int(recipe_data.get("matrix_dim") or 0),
                allocation_state=str(recipe_data.get("allocation_state") or ""),
            )
            if not result.valid and _is_timing_failure(result.reason):
                timing_failure_streak = _consecutive_timing_failures(
                    app.state.db, executor_id,
                )
            from neurons.validator.api.routes.proofs import _executor_hotkey_binding
            hotkey = _executor_hotkey_binding.get(executor_id, "")
            update_executor_on_proof(
                app.state.db, executor_id, hotkey_ss58=hotkey,
                epoch_id=epoch_id, valid=result.valid,
            )

        bt.logging.info(
            f"Verified: executor={executor_id[:16]} cycle={epoch_id} valid={result.valid} tier={result.tier} ({result.verification_time_ms:.1f}ms) {result.reason if not result.valid else ''}"
        )

        # ── AUTO-CANCEL RENTAL ON PROOF FAILURE ────────────────
        # If this executor was rented during the cycle and the proof failed,
        # the renter is paying for a GPU we can't verify. Cut the rental
        # immediately: markAvailable + destroy container + record a
        # failure event. The immediate cutoff prevents the renter from
        # continuing to pay for a broken GPU while settlement policy handles
        # any account-level adjustment.
        if not result.valid and ctx.is_rented:
            if _is_timing_failure(result.reason):
                required_streak = int(os.environ.get(
                    "TIMING_FAILURE_AUTOCANCEL_STREAK", "2",
                ))
                if timing_failure_streak <= 0:
                    timing_failure_streak = required_streak
                if timing_failure_streak < required_streak:
                    bt.logging.warning(
                        f"Timing failure {timing_failure_streak}/{required_streak} "
                        f"for rented executor={executor_id[:16]} cycle={epoch_id}; "
                        "not auto-cancelling until consecutive threshold is reached"
                    )
                    return

            from neurons.validator.api.routes import rent as rent_route
            active_rid = rent_route.rental_state.has_active_rental(executor_id)
            if active_rid:
                bt.logging.warning(
                    f"AUTO-CANCEL rental {active_rid[:8]}: executor={executor_id[:16]} "
                    f"failed proof at cycle={epoch_id} (reason: {result.reason})"
                )
                try:
                    from common.db import open_sybil_flag
                    opened = open_sybil_flag(app.state.db, executor_id, "rental_proof_fail", {
                        "rental_id": active_rid,
                        "cycle": epoch_id,
                        "reason": result.reason,
                        "tier": result.tier,
                    })
                    if opened and rent_route.chain_snapshot is not None:
                        rent_route.chain_snapshot.invalidate_executor(executor_id)
                except Exception as e:
                    bt.logging.warning(
                        f"rental_proof_fail probation flag failed for "
                        f"{executor_id[:16]}: {e}"
                    )
                # Use the internal terminate so auth gates don't fire (no
                # Request, no owner check); reason="proof_fail" so the DB
                # row records the cause. Fire-and-forget but log on failure
                # via add_done_callback — previously the bare create_task
                # swallowed exceptions silently.
                from neurons.validator.api.routes.rent import _terminate_rental_internal
                _t = asyncio.create_task(
                    _terminate_rental_internal(active_rid, reason="proof_fail")
                )
                def _log_done(t, rid=active_rid):
                    if t.cancelled():
                        return
                    exc = t.exception()
                    if exc:
                        bt.logging.error(f"Auto-cancel {rid[:8]} failed: {exc}")
                _t.add_done_callback(_log_done)

    except asyncio.CancelledError:
        raise
    except Exception as e:
        bt.logging.error(f"Verify task error for cycle={epoch_id} executor={executor_id[:16]}: {e}")


async def _verification_loop():
    """Pull recipes from the queue and dispatch each to its own task.

    Concurrent verification: multiple `_verify_one_item` tasks run in flight,
    using the verify_pool's worker processes in parallel. This matters because
    all miners broadcast within seconds of each other when a cycle boundary
    fires, so the queue spikes — serial dispatch would create lag proportional
    to executor count.

    Caps in-flight tasks at MAX_INFLIGHT to bound memory under traffic spikes.
    """
    MAX_INFLIGHT = int(
        os.environ.get("VERIFY_MAX_INFLIGHT") or _verify_max_inflight_default()
    )
    inflight: set[asyncio.Task] = set()

    while True:
        try:
            # Backpressure: if too many tasks in flight, wait for some to finish
            if len(inflight) >= MAX_INFLIGHT:
                done, inflight = await asyncio.wait(
                    inflight, return_when=asyncio.FIRST_COMPLETED,
                )
                continue

            try:
                _priority, _queued_at, item = await asyncio.wait_for(
                    verification_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                # Queue empty — drain finished tasks, recheck stuck-verification timer
                if inflight:
                    done = {t for t in inflight if t.done()}
                    inflight -= done
                continue

            task = asyncio.create_task(_verify_one_item(item))
            try:
                recipe = item.get("recipe") or {}
                bt.logging.info(
                    f"Verification dequeued: executor={str(recipe.get('executor_id', ''))[:16]} "
                    f"cycle={recipe.get('epoch_id')}"
                )
            except Exception:
                pass
            inflight.add(task)
            task.add_done_callback(inflight.discard)

        except asyncio.CancelledError:
            for t in inflight:
                t.cancel()
            break
        except Exception as e:
            bt.logging.error(f"Verification loop error: {e}")
            await asyncio.sleep(1)


async def _scoring_loop():
    """Recompute per-executor scores every 60s using scorer_v2.

    Score model (see neurons/validator/scoring/scorer_v2.py):
      score = GPU_HOURLY_PRICE[gpu_model] × gpu_count × utilization × reliability
      score = 0 if ANY of: stale proof, missing heartbeat, open HARD
              sybil flag, last canary == fail, or executor not active
    """
    import time as _time
    while True:
        try:
            if analyzer is None:
                await asyncio.sleep(5)
                continue

            from neurons.validator.api.routes import browse, rent
            registry = browse.registry_client
            db = rent.db  # access via the route module that already has it
            if registry is None or db is None:
                continue

            # Keep denormalized pass-rate fields fresh before score_one reads
            # them from ExecutorStats. Without this, proof reliability would lag
            # one scoring interval behind the verifier.
            analyzer.sync_stats_to_db()

            from neurons.validator.scoring.scorer import (
                score_one, build_context_from_db, scoring_config_from_runtime,
                stake_requirement_for_context,
            )
            from neurons.validator.api.routes.browse import _name_for_hash

            # If the canary scheduler is OFF on this validator, scoring
            # must not require a canary record (every executor would be
            # canary_pending forever and score 0). When ON, scoring is
            # strict and requires a passing canary.
            _canary_required = os.environ.get("CANARY_ENABLED", "0") == "1"
            scoring_cfg = scoring_config_from_runtime(
                canary_required=_canary_required,
            )

            now_ts = _time.time()
            try:
                execs = await asyncio.to_thread(registry.get_all_active_executors)
            except Exception as e:
                bt.logging.debug(f"scoring: chain query failed: {e}")
                continue

            # Pull the metagraph once per loop so stake lookups don't
            # hammer the chain per-executor. get_metagraph is cached
            # (3-min TTL inside rpc) so this is effectively a dict read.
            mg = None
            stake_by_hotkey: dict[str, float] = {}
            try:
                mg = await asyncio.to_thread(app.state.rpc.get_metagraph, False)
                hk_list = list(getattr(mg, "hotkeys", []))
                stakes = getattr(mg, "stake", None)
                if stakes is not None:
                    # mg.stake is subnet alpha stake (float) on lite metagraphs.
                    stake_by_hotkey = {
                        hk_list[i]: float(stakes[i])
                        for i in range(min(len(hk_list), len(stakes)))
                    }
            except Exception as e:
                bt.logging.debug(f"scoring: metagraph stake lookup failed: {e}")

            contexts = []
            required_by_hotkey: dict[str, float] = {}
            for spec in execs:
                gpu_model = _name_for_hash(spec.gpu_model_hash)
                # hotkey_ss58 lookup: from Executor row if heartbeat seen,
                # otherwise empty (UID will resolve via chain in weight loop)
                hotkey_ss58 = ""
                try:
                    from common.db import Executor
                    with db.session() as s:
                        row = s.query(Executor).filter_by(
                            executor_id=spec.executor_id,
                        ).first()
                        if row is not None:
                            hotkey_ss58 = row.hotkey_ss58 or ""
                except Exception:
                    pass

                miner_stake_tao = stake_by_hotkey.get(hotkey_ss58, 0.0) if hotkey_ss58 else 0.0

                ctx = build_context_from_db(
                    executor_id=spec.executor_id,
                    hotkey_ss58=hotkey_ss58,
                    gpu_model=gpu_model,
                    gpu_count=spec.gpu_count,
                    is_rented=spec.is_rented,
                    is_active=spec.is_active,
                    db=db, now_ts=now_ts,
                    miner_stake_tao=miner_stake_tao,
                    miner_address=getattr(spec, "miner_address", "") or "",
                    miner_uid=getattr(spec, "miner_uid", None),
                    endpoint=getattr(spec, "endpoint", "") or "",
                )
                contexts.append(ctx)
                if ctx.hotkey_ss58 and ctx.is_active:
                    required_by_hotkey[ctx.hotkey_ss58] = (
                        required_by_hotkey.get(ctx.hotkey_ss58, 0.0)
                        + stake_requirement_for_context(ctx, scoring_cfg)
                    )

            new_scores: dict = {}
            for ctx in contexts:
                if ctx.hotkey_ss58:
                    ctx.miner_required_stake_tao = required_by_hotkey.get(ctx.hotkey_ss58, 0.0)
                new_scores[ctx.executor_id] = score_one(ctx, cfg=scoring_cfg)

            global executor_scores
            executor_scores = new_scores

            browse.scoring_data = new_scores
            rent.scoring_data = new_scores

            n_pos = sum(1 for r in new_scores.values() if r.score > 0)
            bt.logging.debug(
                f"Scores updated: {n_pos}/{len(new_scores)} executors scoring > 0"
            )
            await asyncio.sleep(60)

        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"Scoring loop error: {e}; retry in 60s")
            await asyncio.sleep(60)


async def _validator_liveness_loop(
    db,
    *,
    service_name: str,
    instance_id: str,
    interval_s: float = 10.0,
) -> None:
    """Persist a lightweight validator heartbeat for outage reconstruction."""
    from common.db import record_validator_heartbeat

    interval_s = max(1.0, float(interval_s or 10.0))
    while True:
        try:
            await asyncio.to_thread(
                record_validator_heartbeat,
                db,
                service_name=service_name,
                instance_id=instance_id,
            )
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.debug(f"Validator liveness heartbeat failed: {e}")
            await asyncio.sleep(interval_s)


async def _offline_report_loop(db, registry, chain_snapshot, endpoint_health=None):
    """Report free executors offline only after heartbeat and endpoint failure.

    Public inventory hides stale executors through the live-score gate first.
    This loop is the slower on-chain cleanup path. A validator restart can make
    local heartbeat timestamps stale even while the miner is healthy, so stale
    heartbeat alone is not enough to submit an on-chain offline report.
    """
    import calendar as _calendar
    import time as _time

    check_interval = float(os.environ.get("OFFLINE_REPORT_CHECK_INTERVAL_S", "60"))
    # ComputeRegistry needs 3 recent reports within roughly 24h. Defaults below
    # produce those reports at about 12h, 18h, and 24h of continuous absence.
    stale_after = float(os.environ.get("OFFLINE_REPORT_AFTER_S", "43200"))
    repeat_after = float(os.environ.get("OFFLINE_REPORT_REPEAT_S", "21600"))
    failure_backoff_s = float(os.environ.get("OFFLINE_REPORT_FAILURE_BACKOFF_S", "3600"))
    boot_grace_s = float(os.environ.get("OFFLINE_REPORT_BOOT_GRACE_S", "900"))
    started_at = _time.time()
    last_report_at: dict[str, float] = {}
    last_failure_at: dict[str, float] = {}
    bt.logging.info(
        f"Offline report loop starting "
        f"(stale_after={stale_after:.0f}s repeat_after={repeat_after:.0f}s "
        f"failure_backoff={failure_backoff_s:.0f}s boot_grace={boot_grace_s:.0f}s)"
    )

    def _dt_to_unix(dt) -> float:
        if not dt:
            return 0.0
        if getattr(dt, "tzinfo", None) is not None:
            return float(dt.timestamp())
        return float(_calendar.timegm(dt.timetuple()))

    while True:
        try:
            await asyncio.sleep(check_interval)
            if db is None or registry is None or chain_snapshot is None:
                continue

            execs = chain_snapshot.all_executors()
            if not execs:
                continue
            ids = [e.executor_id for e in execs]

            from common.db import Executor
            with db.session() as s:
                rows = {
                    row.executor_id: row
                    for row in s.query(Executor).filter(Executor.executor_id.in_(ids)).all()
                }

            now = _time.time()
            if now - started_at < boot_grace_s:
                continue
            for spec in execs:
                if getattr(spec, "is_rented", False):
                    continue
                row = rows.get(spec.executor_id)
                last_seen_ts = _dt_to_unix(row.last_seen) if row else 0.0
                age = now - last_seen_ts if last_seen_ts else float("inf")
                if age < stale_after:
                    continue
                if now - last_report_at.get(spec.executor_id, 0.0) < repeat_after:
                    continue
                if now - last_failure_at.get(spec.executor_id, 0.0) < failure_backoff_s:
                    continue
                if endpoint_health is None:
                    bt.logging.warning(
                        f"offline_report skipped: endpoint health unavailable "
                        f"executor={spec.executor_id[:16]} heartbeat_age={age:.0f}s"
                    )
                    continue
                health_state = endpoint_health.state(spec.executor_id)
                if not health_state or not health_state.is_unhealthy:
                    status = getattr(health_state, "status", "unknown") if health_state else "unknown"
                    bt.logging.info(
                        f"offline_report skipped: endpoint not confirmed unhealthy "
                        f"executor={spec.executor_id[:16]} heartbeat_age={age:.0f}s "
                        f"endpoint_health={status}"
                    )
                    continue

                try:
                    tx = await asyncio.to_thread(
                        registry.report_offline,
                        bytes.fromhex(spec.executor_id),
                    )
                    last_report_at[spec.executor_id] = now
                    bt.logging.warning(
                        f"offline_report: executor={spec.executor_id[:16]} "
                        f"heartbeat_age={age:.0f}s tx={tx[:18]}"
                    )
                    try:
                        chain_snapshot.invalidate_executor(spec.executor_id)
                    except Exception:
                        pass
                except Exception as e:
                    msg = str(e)
                    if "submitted (hash=" in msg or "already known" in msg.lower():
                        last_report_at[spec.executor_id] = now
                    else:
                        last_failure_at[spec.executor_id] = now
                    bt.logging.warning(
                        f"offline_report failed: executor={spec.executor_id[:16]} "
                        f"heartbeat_age={age:.0f}s error={e}"
                    )
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"Offline report loop error: {e}; retry in 60s")
            await asyncio.sleep(60)


async def _prune_loop():
    """Prune bounded in-memory and DB telemetry state."""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            if analyzer:
                analyzer.prune_old(max_age_seconds=259200)  # 3 days in memory
            if os.environ.get("VALIDATOR_DB_PRUNE_ENABLED", "1") == "1":
                db_ref = getattr(app.state, "db", None)
                if db_ref is not None:
                    from common.db import prune_validator_db
                    summary = await asyncio.to_thread(prune_validator_db, db_ref)
                    if any(summary.values()):
                        bt.logging.info(f"Validator DB prune summary: {summary}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"Prune loop error: {e}; retry in 60s")
            await asyncio.sleep(60)


async def _metagraph_stats_loop(rpc, uid, interval: float = 60.0, wallet_ss58: str | None = None):
    """Periodic compact on-chain position summary.

    Logs at INFO every `interval` seconds:
        Metagraph | block=N | UID X | vtrust=… | dividends=… | emission=…α/tempo | stake=…α

    Operators want to see their UID's standing without opening a CLI; this
    is the "where do I stand right now" heartbeat. RPC failures are silent
    (DEBUG) so a transient testnet 429 doesn't pollute the log.
    """
    if rpc is None:
        return
    while True:
        try:
            await asyncio.sleep(interval)
            if uid is None and wallet_ss58:
                resolved_uid = await asyncio.to_thread(rpc.get_uid_for_hotkey, wallet_ss58)
                if resolved_uid is None:
                    continue
                uid = int(resolved_uid)
                bt.logging.info(f"Metagraph stats UID resolved: UID {uid}")
            if uid is None:
                continue
            # Single call: get_metagraph returns cached metagraph (3-min TTL)
            # if it was refreshed recently. Calling get_current_block separately
            # would race against verification-loop chain calls on the same websocket.
            mg = await asyncio.to_thread(rpc.get_metagraph, False)
            block = int(mg.block) if hasattr(mg, "block") else 0

            def _g(attr):
                return float(getattr(mg, attr)[uid]) if hasattr(mg, attr) and uid < len(getattr(mg, attr)) else 0.0

            parts = (
                f"UID {uid}",
                f"vtrust={_g('validator_trust'):.2f}",
                f"dividends={_g('dividends'):.4f}",
                f"emission={_g('emission'):.2f}α/tempo",
                f"stake={_g('stake'):.2f}α",
            )
            bt.logging.info(f"Metagraph | block={block} | {' | '.join(parts)}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.debug(f"Metagraph stats refresh failed: {e}")


async def _metagraph_hotkey_cache_warmup(rpc) -> None:
    """Populate the validator-local hotkey UID index after startup."""
    if rpc is None:
        return
    try:
        timeout_s = max(
            1.0,
            float(os.environ.get("NODEXO_METAGRAPH_WARMUP_TIMEOUT_S", "60")),
        )
        await asyncio.wait_for(
            asyncio.to_thread(rpc.get_metagraph, False),
            timeout=timeout_s,
        )
        bt.logging.info("Metagraph hotkey cache warmed")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        bt.logging.warning(f"Metagraph hotkey cache warmup failed: {e}")


async def _weight_setting_loop(rpc, wallet, netuid):
    """Set weights on Bittensor chain every tempo (~360 blocks / 72 min).

    Resolution chain: executor_id → miner_address (from ComputeRegistry) → UID
    (from ComputeRegistry.evmToUid). Multiple executors under the same miner
    address aggregate into one UID's weight.
    """
    from neurons.version import spec_version

    BLOCK_TIME = 12

    def _commit_reveal_pending(message: object) -> bool:
        text = str(message or "")
        return (
            "TooManyUnrevealedCommits" in text
            or "Maximum commit limit reached" in text
        )

    while True:
        try:
            from common.subnet_runtime_config import get_subnet_runtime_config
            weights_cfg = get_subnet_runtime_config().weights
            tempo_seconds = max(10, int(weights_cfg.set_weights_interval_blocks)) * BLOCK_TIME
            await asyncio.sleep(tempo_seconds)

            if not executor_scores:
                bt.logging.debug("No executor scores yet, skipping weight setting")
                continue

            from neurons.validator.api.routes import browse
            if not browse.registry_client:
                continue

            registry = browse.registry_client
            mg = await asyncio.to_thread(rpc.get_metagraph, True)
            try:
                hotkey_to_uid = {
                    hotkey: idx
                    for idx, hotkey in enumerate(list(getattr(mg, "hotkeys", [])))
                }
            except Exception:
                hotkey_to_uid = {}

            # ── Step 1: Map executor_id → miner_address → UID ──
            # Build address → UID lookup from the ComputeRegistry contract
            uid_scores: dict[int, float] = {}

            for eid, es in executor_scores.items():
                if es.score <= 0:
                    continue
                try:
                    if getattr(es, "hotkey_ss58", "") and es.hotkey_ss58 in hotkey_to_uid:
                        uid = hotkey_to_uid[es.hotkey_ss58]
                        uid_scores[uid] = uid_scores.get(uid, 0.0) + es.score
                        continue

                    spec = await asyncio.to_thread(registry.get_executor_info, bytes.fromhex(eid))
                    if not spec or not spec.miner_address:
                        continue

                    # Read UID for this miner address from ComputeRegistry
                    miner_addr = spec.miner_address
                    uid = await asyncio.to_thread(registry.evm_to_uid, miner_addr)

                    # Validate UID is in range
                    if uid >= mg.n.item():
                        continue

                    # Aggregate: if miner has multiple executors, sum their scores
                    if uid in uid_scores:
                        uid_scores[uid] += es.score
                    else:
                        uid_scores[uid] = es.score

                except Exception as e:
                    bt.logging.debug(f"UID resolution failed for {eid[:16]}: {e}")
                    continue

            if not uid_scores:
                bt.logging.debug("No UID scores resolved, skipping weight setting")
                continue

            # Normalize miner weights to sum to 1, then redirect the configured
            # fraction to the subnet-owner UID unless an explicit burn UID is
            # set in runtime config.
            total = sum(uid_scores.values())
            if total > 0:
                uid_scores = {uid: score / total for uid, score in uid_scores.items()}

            burn_fraction = max(0.0, min(0.95, float(weights_cfg.emission_burn_fraction)))
            if burn_fraction > 0 and uid_scores:
                try:
                    n_uids = int(mg.n.item())
                except Exception:
                    n_uids = 0
                burn_uid = (
                    int(weights_cfg.burn_uid)
                    if weights_cfg.burn_uid is not None
                    else await asyncio.to_thread(rpc.get_subnet_owner_uid)
                )
                if burn_uid is None:
                    bt.logging.warning("Skipping configured emission burn: subnet owner UID unresolved")
                    burn_uid = -1
                if 0 <= burn_uid < n_uids:
                    for uid in list(uid_scores.keys()):
                        uid_scores[uid] *= (1.0 - burn_fraction)
                    uid_scores[burn_uid] = uid_scores.get(burn_uid, 0.0) + burn_fraction
                else:
                    bt.logging.warning(
                        f"Skipping configured emission burn: burn_uid={burn_uid} "
                        f"outside metagraph size {n_uids}"
                    )

            uids = list(uid_scores.keys())
            weights = list(uid_scores.values())

            if hasattr(rpc, "set_weights_result"):
                success, message = await asyncio.to_thread(
                    rpc.set_weights_result,
                    wallet=wallet,
                    uids=uids,
                    weights=weights,
                    version_key=spec_version,
                )
            else:
                success = await asyncio.to_thread(
                    rpc.set_weights,
                    wallet=wallet,
                    uids=uids,
                    weights=weights,
                    version_key=spec_version,
                )
                message = ""

            top_uid = uids[weights.index(max(weights))] if weights else -1
            top_weight = max(weights) if weights else 0.0
            if success:
                bt.logging.info(
                    f"Weights set: OK ({len(uids)} UIDs, top: UID {top_uid} = {top_weight:.4f})"
                )
            elif _commit_reveal_pending(message):
                bt.logging.warning(
                    "Weights commit-reveal pending: Subtensor rejected this "
                    f"commit because unrevealed commits are still pending; "
                    f"next attempt follows configured cadence ({tempo_seconds}s). "
                    f"({len(uids)} UIDs, top: UID {top_uid} = {top_weight:.4f})"
                )
            else:
                bt.logging.error(
                    f"Weights set: FAILED ({len(uids)} UIDs, top: UID {top_uid} = {top_weight:.4f}): {message}"
                )

        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"Weight setting error: {e}; retry in 60s")
            await asyncio.sleep(60)


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    bt.logging.enable_info()
    runtime_args = _parse_runtime_args()
    app.state.auto_update_enabled = bool(runtime_args.auto_update)
    app.state.auto_update_interval = int(runtime_args.auto_update_interval)
    app.state.auto_update_restart_delay = int(runtime_args.auto_update_restart_delay)
    if not os.environ.get("NODEXO_ADMIN_TOKEN") and not os.environ.get("NODEXO_ADMIN_HOTKEYS"):
        bt.logging.warning(
            "No validator admin auth configured. This is fine for a verifier/"
            "weight-setting validator, but rental control, sybil admin, and "
            "history/admin endpoints will return 403. Set NODEXO_ADMIN_TOKEN or "
            "NODEXO_ADMIN_HOTKEYS on the primary validator."
        )

    # ── Pre-init ALL blocking chain work before uvicorn starts ──
    network = runtime_args.subtensor_network
    chain_endpoint = runtime_args.subtensor_endpoint
    connect_target = chain_endpoint or network
    netuid_env = str(runtime_args.netuid or _default_netuid_for_network(network) or "")
    wallet_env = runtime_args.wallet
    hotkey_env = runtime_args.hotkey

    if network and netuid_env:
        from common.chain.rpc import SubtensorRPC

        SubCls = getattr(bt, "Subtensor", None) or bt.subtensor
        WalletCls = getattr(bt, "Wallet", None) or bt.wallet
        AxonCls = getattr(bt, "Axon", None) or getattr(bt, "axon", None)

        # Connect with retry-on-429. The testnet RPC endpoint
        # (test.chain.opentensor.ai) rate-limits aggressively; without
        # this, PM2's restart-on-crash loops the daemon every <1s, each
        # retry hits the endpoint and DEEPENS the rate-limit window
        # instead of letting it decay. We saw this in 2026-05-23 testing:
        # 55 restarts in a row, all rejected with HTTP 429 on the
        # websocket handshake. With this backoff the process holds open
        # across the rate-limit cooldown so PM2 doesn't churn.
        sub = None
        delay = 5.0
        max_attempts = 12  # ~10 minutes worth of cumulative backoff
        non_rate_limit_retries_left = 2  # fail fast on real misconfig
        for attempt in range(1, max_attempts + 1):
            try:
                if chain_endpoint:
                    bt.logging.info(
                        f"Connecting to subtensor ({network}, endpoint={chain_endpoint}, "
                        f"attempt {attempt}/{max_attempts})..."
                    )
                else:
                    bt.logging.info(
                        f"Connecting to subtensor ({network}, attempt {attempt}/{max_attempts})..."
                    )
                sub = SubCls(network=connect_target)
                break
            except Exception as e:
                # Match the websockets InvalidStatus 429 by string — the
                # exception type is buried under bittensor / async-substrate
                # layers and varies by version. The substring is stable.
                msg = str(e)
                # Strip control chars (\r\n etc) from log output —
                # the RPC endpoint controls this string, so CRLF could
                # otherwise inject lines into log scrapers.
                msg_safe = msg.encode("unicode_escape").decode()[:200]
                is_rate_limited = ("429" in msg) or ("Too Many Requests" in msg)
                if attempt == max_attempts:
                    bt.logging.error(
                        f"Subtensor connect failed after {max_attempts} attempts: {msg_safe}"
                    )
                    raise
                if not is_rate_limited:
                    non_rate_limit_retries_left -= 1
                    if non_rate_limit_retries_left < 0:
                        bt.logging.error(
                            f"Subtensor connect: non-429 failure budget exhausted "
                            f"({type(e).__name__}); failing fast: {msg_safe}"
                        )
                        raise
                    bt.logging.warning(
                        f"Subtensor connect failed ({type(e).__name__}); "
                        f"sleeping {delay:.0f}s before retry: {msg_safe}"
                    )
                else:
                    bt.logging.warning(
                        f"Subtensor 429-rate-limited; sleeping {delay:.0f}s before retry"
                    )
                import time as _t
                _t.sleep(delay)
                delay = min(delay * 2, 120.0)
        rpc = SubtensorRPC(
            network=network,
            netuid=int(netuid_env),
            subtensor=sub,
            chain_endpoint=chain_endpoint,
        )
        bt.logging.info(f"Subtensor connected (block={sub.get_current_block()})")

        wallet = WalletCls(name=wallet_env, hotkey=hotkey_env)
        if AxonCls and runtime_args.discovery_mode in {"native", "both"}:
            try:
                endpoint = runtime_args.endpoint or f"http://0.0.0.0:{runtime_args.port}"
                _serve_native_validator_axon(
                    rpc, AxonCls, wallet, int(netuid_env), endpoint, int(runtime_args.port),
                )
            except Exception as e:
                bt.logging.warning(f"serve_axon: {e}")
        elif AxonCls:
            bt.logging.info(
                f"Native validator axon disabled (discovery_mode={runtime_args.discovery_mode})"
            )

        # Store on app for lifespan to pick up
        app._vali_pre_rpc = rpc
        app._vali_pre_wallet = wallet

    import uvicorn
    uvicorn.run(
        app, host=runtime_args.bind_host, port=int(runtime_args.port),
        log_level="warning", access_log=False, loop="asyncio",
    )
