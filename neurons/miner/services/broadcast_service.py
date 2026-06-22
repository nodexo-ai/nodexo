"""
Broadcast service — pushes proof commitments and recipes to all validators.

Discovers validators from ValidatorRegistry on-chain, sends two-phase
proof data to each validator's API.

Reference: see protocol design notes for proof submission patterns.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import bittensor as bt

from common.types import ProofCommitment, ProofRecipe, ValidatorEndpoint


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


POST_MAX_ATTEMPTS = _env_int("NODEXO_VALIDATOR_POST_MAX_ATTEMPTS", 2)
# Timeout for individual validator POST. Validator ingress can spend up to
# ~21s on first-bind registry/auth checks (registry lookup + ownership check),
# so miner defaults must be longer than the validator-side auth budget.
POST_TIMEOUT_S = _env_float("NODEXO_VALIDATOR_POST_TIMEOUT_SECONDS", 30)
POST_CONNECT_TIMEOUT_S = _env_float("NODEXO_VALIDATOR_CONNECT_TIMEOUT_SECONDS", 2)
REQUEST_TIMEOUT = aiohttp.ClientTimeout(
    total=POST_TIMEOUT_S,
    connect=POST_CONNECT_TIMEOUT_S,
)
RECEIPT_POST_MAX_ATTEMPTS = _env_int("NODEXO_RECEIPT_POST_MAX_ATTEMPTS", 3)
RECEIPT_POST_TIMEOUT_S = _env_float("NODEXO_RECEIPT_POST_TIMEOUT_SECONDS", 6.0)
RECEIPT_CONNECT_TIMEOUT_S = _env_float("NODEXO_RECEIPT_CONNECT_TIMEOUT_SECONDS", 2.0)
RECEIPT_REQUEST_TIMEOUT = aiohttp.ClientTimeout(
    total=RECEIPT_POST_TIMEOUT_S,
    connect=RECEIPT_CONNECT_TIMEOUT_S,
)
BROADCAST_DEADLINE_S = _env_float(
    "NODEXO_VALIDATOR_BROADCAST_DEADLINE_SECONDS",
    max(30.0, POST_TIMEOUT_S * max(1, POST_MAX_ATTEMPTS) + 10.0),
)
RECEIPT_BROADCAST_DEADLINE_S = _env_float(
    "NODEXO_RECEIPT_BROADCAST_DEADLINE_SECONDS",
    max(1.0, RECEIPT_POST_TIMEOUT_S * max(1, RECEIPT_POST_MAX_ATTEMPTS) + 3.0),
)
DISCOVERY_TIMEOUT_S = _env_float("NODEXO_VALIDATOR_DISCOVERY_TIMEOUT_SECONDS", 3)
DISCOVERY_RETRY_S = _env_float("NODEXO_VALIDATOR_DISCOVERY_RETRY_SECONDS", 120)
DISCOVERY_READY_RETRY_S = _env_float("NODEXO_VALIDATOR_DISCOVERY_READY_RETRY_SECONDS", 5)
DISCOVERY_READY_TIMEOUT_S = _env_float(
    "NODEXO_VALIDATOR_DISCOVERY_READY_TIMEOUT_SECONDS", 0,
)
DISCOVERY_RETRY_MAX_S = _env_float("NODEXO_VALIDATOR_DISCOVERY_RETRY_MAX_SECONDS", 1800)
DISCOVERY_CACHE_TTL_S = _env_float("NODEXO_VALIDATOR_DISCOVERY_CACHE_SECONDS", 1800)
ENDPOINT_FAILURE_THRESHOLD = _env_int("NODEXO_VALIDATOR_ENDPOINT_FAILURE_THRESHOLD", 2)
ENDPOINT_TRANSIENT_FAILURE_THRESHOLD = _env_int(
    "NODEXO_VALIDATOR_ENDPOINT_TRANSIENT_FAILURE_THRESHOLD", 12,
)
ENDPOINT_QUARANTINE_S = _env_float("NODEXO_VALIDATOR_ENDPOINT_QUARANTINE_SECONDS", 900)


@dataclass(frozen=True)
class _BroadcastFailure:
    kind: str

    @property
    def is_hard(self) -> bool:
        return self.kind == "hard"

    @property
    def is_transient(self) -> bool:
        return self.kind == "transient"


def _broadcast_failure_from_exception(exc: BaseException) -> _BroadcastFailure:
    if isinstance(exc, asyncio.TimeoutError):
        return _BroadcastFailure("transient")
    if isinstance(exc, aiohttp.ClientConnectorError):
        return _BroadcastFailure("transient")
    if isinstance(exc, aiohttp.ClientOSError):
        return _BroadcastFailure("transient")
    return _BroadcastFailure("transient")


def _coerce_broadcast_failure(result) -> _BroadcastFailure | None:
    if result is True:
        return None
    if isinstance(result, _BroadcastFailure):
        return result
    if isinstance(result, BaseException):
        return _broadcast_failure_from_exception(result)
    # Preserve older tests and any future callers that still pass False as a
    # hard endpoint failure.
    if result is False:
        return _BroadcastFailure("hard")
    return _BroadcastFailure("transient")


def _is_native_validator(v: ValidatorEndpoint) -> bool:
    return str(getattr(v, "address", "") or "").startswith("axon:")


def _has_evm_or_configured_validator(validators: list[ValidatorEndpoint]) -> bool:
    return any(not _is_native_validator(v) for v in validators or [])


def _valid_endpoint_url(endpoint: str) -> bool:
    parsed = urlparse((endpoint or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        _ = parsed.port
    except ValueError:
        return False
    return True


def _default_cache_path() -> str:
    explicit = os.environ.get("NODEXO_VALIDATOR_ENDPOINT_CACHE_PATH", "").strip()
    if explicit:
        return explicit
    data_dir = os.environ.get("NODEXO_DATA_DIR", "").strip()
    if not data_dir:
        return ""
    return str(Path(data_dir).expanduser() / "validator_endpoints_cache.json")


class BroadcastService:
    """Broadcasts proof data to all active validators."""

    def __init__(
        self,
        discover_validators: callable,  # Returns list[ValidatorEndpoint]
        sign_payload: Optional[callable] = None,  # SR25519 signing
        endpoint: str = "",  # Public miner endpoint reported in heartbeats
    ):
        self._discover_validators = discover_validators
        self._sign_payload = sign_payload
        self._endpoint = endpoint.strip()
        self._session: Optional[aiohttp.ClientSession] = None
        self._endpoint_failures: dict[str, int] = {}
        self._endpoint_transient_failures: dict[str, int] = {}
        self._endpoint_quarantine_until: dict[str, float] = {}
        self._validator_cache: list[ValidatorEndpoint] = self._load_validator_cache()
        self._cache_updated_at: float = 0
        self._cache_ttl: float = DISCOVERY_CACHE_TTL_S
        self._next_refresh_after: float = 0
        self._refresh_failures: int = 0
        self._refresh_lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None
        self._receipt_locks: dict[tuple[str, str, int, int], asyncio.Lock] = {}
        self._receipt_lock_seen_at: dict[tuple[str, str, int, int], float] = {}
        self._receipt_pass0_events: dict[tuple[str, str, int, int], asyncio.Event] = {}
        self._receipt_pass0_result: dict[tuple[str, str, int, int], object] = {}

    @property
    def endpoint(self) -> str:
        """Public miner endpoint advertised to validators."""
        return self._endpoint

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            force_close = _env_bool("NODEXO_VALIDATOR_BROADCAST_FORCE_CLOSE", False)
            connector_kwargs = {
                "force_close": force_close,
                "limit": int(os.environ.get("NODEXO_VALIDATOR_BROADCAST_CONN_LIMIT", "256")),
                "limit_per_host": int(os.environ.get("NODEXO_VALIDATOR_BROADCAST_CONN_LIMIT_PER_HOST", "16")),
                "ttl_dns_cache": int(os.environ.get("NODEXO_VALIDATOR_BROADCAST_DNS_TTL_SECONDS", "300")),
                "enable_cleanup_closed": True,
            }
            if not force_close:
                connector_kwargs["keepalive_timeout"] = _env_float(
                    "NODEXO_VALIDATOR_BROADCAST_KEEPALIVE_SECONDS", 75.0,
                )
            connector = aiohttp.TCPConnector(**connector_kwargs)
            self._session = aiohttp.ClientSession(
                timeout=REQUEST_TIMEOUT, connector=connector,
            )

    def _clean_validator_list(self, validators: list[ValidatorEndpoint]) -> list[ValidatorEndpoint]:
        cleaned: list[ValidatorEndpoint] = []
        seen: set[tuple[str, int]] = set()
        for v in validators or []:
            endpoint = str(getattr(v, "proxy_endpoint", "") or "").strip().rstrip("/")
            if not endpoint or not _valid_endpoint_url(endpoint):
                continue
            uid = int(getattr(v, "uid", -1) or -1)
            key = (endpoint.lower(), uid)
            if key in seen:
                continue
            seen.add(key)
            v.proxy_endpoint = endpoint
            cleaned.append(v)
        return cleaned

    def _load_validator_cache(self) -> list[ValidatorEndpoint]:
        path = _default_cache_path()
        if not path:
            return []
        try:
            data = json.loads(Path(path).read_text())
            rows = data.get("validators") if isinstance(data, dict) else data
            validators = [
                ValidatorEndpoint(
                    address=str(row.get("address") or ""),
                    proxy_endpoint=str(row.get("proxy_endpoint") or ""),
                    uid=int(row.get("uid", -1) or -1),
                    is_active=bool(row.get("is_active", True)),
                )
                for row in rows or []
                if isinstance(row, dict)
            ]
            cleaned = self._clean_validator_list(validators)
            self._load_endpoint_health_state(data, cleaned)
            if cleaned:
                bt.logging.info(
                    f"Loaded validator endpoint cache: {len(cleaned)} endpoint(s)"
                )
            return cleaned
        except FileNotFoundError:
            return []
        except Exception as e:
            bt.logging.warning(f"Ignoring invalid validator endpoint cache {path}: {e}")
            return []

    def set_validator_cache(
        self,
        validators: list[ValidatorEndpoint],
        *,
        source: str = "discovery",
        save: bool = True,
    ) -> list[ValidatorEndpoint]:
        """Replace the broadcast cache with a freshly discovered validator list."""
        fresh = self._clean_validator_list(validators)
        if not fresh:
            return []
        self._validator_cache = fresh
        self._prune_endpoint_health_state(fresh)
        now = time.time()
        self._cache_updated_at = now
        self._next_refresh_after = now + self._cache_ttl
        self._refresh_failures = 0
        if save:
            self._save_validator_cache()
        bt.logging.info(
            f"Validator endpoint cache set from {source}: {len(fresh)} endpoint(s)"
        )
        return fresh

    def current_validators(self) -> list[ValidatorEndpoint]:
        """Return currently usable non-quarantined validator endpoints."""
        return list(self._validators_for_broadcast())

    def _save_validator_cache(self) -> None:
        path = _default_cache_path()
        if not path or not self._validator_cache:
            return
        try:
            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": time.time(),
                "validators": [asdict(v) for v in self._validator_cache],
                "endpoint_health": self._endpoint_health_payload(),
            }
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True))
            tmp.replace(target)
        except Exception as e:
            bt.logging.debug(f"validator endpoint cache write failed: {e}")

    @staticmethod
    def _endpoint_key_for_url(endpoint: str) -> str:
        return str(endpoint or "").strip().rstrip("/").lower()

    def _endpoint_key(self, validator: ValidatorEndpoint) -> str:
        return self._endpoint_key_for_url(getattr(validator, "proxy_endpoint", ""))

    def _load_endpoint_health_state(
        self,
        data,
        validators: list[ValidatorEndpoint],
    ) -> None:
        if not isinstance(data, dict):
            return
        valid_keys = {self._endpoint_key(v) for v in validators}
        rows = data.get("endpoint_health") or {}
        if not isinstance(rows, dict):
            return
        now = time.time()
        for key, row in rows.items():
            key = self._endpoint_key_for_url(key)
            if key not in valid_keys or not isinstance(row, dict):
                continue
            failures = int(row.get("failures") or 0)
            transient_failures = int(row.get("transient_failures") or 0)
            quarantine_until = float(row.get("quarantine_until") or 0)
            if failures > 0:
                self._endpoint_failures[key] = failures
            if transient_failures > 0:
                self._endpoint_transient_failures[key] = transient_failures
            if quarantine_until > now:
                self._endpoint_quarantine_until[key] = quarantine_until

    def _endpoint_health_payload(self) -> dict:
        keys = (
            set(self._endpoint_failures)
            | set(self._endpoint_transient_failures)
            | set(self._endpoint_quarantine_until)
        )
        now = time.time()
        payload = {}
        for key in keys:
            failures = int(self._endpoint_failures.get(key, 0) or 0)
            transient_failures = int(self._endpoint_transient_failures.get(key, 0) or 0)
            quarantine_until = float(self._endpoint_quarantine_until.get(key, 0) or 0)
            if failures <= 0 and transient_failures <= 0 and quarantine_until <= now:
                continue
            payload[key] = {
                "failures": failures,
                "transient_failures": transient_failures,
                "quarantine_until": quarantine_until,
            }
        return payload

    def _prune_endpoint_health_state(self, validators: list[ValidatorEndpoint]) -> None:
        valid_keys = {self._endpoint_key(v) for v in validators}
        for key in list(self._endpoint_failures):
            if key not in valid_keys:
                self._endpoint_failures.pop(key, None)
        for key in list(self._endpoint_transient_failures):
            if key not in valid_keys:
                self._endpoint_transient_failures.pop(key, None)
        for key in list(self._endpoint_quarantine_until):
            if key not in valid_keys:
                self._endpoint_quarantine_until.pop(key, None)

    def _is_endpoint_quarantined(self, key: str, now: float | None = None) -> bool:
        if not key:
            return False
        now = time.time() if now is None else now
        until = float(self._endpoint_quarantine_until.get(key, 0) or 0)
        if until <= 0:
            return False
        if until <= now:
            self._endpoint_quarantine_until.pop(key, None)
            return False
        return True

    def _validators_for_broadcast(self) -> list[ValidatorEndpoint]:
        now = time.time()
        selected: list[ValidatorEndpoint] = []
        skipped = 0
        for validator in self._validator_cache:
            key = self._endpoint_key(validator)
            if self._is_endpoint_quarantined(key, now):
                skipped += 1
                continue
            selected.append(validator)
        if skipped:
            bt.logging.debug(
                f"Skipped {skipped} quarantined validator endpoint(s) for broadcast"
            )
        return selected

    def _record_broadcast_results(
        self,
        validators: list[ValidatorEndpoint],
        results: list,
    ) -> None:
        changed = False
        now = time.time()
        hard_threshold = max(1, int(ENDPOINT_FAILURE_THRESHOLD))
        transient_threshold = max(1, int(ENDPOINT_TRANSIENT_FAILURE_THRESHOLD))
        quarantine_s = max(1.0, float(ENDPOINT_QUARANTINE_S))
        for validator, result in zip(validators, results):
            key = self._endpoint_key(validator)
            if not key:
                continue
            endpoint = getattr(validator, "proxy_endpoint", key)
            if result is True:
                if self._endpoint_failures.pop(key, None) is not None:
                    changed = True
                if self._endpoint_transient_failures.pop(key, None) is not None:
                    changed = True
                if self._endpoint_quarantine_until.pop(key, None) is not None:
                    changed = True
                continue

            failure = _coerce_broadcast_failure(result)
            if failure is None:
                continue

            if failure.kind == "client":
                if self._endpoint_failures.pop(key, None) is not None:
                    changed = True
                if self._endpoint_transient_failures.pop(key, None) is not None:
                    changed = True
                continue

            if failure.is_hard:
                failures = int(self._endpoint_failures.get(key, 0) or 0) + 1
                self._endpoint_failures[key] = failures
                threshold = hard_threshold
                failure_label = "hard"
            else:
                failures = int(self._endpoint_transient_failures.get(key, 0) or 0) + 1
                self._endpoint_transient_failures[key] = failures
                threshold = transient_threshold
                failure_label = "transient"
            changed = True
            if failures >= threshold:
                until = now + quarantine_s
                previous_until = float(self._endpoint_quarantine_until.get(key, 0) or 0)
                self._endpoint_quarantine_until[key] = until
                if previous_until <= now:
                    bt.logging.warning(
                        f"Quarantining validator endpoint {endpoint} for "
                        f"{quarantine_s:.0f}s after {failures} {failure_label} failed broadcast(s)"
                    )

        if changed:
            self._save_validator_cache()

    def _next_retry_delay(self) -> float:
        delay = DISCOVERY_RETRY_S * (2 ** max(0, self._refresh_failures - 1))
        return min(DISCOVERY_RETRY_MAX_S, delay)

    async def _run_broadcast_posts(
        self,
        coros: list,
        *,
        timeout_s: float | None = None,
    ) -> list:
        if not coros:
            return []
        if timeout_s is None:
            timeout_s = BROADCAST_DEADLINE_S
        tasks = [asyncio.create_task(coro) for coro in coros]
        try:
            return await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            done = sum(1 for task in tasks if task.done() and not task.cancelled())
            bt.logging.warning(
                f"Validator broadcast deadline hit ({timeout_s:.1f}s); "
                f"completed {done}/{len(tasks)} posts"
            )
            results = []
            for task in tasks:
                if not task.done() or task.cancelled():
                    results.append(_BroadcastFailure("transient"))
                    continue
                try:
                    results.append(task.result())
                except Exception as e:
                    results.append(_broadcast_failure_from_exception(e))
            return results

    def _receipt_lock_key(
        self,
        validator: ValidatorEndpoint,
        receipt: dict,
    ) -> tuple[str, str, int, int]:
        endpoint = self._endpoint_key(validator)
        executor_id = str(receipt.get("executor_id") or "")
        try:
            epoch_id = int(receipt.get("epoch_id") or 0)
        except Exception:
            epoch_id = 0
        try:
            gpu_index = int(receipt.get("gpu_index") or -1)
        except Exception:
            gpu_index = -1
        return endpoint, executor_id, epoch_id, gpu_index

    def _prune_receipt_locks(self) -> None:
        if len(self._receipt_locks) < 10000:
            return
        cutoff = time.time() - 3600.0
        for key, seen_at in list(self._receipt_lock_seen_at.items()):
            if seen_at >= cutoff:
                continue
            lock = self._receipt_locks.get(key)
            if lock is not None and lock.locked():
                continue
            self._receipt_locks.pop(key, None)
            self._receipt_lock_seen_at.pop(key, None)
            self._receipt_pass0_events.pop(key, None)
            self._receipt_pass0_result.pop(key, None)

    @staticmethod
    def _receipt_pass0_wait_s() -> float:
        try:
            return max(0.1, float(os.environ.get(
                "NODEXO_RECEIPT_PASS0_WAIT_S",
                str(RECEIPT_BROADCAST_DEADLINE_S),
            )))
        except Exception:
            return RECEIPT_BROADCAST_DEADLINE_S

    async def _post_receipt_ordered(
        self,
        validator: ValidatorEndpoint,
        url: str,
        payload_json: str,
        receipt: dict,
        headers: dict,
    ) -> bool | _BroadcastFailure:
        key = self._receipt_lock_key(validator, receipt)
        try:
            pass_index = int(receipt.get("u") or 0)
        except Exception:
            pass_index = 0
        pass0_event = self._receipt_pass0_events.get(key)
        if pass0_event is None:
            pass0_event = asyncio.Event()
            self._receipt_pass0_events[key] = pass0_event

        if pass_index == 1 and not pass0_event.is_set():
            try:
                await asyncio.wait_for(
                    pass0_event.wait(),
                    timeout=self._receipt_pass0_wait_s(),
                )
            except asyncio.TimeoutError:
                bt.logging.warning(
                    f"Receipt pass_1 held because pass_0 did not finish first: "
                    f"validator={validator.address[:16]} "
                    f"executor={str(receipt.get('executor_id') or '')[:16]} "
                    f"epoch={receipt.get('epoch_id')} gpu={receipt.get('gpu_index')}"
                )
                return _BroadcastFailure("transient")
        if pass_index == 1:
            pass0_result = self._receipt_pass0_result.get(key)
            if isinstance(pass0_result, _BroadcastFailure) and pass0_result.kind == "client":
                bt.logging.warning(
                    f"Receipt pass_1 skipped because pass_0 was rejected by validator: "
                    f"validator={validator.address[:16]} "
                    f"executor={str(receipt.get('executor_id') or '')[:16]} "
                    f"epoch={receipt.get('epoch_id')} gpu={receipt.get('gpu_index')}"
                )
                return _BroadcastFailure("client")
            if pass0_result is not True:
                bt.logging.debug(
                    f"Receipt pass_1 proceeding after ambiguous pass_0 result: "
                    f"validator={validator.address[:16]} "
                    f"executor={str(receipt.get('executor_id') or '')[:16]} "
                    f"epoch={receipt.get('epoch_id')} gpu={receipt.get('gpu_index')}"
                )

        lock = self._receipt_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._receipt_locks[key] = lock
        self._receipt_lock_seen_at[key] = time.time()

        async with lock:
            self._receipt_lock_seen_at[key] = time.time()
            result: bool | _BroadcastFailure = _BroadcastFailure("transient")
            try:
                result = await self._post_raw(
                    url,
                    payload_json,
                    validator.address[:16],
                    headers,
                    timeout=RECEIPT_REQUEST_TIMEOUT,
                    max_attempts=RECEIPT_POST_MAX_ATTEMPTS,
                )
                return result
            finally:
                if pass_index == 0:
                    self._receipt_pass0_result[key] = result
                    pass0_event.set()

    async def _refresh_validators_now(self):
        async with self._refresh_lock:
            now = time.time()
            if now - self._cache_updated_at < self._cache_ttl and self._validator_cache:
                return
            if now < self._next_refresh_after:
                return

            try:
                fresh = await asyncio.wait_for(
                    asyncio.to_thread(self._discover_validators),
                    timeout=DISCOVERY_TIMEOUT_S,
                )
                fresh = self._clean_validator_list(fresh)
                if fresh:
                    if (
                        self._validator_cache
                        and _has_evm_or_configured_validator(self._validator_cache)
                        and not _has_evm_or_configured_validator(fresh)
                    ):
                        self._refresh_failures += 1
                        self._next_refresh_after = now + self._next_retry_delay()
                        bt.logging.warning(
                            "Validator discovery returned native-only fallback while "
                            "an EVM/configured validator cache exists; keeping current cache"
                        )
                        return
                    self.set_validator_cache(
                        fresh,
                        source="background discovery",
                        save=True,
                    )
                    bt.logging.debug(
                        f"Refreshed validator list: {len(self._validator_cache)} active validators"
                    )
                elif not self._validator_cache:
                    self._refresh_failures += 1
                    self._next_refresh_after = now + self._next_retry_delay()
                    bt.logging.warning("Validator discovery returned empty list and cache is empty")
                else:
                    self._refresh_failures += 1
                    self._next_refresh_after = now + self._next_retry_delay()
                    bt.logging.debug(
                        f"Validator discovery returned empty list; keeping {len(self._validator_cache)} cached"
                    )
            except asyncio.TimeoutError:
                self._refresh_failures += 1
                self._next_refresh_after = now + self._next_retry_delay()
                bt.logging.warning(
                    f"Validator discovery timed out after {DISCOVERY_TIMEOUT_S:.1f}s; "
                    f"keeping {len(self._validator_cache)} cached"
                )
            except Exception as e:
                self._refresh_failures += 1
                self._next_refresh_after = now + self._next_retry_delay()
                bt.logging.error(f"Failed to refresh validators: {e}")

    def _start_background_refresh(self) -> None:
        if self._refresh_task is not None and not self._refresh_task.done():
            return

        async def _runner():
            try:
                await self._refresh_validators_now()
            except Exception as e:
                bt.logging.debug(f"background validator discovery failed: {e}")

        self._refresh_task = asyncio.create_task(_runner())

    async def _refresh_validators(self):
        """Keep validator cache fresh without stalling normal broadcasts.

        If a cache exists, broadcasts use it immediately and refresh discovery in
        the background. Only the first ever discovery waits, and even that is
        bounded by DISCOVERY_TIMEOUT_S.
        """
        now = time.time()
        if self._validator_cache:
            if (
                now - self._cache_updated_at >= self._cache_ttl
                and now >= self._next_refresh_after
            ):
                self._start_background_refresh()
            return
        if now < self._next_refresh_after:
            return
        await self._refresh_validators_now()

    async def refresh_until_ready(
        self,
        *,
        discover_validators: Optional[callable] = None,
        timeout_s: Optional[float] = None,
        retry_s: Optional[float] = None,
    ) -> list[ValidatorEndpoint]:
        """Block startup until fresh validator discovery succeeds.

        Disk cache is deliberately ignored here. Startup proofs must only begin
        after a fresh chain/RPC-backed discovery so stale validator endpoints
        cannot receive or delay proof broadcasts.
        """
        discover = discover_validators or self._discover_validators
        timeout = DISCOVERY_READY_TIMEOUT_S if timeout_s is None else float(timeout_s)
        retry = DISCOVERY_READY_RETRY_S if retry_s is None else float(retry_s)
        retry = max(0.1, retry)
        deadline = None if timeout <= 0 else time.monotonic() + timeout
        attempts = 0

        while True:
            attempts += 1
            try:
                fresh = await asyncio.wait_for(
                    asyncio.to_thread(discover),
                    timeout=DISCOVERY_TIMEOUT_S,
                )
                fresh = self._clean_validator_list(fresh)
                if fresh:
                    return self.set_validator_cache(
                        fresh,
                        source="startup discovery",
                        save=True,
                    )
                bt.logging.warning(
                    "Startup validator discovery returned no usable endpoints; "
                    f"retrying in {retry:.1f}s"
                )
            except asyncio.TimeoutError:
                bt.logging.warning(
                    f"Startup validator discovery timed out after "
                    f"{DISCOVERY_TIMEOUT_S:.1f}s; retrying in {retry:.1f}s"
                )
            except Exception as e:
                bt.logging.warning(
                    f"Startup validator discovery failed ({type(e).__name__}): "
                    f"{e}; retrying in {retry:.1f}s"
                )

            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(
                    "fresh validator discovery did not become ready within "
                    f"{timeout:.1f}s after {attempts} attempt(s)"
                )
            sleep_s = retry
            if deadline is not None:
                sleep_s = min(sleep_s, max(0.1, deadline - time.monotonic()))
            await asyncio.sleep(sleep_s)

    async def broadcast_receipt(self, receipt: dict) -> dict:
        """Phase A: per-GPU per-pass receipt to all validators.

        Sent immediately after each GPU completes pass_0 or pass_1.
        Contains: executor_id, epoch_id, gpu_index, u, root, T_commit, ts_ns.
        Validator records arrival time for per-GPU delta measurement.
        """
        await self._refresh_validators()
        await self._ensure_session()

        payload_json = json.dumps(receipt, sort_keys=True)

        extra_headers = {"Content-Type": "application/json"}
        if self._sign_payload:
            extra_headers.update(self._sign_payload(payload_json))

        tasks = []
        posted_validators = []
        validators = self._validators_for_broadcast()
        for v in validators:
            if not v.proxy_endpoint:
                continue
            url = f"{v.proxy_endpoint.rstrip('/')}/proofs/receipt"
            tasks.append(self._post_receipt_ordered(
                v,
                url,
                payload_json,
                receipt,
                extra_headers,
            ))
            posted_validators.append(v)
        if not tasks:
            bt.logging.warning(
                f"Receipt broadcast has no validator endpoints: "
                f"gpu={receipt.get('gpu_index', -1)} u={receipt.get('u', -1)} "
                f"epoch={receipt.get('epoch_id', 0)}"
            )
            return {"success": 0, "total": 0}

        results = await self._run_broadcast_posts(
            tasks,
            timeout_s=RECEIPT_BROADCAST_DEADLINE_S,
        )
        self._record_broadcast_results(posted_validators, results)
        self._prune_receipt_locks()
        success = sum(1 for r in results if r is True)
        bt.logging.debug(
            f"Receipt broadcast: gpu={receipt.get('gpu_index', -1)} u={receipt.get('u', -1)} → {success}/{len(tasks)} validators"
        )
        return {"success": success, "total": len(tasks)}

    async def broadcast_commitment(self, commitment: ProofCommitment):
        """Phase A: broadcast pass_0 commitment to all validators."""
        await self._refresh_validators()
        await self._ensure_session()

        payload = _serialize_commitment(commitment)
        payload_json = json.dumps(payload, sort_keys=True)

        # Sign the payload — returns dict of HTTP headers
        extra_headers = {"Content-Type": "application/json"}
        if self._sign_payload:
            extra_headers.update(self._sign_payload(payload_json))

        tasks = []
        posted_validators = []
        validators = self._validators_for_broadcast()
        for v in validators:
            if not v.proxy_endpoint:
                continue
            url = f"{v.proxy_endpoint.rstrip('/')}/proofs/commit"
            # Send pre-serialized JSON (data=) NOT dict (json=) to match signed digest
            tasks.append(self._post_raw(url, payload_json, v.address[:16], extra_headers))
            posted_validators.append(v)
        if not tasks:
            bt.logging.warning(
                f"Commitment broadcast has no validator endpoints "
                f"(epoch={commitment.epoch_id})"
            )
            return

        results = await self._run_broadcast_posts(tasks)
        self._record_broadcast_results(posted_validators, results)
        success = sum(1 for r in results if r is True)
        bt.logging.debug(
            f"Commitment broadcast: {success}/{len(tasks)} validators received (epoch={commitment.epoch_id})"
        )

    async def broadcast_recipe(self, recipe) -> dict:
        """Phase B: broadcast full recipe to all validators.

        recipe is a plain dict from proof_service (with zkgemm.prover.BlockProof objects).
        """
        await self._refresh_validators()
        await self._ensure_session()

        payload = _serialize_recipe(recipe)
        payload_json = json.dumps(payload, sort_keys=True)

        extra_headers = {"Content-Type": "application/json"}
        if self._sign_payload:
            extra_headers.update(self._sign_payload(payload_json))

        tasks = []
        posted_validators = []
        validators = self._validators_for_broadcast()
        for v in validators:
            if not v.proxy_endpoint:
                continue
            url = f"{v.proxy_endpoint.rstrip('/')}/proofs/recipe"
            tasks.append(self._post_raw(url, payload_json, v.address[:16], extra_headers))
            posted_validators.append(v)
        if not tasks:
            bt.logging.warning(
                f"Recipe broadcast has no validator endpoints "
                f"(epoch={recipe['epoch_id']})"
            )
            return {"success": 0, "total": 0}

        results = await self._run_broadcast_posts(tasks)
        self._record_broadcast_results(posted_validators, results)
        success = sum(1 for r in results if r is True)
        # Log loud only when not all targets succeeded — silent success is fine.
        if tasks and success < len(tasks):
            bt.logging.warning(
                f"Recipe broadcast partial: {success}/{len(tasks)} validators received (epoch={recipe['epoch_id']})"
            )
        else:
            bt.logging.debug(
                f"Recipe broadcast: {success}/{len(tasks)} validators received (epoch={recipe['epoch_id']})"
            )
        return {"success": success, "total": len(tasks)}

    async def broadcast_heartbeat(self, payload: dict):
        """Push hardware heartbeat to all validators."""
        await self._refresh_validators()
        await self._ensure_session()

        payload_json = json.dumps(payload, sort_keys=True)
        extra_headers = {"Content-Type": "application/json"}
        if self._sign_payload:
            extra_headers.update(self._sign_payload(payload_json))

        tasks = []
        posted_validators = []
        validators = self._validators_for_broadcast()
        for v in validators:
            if not v.proxy_endpoint:
                continue
            url = f"{v.proxy_endpoint.rstrip('/')}/heartbeat"
            tasks.append(self._post_raw(url, payload_json, v.address[:16], extra_headers))
            posted_validators.append(v)

        if tasks:
            results = await self._run_broadcast_posts(tasks)
            self._record_broadcast_results(posted_validators, results)
            success = sum(1 for r in results if r is True)
            bt.logging.debug(f"Heartbeat: {success}/{len(tasks)} validators")
        else:
            bt.logging.warning("Heartbeat broadcast has no validator endpoints")

    async def _post_raw(
        self,
        url: str,
        body: str,
        label: str,
        headers: dict = None,
        *,
        timeout: aiohttp.ClientTimeout | None = None,
        max_attempts: int | None = None,
    ) -> bool | _BroadcastFailure:
        """POST pre-serialized JSON to a validator. Returns True on success.

        Uses data= (not json=) to prevent aiohttp from re-serializing,
        ensuring the bytes match what was signed.

        Retries up to 2x on transient failures (timeout, connection error,
        5xx, 429). Permanent client errors (4xx other than 429) fail fast.
        Intermediate failures log at DEBUG; only the final failure logs a WARNING.
        """
        max_attempts = max(1, int(
            POST_MAX_ATTEMPTS if max_attempts is None else max_attempts
        ))
        backoff_s = 0.5
        last_err: str | None = None
        last_failure = _BroadcastFailure("transient")

        for attempt in range(1, max_attempts + 1):
            try:
                async with self._session.post(
                    url,
                    data=body,
                    headers=headers or {},
                    timeout=timeout,
                ) as resp:
                    if resp.status in (200, 201, 202):
                        if attempt > 1:
                            bt.logging.debug(f"POST {url} succeeded on attempt {attempt}")
                        return True
                    resp_body = await resp.text()
                    last_err = f"{resp.status}: {resp_body[:200]}"
                    # 4xx (except 429) are permanent client errors — don't retry.
                    # 409 means a previous attempt reached the validator but
                    # the response was lost, so the broadcast succeeded.
                    if resp.status == 409:
                        bt.logging.debug(f"POST {url} → {last_err}")
                        return True
                    if (
                        resp.status == 401
                        and attempt > 1
                        and "Replay detected" in resp_body
                    ):
                        bt.logging.debug(
                            f"POST {url} replay on retry; treating prior accepted attempt as success"
                        )
                        return True
                    if 400 <= resp.status < 500 and resp.status != 429:
                        bt.logging.warning(f"POST {url} → {last_err}")
                        return _BroadcastFailure("client")
                    last_failure = _BroadcastFailure("transient")
            except asyncio.TimeoutError:
                last_err = "timed out"
                last_failure = _BroadcastFailure("transient")
            except aiohttp.ClientConnectorError as e:
                last_err = str(e)
                last_failure = _BroadcastFailure("transient")
            except aiohttp.ClientOSError as e:
                last_err = str(e)
                last_failure = _BroadcastFailure("transient")
            except Exception as e:
                last_err = str(e)
                last_failure = _BroadcastFailure("transient")

            if attempt < max_attempts:
                bt.logging.debug(
                    f"POST {url} attempt {attempt}/{max_attempts} failed ({last_err}); retrying in {backoff_s}s"
                )
                await asyncio.sleep(backoff_s)
                backoff_s *= 2

        bt.logging.warning(f"POST {url} failed after {max_attempts} attempts: {last_err}")
        return last_failure

    async def _post(
        self,
        url: str,
        payload: dict,
        label: str,
        extra_headers: dict = None,
    ) -> bool | _BroadcastFailure:
        """POST JSON dict to a single validator (unsigned). Returns True on success."""
        try:
            async with self._session.post(url, json=payload, headers=extra_headers or {}) as resp:
                if resp.status in (200, 201, 202):
                    return True
                body = await resp.text()
                bt.logging.warning(f"POST {url} → {resp.status}: {body[:200]}")
                if 400 <= resp.status < 500 and resp.status != 429:
                    return _BroadcastFailure("client")
                return _BroadcastFailure("transient")
        except asyncio.TimeoutError:
            bt.logging.warning(f"POST {url} timed out")
            return _BroadcastFailure("transient")
        except aiohttp.ClientConnectorError as e:
            bt.logging.warning(f"POST {url} failed: {e}")
            return _BroadcastFailure("transient")
        except aiohttp.ClientOSError as e:
            bt.logging.warning(f"POST {url} failed: {e}")
            return _BroadcastFailure("transient")
        except Exception as e:
            bt.logging.warning(f"POST {url} failed: {e}")
            return _BroadcastFailure("transient")

    async def close(self):
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()


def _serialize_commitment(c: ProofCommitment) -> dict:
    """Serialize a commitment to JSON-safe dict."""
    return {
        "epoch_id": c.epoch_id,
        "executor_id": c.executor_id,
        "seed": c.seed.hex(),
        "matrix_dim": c.matrix_dim,
        "pass_0_root": c.pass_0_root.hex(),
        "hw_attestation": {
            "gpu_model": c.hw_attestation.gpu_model,
            "gpu_uuids": c.hw_attestation.gpu_uuids,
            "gpu_count": c.hw_attestation.gpu_count,
            "vram_mb": c.hw_attestation.vram_mb,
            "driver_version": c.hw_attestation.driver_version,
            "nvml_digest": c.hw_attestation.nvml_digest,
            "sysbox_detected": c.hw_attestation.sysbox_detected,
            "cpu_model": c.hw_attestation.cpu_model,
            "ram_gb": c.hw_attestation.ram_gb,
            "software_version": c.hw_attestation.software_version,
            "tee_platform": c.hw_attestation.tee_platform,
        },
        "timestamp_pass0": c.timestamp_pass0,
    }


def _serialize_recipe(r) -> dict:
    """Serialize a recipe dict (with zkgemm.prover.BlockProof objects) to JSON-safe dict.

    The recipe is a plain dict from proof_service.py with:
    - seed: bytes
    - pass_0/pass_1: dict with merkle_root (bytes) and block_proofs (list[zkgemm.prover.BlockProof])

    zkgemm.prover.BlockProof has:
    - bi, bj: int
    - leaf_hash: bytes
    - merkle_path: list[tuple[bytes, bool]]  (hash, is_left)
    - zkgemm_proof: BlockGemmProof (contains sumcheck data + spot checks)
    """
    def _serialize_pass(p):
        return {
            "merkle_root": p["merkle_root"].hex(),
            "block_proofs": [_serialize_block_proof(bp) for bp in p["block_proofs"]],
        }

    def _serialize_block_proof(bp):
        """Serialize a zkgemm.prover.BlockProof to JSON.

        zkgemm.gemm.BlockGemmProof fields:
          bs, n_inner, sumcheck_proof (SumcheckProof), final_A_eval, final_B_eval, spot_A, spot_B
        SumcheckProof fields: claimed_sum, rounds (list[SumcheckRound])
        SumcheckRound fields: evals (tuple[int, int, int])
        """
        zp = bp.zkgemm_proof
        sp = zp.sumcheck_proof
        return {
            "bi": bp.bi,
            "bj": bp.bj,
            "leaf_hash": bp.leaf_hash.hex(),
            # Preserve direction flags: list of [hex_hash, is_left]
            "merkle_path": [[h.hex(), is_left] for h, is_left in bp.merkle_path],
            # BlockGemmProof metadata
            "bs": zp.bs,
            "n_inner": zp.n_inner,
            # SumcheckProof data
            "claimed_sum": sp.claimed_sum,
            "sumcheck_rounds": [list(r.evals) for r in sp.rounds],
            # Final MLE evaluations
            "final_A_eval": zp.final_A_eval,
            "final_B_eval": zp.final_B_eval,
            # PRF spot checks
            "spot_A": [list(t) for t in zp.spot_A],
            "spot_B": [list(t) for t in zp.spot_B],
        }

    # Per-GPU recipe format
    # gpu_proofs may contain:
    #   - Pre-serialized dicts from gpu_worker.py (block_proofs are already plain dicts)
    #   - Raw zkgemm.prover.BlockProof objects (from in-process proof generation)
    # Handle both cases.
    gpu_proofs_serialized = []
    for gp in r.get("gpu_proofs", []):
        seed_val = gp["seed"]
        seed_hex = seed_val.hex() if isinstance(seed_val, bytes) else str(seed_val)

        # Check if pass_0 block_proofs are already serialized (from gpu_worker)
        pass_0 = gp["pass_0"]
        pass_1 = gp["pass_1"]
        if pass_0.get("block_proofs") and isinstance(pass_0["block_proofs"][0], dict):
            # Already serialized by gpu_worker — just ensure merkle_root is hex string
            s_pass_0 = {
                "merkle_root": pass_0["merkle_root"] if isinstance(pass_0["merkle_root"], str) else pass_0["merkle_root"].hex(),
                "block_proofs": pass_0["block_proofs"],
            }
            s_pass_1 = {
                "merkle_root": pass_1["merkle_root"] if isinstance(pass_1["merkle_root"], str) else pass_1["merkle_root"].hex(),
                "block_proofs": pass_1["block_proofs"],
            }
        else:
            # Raw zkgemm objects — serialize them
            s_pass_0 = _serialize_pass(pass_0)
            s_pass_1 = _serialize_pass(pass_1)

        gpu_proofs_serialized.append({
            "gpu_index": gp["gpu_index"],
            "seed": seed_hex,
            "ts_pass0_ns": gp.get("ts_pass0_ns", 0),
            "ts_pass1_ns": gp.get("ts_pass1_ns", 0),
            "T_commit_0": gp.get("T_commit_0", ""),
            "T_commit_1": gp.get("T_commit_1", ""),
            "pass_0": s_pass_0,
            "pass_1": s_pass_1,
        })

    return {
        "epoch_id": r["epoch_id"],
        "executor_id": r["executor_id"],
        "matrix_dim": r["matrix_dim"],
        "gpu_count": r.get("gpu_count", len(gpu_proofs_serialized)),
        "run_id": r.get("run_id", ""),
        "beacon_meta": r.get("beacon_meta", {}),
        "allocation_state": r.get("allocation_state", ""),
        "gpu_proofs": gpu_proofs_serialized,
        "compute_time": r["compute_time"],
        "timestamp_start": r.get("timestamp_start", 0),
        "timestamp_end": r.get("timestamp_end", 0),
    }
