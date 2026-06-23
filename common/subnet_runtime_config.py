"""Runtime subnet policy loaded from the web subnet-config endpoint.

The built-in values are the bootstrap/default policy. Validators and miners may
replace them at runtime from the signed-in-admin web config without requiring a
code release. Only non-secret, subnet-wide knobs belong here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import isfinite
from threading import RLock
from typing import Any
from urllib.parse import urlparse

from common.config import canonical_gpu_model_name


DEFAULT_GPU_HOURLY_PRICE: dict[str, float] = {
    "NVIDIA GeForce RTX 3060": 0.10,
    "NVIDIA GeForce RTX 3060 Ti": 0.10,
    "NVIDIA GeForce RTX 3070": 0.12,
    "NVIDIA GeForce RTX 3070 Ti": 0.13,
    "NVIDIA GeForce RTX 3080": 0.15,
    "NVIDIA GeForce RTX 3080 Ti": 0.17,
    "NVIDIA GeForce RTX 3090": 0.22,
    "NVIDIA GeForce RTX 3090 Ti": 0.25,
    "NVIDIA GeForce RTX 4060 Ti": 0.13,
    "NVIDIA GeForce RTX 4070": 0.18,
    "NVIDIA GeForce RTX 4070 Ti": 0.22,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 0.25,
    "NVIDIA GeForce RTX 4080": 0.30,
    "NVIDIA GeForce RTX 4080 SUPER": 0.32,
    "NVIDIA GeForce RTX 4090": 0.69,
    "NVIDIA GeForce RTX 5090": 0.55,
    "NVIDIA RTX 6000 Ada Generation": 0.85,
    "NVIDIA RTX A6000": 0.49,
    "NVIDIA A100-SXM4-40GB": 0.80,
    "NVIDIA A100-PCIE-40GB": 0.80,
    "NVIDIA A100-SXM4-80GB": 1.49,
    "NVIDIA A100 80GB PCIe": 1.39,
    "NVIDIA L40": 1.00,
    "NVIDIA L40S": 1.20,
    "NVIDIA H100 80GB HBM3": 2.89,
    "NVIDIA H100 NVL": 2.80,
    "NVIDIA H200": 3.50,
    "NVIDIA H200 NVL": 3.50,
    "NVIDIA B200": 5.00,
    "NVIDIA B300 SXM6": 6.00,
}
DEFAULT_PRICE_PER_HOUR = 0.10


@dataclass(frozen=True)
class PricingRuntimeConfig:
    default_price_per_hour: float = DEFAULT_PRICE_PER_HOUR
    renter_quote_multiplier: float = 1.0
    gpu_hourly_price: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GPU_HOURLY_PRICE)
    )


@dataclass(frozen=True)
class ScoringRuntimeConfig:
    proof_recency_s: int = 600
    heartbeat_recency_s: int = 300
    idle_fraction: float = 0.20
    min_miner_stake_tao: float = 0.0
    stake_per_score_tao: float = 1.0
    stake_gpu_count_exponent: float = 1.0
    stake_requirement_multiplier: float = 24.0
    stake_grace_until_ts: float = 0.0
    min_reliability_samples: int = 5
    min_reliability_gate: float = 0.67
    reliability_exponent: float = 2.0
    reliability_miss_grace_slots: int = 20
    timing_grace_count: int = 3
    timing_grace_window_s: int = 3600
    timing_grace_consecutive_count: int = 2
    timing_min_delta_fraction: float = 0.55
    rented_micro_wall_deadline_s: int = 90


@dataclass(frozen=True)
class CanaryRuntimeConfig:
    mean_seconds: int = 24 * 3600
    error_retry_seconds: int = 15 * 60
    error_streak_threshold: int = 3


@dataclass(frozen=True)
class WeightsRuntimeConfig:
    set_weights_interval_blocks: int = 360
    emission_burn_fraction: float = 0.8
    burn_uid: int | None = None


@dataclass(frozen=True)
class NetworkRuntimeConfig:
    rental_validator_hotkeys: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlacklistRuntimeConfig:
    executor_ids: tuple[str, ...] = ()
    miner_hotkeys: tuple[str, ...] = ()
    miner_addresses: tuple[str, ...] = ()
    uids: tuple[int, ...] = ()
    endpoint_hosts: tuple[str, ...] = ()
    active_rental_policy: str = "terminate"


@dataclass(frozen=True)
class SubnetRuntimeConfig:
    source: str = "builtin"
    version: str = "builtin"
    pricing: PricingRuntimeConfig = field(default_factory=PricingRuntimeConfig)
    scoring: ScoringRuntimeConfig = field(default_factory=ScoringRuntimeConfig)
    canary: CanaryRuntimeConfig = field(default_factory=CanaryRuntimeConfig)
    weights: WeightsRuntimeConfig = field(default_factory=WeightsRuntimeConfig)
    network: NetworkRuntimeConfig = field(default_factory=NetworkRuntimeConfig)
    blacklist: BlacklistRuntimeConfig = field(default_factory=BlacklistRuntimeConfig)


_LOCK = RLock()
_RUNTIME_CONFIG = SubnetRuntimeConfig()


def _coerce_float(
    raw: Any,
    *,
    name: str,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    if raw is None:
        return default
    value = float(raw)
    if not isfinite(value) or value < min_value or value > max_value:
        raise ValueError(f"{name} out of range")
    return value


def _coerce_int(
    raw: Any,
    *,
    name: str,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    if raw is None:
        return default
    value = int(raw)
    if value < min_value or value > max_value:
        raise ValueError(f"{name} out of range")
    return value


def _parse_pricing(raw: Any) -> PricingRuntimeConfig:
    if raw is None:
        return PricingRuntimeConfig()
    if not isinstance(raw, dict):
        raise ValueError("pricing must be an object")
    default_price = _coerce_float(
        raw.get("default_price_per_hour"),
        name="pricing.default_price_per_hour",
        default=DEFAULT_PRICE_PER_HOUR,
        min_value=0.0,
        max_value=1_000.0,
    )
    renter_quote_multiplier = _coerce_float(
        raw.get("renter_quote_multiplier"),
        name="pricing.renter_quote_multiplier",
        default=1.0,
        min_value=0.0,
        max_value=100.0,
    )
    prices_raw = raw.get("gpu_hourly_price", DEFAULT_GPU_HOURLY_PRICE)
    if not isinstance(prices_raw, dict):
        raise ValueError("pricing.gpu_hourly_price must be an object")
    prices: dict[str, float] = {}
    for name, value in prices_raw.items():
        canonical_name = canonical_gpu_model_name(str(name))
        if canonical_name.lower() == "default":
            continue
        price = _coerce_float(
            value,
            name=f"pricing.gpu_hourly_price.{canonical_name}",
            default=default_price,
            min_value=0.0,
            max_value=1_000.0,
        )
        if canonical_name in prices:
            raise ValueError(f"duplicate price after canonicalization: {name!r}")
        prices[canonical_name] = price
    return PricingRuntimeConfig(
        default_price_per_hour=default_price,
        renter_quote_multiplier=renter_quote_multiplier,
        gpu_hourly_price=prices,
    )


def _parse_scoring(raw: Any) -> ScoringRuntimeConfig:
    if raw is None:
        return ScoringRuntimeConfig()
    if not isinstance(raw, dict):
        raise ValueError("scoring must be an object")
    return ScoringRuntimeConfig(
        proof_recency_s=_coerce_int(
            raw.get("proof_recency_s"),
            name="scoring.proof_recency_s",
            default=600,
            min_value=60,
            max_value=86_400,
        ),
        heartbeat_recency_s=_coerce_int(
            raw.get("heartbeat_recency_s"),
            name="scoring.heartbeat_recency_s",
            default=300,
            min_value=30,
            max_value=86_400,
        ),
        idle_fraction=_coerce_float(
            raw.get("idle_fraction"),
            name="scoring.idle_fraction",
            default=0.20,
            min_value=0.0,
            max_value=1.0,
        ),
        min_miner_stake_tao=_coerce_float(
            raw.get("min_miner_stake_tao"),
            name="scoring.min_miner_stake_tao",
            default=0.0,
            min_value=0.0,
            max_value=10_000_000.0,
        ),
        stake_per_score_tao=_coerce_float(
            raw.get("stake_per_score_tao"),
            name="scoring.stake_per_score_tao",
            default=1.0,
            min_value=0.0,
            max_value=10_000_000.0,
        ),
        stake_gpu_count_exponent=_coerce_float(
            raw.get("stake_gpu_count_exponent"),
            name="scoring.stake_gpu_count_exponent",
            default=1.0,
            min_value=0.0,
            max_value=4.0,
        ),
        stake_requirement_multiplier=_coerce_float(
            raw.get("stake_requirement_multiplier"),
            name="scoring.stake_requirement_multiplier",
            default=24.0,
            min_value=0.0,
            max_value=1_000.0,
        ),
        stake_grace_until_ts=_coerce_float(
            raw.get("stake_grace_until_ts"),
            name="scoring.stake_grace_until_ts",
            default=0.0,
            min_value=0.0,
            max_value=4_102_444_800.0,  # 2100-01-01 UTC
        ),
        min_reliability_samples=_coerce_int(
            raw.get("min_reliability_samples"),
            name="scoring.min_reliability_samples",
            default=5,
            min_value=1,
            max_value=10_000,
        ),
        min_reliability_gate=_coerce_float(
            raw.get("min_reliability_gate"),
            name="scoring.min_reliability_gate",
            default=0.67,
            min_value=0.0,
            max_value=1.0,
        ),
        reliability_exponent=_coerce_float(
            raw.get("reliability_exponent"),
            name="scoring.reliability_exponent",
            default=2.0,
            min_value=0.0,
            max_value=10.0,
        ),
        reliability_miss_grace_slots=_coerce_int(
            raw.get("reliability_miss_grace_slots"),
            name="scoring.reliability_miss_grace_slots",
            default=20,
            min_value=0,
            max_value=10_000,
        ),
        timing_grace_count=_coerce_int(
            raw.get("timing_grace_count"),
            name="scoring.timing_grace_count",
            default=3,
            min_value=0,
            max_value=20,
        ),
        timing_grace_window_s=_coerce_int(
            raw.get("timing_grace_window_s"),
            name="scoring.timing_grace_window_s",
            default=3600,
            min_value=60,
            max_value=86_400,
        ),
        timing_grace_consecutive_count=_coerce_int(
            raw.get("timing_grace_consecutive_count"),
            name="scoring.timing_grace_consecutive_count",
            default=1,
            min_value=1,
            max_value=10,
        ),
        timing_min_delta_fraction=_coerce_float(
            raw.get("timing_min_delta_fraction"),
            name="scoring.timing_min_delta_fraction",
            default=0.55,
            min_value=0.0,
            max_value=1.0,
        ),
        rented_micro_wall_deadline_s=_coerce_int(
            raw.get("rented_micro_wall_deadline_s"),
            name="scoring.rented_micro_wall_deadline_s",
            default=90,
            min_value=0,
            max_value=600,
        ),
    )


def _parse_canary(raw: Any) -> CanaryRuntimeConfig:
    if raw is None:
        return CanaryRuntimeConfig()
    if not isinstance(raw, dict):
        raise ValueError("canary must be an object")
    return CanaryRuntimeConfig(
        mean_seconds=_coerce_int(
            raw.get("mean_seconds"),
            name="canary.mean_seconds",
            default=24 * 3600,
            min_value=120,
            max_value=30 * 24 * 3600,
        ),
        error_retry_seconds=_coerce_int(
            raw.get("error_retry_seconds"),
            name="canary.error_retry_seconds",
            default=15 * 60,
            min_value=60,
            max_value=24 * 3600,
        ),
        error_streak_threshold=_coerce_int(
            raw.get("error_streak_threshold"),
            name="canary.error_streak_threshold",
            default=3,
            min_value=1,
            max_value=20,
        ),
    )


def _parse_weights(raw: Any) -> WeightsRuntimeConfig:
    if raw is None:
        return WeightsRuntimeConfig()
    if not isinstance(raw, dict):
        raise ValueError("weights must be an object")
    return WeightsRuntimeConfig(
        set_weights_interval_blocks=_coerce_int(
            raw.get("set_weights_interval_blocks"),
            name="weights.set_weights_interval_blocks",
            default=360,
            min_value=10,
            max_value=10_000,
        ),
        emission_burn_fraction=_coerce_float(
            raw.get("emission_burn_fraction"),
            name="weights.emission_burn_fraction",
            default=0.8,
            min_value=0.0,
            max_value=1.0,
        ),
        burn_uid=(
            None
            if raw.get("burn_uid") is None
            else _coerce_int(
                raw.get("burn_uid"),
                name="weights.burn_uid",
                default=0,
                min_value=0,
                max_value=65_535,
            )
        ),
    )


def _parse_network(raw: Any) -> NetworkRuntimeConfig:
    if raw is None:
        return NetworkRuntimeConfig()
    if not isinstance(raw, dict):
        raise ValueError("network must be an object")
    hotkeys_raw = raw.get("rental_validator_hotkeys", [])
    if hotkeys_raw is None:
        hotkeys_raw = []
    if not isinstance(hotkeys_raw, list):
        raise ValueError("network.rental_validator_hotkeys must be a list")
    hotkeys: list[str] = []
    seen: set[str] = set()
    for value in hotkeys_raw:
        hotkey = str(value or "").strip()
        if not hotkey:
            continue
        if len(hotkey) > 80:
            raise ValueError("network.rental_validator_hotkeys contains invalid hotkey")
        if hotkey in seen:
            continue
        seen.add(hotkey)
        hotkeys.append(hotkey)
    return NetworkRuntimeConfig(rental_validator_hotkeys=tuple(hotkeys))


_EXECUTOR_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")


def _list_raw(raw: Any, *, name: str, max_items: int = 2048) -> list[Any]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be a list")
    if len(raw) > max_items:
        raise ValueError(f"{name} has too many entries")
    return raw


def _dedupe_normalized(values: list[Any], normalizer, *, name: str) -> tuple:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        item = normalizer(value)
        if item is None or item == "":
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _normalize_executor_id(value: Any) -> str:
    text = str(value or "").strip().lower().removeprefix("0x")
    if not text:
        return ""
    if not _EXECUTOR_ID_RE.fullmatch(text):
        raise ValueError("blacklist.executor_ids contains invalid executor id")
    return text


def _normalize_hotkey(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > 80:
        raise ValueError("blacklist.miner_hotkeys contains invalid hotkey")
    return text


def _normalize_address(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if not text.startswith("0x") and re.fullmatch(r"[0-9a-f]{40}", text):
        text = f"0x{text}"
    if not _ADDRESS_RE.fullmatch(text):
        raise ValueError("blacklist.miner_addresses contains invalid address")
    return text


def _normalize_uid(value: Any) -> int | None:
    if value is None or value == "":
        return None
    uid = int(value)
    if uid < 0 or uid > 65_535:
        raise ValueError("blacklist.uids contains invalid UID")
    return uid


def _endpoint_host(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    parsed_text = text if "://" in text else f"//{text}"
    try:
        parsed = urlparse(parsed_text)
        host = parsed.hostname or ""
    except Exception:
        host = ""
    if not host:
        host = text.split("/", 1)[0]
        if host.count(":") == 1:
            host = host.rsplit(":", 1)[0]
    host = host.strip().strip("[]").rstrip(".").lower()
    if not host:
        return ""
    if len(host) > 253 or any(ch.isspace() for ch in host):
        raise ValueError("blacklist.endpoint_hosts contains invalid host")
    return host


def _parse_blacklist(raw: Any) -> BlacklistRuntimeConfig:
    if raw is None:
        return BlacklistRuntimeConfig()
    if not isinstance(raw, dict):
        raise ValueError("blacklist must be an object")
    endpoint_hosts_raw = raw.get("endpoint_hosts", raw.get("endpoints"))
    policy = str(raw.get("active_rental_policy") or "terminate").strip().lower()
    if policy not in {"terminate", "keep"}:
        raise ValueError("blacklist.active_rental_policy must be terminate or keep")
    return BlacklistRuntimeConfig(
        executor_ids=_dedupe_normalized(
            _list_raw(raw.get("executor_ids"), name="blacklist.executor_ids"),
            _normalize_executor_id,
            name="blacklist.executor_ids",
        ),
        miner_hotkeys=_dedupe_normalized(
            _list_raw(raw.get("miner_hotkeys"), name="blacklist.miner_hotkeys"),
            _normalize_hotkey,
            name="blacklist.miner_hotkeys",
        ),
        miner_addresses=_dedupe_normalized(
            _list_raw(raw.get("miner_addresses"), name="blacklist.miner_addresses"),
            _normalize_address,
            name="blacklist.miner_addresses",
        ),
        uids=_dedupe_normalized(
            _list_raw(raw.get("uids"), name="blacklist.uids"),
            _normalize_uid,
            name="blacklist.uids",
        ),
        endpoint_hosts=_dedupe_normalized(
            _list_raw(endpoint_hosts_raw, name="blacklist.endpoint_hosts"),
            _endpoint_host,
            name="blacklist.endpoint_hosts",
        ),
        active_rental_policy=policy,
    )


def parse_runtime_config_from_config(
    config: dict[str, Any],
    *,
    source: str = "remote",
) -> SubnetRuntimeConfig:
    if not isinstance(config, dict):
        raise ValueError("subnet config must be an object")
    version = str(config.get("version") or config.get("config_version") or "unknown")[:80]
    pricing = _parse_pricing(config.get("pricing"))
    scoring = _parse_scoring(config.get("scoring"))
    canary = _parse_canary(config.get("canary"))
    weights = _parse_weights(config.get("weights"))
    network = _parse_network(config.get("network"))
    blacklist = _parse_blacklist(config.get("blacklist"))
    if weights.burn_uid is not None and weights.burn_uid in blacklist.uids:
        raise ValueError("weights.burn_uid must not be listed in blacklist.uids")
    return SubnetRuntimeConfig(
        source=str(source),
        version=version,
        pricing=pricing,
        scoring=scoring,
        canary=canary,
        weights=weights,
        network=network,
        blacklist=blacklist,
    )


def apply_runtime_config(config: SubnetRuntimeConfig) -> None:
    global _RUNTIME_CONFIG
    with _LOCK:
        _RUNTIME_CONFIG = config


def apply_runtime_config_from_config(
    config: dict[str, Any],
    *,
    source: str = "remote",
) -> SubnetRuntimeConfig:
    parsed = parse_runtime_config_from_config(config, source=source)
    apply_runtime_config(parsed)
    return parsed


def get_subnet_runtime_config() -> SubnetRuntimeConfig:
    with _LOCK:
        return _RUNTIME_CONFIG


def policy_blacklist_reason(
    executor_id: str = "",
    *,
    hotkey_ss58: str = "",
    miner_address: str = "",
    miner_uid: int | None = None,
    endpoint: str = "",
) -> str:
    """Return a stable policy blacklist reason, or an empty string if allowed."""
    cfg = get_subnet_runtime_config().blacklist
    try:
        normalized_executor_id = str(executor_id or "").strip().lower().removeprefix("0x")
        if normalized_executor_id and normalized_executor_id in cfg.executor_ids:
            return "policy_blacklisted_executor"
    except Exception:
        pass
    if hotkey_ss58 and hotkey_ss58 in cfg.miner_hotkeys:
        return "policy_blacklisted_hotkey"
    try:
        normalized_address = _normalize_address(miner_address)
        if normalized_address and normalized_address in cfg.miner_addresses:
            return "policy_blacklisted_address"
    except Exception:
        pass
    try:
        if miner_uid is not None and int(miner_uid) in cfg.uids:
            return "policy_blacklisted_uid"
    except Exception:
        pass
    try:
        host = _endpoint_host(endpoint)
        if host and host in cfg.endpoint_hosts:
            return "policy_blacklisted_endpoint"
    except Exception:
        pass
    return ""


def runtime_config_metadata() -> dict[str, str]:
    cfg = get_subnet_runtime_config()
    return {"source": cfg.source, "version": cfg.version}


def reset_runtime_config_for_tests() -> None:
    apply_runtime_config(SubnetRuntimeConfig())
