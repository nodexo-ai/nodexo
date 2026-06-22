"""Runtime subnet config fetcher.

Validators use this for operator-controlled parameters that should not require
a subnet code release, such as per-GPU proof timing calibration.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import bittensor as bt

from common.proof_timing import apply_timing_models_from_config, timing_config_metadata
from common.subnet_runtime_config import (
    apply_runtime_config,
    parse_runtime_config_from_config,
)


def _default_cache_path() -> Path:
    base = os.environ.get("NODEXO_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser() / "subnet_config_cache.json"
    return Path.home() / ".nodexo" / "subnet_config_cache.json"


def subnet_config_url() -> str:
    return (
        os.environ.get("NODEXO_SUBNET_CONFIG_URL", "")
        or os.environ.get("SUBNET_CONFIG_URL", "")
    ).strip()


def subnet_config_refresh_seconds() -> float:
    raw = os.environ.get("NODEXO_SUBNET_CONFIG_REFRESH_SECONDS", "600")
    try:
        return max(30.0, float(raw))
    except Exception:
        return 600.0


def _refresh_seconds_from_config(config: dict[str, Any] | None) -> float:
    if config is None:
        return subnet_config_refresh_seconds()
    try:
        return max(30.0, min(86_400.0, float(config.get("refresh_seconds"))))
    except Exception:
        return subnet_config_refresh_seconds()


def subnet_config_cache_path() -> Path:
    raw = os.environ.get("NODEXO_SUBNET_CONFIG_CACHE_PATH", "").strip()
    return Path(raw).expanduser() if raw else _default_cache_path()


def bundled_subnet_config_path() -> Path:
    return Path(__file__).resolve().with_name("bundled_subnet_config.json")


def _timeout_seconds() -> float:
    raw = os.environ.get("NODEXO_SUBNET_CONFIG_TIMEOUT_SECONDS", "5")
    try:
        return max(1.0, min(30.0, float(raw)))
    except Exception:
        return 5.0


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "nodexo-validator/subnet-config",
        },
    )
    with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
        status = int(getattr(response, "status", 200))
        if status < 200 or status >= 300:
            raise RuntimeError(f"subnet config HTTP {status}")
        body = response.read(1_000_000)
    parsed = json.loads(body.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("subnet config response must be an object")
    return parsed


def load_cached_subnet_config() -> bool:
    """Load the validator-local last-known-good config, if present."""
    path = subnet_config_cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        runtime_config = parse_runtime_config_from_config(data, source=f"cache:{path}")
        count = apply_timing_models_from_config(data, source=f"cache:{path}")
        apply_runtime_config(runtime_config)
        meta = timing_config_metadata()
        bt.logging.info(
            f"Loaded cached subnet config: version={meta.get('version')} "
            f"models={count} path={path}"
        )
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        bt.logging.warning(f"Ignoring invalid cached subnet config {path}: {e}")
        return False


def load_bundled_subnet_config() -> bool:
    """Load the release-bundled config snapshot, if present.

    This is the shipped bootstrap fallback for fresh installs or config-server
    outages before a local last-known-good cache exists.
    """
    path = bundled_subnet_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        runtime_config = parse_runtime_config_from_config(data, source=f"bundled:{path}")
        count = apply_timing_models_from_config(data, source=f"bundled:{path}")
        apply_runtime_config(runtime_config)
        meta = timing_config_metadata()
        bt.logging.info(
            f"Loaded bundled subnet config: version={meta.get('version')} "
            f"models={count} path={path}"
        )
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        bt.logging.warning(f"Ignoring invalid bundled subnet config {path}: {e}")
        return False


def _write_cache(config: dict[str, Any]) -> None:
    path = subnet_config_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_cached_subnet_config() -> dict[str, Any] | None:
    """Return the validated last-known-good raw config document, if present."""
    path = subnet_config_cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        parse_runtime_config_from_config(data, source=f"cache:{path}")
        apply_timing_models_from_config(data, source=f"cache:{path}")
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        bt.logging.warning(f"Ignoring invalid cached subnet config {path}: {e}")
        return None


def read_bundled_subnet_config() -> dict[str, Any] | None:
    """Return the validated release-bundled config snapshot, if present."""
    path = bundled_subnet_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        parse_runtime_config_from_config(data, source=f"bundled:{path}")
        apply_timing_models_from_config(data, source=f"bundled:{path}")
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        bt.logging.warning(f"Ignoring invalid bundled subnet config {path}: {e}")
        return None


def read_cached_or_bundled_subnet_config() -> dict[str, Any] | None:
    """Return last-known-good config, falling back to the shipped snapshot."""
    return read_cached_subnet_config() or read_bundled_subnet_config()


async def fetch_and_apply_subnet_config(url: str | None = None) -> dict[str, Any] | None:
    """Fetch one config version and atomically apply it if valid."""
    target = (url or subnet_config_url()).strip()
    if not target:
        return None
    try:
        data = await asyncio.to_thread(_fetch_json, target)
        runtime_config = parse_runtime_config_from_config(data, source=target)
        count = apply_timing_models_from_config(data, source=target)
        apply_runtime_config(runtime_config)
        await asyncio.to_thread(_write_cache, data)
        meta = timing_config_metadata()
        bt.logging.info(
            f"Subnet config refreshed: version={meta.get('version')} "
            f"models={count} source={target}"
        )
        return data
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as e:
        bt.logging.warning(f"Subnet config refresh failed from {target}: {e}")
        return None


async def run_subnet_config_refresh_loop() -> None:
    """Keep runtime subnet config fresh.

    The caller should load the validator-local last-known-good cache before
    starting this loop. On refresh failures, the current in-memory config stays
    active.
    """
    url = subnet_config_url()
    if not url:
        bt.logging.info("Subnet config refresh disabled (NODEXO_SUBNET_CONFIG_URL not set)")
        return

    last_config = await fetch_and_apply_subnet_config(url)
    interval = _refresh_seconds_from_config(last_config)
    bt.logging.info(f"Subnet config refresh loop started (interval={interval:.0f}s)")
    while True:
        await asyncio.sleep(interval)
        next_config = await fetch_and_apply_subnet_config(url)
        if next_config is not None:
            last_config = next_config
        interval = _refresh_seconds_from_config(last_config)
