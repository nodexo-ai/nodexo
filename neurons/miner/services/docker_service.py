"""
Docker + Sysbox container management for rental pods.

Creates, manages, and destroys user rental containers with GPU passthrough,
resource limits, and Sysbox isolation.

Container creation patterns adapted from prior-art compute marketplace
implementations (see protocol design notes for context).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import bittensor as bt

DEFAULT_IMAGE = "ubuntu:22.04"
SYSBOX_RUNTIME = "sysbox-runc"


@dataclass
class ContainerConfig:
    """Configuration for a rental container."""
    name: str                      # Container name (unique per executor)
    image: str = DEFAULT_IMAGE     # Docker image
    gpu_uuids: list[str] = None    # GPU UUIDs to attach
    cpu_count: int = 4             # CPU cores
    memory_gb: int = 16            # RAM limit
    storage_gb: int = 100          # Disk limit
    ssh_pub_key: str = ""          # User's SSH public key
    ports: dict[int, int] = None   # host_port -> container_port mapping
    use_sysbox: bool = True        # Use Sysbox runtime
    ttl_seconds: int = 0           # Auto-terminate after N seconds (0 = no TTL)


@dataclass
class ContainerInfo:
    """Info about a running rental container."""
    container_id: str
    name: str
    ssh_port: int
    ssh_user: str
    status: str
    gpu_uuids: list[str]
    created_at: float
    ttl_seconds: int
    image: str = ""
    image_id: str = ""


CONTAINER_STATE_FILE = os.path.join(
    os.path.abspath(os.path.expanduser(
        os.environ.get("NODEXO_DATA_DIR", "~/.nodexo")
    )),
    "containers.json",
)


class DockerService:
    """Manages Docker containers for GPU rentals. Persists state to disk.

    The constructor takes a `port_range` (LOW, HIGH) for allocating publicly-
    reachable SSH ports per rental. Operators open this range in their
    provider firewall once, and every subsequent rental container's port-22
    gets bound to a free port from the range. Without a range, falls back to
    Docker's random high-port assignment (only usable when the dev has SSH
    access to the host to ProxyJump in — never a real renter).
    """

    def __init__(self, port_range: Optional[tuple[int, int]] = None):
        self._containers: dict[str, ContainerInfo] = {}
        self._port_range = port_range
        self._load_state()

    def _allocate_port(self) -> int:
        """Pick a free port from the rental range. 0 = let Docker auto-assign."""
        if not self._port_range:
            return 0
        lo, hi = self._port_range
        used = {c.ssh_port for c in self._containers.values() if c.ssh_port}
        for p in range(lo, hi + 1):
            if p in used:
                continue
            # Make sure no other process holds the port.
            import socket
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
        raise RuntimeError(
            f"No free port in rental range {lo}-{hi}. "
            f"Currently {len(used)} containers in use; widen --rental-port-range."
        )

    _NAME_RE = None
    _GPU_UUID_RE = None

    def _ensure_regex(self):
        import re
        if DockerService._NAME_RE is None:
            DockerService._NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
            DockerService._GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{1,64}$")

    def has_active_containers(self) -> bool:
        """Fast local allocation check for proof mode decisions.

        The chain rental flag is authoritative for global coordination, but a
        just-created canary/rental container is local ground truth for GPU
        pressure. Do not run a heavy proof while a managed container exists,
        even if the miner's chain read has not observed markRented yet.
        """
        active_statuses = {"created", "restarting", "running", "paused"}
        return any(
            str(info.status or "").lower() in active_statuses
            for info in self._containers.values()
        )

    def create_container(self, config: ContainerConfig) -> ContainerInfo:
        """Create a new rental container with GPU passthrough and Sysbox.

        The executor daemon manages its own containers via Docker API
        (creating directly on the miner host, called from the validator's
        rental orchestrator over a signed HTTP request).

        IMPORTANT: every external string (name, image, gpu_uuids) is
        passed as a separate argv element to subprocess (shell=False),
        not joined into a shell command. Earlier versions used
        `subprocess.run(" ".join(cmd), shell=True, ...)` which let a
        crafted `image` or `name` break out via shell metacharacters.
        """
        self._refresh_container_state()
        self._ensure_regex()
        # Defense in depth — the HTTP route also validates, but the docker
        # service is reachable from in-process callers too. Reject anything
        # weird before docker sees it.
        if not DockerService._NAME_RE.match(config.name):
            raise RuntimeError(f"Invalid container name: {config.name!r}")
        if config.gpu_uuids:
            for u in config.gpu_uuids:
                if not DockerService._GPU_UUID_RE.match(u):
                    raise RuntimeError(f"Invalid GPU UUID: {u!r}")

        expected_image_id = self._image_id(config.image)
        if not expected_image_id:
            raise RuntimeError(f"Docker image is not inspectable: {config.image}")

        # Build docker run command (argv list, NEVER shell-joined)
        cmd = ["docker", "run", "-d", "--name", config.name]

        # Sysbox runtime
        if config.use_sysbox and self._sysbox_available():
            cmd.extend(["--runtime", SYSBOX_RUNTIME])
        elif config.use_sysbox:
            if os.environ.get("NODEXO_ALLOW_DOCKER_RUNTIME_FALLBACK", "0") == "1":
                bt.logging.warning(
                    "Sysbox not available; using default Docker runtime because "
                    "NODEXO_ALLOW_DOCKER_RUNTIME_FALLBACK=1"
                )
            else:
                raise RuntimeError(
                    "Sysbox runtime sysbox-runc is required for rental isolation. "
                    "Install sysbox or set NODEXO_ALLOW_DOCKER_RUNTIME_FALLBACK=1 "
                    "only for local development."
                )

        # GPU passthrough — pass as a single device-list value, no shell
        # interpolation. Docker accepts `device=UUID1,UUID2,...` directly.
        if config.gpu_uuids:
            device_str = ",".join(config.gpu_uuids)
            cmd.extend(["--gpus", f"device={device_str}"])
        else:
            cmd.extend(["--gpus", "all"])

        # Resource limits
        cmd.extend(["--cpus", str(config.cpu_count)])
        cmd.extend(["--memory", f"{config.memory_gb}g"])
        # --storage-opt size=... only works on devicemapper / btrfs; the
        # default overlay2 driver rejects it. Disabled until we ship a
        # quota-aware storage driver. Renter quota will likely come via
        # a separately-mounted volume with its own filesystem quota.
        if config.storage_gb > 0 and self._supports_storage_opt():
            cmd.extend(["--storage-opt", f"size={config.storage_gb}g"])

        # Network capabilities (for VPN support)
        cmd.extend(["--cap-add", "NET_ADMIN", "--device", "/dev/net/tun"])

        # Port mappings — pick from the publicly-reachable rental range so
        # the renter can ssh root@host:port directly. Falls back to 0 (random
        # docker-assigned high port) only if the operator didn't configure
        # --rental-port-range, which works for dev (ProxyJump through host)
        # but not for a real renter.
        ssh_host_port = 0
        if config.ports:
            for host_port, container_port in config.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
                if container_port == 22:
                    ssh_host_port = host_port
        else:
            ssh_host_port = self._allocate_port()
            cmd.extend(["-p", f"{ssh_host_port}:22" if ssh_host_port else "0:22"])

        # Restart policy
        cmd.extend(["--restart", "unless-stopped"])

        # Image + command. We default to `sleep infinity` so a vanilla base
        # image (ubuntu:22.04 etc.) stays alive while the SSH setup runs
        # via `docker exec` below. Without a long-running CMD the container
        # exits immediately and `--restart=unless-stopped` makes it loop.
        # Renters who need real systemd should ship an image with their
        # own init and override CMD via the (future) container_command field.
        cmd.append(config.image)
        cmd.extend(["sleep", "infinity"])

        # Logged for operator audit. Argv list reads cleanly without quoting
        # gymnastics; we DO NOT shell-join when running.
        bt.logging.info(f"Creating container: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=False, capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"docker run failed: {result.stderr}")

            container_id = result.stdout.strip()[:12]
            container_image_id = self._container_image_id(config.name)
            if not container_image_id:
                subprocess.run(
                    ["docker", "rm", "-f", config.name],
                    shell=False,
                    capture_output=True,
                    timeout=30,
                )
                raise RuntimeError(
                    f"Could not inspect container image for {config.name}"
                )
            if container_image_id != expected_image_id:
                subprocess.run(
                    ["docker", "rm", "-f", config.name],
                    shell=False,
                    capture_output=True,
                    timeout=30,
                )
                raise RuntimeError(
                    "Container image identity mismatch: "
                    f"requested={config.image} expected={expected_image_id[:20]} "
                    f"got={container_image_id[:20]}"
                )

            # If we used -p 0:22, find the assigned port
            if ssh_host_port == 0:
                ssh_host_port = self._get_mapped_port(config.name, 22)

            # Inject SSH key and start SSHD
            if config.ssh_pub_key:
                self._setup_ssh(config.name, config.ssh_pub_key)

            info = ContainerInfo(
                container_id=container_id,
                name=config.name,
                image=config.image,
                image_id=container_image_id,
                ssh_port=ssh_host_port,
                ssh_user="root",
                status="running",
                gpu_uuids=config.gpu_uuids or [],
                created_at=time.time(),
                ttl_seconds=config.ttl_seconds,
            )
            self._containers[config.name] = info
            self._save_state()
            bt.logging.info(f"Container {config.name} created (port {ssh_host_port})")
            return info

        except subprocess.TimeoutExpired:
            raise RuntimeError("Container creation timed out")

    def image_present(self, image: str) -> bool:
        """True if Docker already has the image locally."""
        try:
            from common.images import image_runtime_reference

            inspect_ref = image
            result = subprocess.run(
                ["docker", "image", "inspect", inspect_ref],
                shell=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                inspect_ref = image_runtime_reference(image)
                if inspect_ref != image:
                    result = subprocess.run(
                        ["docker", "image", "inspect", inspect_ref],
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
            if result.returncode != 0:
                return False
            import json
            rows = json.loads(result.stdout or "[]")
            row = rows[0] if rows else {}
            repo_digests = row.get("RepoDigests") or []
            from common.images import image_digest_matches, image_expected_digest

            expected = image_expected_digest(image)
            if expected and not image_digest_matches(image, repo_digests):
                bt.logging.warning(
                    "Rental image digest mismatch; treating as not ready: "
                    f"{image} expected={expected} got={repo_digests or ['<none>']}"
                )
                return False
            return True
        except Exception:
            return False

    def _image_id(self, image: str) -> str:
        """Return Docker's immutable local image ID for an image reference."""
        try:
            from common.images import image_runtime_reference

            inspect_ref = image
            result = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", inspect_ref],
                shell=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                inspect_ref = image_runtime_reference(image)
                if inspect_ref != image:
                    result = subprocess.run(
                        ["docker", "image", "inspect", "--format", "{{.Id}}", inspect_ref],
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
            if result.returncode != 0:
                return ""
            return (result.stdout or "").strip()
        except Exception:
            return ""

    def _container_image_id(self, container_name: str) -> str:
        """Return the immutable image ID a container was actually started from."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.Image}}", container_name],
                shell=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return ""
            return (result.stdout or "").strip()
        except Exception:
            return ""

    def add_ssh_key(self, container_name: str, ssh_pub_key: str) -> bool:
        """Append a pubkey to the rental container's authorized_keys.

        Validator-orchestrator → POST /containers/{name}/ssh_keys hits
        this. The key text is piped via stdin to `docker exec … tee -a`
        so renter-controlled content never reaches a shell.
        """
        self._ensure_regex()
        if not DockerService._NAME_RE.match(container_name):
            return False
        if not ssh_pub_key or not ssh_pub_key.strip():
            return False
        try:
            payload = (ssh_pub_key.strip() + "\n").encode()
            r = subprocess.run(
                ["docker", "exec", "-i", container_name,
                 "tee", "-a", "/root/.ssh/authorized_keys"],
                input=payload, capture_output=True, timeout=30,
            )
            if r.returncode != 0:
                bt.logging.warning(f"add_ssh_key tee failed: {r.stderr!r}")
                return False
            subprocess.run(
                ["docker", "exec", container_name,
                 "chmod", "600", "/root/.ssh/authorized_keys"],
                capture_output=True, timeout=10,
            )
            return True
        except Exception as e:
            bt.logging.warning(f"add_ssh_key error in {container_name}: {e}")
            return False

    def remove_ssh_key(self, container_name: str, ssh_pub_key: str) -> bool:
        """Remove an authorized_keys entry matching `ssh_pub_key`.

        Matching: by the key body (the AAAA… field), not the comment, so
        a renter who passes their full pubkey with a slightly different
        host comment still removes the right line. Read-filter-write
        happens entirely on the host with python string ops; nothing
        renter-controlled is interpolated into a shell.
        """
        self._ensure_regex()
        if not DockerService._NAME_RE.match(container_name):
            return False
        parts = (ssh_pub_key or "").strip().split()
        if len(parts) < 2:
            return False
        key_body = parts[1]

        # 1) read current authorized_keys
        r = subprocess.run(
            ["docker", "exec", container_name,
             "cat", "/root/.ssh/authorized_keys"],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            # File missing (no keys ever added) → nothing to remove.
            return True
        try:
            lines = r.stdout.decode("utf-8", errors="replace").splitlines()
        except Exception:
            return False
        out = [l for l in lines if key_body not in l]
        if len(out) == len(lines):
            # Nothing matched — caller asked to remove a key that isn't there.
            # Treat as success (idempotent).
            return True
        payload = ("\n".join(out) + ("\n" if out else "")).encode()

        # 2) pipe new contents back via stdin → tee (overwrite, no -a)
        w = subprocess.run(
            ["docker", "exec", "-i", container_name,
             "tee", "/root/.ssh/authorized_keys"],
            input=payload, capture_output=True, timeout=30,
        )
        if w.returncode != 0:
            bt.logging.warning(f"remove_ssh_key tee failed: {w.stderr!r}")
            return False
        subprocess.run(
            ["docker", "exec", container_name,
             "chmod", "600", "/root/.ssh/authorized_keys"],
            capture_output=True, timeout=10,
        )
        return True

    def destroy_container(self, name: str) -> bool:
        """Stop and remove a container."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True, timeout=30,
            )
            self._containers.pop(name, None)
            self._save_state()
            bt.logging.info(f"Container {name} destroyed")
            return True
        except Exception as e:
            bt.logging.error(f"Failed to destroy container {name}: {e}")
            return False

    def list_containers(self) -> list[ContainerInfo]:
        """List all managed rental containers.

        Reconcile against Docker before answering validators. The JSON
        state file is not authoritative if an operator deletes or stops a
        container out of band.
        """
        self._refresh_container_state()
        return list(self._containers.values())

    def port_status(self) -> dict:
        """Current rental SSH port allocation.

        Only container port 22 is exposed today. Host ports are allocated
        from the configured rental range, defaulting to 20000-20100 in
        miner.py.
        """
        used_ports = sorted(c.ssh_port for c in self._containers.values() if c.ssh_port)
        if not self._port_range:
            return {
                "configured": False,
                "start": None,
                "end": None,
                "total": None,
                "used": len(used_ports),
                "free": None,
                "used_ports": used_ports,
            }
        lo, hi = self._port_range
        total = hi - lo + 1
        return {
            "configured": True,
            "start": lo,
            "end": hi,
            "total": total,
            "used": len([p for p in used_ports if lo <= p <= hi]),
            "free": max(0, total - len([p for p in used_ports if lo <= p <= hi])),
            "used_ports": used_ports,
        }

    def get_container(self, name: str) -> Optional[ContainerInfo]:
        self._refresh_container_state([name])
        return self._containers.get(name)

    def check_ttl_expiry(self):
        """Check for and terminate containers that have exceeded their TTL."""
        now = time.time()
        expired = []
        for name, info in self._containers.items():
            if info.ttl_seconds > 0:
                if now - info.created_at > info.ttl_seconds:
                    expired.append(name)

        for name in expired:
            bt.logging.info(f"Container {name} TTL expired, destroying")
            self.destroy_container(name)

    def _setup_ssh(self, container_name: str, ssh_pub_key: str):
        """Install SSH server and inject user's public key into container.

        SECURITY: the renter controls `ssh_pub_key` content. Earlier code
        shell-interpolated it (`f'echo "{ssh_pub_key}" >> …'`), letting a
        crafted key break out via `";` plus arbitrary commands. We pipe
        the key via stdin to `tee -a` so the bytes go straight into the
        file with no shell interpretation.
        """
        # Setup commands that are CONSTANT — safe to use bash -c.
        setup_cmds = [
            "apt-get update -qq && apt-get install -y -qq openssh-server > /dev/null 2>&1",
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh",
            "chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true",
            "sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config",
            "sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config",
            "service ssh start || /usr/sbin/sshd",
        ]
        for cmd in setup_cmds:
            try:
                subprocess.run(
                    ["docker", "exec", container_name, "bash", "-c", cmd],
                    capture_output=True, timeout=60,
                )
            except Exception as e:
                bt.logging.warning(f"SSH setup command failed in {container_name}: {e}")

        # Pipe the pubkey via stdin into authorized_keys — no shell, no
        # interpolation, content is just bytes appended to the file.
        if ssh_pub_key:
            try:
                payload = (ssh_pub_key.strip() + "\n").encode()
                subprocess.run(
                    ["docker", "exec", "-i", container_name,
                     "tee", "-a", "/root/.ssh/authorized_keys"],
                    input=payload, capture_output=True, timeout=30,
                )
                subprocess.run(
                    ["docker", "exec", container_name,
                     "chmod", "600", "/root/.ssh/authorized_keys"],
                    capture_output=True, timeout=10,
                )
            except Exception as e:
                bt.logging.warning(f"SSH key inject failed in {container_name}: {e}")

    def _get_mapped_port(self, container_name: str, container_port: int) -> int:
        """Get the host port mapped to a container port."""
        try:
            result = subprocess.run(
                ["docker", "port", container_name, str(container_port)],
                capture_output=True, text=True, timeout=10,
            )
            # Output format: "0.0.0.0:32768" or ":::32768"
            for line in result.stdout.strip().split("\n"):
                if ":" in line:
                    return int(line.split(":")[-1])
        except Exception:
            pass
        return 0

    def _sysbox_available(self) -> bool:
        """Check if Sysbox runtime is installed."""
        return os.path.exists("/usr/bin/sysbox-runc")

    _storage_opt_cache: Optional[bool] = None

    def _supports_storage_opt(self) -> bool:
        """Probe whether Docker's storage driver accepts --storage-opt size=.

        overlay2 (the Ubuntu default) rejects it; devicemapper / btrfs /
        zfs accept it. Cached on first call.
        """
        if DockerService._storage_opt_cache is not None:
            return DockerService._storage_opt_cache
        try:
            r = subprocess.run(
                ["docker", "info", "--format", "{{.Driver}}"],
                capture_output=True, text=True, timeout=5,
            )
            driver = (r.stdout or "").strip().lower()
            DockerService._storage_opt_cache = driver in {"devicemapper", "btrfs", "zfs"}
        except Exception:
            DockerService._storage_opt_cache = False
        return DockerService._storage_opt_cache

    def _save_state(self):
        """Persist container state to disk for survival across restarts."""
        import json
        state_dir = os.path.dirname(CONTAINER_STATE_FILE)
        os.makedirs(state_dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(state_dir, 0o700)
        except OSError:
            pass
        data = {}
        for name, info in self._containers.items():
            data[name] = {
                "container_id": info.container_id,
                "name": info.name,
                "image": info.image,
                "image_id": info.image_id,
                "ssh_port": info.ssh_port,
                "ssh_user": info.ssh_user,
                "status": info.status,
                "gpu_uuids": info.gpu_uuids,
                "created_at": info.created_at,
                "ttl_seconds": info.ttl_seconds,
            }
        tmp_path = f"{CONTAINER_STATE_FILE}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, CONTAINER_STATE_FILE)

    def _refresh_container_state(self, names: Optional[list[str]] = None) -> None:
        """Reconcile tracked containers with Docker's live state."""
        check_names = list(names) if names is not None else list(self._containers.keys())
        if not check_names:
            return
        dirty = False
        for name in check_names:
            info = self._containers.get(name)
            if info is None:
                continue
            try:
                result = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Status}}", name],
                    capture_output=True, text=True, timeout=10,
                )
            except Exception as e:
                bt.logging.debug(f"container state refresh failed for {name}: {e}")
                continue
            if result.returncode != 0:
                bt.logging.warning(
                    f"Tracked rental container {name} no longer exists; "
                    "removing from miner state"
                )
                self._containers.pop(name, None)
                dirty = True
                continue
            status = (result.stdout or "").strip()
            if status and status != info.status:
                info.status = status
                dirty = True
        if dirty:
            self._save_state()

    def _load_state(self):
        """Restore container state from disk after restart."""
        import json
        if not os.path.exists(CONTAINER_STATE_FILE):
            return
        try:
            with open(CONTAINER_STATE_FILE) as f:
                data = json.load(f)
            dirty = False
            for name, info in data.items():
                # Verify container still exists in Docker
                result = subprocess.run(
                    ["docker", "inspect", name],
                    capture_output=True, timeout=10,
                )
                if result.returncode == 0:
                    self._containers[name] = ContainerInfo(**info)
                    bt.logging.info(f"Restored container {name} from state file")
                else:
                    bt.logging.info(f"Container {name} no longer exists, removing from state")
                    dirty = True
            if dirty:
                self._save_state()
        except Exception as e:
            bt.logging.warning(f"Failed to load container state: {e}")

    def reap_orphans(self, executor_id: str, validator_urls: list[str],
                      hotkey_seed: Optional[bytes] = None,
                      grace_seconds: int = 120) -> int:
        """Destroy local rental containers no validator knows about.

        This is the cure for the failure mode that produced the
        `nodexo-rental-1779383655` ghost earlier: validator-side DB
        wipe / crash / re-deploy leaves the miner with running
        containers that have no on-chain or DB counterpart. Without
        this reap step, those containers sit holding GPU resources
        forever and the validator can never see them to clean up.

        Logic:
          1. Ask every configured validator for its known container
             names for this executor.
          2. Union the responses.
          3. Any local container NOT in that union is an orphan.
             Destroy it.

        Safe-by-design:
          - If ALL validators are unreachable, do NOTHING (don't
            destroy real rentals just because the network is flaky).
          - If at least one validator responds, trust the union of
            responses (a peer validator's rental is still in someone's
            answer).
        Returns the number of containers destroyed.
        """
        if not self._containers:
            return 0
        if not validator_urls:
            bt.logging.debug("reap_orphans: no validator URLs; skipping")
            return 0
        import json as _json
        import urllib.request
        import urllib.error

        known: set[str] = set()
        any_validator_responded = False
        # Sign the request — the endpoint is no longer public (audit
        # C-9). Signature is over the canonical "GET <path>" string;
        # the validator's binding check confirms the hotkey actually
        # owns this executor_id before responding.
        from common.crypto import sign_payload
        for vurl in validator_urls:
            path = f"/executors/{executor_id}/active_containers"
            url = f"{vurl.rstrip('/')}{path}"
            headers = {}
            if hotkey_seed:
                try:
                    body = f"GET {path}".encode()
                    headers = sign_payload(body, hotkey_seed)
                except Exception as e:
                    bt.logging.debug(f"reap_orphans: sign failed: {e}")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = _json.loads(resp.read())
                    for name in data.get("container_names") or []:
                        if name:
                            known.add(name)
                    any_validator_responded = True
            except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
                bt.logging.debug(f"reap_orphans: {url} unreachable ({e})")

        if not any_validator_responded:
            bt.logging.warning(
                "reap_orphans: no validators responded — refusing to "
                "destroy local containers. Will retry on next call."
            )
            return 0

        destroyed = 0
        now = time.time()
        for name, info in list(self._containers.items()):
            # The validator records a freshly-created canary/rental container
            # immediately after POST /containers returns. A periodic reap can
            # race inside that small transition window; do not destroy brand-new
            # containers before validators have had a chance to publish them.
            if grace_seconds > 0 and now - info.created_at < grace_seconds:
                continue
            if name in known:
                continue
            bt.logging.warning(
                f"reap_orphans: container '{name}' not known to any "
                f"validator — destroying as orphan"
            )
            if self.destroy_container(name):
                destroyed += 1
        return destroyed
