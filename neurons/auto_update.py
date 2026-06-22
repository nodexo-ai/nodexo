"""Role-aware auto-updater for Nodexo neurons.

The updater fetches the configured git remote, reads ``neurons/version.py``
from the remote branch, and updates only when the role-specific numeric
version increased.

The updater intentionally uses ``git pull --ff-only`` and never resets local
state. Dirty or diverged production checkouts should be fixed by an operator,
not overwritten by a daemon.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import bittensor as bt

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_git(*args: str, cwd: Optional[Path] = None, timeout: int = 120) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or _REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "git command timed out"
    except Exception as exc:
        return 1, str(exc)


def get_local_head() -> Optional[str]:
    rc, out = _run_git("rev-parse", "HEAD")
    return out if rc == 0 else None


def get_current_branch() -> Optional[str]:
    rc, out = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0 or not out or out == "HEAD":
        return None
    return out


def fetch_origin() -> bool:
    rc, out = _run_git("fetch", "origin", "--quiet")
    if rc != 0:
        if "not a git repository" in out:
            bt.logging.debug("Auto-update skipped: not a git repository")
        else:
            bt.logging.warning(f"Auto-update git fetch failed: {out[:240]}")
        return False
    return True


def get_remote_head(branch: Optional[str] = None) -> Optional[str]:
    branch = branch or get_current_branch()
    if not branch:
        return None
    rc, out = _run_git("rev-parse", f"origin/{branch}")
    return out if rc == 0 else None


def _read_remote_version_file(branch: Optional[str] = None) -> Optional[str]:
    branch = branch or get_current_branch()
    if not branch:
        return None
    rc, out = _run_git("show", f"origin/{branch}:neurons/version.py")
    if rc != 0:
        bt.logging.warning(f"Cannot read remote neurons/version.py: {out[:240]}")
        return None
    return out


def _parse_version_from_source(source: str, var_name: str) -> Optional[int]:
    prefix = {
        "spec_version": "SPEC_",
        "miner_version": "MINER_",
        "validator_version": "VALIDATOR_",
    }.get(var_name)
    if not prefix:
        return None

    def _find_int(name: str) -> Optional[int]:
        match = re.search(rf"^{re.escape(name)}\s*=\s*(\d+)", source, re.MULTILINE)
        return int(match.group(1)) if match else None

    major = _find_int(f"{prefix}MAJOR")
    minor = _find_int(f"{prefix}MINOR")
    patch = _find_int(f"{prefix}PATCH")
    if major is None or minor is None or patch is None:
        return None
    return major * 1_000_000 + minor * 1_000 + patch


def get_remote_version(role: str) -> Optional[int]:
    var_name = "miner_version" if role == "miner" else "validator_version"
    source = _read_remote_version_file()
    if source is None:
        return None
    return _parse_version_from_source(source, var_name)


def check_remote_version(role: str) -> Optional[tuple[str, int, int]]:
    """Return (remote_head, remote_version, local_version) if update is needed."""
    if not fetch_origin():
        return None
    local_head = get_local_head()
    remote_head = get_remote_head()
    if not local_head or not remote_head or local_head == remote_head:
        return None

    role = "miner" if role == "miner" else "validator"
    var_name = "miner_version" if role == "miner" else "validator_version"
    from neurons.version import miner_version, validator_version

    local_version = miner_version if role == "miner" else validator_version
    source = _read_remote_version_file()
    if source is None:
        return None
    remote_version = _parse_version_from_source(source, var_name)
    if remote_version is None:
        bt.logging.warning(f"Cannot parse remote {var_name}; auto-update skipped")
        return None
    if remote_version <= local_version:
        bt.logging.debug(
            f"Remote commits available but {var_name} unchanged "
            f"({local_version}); auto-update skipped"
        )
        return None
    return remote_head, remote_version, local_version


def pull_and_install() -> bool:
    branch = get_current_branch()
    if not branch:
        bt.logging.error("Auto-update cannot determine current branch")
        return False

    rc, out = _run_git("pull", "origin", branch, "--ff-only", timeout=180)
    if rc != 0:
        bt.logging.error(
            "Auto-update pull failed; checkout is probably dirty or diverged: "
            f"{out[:400]}"
        )
        return False

    bt.logging.info("Auto-update installing package")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        bt.logging.error("Auto-update pip install timed out")
        return False
    if result.returncode != 0:
        bt.logging.error(f"Auto-update pip install failed: {result.stderr[:400]}")
        return False
    return True


def _detect_pm2() -> Optional[str]:
    if "pm_id" not in os.environ:
        return None
    try:
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        my_pid = os.getpid()
        pm_id = os.environ.get("pm_id")
        for proc in json.loads(result.stdout):
            if proc.get("pid") == my_pid or str(proc.get("pm_id")) == pm_id:
                return proc.get("name")
    except Exception as exc:
        bt.logging.debug(f"PM2 detection failed: {exc}")
    return None


def restart_process() -> None:
    pm2_name = _detect_pm2()
    if pm2_name:
        bt.logging.info(f"Auto-update restarting via PM2 process {pm2_name}")
        try:
            subprocess.Popen(
                ["pm2", "restart", pm2_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)
            sys.exit(0)
        except Exception as exc:
            bt.logging.error(f"PM2 restart failed, falling back to execv: {exc}")
    bt.logging.info(f"Auto-update re-exec: {sys.executable} {' '.join(sys.argv)}")
    os.execv(sys.executable, [sys.executable, *sys.argv])


class AutoUpdater:
    """Background version watcher.

    ``role`` is either ``miner`` or ``validator``. The updater compares only the
    role's numeric version, so validator-only releases do not restart miners.
    """

    def __init__(
        self,
        *,
        role: str,
        check_interval: int = 1800,
        restart_delay: int = 5,
        busy_check: Optional[Callable[[], bool]] = None,
        jitter_seconds: int = 0,
        jitter_seed: bytes = b"",
    ):
        self.role = "miner" if role == "miner" else "validator"
        self.check_interval = max(60, int(check_interval))
        self.restart_delay = max(0, int(restart_delay))
        self.busy_check = busy_check
        self.jitter_seconds = max(0, int(jitter_seconds))
        self.jitter_seed = bytes(jitter_seed or b"")
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._update_pending = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="nodexo-auto-updater",
            daemon=True,
        )
        self._thread.start()
        watched = "miner_version" if self.role == "miner" else "validator_version"
        bt.logging.info(
            f"Auto-updater started (role={self.role}, watching={watched}, "
            f"interval={self.check_interval}s, branch={get_current_branch() or 'unknown'})"
        )

    def stop(self) -> None:
        self._stop_event.set()

    def notify_not_busy(self) -> None:
        if not self._update_pending:
            return
        bt.logging.info("Deferred auto-update applying now")
        self._update_pending = False
        self._apply_update()

    def _run(self) -> None:
        self._wait(60)
        while not self._stop_event.is_set():
            try:
                self._check_once()
            except Exception as exc:
                bt.logging.error(f"Auto-update check failed: {exc}")
            self._wait(self.check_interval)

    def _wait(self, seconds: int) -> None:
        self._stop_event.wait(timeout=seconds)

    def _stagger_sleep(self) -> None:
        if self.jitter_seconds <= 0 or not self.jitter_seed:
            return
        delay = int.from_bytes(
            hashlib.sha256(self.jitter_seed).digest()[:4], "big",
        ) % self.jitter_seconds
        if delay > 0:
            bt.logging.info(
                f"Auto-update restart stagger: {delay}s "
                f"(window={self.jitter_seconds}s)"
            )
            self._wait(delay)

    def _check_once(self) -> None:
        result = check_remote_version(self.role)
        if result is None:
            return
        remote_head, remote_version, local_version = result
        watched = "miner_version" if self.role == "miner" else "validator_version"
        bt.logging.info(
            f"Auto-update available: {watched} {local_version} -> "
            f"{remote_version} ({remote_head[:8]})"
        )
        if self.busy_check and self.busy_check():
            bt.logging.info("Auto-update deferred because process is busy")
            self._update_pending = True
            return
        self._apply_update()

    def _apply_update(self) -> None:
        if not pull_and_install():
            bt.logging.error("Auto-update failed; will retry on next check")
            return
        self._wait(self.restart_delay)
        if self.busy_check and self.busy_check():
            bt.logging.info("Auto-update became busy during restart delay; deferred")
            self._update_pending = True
            return
        self._stagger_sleep()
        restart_process()
