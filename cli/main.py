"""
nodexo CLI — fleet management tool for miner operators.

Runs on your local secure machine (where the coldkey lives).
By default, pulls fleet data from the public Nodexo API. Chain/RPC
operator diagnostics are available with `fleet --chain-direct`.

Interactive navigation:
  Table view → type number → Detail view
  Detail view: n=next, p=prev, b=back to table, q=quit

Commands:
  fleet/miner       — interactive miner/operator fleet dashboard
  inventory         — public capacity list
  marketplace       — public renter marketplace
  quote             — preview x402 payment requirements without paying
  rent              — create an x402 or credit-backed rental
  credits           — show account credit balance
  account-ssh-keys  — list/add/remove saved account SSH public keys
  operator-claim    — sign an operator dashboard hotkey claim
  fund              — fund EVM wallet from coldkey

Usage:
  nodexo --wallet miner fleet
  nodexo --wallet miner fleet --chain-direct --chain-config chain_config.json
  nodexo inventory
  nodexo rent --gpu A6000 --duration 1h --ssh-key ~/.ssh/id_ed25519.pub
  nodexo rent --payment credits --gpu A6000 --api-key vc_... --ssh-key ~/.ssh/id_ed25519.pub
  nodexo fund --wallet miner --amount 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _exit(code: int = 0) -> None:
    """Exit without bittensor websocket cleanup hangs, but keep CLI output."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        getattr(os, "_exit")(code)


# ── Wallet discovery + chain config auto-pick ──────────────────

def _list_wallets() -> list[dict]:
    """List existing Bittensor wallets with their hotkeys.

    Returns a list of {'name': coldkey_name, 'hotkeys': [hotkey_name, ...]}.
    Used by the interactive wallet selector when --wallet/--hotkey are omitted.
    """
    wallets_dir = Path.home() / ".bittensor" / "wallets"
    if not wallets_dir.exists():
        return []
    out = []
    for w in sorted(wallets_dir.iterdir()):
        if not w.is_dir():
            continue
        hotkeys_dir = w / "hotkeys"
        hotkeys: list[str] = []
        if hotkeys_dir.exists():
            hotkeys = sorted(
                f.name for f in hotkeys_dir.iterdir()
                if f.is_file() and not f.name.endswith(".txt")
            )
        if hotkeys:
            out.append({"name": w.name, "hotkeys": hotkeys})
    return out


def _detect_pm2_miners(network: str = "") -> list[dict]:
    """Find running Nodexo miner processes via PM2 and parse their --wallet/--hotkey.

    Best-effort: if PM2 isn't installed or no Nodexo process is found, returns [].
    Useful when the operator runs the CLI on the same machine as their fleet —
    saves them from picking the wallet manually. Filters by netuid match if
    `network` is given (test→468, finney→106).
    """
    import json
    import shlex
    import subprocess

    expected_netuid = {"test": 468, "finney": 106}.get(network)

    try:
        out = subprocess.run(
            ["pm2", "jlist"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return []
        procs = json.loads(out.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []

    found: list[dict] = []
    for p in procs:
        env = p.get("pm2_env", {}) or {}
        script = env.get("pm_exec_path", "") or ""
        name = p.get("name", "") or ""
        # Heuristic: a Nodexo miner has nodexo in its PM2 name
        # OR launches from a script under the nodexo repo
        is_nodexo = ("nodexo" in name.lower()) or ("nodexo" in script)
        if not is_nodexo:
            continue
        # Parse the launch shell script for --wallet/--hotkey/--netuid
        if not (script.endswith(".sh") and os.path.exists(script)):
            continue
        try:
            with open(script) as f:
                body = f.read()
        except Exception:
            continue
        wallet = hotkey = ""
        netuid: int | None = None
        try:
            tokens = shlex.split(body, comments=True, posix=True)
        except ValueError:
            continue
        for i, tok in enumerate(tokens):
            if tok == "--wallet" and i + 1 < len(tokens):
                wallet = tokens[i + 1].strip("'\"")
            elif tok == "--hotkey" and i + 1 < len(tokens):
                hotkey = tokens[i + 1].strip("'\"")
            elif tok == "--netuid" and i + 1 < len(tokens):
                try:
                    netuid = int(tokens[i + 1].strip("'\""))
                except ValueError:
                    pass
        if not wallet or not hotkey:
            continue
        if expected_netuid is not None and netuid is not None and netuid != expected_netuid:
            continue
        found.append({
            "pm2_name": name, "wallet": wallet, "hotkey": hotkey, "netuid": netuid,
        })
    return found


def _select_wallet(default_wallet: str = "", default_hotkey: str = "",
                   network: str = "") -> tuple[str, str]:
    """Prompt operator to pick a (wallet, hotkey) pair if not specified.

    If both default_wallet and default_hotkey are non-empty AND the pair
    exists, returns them unchanged. Otherwise:
      1. Check PM2 for running Nodexo miners (best-effort) and surface those
         as the first options.
      2. Fall back to listing all wallets in ~/.bittensor/wallets/ and prompt.
    """
    wallets = _list_wallets()
    if not wallets:
        print("  No wallets found in ~/.bittensor/wallets/. Create one with `btcli wallet new_coldkey`.")
        sys.exit(1)

    # If user passed both and the pair exists on disk, accept
    if default_wallet and default_hotkey:
        for w in wallets:
            if w["name"] == default_wallet and default_hotkey in w["hotkeys"]:
                return default_wallet, default_hotkey

    # PM2 fast-path: if a Nodexo miner is running locally and uses a wallet
    # that exists on disk, surface it as option [1]. Common case: operator
    # is on the same box as their fleet.
    pm2 = _detect_pm2_miners(network)
    pm2_options: list[tuple[str, str, str]] = []  # (label, wallet, hotkey)
    seen: set[tuple[str, str]] = set()
    for m in pm2:
        key = (m["wallet"], m["hotkey"])
        if key in seen:
            continue
        # Confirm the wallet/hotkey pair exists on disk before suggesting
        for w in wallets:
            if w["name"] == m["wallet"] and m["hotkey"] in w["hotkeys"]:
                label = f"{m['pm2_name']} → {m['wallet']}/{m['hotkey']}"
                if m.get("netuid") is not None:
                    label += f" (netuid {m['netuid']})"
                pm2_options.append((label, m["wallet"], m["hotkey"]))
                seen.add(key)
                break

    options: list[tuple[str, str, str]] = []  # (label, wallet, hotkey)
    if pm2_options:
        print("\n  Detected running miners (via PM2):")
        for opt in pm2_options:
            options.append(opt)
            print(f"    [{len(options)}] {opt[0]}")
        print("\n  Other wallets:")
    else:
        print("\n  Available wallets:")
    for w in wallets:
        for hk in w["hotkeys"]:
            if (w["name"], hk) in seen:
                continue
            label = f"{w['name']}/{hk}"
            options.append((label, w["name"], hk))
            print(f"    [{len(options)}] {label}")

    while True:
        choice = input(f"\n  Select [1-{len(options)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            _, picked_w, picked_hk = options[int(choice) - 1]
            return picked_w, picked_hk
        print(f"  Invalid choice. Enter 1-{len(options)}.")


def _wallet_hotkey_ss58(wallet_name: str, hotkey_name: str) -> str:
    import json
    from substrateinterface.utils.ss58 import ss58_encode

    hotkey_path = Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys" / hotkey_name
    data = json.loads(hotkey_path.read_text(encoding="utf-8"))
    if isinstance(data.get("ss58Address"), str) and data["ss58Address"]:
        return data["ss58Address"]
    for field in ("accountId", "publicKey"):
        value = data.get(field)
        if isinstance(value, str) and value:
            raw = bytes.fromhex(value.removeprefix("0x"))
            if len(raw) == 32:
                return ss58_encode(raw, ss58_format=42)
    raise RuntimeError(f"Cannot read hotkey address from {hotkey_path}")


def _find_wallet_for_hotkey_ss58(hotkey_ss58: str) -> tuple[str, str] | None:
    matches: list[tuple[str, str]] = []
    for wallet in _list_wallets():
        for hotkey in wallet["hotkeys"]:
            try:
                if _wallet_hotkey_ss58(wallet["name"], hotkey) == hotkey_ss58:
                    matches.append((wallet["name"], hotkey))
            except Exception:
                continue
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and sys.stdin.isatty():
        print("\n  Matching local hotkeys:")
        for i, (wallet, hotkey) in enumerate(matches, start=1):
            print(f"    [{i}] {wallet}/{hotkey}")
        while True:
            choice = input(f"\n  Select [1-{len(matches)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                return matches[int(choice) - 1]
            print(f"  Invalid choice. Enter 1-{len(matches)}.")
    if len(matches) > 1:
        print(
            "  Multiple local wallets match the challenge hotkey. "
            "Pass --wallet and --hotkey explicitly.",
            file=sys.stderr,
        )
        _exit(1)
    return None


def _resolve_chain_config(explicit_path: str, network: str) -> str:
    """Pick the right chain_config_*.json from --subtensor-network if not given.

    Looks in this repo root.
    """
    if explicit_path:
        return explicit_path
    repo = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if network == "test":
        candidates = [repo / "chain_config_testnet.json", repo / "chain_config.json"]
    elif network == "finney":
        candidates = [repo / "chain_config_mainnet.json", repo / "chain_config.json"]
    else:
        candidates = [repo / "chain_config.json"]
    for p in candidates:
        if p.exists():
            return str(p)
    print(f"  Could not auto-resolve chain config for network='{network}'.")
    print(f"  Tried: {', '.join(str(p) for p in candidates)}")
    print(f"  Pass --chain-config explicitly.")
    sys.exit(1)


# ── Async data fetching ────────────────────────────────────────

async def _fetch_json(session, url: str, timeout: float = 4) -> dict | None:
    try:
        import aiohttp
        headers = {}
        admin = os.environ.get("NODEXO_ADMIN_TOKEN", "").strip()
        if admin:
            headers["X-Admin-Token"] = admin
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


async def _gather_miner_data(executors: list, validator_url: str = "") -> list[dict]:
    """Fetch live data from all miners + validator in parallel.

    Joins three views per executor:
      - on-chain registration (passed in `executors`)
      - operational health from miner endpoints (/health, /hardware, /proof/status)
      - attestation/verification stats from validator (/scores, /instances)
      - rental status from validator (/rentals)
    """
    import aiohttp
    results = []

    async with aiohttp.ClientSession() as session:
        scores: dict = {}
        instances_by_id: dict[str, dict] = {}
        rentals_by_executor: dict[str, dict] = {}
        if validator_url:
            v = validator_url.rstrip('/')
            scores_data, instances_data, rentals_data = await asyncio.gather(
                _fetch_json(session, f"{v}/scores"),
                _fetch_json(session, f"{v}/instances"),
                _fetch_json(session, f"{v}/rentals"),
                return_exceptions=True,
            )
            if isinstance(scores_data, dict):
                scores = scores_data.get("scores", {})
            if isinstance(instances_data, dict):
                for inst in instances_data.get("instances", []):
                    instances_by_id[inst.get("instance_id", "")] = inst
            if isinstance(rentals_data, dict):
                for r in rentals_data.get("rentals", []):
                    rentals_by_executor[r.get("executor_id", "")] = r

        tasks = []
        for ex in executors:
            endpoint = ex["endpoint"]
            tasks.append(_fetch_json(session, f"{endpoint}/health"))
            tasks.append(_fetch_json(session, f"{endpoint}/hardware"))
            tasks.append(_fetch_json(session, f"{endpoint}/proof/status"))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, ex in enumerate(executors):
            health = responses[i * 3] if not isinstance(responses[i * 3], Exception) else None
            hardware = responses[i * 3 + 1] if not isinstance(responses[i * 3 + 1], Exception) else None
            proof = responses[i * 3 + 2] if not isinstance(responses[i * 3 + 2], Exception) else None

            eid = ex["executor_id"]
            score_data = scores.get(eid, {})
            instance = instances_by_id.get(eid, {})
            rental = rentals_by_executor.get(eid)

            # GPU name preference: validator /instances → miner /hardware → unknown.
            # Miner /hardware returns gpus: [{name, uuid, vram_mb}, ...] — pull from index 0.
            gpu_name = ""
            if instance.get("hardware", {}).get("gpu_name"):
                gpu_name = instance["hardware"]["gpu_name"]
            elif hardware:
                gpus_list = hardware.get("gpus") or []
                if gpus_list and gpus_list[0].get("name"):
                    gpu_name = gpus_list[0]["name"]
                elif hardware.get("gpu_name"):  # legacy schema fallback
                    gpu_name = hardware["gpu_name"]
            if gpu_name:
                gpu_name = gpu_name.replace("NVIDIA ", "")

            gpu_pct = "—"
            if hardware and "gpu_utilization" in hardware:
                utils = hardware["gpu_utilization"]
                if utils:
                    avg = sum(g.get("gpu_util_pct", 0) for g in utils) / len(utils)
                    gpu_pct = f"{avg:.0f}%"

            proof_epoch = "—"
            proof_running = False
            if proof and isinstance(proof, dict):
                proof_epoch = f"ep{proof.get('last_proven_epoch', '?')}"
                proof_running = proof.get("running", False)

            # last verified relative time (from validator /instances age_sec)
            verified_age_sec = instance.get("age_sec")

            results.append({
                **ex,
                "reachable": health is not None,
                "gpu_name": gpu_name,
                "gpu_pct": gpu_pct,
                "proof_epoch": proof_epoch,
                "proof_running": proof_running,
                # Attestation
                "score": score_data.get("pass_rate_24h"),
                "proof_status_str": score_data.get("status", "—"),
                "avg_delta": score_data.get("avg_delta"),
                "trust_level": score_data.get("trust_level"),
                "verified_age_sec": verified_age_sec,
                # Rental
                "rental": rental,
                # Raw
                "hardware_full": hardware,
                "proof_full": proof,
                "instance": instance,
            })

    return results


# ── Rendering ──────────────────────────────────────────────────

# ANSI color helpers. Disabled if NO_COLOR env var is set or not a TTY.
def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


_C = {
    "reset":   "\033[0m" if _colors_enabled() else "",
    "bold":    "\033[1m" if _colors_enabled() else "",
    "dim":     "\033[2m" if _colors_enabled() else "",
    "red":     "\033[31m" if _colors_enabled() else "",
    "green":   "\033[32m" if _colors_enabled() else "",
    "yellow":  "\033[33m" if _colors_enabled() else "",
    "blue":    "\033[34m" if _colors_enabled() else "",
    "magenta": "\033[35m" if _colors_enabled() else "",
    "cyan":    "\033[36m" if _colors_enabled() else "",
    "white":   "\033[37m" if _colors_enabled() else "",
    "bg_red":     "\033[41m" if _colors_enabled() else "",
    "bg_green":   "\033[42m" if _colors_enabled() else "",
}


def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI colors. Each `codes` is a key in `_C`."""
    if not codes:
        return text
    pre = "".join(_C[k] for k in codes if k in _C)
    return f"{pre}{text}{_C['reset']}"


def _visible_len(s: str) -> int:
    """Length of `s` ignoring ANSI escape codes (for column alignment)."""
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def _pad(s: str, width: int, align: str = "<") -> str:
    """Left/right-pad `s` to `width` accounting for invisible ANSI codes."""
    visible = _visible_len(s)
    pad = max(0, width - visible)
    return s + " " * pad if align == "<" else " " * pad + s


def _clear():
    if not sys.stdout.isatty():
        return
    os.system("clear" if os.name != "nt" else "cls")


def _render_header(wallet: str, hotkey: str, hotkey_ss58: str, evm_addr: str,
                   uid, netuid: int, mg, source: str = "chain"):
    """Pretty role-aware header with color and aligned grid layout.

    Validators see vTrust + Dividends; miners see Incentive + Consensus.
    Both see Stake + Emission.
    """
    # Role detection: validator_permit is unreliable (everyone gets it on
    # small subnets / testnet). Use incentive vs vtrust signals — a miner
    # earns incentive but no vtrust/dividends, a validator earns vtrust +
    # dividends but no incentive.
    is_validator = False
    role = "MINER"
    role_color = "cyan"
    if source == "public-api":
        role = "FLEET"
    elif uid is not None:
        try:
            vt = float(mg.validator_trust[uid]) if hasattr(mg, "validator_trust") else 0.0
            div = float(mg.dividends[uid]) if hasattr(mg, "dividends") else 0.0
            inc = float(mg.incentive[uid]) if hasattr(mg, "incentive") else 0.0
        except Exception:
            vt = div = inc = 0.0
        if (vt > 0.01 or div > 0.01) and inc < 0.01:
            is_validator = True
            role = "VALIDATOR"
            role_color = "magenta"
        elif inc < 0.01 and vt < 0.01:
            role = "INACTIVE"
            role_color = "yellow"

    print()
    title = f" {_c('▎', role_color, 'bold')} {_c('NODEXO', 'bold', 'white')}  {_c('—', 'dim')}  {_c(role, role_color, 'bold')} {_c('Dashboard', 'dim')}"
    print(title)
    print(f" {_c('━' * 84, 'dim')}")

    # Identity block — two columns
    if uid is not None:
        uid_str = _c(str(uid), 'green', 'bold')
    elif source == "public-api":
        uid_str = _c("not queried", "dim")
    elif mg is None:
        uid_str = _c("metagraph unavailable", "yellow", "bold")
    else:
        uid_str = _c('NOT REGISTERED', 'red', 'bold')
    print(f"  {_c('Wallet', 'dim'):<22} {wallet}/{hotkey}")
    print(f"  {_c('UID', 'dim'):<22} {uid_str}    {_c('Subnet', 'dim')} {netuid}")
    print(f"  {_c('Hotkey SS58', 'dim'):<22} {_c(hotkey_ss58, 'white')}")
    print(f"  {_c('EVM address', 'dim'):<22} {_c(evm_addr, 'white')}")
    if source == "public-api":
        print(f"  {_c('Data source', 'dim'):<22} public API")
        print(f" {_c('━' * 84, 'dim')}")

    if uid is not None:
        def _g(attr):
            arr = getattr(mg, attr, None)
            try:
                return float(arr[uid]) if arr is not None and uid < len(arr) else 0.0
            except Exception:
                return 0.0

        emission = _g("emission")
        stake = _g("stake")
        # Color-tier helper: green if "good", yellow if mid, red if zero/bad.
        def _tier(val: float, good: float, mid: float) -> str:
            if val >= good: return "green"
            if val >= mid: return "yellow"
            return "red"

        print()
        print(f" {_c('━' * 84, 'dim')}")
        if is_validator:
            vtrust = _g("validator_trust")
            dividends = _g("dividends")
            row = [
                ("Stake",     f"{stake:>9.2f} α", _tier(stake, 1000, 100)),
                ("vTrust",    f"{vtrust:>9.2f}",  _tier(vtrust, 0.7, 0.3)),
                ("Dividends", f"{dividends:>9.4f}",_tier(dividends, 0.5, 0.05)),
                ("Emission",  f"{emission:>7.2f} α/tempo", _tier(emission, 50, 5)),
            ]
        else:
            incentive = _g("incentive")
            consensus = _g("consensus")
            row = [
                ("Stake",     f"{stake:>9.2f} α", _tier(stake, 1000, 100)),
                ("Incentive", f"{incentive:>9.4f}", _tier(incentive, 0.5, 0.05)),
                ("Consensus", f"{consensus:>9.2f}", _tier(consensus, 0.7, 0.3)),
                ("Emission",  f"{emission:>7.2f} α/tempo", _tier(emission, 50, 5)),
            ]
        # Render as 2x2 grid
        for i in range(0, len(row), 2):
            left = row[i]
            right = row[i + 1] if i + 1 < len(row) else None
            left_str = f"  {_c(left[0], 'dim'):<22} {_c(left[1], left[2], 'bold')}"
            print(_pad(left_str, 50) + (
                f"  {_c(right[0], 'dim'):<22} {_c(right[1], right[2], 'bold')}" if right else ""
            ))
        print(f" {_c('━' * 84, 'dim')}")


def _print_section(title: str):
    print()
    print(f" {_c(title, 'bold', 'white')}")


def _print_table(rows: list[list[tuple[str, int, str]]], col_specs: list[tuple[str, int, str]]):
    """Render a table given column specs [(name, width, align)] and rows of cells.

    Each cell is the raw string (may contain ANSI codes); width/align are reused
    from col_specs.
    """
    hdr = " " + " ".join(_c(_pad(n, w, a), 'dim', 'bold') for n, w, a in col_specs)
    sep = " " + " ".join(_c("─" * w, 'dim') for _, w, _a in col_specs)
    print(hdr)
    print(sep)
    for row in rows:
        cells = [_pad(cell, w, a) for cell, (_, w, a) in zip(row, col_specs)]
        print(" " + " ".join(cells))


class _Spinner:
    """Braille-dot spinner. Use as a context manager:

        with _Spinner("Requesting rental"):
            # do slow work

    Stops on __exit__ and clears the line. Cheap (50ms tick).
    """
    CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str):
        self.message = message
        self._thread = None
        self._stop = None

    def __enter__(self):
        import threading
        # No animation when not a TTY (CI / piped) — just print the message
        # once so logs aren't littered with control codes.
        if not sys.stderr.isatty():
            print(f"  {self.message}…", file=sys.stderr, flush=True)
            return self
        self._stop = threading.Event()
        def _loop():
            i = 0
            while not self._stop.is_set():
                ch = self.CHARS[i % len(self.CHARS)]
                sys.stderr.write(f"\r  {_C['cyan']}{ch}{_C['reset']} {self.message}…")
                sys.stderr.flush()
                self._stop.wait(0.08)
                i += 1
            sys.stderr.write("\r" + " " * (len(self.message) + 8) + "\r")
            sys.stderr.flush()
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)


def _wait_for_key(timeout: float) -> str | None:
    """Wait up to `timeout` seconds for a single keypress, return None on timeout.

    Uses cbreak so a single char comes through without Enter. Restores the
    terminal regardless. Falls back to plain sleep when stdin isn't a TTY
    (piped input, captured output) — the watch loop then degrades to
    refresh-only mode, ctrl-C to exit.
    """
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return None
    import select as _select
    import termios as _termios
    import tty as _tty
    fd = sys.stdin.fileno()
    old = _termios.tcgetattr(fd)
    try:
        _tty.setcbreak(fd)
        rlist, _, _ = _select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            return ch
        return None
    finally:
        _termios.tcsetattr(fd, _termios.TCSADRAIN, old)


def _reliability_str(rel, samples) -> str:
    """Render reliability for marketplace tables.

    Render confidence instead of a raw check count; detailed API fields keep the
    count available, while the renter-facing table stays readable.
    """
    if rel is None or samples <= 0:
        return _c("warming up", "dim")
    pct = rel * 100
    color = "green" if pct >= 95 else ("yellow" if pct >= 80 else "red")
    if samples >= 100:
        confidence = "high"
    elif samples >= 20:
        confidence = "medium"
    elif samples >= 5:
        confidence = "limited"
    else:
        confidence = "warming"
    return f"{_c(f'{pct:.1f}%', color, 'bold')} {_c(confidence, 'dim')}"


def _humanize_age(seconds) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    if s < 60: return f"{s}s ago"
    if s < 3600: return f"{s // 60}m ago"
    if s < 86400: return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def _format_alpha(value) -> str:
    try:
        v = float(value or 0.0)
    except Exception:
        v = 0.0
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _format_unix(ts) -> str:
    try:
        value = float(ts or 0.0)
    except Exception:
        value = 0.0
    if value <= 0:
        return "—"
    from datetime import datetime
    return datetime.fromtimestamp(value).strftime("%b %d %H:%M")


def _stake_summary(miners: list[dict]) -> dict:
    current = 0.0
    required = 0.0
    grace_until = 0.0
    ok = True
    in_grace = False
    for miner in miners:
        score = ((miner.get("instance") or {}).get("score") or {})
        req = float(score.get("miner_required_stake_alpha") or 0.0)
        if req <= 0:
            continue
        required = max(required, req)
        current = max(current, float(score.get("miner_stake_alpha") or 0.0))
        grace_until = max(grace_until, float(score.get("stake_grace_until_ts") or 0.0))
        ok = ok and bool(score.get("stake_ok", False))
        in_grace = in_grace or bool(score.get("stake_in_grace", False))
    return {
        "current": current,
        "required": required,
        "grace_until": grace_until,
        "ok": ok,
        "in_grace": in_grace,
    }


def _is_rental_locked(miner: dict) -> bool:
    """True when the executor is locked for a rental even if details are gated.

    Fleet views must not rely only on the validator /rentals endpoint: that
    route is internal-auth protected and also only knows rentals created by
    this validator. The registry is the availability source of truth.
    """
    if miner.get("rental"):
        return True
    if miner.get("is_rented"):
        return True
    instance = miner.get("instance") or {}
    return instance.get("alloc_state") == "rented"


def _netuid_hint(network: str) -> int:
    return {"test": 468, "finney": 106}.get((network or "").strip(), 0)


def _public_instance_to_miner(inst: dict, offer: dict | None = None) -> dict:
    """Map public /api/instances rows into the fleet renderer shape."""
    hw = inst.get("hardware") or {}
    proof = inst.get("proof") or {}
    live = inst.get("live") or {}

    gpu_name = (hw.get("gpu_name") or "GPU").replace("NVIDIA ", "")
    gpu_count = int(hw.get("gpu_count") or 1)
    vram_bytes = int(hw.get("gpu_vram_bytes") or 0)
    vram_mb = vram_bytes // (1 << 20)
    if not vram_mb and hw.get("gpu_vram_mb"):
        vram_mb = int(hw.get("gpu_vram_mb") or 0)

    status = (inst.get("status") or "").lower()
    alloc_state = inst.get("alloc_state") or "free"
    endpoint = inst.get("endpoint") or ""
    last_epoch = proof.get("last_epoch")

    # Public API rows are validator-observed live rows, not raw registry rows.
    # `fail` still means reachable but currently failing verification.
    reachable = bool(endpoint) and status not in {"offline", "unreachable", "stale"}
    active = status not in {"offline", "inactive", "deregistered"}

    return {
        "executor_id": inst.get("instance_id") or "",
        "endpoint": endpoint,
        "gpu_model_hash": "",
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "vram_mb": vram_mb,
        "price_rao": 0,
        "price_usdc_per_hour": (offer or {}).get("price_usdc_per_hour"),
        "registered_at": None,
        "expires_at": None,
        "is_active": active,
        "is_rented": alloc_state == "rented",
        "reachable": reachable,
        "gpu_pct": f"{float(live.get('gpu_pct') or 0):.0f}%" if live else "—",
        "proof_epoch": f"ep{last_epoch}" if last_epoch is not None else "—",
        "proof_running": False,
        "score": proof.get("pass_rate_24h"),
        "proof_status_str": status or "—",
        "avg_delta": None,
        "trust_level": proof.get("trust_level"),
        "verified_age_sec": inst.get("age_sec"),
        "rental": None,
        "hardware_full": hw,
        "proof_full": proof,
        "instance": inst,
    }


def _fleet_price_str(miner: dict) -> str:
    if miner.get("price_usdc_per_hour") is not None:
        return _c(f"${float(miner['price_usdc_per_hour']):.2f}/hr", "white")
    price_per_gpu_hour = miner.get("price_rao", 0)
    if price_per_gpu_hour:
        return _c(f"{price_per_gpu_hour} RAO/hr", "white")
    return _c("not priced", "dim")


def _render_table(miners: list[dict]):
    """Three-section dashboard: Executors / Attestation / Rental.

    Each section answers one question:
      - EXECUTORS: identity, hardware, on-chain reachability
      - ATTESTATION: how the validator verifies these executors
      - RENTAL: business state — who's renting, when, ssh/container details
    """

    # ── Section 1: EXECUTORS ──────────────────────────────────────
    _print_section("EXECUTORS")
    cols = [
        ("#",        4,  ">"),
        ("Executor", 18, "<"),
        ("GPU",      22, "<"),
        ("Cnt",      4,  ">"),
        ("VRAM",     7,  ">"),
        ("Status",   10, "<"),
        ("Endpoint", 32, "<"),
    ]
    rows = []
    dash = _c("—", "dim")
    for i, m in enumerate(miners):
        eid = m["executor_id"][:16] + "…"
        if not m["reachable"] and m["is_active"]:
            status_str = _c("UNREACH", "red", "bold")
        elif m["is_active"] and m["reachable"]:
            status_str = _c("ACTIVE", "green", "bold")
        else:
            status_str = _c("OFFLINE", "yellow", "bold")
        host = m["endpoint"].replace("https://", "").replace("http://", "").split("/")[0]
        if len(host) > 32:
            host = host[:29] + "…"

        # If we can't reach the executor, the on-chain spec is unverifiable —
        # dim the row and don't pretend we know the hardware.
        if m["reachable"]:
            gpu_name = (m.get("gpu_name") or "GPU")[:22]
            gpu_cnt = str(m["gpu_count"])
            vram = f"{m['vram_mb'] // 1024} GB"
            host_str = host
        else:
            gpu_name = dash
            gpu_cnt = dash
            vram = dash
            host_str = _c(host, "dim")

        rows.append([str(i), eid, gpu_name, gpu_cnt, vram, status_str, host_str])
    _print_table(rows, cols)

    # ── Section 2: ATTESTATION ────────────────────────────────────
    _print_section("ATTESTATION")
    cols = [
        ("#",         4,  ">"),
        ("Executor",  18, "<"),
        ("Method",    9,  "<"),
        ("Tier",      7,  "<"),
        ("Pass",      9,  ">"),
        ("Cycle",     11, "<"),
        ("Verified",  12, "<"),
    ]
    rows = []
    for i, m in enumerate(miners):
        eid = m["executor_id"][:16] + "…"
        # Method: ZkGEMM is the only method today; future TEE will add another value
        if m.get("score") is None:
            method = _c("—", "dim")
            tier = _c("—", "dim")
            pass_str = _c("—", "dim")
            verified = _c("—", "dim")
        else:
            method = _c("ZkGEMM", "white")
            tier_val = m.get("trust_level") or "—"
            tier_color = {"light": "green", "spot": "cyan", "full": "yellow"}.get(tier_val, "dim")
            tier = _c(tier_val, tier_color)
            score = m["score"]
            pct = score * 100
            pass_color = "green" if pct >= 95 else ("yellow" if pct >= 70 else "red")
            pass_str = _c(f"{pct:.0f}%", pass_color, "bold")
            verified = _c(_humanize_age(m.get("verified_age_sec")), "white")
        cycle = m["proof_epoch"]
        cycle_str = _c(cycle, "white") if cycle != "—" else _c(cycle, "dim")
        rows.append([str(i), eid, method, tier, pass_str, cycle_str, verified])
    _print_table(rows, cols)

    # ── Section 3: RENTAL ─────────────────────────────────────────
    _print_section("RENTAL")
    cols = [
        ("#",         4,  ">"),
        ("Executor",  18, "<"),
        ("Status",    11, "<"),
        ("Started",   12, "<"),
        ("TTL Left",  10, ">"),
        ("Price",     14, ">"),
        ("Renter / SSH", 28, "<"),
    ]
    rows = []
    for i, m in enumerate(miners):
        eid = m["executor_id"][:16] + "…"
        rental = m.get("rental")
        price_str = _fleet_price_str(m)

        if rental:
            status_str = _c("RENTED", "magenta", "bold")
            from datetime import datetime
            try:
                started = datetime.fromisoformat(rental.get("created_at", ""))
                started_str = started.strftime("%H:%M:%S")
            except Exception:
                started_str = "—"
            ttl = rental.get("ttl_seconds", 0)
            ttl_left_str = _humanize_age(ttl) if ttl else "—"
            ssh = f"{rental.get('ssh_host', '')}:{rental.get('ssh_port', '')}"
            renter = ssh if ssh != ":" else "—"
        elif _is_rental_locked(m):
            status_str = _c("RENTED", "magenta", "bold")
            started_str = _c("—", "dim")
            ttl_left_str = _c("—", "dim")
            renter = _c("renter private", "dim")
        elif m["is_active"] and m["reachable"]:
            status_str = _c("AVAILABLE", "green", "bold")
            started_str = _c("—", "dim")
            ttl_left_str = _c("—", "dim")
            renter = _c("—", "dim")
        else:
            status_str = _c("—", "dim")
            started_str = _c("—", "dim")
            ttl_left_str = _c("—", "dim")
            renter = _c("—", "dim")
        rows.append([str(i), eid, status_str, started_str, ttl_left_str, price_str, renter])
    _print_table(rows, cols)

    # ── Footer ────────────────────────────────────────────────────
    print()
    active = sum(1 for m in miners if m["is_active"] and m["reachable"] and not _is_rental_locked(m))
    rented = sum(1 for m in miners if _is_rental_locked(m))
    offline = sum(1 for m in miners if not m["is_active"])
    unreach = sum(1 for m in miners if m["is_active"] and not m["reachable"])
    total_gpus = sum(m["gpu_count"] for m in miners if m["is_active"] and m["reachable"])
    total_vram = sum(m["vram_mb"] // 1024 for m in miners if m["is_active"] and m["reachable"])

    parts = [
        _c(f"Available {active}", "green", "bold"),
        _c(f"Rented {rented}", "magenta", "bold") if rented else _c(f"Rented {rented}", "dim"),
        _c(f"Unreach {unreach}", "red", "bold") if unreach else _c(f"Unreach {unreach}", "dim"),
        _c(f"Offline {offline}", "yellow", "bold") if offline else _c(f"Offline {offline}", "dim"),
        _c(f"GPUs {total_gpus} ({total_vram} GB total)", "cyan", "bold"),
    ]
    stake = _stake_summary(miners)
    if stake["required"] > 0:
        if stake["ok"]:
            stake_part = _c(
                f"Stake {_format_alpha(stake['current'])}/{_format_alpha(stake['required'])} α",
                "green",
                "bold",
            )
        elif stake["in_grace"] and stake["grace_until"] > time.time():
            stake_part = _c(
                f"Stake grace {_format_alpha(stake['current'])}/{_format_alpha(stake['required'])} α until {_format_unix(stake['grace_until'])}",
                "yellow",
                "bold",
            )
        else:
            stake_part = _c(
                f"Stake low {_format_alpha(stake['current'])}/{_format_alpha(stake['required'])} α",
                "red",
                "bold",
            )
        parts.append(stake_part)
    print("  " + _c("│", "dim").join(f" {p} " for p in parts))


def _render_detail(miner: dict, idx: int, total: int):
    """Detail view — same three-section layout as the main table."""
    print()
    print(f" {_c('▎', 'cyan', 'bold')} {_c(f'Executor [{idx + 1}/{total}]', 'bold', 'white')}  {_c(miner['executor_id'][:32] + '…', 'dim')}")
    print(f" {_c('━' * 84, 'dim')}")

    # ── IDENTITY ──────────────────────────────────────────────
    if not miner["reachable"] and miner["is_active"]:
        status_str = _c("UNREACHABLE", "red", "bold")
    elif _is_rental_locked(miner):
        status_str = _c("RENTED", "magenta", "bold")
    elif miner["is_active"]:
        status_str = _c("ACTIVE", "green", "bold")
    else:
        status_str = _c("OFFLINE", "yellow", "bold")
    print(f"\n {_c('IDENTITY', 'bold', 'white')}")
    print(f"   {_c('Executor ID', 'dim'):<22} {miner['executor_id']}")
    print(f"   {_c('Endpoint', 'dim'):<22} {miner['endpoint']}")
    print(f"   {_c('Status', 'dim'):<22} {status_str}")
    print(f"   {_c('GPU', 'dim'):<22} {miner['gpu_count']} × {miner.get('gpu_name', '?')} ({miner['vram_mb']} MB each)")
    expires_at = miner.get("expires_at")
    if expires_at:
        ttl_h = max(0, (expires_at - int(time.time())) / 3600)
        ttl_color = "red" if ttl_h < 1 else ("yellow" if ttl_h < 6 else "green")
        ttl_str = _c(f"{ttl_h:.1f}h", ttl_color)
        print(f"   {_c('Registration lease', 'dim'):<22} expires in {ttl_str} {_c('(auto-renewed by miner heartbeat)', 'dim')}")
    else:
        print(f"   {_c('Registration lease', 'dim'):<22} {_c('not exposed by public API', 'dim')}")

    # ── ATTESTATION ────────────────────────────────────────────
    print(f"\n {_c('ATTESTATION', 'bold', 'white')}")
    if miner.get("score") is None:
        print(f"   {_c('No data — pass --validator-url to fetch', 'dim')}")
    else:
        score_pct = miner["score"] * 100
        score_color = "green" if score_pct >= 95 else ("yellow" if score_pct >= 70 else "red")
        tier_val = miner.get("trust_level") or "—"
        tier_color = {"light": "green", "spot": "cyan", "full": "yellow"}.get(tier_val, "dim")
        instance = miner.get("instance") or {}
        proof = instance.get("proof") or {}
        samples_1h = proof.get("samples_1h", "—")
        samples_24h = proof.get("samples_24h", "—")
        last_cycle = proof.get("last_epoch", miner["proof_epoch"])
        verified_age = _humanize_age(miner.get("verified_age_sec"))

        print(f"   {_c('Method', 'dim'):<22} {_c('ZkGEMM', 'white')}")
        print(f"   {_c('Tier', 'dim'):<22} {_c(tier_val, tier_color, 'bold')}")
        print(f"   {_c('Pass rate (24h)', 'dim'):<22} {_c(f'{score_pct:.1f}%', score_color, 'bold')}  ({samples_24h} checks)")
        print(f"   {_c('Pass rate (1h)', 'dim'):<22} ({samples_1h} checks)")
        print(f"   {_c('Last verified', 'dim'):<22} {_c(verified_age, 'white')} (cycle {last_cycle})")
        score_obj = instance.get("score") or {}
        required_alpha = float(score_obj.get("miner_required_stake_alpha") or 0.0)
        if required_alpha > 0:
            current_alpha = float(score_obj.get("miner_stake_alpha") or 0.0)
            if score_obj.get("stake_ok"):
                stake_state = _c("pass", "green", "bold")
            elif score_obj.get("stake_in_grace") and float(score_obj.get("stake_grace_until_ts") or 0.0) > time.time():
                stake_state = _c(f"grace until {_format_unix(score_obj.get('stake_grace_until_ts'))}", "yellow", "bold")
            else:
                stake_state = _c("low", "red", "bold")
            print(
                f"   {_c('Alpha stake', 'dim'):<22} "
                f"{_format_alpha(current_alpha)} / {_format_alpha(required_alpha)} α  {stake_state}"
            )
        if miner.get("avg_delta") is not None and miner["avg_delta"] > 0:
            print(f"   {_c('Avg per-GPU delta', 'dim'):<22} {miner['avg_delta']:.2f}s")

    # ── RENTAL ─────────────────────────────────────────────────
    print(f"\n {_c('RENTAL', 'bold', 'white')}")
    rental = miner.get("rental")
    if rental:
        print(f"   {_c('Status', 'dim'):<22} {_c('RENTED', 'magenta', 'bold')}")
        print(f"   {_c('Rental ID', 'dim'):<22} {rental.get('rental_id', '—')}")
        print(f"   {_c('Started at', 'dim'):<22} {rental.get('created_at', '—')}")
        ttl_s = rental.get("ttl_seconds", 0)
        print(f"   {_c('TTL', 'dim'):<22} {_humanize_age(ttl_s)}")
        print(f"   {_c('Container', 'dim'):<22} {rental.get('container_name', '—')}")
        ssh = f"{rental.get('ssh_user', 'user')}@{rental.get('ssh_host', '')}:{rental.get('ssh_port', '')}"
        print(f"   {_c('SSH endpoint', 'dim'):<22} {_c(ssh, 'cyan')}")
    elif _is_rental_locked(miner):
        print(f"   {_c('Status', 'dim'):<22} {_c('RENTED', 'magenta', 'bold')}")
        print(f"   {_c('Details', 'dim'):<22} {_c('renter details are private in public API mode', 'dim')}")
        print(f"   {_c('Price', 'dim'):<22} {_fleet_price_str(miner)}")
    elif miner["is_active"] and miner["reachable"]:
        print(f"   {_c('Status', 'dim'):<22} {_c('AVAILABLE', 'green', 'bold')}")
        print(f"   {_c('Price', 'dim'):<22} {_fleet_price_str(miner)}")
    else:
        print(f"   {_c('Status', 'dim'):<22} {_c('—', 'dim')}  {_c('(executor not reachable)', 'dim')}")

    # ── LIVE METRICS ───────────────────────────────────────────
    instance = miner.get("instance") or {}
    live = instance.get("live") or {}
    hw = instance.get("hardware") or {}
    if live or hw:
        print(f"\n {_c('LIVE METRICS', 'bold', 'white')}")
        if "gpu_pct" in live:
            gpu_pct = live["gpu_pct"]
            bar_len = int(gpu_pct / 5)
            bar = _c("█" * bar_len, "cyan") + _c("░" * (20 - bar_len), "dim")
            mem_used = live.get("gpu_vram_used_bytes", 0) // (1 << 30)
            # Live total is unreliable on some miners; fall back to on-chain vram_mb.
            mem_total = (live.get("gpu_vram_total_bytes", 0) // (1 << 30)) or (miner.get("vram_mb", 0) // 1024)
            vram_str = f"{mem_used} / {mem_total} GB VRAM" if mem_total else f"{mem_used} GB used"
            print(f"   {_c('GPU util', 'dim'):<22} {bar} {gpu_pct:>3.0f}%   ({vram_str})")
        if "gpu_temp_c" in live:
            print(f"   {_c('GPU temp / power', 'dim'):<22} {live['gpu_temp_c']:.0f}°C  /  {live.get('gpu_power_w', 0):.0f} W")
        if "cpu_pct" in live:
            ram_used = live.get("ram_used_bytes", 0) // (1 << 30)
            print(f"   {_c('CPU / RAM', 'dim'):<22} {live['cpu_pct']:.0f}%  /  {ram_used} GB used")
        if hw.get("cpu_model"):
            print(f"   {_c('Host', 'dim'):<22} {hw['cpu_model']}, {hw.get('cpu_cores', '?')} cores")

    print(f"\n {_c('━' * 84, 'dim')}")
    # Footer is intentionally NOT printed here — caller chooses what to show:
    # - static interactive mode (`_interactive`) prints its own one-liner
    # - watch mode prints its footer with `auto-refresh Ns` appended
    # Mixing both produced a double footer, so this stays caller-driven.


def _render_help():
    """Help overlay — shown when the operator presses [h]."""
    print()
    print(f" {_c('▎', 'cyan', 'bold')} {_c('Help', 'bold', 'white')}")
    print(f" {_c('━' * 84, 'dim')}")

    print(f"\n {_c('SECTIONS', 'bold', 'white')}")
    print(f"   {_c('EXECUTORS', 'dim'):<22} on-chain identity, GPU, reachability")
    print(f"   {_c('ATTESTATION', 'dim'):<22} how the validator verifies each executor")
    print(f"   {_c('RENTAL', 'dim'):<22} who is renting, when, ssh/container details")

    print(f"\n {_c('ATTESTATION COLUMNS', 'bold', 'white')}")
    print(f"   {_c('Method', 'dim'):<22} proof system in use (ZkGEMM today; TEE in future)")
    print(f"   {_c('Tier', 'dim'):<22} verification depth — full (heavy crypto, ~2.2s)")
    print(f"   {' ':<22} or spot (probabilistic, ~16ms)")
    print(f"   {_c('Pass', 'dim'):<22} 24h pass rate; <50% retriggers full-mode penalty verify")
    print(f"   {_c('Cycle', 'dim'):<22} last proof cycle the validator scored")
    print(f"   {_c('Verified', 'dim'):<22} time since the last verification result")

    print(f"\n {_c('KEYS', 'bold', 'white')}")
    print(f"   {_c('0-9', 'cyan'):<22} drill into executor detail")
    print(f"   {_c('h', 'cyan'):<22} this help")
    print(f"   {_c('q', 'cyan'):<22} quit")
    print(f"   {_c('n / p', 'cyan'):<22} next / prev (in detail view)")
    print(f"   {_c('b', 'cyan'):<22} back to table (in detail view)")
    print(f"   {_c('d', 'cyan'):<22} deregister this executor on-chain (in detail view)")

    print(f"\n {_c('CLI FLAGS', 'bold', 'white')}")
    print(f"   {_c('--watch [N]', 'cyan'):<22} non-interactive auto-refresh every N seconds (default 30)")
    print(f"   {_c('--validator-url', 'cyan'):<22} pull pass-rate / live metrics from a validator API")

    print(f"\n {_c('━' * 84, 'dim')}")
    # Caller prints the footer (interactive: "  > " prompt; watch: nav keys
    # + auto-refresh). Keeping the back-hint here would duplicate.


# ── Interactive loop ───────────────────────────────────────────

def _interactive(miners: list[dict], header_fn, deregister_fn=None):
    """Interactive navigation: table view ↔ detail view ↔ help overlay.

    `deregister_fn(executor_id_hex) -> tx_hash | None` — invoked by `[d]` in the
    detail view. None means caller skipped (user said no, or call failed).
    On success the row is removed from `miners` in-place and the caller is
    bounced back to the table view.
    """
    mode = "table"  # "table" | "detail" | "help"
    prev_mode = "table"
    selected = 0

    while True:
        _clear()
        if mode != "help":
            header_fn()

        if mode == "table":
            _render_table(miners)
            print(f"\n  Select [0-{len(miners)-1}] · {_c('[h]', 'cyan')} help · {_c('[q]', 'cyan')} quit: ", end="", flush=True)

            try:
                inp = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if inp in ("q", "quit"):
                break
            elif inp in ("h", "help", "?"):
                prev_mode = "table"
                mode = "help"
                continue
            try:
                idx = int(inp)
                if 0 <= idx < len(miners):
                    selected = idx
                    mode = "detail"
            except ValueError:
                pass

        elif mode == "detail":
            _render_detail(miners[selected], selected, len(miners))
            print(
                f" {_c('[n]', 'cyan')} next  {_c('[p]', 'cyan')} prev  "
                f"{_c('[b]', 'cyan')} back  {_c('[d]', 'red')} deregister  "
                f"{_c('[h]', 'cyan')} help  {_c('[q]', 'cyan')} quit"
            )
            print("  > ", end="", flush=True)

            try:
                inp = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if inp in ("q", "quit"):
                break
            elif inp in ("h", "help", "?"):
                prev_mode = "detail"
                mode = "help"
                continue
            elif inp in ("b", "back", "t", "table"):
                mode = "table"
            elif inp in ("d", "deregister"):
                if deregister_fn is None:
                    print(f"  {_c('Deregister not available in this view.', 'yellow')}")
                    input("  press enter to continue ")
                    continue
                m = miners[selected]
                eid_hex = m["executor_id"]
                print(
                    f"\n  {_c('Deregister executor', 'red', 'bold')} "
                    f"{_c(eid_hex[:16] + '…', 'white')}?"
                )
                ep = m["endpoint"]
                print(f"  {_c(f'  endpoint: {ep}', 'dim')}")
                print(f"  {_c('  this is on-chain and irreversible.', 'dim')}")
                confirm = input(f"  type {_c('yes', 'red', 'bold')} to confirm: ").strip()
                if confirm != "yes":
                    print(f"  {_c('Aborted.', 'dim')}")
                    input("  press enter to continue ")
                    continue
                print(f"  {_c('Submitting tx...', 'dim')}", end="", flush=True)
                try:
                    tx = deregister_fn(eid_hex)
                except Exception as e:
                    print(f" {_c('failed', 'red')}: {e}")
                    input("  press enter to continue ")
                    continue
                if not tx:
                    print(f" {_c('cancelled', 'yellow')}")
                    input("  press enter to continue ")
                    continue
                print(f" {_c('done', 'green')} (tx={tx[:10]}…)")
                miners.pop(selected)
                if not miners:
                    print(f"\n  {_c('All executors deregistered.', 'dim')}")
                    return
                selected = min(selected, len(miners) - 1)
                mode = "table"
                input("  press enter to continue ")
            elif inp in ("n", "next", "right", ""):
                selected = (selected + 1) % len(miners)
            elif inp in ("p", "prev", "left"):
                selected = (selected - 1) % len(miners)
            else:
                try:
                    idx = int(inp)
                    if 0 <= idx < len(miners):
                        selected = idx
                except ValueError:
                    pass

        elif mode == "help":
            _render_help()
            print(f" {_c('[enter]', 'cyan')} back")
            print("  > ", end="", flush=True)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                break
            mode = prev_mode

    print()


# ── Commands ───────────────────────────────────────────────────

def cmd_status(args):
    """Interactive fleet dashboard."""
    from common.chain.wallet import load_hotkey_seed, derive_evm_account
    import bittensor as bt

    args.wallet, args.hotkey = _select_wallet(args.wallet, args.hotkey, args.subtensor_network)

    hotkey_seed = load_hotkey_seed(args.wallet, args.hotkey)
    evm_account = derive_evm_account(hotkey_seed)
    WalletCls = getattr(bt, "Wallet", None) or bt.wallet
    wallet = WalletCls(name=args.wallet, hotkey=args.hotkey)
    hotkey_ss58 = wallet.hotkey.ss58_address

    use_chain_direct = bool(getattr(args, "chain_direct", False))
    source = "chain" if use_chain_direct else "public-api"
    netuid = _netuid_hint(args.subtensor_network)
    rpc = None
    client = None

    if not use_chain_direct:
        async def _load_all():
            print("\n  Loading", end="", flush=True)
            t0 = time.time()
            import aiohttp
            import urllib.parse
            hotkey_q = urllib.parse.quote(hotkey_ss58, safe="")
            async with aiohttp.ClientSession() as session:
                instances_payload, inventory_payload = await asyncio.gather(
                    _fetch_json(session, f"{_api_url(args)}/instances?hotkey={hotkey_q}", timeout=8),
                    _fetch_json(session, f"{_api_url(args)}/inventory", timeout=8),
                    return_exceptions=True,
                )
            payload = instances_payload if isinstance(instances_payload, dict) else {}
            inventory = inventory_payload if isinstance(inventory_payload, dict) else {}
            offer_by_executor: dict[str, dict] = {}
            for offer in inventory.get("offers", []) or []:
                for eid in offer.get("executor_ids", []) or []:
                    offer_by_executor[eid] = offer
            rows = (payload or {}).get("instances", []) if isinstance(payload, dict) else []
            miners = [
                _public_instance_to_miner(inst, offer_by_executor.get(inst.get("instance_id", "")))
                for inst in rows
            ]
            if not getattr(args, "show_stale", False):
                miners = [m for m in miners if m.get("is_active")]
            print(f" · public-api {time.time()-t0:.1f}s")
            return None, miners

        uid = None
    else:
        from common.config import ChainConfig
        from common.chain.compute_registry import ComputeRegistryClient
        from common.chain.rpc import SubtensorRPC

        chain_config_path = _resolve_chain_config(args.chain_config, args.subtensor_network)
        cfg = ChainConfig.from_json(
            chain_config_path,
            subtensor_network=args.subtensor_network,
            chain_endpoint=args.subtensor_endpoint,
        )
        netuid = cfg.netuid
        rpc = SubtensorRPC(
            network=args.subtensor_network,
            netuid=cfg.netuid,
            chain_endpoint=args.subtensor_endpoint,
        )
        client = ComputeRegistryClient(cfg, private_key=evm_account.key.hex())

        # Run all I/O concurrently: bittensor connect+metagraph, EVM
        # getMinerExecutors, then aiohttp fan-out to all executors + validator.
        # This is opt-in because the normal CLI should not require RPC.
        async def _timed(label: str, fn, timeout: float, *, required: bool = True):
            t0 = time.time()
            try:
                result = await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
                print(f" · {label} {time.time()-t0:.1f}s", end="", flush=True)
                return result
            except asyncio.TimeoutError as e:
                print(f" · {label} {_c('TIMEOUT', 'red', 'bold')} after {timeout:.0f}s", end="", flush=True)
                msg = (
                    f"{label} timed out. Check --subtensor-network "
                    f"({args.subtensor_network}) and the chain config "
                    f"(compute_registry={cfg.compute_registry_address})."
                )
                if required:
                    raise RuntimeError(msg) from e
                return e
            except Exception as e:
                print(f" · {label} {_c('WARN', 'yellow', 'bold')}", end="", flush=True)
                if required:
                    raise RuntimeError(f"{label} failed: {e}") from e
                return e

        async def _load_all():
            print("\n  Loading", end="", flush=True)
            t0 = time.time()
            mg_result, raw_executors = await asyncio.gather(
                _timed("metagraph", rpc.get_metagraph, timeout=15.0, required=False),
                _timed("executors", client.contract.functions.getMinerExecutors(evm_account.address).call, timeout=15.0, required=True),
            )
            mg = None if isinstance(mg_result, Exception) else mg_result

            executors = [{
                "executor_id": ex[0].hex(),
                "endpoint": ex[1],
                "gpu_model_hash": ex[2].hex(),
                "gpu_name": "GPU",
                "gpu_count": ex[3],
                "vram_mb": ex[4],
                "price_rao": ex[5],
                "registered_at": ex[6],
                "expires_at": ex[7],
                "is_active": ex[8],
                "is_rented": ex[9],
            } for ex in raw_executors]

            if not getattr(args, "show_stale", False):
                now = int(time.time())
                executors = [e for e in executors if e["is_active"] and e["expires_at"] > now]

            if not executors:
                return mg, []

            t1 = time.time()
            miners = await _gather_miner_data(executors, args.validator_url)
            print(f" · live {time.time()-t1:.1f}s · total {time.time()-t0:.1f}s")
            return mg, miners

        uid = None

    try:
        mg, miners = asyncio.run(_load_all())
    except Exception as e:
        print(f"\n  {_c('Fleet load failed:', 'red', 'bold')} {e}", file=sys.stderr)
        if use_chain_direct:
            print(
                "  Public subtensor RPCs can rate-limit testnet diagnostics. "
                "Try `nodexo fleet` for the public API view, or retry "
                "`fleet --chain-direct` later.",
                file=sys.stderr,
            )
        _exit(1)

    def _refresh_uid(current_mg=None):
        if not use_chain_direct or rpc is None or current_mg is None:
            return None
        try:
            return rpc.get_uid_for_hotkey(hotkey_ss58)
        except Exception:
            return None

    uid = _refresh_uid(mg)

    # ── Watch mode ────────────────────────────────────────────────
    # Auto-refreshing dashboard with live keyboard nav. Single-key cbreak
    # input + select() lets us either: (a) timeout → refetch data, or (b)
    # respond to a keypress immediately. Same key bindings as the static
    # interactive mode (`[h]` help, `[0-9]` detail, `[d]` deregister, `[q]`
    # quit), plus `[r]` force-refresh-now.
    if args.watch and args.watch > 0:
        watch_interval = max(2, int(args.watch))
        mode = "table"   # "table" | "detail" | "help"
        selected = 0
        last_msg: str | None = None

        def deregister_now(executor_id_hex: str) -> str | None:
            if not use_chain_direct or client is None:
                raise RuntimeError("deregister requires fleet --chain-direct")
            try:
                return client.deregister_executor(bytes.fromhex(executor_id_hex))
            except Exception as e:
                raise RuntimeError(str(e)) from e

        def _footer(mode: str) -> str:
            base_keys = f"{_c('[h]', 'cyan')} help  {_c('[r]', 'cyan')} refresh  {_c('[q]', 'cyan')} quit"
            if mode == "table" and miners:
                keys = f"{_c(f'[0-{len(miners)-1}]', 'cyan')} detail  " + base_keys
            elif mode == "detail":
                keys = (
                    f"{_c('[n]', 'cyan')} next  {_c('[p]', 'cyan')} prev  "
                    f"{_c('[b]', 'cyan')} back  {_c('[d]', 'red')} deregister  " + base_keys
                )
            else:
                keys = f"{_c('[any]', 'cyan')} back  " + base_keys
            return f"  {keys}  ·  {_c(f'auto-refresh {watch_interval}s', 'dim')}"

        try:
            while True:
                _clear()
                if mode == "help":
                    _render_help()
                else:
                    _render_header(args.wallet, args.hotkey, hotkey_ss58, evm_account.address, uid, netuid, mg, source)
                    if not miners:
                        print("\n  No executors found for this hotkey in the selected data source.")
                    elif mode == "detail":
                        if selected >= len(miners):
                            selected = 0
                        _render_detail(miners[selected], selected, len(miners))
                    else:
                        _render_table(miners)

                if last_msg:
                    print(f"\n  {last_msg}")
                    last_msg = None
                print(f"\n{_footer(mode)}", flush=True)

                key = _wait_for_key(watch_interval)

                # Timeout → refresh and continue
                if key is None:
                    try:
                        mg, miners = asyncio.run(_load_all())
                        uid = _refresh_uid(mg)
                    except SystemExit:
                        raise
                    except Exception as e:
                        last_msg = _c(f"Refresh failed: {e}", "yellow")
                    continue

                k = key.lower()
                if k in ("q", "\x03", "\x04"):  # q, ctrl-c, ctrl-d
                    break
                if k == "r":
                    try:
                        mg, miners = asyncio.run(_load_all())
                        uid = _refresh_uid(mg)
                    except Exception as e:
                        last_msg = _c(f"Refresh failed: {e}", "yellow")
                    continue
                if k in ("h", "?"):
                    mode = "help" if mode != "help" else "table"
                    continue
                if mode == "help":
                    mode = "table"
                    continue
                if mode == "table":
                    if k.isdigit() and miners:
                        idx = int(k)
                        if 0 <= idx < len(miners):
                            selected = idx
                            mode = "detail"
                    continue
                # mode == "detail"
                if k in ("b", "t"):
                    mode = "table"
                elif k == "n":
                    selected = (selected + 1) % len(miners)
                elif k == "p":
                    selected = (selected - 1) % len(miners)
                elif k == "d":
                    # Pause auto-refresh during the on-chain confirmation prompt.
                    m = miners[selected]
                    ep = m["endpoint"]
                    _clear()
                    _render_header(args.wallet, args.hotkey, hotkey_ss58, evm_account.address, uid, netuid, mg, source)
                    print(
                        f"\n  {_c('Deregister', 'red', 'bold')} "
                        f"{_c(m['executor_id'][:16] + '…', 'white')}? "
                        f"{_c(f'(endpoint: {ep})', 'dim')}"
                    )
                    print(f"  {_c('  on-chain and irreversible.', 'dim')}")
                    print(f"  type {_c('yes', 'red', 'bold')} to confirm: ", end="", flush=True)
                    try:
                        confirm = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        confirm = ""
                    if confirm == "yes":
                        try:
                            tx = deregister_now(m["executor_id"])
                            last_msg = _c(f"Deregistered (tx={tx[:10]}…)", "green")
                        except Exception as e:
                            last_msg = _c(f"Deregister failed: {e}", "red")
                    else:
                        last_msg = _c("Deregister aborted.", "dim")
                    # Force a fresh fetch so the row drops on the next render.
                    try:
                        mg, miners = asyncio.run(_load_all())
                        uid = _refresh_uid(mg)
                    except Exception as e:
                        last_msg = _c(f"Refresh after deregister failed: {e}", "yellow")
                    if not miners:
                        mode = "table"
                    elif selected >= len(miners):
                        selected = 0
                        mode = "table"
                    else:
                        mode = "table"
        except KeyboardInterrupt:
            pass
        print()
        _exit(0)

    if not miners:
        _clear()
        _render_header(args.wallet, args.hotkey, hotkey_ss58, evm_account.address, uid, netuid, mg, source)
        print("\n  No executors found for this hotkey in the selected data source.\n")
        _exit(0)

    def header_fn():
        _render_header(args.wallet, args.hotkey, hotkey_ss58, evm_account.address, uid, netuid, mg, source)

    deregister_fn = None
    if use_chain_direct and client is not None:
        def deregister_fn(executor_id_hex: str) -> str | None:
            """Submit deregisterExecutor tx synchronously. Returns tx hash on success."""
            try:
                return client.deregister_executor(bytes.fromhex(executor_id_hex))
            except Exception as e:
                raise RuntimeError(str(e)) from e

    # Interactive loop
    _interactive(miners, header_fn, deregister_fn=deregister_fn)

    # bt v10's substrate websocket cleanup hangs ~30s at exit (library bug).
    # The CLI is read-only, so skipping clean shutdown is safe.
    _exit(0)


def cmd_deregister(args):
    """Deregister an on-chain executor. Accepts a full or prefix executor_id."""
    from common.config import ChainConfig
    from common.chain.compute_registry import ComputeRegistryClient
    from common.chain.wallet import load_hotkey_seed, derive_evm_account

    chain_config_path = _resolve_chain_config(args.chain_config, args.subtensor_network)
    args.wallet, args.hotkey = _select_wallet(args.wallet, args.hotkey, args.subtensor_network)

    cfg = ChainConfig.from_json(
        chain_config_path,
        subtensor_network=args.subtensor_network,
        chain_endpoint=args.subtensor_endpoint,
    )
    hotkey_seed = load_hotkey_seed(args.wallet, args.hotkey)
    evm_account = derive_evm_account(hotkey_seed)
    client = ComputeRegistryClient(cfg, private_key=evm_account.key.hex())

    raw = client.contract.functions.getMinerExecutors(evm_account.address).call()
    if not raw:
        print("\n  No executors registered to this wallet.\n")
        _exit(0)

    target = (args.executor_id or "").lower().lstrip("0x")
    matches = [ex for ex in raw if ex[0].hex().lower().startswith(target)] if target else []
    if not matches:
        print(f"\n  No executor matches prefix '{args.executor_id}'. Registered executors:")
        for ex in raw:
            print(f"    {ex[0].hex()}  endpoint={ex[1]}  active={ex[8]}")
        _exit(1)
    if len(matches) > 1:
        print(f"\n  Prefix '{args.executor_id}' is ambiguous ({len(matches)} matches):")
        for ex in matches:
            print(f"    {ex[0].hex()}  endpoint={ex[1]}")
        _exit(1)

    ex = matches[0]
    eid_hex = ex[0].hex()
    print(f"\n  Deregister executor {eid_hex}")
    print(f"    endpoint: {ex[1]}")
    print(f"    active:   {ex[8]}")
    print(f"    rented:   {ex[9]}")

    if not args.yes:
        confirm = input("\n  Type 'yes' to confirm: ").strip()
        if confirm != "yes":
            print("  Aborted.\n")
            _exit(0)

    print("  Submitting tx...", end="", flush=True)
    try:
        tx = client.deregister_executor(bytes.fromhex(eid_hex))
    except Exception as e:
        print(f" failed: {e}")
        _exit(1)
    print(f" done (tx={tx})\n")
    _exit(0)


_CONFIG_PATH = Path.home() / ".config" / "nodexo" / "config.toml"


def _default_subtensor_network() -> str:
    """Choose a sane default for the local checkout.

    Production installs normally ship chain_config_mainnet.json and should
    default to finney. Dev/testnet checkouts often only have
    chain_config_testnet.json; defaulting those to finney makes `nodexo
    status` fail before it can show the operator anything useful.
    """
    env = os.environ.get("NODEXO_SUBTENSOR_NETWORK", "").strip()
    if env:
        return env
    repo = Path(__file__).resolve().parent.parent
    if (repo / "chain_config_mainnet.json").exists():
        return "finney"
    if (repo / "chain_config_testnet.json").exists():
        return "test"
    return "finney"


def _default_api_url() -> str:
    """Read public renter API URL from env or ~/.config/nodexo/config.toml.

    This is the web app API, not the validator control plane. Public rental
    commands use it so x402, recovery tokens, rate limits, and billing stay in
    one place.
    """
    env = os.environ.get("NODEXO_API_URL", "").strip()
    if env:
        return env
    try:
        if _CONFIG_PATH.exists():
            try:
                import tomllib  # 3.11+
            except ImportError:
                import tomli as tomllib  # 3.10 fallback
            with open(_CONFIG_PATH, "rb") as f:
                cfg = tomllib.load(f)
            configured = (cfg.get("default", {}).get("api_url") or "").strip()
            if configured:
                return configured
    except Exception:
        pass
    return "https://nodexo.ai/api"


def _api_url(args) -> str:
    url = (getattr(args, "api_url", "") or _default_api_url()).strip()
    if not url:
        print("  No public API URL configured. Set one of:", file=sys.stderr)
        print("    --api-url https://nodexo.ai/api", file=sys.stderr)
        print("    export NODEXO_API_URL=https://nodexo.ai/api", file=sys.stderr)
        print(f"    {_CONFIG_PATH}:  [default]\\n    api_url = \"https://nodexo.ai/api\"", file=sys.stderr)
        _exit(2)
    url = url.rstrip("/")
    return url if url.endswith("/api") else f"{url}/api"


def _find_public_cli_script() -> str:
    """Locate the Node public API helper used for x402 signing.

    The Python CLI remains the main user/operator CLI. Public x402 signing uses
    the Node helper shipped in the release tree.
    """
    explicit = os.environ.get("NODEXO_PUBLIC_CLI", "").strip()
    if explicit and Path(explicit).exists():
        return explicit

    repo_helper = Path(__file__).resolve().parent.parent / "scripts" / "public-cli.mjs"
    if repo_helper.exists():
        return str(repo_helper)

    print("  Public API helper not found.", file=sys.stderr)
    print("  Set NODEXO_PUBLIC_CLI=/path/to/public-cli.mjs", file=sys.stderr)
    print("  or install the public helper dependencies in this repo:", file=sys.stderr)
    print("    npm install", file=sys.stderr)
    print("    node scripts/public-cli.mjs rent --gpu A6000 --duration 1h --ssh-key ~/.ssh/id_ed25519.pub", file=sys.stderr)
    _exit(2)


def _duration_to_hours(duration: str) -> int:
    """Convert CLI duration strings to the public x402 API's integer hours."""
    raw = (duration or "").strip().lower()
    if raw in ("", "1", "1h"):
        return 1
    if raw in ("unlimited", "infinite", "inf"):
        print("  Public x402 rentals require --duration 1h..168h.", file=sys.stderr)
        print("  Credit-metered unlimited rentals require wallet login in the web app.", file=sys.stderr)
        _exit(2)
    import re
    m = re.fullmatch(r"(\d+)([smhd]?)", raw)
    if not m:
        print(f"  Invalid duration '{duration}'. Use 1h, 4h, 24h, or 7d.", file=sys.stderr)
        _exit(2)
    value = int(m.group(1))
    unit = m.group(2) or "h"
    seconds = value
    if unit == "m":
        seconds = value * 60
    elif unit == "h":
        seconds = value * 3600
    elif unit == "d":
        seconds = value * 86400
    hours = max(1, (seconds + 3599) // 3600)
    if hours > 168:
        print("  Public x402 rentals are capped at 168h. Top up before expiry.", file=sys.stderr)
        _exit(2)
    if seconds % 3600:
        print(f"  Rounded duration '{duration}' up to {hours}h for x402.", file=sys.stderr)
    return hours


def _run_public_cli(args, command: str, extra: list[str]) -> None:
    import subprocess
    node = os.environ.get("NODE_BIN", "node")
    script = _find_public_cli_script()
    cmd = [node, script, command, "--api", _api_url(args), *extra]
    raise SystemExit(subprocess.run(cmd).returncode)


def _recovery_token_args(args) -> list[str]:
    token = (getattr(args, "recovery_token", "") or "").strip()
    return ["--recovery-token", token] if token else []


def _api_key_args(args) -> list[str]:
    key = (
        (getattr(args, "api_key", "") or "").strip()
        or os.environ.get("NODEXO_API_KEY", "").strip()
    )
    return ["--api-key", key] if key else []


def _public_auth_args(args) -> list[str]:
    api_key = _api_key_args(args)
    return api_key if api_key else _recovery_token_args(args)


def _public_rent_args(args) -> list[str]:
    pubkey_path, _pubkey = _resolve_ssh_pubkey(args.ssh_key)
    payment_mode = (getattr(args, "payment_mode", "") or "x402").strip().lower()
    if payment_mode not in ("x402", "credits"):
        print("  --payment must be x402 or credits.", file=sys.stderr)
        _exit(2)
    extra = [
        "--payment", payment_mode,
        "--gpu", args.gpu_model or "",
        "--gpu-count", str(int(args.gpu_count)),
        "--image", args.image,
        "--ssh-key", pubkey_path,
        "--storage-gb", str(int(args.storage_gb)),
        "--memory-gb", str(int(args.memory_gb)),
        *_api_key_args(args),
    ]
    if payment_mode == "x402":
        extra.extend(["--hours", str(_duration_to_hours(args.duration))])
    if getattr(args, "json", False):
        extra.append("--json")
    if getattr(args, "save", ""):
        extra.extend(["--save", args.save])
    if getattr(args, "idempotency_key", ""):
        extra.extend(["--idempotency-key", args.idempotency_key])
    if getattr(args, "max_usdc", ""):
        extra.extend(["--max-usdc", args.max_usdc])
    if getattr(args, "private_key_env", ""):
        extra.extend(["--private-key-env", args.private_key_env])
    return extra


def _recovery_token_value(args) -> str:
    api_key = (
        (getattr(args, "api_key", "") or "").strip()
        or os.environ.get("NODEXO_API_KEY", "").strip()
    )
    if api_key:
        return api_key
    token = (
        (getattr(args, "recovery_token", "") or "").strip()
        or os.environ.get("NODEXO_RENTAL_RECOVERY_TOKEN", "").strip()
    )
    if not token:
        print(
            "  Missing recovery token. Pass --recovery-token or set NODEXO_RENTAL_RECOVERY_TOKEN.",
            file=sys.stderr,
        )
        _exit(2)
    return token


def _fetch_public_rental(args, rental_id: str) -> dict:
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    rid = urllib.parse.quote(rental_id.strip(), safe="")
    req = urllib.request.Request(
        f"{_api_url(args)}/rentals/{rid}",
        headers={"Authorization": f"Bearer {_recovery_token_value(args)}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            pass
        print(f"  HTTP {e.code}: {detail or e.reason}", file=sys.stderr)
        _exit(1)
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        _exit(1)


def _default_validator_url() -> str:
    """Read default --validator-url from env or ~/.config/nodexo/config.toml.

    Resolution order (first match wins):
      1. NODEXO_VALIDATOR_URL environment variable
      2. [default] validator_url in config.toml
      3. empty string (the CLI will then refuse rental commands)
    """
    env = os.environ.get("NODEXO_VALIDATOR_URL", "").strip()
    if env:
        return env
    try:
        if _CONFIG_PATH.exists():
            try:
                import tomllib  # 3.11+
            except ImportError:
                import tomli as tomllib  # 3.10 fallback
            with open(_CONFIG_PATH, "rb") as f:
                cfg = tomllib.load(f)
            return (cfg.get("default", {}).get("validator_url") or "").strip()
    except Exception:
        pass
    return ""


def _validator_url(args) -> str:
    url = (args.validator_url or "").strip()
    if not url:
        print("  No validator URL configured. Set one of:", file=sys.stderr)
        print("    --validator-url https://validator.nodexo.ai", file=sys.stderr)
        print("    export NODEXO_VALIDATOR_URL=https://validator.nodexo.ai", file=sys.stderr)
        print(f"    {_CONFIG_PATH}:  [default]\\n    validator_url = \"https://validator.nodexo.ai\"", file=sys.stderr)
        _exit(2)
    return url.rstrip("/")


def _validator_request_headers(headers: dict | None = None) -> dict:
    out = dict(headers or {})
    admin = os.environ.get("NODEXO_ADMIN_TOKEN", "").strip()
    if admin:
        out["X-Admin-Token"] = admin
    return out


def _read_ssh_pubkey(path: str) -> str:
    """Resolve --ssh-key, defaulting to id_ed25519.pub then id_rsa.pub.

    Returns the key contents. For callers that ALSO need the path
    (to write IdentityFile into ~/.ssh/config), use _resolve_ssh_pubkey
    below which returns (content, path).
    """
    _, content = _resolve_ssh_pubkey(path)
    return content


def _resolve_ssh_pubkey(path: str) -> tuple[str, str]:
    """Resolve a usable SSH pubkey, returning (resolved_path, content).

    Lookup order:
      1. Explicit `path` (--ssh-key).
      2. ~/.ssh/id_ed25519.pub
      3. ~/.ssh/id_rsa.pub
      4. Offer to generate ~/.ssh/id_ed25519 in-place (interactive yes/no,
         or auto-yes when NODEXO_AUTO_GENERATE_KEY=1 / stdin not a TTY).

    Lets `rent` write the right IdentityFile into the ssh-config block
    even when --ssh-key wasn't passed (we'd previously lost the path).
    """
    import os as _os
    import subprocess as _sub
    candidates = [path] if path else [
        _os.path.expanduser("~/.ssh/id_ed25519.pub"),
        _os.path.expanduser("~/.ssh/id_rsa.pub"),
    ]
    for p in candidates:
        if p and _os.path.isfile(p):
            with open(p) as f:
                return p, f.read().strip()

    # No key found. Offer to generate one — the most common reason a renter
    # would hit this branch is "first rental on a fresh laptop".
    target_priv = _os.path.expanduser("~/.ssh/id_ed25519")
    target_pub = target_priv + ".pub"
    auto = _os.environ.get("NODEXO_AUTO_GENERATE_KEY", "") == "1" or not sys.stdin.isatty()
    if auto:
        confirm = "y"
        print(f"  No SSH key found; generating {target_priv} (auto)", file=sys.stderr)
    else:
        print(f"  No SSH key found. Generate one at {target_priv} now? [Y/n] ",
              end="", file=sys.stderr, flush=True)
        try:
            confirm = (input() or "y").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = "n"
    if confirm not in ("", "y", "yes"):
        print(f"  Aborted. Generate one yourself with: "
              f"ssh-keygen -t ed25519 -f {target_priv}", file=sys.stderr)
        _exit(2)
    _os.makedirs(_os.path.expanduser("~/.ssh"), mode=0o700, exist_ok=True)
    try:
        # -N "" → empty passphrase (non-interactive, dev convenience).
        # Operators with stricter posture can pre-create a passphrase-
        # protected key and pass --ssh-key explicitly.
        _sub.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", target_priv],
            check=True,
        )
    except (_sub.CalledProcessError, FileNotFoundError) as e:
        print(f"  ssh-keygen failed: {e}", file=sys.stderr)
        _exit(2)
    if not _os.path.isfile(target_pub):
        print(f"  Generated key not found at {target_pub}", file=sys.stderr)
        _exit(2)
    with open(target_pub) as f:
        return target_pub, f.read().strip()


def _ssh_jump_host_from_endpoint(endpoint: str) -> str | None:
    """Pull the host portion out of an http(s)://HOST:PORT endpoint, for ProxyJump.

    Returns None if the parse fails or the host is loopback (no jump needed).
    """
    try:
        from urllib.parse import urlparse
        u = urlparse(endpoint)
        host = u.hostname
        if not host or host in ("127.0.0.1", "::1", "localhost"):
            return None
        return host
    except Exception:
        return None


def cmd_rent(args):
    """Submit a public rental request and print the SSH command."""
    if not getattr(args, "validator_direct", False):
        _run_public_cli(args, "rent", _public_rent_args(args))

    """Submit a rental request to a validator gateway and print the SSH command."""
    import json
    import urllib.request
    import urllib.error

    url = _validator_url(args) + "/rent"
    pubkey_path, pubkey = _resolve_ssh_pubkey(args.ssh_key)
    # Remember the resolved path so the ~/.ssh/config block gets IdentityFile.
    # Was a real bug — when --ssh-key was omitted, args.ssh_key stayed "" and
    # the config block had no IdentityFile, breaking renters whose default
    # key wasn't in the standard search order.
    args.ssh_key = pubkey_path
    body = {
        "gpu_model": args.gpu_model or None,
        "gpu_count": int(args.gpu_count),
        "duration": args.duration,
        "ssh_pub_key": pubkey,
        "image": args.image,
        "storage_gb": int(args.storage_gb),
        "min_storage_gb": int(args.storage_gb),
        "memory_gb": int(args.memory_gb),
        "min_ram_gb": int(args.memory_gb),
    }

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers=_validator_request_headers({"Content-Type": "application/json"}),
    )
    try:
        with _Spinner(f"Requesting rental ({args.duration}, {args.gpu_count}× {args.gpu_model or 'GPU'})"):
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            pass
        print(f"  {_c('Rental failed:', 'red', 'bold')} HTTP {e.code} {detail}")
        _exit(1)
    except Exception as e:
        print(f"  {_c('Rental failed:', 'red', 'bold')} {e}")
        _exit(1)

    print(f"\n {_c('▎', 'green', 'bold')} {_c('Rental active', 'bold', 'green')}")
    print(f" {_c('━' * 76, 'dim')}")
    print(f"   {_c('Rental ID', 'dim'):<22} {_c(data['rental_id'], 'white', 'bold')}")
    gpu = data.get("gpu", {})
    if gpu:
        gpu_str = f"{gpu.get('count', 1)}× {gpu.get('model', '?').replace('NVIDIA ', '')}  ({gpu.get('vram_gb', '?')} GB VRAM)"
        print(f"   {_c('GPU', 'dim'):<22} {_c(gpu_str, 'cyan', 'bold')}")
    host = data.get("host", {})
    if host and (host.get("cpu_cores") or host.get("ram_gb")):
        host_str = f"{host.get('cpu_model', '?')[:40]}, {host.get('cpu_cores', '?')} cores, {host.get('ram_gb', '?')} GB RAM"
        if host.get("disk_gb"):
            host_str += f", {host['disk_gb']} GB disk"
        print(f"   {_c('Host', 'dim'):<22} {host_str}")
    resources = data.get("resources", {})
    if resources:
        resource_str = (
            f"{resources.get('storage_gb', '?')} GB storage, "
            f"{resources.get('memory_gb', '?')} GB RAM"
        )
        print(f"   {_c('Requested', 'dim'):<22} {resource_str}")
    conn = data["connection"]
    print(f"   {_c('SSH', 'dim'):<22} {conn['ssh_user']}@{conn['ssh_host']}:{conn['ssh_port']}")
    ttl = data.get("ttl_seconds", 0)
    if ttl <= 0:
        ttl_str = _c("open-ended", "yellow") + _c("  (no TTL; terminate explicitly via nodexo rental-end)", "dim")
    else:
        if ttl >= 86400:
            ttl_str = f"{ttl // 86400}d"
        elif ttl >= 3600:
            ttl_str = f"{ttl // 3600}h"
        elif ttl >= 60:
            ttl_str = f"{ttl // 60}m"
        else:
            ttl_str = f"{ttl}s"
        ttl_str = f"{ttl_str}  ({ttl}s)"
    print(f"   {_c('Duration', 'dim'):<22} {ttl_str}")
    price = data.get("price", {})
    if price.get("per_gpu_hour_rao"):
        price_str = f"{price['per_gpu_hour_rao']} RAO/GPU/h"
        if price.get("estimated_total_rao"):
            price_str += f"  ·  est. total {price['estimated_total_rao']} RAO"
        print(f"   {_c('Price', 'dim'):<22} {price_str}")
    else:
        print(f"   {_c('Price', 'dim'):<22} {_c('free (not listed on testnet)', 'dim')}")
    if data.get("image"):
        print(f"   {_c('Image', 'dim'):<22} {data['image']}")
    print(f" {_c('━' * 76, 'dim')}")

    # Build the actual ssh command. If the container's high port isn't open on
    # the provider firewall, we need ProxyJump through the executor host's
    # already-open SSH port. Add --no-proxy-jump to opt out.
    # ssh -i wants the PRIVATE key path; the user passed the public key, so
    # strip the .pub suffix when constructing the command.
    privkey = args.ssh_key[:-4] if args.ssh_key.endswith(".pub") else args.ssh_key
    keypart = f"-i {privkey} " if privkey else ""
    if args.no_proxy_jump or not args.jump_user:
        cmd = (
            f"ssh {keypart}"
            f"-o StrictHostKeyChecking=accept-new "
            f"-p {conn['ssh_port']} "
            f"{conn['ssh_user']}@{conn['ssh_host']}"
        )
        note = "(direct — requires container's port to be reachable)"
    else:
        # Inside the rented container the SSH server binds to localhost on
        # the host's high port, so the inner SSH goes to 127.0.0.1.
        cmd = (
            f"ssh {keypart}"
            f"-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null "
            f"-J {args.jump_user}@{conn['ssh_host']} "
            f"-p {conn['ssh_port']} "
            f"{conn['ssh_user']}@127.0.0.1"
        )
        note = f"(via ProxyJump through {args.jump_user}@{conn['ssh_host']} — only host SSH port needs to be open)"

    # Auto-write an ~/.ssh/config block so subsequent connects are short.
    rid = data["rental_id"]
    short_id = rid[:12]
    alias = f"nodexo-{short_id}"
    try:
        _write_ssh_config_block(
            alias=alias,
            ssh_host=conn["ssh_host"],
            ssh_port=conn["ssh_port"],
            ssh_user=conn["ssh_user"],
            jump_user=args.jump_user if not args.no_proxy_jump else "",
            identity_file=privkey,
        )
        config_written = True
    except Exception as e:
        config_written = False
        print(f"  {_c(f'(ssh-config write failed: {e})', 'yellow')}")

    print(f"\n  {_c('Connect:', 'bold', 'white')}")
    if config_written:
        print(f"    {_c(f'ssh {alias}', 'cyan', 'bold')}    {_c('(added to ~/.ssh/config)', 'dim')}")
        print(f"    {_c('— or the long form —', 'dim')}")
    print(f"    {_c(cmd, 'cyan')}")
    print(f"    {_c(note, 'dim')}")
    print(f"\n  {_c('Terminate when done:', 'bold', 'white')}")
    vurl = _validator_url(args)
    terminate_cmd = f"nodexo rental-end {rid}"
    print(f"    {_c(terminate_cmd, 'cyan')}")
    print()
    _exit(0)


def cmd_quote(args):
    """Preview x402 payment requirements without signing or provisioning."""
    _run_public_cli(args, "quote", _public_rent_args(args))


def cmd_credits(args):
    """Show account credit balance using an account API key."""
    extra = [*_api_key_args(args)]
    if getattr(args, "json", False):
        extra.append("--json")
    _run_public_cli(args, "credits", extra)


def cmd_account_rentals(args):
    """Show rentals attached to the account API key."""
    extra = [*_api_key_args(args)]
    if getattr(args, "json", False):
        extra.append("--json")
    _run_public_cli(args, "account-rentals", extra)


def cmd_account_ssh_keys(args):
    """List/add/remove saved SSH public keys attached to an account API key."""
    action = getattr(args, "account_ssh_key_action", "") or "list"
    extra = [*_api_key_args(args)]
    if getattr(args, "json", False):
        extra.append("--json")
    if action == "list":
        _run_public_cli(args, "account-ssh-keys", extra)
    if action == "add":
        if getattr(args, "ssh_key", ""):
            extra.extend(["--ssh-key", args.ssh_key])
        if getattr(args, "label", ""):
            extra.extend(["--label", args.label])
        _run_public_cli(args, "account-ssh-key-add", extra)
    if action == "remove":
        extra.extend(["--key-id", args.key_id])
        _run_public_cli(args, "account-ssh-key-remove", extra)
    print("  account-ssh-keys action must be list, add, or remove.", file=sys.stderr)
    _exit(2)


def _operator_claim_message(args) -> str:
    raw_b64 = (getattr(args, "message_base64", "") or "").strip()
    raw_file = (getattr(args, "message_file", "") or "").strip()
    raw_message = (getattr(args, "message", "") or "").strip()
    if sum(bool(x) for x in (raw_b64, raw_file, raw_message)) != 1:
        print(
            "  Pass exactly one of --message-base64, --message-file, or --message.",
            file=sys.stderr,
        )
        _exit(2)
    if raw_b64:
        import base64
        padded = raw_b64 + "=" * ((4 - len(raw_b64) % 4) % 4)
        try:
            return base64.urlsafe_b64decode(padded.encode()).decode("utf-8")
        except Exception as e:
            print(f"  Invalid --message-base64: {e}", file=sys.stderr)
            _exit(2)
    if raw_file:
        try:
            return Path(raw_file).expanduser().read_text(encoding="utf-8")
        except Exception as e:
            print(f"  Could not read --message-file: {e}", file=sys.stderr)
            _exit(2)
    return raw_message.replace("\\n", "\n")


def _operator_claim_hotkey_from_message(message: str) -> str:
    lines = message.splitlines()
    if len(lines) < 2 or "claim a Nodexo miner hotkey" not in lines[0]:
        print("  Message is not a Nodexo operator-claim challenge.", file=sys.stderr)
        _exit(2)
    return lines[1].strip()


def cmd_operator_claim(args):
    """Sign an operator dashboard claim with a local Bittensor hotkey."""
    if getattr(args, "operator_claim_command", "") != "sign":
        print("  Usage: nodexo operator-claim sign --message-base64 <challenge>", file=sys.stderr)
        _exit(2)

    from substrateinterface import Keypair
    from common.chain.wallet import load_hotkey_seed
    import bittensor as bt
    import json

    message = _operator_claim_message(args)
    claimed_hotkey = _operator_claim_hotkey_from_message(message)
    if args.wallet and args.hotkey:
        args.wallet, args.hotkey = _select_wallet(args.wallet, args.hotkey, args.subtensor_network)
    else:
        match = _find_wallet_for_hotkey_ss58(claimed_hotkey)
        if match is not None:
            args.wallet, args.hotkey = match
        elif sys.stdin.isatty():
            args.wallet, args.hotkey = _select_wallet(args.wallet, args.hotkey, args.subtensor_network)
        else:
            print(
                "  No local wallet/hotkey matching the challenge hotkey was found. "
                "Pass --wallet and --hotkey explicitly.",
                file=sys.stderr,
            )
            _exit(1)

    local_hotkey = _wallet_hotkey_ss58(args.wallet, args.hotkey)
    if claimed_hotkey != local_hotkey:
        print("  Hotkey mismatch.", file=sys.stderr)
        print(f"    challenge: {claimed_hotkey}", file=sys.stderr)
        print(f"    selected:  {local_hotkey} ({args.wallet}/{args.hotkey})", file=sys.stderr)
        _exit(1)

    seed = load_hotkey_seed(args.wallet, args.hotkey)
    keypair = Keypair.create_from_seed(seed[:32].hex())
    raw_sig = keypair.sign(message.encode("utf-8"))
    sig_bytes = raw_sig if isinstance(raw_sig, bytes) else bytes.fromhex(str(raw_sig).replace("0x", ""))
    verifier = Keypair(ss58_address=local_hotkey)
    if not verifier.verify(message.encode("utf-8"), sig_bytes):
        print("  Local signature verification failed.", file=sys.stderr)
        _exit(1)
    signature = "0x" + sig_bytes.hex()
    payload = {
        "hotkey": local_hotkey,
        "wallet": args.wallet,
        "hotkey_name": args.hotkey,
        "signature": signature,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
    else:
        print()
        print(f"  {_c('Operator claim signature', 'bold', 'white')}")
        print(f"  {_c('Hotkey', 'dim'):<16} {local_hotkey}")
        print(f"  {_c('Wallet', 'dim'):<16} {args.wallet}/{args.hotkey}")
        print(f"  {_c('Signature', 'dim'):<16} {signature}")
        print()


def _write_ssh_config_block(alias: str, ssh_host: str, ssh_port: int,
                            ssh_user: str, jump_user: str, identity_file: str) -> None:
    """Write/refresh a nodexo-rental block in ~/.ssh/config.

    Replaces any existing block with the same alias. Idempotent.
    """
    import os as _os
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    cfg_path = ssh_dir / "config"
    cfg_path.touch(mode=0o600, exist_ok=True)

    block_start = f"# BEGIN nodexo rental {alias}"
    block_end = f"# END nodexo rental {alias}"
    lines = []
    if identity_file:
        lines.append(f"  IdentityFile {identity_file}")
        lines.append(f"  IdentitiesOnly yes")
    if jump_user:
        lines.append(f"  ProxyJump {jump_user}@{ssh_host}")
        lines.append(f"  Hostname 127.0.0.1")
    else:
        lines.append(f"  Hostname {ssh_host}")
    lines.append(f"  Port {ssh_port}")
    lines.append(f"  User {ssh_user}")
    lines.append(f"  StrictHostKeyChecking accept-new")
    lines.append(f"  UserKnownHostsFile /dev/null")
    lines.append(f"  LogLevel ERROR")
    new_block = "\n".join([block_start, f"Host {alias}"] + lines + [block_end])

    existing = cfg_path.read_text() if cfg_path.exists() else ""
    if block_start in existing and block_end in existing:
        # Replace existing block
        pre, _, rest = existing.partition(block_start)
        _, _, post = rest.partition(block_end)
        existing = pre + post
    existing = existing.rstrip() + "\n\n" + new_block + "\n"
    cfg_path.write_text(existing)
    _os.chmod(cfg_path, 0o600)


def cmd_connect(args):
    """SSH into a rental. Recreates the ~/.ssh/config block if missing.

    Works on a different machine than the one that did the rent: queries
    the validator for the rental's ssh host/port, writes a fresh config
    block, then execs ssh. Without this, users hitting `connect` from a
    laptop/CI/jump-box that hadn't seen the original rent saw a cryptic
    'Could not resolve hostname nodexo-...'.
    """
    import os as _os
    import hashlib
    import urllib.request, urllib.error
    import json
    alias = args.rental_id if args.rental_id.startswith("nodexo-") else f"nodexo-{args.rental_id[:12]}"

    if not getattr(args, "validator_direct", False):
        data = _fetch_public_rental(args, args.rental_id.removeprefix("nodexo-"))
        rental = data.get("rental") or {}
        if data.get("status") != "active" or not rental:
            print(f"  Rental is not active: {data.get('status') or data.get('error') or 'unknown'}", file=sys.stderr)
            _exit(1)
        conn = rental.get("connection") or {}
        if not conn.get("ssh_host") or not conn.get("ssh_port"):
            print("  Rental has no SSH connection details yet.", file=sys.stderr)
            _exit(1)
        rid = rental.get("rental_id") or args.rental_id.removeprefix("nodexo-")
        alias = f"nodexo-{rid[:12]}"
        pubkey_path, _pub = _resolve_ssh_pubkey(args.ssh_key)
        privkey = pubkey_path[:-4] if pubkey_path.endswith(".pub") else pubkey_path
        _write_ssh_config_block(
            alias=alias,
            ssh_host=conn["ssh_host"],
            ssh_port=int(conn["ssh_port"]),
            ssh_user=conn.get("ssh_user", "root"),
            jump_user="",
            identity_file=privkey,
        )
        _os.execvp("ssh", ["ssh", alias])

    cfg_path = Path.home() / ".ssh" / "config"
    has_block = False
    if cfg_path.exists():
        has_block = f"Host {alias}\n" in cfg_path.read_text() or \
                    f"Host {alias} " in cfg_path.read_text() or \
                    cfg_path.read_text().rstrip().endswith(f"Host {alias}")

    if not has_block:
        # Block missing — fetch rental from validator and write it.
        print(f"  no ssh-config block for {alias}, fetching…", file=sys.stderr)
        pubkey_path, pub = _resolve_ssh_pubkey(args.ssh_key)
        fp = hashlib.sha256(pub.encode()).hexdigest()[:16]
        url = _validator_url(args) + f"/rentals?renter_pubkey_fp={fp}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                rentals = json.loads(resp.read()).get("rentals", [])
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}: {e.reason}", file=sys.stderr)
            _exit(1)
        # Match by full id or by prefix; the alias's tail is the rental id prefix.
        wanted = alias.removeprefix("nodexo-")
        target = next((r for r in rentals if r["rental_id"].startswith(wanted)), None)
        if target is None:
            print(f"  No active rental matching '{args.rental_id}' for your key. "
                  f"Run `nodexo rentals` to list yours.", file=sys.stderr)
            _exit(1)
        conn = target.get("connection") or {}
        if not conn.get("ssh_port"):
            print(f"  Rental row has no SSH info — likely still provisioning.", file=sys.stderr)
            _exit(1)
        try:
            _write_ssh_config_block(
                alias=alias, ssh_host=conn["ssh_host"], ssh_port=conn["ssh_port"],
                ssh_user=conn.get("ssh_user", "root"),
                jump_user="", identity_file=pubkey_path,
            )
            print(f"  wrote ~/.ssh/config block for {alias}", file=sys.stderr)
        except Exception as e:
            print(f"  Failed to write ~/.ssh/config: {e}", file=sys.stderr)
            _exit(1)

    _os.execvp("ssh", ["ssh", alias])


def cmd_ssh_config(args):
    """Emit the ~/.ssh/config block for a rental to stdout.

    Lets renters copy the block to another machine:
      nodexo ssh-config <id> >> ~/.ssh/config
      ssh user@other 'cat >> ~/.ssh/config' < <(nodexo ssh-config <id>)
    """
    import hashlib
    import urllib.request, urllib.error
    import json

    if not getattr(args, "validator_direct", False):
        data = _fetch_public_rental(args, args.rental_id.removeprefix("nodexo-"))
        rental = data.get("rental") or {}
        if data.get("status") != "active" or not rental:
            print(f"# rental is not active: {data.get('status') or data.get('error') or 'unknown'}", file=sys.stderr)
            _exit(1)
        conn = rental.get("connection") or {}
        if not conn.get("ssh_host") or not conn.get("ssh_port"):
            print("# rental has no SSH connection details yet", file=sys.stderr)
            _exit(1)
        pubkey_path, _pub = _resolve_ssh_pubkey(args.ssh_key)
        rid = rental.get("rental_id") or args.rental_id.removeprefix("nodexo-")
        alias = f"nodexo-{rid[:12]}"
        lines = [
            f"# BEGIN nodexo rental {alias}",
            f"Host {alias}",
            f"  IdentityFile {pubkey_path[:-4] if pubkey_path.endswith('.pub') else pubkey_path}",
            f"  IdentitiesOnly yes",
            f"  Hostname {conn.get('ssh_host', '')}",
            f"  Port {conn.get('ssh_port', 0)}",
            f"  User {conn.get('ssh_user', 'root')}",
            f"  StrictHostKeyChecking accept-new",
            f"  UserKnownHostsFile /dev/null",
            f"  LogLevel ERROR",
            f"# END nodexo rental {alias}",
            "",
        ]
        print("\n".join(lines))
        _exit(0)

    pubkey_path, pub = _resolve_ssh_pubkey(args.ssh_key)
    fp = hashlib.sha256(pub.encode()).hexdigest()[:16]
    url = _validator_url(args) + f"/rentals?renter_pubkey_fp={fp}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            rentals = json.loads(resp.read()).get("rentals", [])
    except Exception as e:
        print(f"# fetch failed: {e}", file=sys.stderr)
        _exit(1)
    target = next((r for r in rentals if r["rental_id"].startswith(args.rental_id)), None)
    if target is None:
        print(f"# no active rental matching '{args.rental_id}'", file=sys.stderr)
        _exit(1)
    rid = target["rental_id"]
    alias = f"nodexo-{rid[:12]}"
    conn = target.get("connection") or {}
    lines = [
        f"# BEGIN nodexo rental {alias}",
        f"Host {alias}",
        f"  IdentityFile {pubkey_path[:-4] if pubkey_path.endswith('.pub') else pubkey_path}",
        f"  IdentitiesOnly yes",
        f"  Hostname {conn.get('ssh_host', '')}",
        f"  Port {conn.get('ssh_port', 0)}",
        f"  User {conn.get('ssh_user', 'root')}",
        f"  StrictHostKeyChecking accept-new",
        f"  UserKnownHostsFile /dev/null",
        f"  LogLevel ERROR",
        f"# END nodexo rental {alias}",
        "",
    ]
    print("\n".join(lines))
    _exit(0)


def cmd_rental_keys(args):
    """Add or remove an SSH key on a live rental's container.

    Goes through the validator (owner-scoped: caller must hold the
    original SSH key whose fingerprint is recorded on the rental).
    Validator forwards a signed request to the miner; the miner pipes
    the pubkey via stdin into the container's authorized_keys (no
    shell interpolation).
    """
    if not getattr(args, "validator_direct", False):
        command = "ssh-key-add" if args.subcommand == "add" else "ssh-key-remove"
        extra = ["--rental-id", args.rental_id, *_public_auth_args(args)]
        if args.subcommand == "remove" and args.key_text:
            extra.extend(["--ssh-pub-key", args.key_text])
        elif args.ssh_key:
            extra.extend(["--ssh-key", args.ssh_key])
        if getattr(args, "json", False):
            extra.append("--json")
        _run_public_cli(args, command, extra)

    import hashlib as _hl
    import json
    import urllib.request
    import urllib.error

    vurl = _validator_url(args)

    # 1) Read the pubkey text we're adding/removing.
    if args.subcommand == "add":
        _, pubkey = _resolve_ssh_pubkey(args.ssh_key)
    else:  # remove
        if args.key_text:
            pubkey = args.key_text.strip()
        else:
            _, pubkey = _resolve_ssh_pubkey(args.ssh_key)

    # 2) Identity for ownership check: fingerprint the renter's ORIGINAL key
    #    (the one they rented with). For `add`, the original key is whoever
    #    they have on disk already (same fingerprint they used at rent time
    #    when they didn't pass --ssh-key). Reuse --ssh-key for that too; if
    #    we just resolved it for the pubkey above, fingerprint that.
    orig_path, orig_pub = _resolve_ssh_pubkey(args.ssh_key)
    fp = _hl.sha256(orig_pub.encode()).hexdigest()[:16]

    # 3) Find the rental.
    list_url = f"{vurl}/rentals?renter_pubkey_fp={fp}"
    try:
        with urllib.request.urlopen(list_url, timeout=10) as resp:
            rentals = json.loads(resp.read()).get("rentals", [])
    except urllib.error.HTTPError as e:
        print(f"  Failed to list rentals: HTTP {e.code}", file=sys.stderr)
        _exit(1)
    target = next((r for r in rentals if r["rental_id"].startswith(args.rental_id)), None)
    if target is None:
        print(f"  No active rental matching '{args.rental_id}' for your key.",
              file=sys.stderr)
        if rentals:
            print(f"  Yours: {[r['rental_id'][:12] for r in rentals]}", file=sys.stderr)
        _exit(1)
    rid = target["rental_id"]

    # 4) Fire the validator endpoint.
    method = "POST" if args.subcommand == "add" else "DELETE"
    url = f"{vurl}/rentals/{rid}/ssh_keys?renter_pubkey_fp={fp}"
    body = json.dumps({"ssh_pub_key": pubkey}).encode()
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"  {_c('OK', 'green', 'bold')}: "
                  f"{'added' if args.subcommand == 'add' else 'removed'} key on rental {rid[:12]}")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            pass
        print(f"  {_c('Failed', 'red', 'bold')}: HTTP {e.code} {detail}",
              file=sys.stderr)
        _exit(1)
    except Exception as e:
        print(f"  {_c('Failed', 'red', 'bold')}: {e}", file=sys.stderr)
        _exit(1)
    _exit(0)


def cmd_rentals(args):
    """List YOUR active rentals (filtered by your SSH-key fingerprint).

    Pass --all to see every rental on this validator (requires
    NODEXO_ADMIN_TOKEN env var). Without --all, the API filters by your
    SSH key fingerprint so renters can only see what they own.
    """
    import json
    import hashlib
    import urllib.request
    base = _validator_url(args) + "/rentals"
    headers = {}
    if getattr(args, "all", False):
        admin = os.environ.get("NODEXO_ADMIN_TOKEN", "")
        if admin:
            headers["X-Admin-Token"] = admin
        url = base  # no fingerprint filter; admin-gated server-side
    else:
        pub = _read_ssh_pubkey(args.ssh_key)
        fp = hashlib.sha256(pub.encode()).hexdigest()[:16]
        url = f"{base}?renter_pubkey_fp={fp}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            detail = ""
        print(f"  HTTP {e.code}: {detail}", file=sys.stderr)
        _exit(1)
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        _exit(1)
    rentals = data.get("rentals", [])
    if not rentals:
        print(f"\n  {_c('No active rentals.', 'dim')}\n")
        _exit(0)
    print(f"\n {_c('Active rentals', 'bold', 'white')}")
    print(f" {_c('━' * 110, 'dim')}")
    cols = [
        ("Rental",      18, "<"),
        ("GPU",         24, "<"),
        ("Elapsed",     10, ">"),
        ("Remaining",   10, ">"),
        ("Spent",       12, ">"),
        ("SSH",         28, "<"),
    ]
    table_rows = []
    for r in rentals:
        rid = r.get("rental_id", "")[:8] + "…"
        gpu = r.get("gpu") or {}
        gpu_str = f"{gpu.get('count', 1)}× {gpu.get('model', '?').replace('NVIDIA ', '')[:18]}"
        elapsed = int(r.get("elapsed_seconds") or 0)
        remaining = int(r.get("remaining_seconds") or 0)
        def fmt_dur(s):
            if s >= 3600:
                return f"{s//3600}h{(s%3600)//60:02d}m"
            if s >= 60:
                return f"{s//60}m{s%60:02d}s"
            return f"{s}s"
        spend = (r.get("price") or {}).get("spend_so_far_rao") or 0
        spend_str = f"{spend} RAO" if spend else _c("—", "dim")
        ssh = ""
        if r.get("executor_endpoint"):
            host = r["executor_endpoint"].replace("https://", "").replace("http://", "").split(":")[0]
            ssh = f"ssh nodexo-{r['rental_id'][:8]}"
        table_rows.append([rid, gpu_str, fmt_dur(elapsed), fmt_dur(remaining), spend_str, ssh])
    _print_table(table_rows, cols)
    print()
    _exit(0)


def cmd_inventory(args):
    """Static public inventory list.

    The normal path uses the web app public API. Pass --validator-direct for
    admin/operator debugging against validator internals.
    """
    if not getattr(args, "validator_direct", False):
        if getattr(args, "raw", False):
            print("  --raw is validator-internal. Re-run with --validator-direct and NODEXO_ADMIN_TOKEN.", file=sys.stderr)
            _exit(2)
        _run_public_cli(args, "inventory", ["--json"] if getattr(args, "json", False) else [])

    import json
    import urllib.request

    if getattr(args, "raw", False):
        return _cmd_inventory_raw(args)

    url = _validator_url(args) + "/marketplace/tiers"
    try:
        req = urllib.request.Request(url, headers=_validator_request_headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        _exit(1)
    tiers = data.get("tiers", [])
    if not tiers:
        print(f"\n  {_c('No GPUs available right now.', 'dim')}\n")
        _exit(0)

    print()
    print(f" {_c('▎', 'cyan', 'bold')} {_c('NODEXO  —  Marketplace', 'bold', 'white')}")
    print(f" {_c('━' * 96, 'dim')}")
    cols = [
        ("GPU",            26, "<"),    # e.g. "4× RTX A6000"
        ("VRAM total",      12, ">"),    # e.g. "192 GB (4×48)"
        ("CPU",             8, ">"),
        ("RAM",             8, ">"),
        ("Disk",            8, ">"),
        ("Avail",           6, ">"),
        ("Attest",         8, "<"),
        ("Reliability",   18, "<"),
        ("Uptime",          7, ">"),
        ("Price/h",        12, ">"),
    ]
    rows = []
    for t in tiers:
        name = t["gpu_model"].replace("NVIDIA ", "")
        count = t.get("gpu_count", 1)
        gpu_label = f"{count}× {name}"
        # VRAM column shows total + per-GPU when count > 1
        if count > 1:
            vram_label = f"{count * t['vram_gb']} GB ({count}×{t['vram_gb']})"
        else:
            vram_label = f"{t['vram_gb']} GB"
        avail = t["available"]
        attest = t.get("attestation", "ZkGEMM")
        rel = t.get("reliability_24h")
        samples = t.get("reliability_samples", 0)
        rel_str = _reliability_str(rel, samples)
        up = t.get("uptime_days")
        if up is None:
            up_str = _c("—", "dim")
        elif up >= 30:
            up_str = f"{int(up)}d"
        else:
            up_str = f"{up:.1f}d"
        price = t.get("min_price_rao")
        price_str = f"{price} RAO" if price else _c("free*", "dim")
        cpu_s = t.get("cpu_cores") or _c("—", "dim")
        ram_s = (f"{t['ram_gb']} GB" if t.get("ram_gb") else _c("—", "dim"))
        disk_s = (f"{t['disk_gb']} GB" if t.get("disk_gb") else _c("—", "dim"))
        rows.append([
            gpu_label[:26],
            vram_label,
            cpu_s,
            ram_s,
            disk_s,
            _c(str(avail), "green" if avail > 0 else "dim", "bold"),
            _c(attest, "cyan"),
            rel_str,
            up_str,
            price_str,
        ])
    _print_table(rows, cols)
    total = sum(t["available"] * t["gpu_count"] for t in tiers)
    print()
    print(f"  {_c(f'{total} GPU(s) available across {len(tiers)} tier(s).', 'green', 'bold')}")
    # Generate examples from the actual tiers in front of the user so the
    # arbitrary-naming confusion goes away ("am I supposed to type A6000
    # or RTX A6000?"). Pick the first tier as the example baseline.
    if tiers:
        first = tiers[0]
        # Use the readable model name minus the NVIDIA prefix as the
        # short form the renter passes. We accept substring match
        # server-side so any unambiguous fragment works.
        short = first["gpu_model"].replace("NVIDIA ", "").split()[-1]  # 'A6000', '4090', 'H100', …
        cnt = first.get("gpu_count", 1)
        print(f"  Rent: {_c(f'nodexo rent --gpu {short} --duration 1h', 'cyan')}")
        if cnt > 1:
            print(f"        {_c(f'(this tier ships {cnt}× per executor — pass --gpu-count {cnt} to ask for it)', 'dim')}")
        else:
            print(f"        {_c('For multi-GPU configs (when listed), add --gpu-count N', 'dim')}")
    print(f"  Glossary: {_c('nodexo inventory --help', 'dim')}")
    print()
    _exit(0)


def _cmd_inventory_raw(args):
    """Per-executor validator view for admin/operator debugging."""
    import json
    import urllib.request
    url = _validator_url(args) + "/executors"
    try:
        req = urllib.request.Request(url, headers=_validator_request_headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        _exit(1)
    execs = data.get("executors", [])
    if not execs:
        print(f"\n  {_c('No executors available.', 'dim')}\n")
        _exit(0)
    print()
    print(f" {_c('▎', 'yellow', 'bold')} {_c('NODEXO  —  Raw executor list (ops view)', 'bold', 'white')}")
    print(f" {_c('━' * 96, 'dim')}")
    cols = [
        ("Executor",   18, "<"),
        ("GPU model hash", 18, "<"),
        ("Cnt",         4, ">"),
        ("VRAM",        8, ">"),
        ("Rented",      8, "<"),
        ("Score",       8, ">"),
        ("Endpoint",   28, "<"),
    ]
    rows = []
    for ex in execs:
        eid = ex["executor_id"][:16] + "…"
        gpu_hash = ex["gpu_model_hash"][:16] + "…"
        rented = _c("YES", "magenta", "bold") if ex["is_rented"] else "no"
        sc = ex.get("score") or {}
        comp = sc.get("composite", 0)
        ep = (ex.get("endpoint") or "—").replace("https://", "").replace("http://", "").split("/")[0]
        rows.append([eid, gpu_hash, str(ex["gpu_count"]), f"{ex['vram_mb']//1024}GB",
                     rented, f"{comp:.2f}", ep[:28]])
    _print_table(rows, cols)
    print()
    _exit(0)


def cmd_rental_info(args):
    """Detail view of a rental: validator-side state + live miner health."""
    if not getattr(args, "validator_direct", False):
        extra = ["--rental-id", args.rental_id, *_public_auth_args(args)]
        if getattr(args, "json", False):
            extra.append("--json")
        _run_public_cli(args, "get", extra)

    import json
    import urllib.request
    vurl = _validator_url(args)
    # Validator state
    req = urllib.request.Request(f"{vurl}/rentals", headers=_validator_request_headers())
    with urllib.request.urlopen(req, timeout=10) as resp:
        rentals = json.loads(resp.read()).get("rentals", [])
    target = next((r for r in rentals if r["rental_id"].startswith(args.rental_id)), None)
    if target is None:
        print(f"  No active rental matching '{args.rental_id}'.", file=sys.stderr)
        if rentals:
            print(f"  Active IDs: {[r['rental_id'] for r in rentals]}", file=sys.stderr)
        _exit(1)

    rid = target["rental_id"]
    eid = target["executor_id"]
    ep = target["executor_endpoint"]
    container = target["container_name"]
    created = target["created_at"]
    ttl = target.get("ttl_seconds", 0)
    age = int(time.time() - created)
    remaining = max(0, ttl - age)

    print()
    print(f" {_c('▎', 'magenta', 'bold')} {_c(f'Rental {rid}', 'bold', 'white')}")
    print(f" {_c('━' * 80, 'dim')}")
    print(f"   {_c('Executor', 'dim'):<22} {eid}")
    print(f"   {_c('Endpoint', 'dim'):<22} {ep}")
    print(f"   {_c('Container', 'dim'):<22} {container}")
    print(f"   {_c('Age', 'dim'):<22} {_humanize_age(age)}")
    print(f"   {_c('TTL remaining', 'dim'):<22} {remaining}s")

    # Live miner info
    try:
        with urllib.request.urlopen(f"{ep}/health", timeout=4) as r:
            print(f"   {_c('Miner /health', 'dim'):<22} {_c('ok', 'green')} ({r.read().decode().strip()[:60]})")
    except Exception as e:
        print(f"   {_c('Miner /health', 'dim'):<22} {_c('unreachable', 'red')} ({e})")
    try:
        with urllib.request.urlopen(f"{ep}/hardware", timeout=4) as r:
            hw = json.loads(r.read())
        gpus = hw.get("gpus", [])
        if gpus:
            print(f"   {_c('GPU', 'dim'):<22} {gpus[0].get('name', '?')} ({gpus[0].get('vram_mb', 0)} MB)")
        utils = hw.get("gpu_utilization") or []
        if utils:
            u = utils[0]
            print(f"   {_c('Live util', 'dim'):<22} "
                  f"GPU {u.get('gpu_util_pct', 0)}%  "
                  f"VRAM {u.get('mem_used_mb', 0)}/{u.get('mem_total_mb', 0)} MB")
    except Exception:
        pass

    # SSH commands — show both the local alias and a portable direct
    # form. The alias depends on ~/.ssh/config; if the renter is on
    # another machine, or the config gets wiped, the direct form is the
    # copy-pasteable fallback. Same shape any web dashboard would
    # render to a user (the /rentals API returns connection.ssh_host /
    # ssh_port / ssh_user verbatim).
    short = rid[:12]
    alias = f"nodexo-{short}"
    conn = target.get("connection") or {}
    ssh_host = conn.get("ssh_host", "")
    ssh_port = conn.get("ssh_port", 0)
    ssh_user = conn.get("ssh_user", "root")
    print()
    print(f"  {_c('Connect (alias)', 'bold', 'white'):<32} {_c(f'ssh {alias}', 'cyan', 'bold')}")
    if ssh_host and ssh_port:
        direct = f"ssh -p {ssh_port} {ssh_user}@{ssh_host}"
        print(f"  {_c('Connect (direct)', 'bold', 'white'):<32} {_c(direct, 'cyan')}")
        print(f"  {_c('Add a key', 'dim'):<32} nodexo rental-key add {rid[:12]}")
        print(f"  {_c('Remove a key', 'dim'):<32} nodexo rental-key remove {rid[:12]} --key-text \"...\"")
    print(f"  {_c('Terminate', 'bold', 'white'):<32} {_c(f'nodexo rental-end {rid}', 'cyan')}\n")
    _exit(0)


def cmd_history(args):
    """Show historical rentals (terminated + active). Filter by --mine for
    rentals matching your default SSH key fingerprint.
    """
    import json
    import urllib.request
    import hashlib
    url = _validator_url(args) + "/rentals/history"
    qs = []
    if args.limit:
        qs.append(f"limit={int(args.limit)}")
    if args.since:
        # Accept either a unix timestamp or '24h' / '7d' relative
        s = args.since.strip().lower()
        if s.endswith("h"):
            since_ts = time.time() - float(s[:-1]) * 3600
        elif s.endswith("d"):
            since_ts = time.time() - float(s[:-1]) * 86400
        else:
            since_ts = float(s)
        qs.append(f"since_ts={since_ts}")
    if args.mine:
        pub = _read_ssh_pubkey(args.ssh_key)
        fp = hashlib.sha256(pub.encode()).hexdigest()[:16]
        qs.append(f"renter_pubkey_fp={fp}")
    if qs:
        url += "?" + "&".join(qs)

    try:
        req = urllib.request.Request(url, headers=_validator_request_headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        _exit(1)

    rentals = data.get("rentals", [])
    if not rentals:
        print(f"\n  {_c('No rental history found.', 'dim')}\n")
        _exit(0)

    print()
    print(f" {_c('▎', 'cyan', 'bold')} {_c('Rental history', 'bold', 'white')}")
    print(f" {_c('━' * 110, 'dim')}")
    cols = [
        ("Rental",      10, "<"),
        ("Started",     20, "<"),
        ("Status",      18, "<"),
        ("GPU",         22, "<"),
        ("Duration",    10, ">"),
        ("Cost (RAO)",  12, ">"),
        ("Renter fp",   10, "<"),
    ]
    rows = []
    for r in rentals:
        rid = r.get("rental_id", "")[:8] + "…"
        created = (r.get("created_at") or "")[:19].replace("T", " ")
        status = r.get("status", "?")
        status_color = ("green" if status == "active" else
                        ("dim" if status.startswith("terminated_user") else "yellow"))
        gpu = r.get("gpu_model", "?").replace("NVIDIA ", "")[:22]
        secs = int(r.get("actual_seconds") or 0)
        if secs >= 3600:
            dur = f"{secs//3600}h{(secs % 3600)//60:02d}m"
        elif secs >= 60:
            dur = f"{secs//60}m{secs%60:02d}s"
        elif secs > 0:
            dur = f"{secs}s"
        else:
            dur = _c("active", "green")
        cost = r.get("total_paid_rao") or 0
        cost_str = f"{cost}" if cost else "—"
        fp = (r.get("renter_pubkey_fp") or "")[:8]
        rows.append([rid, created, _c(status, status_color), gpu, dur, cost_str, fp])
    _print_table(rows, cols)
    print()
    _exit(0)


def cmd_marketplace(args):
    """Interactive renter dashboard — like `status` for miners but for renters.

    Single screen with: marketplace tiers + your active rentals.
    Keys: r=rent, 0-9=open rental, d=end rental, R=refresh, h=help, q=quit.
    Drops out of cbreak for multi-char prompts (duration, image, etc.).
    """
    if not getattr(args, "validator_direct", False):
        _run_public_cli(args, "marketplace", ["--json"] if getattr(args, "json", False) else [])

    import json
    import hashlib
    import urllib.request

    vurl = _validator_url(args)
    pub = _read_ssh_pubkey(args.ssh_key)
    fp = hashlib.sha256(pub.encode()).hexdigest()[:16]

    def _load():
        """Pull tiers + my active rentals in parallel. Returns dicts."""
        async def _both():
            import aiohttp
            async with aiohttp.ClientSession() as s:
                t_task = _fetch_json(s, f"{vurl}/marketplace/tiers", timeout=4)
                r_task = _fetch_json(s, f"{vurl}/rentals?renter_pubkey_fp={fp}", timeout=4)
                t, r = await asyncio.gather(t_task, r_task, return_exceptions=True)
                return (
                    (t or {}).get("tiers", []) if isinstance(t, dict) else [],
                    (r or {}).get("rentals", []) if isinstance(r, dict) else [],
                )
        return asyncio.run(_both())

    def _render_marketplace(tiers, rentals):
        _clear()
        print()
        print(f" {_c('▎', 'cyan', 'bold')} {_c('NODEXO — Renter Dashboard', 'bold', 'white')}")
        print(f" {_c('━' * 96, 'dim')}")

        # ── Tier table (numbered for selection) ─────────────────
        print(f"\n {_c('MARKETPLACE', 'bold', 'white')}")
        if not tiers:
            print(f"   {_c('No GPUs available right now.', 'dim')}")
        else:
            cols = [
                ("#",            3, ">"),
                ("GPU",         24, "<"),
                ("VRAM",        14, ">"),
                ("CPU",          5, ">"),
                ("RAM",          7, ">"),
                ("Avail",        6, ">"),
                ("Attest",       8, "<"),
                ("Reliability", 18, "<"),
                ("Price/h",     12, ">"),
            ]
            rows = []
            for i, t in enumerate(tiers):
                name = t["gpu_model"].replace("NVIDIA ", "")
                cnt = t.get("gpu_count", 1)
                gpu_label = f"{cnt}× {name}"[:24]
                vram = f"{cnt * t['vram_gb']} GB"
                if cnt > 1:
                    vram = f"{cnt * t['vram_gb']} ({cnt}×{t['vram_gb']})"
                rel = t.get("reliability_24h")
                samples = t.get("reliability_samples", 0)
                rel_str = _reliability_str(rel, samples)
                price = t.get("min_price_rao")
                price_str = f"{price} RAO" if price else _c("free*", "dim")
                rows.append([
                    _c(str(i), "cyan", "bold"),
                    gpu_label, vram,
                    str(t.get("cpu_cores") or "—"),
                    f"{t['ram_gb']}GB" if t.get("ram_gb") else "—",
                    _c(str(t["available"]), "green" if t["available"] else "dim", "bold"),
                    _c(t.get("attestation", "ZkGEMM"), "cyan"),
                    rel_str,
                    price_str,
                ])
            _print_table(rows, cols)

        # ── My rentals ───────────────────────────────────────────
        print(f"\n {_c('YOUR ACTIVE RENTALS', 'bold', 'white')}")
        if not rentals:
            print(f"   {_c('You have no active rentals.', 'dim')}")
        else:
            cols = [
                ("#",          3, ">"),
                ("Rental",    10, "<"),
                ("GPU",       22, "<"),
                ("Elapsed",   10, ">"),
                ("Remaining", 12, ">"),
                ("Spent",     12, ">"),
                ("SSH",       26, "<"),
            ]
            rows = []
            for i, r in enumerate(rentals):
                gpu = r.get("gpu") or {}
                rid = r.get("rental_id", "")[:8] + "…"
                gpu_str = f"{gpu.get('count', 1)}× {gpu.get('model', '?').replace('NVIDIA ', '')[:18]}"
                el = int(r.get("elapsed_seconds") or 0)
                ttl = int(r.get("ttl_seconds") or 0)
                rem = int(r.get("remaining_seconds") or 0)
                def fmt(s):
                    if s >= 3600:
                        return f"{s//3600}h{(s%3600)//60:02d}m"
                    if s >= 60:
                        return f"{s//60}m{s%60:02d}s"
                    return f"{s}s"
                rem_str = _c("∞", "yellow") if ttl == 0 else fmt(rem)
                spend = (r.get("price") or {}).get("spend_so_far_rao") or 0
                spend_str = f"{spend} RAO" if spend else _c("—", "dim")
                rows.append([
                    _c(str(i), "magenta", "bold"),
                    rid, gpu_str,
                    fmt(el), rem_str, spend_str,
                    f"ssh nodexo-{r['rental_id'][:8]}",
                ])
            _print_table(rows, cols)

        # ── Footer ──────────────────────────────────────────────
        # 0-9 is the fast path for the first ten rentals; `s` opens a
        # line-input prompt for any index (works for 10+ rentals).
        print()
        sel_hint = f"{_c('[0-9]', 'cyan')} rental detail"
        if len(rentals) > 10:
            sel_hint = f"{_c('[0-9 / s]', 'cyan')} rental detail (s = pick by #)"
        print(f"  {_c('[r]', 'cyan', 'bold')} rent    "
              f"{sel_hint}    "
              f"{_c('[R]', 'cyan')} refresh    "
              f"{_c('[h]', 'cyan')} help    "
              f"{_c('[q]', 'cyan')} quit", flush=True)

    def _render_help():
        _clear()
        print()
        print(f" {_c('▎', 'cyan', 'bold')} {_c('Help — Renter Dashboard', 'bold', 'white')}")
        print(f" {_c('━' * 80, 'dim')}")
        print(f"\n {_c('KEYS', 'bold', 'white')}")
        keys = [
            ("r",       "Start a new rental — pick a GPU, duration, image, confirm"),
            ("0-9",     "Open detail for rentals 0-9 (single keypress)"),
            ("s",       "Open detail by number — type any index then Enter (10+ rentals)"),
            ("R",       "Force-refresh marketplace + your rentals"),
            ("h or ?",  "This help"),
            ("q",       "Quit"),
        ]
        for k, v in keys:
            print(f"   {_c(k, 'cyan'):<22} {v}")
        print(f"\n {_c('COLUMN GLOSSARY', 'bold', 'white')}")
        for k, v in [
            ("GPU",         "model + count per executor; multi-GPU configs are their own tier"),
            ("VRAM",        "total VRAM for one rental in this tier"),
            ("Reliability", "24h expected proof-check pass rate; 'high' = ≥100 checks"),
            ("Attest",      "ZkGEMM = cryptographic proof (today). TEE coming."),
            ("Price/h",     "miner-set RAO per GPU-hour. 'free*' on testnet."),
            ("Remaining",   "TTL minus elapsed; ∞ for open-ended rentals"),
            ("Spent",       "accrued cost so far (price × elapsed)"),
        ]:
            print(f"   {_c(k, 'dim'):<22} {v}")
        print(f"\n  {_c('[any key] back', 'dim')}")

    def _do_rent(tiers):
        """Multi-prompt rent flow. Drops out of cbreak for input."""
        if not tiers:
            print(f"\n  {_c('No GPUs available to rent right now.', 'yellow')}")
            input("  press enter to continue ")
            return None
        print()
        try:
            sel = input(f"  {_c('Choose GPU #', 'cyan', 'bold')} [0-{len(tiers)-1}]: ").strip()
            idx = int(sel)
            if not (0 <= idx < len(tiers)):
                raise ValueError
        except (ValueError, EOFError, KeyboardInterrupt):
            print(f"  {_c('Aborted.', 'dim')}")
            input("  press enter to continue ")
            return None
        tier = tiers[idx]
        gpu_short = tier["gpu_model"].replace("NVIDIA ", "").split()[-1]
        gpu_count = tier.get("gpu_count", 1)
        pretty = tier["gpu_model"].replace("NVIDIA ", "")
        print(f"  {_c('Picked:', 'dim')} {gpu_count}× {pretty}")
        try:
            duration = input(f"  {_c('Duration', 'cyan')} [unlimited / 30m / 2h / 1d]: ").strip() or "unlimited"
            image = input(f"  {_c('Image', 'cyan')} [ubuntu:22.04]: ").strip() or "ubuntu:22.04"
            confirm = input(f"  {_c('Confirm rent? [Y/n]:', 'green', 'bold')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"  {_c('Aborted.', 'dim')}")
            input("  press enter to continue ")
            return None
        if confirm in ("n", "no"):
            print(f"  {_c('Cancelled.', 'dim')}")
            input("  press enter to continue ")
            return None

        # Submit
        body = {
            "gpu_model": gpu_short,
            "gpu_count": gpu_count,
            "duration": duration,
            "ssh_pub_key": pub,
            "image": image,
        }
        req = urllib.request.Request(
            f"{vurl}/rent", data=json.dumps(body).encode(),
            method="POST", headers={"Content-Type": "application/json"},
        )
        try:
            with _Spinner(f"Requesting rental ({gpu_count}× {gpu_short}, {duration})"):
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail", "")
            except Exception:
                detail = e.reason or ""
            print(f"  {_c('Failed:', 'red', 'bold')} HTTP {e.code} {detail}")
            input("  press enter to continue ")
            return None
        except Exception as e:
            print(f"  {_c('Failed:', 'red', 'bold')} {e}")
            input("  press enter to continue ")
            return None

        rid = data["rental_id"]
        # Auto-write ssh-config
        conn = data["connection"]
        privkey = args.ssh_key[:-4] if args.ssh_key.endswith(".pub") else (args.ssh_key or "")
        alias = f"nodexo-{rid[:12]}"
        try:
            _write_ssh_config_block(
                alias=alias, ssh_host=conn["ssh_host"], ssh_port=conn["ssh_port"],
                ssh_user=conn["ssh_user"], jump_user="", identity_file=privkey,
            )
        except Exception:
            pass
        print(f"\n  {_c('✓ Rented!', 'green', 'bold')}  "
              f"Connect: {_c(f'ssh {alias}', 'cyan', 'bold')}")
        input("  press enter to continue ")
        return rid

    def _fetch_executor_live(endpoint: str) -> dict:
        """Hit the miner's /hardware to pull live GPU stats — same fields
        the operator's fleet dashboard renders. Public read-only endpoint,
        same as cmd_rental_info uses. Short timeout so a slow miner
        doesn't freeze the detail loop.
        """
        if not endpoint:
            return {}
        try:
            with urllib.request.urlopen(f"{endpoint.rstrip('/')}/hardware",
                                        timeout=3) as resp:
                return json.loads(resp.read())
        except Exception:
            return {}

    def _render_rental_detail(r, live):
        rid = r.get("rental_id", "")
        _clear()
        print()
        print(f" {_c('▎', 'magenta', 'bold')} {_c(f'Rental {rid[:16]}', 'bold', 'white')}")
        print(f" {_c('━' * 80, 'dim')}")
        gpu = r.get("gpu") or {}
        print(f"   {_c('GPU', 'dim'):<22} {gpu.get('count', 1)}× {gpu.get('model', '?').replace('NVIDIA ', '')}  ({gpu.get('vram_gb', '?')} GB)")
        print(f"   {_c('Executor', 'dim'):<22} {r.get('executor_id', '')}")
        print(f"   {_c('Container', 'dim'):<22} {r.get('container_name', '—')}")
        # Both forms — alias depends on ~/.ssh/config; direct is portable.
        conn = r.get("connection") or {}
        s_host = conn.get("ssh_host", "")
        s_port = conn.get("ssh_port", 0)
        s_user = conn.get("ssh_user", "root")
        print(f"   {_c('SSH (alias)', 'dim'):<22} ssh nodexo-{rid[:12]}")
        if s_host and s_port:
            print(f"   {_c('SSH (direct)', 'dim'):<22} ssh -p {s_port} {s_user}@{s_host}")

        # ── Live GPU block (same fields the miner's fleet dashboard sees) ──
        utils = (live or {}).get("gpu_utilization") or []
        if utils:
            print()
            for u in utils:
                idx = u.get("index", 0)
                util_pct = u.get("gpu_util_pct", 0)
                mem_used = u.get("mem_used_mb", 0)
                mem_total = u.get("mem_total_mb", 0)
                temp = u.get("temp_c", 0)
                power = u.get("power_w", 0)
                bar_len = int(util_pct / 5) if util_pct else 0
                bar = _c("█" * bar_len, "cyan") + _c("░" * (20 - bar_len), "dim")
                label = f"GPU{idx}" if len(utils) > 1 else "GPU live"
                print(f"   {_c(label, 'dim'):<22} {bar} {util_pct:>3.0f}%   "
                      f"VRAM {mem_used:>5}/{mem_total} MB   "
                      f"{temp:>2}°C / {power:>4.0f} W")
        elif s_host:
            # Miner unreachable / no GPUs reported — surface this so the
            # renter knows live stats are stale, not zero.
            print(f"\n   {_c('GPU live', 'dim'):<22} {_c('miner unreachable', 'yellow')}")

        ttl = int(r.get("ttl_seconds") or 0)
        el = int(r.get("elapsed_seconds") or 0)
        rem = int(r.get("remaining_seconds") or 0)
        ttl_str = "open-ended (no TTL)" if ttl == 0 else f"{ttl}s ({ttl // 60}m)"
        print()
        print(f"   {_c('TTL', 'dim'):<22} {ttl_str}")
        print(f"   {_c('Elapsed', 'dim'):<22} {el}s")
        print(f"   {_c('Remaining', 'dim'):<22} {'∞' if ttl == 0 else f'{rem}s'}")
        price = r.get("price") or {}
        print(f"   {_c('Price', 'dim'):<22} {price.get('per_gpu_hour_rao', 0)} RAO/GPU/h")
        print(f"   {_c('Spent', 'dim'):<22} {price.get('spend_so_far_rao', 0)} RAO")

        print()
        print(f"  {_c('[a]', 'cyan')} add key  "
              f"{_c('[r]', 'cyan')} remove key  "
              f"{_c('[d]', 'red')} terminate  "
              f"{_c('[u]', 'cyan')} refresh  "
              f"{_c('[b]', 'cyan')} back  "
              f"{_c('[q]', 'cyan')} quit", flush=True)

    def _prompt_pubkey_path(default: str = "~/.ssh/id_ed25519.pub") -> str | None:
        """Inline prompt for an SSH pubkey path. Empty = default."""
        try:
            v = input(f"  {_c('Pubkey path', 'cyan')} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return v or default

    def _do_key_add(r):
        rid = r["rental_id"]
        path = _prompt_pubkey_path()
        if path is None:
            print(f"  {_c('Aborted.', 'dim')}")
            return
        import os as _os
        path = _os.path.expanduser(path)
        try:
            with open(path) as f:
                pubkey = f.read().strip()
        except Exception as e:
            print(f"  {_c('Read failed:', 'red')} {e}")
            input("  press enter to continue ")
            return
        url = f"{vurl}/rentals/{rid}/ssh_keys?renter_pubkey_fp={fp}"
        body = json.dumps({"ssh_pub_key": pubkey}).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with _Spinner("Adding SSH key"):
                urllib.request.urlopen(req, timeout=30)
            print(f"  {_c('✓ Added.', 'green', 'bold')}")
        except urllib.error.HTTPError as e:
            print(f"  {_c('Failed', 'red')}: HTTP {e.code}")
        except Exception as e:
            print(f"  {_c('Failed', 'red')}: {e}")
        input("  press enter to continue ")

    def _do_key_remove(r):
        rid = r["rental_id"]
        print(f"  Paste the full pubkey line to remove (or empty for "
              f"~/.ssh/id_ed25519.pub):")
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"  {_c('Aborted.', 'dim')}")
            return
        if not line:
            import os as _os
            path = _os.path.expanduser("~/.ssh/id_ed25519.pub")
            try:
                with open(path) as f:
                    line = f.read().strip()
            except Exception as e:
                print(f"  {_c('Read failed:', 'red')} {e}")
                input("  press enter to continue ")
                return
        url = f"{vurl}/rentals/{rid}/ssh_keys?renter_pubkey_fp={fp}"
        body = json.dumps({"ssh_pub_key": line}).encode()
        req = urllib.request.Request(
            url, data=body, method="DELETE",
            headers={"Content-Type": "application/json"},
        )
        try:
            with _Spinner("Removing SSH key"):
                urllib.request.urlopen(req, timeout=30)
            print(f"  {_c('✓ Removed.', 'green', 'bold')}")
        except urllib.error.HTTPError as e:
            print(f"  {_c('Failed', 'red')}: HTTP {e.code}")
        except Exception as e:
            print(f"  {_c('Failed', 'red')}: {e}")
        input("  press enter to continue ")

    def _do_rental_detail(rentals, idx):
        if not (0 <= idx < len(rentals)):
            return
        r = rentals[idx]
        rid = r.get("rental_id", "")
        ep = r.get("executor_endpoint", "")
        # Initial fetch — synchronous so the first render has stats.
        live = _fetch_executor_live(ep)
        while True:
            _render_rental_detail(r, live)
            # Short refresh window — live stats update every 5s. The
            # outer marketplace loop uses 30s; here we tune tighter
            # because GPU util is the live signal renters watch.
            k = _wait_for_key(5)
            if k is None:
                # timeout → just refresh live stats and re-render
                live = _fetch_executor_live(ep)
                continue
            kl = k.lower()
            if kl in ("b", "\x1b"):
                return
            if kl == "q":
                _exit(0)
            if kl == "u":
                live = _fetch_executor_live(ep)
                continue
            if kl == "a":
                _do_key_add(r)
                continue
            if kl == "r":
                _do_key_remove(r)
                continue
            if kl == "d":
                print(f"\n  {_c('Terminate', 'red', 'bold')} rental {rid[:12]}? "
                      f"type {_c('yes', 'red', 'bold')}: ", end="", flush=True)
                try:
                    confirm = input().strip()
                except (EOFError, KeyboardInterrupt):
                    confirm = ""
                if confirm == "yes":
                    try:
                        url = f"{vurl}/rentals/{rid}?renter_pubkey_fp={fp}"
                        with _Spinner("Terminating rental"):
                            urllib.request.urlopen(
                                urllib.request.Request(url, method="DELETE"),
                                timeout=90,
                            )
                        print(f"  {_c('Terminated.', 'green')}")
                    except urllib.error.HTTPError as e:
                        print(f"  {_c('Failed', 'red')}: HTTP {e.code}")
                    except Exception as e:
                        print(f"  {_c('Failed', 'red')}: {e}")
                    input("  press enter to continue ")
                    return
                else:
                    print(f"  {_c('Aborted.', 'dim')}")

    # ── Main loop ────────────────────────────────────────────────
    tiers, rentals = _load()
    while True:
        _render_marketplace(tiers, rentals)
        key = _wait_for_key(30)  # auto-refresh every 30s
        if key is None:
            tiers, rentals = _load()
            continue
        k = key.lower()
        if k in ("q", "\x03"):
            print()
            _exit(0)
        if k in ("h", "?"):
            _render_help()
            _wait_for_key(300)
            continue
        if key == "R":  # capital — case-sensitive refresh
            tiers, rentals = _load()
            continue
        if k == "r":
            _do_rent(tiers)
            tiers, rentals = _load()
            continue
        if k == "s" and rentals:
            # Line-input fallback for >10 rentals.
            print(f"\n  {_c(f'Rental #', 'cyan', 'bold')} [0-{len(rentals)-1}]: ", end="", flush=True)
            try:
                sel = input().strip()
                idx = int(sel)
            except (ValueError, EOFError, KeyboardInterrupt):
                continue
            if 0 <= idx < len(rentals):
                _do_rental_detail(rentals, idx)
                tiers, rentals = _load()
            continue
        if k.isdigit() and rentals:
            idx = int(k)
            if 0 <= idx < len(rentals):
                _do_rental_detail(rentals, idx)
                tiers, rentals = _load()
                continue


def cmd_rental_end(args):
    """Terminate a rental by id (proves ownership via SSH-key fingerprint)."""
    if not getattr(args, "validator_direct", False):
        extra = ["--rental-id", args.rental_id, *_public_auth_args(args)]
        if getattr(args, "json", False):
            extra.append("--json")
        _run_public_cli(args, "end", extra)

    import json
    import hashlib
    import urllib.request
    # Compute renter fingerprint from local SSH key so the API can verify
    # we're the same person who created the rental.
    pub = _read_ssh_pubkey(args.ssh_key)
    fp = hashlib.sha256(pub.encode()).hexdigest()[:16]
    url = _validator_url(args) + f"/rentals/{args.rental_id}?renter_pubkey_fp={fp}"
    req = urllib.request.Request(url, method="DELETE")
    if os.environ.get("NODEXO_ADMIN_TOKEN"):
        req.add_header("X-Admin-Token", os.environ["NODEXO_ADMIN_TOKEN"])
    try:
        with _Spinner("Terminating rental (destroying container + chain markAvailable)"):
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            detail = ""
        print(f"  HTTP {e.code}: {detail}", file=sys.stderr)
        _exit(1)
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        _exit(1)
    print(f"\n  {_c('Terminated', 'green')}: {data}\n")
    _exit(0)


def cmd_rental_extend(args):
    """Extend an active x402 rental via the public API."""
    extra = [
        "--rental-id", args.rental_id,
        "--hours", str(int(args.hours)),
        *_public_auth_args(args),
    ]
    if getattr(args, "max_usdc", ""):
        extra.extend(["--max-usdc", args.max_usdc])
    if getattr(args, "private_key_env", ""):
        extra.extend(["--private-key-env", args.private_key_env])
    if getattr(args, "json", False):
        extra.append("--json")
    _run_public_cli(args, "extend", extra)


def cmd_fund(args):
    """Fund EVM wallet from coldkey."""
    from common.chain.wallet import load_hotkey_seed, derive_evm_account, get_ss58_mirror_address
    import bittensor as bt

    args.wallet, args.hotkey = _select_wallet(args.wallet, args.hotkey, args.subtensor_network)
    hotkey_seed = load_hotkey_seed(args.wallet, args.hotkey)
    evm_account = derive_evm_account(hotkey_seed)
    mirror = get_ss58_mirror_address(evm_account.address)

    print(f"\n  EVM address: {evm_account.address}")
    print(f"  SS58 mirror: {mirror}")
    print(f"  Amount:      {args.amount} TAO")

    if not args.yes:
        confirm = input("\n  Send TAO? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("  Aborted.\n")
            return

    WalletCls = getattr(bt, "Wallet", None) or bt.wallet
    SubCls = getattr(bt, "Subtensor", None) or bt.subtensor
    wallet = WalletCls(name=args.wallet)
    subtensor = SubCls(network=args.subtensor_endpoint or args.subtensor_network)
    import inspect

    transfer_kwargs = {
        "wallet": wallet,
        "amount": bt.Balance.from_tao(float(args.amount)),
    }
    params = inspect.signature(subtensor.transfer).parameters
    if "destination_ss58" in params:
        transfer_kwargs["destination_ss58"] = mirror
    else:
        transfer_kwargs["dest"] = mirror
    result = subtensor.transfer(**transfer_kwargs)
    success_value = getattr(result, "is_success", None)
    if callable(success_value):
        success_value = success_value()
    if success_value is None:
        success_value = result
    success = bool(success_value)
    print(f"  {'Success!' if success else 'Failed.'}")
    tx_hash = getattr(result, "extrinsic_hash", None)
    if callable(tx_hash):
        tx_hash = tx_hash()
    if tx_hash:
        print(f"  Tx: {tx_hash}")
    if not success:
        err = (
            getattr(result, "error_message", None)
            or getattr(result, "message", None)
            or getattr(result, "error", None)
        )
        if callable(err):
            err = err()
        if err:
            print(f"  Error: {err}")
        _exit(1)
    print()


# ── Entry point ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="nodexo",
        description="Nodexo CLI — public rentals, agent workflows, and miner fleet operations",
        epilog=(
            "Common commands:\n"
            "  nodexo inventory\n"
            "  nodexo quote --gpu A6000 --duration 1h --ssh-key ~/.ssh/id_ed25519.pub\n"
            "  nodexo rent  --gpu A6000 --duration 1h --ssh-key ~/.ssh/id_ed25519.pub\n"
            "  nodexo rental-info <rental_id> --recovery-token <token>\n"
            "  nodexo --wallet <coldkey> --hotkey <hotkey> fleet\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wallet", default="", help="Bittensor coldkey name (fleet/fund/deregister only; prompts if omitted)")
    parser.add_argument("--hotkey", default="", help="Bittensor hotkey name (fleet/fund/deregister only; prompts if omitted)")
    parser.add_argument("--chain-config", default="",
                        help="Chain config JSON. Auto-picked from --subtensor-network if omitted "
                             "(test→chain_config_testnet.json, finney→chain_config_mainnet.json).")
    parser.add_argument("--subtensor-network", default=_default_subtensor_network(),
                        help="finney | test. Defaults to $NODEXO_SUBTENSOR_NETWORK, then available local chain_config_*.json.")
    parser.add_argument("--subtensor-endpoint",
                        default=(
                            os.environ.get("NODEXO_SUBTENSOR_ENDPOINT")
                            or os.environ.get("SUBTENSOR_ENDPOINT")
                            or ""
                        ),
                        help="Optional private/local subtensor RPC endpoint for the selected network.")
    # Resolution order for validator URL:
    #   1. --validator-url on the command line
    #   2. NODEXO_VALIDATOR_URL env var
    #   3. ~/.config/nodexo/config.toml -> [default] validator_url
    parser.add_argument("--validator-url", default=_default_validator_url(),
                        help="Internal validator URL for --validator-direct / --chain-direct operator modes.")
    parser.add_argument("--api-url", default=_default_api_url(),
                        help="Public renter API URL. Defaults to $NODEXO_API_URL, "
                             "~/.config/nodexo/config.toml, or https://nodexo.ai/api.")
    parser.add_argument("--api-key", default="",
                        help="Account API key for credit-backed rentals and account reads. "
                             "Defaults to $NODEXO_API_KEY.")
    parser.add_argument("--watch", nargs="?", const=30, default=0, type=int,
                        help="Fleet only: auto-refresh dashboard every N seconds (default: 30 if "
                             "passed without a value). Disables interactive navigation. "
                             "Ctrl+C to exit.")
    parser.add_argument("--show-stale", action="store_true",
                        help="Fleet only: include offline / expired-lease executors in the dashboard. "
                             "Hidden by default — they're contract dead-rows that the miner "
                             "can deregister to clean up.")

    sub = parser.add_subparsers(dest="command")

    def add_common_options(p):
        """Allow global options before or after the subcommand.

        Argparse only parses parent options before the subcommand. Operators
        naturally type `nodexo fleet --chain-config ...`; registering
        the same destinations on subparsers keeps that form working without
        changing the public API.
        """
        p.add_argument("--wallet", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--hotkey", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--chain-config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--subtensor-network", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--validator-url", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--api-url", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--api-key", default=argparse.SUPPRESS,
                       help="Account API key for credit rentals and account reads. "
                            "Defaults to $NODEXO_API_KEY.")
        return p

    # `fleet` accepts --watch / --show-stale either before or after the
    # subcommand. Argparse won't pick up parent flags placed *after* the
    # subcommand, so register them here too — last writer wins (subparser
    # default of 0/False would clobber a parent value otherwise, so we use
    # SUPPRESS to fall through to the parent's parse).
    status_p = add_common_options(sub.add_parser(
        "fleet",
        aliases=["miner"],
        help="Miner fleet dashboard via public API",
        description=(
            "Shows the executors attached to a miner hotkey. Default mode uses\n"
            "the public web API, so it does not perform registry or metagraph\n"
            "RPC calls. Use --chain-direct only when you need on-chain lease,\n"
            "UID/stake/emission, or deregister diagnostics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ))
    status_p.add_argument("--watch", nargs="?", const=30, default=argparse.SUPPRESS, type=int,
                          help="Auto-refresh every N seconds (default 30). Same as global --watch.")
    status_p.add_argument("--show-stale", action="store_true", default=argparse.SUPPRESS,
                          help="Include offline / expired rows. Same as global --show-stale.")
    status_p.add_argument("--chain-direct", action="store_true",
                          help="Operator diagnostics: query registry/metagraph directly and enable deregister.")

    fund_p = add_common_options(sub.add_parser("fund", help="Fund EVM wallet"))
    fund_p.add_argument("--amount", required=True)
    fund_p.add_argument("--yes", "-y", action="store_true")

    dereg_p = add_common_options(sub.add_parser("deregister", help="Deregister an on-chain executor"))
    dereg_p.add_argument("--executor-id", required=True,
                         help="Executor ID hex (full or unique prefix; with or without 0x)")
    dereg_p.add_argument("--yes", "-y", action="store_true",
                         help="Skip the confirmation prompt")

    def add_rent_selection_options(p):
        p.add_argument("--gpu", "--gpu-model", default="", dest="gpu_model",
                       help=(
                           "GPU model. Substring of the model name, case-insensitive: "
                           "'A6000', '4090', 'H100', etc."
                       ))
        p.add_argument("--gpu-count", type=int, default=1,
                       help="GPUs to allocate (default 1)")
        p.add_argument("--duration", default="1h",
                       help="x402 rental duration: '1h', '4h', '24h', '7d' (max 168h). Ignored for credit rentals.")
        p.add_argument("--image", default="ubuntu:22.04",
                       help=(
                           "Container image. Examples: ubuntu:22.04, "
                           "nvidia/cuda:12.4.0-runtime-ubuntu22.04, "
                           "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime."
                       ))
        p.add_argument("--storage-gb", type=int, default=30,
                       help="Minimum free Docker-backed storage to place on (default 30GB)")
        p.add_argument("--memory-gb", type=int, default=16,
                       help="Minimum host RAM to place on and requested container memory (default 16GB)")
        p.add_argument("--ssh-key", default="",
                       help="Path to SSH public key (default: ~/.ssh/id_ed25519.pub then id_rsa.pub)")
        return p

    quote_p = add_rent_selection_options(add_common_options(sub.add_parser(
        "quote",
        help="Preview x402 payment requirements without paying",
        description=(
            "Preview the selected inventory, duration, and x402 upto payment "
            "requirements. This does not sign, pay, provision, or hold capacity."
        ),
    )))
    quote_p.add_argument("--json", action="store_true", help="Print raw machine-readable quote JSON.")

    rent_p = add_rent_selection_options(add_common_options(sub.add_parser(
        "rent",
        help="Create an x402 or credit-backed GPU rental",
        description=(
            "Rent a GPU. The validator picks an executor matching your spec\n"
            "via weighted-random over candidates — you never see or target\n"
            "a specific miner.\n"
            "\n"
            "Examples:\n"
            "  nodexo quote --gpu A6000 --duration 1h\n"
            "  nodexo rent --gpu 4090 --duration 2h\n"
            "  nodexo rent --payment credits --gpu A6000 --api-key vc_...\n"
            "  nodexo rent --gpu A6000 --gpu-count 4 --duration 1h\n"
            "  nodexo rent --gpu H100  --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime\n"
            "\n"
            "x402 mode signs with $NODEXO_EVM_PRIVATE_KEY and needs --duration.\n"
            "credits mode uses account balance and needs $NODEXO_API_KEY or --api-key.\n"
            "Output: rental id, recovery token, direct SSH command, and next actions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )))
    rent_p.add_argument("--payment", "--payment-mode", default="x402", choices=["x402", "credits"],
                        dest="payment_mode",
                        help="Payment path: x402 accountless authorization or metered account credits.")
    rent_p.add_argument("--jump-user", default="",
                        help="Validator-direct only: SSH user for ProxyJump through the executor host "
                             "(e.g. 'ubuntu'). Without this the printed ssh command "
                             "is direct and assumes the container port is reachable.")
    rent_p.add_argument("--no-proxy-jump", action="store_true",
                        help="Validator-direct only: always print a direct ssh command, no ProxyJump even if --jump-user is given.")
    rent_p.add_argument("--save", default="",
                        help="Public API only: write successful rental response JSON to this file.")
    rent_p.add_argument("--max-usdc", default="", dest="max_usdc",
                        help="Public API only: override maximum x402 authorization cap.")
    rent_p.add_argument("--private-key-env", default="NODEXO_EVM_PRIVATE_KEY",
                        help="Public API only: env var containing the x402 EVM private key.")
    rent_p.add_argument("--idempotency-key", default="",
                        help="Public API only: idempotency key for agent/job retries.")
    rent_p.add_argument("--validator-direct", action="store_true",
                        help="Admin/operator mode: call validator /rent directly instead of public /api/rent.")

    credits_p = add_common_options(sub.add_parser(
        "credits",
        help="Show account credit balance",
        description="Reads account credit balance using --api-key or $NODEXO_API_KEY.",
    ))
    credits_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")

    account_rentals_p = add_common_options(sub.add_parser(
        "account-rentals",
        aliases=["account-rentals-list"],
        help="Show rentals attached to an account API key",
        description="Reads active and recent account rentals using --api-key or $NODEXO_API_KEY.",
    ))
    account_rentals_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")

    account_keys_p = add_common_options(sub.add_parser(
        "account-ssh-keys",
        help="List, add, or remove saved account SSH public keys",
        description=(
            "Manage SSH public keys stored on a signed-in account through an account API key.\n"
            "Requires the API key's `keys` scope."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ))
    account_keys_sub = account_keys_p.add_subparsers(
        dest="account_ssh_key_action",
    )
    account_keys_list_p = add_common_options(account_keys_sub.add_parser("list", help="List saved SSH public keys"))
    account_keys_list_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")
    account_keys_add_p = add_common_options(account_keys_sub.add_parser("add", help="Save an SSH public key"))
    account_keys_add_p.add_argument("--ssh-key", default="",
                                    help="Path to SSH public key (default: ~/.ssh/id_ed25519.pub then id_rsa.pub)")
    account_keys_add_p.add_argument("--label", default="", help="Optional display label")
    account_keys_add_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")
    account_keys_remove_p = add_common_options(account_keys_sub.add_parser("remove", help="Remove a saved SSH public key"))
    account_keys_remove_p.add_argument("--key-id", required=True, help="Key ID returned by list/add")
    account_keys_remove_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")

    operator_claim_p = add_common_options(sub.add_parser(
        "operator-claim",
        help="Sign operator dashboard hotkey claims",
        description=(
            "Operator dashboard claims are account-bound in the browser but signed\n"
            "locally with the miner hotkey. Copy the challenge command from the\n"
            "web app, run it on the machine with the Bittensor hotkey, then paste\n"
            "the signature back into the web app."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ))
    operator_claim_sub = operator_claim_p.add_subparsers(dest="operator_claim_command")
    operator_claim_sign_p = add_common_options(operator_claim_sub.add_parser(
        "sign",
        help="Sign a browser-generated operator claim challenge",
    ))
    operator_claim_sign_p.add_argument("--message-base64", default="", dest="message_base64",
                                       help="Base64url challenge generated by the operator page.")
    operator_claim_sign_p.add_argument("--message-file", default="",
                                       help="File containing the exact challenge message.")
    operator_claim_sign_p.add_argument("--message", default="",
                                       help="Exact challenge message. Use \\n for newlines.")
    operator_claim_sign_p.add_argument("--json", action="store_true",
                                       help="Print machine-readable JSON.")

    end_p = add_common_options(sub.add_parser("rental-end", help="Terminate a rental by id"))
    end_p.add_argument("rental_id", help="Rental ID returned by `rent`")
    end_p.add_argument("--recovery-token", default="",
                       help="Recovery token returned by `rent` or $NODEXO_RENTAL_RECOVERY_TOKEN")
    end_p.add_argument("--ssh-key", default="",
                       help=argparse.SUPPRESS)
    end_p.add_argument("--validator-direct", action="store_true",
                       help="Admin/operator mode: call validator internals directly.")
    end_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")

    extend_p = add_common_options(sub.add_parser("rental-extend", help="Top up an x402 rental"))
    extend_p.add_argument("rental_id", help="Rental ID returned by `rent`")
    extend_p.add_argument("--hours", type=int, required=True, help="Hours to add, 1..168")
    extend_p.add_argument("--recovery-token", default="",
                          help="Recovery token returned by `rent` or $NODEXO_RENTAL_RECOVERY_TOKEN")
    extend_p.add_argument("--max-usdc", default="", dest="max_usdc",
                          help="Override maximum x402 authorization cap.")
    extend_p.add_argument("--private-key-env", default="NODEXO_EVM_PRIVATE_KEY",
                          help="Env var containing the x402 EVM private key.")
    extend_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")

    mkt_p = add_common_options(sub.add_parser(
        "marketplace",
        help="Interactive public marketplace",
        description=(
            "Interactive renter marketplace through the web app /api routes.\n"
            "\n"
            "Keys in public mode:\n"
            "  p       preview x402 quote without paying\n"
            "  r       rent a GPU\n"
            "  Enter   refresh\n"
            "  q       quit\n"
            "\n"
            "For scripts / agents, use the one-shot subcommands:\n"
            "  nodexo inventory, quote, rent, rental-info, rental-extend, rental-end."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ))
    mkt_p.add_argument("--ssh-key", default="",
                       help="Path to YOUR SSH pubkey (default: ~/.ssh/id_ed25519.pub). "
                            "Used for ownership filter on /rentals + injected into rented containers.")
    mkt_p.add_argument("--json", action="store_true",
                       help="Public API mode: print raw JSON.")
    mkt_p.add_argument("--validator-direct", action="store_true",
                       help="Admin/operator mode: use the legacy validator-direct interactive dashboard.")

    inventory_p = add_common_options(sub.add_parser(
        "inventory",
        help="List public inventory once",
        description=(
            "Inventory view — aggregated GPU tiers available through the public API.\n"
            "\n"
            "Column glossary:\n"
            "  GPU           — model + count per executor (e.g. '4× RTX A6000').\n"
            "                  Multi-GPU configs are their own tier; rent the\n"
            "                  count you want with --gpu-count.\n"
            "  VRAM total    — total VRAM you get from one executor in this tier.\n"
            "                  For multi-GPU: 'TOTAL_GB (count×perGPU_GB)'.\n"
            "  CPU/RAM/Disk  — host hardware available beside the GPU.\n"
            "  Avail         — number of executors in this tier currently rentable.\n"
            "  Verify        — how the validator verifies the GPU.\n"
            "                  ZkGEMM = cryptographic proof of compute (today).\n"
            "                  TEE/CC = hardware-backed confidential compute when supported.\n"
            "  Reliability   — last 24h expected proof-check pass rate. Missing\n"
            "                  expected checks reduce reliability after the configured\n"
            "                  grace window; failed checks are never forgiven.\n"
            "                  Confidence: high ≥100 checks, medium ≥20,\n"
            "                  limited ≥5, warming up <5 checks.\n"
            "  Uptime        — average days the executors in this tier have been\n"
            "                  online (since first heartbeat).\n"
            "  Price/h       — public USDC rate per executor-hour.\n"
            "\n"
            "Renting:\n"
            "  nodexo rent --gpu <name> --duration <dur>\n"
            "    <name>     substring of the model (e.g. A6000, 4090, H100).\n"
            "               If it matches multiple models, you get an error\n"
            "               with the list of options.\n"
            "    <dur>      1h | 4h | 24h | 7d  (x402 max 168h)\n"
            "    --gpu-count N  ask for an N-GPU executor (defaults to 1).\n"
            "  See: nodexo quote --help and nodexo rent --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ))
    inventory_p.add_argument("--raw", action="store_true",
                             help="Show per-executor validator view (admin/operator debug only).")
    inventory_p.add_argument("--json", action="store_true",
                             help="Public API mode: print raw JSON.")
    inventory_p.add_argument("--validator-direct", action="store_true",
                             help="Admin/operator mode: call validator marketplace routes directly.")

    rentals_p = add_common_options(sub.add_parser(
        "rentals",
        help="Operator/debug: list active validator rentals",
        description=(
            "List active rentals from the validator control plane.\n"
            "By default this filters to rentals owned by your SSH public key. "
            "Use --all with NODEXO_ADMIN_TOKEN for the operator-wide view."
        ),
    ))
    rentals_p.add_argument("--ssh-key", default="", help="Path to SSH public key used to identify your rentals")
    rentals_p.add_argument("--all", action="store_true", help="Show all rentals; requires validator admin token")

    history_p = add_common_options(sub.add_parser(
        "history",
        aliases=["rental-history"],
        help="Show validator rental history",
        description="Operator/debug view of rental history. Uses the validator control plane and requires admin access.",
    ))
    history_p.add_argument("--mine", action="store_true",
                           help="Only rentals matching your default SSH key fingerprint")
    history_p.add_argument("--ssh-key", default="",
                           help="Path to SSH public key for --mine")
    history_p.add_argument("--since", default="",
                           help="Only since timestamp (epoch) or relative duration such as 24h or 7d")
    history_p.add_argument("--limit", type=int, default=100,
                           help="Max rows (default 100, max 1000)")

    info_p = add_common_options(sub.add_parser("rental-info", help="Show public rental status, SSH access, and actions"))
    info_p.add_argument("rental_id", help="Rental ID (full or prefix)")
    info_p.add_argument("--recovery-token", default="",
                        help="Recovery token returned by `rent` or $NODEXO_RENTAL_RECOVERY_TOKEN")
    info_p.add_argument("--validator-direct", action="store_true",
                        help="Admin/operator mode: call validator internals directly.")
    info_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")

    connect_p = add_common_options(sub.add_parser("connect", help="SSH into a rental by id or prefix"))
    connect_p.add_argument("rental_id", help="Rental ID or prefix")
    connect_p.add_argument("--ssh-key", default="", help="Path to SSH public key used to identify the rental")
    connect_p.add_argument("--recovery-token", default="",
                           help="Recovery token returned by `rent` or $NODEXO_RENTAL_RECOVERY_TOKEN")
    connect_p.add_argument("--validator-direct", action="store_true",
                           help="Admin/operator mode: recover connection from validator internals.")

    ssh_cfg_p = add_common_options(sub.add_parser("ssh-config", help="Print an SSH config block for a rental"))
    ssh_cfg_p.add_argument("rental_id", help="Rental ID or prefix")
    ssh_cfg_p.add_argument("--ssh-key", default="", help="Path to SSH public key used to identify the rental")
    ssh_cfg_p.add_argument("--recovery-token", default="",
                           help="Recovery token returned by `rent` or $NODEXO_RENTAL_RECOVERY_TOKEN")
    ssh_cfg_p.add_argument("--validator-direct", action="store_true",
                           help="Admin/operator mode: recover connection from validator internals.")

    keys_p = add_common_options(sub.add_parser("rental-key", help="Add or remove SSH keys on a live rental"))
    keys_p.add_argument("subcommand", choices=["add", "remove"], help="add | remove")
    keys_p.add_argument("rental_id", help="Rental ID (full or prefix)")
    keys_p.add_argument("--ssh-key", default="", help="Path to pubkey (for add)")
    keys_p.add_argument("--key-text", default="", help="Literal pubkey string (for remove)")
    keys_p.add_argument("--recovery-token", default="",
                        help="Recovery token returned by `rent` or $NODEXO_RENTAL_RECOVERY_TOKEN")
    keys_p.add_argument("--validator-direct", action="store_true",
                        help="Admin/operator mode: call validator internals directly.")
    keys_p.add_argument("--json", action="store_true", help="Print raw machine-readable JSON.")

    args = parser.parse_args()

    if args.command in ("fleet", "miner"):
        cmd_status(args)
    elif args.command == "fund":
        cmd_fund(args)
    elif args.command == "deregister":
        cmd_deregister(args)
    elif args.command == "quote":
        cmd_quote(args)
    elif args.command == "rent":
        cmd_rent(args)
    elif args.command == "credits":
        cmd_credits(args)
    elif args.command in ("account-rentals", "account-rentals-list"):
        cmd_account_rentals(args)
    elif args.command == "account-ssh-keys":
        cmd_account_ssh_keys(args)
    elif args.command == "operator-claim":
        cmd_operator_claim(args)
    elif args.command == "rental-end":
        cmd_rental_end(args)
    elif args.command == "rental-extend":
        cmd_rental_extend(args)
    elif args.command == "rental-key":
        cmd_rental_keys(args)
    elif args.command == "rentals":
        cmd_rentals(args)
    elif args.command in ("history", "rental-history"):
        cmd_history(args)
    elif args.command in ("connect",):
        cmd_connect(args)
    elif args.command in ("ssh-config",):
        cmd_ssh_config(args)
    elif args.command == "inventory":
        cmd_inventory(args)
    elif args.command == "rental-info":
        cmd_rental_info(args)
    elif args.command == "marketplace":
        cmd_marketplace(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
