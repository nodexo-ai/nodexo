"""
Sybil detection scanner.

Every SCAN_INTERVAL seconds, scan the validator's `executor_hardware` +
`executor_stats` + chain spec data for evidence that two `executor_id`s
are actually backed by the same physical machine. Flag offenders, drop
them from the rentable set, and zero their score. Active renter containers
are not force-cancelled by default; a hard flag is a miner-trust event, not
necessarily an immediate renter-connectivity failure.

Signals used (all OBJECTIVE — no renter-reported complaints, no
subjective judgement; nothing here can be triggered by a malicious
renter):

  1. **GPU UUID collision** — two distinct executor_ids report the
     same `primary_gpu_uuid`. Clear evidence: the GPU UUID is a hard
     hardware fingerprint and shouldn't appear twice in a healthy
     network. Honest case is impossible (one GPU = one UUID).

  2. **System UUID collision** — same `system_uuid` across multiple
     executor_ids. A miner may legitimately run multiple GPUs on one
     box; each is a separate executor BUT the hardware-fingerprint set
     should be disjoint. We flag any case where TWO executor_ids on
     the same system also share a GPU UUID (i.e., both registering
     the same GPU).

  3. **NVML digest outlier** — a miner reports an `nvml_digest` not
     present in the network-consensus set for their `driver_version`.
     Soft flag (downgrades trust); doesn't auto-ban (could be a brand-
     new driver release). Operator review unblocks.

  4. **Endpoint IP collision with GPU UUID collision** — two executor
     endpoints resolve to the same IP AND share a GPU UUID. Combined
     signal raises confidence to "ban".

  5. **Active sybil rental** — if more than one of OUR rentals lands
     on executors sharing a hardware fingerprint, immediate auto-cancel
     of the newer rentals + ban.

These signals are independent. Multiple signals on one executor
compound, but ANY of (1), (2)-with-shared-gpu, or (4) triggers a ban
on its own.

What this DOES NOT do:
  - Trust any renter complaint (out of scope; see project docs on the
    dispute design).
  - Spoofing-resistant hardware attestation (that's the monitor
    container + TEE roadmap; this is the cheap defense-in-depth layer
    that runs against current heartbeat data).
  - Re-verify reported UUIDs via active probing (the monitor container
    will do that; here we only correlate what miners voluntarily report
    through the existing heartbeat path).

Operator workflow on a flag:
  - Background scanner inserts a row in `sybil_flags`.
  - `available_executors` excludes the flagged executor (browse.py +
    rent.py read `get_banned_executor_ids`).
  - Active rentals on the flagged executor continue by default. Operators can
    enable SYBIL_AUTOCANCEL_ACTIVE_RENTALS=1 if they prefer strict cutoff.
  - Admin GET /sybil/flags lists open flags + evidence.
  - Admin POST /sybil/flags/{id}/clear unflags after manual review.
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import re
import time
from collections import defaultdict
from urllib.parse import urlparse

import bittensor as bt


# Match a 64-char hex string (Docker full container ID) anywhere in a
# cgroup path. Handles both cgroup v1 (`/docker/<id>`), v2
# (`0::/system.slice/docker-<id>.scope`), and Kubernetes (`/kubepods/.../<id>`).
_DOCKER_ID_RE = re.compile(r'\b([0-9a-f]{64})\b')


def _extract_container_id(cgroup_str: str) -> str | None:
    """Pull the 64-char container ID from a cgroup string. None if none."""
    if not cgroup_str:
        return None
    m = _DOCKER_ID_RE.search(cgroup_str)
    return m.group(1) if m else None


def _is_docker_cgroup(cgroup_str: str) -> bool:
    """True iff this cgroup describes a docker container (not host)."""
    return bool(cgroup_str) and (
        "docker" in cgroup_str or _DOCKER_ID_RE.search(cgroup_str) is not None
    )


SCAN_INTERVAL = 600.0   # 10 minutes between scans. Frequent enough to
                        # catch a sybil within one rental cycle; rare
                        # enough not to spam the DB.
MAX_MONITOR_SILENCE_S = 300.0  # If a once-seen monitor goes silent for
                               # this long, the miner has lost its
                               # attestable surface → ban.

# Proof-suppression detection: if validator verified >=N proofs in the
# window AND too few of them were witnessed by the monitor, the monitor
# is hiding PIDs while the validator still cryptographically verifies.
# The flag is windowed and self-healing: once recent monitor evidence no
# longer supports the claim, the stale hard flag is cleared automatically.
PROOF_SUPPRESSION_MIN_SAMPLES = 12      # min verified proofs before we fire
PROOF_SUPPRESSION_WITNESS_RATIO = 0.2   # hard only when almost never witnessed
PROOF_SUPPRESSION_RECOVERY_SAMPLES = 3  # latest clean verified+observed cycles
PROOF_SUPPRESSION_MISS_STREAK = 5       # latest verified+observed cycles all missed

# Validator-external arrival-latency signal: detect persistent GPU
# contention (sybil running concurrent compute on same chip).
ARRIVAL_MIN_BASELINE_SAMPLES = 15     # need this much history to build
                                       # a stable rolling baseline
ARRIVAL_WINDOW_SIZE = 10              # recent cycles to test against
ARRIVAL_Z_THRESHOLD = 3.0             # z-score above which we flag.
                                       # Z>3 is ~p<0.003 per sample;
                                       # the streak check below makes
                                       # false positives vanishing.
ARRIVAL_STREAK_THRESHOLD = 5          # need this many recent cycles
                                       # above z-threshold in the window
ARRIVAL_TIMING_SAMPLE_LIMIT = 80      # look back far enough to find a
                                       # same-workload baseline across
                                       # rent/free mode transitions


async def sybil_scanner_loop(db, get_active_rentals_callable,
                             terminate_callable, chain_snapshot,
                             orchestrator=None,
                             interval_s: float = SCAN_INTERVAL) -> None:
    """Background scanner. Cancel via task.cancel() at shutdown.

    Dependencies are injected (db, the active-rentals lookup, the
    internal terminate fn, the chain_snapshot) so this module doesn't
    pull in the full rent.py route surface.
    """
    bt.logging.info(f"Sybil scanner starting (interval={interval_s:.0f}s)")
    while True:
        try:
            await asyncio.sleep(interval_s)
            await _scan_once(
                db, get_active_rentals_callable, terminate_callable,
                chain_snapshot, orchestrator=orchestrator,
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.error(f"Sybil scanner loop error: {e}; retry in 60s")
            await asyncio.sleep(60)


async def _scan_once(db, get_active_rentals_callable, terminate_callable,
                     chain_snapshot, orchestrator=None) -> None:
    """One scan pass. Pulls the current ExecutorHardware + chain spec,
    computes fingerprint collisions, opens new sybil flags as needed.
    """
    from common.db import (
        ExecutorHardware, open_sybil_flag, get_banned_executor_ids,
    )

    # Collect (executor_id, hardware_row) for all known executors.
    rows: list = []
    with db.session() as s:
        for hw in s.query(ExecutorHardware).all():
            rows.append({
                "executor_id": hw.executor_id,
                "system_uuid": hw.system_uuid or "",
                "primary_gpu_uuid": hw.primary_gpu_uuid or "",
                "gpu_uuids_json": hw.gpu_uuids_json or "[]",
                "hw_static_json": hw.hw_static_json or "{}",
            })

    # Chain endpoints (for IP correlation). Read from the snapshot —
    # don't hit chain on every scan; the snapshot refresh handles
    # freshness.
    endpoint_by_eid: dict[str, str] = {}
    if chain_snapshot is not None:
        for ex in chain_snapshot.all_executors():
            endpoint_by_eid[ex.executor_id] = ex.endpoint or ""

    new_flags: list[tuple[str, str, dict]] = []  # (eid, reason, evidence)

    # ── Signal 1: GPU UUID collision ──────────────────────────────
    # The strongest individual signal: two executor_ids reporting the
    # SAME primary GPU UUID. This is mathematically impossible for
    # honest miners.
    by_gpu_uuid: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["primary_gpu_uuid"]:
            by_gpu_uuid[r["primary_gpu_uuid"]].append(r["executor_id"])
    for gpu_uuid, eids in by_gpu_uuid.items():
        if len(eids) <= 1:
            continue
        evidence = {"gpu_uuid": gpu_uuid, "peers": eids}
        for eid in eids:
            new_flags.append((eid, "gpu_uuid_dup", evidence))

    # ── Signal 2: System UUID + shared GPU UUID set ───────────────
    # A miner can legitimately register multiple executors on one host
    # IF each backs a DIFFERENT GPU. We flag system_uuid collisions
    # only when the gpu_uuids sets also overlap (one GPU claimed by
    # multiple executors).
    by_system_uuid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["system_uuid"]:
            try:
                gpu_uuids = set(_json.loads(r["gpu_uuids_json"]))
            except Exception:
                gpu_uuids = set()
            by_system_uuid[r["system_uuid"]].append({
                "executor_id": r["executor_id"],
                "gpu_uuids": gpu_uuids,
            })
    for sys_uuid, entries in by_system_uuid.items():
        if len(entries) <= 1:
            continue
        # Pairwise check: any two executors on the same system_uuid
        # whose gpu_uuids sets overlap? That's a sybil.
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                overlap = entries[i]["gpu_uuids"] & entries[j]["gpu_uuids"]
                if overlap:
                    evidence = {
                        "system_uuid": sys_uuid,
                        "peers": [entries[i]["executor_id"],
                                  entries[j]["executor_id"]],
                        "overlapping_gpus": sorted(overlap),
                    }
                    new_flags.append(
                        (entries[i]["executor_id"], "system_uuid_dup", evidence)
                    )
                    new_flags.append(
                        (entries[j]["executor_id"], "system_uuid_dup", evidence)
                    )

    # ── Signal 3: NVML digest outlier (soft signal) ───────────────
    # Group executors by (driver_version, OS arch). Within each group,
    # any executor whose nvml_digest is not the modal value gets a
    # soft flag. Doesn't auto-ban (could be a legit driver update); we
    # write the flag so operator can review.
    by_driver: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for r in rows:
        try:
            hw_static = _json.loads(r["hw_static_json"])
        except Exception:
            continue
        drv = (hw_static.get("gpus") or [{}])[0].get("driver_version", "")
        os_arch = (hw_static.get("os") or "") + "/" + (hw_static.get("cpu", {}).get("arch", ""))
        digest = hw_static.get("nvml_digest")
        # Tolerate both legacy string-form and new dict-form digests.
        if isinstance(digest, dict):
            digest_key = f"{digest.get('md5','')}:{digest.get('sha256','')}"
        else:
            digest_key = str(digest or "")
        if drv and digest_key and digest_key != ":":
            by_driver[(drv, os_arch)].append((r["executor_id"], digest_key))
    for (drv, os_arch), entries in by_driver.items():
        if len(entries) < 3:
            # Need a minimum sample size before any digest can be called
            # "modal" — small groups produce noisy flags.
            continue
        counts: dict[str, int] = defaultdict(int)
        for _, dk in entries:
            counts[dk] += 1
        modal = max(counts.values())
        for eid, dk in entries:
            if counts[dk] < modal:
                new_flags.append((eid, "nvml_digest_outlier", {
                    "driver_version": drv,
                    "os_arch": os_arch,
                    "reported_digest": dk,
                    "modal_count": modal,
                    "this_count": counts[dk],
                }))

    # ── Signal -1: Arrival-latency anomaly (validator-external) ──
    # Compute z-score of each executor's recent proof arrival latencies
    # against its own rolling baseline. Persistent slowdown indicates
    # GPU contention (sybil running concurrent work). The miner cannot
    # fake recv_time or chain block timestamps — both are external.
    # This is the cryptographically-anchored statistical signal.
    try:
        from common.db import ExecutorHardware, get_recent_proof_timing_samples
        with db.session() as s:
            all_eids = [r.executor_id for r in s.query(ExecutorHardware).all()]
        for eid in all_eids:
            required = ARRIVAL_MIN_BASELINE_SAMPLES + ARRIVAL_WINDOW_SIZE
            samples = get_recent_proof_timing_samples(
                db, eid, limit=max(ARRIVAL_TIMING_SAMPLE_LIMIT, required),
            )

            selected = None
            # Prefer external_pass_delta (network jitter cancels). Fall back
            # to exact beacon-relative wall clock, then legacy cycle-start
            # arrival latency for older rows.
            for metric_name, field_name in (
                ("external_pass_delta", "external_pass_delta_s"),
                ("wall_clock", "wall_clock_s"),
                ("arrival_latency", "arrival_latency_s"),
            ):
                metric_samples = [
                    smp for smp in samples if float(smp.get(field_name) or 0) > 0
                ]
                if len(metric_samples) < required:
                    continue

                # Critical guard: free executors run heavy proofs; rented
                # executors run micro proofs. Their timings are both
                # honest but live in different distributions, so compare
                # only the newest workload's own matrix_dim history.
                latest_matrix_dim = next(
                    (
                        int(smp.get("matrix_dim") or 0)
                        for smp in metric_samples
                        if int(smp.get("matrix_dim") or 0) > 0
                    ),
                    0,
                )
                if latest_matrix_dim <= 0:
                    # Old rows lack workload metadata. Skipping is safer
                    # than rebuilding the bug by mixing micro/heavy proofs.
                    continue
                same_workload = [
                    smp for smp in metric_samples
                    if int(smp.get("matrix_dim") or 0) == latest_matrix_dim
                ]
                if len(same_workload) < required:
                    continue
                selected = (metric_name, field_name, latest_matrix_dim, same_workload[:required])
                break

            if selected is None:
                continue
            metric_name, field_name, matrix_dim, selected_samples = selected
            recent = [float(smp[field_name]) for smp in selected_samples]
            # Most-recent values first. Window = newest ARRIVAL_WINDOW_SIZE;
            # baseline = the rest.
            window = recent[:ARRIVAL_WINDOW_SIZE]
            baseline = recent[ARRIVAL_WINDOW_SIZE:]
            # Robust statistics: median + MAD (median absolute deviation).
            # MAD * 1.4826 ≈ σ for normal distributions, but handles
            # outliers MUCH better than mean+stdev.
            #
            # Scale floor matters: in steady state the baseline is often
            # eerily consistent (same SSH tunnel, same thermal regime),
            # making MAD shrink to ~0.1s. With a tight scale, perfectly
            # normal operational drift (1–3 seconds) registers as a
            # 10+ sigma event and the detector misfires every time the
            # miner restarts or the network path has a hiccup.
            #
            # Observed real-world variability of `external_pass_delta`
            # on a stable A6000 + reverse-SSH-tunnel topology:
            #   median ~15s, window swings 15-18s under normal load.
            # Floor the scale at 2.5s so a single 7-second jitter does
            # NOT register as a sigma event, but a sustained 10-second+
            # asymmetry (the kind a sybil-by-relay would actually show)
            # still trips. This is the standard MAD-floor practice for
            # ops anomaly detection — pick a floor that reflects the
            # noise level of the actual deployment, not the eerily-
            # consistent steady-state.
            import statistics
            try:
                med = statistics.median(baseline)
                abs_devs = [abs(x - med) for x in baseline]
                mad = statistics.median(abs_devs) or 0.05
                scale = max(mad * 1.4826, 2.5)  # 2.5-second minimum noise floor
            except statistics.StatisticsError:
                continue
            # Count window cycles whose z-score exceeds threshold.
            z_scores = [(x - med) / scale for x in window]
            elevated = sum(1 for z in z_scores if z > ARRIVAL_Z_THRESHOLD)
            if elevated >= ARRIVAL_STREAK_THRESHOLD:
                new_flags.append((eid, "proof_arrival_anomaly", {
                    "metric": metric_name,
                    "baseline_median_s": round(med, 3),
                    "baseline_mad_s": round(mad, 3),
                    "scale_s": round(scale, 3),
                    "window_size": ARRIVAL_WINDOW_SIZE,
                    "elevated_count": elevated,
                    "z_threshold": ARRIVAL_Z_THRESHOLD,
                    "matrix_dim": matrix_dim,
                    "mode_guard": "same_matrix_dim",
                    "window_epochs": [
                        int(smp.get("epoch_id") or 0)
                        for smp in selected_samples[:ARRIVAL_WINDOW_SIZE]
                    ],
                    "window_values": [round(x, 3) for x in window],
                    "window_z_scores": [round(z, 2) for z in z_scores],
                }))
    except Exception as e:
        bt.logging.debug(f"proof_arrival_anomaly check failed: {e}")

    # ── Signal 0: Proof-verified-but-not-witnessed ────────────────
    # The strongest non-canary signal we have. Validator independently
    # verifies proofs cryptographically. Monitor reports per-cycle
    # whether the proof process was seen. An honest miner produces
    # matched cycles (both True). A rootkit hiding PIDs produces
    # "verified=True, witnessed=False" — a chain-anchored asymmetry
    # only a kernel-mode liar can generate.
    try:
        from common.db import (
            ExecutorHardware, clear_open_sybil_flag,
            get_monitor_recent_witness_stats, get_monitor_suppression_stats,
        )
        with db.session() as s:
            executor_ids_seen = {
                r.executor_id for r in s.query(ExecutorHardware).all()
            }
        for eid in executor_ids_seen:
            stats = get_monitor_suppression_stats(db, eid, window_hours=1.0)
            if stats["verified"] < PROOF_SUPPRESSION_MIN_SAMPLES:
                if clear_open_sybil_flag(
                    db, eid, "monitor_proof_suppression",
                    cleared_by="auto-monitor-window-recovered",
                ):
                    bt.logging.info(
                        f"monitor_proof_suppression cleared executor={eid[:16]} "
                        f"verified={stats['verified']} min={PROOF_SUPPRESSION_MIN_SAMPLES}"
                    )
                continue
            ratio = stats["matched"] / stats["verified"]
            recent = get_monitor_recent_witness_stats(
                db,
                eid,
                limit=max(PROOF_SUPPRESSION_RECOVERY_SAMPLES, PROOF_SUPPRESSION_MISS_STREAK),
                window_hours=2.0,
            )
            if ratio < PROOF_SUPPRESSION_WITNESS_RATIO:
                if recent["verified"] >= PROOF_SUPPRESSION_RECOVERY_SAMPLES and recent["all_matched"]:
                    if clear_open_sybil_flag(
                        db, eid, "monitor_proof_suppression",
                        cleared_by="auto-monitor-clean-streak",
                    ):
                        bt.logging.info(
                            f"monitor_proof_suppression cleared executor={eid[:16]} "
                            f"after {recent['verified']} clean recent cycles"
                        )
                    continue
                recent_miss_streak = (
                    recent["verified"] >= PROOF_SUPPRESSION_MISS_STREAK
                    and recent["matched"] == 0
                )
                if recent_miss_streak:
                    new_flags.append((eid, "monitor_proof_suppression", {
                        "observed": stats.get("observed", 0),
                        "verified": stats["verified"],
                        "witnessed": stats["witnessed"],
                        "matched": stats["matched"],
                        "verified_unwitnessed": stats["verified_unwitnessed"],
                        "witness_ratio": round(ratio, 3),
                        "threshold": PROOF_SUPPRESSION_WITNESS_RATIO,
                        "recent_miss_streak": PROOF_SUPPRESSION_MISS_STREAK,
                        "sample_cycles": stats["verified_unwitnessed_cycles"][:10],
                    }))
                elif clear_open_sybil_flag(
                    db, eid, "monitor_proof_suppression",
                    cleared_by="auto-monitor-signal-insufficient",
                ):
                    bt.logging.info(
                        f"monitor_proof_suppression cleared executor={eid[:16]} "
                        f"ratio={ratio:.3f} recent_matched={recent['matched']}/{recent['verified']}"
                    )
            else:
                if clear_open_sybil_flag(
                    db, eid, "monitor_proof_suppression",
                    cleared_by="auto-monitor-witness-recovered",
                ):
                    bt.logging.info(
                        f"monitor_proof_suppression cleared executor={eid[:16]} "
                        f"ratio={ratio:.3f} threshold={PROOF_SUPPRESSION_WITNESS_RATIO}"
                    )
    except Exception as e:
        bt.logging.debug(f"proof-suppression check failed: {e}")

    # ── Signal 0b: GPU process outside rental cgroup ─────────────
    # Direct sybil-during-rental detection. When executor X
    # is rented, the ONLY docker-cgroup processes touching the GPU
    # should be those running in X's rental container(s). Any docker
    # process from an UNKNOWN container = unauthorized compute on
    # this GPU = either:
    #   (a) the miner is running a sybil rental container they didn't
    #       tell us about, OR
    #   (b) the miner is running other docker workloads stealing
    #       compute during a rental.
    # Either way → HARD ban.
    #
    # The miner's own gpu_worker subprocess runs on the HOST (not
    # docker), so its cgroup is `/system.slice/...` not docker → not
    # flagged. Renter's processes inside the rental container's cgroup
    # match expected_container_ids → not flagged.
    #
    # The rootkit can defeat by lying about cgroup for sybil PIDs, but
    # that's a deeper attack than lying about NVML PID lists alone.
    try:
        from neurons.validator.api.routes.monitor import all_reports
        monitor_reports = all_reports()
    except Exception:
        monitor_reports = {}

    if orchestrator is not None:
        try:
            active_rentals = (
                get_active_rentals_callable()
                if callable(get_active_rentals_callable) else []
            )
            # group rentals by executor; one executor may have multiple containers
            rentals_by_exec: dict[str, list[dict]] = defaultdict(list)
            for r in active_rentals:
                rentals_by_exec[r.get("executor_id", "")].append(r)

            for eid, rentals in rentals_by_exec.items():
                if not eid or eid not in monitor_reports:
                    continue
                payload = (monitor_reports[eid] or {}).get("payload") or {}
                procs = payload.get("processes") or []
                if not procs:
                    continue
                # Fetch the miner's actual container list (signed call).
                endpoint = rentals[0].get("executor_endpoint", "")
                if not endpoint:
                    continue
                try:
                    miner_containers = await orchestrator.list_containers(endpoint)
                except Exception:
                    miner_containers = []
                # expected_container_ids: full + short prefix forms
                expected_ids: set[str] = set()
                expected_names: set[str] = {
                    r.get("container_name", "") for r in rentals
                    if r.get("container_name")
                }
                for c in miner_containers or []:
                    if c.get("name") in expected_names:
                        cid = c.get("container_id", "") or ""
                        if cid:
                            expected_ids.add(cid)
                            # short ID is first 12 chars
                            expected_ids.add(cid[:12])
                if not expected_ids:
                    # Couldn't resolve container IDs (miner unreachable
                    # or no overlap). Skip rather than false-positive.
                    continue
                # Examine each GPU process. Skip ones in proof-shape range
                # only when their cgroup is the host (miner's own gpu_worker
                # is allowed).
                for proc in procs:
                    cgroup = proc.get("cgroup", "")
                    if not _is_docker_cgroup(cgroup):
                        continue  # host process — miner's own work, allowed
                    cid = _extract_container_id(cgroup) or ""
                    # Match full or short ID (cgroup may contain either)
                    if cid in expected_ids:
                        continue
                    if any(cid.startswith(eid_short) for eid_short in expected_ids
                           if len(eid_short) <= 12 and eid_short):
                        continue
                    # Unauthorized docker process on this GPU.
                    new_flags.append((eid, "gpu_process_outside_rental", {
                        "unauthorized_pid": proc.get("pid"),
                        "unauthorized_cgroup": cgroup[:200],
                        "unauthorized_container_id": cid,
                        "expected_container_names": sorted(expected_names),
                        "expected_container_id_count": len(expected_ids),
                    }))
                    break  # one flag per executor; no point listing every PID
        except Exception as e:
            bt.logging.debug(f"gpu_process_outside_rental check failed: {e}")

    # ── Signal 3a: Monitor silent ─────────────────────────────────
    # Once we've heard from an executor's monitor, going silent for
    # MAX_MONITOR_SILENCE_S means the miner has lost its attestable
    # witness. Either the operator killed the container, or the
    # monitor crashed and the miner's supervisor can't restart it.
    # Either way, the executor is not safe to rent and not safe to
    # score against. Flag = ban.
    now = time.time()
    for eid, rep in monitor_reports.items():
        last_ts = (rep or {}).get("received_ts", 0)
        if last_ts and (now - last_ts) > MAX_MONITOR_SILENCE_S:
            new_flags.append((eid, "monitor_silent", {
                "last_report_ts": last_ts,
                "silence_s": int(now - last_ts),
            }))

    # ── Signal 3b: Cross-monitor PID overlap ──────────────────────
    # The actual sybil-defense surface. If TWO monitors (each bound
    # to a different executor_id) report the SAME host PID in their
    # CUDA-process list, they are watching the same physical GPU —
    # which means the two executor_ids are the same physical box,
    # i.e. a sybil. PIDs are host-wide under --pid=host, so two
    # monitors on the same machine see the same PID space.
    #
    # The miner's only escapes:
    #   - Run only ONE monitor and skip the other → monitor_silent
    #     flag fires on the un-monitored executor (5 min threshold).
    #   - Patch the kernel module to hide PIDs differently for each
    #     monitor → kernel rootkit, out of scope without TEE.
    #
    # Replaced the previous "rental container name" signal, which
    # was DEAD CODE: monitor populated container names from NVML's
    # 15-char comm field which never contains "nodexo-rental-…".
    try:
        from neurons.validator.api.routes.monitor import all_reports
        monitor_reports = all_reports()
    except Exception:
        monitor_reports = {}

    # Only consider reports we received in the last MONITOR_FRESHNESS_S
    # — stale reports across a validator restart could falsely overlap.
    MONITOR_FRESHNESS_S = 180
    fresh_reports: dict[str, dict] = {}
    for eid, rep in monitor_reports.items():
        if rep and (now - (rep.get("received_ts") or 0)) <= MONITOR_FRESHNESS_S:
            fresh_reports[eid] = rep

    # Build: pid → list[executor_id]. A pid appearing under multiple
    # executors is an unambiguous sybil signal.
    pid_to_eids: dict[int, list[str]] = defaultdict(list)
    for eid, rep in fresh_reports.items():
        payload = (rep or {}).get("payload") or {}
        for proc in payload.get("processes") or []:
            pid = proc.get("pid")
            if isinstance(pid, int) and pid > 0:
                pid_to_eids[pid].append(eid)

    seen_sybil_pairs: set[frozenset] = set()
    for pid, eids in pid_to_eids.items():
        uniq = set(eids)
        if len(uniq) <= 1:
            continue
        # Dedupe by the exact set of peers — one flag per peer-set is
        # enough; a sybil with 5 PIDs overlapping doesn't need 5 flags.
        peers_key = frozenset(uniq)
        if peers_key in seen_sybil_pairs:
            continue
        seen_sybil_pairs.add(peers_key)
        evidence = {"shared_pid": pid, "peers": sorted(uniq)}
        for eid in uniq:
            new_flags.append((eid, "cross_monitor_pid_overlap", evidence))

    # ── Signal 4: Endpoint IP collision + GPU UUID collision ─────
    # Two miners sharing an IP is plausible (NAT, datacenter rack).
    # Two miners sharing an IP AND a GPU UUID is not.
    ip_by_eid: dict[str, str] = {}
    for eid, ep in endpoint_by_eid.items():
        try:
            host = urlparse(ep).hostname
            if host:
                ip_by_eid[eid] = host
        except Exception:
            pass
    by_ip: dict[str, list[str]] = defaultdict(list)
    for eid, ip in ip_by_eid.items():
        by_ip[ip].append(eid)
    for ip, eids in by_ip.items():
        if len(eids) <= 1:
            continue
        # For every pair on the same IP, check GPU UUID overlap.
        rows_by_eid = {r["executor_id"]: r for r in rows}
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                ri = rows_by_eid.get(eids[i], {})
                rj = rows_by_eid.get(eids[j], {})
                gi = ri.get("primary_gpu_uuid", "")
                gj = rj.get("primary_gpu_uuid", "")
                if gi and gi == gj:
                    evidence = {
                        "endpoint_ip": ip,
                        "shared_gpu_uuid": gi,
                        "peers": [eids[i], eids[j]],
                    }
                    new_flags.append((eids[i], "endpoint_ip_dup", evidence))
                    new_flags.append((eids[j], "endpoint_ip_dup", evidence))

    if not new_flags:
        bt.logging.debug("Sybil scanner: no new flags")
        return

    # Persist flags (idempotent — open_sybil_flag skips duplicates).
    inserted = 0
    for eid, reason, evidence in new_flags:
        if open_sybil_flag(db, eid, reason, evidence):
            inserted += 1
            bt.logging.warning(
                f"Sybil flag: executor={eid[:16]} reason={reason} "
                f"evidence={_json.dumps(evidence)[:200]}"
            )
    if inserted:
        bt.logging.warning(f"Sybil scanner: {inserted} new flag(s) opened")

    # Active-rental policy:
    #   - HARD flags already remove the executor from future rental flow and
    #     score it at zero through get_banned_executor_ids().
    #   - Do not kill an active renter's container by default. For the renter,
    #     a running SSH session on a degraded-but-reachable host is usually
    #     better than abrupt termination. Proof failures still auto-cancel via
    #     validator.py because those mean the paid GPU itself failed verification.
    #   - Operators who want strict behavior can opt in.
    banned = get_banned_executor_ids(db)
    if not banned:
        return
    rentals = get_active_rentals_callable() if callable(get_active_rentals_callable) else []
    impacted = [r for r in rentals if r.get("executor_id") in banned]
    if impacted and not callable(terminate_callable):
        bt.logging.warning(
            f"Sybil scanner: {len(impacted)} active rental(s) on flagged executor(s); "
            "auto-cancel unavailable on this validator"
        )
        return
    if impacted and os.environ.get("SYBIL_AUTOCANCEL_ACTIVE_RENTALS", "0") != "1":
        bt.logging.warning(
            f"Sybil scanner: {len(impacted)} active rental(s) on flagged executor(s); "
            "not auto-cancelling (set SYBIL_AUTOCANCEL_ACTIVE_RENTALS=1 to enable)"
        )
        return
    for r in rentals:
        if r.get("executor_id") in banned:
            rid = r.get("rental_id", "")
            if not rid:
                continue
            bt.logging.warning(
                f"Sybil scanner: auto-cancelling rental {rid[:8]}… "
                f"(executor {r['executor_id'][:16]} flagged)"
            )
            try:
                # reason="sybil" so the audit trail distinguishes this
                # from routine orphan cleanup. _VALID_REASONS in
                # rent.py was extended to accept this.
                await terminate_callable(rid, "sybil")
            except Exception as e:
                bt.logging.error(
                    f"Sybil scanner: terminate({rid[:8]}) failed: {e}"
                )
