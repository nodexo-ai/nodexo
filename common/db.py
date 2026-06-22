"""
Database layer for Nodexo — SQLAlchemy models + engine setup.

Supports PostgreSQL (production) and SQLite (dev/testing).

Tables:
  - executors: identity + binding record
  - executor_hardware: static hardware fingerprint
  - executor_stats: live metrics + computed analytics
  - proof_results, proof_receipts: verification verdicts and timing ingress
  - rentals: active + historical rental tracking
  - sybil_flags, monitor_*, canary_records: sybil + canary state

Prior-art context: protocol design notes.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

import bittensor as bt

from sqlalchemy import (
    Column, BigInteger, Integer, Float, String, Text, Boolean, DateTime,
    Index, UniqueConstraint, create_engine, event,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


def _nodexo_data_dir() -> str:
    return os.path.abspath(os.path.expanduser(
        os.environ.get("NODEXO_DATA_DIR", "~/.nodexo")
    ))


def default_db_url() -> str:
    return "sqlite:///" + os.path.join(_nodexo_data_dir(), "validator.db")


# ── Table 1: Executor Identity ─────────────────────────────────

class Executor(Base):
    """Executor identity + binding record.

    Created on first contact (heartbeat or proof submission).
    One row per physical GPU machine.
    """
    __tablename__ = "executors"

    executor_id = Column(String(64), primary_key=True)    # SHA256 hex
    hotkey_ss58 = Column(String(48))                       # Bittensor SS58
    coldkey_ss58 = Column(String(48), default="")
    miner_address = Column(String(42), default="")         # EVM address
    endpoint = Column(String(256), default="")
    meta = Column(Text, default="")
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


# ── Table 2: Executor Hardware ─────────────────────────────────

class ExecutorHardware(Base):
    """Static hardware fingerprint (updated on heartbeat).

    Full system spec snapshot — captures everything reported by the
    miner in the `hw_static` heartbeat payload.
    """
    __tablename__ = "executor_hardware"

    executor_id = Column(String(64), primary_key=True)

    # System
    system_uuid = Column(String(64), default="")

    # CPU
    cpu_model = Column(String(128), default="")
    cpu_arch = Column(String(16), default="")
    cpu_cores_physical = Column(Integer, default=0)
    cpu_cores_logical = Column(Integer, default=0)
    cpu_freq_max_mhz = Column(Float, default=0)

    # RAM
    ram_total_bytes = Column(BigInteger, default=0)

    # GPU
    gpu_count = Column(Integer, default=0)
    primary_gpu_name = Column(String(128), default="")
    primary_gpu_uuid = Column(String(64), default="")
    primary_gpu_vram_bytes = Column(BigInteger, default=0)
    gpu_vram_total_bytes = Column(BigInteger, default=0)
    gpu_uuids_json = Column(Text, default="[]")
    driver_version = Column(String(32), default="")

    # Disk
    disk_count = Column(Integer, default=0)
    disk_total_bytes = Column(BigInteger, default=0)
    primary_disk_type = Column(String(16), default="")       # ssd/hdd/nvme

    # Container isolation
    sysbox_detected = Column(Boolean, default=False)
    tee_platform = Column(String(32), default="")

    # Raw blob (full hw_static from heartbeat)
    hw_static_json = Column(Text, default="{}")
    software_version = Column(String(32), default="")

    updated_at = Column(DateTime, default=datetime.utcnow)


# ── Table 3: Executor Stats ───────────────────────────────────

class ExecutorStats(Base):
    """Live metrics + computed analytics. One row per executor, updated by
    heartbeats (live metrics) and analyzer loop (scoring + status).

    Denormalized for fast queries — one row per executor.
    """
    __tablename__ = "executor_stats"

    executor_id = Column(String(64), primary_key=True)

    # Identity (denormalized for fast queries)
    hotkey_ss58 = Column(String(48), default="")
    miner_address = Column(String(42), default="")
    endpoint = Column(String(256), default="")

    # Allocation state
    alloc_state = Column(String(16), default="free")         # free/allocated/banned
    alloc_updated_at = Column(DateTime, nullable=True)

    # Live metrics (from heartbeat hw_live)
    cpu_usage_pct = Column(Float, default=0)
    ram_used_bytes = Column(BigInteger, default=0)
    ram_total_bytes = Column(BigInteger, default=0)
    swap_used_bytes = Column(BigInteger, default=0)
    swap_total_bytes = Column(BigInteger, default=0)
    gpu_usage_pct = Column(Float, default=0)
    gpu_vram_used_bytes = Column(BigInteger, default=0)
    gpu_vram_total_bytes = Column(BigInteger, default=0)
    gpu_temp_c = Column(Float, default=0)
    gpu_power_w = Column(Float, default=0)
    disk_used_bytes = Column(BigInteger, default=0)
    disk_total_bytes = Column(BigInteger, default=0)
    net_rx_bytes = Column(BigInteger, default=0)
    net_tx_bytes = Column(BigInteger, default=0)
    hw_live_json = Column(Text, default="{}")

    # Proof scoring (rolling windows)
    pass_rate_1h = Column(Float, default=0)
    pass_rate_24h = Column(Float, default=0)
    pass_rate_all = Column(Float, default=0)
    avg_delta_1h = Column(Float, default=0)
    avg_delta_24h = Column(Float, default=0)
    avg_wall_1h = Column(Float, default=0)
    avg_wall_24h = Column(Float, default=0)
    samples_1h = Column(Integer, default=0)
    samples_24h = Column(Integer, default=0)
    samples_all = Column(Integer, default=0)

    # Status (computed by analyzer_loop every 60s)
    current_status = Column(String(16), default="waiting")   # ok/stale/offline/fail/pending/waiting
    current_reason = Column(String(256), default="")
    last_status = Column(String(16), default="")
    last_success_at = Column(DateTime, nullable=True)
    last_fail_at = Column(DateTime, nullable=True)
    last_activity_at = Column(DateTime, nullable=True)
    age_sec = Column(Float, default=0)

    # Scoring
    composite_score = Column(Float, default=0)
    proof_score = Column(Float, default=0)
    availability_score = Column(Float, default=0)
    hardware_score = Column(Float, default=0)
    trust_level = Column(String(8), default="full")          # light/spot/full
    utilization_mult = Column(Float, default=1.0)

    # Run tracking
    last_run_id = Column(String(64), default="")
    last_epoch_id = Column(Integer, default=0)
    last_b_num = Column(Integer, default=0)

    first_seen = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


# ── Table 4: Proof Results ────────────────────────────────────

class ProofResult(Base):
    """Verification result for one executor's proof in one epoch."""
    __tablename__ = "proof_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    executor_id = Column(String(64), index=True)
    epoch_id = Column(Integer)
    valid = Column(Boolean)
    tier = Column(String(8))              # light/spot/full
    reason = Column(String(256), default="")
    delta_seconds = Column(Float, default=0)
    compute_time = Column(Float, default=0)
    verification_time_ms = Column(Float, default=0)
    verified_at = Column(DateTime, default=datetime.utcnow)
    # Validator-external timing signal. recv_time
    # is the validator's wall-clock when the recipe arrived. arrival_
    # latency_s is recv_time - estimated_cycle_start_ts (the wall-clock
    # gap from chain cycle boundary to recipe receipt). Slowdowns from
    # GPU contention show up here without trusting the miner's self-
    # reported compute_time. See sybil/scanner.py proof_arrival_anomaly.
    recv_time = Column(Float, default=0)
    arrival_latency_s = Column(Float, default=0)
    # External metric: time-delta between validator receiving pass_0 and
    # pass_1 receipts. This is accepted only together with receipt freshness
    # checks; otherwise a miner could withhold pass_0 and shrink the delta.
    external_pass_delta_s = Column(Float, default=0)
    # Exact wall-clock timing used by the validator's free-capacity timing
    # gate: recv_time minus the validator-local beacon hash availability time.
    # This differs from arrival_latency_s, which is measured from cycle start
    # for older telemetry compatibility.
    wall_clock_s = Column(Float, default=0)
    # Multi-GPU receipt spread, bound to the final recipe roots/T_commit values.
    # Used for calibrating the receipt-spread deadline that catches serialized
    # fake multi-GPU execution.
    receipt_pass0_spread_s = Column(Float, default=0)
    receipt_pass1_spread_s = Column(Float, default=0)
    receipt_pass0_wall_s = Column(Float, default=0)
    receipt_pass1_wall_s = Column(Float, default=0)
    # Proof workload metadata. The miner intentionally runs a heavy
    # proof while free and a micro proof while rented; timing baselines
    # must not mix those modes.
    matrix_dim = Column(Integer, default=0)
    allocation_state = Column(String(16), default="")

    __table_args__ = (
        UniqueConstraint("executor_id", "epoch_id", name="uniq_proof_exec_epoch"),
        Index("ix_proof_exec_verified", "executor_id", "verified_at"),
        Index("ix_proof_valid_verified", "valid", "verified_at"),
    )


# ── Table 4a: Proof Timing Receipts ────────────────────────────

class ProofReceipt(Base):
    """Timing-critical receipt captured when a GPU pass reaches a validator.

    This table is intentionally separate from proof_results. Receipt ingress can
    run in a lightweight process while the main validator verifies recipes later
    and binds these rows to the final recipe roots/T_commit values.
    """
    __tablename__ = "proof_receipts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    executor_id = Column(String(64), index=True)
    epoch_id = Column(Integer)
    gpu_index = Column(Integer)
    u = Column(Integer)  # 0 = pass_0, 1 = pass_1
    recv_time = Column(Float, default=0)
    root = Column(String(128), default="")
    t_commit = Column(String(128), default="")
    hotkey_ss58 = Column(String(48), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "executor_id", "epoch_id", "gpu_index", "u", "root", "t_commit",
            name="uniq_proof_receipt_bound_pass",
        ),
        Index("ix_proof_receipts_exec_epoch", "executor_id", "epoch_id"),
        Index("ix_proof_receipts_created", "created_at"),
    )


# ── Table 4b: Validator Liveness / Outages ─────────────────────

class ValidatorLiveness(Base):
    """Latest heartbeat for a validator process role.

    Used to reconstruct validator/proxy outage windows after restart. If the
    validator was unavailable, missing proof/heartbeat submissions are neutral
    for miner scoring instead of counted as miner failures.
    """
    __tablename__ = "validator_liveness"

    service_name = Column(String(64), primary_key=True)
    instance_id = Column(String(128), default="")
    started_at = Column(DateTime, default=datetime.utcnow)
    ready_at = Column(DateTime, default=datetime.utcnow)
    last_heartbeat_at = Column(DateTime, default=datetime.utcnow)
    shutdown_at = Column(DateTime)


class ValidatorOutage(Base):
    """Persisted validator unavailable window."""
    __tablename__ = "validator_outages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    service_name = Column(String(64), index=True)
    started_at = Column(DateTime, index=True)
    ended_at = Column(DateTime, index=True)
    duration_s = Column(Float, default=0)
    reason = Column(String(64), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_validator_outage_window", "service_name", "started_at", "ended_at"),
    )


# ── Table 5: Rentals ──────────────────────────────────────────

class Rental(Base):
    """Active + historical rental tracking.

    Every rental ever created leaves a row here. Active rentals have
    status='active' and terminated_at IS NULL; everything else is historical.
    Snapshot fields (gpu_model, vram_mb, price_per_gpu_hour_rao) are stored
    at rental START so historical reports don't depend on the executor still
    being registered with the same specs.
    """
    __tablename__ = "rentals"

    rental_id = Column(String(32), primary_key=True)
    executor_id = Column(String(64), index=True)
    executor_endpoint = Column(String(256))
    container_name = Column(String(128))
    ssh_host = Column(String(128))
    ssh_port = Column(Integer)
    ssh_user = Column(String(32))
    client_request_id = Column(String(128), default="", index=True)
    ttl_seconds = Column(Integer, default=0)
    status = Column(String(32), default="active")
        # active | terminated_user | terminated_ttl | terminated_proof_fail | terminated_orphan | ...
    release_reason = Column(String(32), default="")
        # Non-empty only while status='release_pending'. This preserves the
        # intended terminal reason across validator restarts when chain release
        # failed transiently.
    release_requested_at = Column(DateTime, nullable=True)
    release_attempts = Column(Integer, default=0)
    last_release_error = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    terminated_at = Column(DateTime, nullable=True)

    # Snapshot of executor specs at rent time (so historical reports work
    # even after the executor's chain spec changes / deregisters).
    gpu_model = Column(String(64), default="")    # e.g. "NVIDIA RTX A6000"
    vram_mb = Column(Integer, default=0)
    gpu_count = Column(Integer, default=1)
    price_per_gpu_hour_rao = Column(BigInteger, default=0)

    # Renter identity. Today only an SSH-pubkey fingerprint (sha256[:16]) —
    # weak correlation across rentals from the same person. When auth lands,
    # replace with renter_id (account/wallet/api-key fk).
    renter_pubkey_fp = Column(String(16), default="", index=True)

    # Chain bookkeeping
    start_block = Column(BigInteger, default=0)   # block of markRented
    end_block = Column(BigInteger, default=0)     # block of markAvailable

    # Billing (computed at terminate time)
    actual_seconds = Column(Integer, default=0)   # terminated_at - created_at
    total_paid_rao = Column(BigInteger, default=0)
        # price_per_gpu_hour_rao * gpu_count * (actual_seconds / 3600)

    __table_args__ = (
        Index("ix_rental_status", "status"),
        Index("ix_rental_exec", "executor_id"),
        Index("ix_rental_renter", "renter_pubkey_fp"),
        Index("ix_rental_client_request", "client_request_id"),
        Index("ix_rental_created", "created_at"),
    )


# ── Table 6: Executor Hotkey Bindings (proof-side, persistent) ──

class ExecutorHotkeyBinding(Base):
    """Persistent first-claim binding between an executor_id and the
    hotkey that proved ownership via SR25519 sig.

    Without persistence (the pre-audit state), every validator restart
    flushed `_executor_hotkey_binding` in proofs.py and the next
    proof-recipe-claim from ANY hotkey re-bound the executor to that
    hotkey. That's a 60-second sybil window per validator restart.
    """
    __tablename__ = "executor_hotkey_bindings"
    executor_id = Column(String(64), primary_key=True)
    hotkey_ss58 = Column(String(48))
    bound_at = Column(DateTime, default=datetime.utcnow)


# ── Table 7: Monitor Pubkey TOFU (persistent) ────────────────

class MonitorPubkeyBinding(Base):
    """Persistent TOFU binding between an executor_id and its monitor's
    SR25519 pubkey. Without persistence, validator restart wipes the
    binding and a pubkey-rotated monitor could TOFU itself a new
    binding with no `monitor_pubkey_rotation` flag firing.
    """
    __tablename__ = "monitor_pubkey_bindings"
    executor_id = Column(String(64), primary_key=True)
    monitor_ss58 = Column(String(48))
    bound_at = Column(DateTime, default=datetime.utcnow)


# ── Table 8: Rental Windows (block-anchored timeline) ────────

class MonitorCycleObservation(Base):
    """Per-cycle record of what an executor's monitor reported.

    Built incrementally as monitor reports arrive: each report covers
    one cycle, and we record whether `expected_proof_seen` was True
    for that cycle. The sybil scanner cross-references this with
    `proof_results` (validator-verified proofs) to detect rootkit-
    style PID suppression.

    Suppression invariant: if the validator verifies a valid proof
    for executor X at cycle N, an honest miner's monitor MUST have
    seen the proof process (expected_proof_seen=True). A rootkit
    that hides the proof PID from NVML produces False here while
    the proof is still validly submitted to the validator. That
    asymmetry is the unfakeable signal.
    """
    __tablename__ = "monitor_cycle_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    executor_id = Column(String(64), index=True)
    cycle_id = Column(Integer)
    expected_proof_seen = Column(Boolean, default=False)
    extra_proofs_seen = Column(Integer, default=0)
    observed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("executor_id", "cycle_id",
                         name="uniq_mco_exec_cycle"),
        Index("ix_mco_exec_observed", "executor_id", "observed_at"),
    )


class RentalWindowRow(Base):
    """Persistent block-anchored rental timeline.

    Mirrors neurons/validator/rental/state.py's in-memory `_rented`
    dict. Each row is one rental window for one executor. end_block=0
    means still active.

    Without persistence, the validator restart loses the whole
    timeline → `was_rented_at(block)` falls through to live chain
    state, re-opening the cycle-window race we explicitly fixed.
    """
    __tablename__ = "rental_windows"
    rental_id = Column(String(32), primary_key=True)
    executor_id = Column(String(64), index=True)
    start_block = Column(BigInteger)
    end_block = Column(BigInteger, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class CanaryRecord(Base):
    """One row per executor: last canary outcome + consecutive-error
    streak used to detect canary sabotage (a miner who reliably breaks
    pip-install or network egress would keep last_status flapping
    between pass/error indefinitely without ever flipping to fail —
    bypassing the canary_failed gate. After N errors in a row the
    scheduler treats it as effective failure).
    """
    __tablename__ = "canary_records"
    executor_id = Column(String(64), primary_key=True)
    last_canaried_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_status = Column(String(16), default="")    # pass | fail | error
    last_summary = Column(Text, default="")
    last_result_json = Column(Text, default="")
    # Resets to 0 on any pass/fail outcome; increments on consecutive
    # error outcomes. Threshold check lives in the scheduler.
    consecutive_errors = Column(Integer, default=0)
    # Set when the scheduler calls markRented; cleared in the runner's
    # finally block. If validator crashes between markRented and the
    # finally (~8min window), `is_rented=True` stays on chain forever
    # (or until 30min lease expiry). On boot, reconcile any rows with
    # inflight_at set within the lease window by calling markAvailable.
    inflight_at = Column(DateTime, default=None, nullable=True)


# ── Table: Idempotency keys ──────────────────────────────────
#
# Generic key → response cache for endpoints that need replay-safe
# write semantics. Currently used by POST /rentals/{id}/extend (the
# webapp's x402 retry path). Same key + same context = same cached
# response, so a retried payment cannot double-extend a rental.
#
# Concurrency: PRIMARY KEY on `key` + INSERT-then-SELECT inside a
# single transaction. Across processes the DB-level unique-violation
# is the arbiter. SQLite serialises writes within a single process
# and the busy timeout (5s default) handles cross-process contention.

class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    key = Column(String(128), primary_key=True)
    scope = Column(String(64), index=True)            # "extend_rental", future: "topup_balance", etc.
    response_json = Column(Text, nullable=True)       # filled after upstream call lands
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class ChainCursor(Base):
    """Small persistent cursor table for chain event indexers.

    Used by the ComputeRegistry rental-event indexer so validators can
    reconstruct rental windows created by peer validators without rescanning
    old blocks on every restart.
    """
    __tablename__ = "chain_cursors"
    key = Column(String(128), primary_key=True)
    block_number = Column(BigInteger, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class RegistryExecutorState(Base):
    """Durable ComputeRegistry inventory index.

    This is the validator-local current view reconstructed from
    ComputeRegistry events plus periodic paginated reconciliation. It keeps
    renter/API reads off the chain hot path while preserving chain as the
    authority for writes and fallback reads.
    """
    __tablename__ = "registry_executor_state"

    executor_id = Column(String(64), primary_key=True)
    miner_address = Column(String(42), default="", index=True)
    endpoint = Column(String(256), default="")
    gpu_model_hash = Column(String(64), default="")
    gpu_count = Column(Integer, default=0)
    vram_mb = Column(Integer, default=0)
    price_per_gpu_hour = Column(BigInteger, default=0)
    expires_at = Column(BigInteger, default=0, index=True)
    is_active = Column(Boolean, default=True, index=True)
    is_rented = Column(Boolean, default=False, index=True)
    is_removed = Column(Boolean, default=False, index=True)
    miner_uid = Column(Integer, default=0)
    miner_registered = Column(Boolean, default=False)
    uid_owner_match = Column(Boolean, default=False)
    last_event_block = Column(BigInteger, default=0)
    last_event_log_index = Column(Integer, default=0)
    last_event_kind = Column(String(32), default="")
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_registry_exec_active_rented_exp", "is_active", "is_rented", "expires_at"),
        Index("ix_registry_exec_removed_updated", "is_removed", "updated_at"),
    )


# ── Table 9: Sybil Flags ──────────────────────────────────────

class SybilFlag(Base):
    """Sybil-detection findings raised by the validator's background scanner.

    Severity tiers:
      - "hard" — high-confidence active or cross-correlated evidence:
        duplicate physical fingerprints, active canary failure, or rental
        SLA failure. Auto-bans the executor from score/rental selection.
      - "soft" — passive monitor findings and single self-reported signals.
        Operator review required; no auto-ban. The monitor sidecar is useful
        observability, but process sampling alone is not stable enough to
        be an enforcement primitive.

    Evidence is structured JSON so an operator can review WHY the flag
    fired before clearing it. We never trust renter-reported complaints;
    this table only stores objective evidence.
    """
    __tablename__ = "sybil_flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    executor_id = Column(String(64), index=True)
    reason = Column(String(64))
        # hard: cross_monitor_pid_overlap | endpoint_ip_dup |
        #       canary_* | rental_*
        # soft: gpu_uuid_dup | system_uuid_dup | nvml_digest_outlier |
        #       monitor_* | behavioral_extra_proof
    severity = Column(String(8), default="hard")  # "hard" | "soft"
    evidence_json = Column(Text, default="{}")
    flagged_at = Column(DateTime, default=datetime.utcnow)
    cleared_at = Column(DateTime, nullable=True)
    cleared_by = Column(String(64), default="")

    __table_args__ = (
        Index("ix_sybil_exec_flagged", "executor_id", "flagged_at"),
        Index("ix_sybil_open", "executor_id", "cleared_at"),
        Index("ix_sybil_severity_open", "severity", "cleared_at"),
    )


# ── Database connection ────────────────────────────────────────

class Database:
    """Database connection manager. Supports PostgreSQL and SQLite."""

    def __init__(self, url: Optional[str] = None):
        if url is None:
            url = os.environ.get("DB_URL") or default_db_url()
        self.url = url
        self._prepare_sqlite_path(url)
        engine_kwargs = self._engine_kwargs(url)
        self.engine = create_engine(url, **engine_kwargs)
        self._configure_engine(url)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self._apply_inline_migrations()
        bt.logging.info(f"Database initialized: {url.split('@')[-1] if '@' in url else url}")

    @staticmethod
    def _parsed_url(url: str):
        try:
            return make_url(url)
        except Exception:
            return None

    def _prepare_sqlite_path(self, url: str) -> None:
        parsed = self._parsed_url(url)
        if parsed is None or not parsed.drivername.startswith("sqlite"):
            return
        path = parsed.database
        if not path or path == ":memory:":
            return
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        try:
            os.chmod(os.path.dirname(path), 0o700)
        except OSError:
            pass

    def _engine_kwargs(self, url: str) -> dict:
        parsed = self._parsed_url(url)
        kwargs = {"future": True, "pool_pre_ping": True}
        if parsed is not None and parsed.drivername.startswith("sqlite"):
            kwargs["connect_args"] = {
                "timeout": float(os.environ.get("SQLITE_BUSY_TIMEOUT_SECONDS", "5")),
                "check_same_thread": False,
            }
        return kwargs

    def _configure_engine(self, url: str) -> None:
        parsed = self._parsed_url(url)
        if parsed is None or not parsed.drivername.startswith("sqlite"):
            return
        database_path = parsed.database or ""
        file_backed = database_path and database_path != ":memory:"

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.execute("PRAGMA foreign_keys=ON")
                if file_backed:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
            finally:
                cursor.close()

    def _apply_inline_migrations(self) -> None:
        """Add columns that were introduced after the original table was
        created. SQLAlchemy's `create_all` is a no-op when the table
        already exists, so model additions don't propagate to existing
        DBs. We don't ship Alembic; instead, every additive column gets
        a defensive `ADD COLUMN IF NOT EXISTS` here.

        Keep this list tight. If schema churn outgrows it, add Alembic.
        """
        from sqlalchemy import inspect, text
        # SQLite uses DATETIME, PostgreSQL needs TIMESTAMP. Both
        # SQLAlchemy DateTime columns map to either with the same
        # python-side semantics; we just have to emit the right
        # token in raw ALTER. Without this, the migration on a
        # PostgreSQL-backed validator failed silently (logged as a
        # warning) and subsequent `inflight_at` writes 500.
        dialect = self.engine.dialect.name
        datetime_col_type = (
            "TIMESTAMP" if dialect.startswith("postgres") else "DATETIME"
        )
        inspector = inspect(self.engine)
        # Tracks canary receipts that were already in progress at epoch close.
        try:
            cols = {c["name"] for c in inspector.get_columns("canary_records")}
            if "inflight_at" not in cols:
                with self.engine.begin() as conn:
                    conn.execute(text(
                        f"ALTER TABLE canary_records ADD COLUMN inflight_at {datetime_col_type}"
                    ))
                bt.logging.info(
                    f"Migrated canary_records: added inflight_at ({datetime_col_type})"
                )
        except Exception as e:
            bt.logging.warning(f"inline migration (canary_records.inflight_at): {e}")

        # proof_results timing metadata — added 2026-05-24 after the
        # proof-arrival anomaly scanner falsely compared rented micro
        # proofs against free heavy proofs.
        try:
            cols = {c["name"] for c in inspector.get_columns("proof_results")}
            with self.engine.begin() as conn:
                if "matrix_dim" not in cols:
                    conn.execute(text(
                        "ALTER TABLE proof_results ADD COLUMN matrix_dim INTEGER DEFAULT 0"
                    ))
                    bt.logging.info("Migrated proof_results: added matrix_dim")
                if "allocation_state" not in cols:
                    conn.execute(text(
                        "ALTER TABLE proof_results ADD COLUMN allocation_state VARCHAR(16) DEFAULT ''"
                    ))
                    bt.logging.info("Migrated proof_results: added allocation_state")
                if "wall_clock_s" not in cols:
                    conn.execute(text(
                        "ALTER TABLE proof_results ADD COLUMN wall_clock_s FLOAT DEFAULT 0"
                    ))
                    bt.logging.info("Migrated proof_results: added wall_clock_s")
                if "receipt_pass0_spread_s" not in cols:
                    conn.execute(text(
                        "ALTER TABLE proof_results ADD COLUMN receipt_pass0_spread_s FLOAT DEFAULT 0"
                    ))
                    bt.logging.info("Migrated proof_results: added receipt_pass0_spread_s")
                if "receipt_pass1_spread_s" not in cols:
                    conn.execute(text(
                        "ALTER TABLE proof_results ADD COLUMN receipt_pass1_spread_s FLOAT DEFAULT 0"
                    ))
                    bt.logging.info("Migrated proof_results: added receipt_pass1_spread_s")
                if "receipt_pass0_wall_s" not in cols:
                    conn.execute(text(
                        "ALTER TABLE proof_results ADD COLUMN receipt_pass0_wall_s FLOAT DEFAULT 0"
                    ))
                    bt.logging.info("Migrated proof_results: added receipt_pass0_wall_s")
                if "receipt_pass1_wall_s" not in cols:
                    conn.execute(text(
                        "ALTER TABLE proof_results ADD COLUMN receipt_pass1_wall_s FLOAT DEFAULT 0"
                    ))
                    bt.logging.info("Migrated proof_results: added receipt_pass1_wall_s")
        except Exception as e:
            bt.logging.warning(f"inline migration (proof_results timing metadata): {e}")

        # rentals release retry metadata — added 2026-05-26. A transient
        # markAvailable failure must not make the validator forget an active
        # chain lock. These fields keep that retry state durable.
        try:
            cols = {c["name"] for c in inspector.get_columns("rentals")}
            with self.engine.begin() as conn:
                if "release_reason" not in cols:
                    conn.execute(text(
                        "ALTER TABLE rentals ADD COLUMN release_reason VARCHAR(32) DEFAULT ''"
                    ))
                    bt.logging.info("Migrated rentals: added release_reason")
                if "release_requested_at" not in cols:
                    conn.execute(text(
                        f"ALTER TABLE rentals ADD COLUMN release_requested_at {datetime_col_type}"
                    ))
                    bt.logging.info(
                        f"Migrated rentals: added release_requested_at ({datetime_col_type})"
                    )
                if "release_attempts" not in cols:
                    conn.execute(text(
                        "ALTER TABLE rentals ADD COLUMN release_attempts INTEGER DEFAULT 0"
                    ))
                    bt.logging.info("Migrated rentals: added release_attempts")
                if "last_release_error" not in cols:
                    conn.execute(text(
                        "ALTER TABLE rentals ADD COLUMN last_release_error TEXT DEFAULT ''"
                    ))
                    bt.logging.info("Migrated rentals: added last_release_error")
                if "client_request_id" not in cols:
                    conn.execute(text(
                        "ALTER TABLE rentals ADD COLUMN client_request_id VARCHAR(128) DEFAULT ''"
                    ))
                    bt.logging.info("Migrated rentals: added client_request_id")
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_rental_client_request "
                    "ON rentals (client_request_id)"
                ))
        except Exception as e:
            bt.logging.warning(f"inline migration (rentals release retry metadata): {e}")

        # proof_arrival_anomaly used to be classified as hard. It is a
        # statistical audit signal, so downgrade any flags created by
        # older validators; otherwise a historical false-positive keeps
        # banning the executor after this code is deployed.
        try:
            with self.engine.begin() as conn:
                res = conn.execute(text(
                    "UPDATE sybil_flags "
                    "SET severity = 'soft' "
                    "WHERE reason = 'proof_arrival_anomaly' "
                    "AND severity = 'hard'"
                ))
            if getattr(res, "rowcount", 0):
                bt.logging.info(
                    f"Migrated sybil_flags: downgraded {res.rowcount} "
                    "proof_arrival_anomaly flag(s) to soft"
                )
        except Exception as e:
            bt.logging.warning(f"inline migration (sybil_flags proof_arrival_anomaly): {e}")

        # idempotency_keys — added 2026-05-23 for POST /rentals/{id}/extend
        # replay protection. `create_all` above creates it on fresh DBs;
        # this branch is only needed if the table happened to be
        # created without this exact shape (defensive no-op normally).
        try:
            inspector.get_columns("idempotency_keys")
        except Exception:
            # Table doesn't exist — create_all above should have made
            # it, but if a transient driver error happened we'd see
            # this. Surface so the operator notices.
            bt.logging.warning("idempotency_keys table missing after create_all")

    @contextmanager
    def session(self):
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


# ── Helper queries ─────────────────────────────────────────────

def try_claim_idempotency_key(db: Database, key: str, scope: str) -> Optional[str]:
    """Atomic claim of an idempotency key for a write endpoint.

    Returns:
      None        — the caller WON the claim; proceed with the upstream
                    write, then call `attach_idempotency_response`.
      str (json)  — the key was already claimed AND has a cached
                    response. Caller MUST return this verbatim.
      ""          — the key was already claimed but the original
                    request is still in flight (response_json IS NULL).
                    Caller should return 409 to nudge the client to
                    retry after the original lands.

    Concurrency: SQLAlchemy session is wrapped in a single transaction.
    PRIMARY KEY on `key` is the cross-process arbiter — exactly one
    INSERT wins; the loser's SELECT inside the same transaction sees
    the winner's row.
    """
    with db.session() as s:
        # INSERT-if-not-exists. On unique-constraint violation, abort
        # the insert silently — we just need to know if we won.
        existing = s.query(IdempotencyKey).filter_by(key=key).first()
        if existing is not None:
            return existing.response_json if existing.response_json else ""
        s.add(IdempotencyKey(key=key, scope=scope, response_json=None))
        # Commit happens on session exit. After commit, we WON: a
        # racing process that tries to insert the same key gets the
        # PK violation and falls back to the SELECT path on retry.
    return None


def attach_idempotency_response(db: Database, key: str, response_json: str) -> None:
    """Fill in the cached response for a previously-claimed key. Safe
    to call repeatedly; only the first non-empty payload is recorded."""
    with db.session() as s:
        row = s.query(IdempotencyKey).filter_by(key=key).first()
        if row is None:
            return
        if not row.response_json:
            row.response_json = response_json


def upsert_executor(db: Database, executor_id: str, **kwargs):
    """Insert or update executor identity record."""
    with db.session() as s:
        ex = s.query(Executor).filter_by(executor_id=executor_id).first()
        if ex is None:
            ex = Executor(executor_id=executor_id, **kwargs)
            s.add(ex)
        else:
            for k, v in kwargs.items():
                # Empty heartbeat endpoint means "unknown", not "clear it".
                if k == "endpoint" and not v:
                    continue
                setattr(ex, k, v)
            ex.last_seen = datetime.utcnow()


def upsert_executor_hardware(db: Database, executor_id: str, **kwargs):
    """Insert or update static hardware fingerprint."""
    with db.session() as s:
        hw = s.query(ExecutorHardware).filter_by(executor_id=executor_id).first()
        if hw is None:
            hw = ExecutorHardware(executor_id=executor_id, **kwargs)
            s.add(hw)
        else:
            for k, v in kwargs.items():
                setattr(hw, k, v)
            hw.updated_at = datetime.utcnow()


def upsert_executor_stats(db: Database, executor_id: str, **kwargs):
    """Insert or update executor stats (live metrics + scoring)."""
    with db.session() as s:
        stats = s.query(ExecutorStats).filter_by(executor_id=executor_id).first()
        if stats is None:
            stats = ExecutorStats(executor_id=executor_id, **kwargs)
            s.add(stats)
        else:
            for k, v in kwargs.items():
                # Empty heartbeat endpoint means "unknown", not "clear it".
                if k == "endpoint" and not v:
                    continue
                setattr(stats, k, v)
            stats.updated_at = datetime.utcnow()


def store_proof_result(db: Database, executor_id: str, epoch_id: int,
                       valid: bool, tier: str, reason: str = "",
                       delta: float = 0, compute_time: float = 0,
                       verify_ms: float = 0,
                       recv_time: float = 0,
                       arrival_latency_s: float = 0,
                       external_pass_delta_s: float = 0,
                       wall_clock_s: float | None = None,
                       receipt_pass0_spread_s: float = 0,
                       receipt_pass1_spread_s: float = 0,
                       receipt_pass0_wall_s: float = 0,
                       receipt_pass1_wall_s: float = 0,
                       matrix_dim: int = 0,
                       allocation_state: str = ""):
    """Store a proof verification result."""
    with db.session() as s:
        existing = s.query(ProofResult).filter_by(
            executor_id=executor_id, epoch_id=epoch_id,
        ).first()
        if existing:
            existing.valid = valid
            existing.tier = tier
            existing.reason = reason
            existing.delta_seconds = delta
            existing.compute_time = compute_time
            existing.verification_time_ms = verify_ms
            existing.verified_at = datetime.utcnow()
            if recv_time:
                existing.recv_time = recv_time
            if arrival_latency_s:
                existing.arrival_latency_s = arrival_latency_s
            if external_pass_delta_s:
                existing.external_pass_delta_s = external_pass_delta_s
            if wall_clock_s is not None:
                existing.wall_clock_s = float(wall_clock_s or 0)
            existing.receipt_pass0_spread_s = float(receipt_pass0_spread_s or 0)
            existing.receipt_pass1_spread_s = float(receipt_pass1_spread_s or 0)
            existing.receipt_pass0_wall_s = float(receipt_pass0_wall_s or 0)
            existing.receipt_pass1_wall_s = float(receipt_pass1_wall_s or 0)
            if matrix_dim:
                existing.matrix_dim = int(matrix_dim)
            if allocation_state:
                existing.allocation_state = allocation_state[:16]
        else:
            s.add(ProofResult(
                executor_id=executor_id, epoch_id=epoch_id,
                valid=valid, tier=tier, reason=reason,
                delta_seconds=delta, compute_time=compute_time,
                verification_time_ms=verify_ms,
                recv_time=recv_time, arrival_latency_s=arrival_latency_s,
                external_pass_delta_s=external_pass_delta_s,
                wall_clock_s=float(wall_clock_s or 0),
                receipt_pass0_spread_s=float(receipt_pass0_spread_s or 0),
                receipt_pass1_spread_s=float(receipt_pass1_spread_s or 0),
                receipt_pass0_wall_s=float(receipt_pass0_wall_s or 0),
                receipt_pass1_wall_s=float(receipt_pass1_wall_s or 0),
                matrix_dim=int(matrix_dim or 0),
                allocation_state=(allocation_state or "")[:16],
            ))


def _clean_receipt_hex(value: str) -> str:
    return str(value or "").lower().removeprefix("0x")[:128]


def store_proof_receipt(
    db: Database,
    *,
    executor_id: str,
    epoch_id: int,
    gpu_index: int,
    u: int,
    recv_time: float,
    root: str,
    t_commit: str,
    hotkey_ss58: str = "",
) -> None:
    """Persist a timing receipt, keeping the earliest matching arrival time."""
    executor_id = str(executor_id or "").removeprefix("0x")[:64]
    root = _clean_receipt_hex(root)
    t_commit = _clean_receipt_hex(t_commit)
    if not executor_id or not root or not t_commit:
        return
    with db.session() as s:
        existing = s.query(ProofReceipt).filter_by(
            executor_id=executor_id,
            epoch_id=int(epoch_id),
            gpu_index=int(gpu_index),
            u=int(u),
            root=root,
            t_commit=t_commit,
        ).first()
        if existing:
            if float(recv_time or 0) > 0 and (
                not existing.recv_time or float(recv_time) < float(existing.recv_time)
            ):
                existing.recv_time = float(recv_time)
            if hotkey_ss58 and not existing.hotkey_ss58:
                existing.hotkey_ss58 = str(hotkey_ss58)[:48]
            return
        s.add(ProofReceipt(
            executor_id=executor_id,
            epoch_id=int(epoch_id),
            gpu_index=int(gpu_index),
            u=int(u),
            recv_time=float(recv_time or 0),
            root=root,
            t_commit=t_commit,
            hotkey_ss58=str(hotkey_ss58 or "")[:48],
        ))


def _empty_bound_receipt_timing_stats() -> dict[str, float]:
    return {
        "count": 0.0,
        "mean_delta": 0.0,
        "min_delta": 0.0,
        "max_delta": 0.0,
        "pass0_spread": 0.0,
        "pass1_spread": 0.0,
        "min_pass0_recv": 0.0,
        "max_pass0_recv": 0.0,
        "min_pass1_recv": 0.0,
        "max_pass1_recv": 0.0,
    }


def _receipt_stats_from_pairs(
    pairs: list[tuple[float, float]],
) -> dict[str, float]:
    if not pairs:
        return _empty_bound_receipt_timing_stats()
    deltas = [t1 - t0 for t0, t1 in pairs if t0 > 0 and t1 > t0]
    if not deltas:
        return _empty_bound_receipt_timing_stats()
    pass0_times = [t0 for t0, t1 in pairs if t0 > 0 and t1 > t0]
    pass1_times = [t1 for t0, t1 in pairs if t0 > 0 and t1 > t0]
    return {
        "count": float(len(deltas)),
        "mean_delta": sum(deltas) / len(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "pass0_spread": (
            max(pass0_times) - min(pass0_times) if len(pass0_times) > 1 else 0.0
        ),
        "pass1_spread": (
            max(pass1_times) - min(pass1_times) if len(pass1_times) > 1 else 0.0
        ),
        "min_pass0_recv": min(pass0_times),
        "max_pass0_recv": max(pass0_times),
        "min_pass1_recv": min(pass1_times),
        "max_pass1_recv": max(pass1_times),
    }


def get_bound_proof_receipt_timing_stats(
    db: Database,
    recipe_data: dict,
) -> dict[str, float]:
    """Return receipt timing stats from durable rows bound to a recipe.

    A row only counts when executor, epoch, GPU index, pass number, Merkle root,
    T_commit, and, when available, the final recipe signer hotkey all match.
    """
    executor_id = str(recipe_data.get("executor_id") or "").removeprefix("0x")
    epoch_id = int(recipe_data.get("epoch_id") or 0)
    if not executor_id or not epoch_id:
        return _empty_bound_receipt_timing_stats()
    allowed_hotkey = str(recipe_data.get("_nodexo_hotkey_ss58") or "")
    with db.session() as s:
        rows = s.query(ProofReceipt).filter(
            ProofReceipt.executor_id == executor_id,
            ProofReceipt.epoch_id == epoch_id,
        ).all()
    by_key: dict[tuple[int, int, str, str], tuple[float, str]] = {}
    for r in rows:
        hotkey = str(r.hotkey_ss58 or "")
        if allowed_hotkey and hotkey and hotkey != allowed_hotkey:
            continue
        key = (
            int(r.gpu_index or 0),
            int(r.u or 0),
            _clean_receipt_hex(r.root),
            _clean_receipt_hex(r.t_commit),
        )
        recv_time = float(r.recv_time or 0)
        previous = by_key.get(key)
        if recv_time > 0 and (previous is None or recv_time < previous[0]):
            by_key[key] = (recv_time, hotkey)

    pairs: list[tuple[float, float]] = []
    for gp in recipe_data.get("gpu_proofs") or []:
        try:
            gpu_index = int(gp.get("gpu_index"))
        except Exception:
            continue
        p0 = gp.get("pass_0") or {}
        p1 = gp.get("pass_1") or {}
        root0 = _clean_receipt_hex(p0.get("merkle_root") or "")
        root1 = _clean_receipt_hex(p1.get("merkle_root") or "")
        tc0 = _clean_receipt_hex(gp.get("T_commit_0") or "")
        tc1 = _clean_receipt_hex(gp.get("T_commit_1") or "")
        r0 = by_key.get((gpu_index, 0, root0, tc0))
        r1 = by_key.get((gpu_index, 1, root1, tc1))
        if r0 and r1:
            pairs.append((float(r0[0]), float(r1[0])))
    return _receipt_stats_from_pairs(pairs)


def _dt_to_unix_utc(dt: Optional[datetime]) -> float:
    if dt is None:
        return 0.0
    import calendar
    return float(calendar.timegm(dt.utctimetuple()))


def _unix_to_dt_utc(ts: float) -> datetime:
    return datetime.utcfromtimestamp(float(ts))


def record_validator_startup(
    db: Database,
    *,
    service_name: str = "primary",
    instance_id: str = "",
    ready_ts: float | None = None,
    stale_after_s: float = 45.0,
    max_outage_s: float = 24 * 3600,
) -> dict:
    """Mark validator ready and reconstruct the prior unavailable window.

    The validator cannot write while it is down. On the next startup we compare
    the previous heartbeat to the current ready time and persist the gap as a
    validator outage. Scoring uses these windows to neutralize proof/heartbeat
    recency during validator-side downtime.
    """
    ready_ts = float(ready_ts or __import__("time").time())
    ready_at = _unix_to_dt_utc(ready_ts)
    stale_after_s = max(1.0, float(stale_after_s or 1.0))
    max_outage_s = max(stale_after_s, float(max_outage_s or stale_after_s))
    created_outage = False
    outage_duration_s = 0.0
    with db.session() as s:
        row = s.query(ValidatorLiveness).filter_by(
            service_name=service_name,
        ).first()
        if row is not None and row.last_heartbeat_at is not None:
            last_ts = _dt_to_unix_utc(row.last_heartbeat_at)
            gap = ready_ts - last_ts
            if stale_after_s <= gap <= max_outage_s:
                started_at = row.last_heartbeat_at
                # Avoid duplicate outage rows when a process is restarted twice
                # quickly before heartbeat state advances.
                existing = s.query(ValidatorOutage).filter(
                    ValidatorOutage.service_name == service_name,
                    ValidatorOutage.started_at == started_at,
                    ValidatorOutage.ended_at == ready_at,
                ).first()
                if existing is None:
                    s.add(ValidatorOutage(
                        service_name=service_name,
                        started_at=started_at,
                        ended_at=ready_at,
                        duration_s=gap,
                        reason="validator_process_unavailable",
                    ))
                    created_outage = True
                    outage_duration_s = gap
        if row is None:
            row = ValidatorLiveness(
                service_name=service_name,
                instance_id=instance_id,
                started_at=ready_at,
                ready_at=ready_at,
                last_heartbeat_at=ready_at,
            )
            s.add(row)
        else:
            row.instance_id = instance_id
            row.started_at = ready_at
            row.ready_at = ready_at
            row.last_heartbeat_at = ready_at
            row.shutdown_at = None
    return {
        "created_outage": created_outage,
        "duration_s": outage_duration_s,
        "service_name": service_name,
    }


def record_validator_heartbeat(
    db: Database,
    *,
    service_name: str = "primary",
    instance_id: str = "",
    now_ts: float | None = None,
) -> None:
    now_ts = float(now_ts or __import__("time").time())
    now = _unix_to_dt_utc(now_ts)
    with db.session() as s:
        row = s.query(ValidatorLiveness).filter_by(
            service_name=service_name,
        ).first()
        if row is None:
            s.add(ValidatorLiveness(
                service_name=service_name,
                instance_id=instance_id,
                started_at=now,
                ready_at=now,
                last_heartbeat_at=now,
            ))
        else:
            row.instance_id = instance_id
            row.last_heartbeat_at = now


def record_validator_shutdown(
    db: Database,
    *,
    service_name: str = "primary",
    now_ts: float | None = None,
) -> None:
    now = _unix_to_dt_utc(float(now_ts or __import__("time").time()))
    with db.session() as s:
        row = s.query(ValidatorLiveness).filter_by(
            service_name=service_name,
        ).first()
        if row is not None:
            row.shutdown_at = now
            row.last_heartbeat_at = now


def validator_outage_overlap_seconds(
    db: Database,
    start_ts: float,
    end_ts: float,
    *,
    service_name: str = "primary",
) -> float:
    """Return total persisted validator-outage overlap with [start, end]."""
    start_ts = float(start_ts or 0.0)
    end_ts = float(end_ts or 0.0)
    if start_ts <= 0 or end_ts <= start_ts:
        return 0.0
    start_dt = _unix_to_dt_utc(start_ts)
    end_dt = _unix_to_dt_utc(end_ts)
    total = 0.0
    with db.session() as s:
        rows = s.query(ValidatorOutage).filter(
            ValidatorOutage.service_name == service_name,
            ValidatorOutage.started_at < end_dt,
            ValidatorOutage.ended_at > start_dt,
        ).all()
        for row in rows:
            a = max(start_ts, _dt_to_unix_utc(row.started_at))
            b = min(end_ts, _dt_to_unix_utc(row.ended_at))
            if b > a:
                total += b - a
    return max(0.0, total)


def validator_outage_overlaps(
    db: Database,
    start_ts: float,
    end_ts: float,
    *,
    service_name: str = "primary",
) -> bool:
    return validator_outage_overlap_seconds(
        db, start_ts, end_ts, service_name=service_name,
    ) > 0.0


def get_recent_proof_timing_samples(db: Database, executor_id: str,
                                    limit: int = 80) -> list[dict]:
    """Latest valid proof timing samples with workload metadata.

    Reverse-chronological. The sybil scanner uses `matrix_dim` to keep
    rented micro proofs and free heavy proofs in separate timing
    baselines.
    """
    with db.session() as s:
        rows = s.query(ProofResult).filter(
            ProofResult.executor_id == executor_id,
            ProofResult.valid == True,                  # noqa: E712
            ProofResult.tier != "skipped",
        ).order_by(
            ProofResult.verified_at.desc()
        ).limit(limit).all()
        return [{
            "epoch_id": int(r.epoch_id or 0),
            "tier": r.tier or "",
            "matrix_dim": int(r.matrix_dim or 0),
            "allocation_state": r.allocation_state or "",
            "arrival_latency_s": float(r.arrival_latency_s or 0),
            "external_pass_delta_s": float(r.external_pass_delta_s or 0),
            "wall_clock_s": float(r.wall_clock_s or 0),
            "receipt_pass0_spread_s": float(r.receipt_pass0_spread_s or 0),
            "receipt_pass1_spread_s": float(r.receipt_pass1_spread_s or 0),
            "receipt_pass0_wall_s": float(r.receipt_pass0_wall_s or 0),
            "receipt_pass1_wall_s": float(r.receipt_pass1_wall_s or 0),
        } for r in rows]


def get_recent_external_pass_deltas(db: Database, executor_id: str,
                                      limit: int = 30) -> list[float]:
    """Return latest external_pass_delta_s values for an executor
    (valid proofs only, reverse chronological). This is the cleaner
    of the two external timing metrics — network jitter cancels
    because both pass receipts traverse the same path."""
    with db.session() as s:
        rows = s.query(ProofResult).filter(
            ProofResult.executor_id == executor_id,
            ProofResult.valid == True,                  # noqa: E712
            ProofResult.tier != "skipped",
            ProofResult.external_pass_delta_s > 0,
        ).order_by(
            ProofResult.verified_at.desc()
        ).limit(limit).all()
        return [float(r.external_pass_delta_s) for r in rows]


def get_recent_arrival_latencies(db: Database, executor_id: str,
                                  limit: int = 30) -> list[float]:
    """Return the latest `limit` arrival_latency_s values for an
    executor (only valid proofs, in reverse-chronological order).
    The sybil scanner's `proof_arrival_anomaly` uses this to compute
    rolling baseline (median, MAD) and per-cycle z-scores.

    Only valid proofs are included — a failed proof skews the timing
    metric because failures often involve retries / partial work.
    """
    with db.session() as s:
        rows = s.query(ProofResult).filter(
            ProofResult.executor_id == executor_id,
            ProofResult.valid == True,                  # noqa: E712
            ProofResult.tier != "skipped",
            ProofResult.arrival_latency_s > 0,
        ).order_by(
            ProofResult.verified_at.desc()
        ).limit(limit).all()
        return [float(r.arrival_latency_s) for r in rows]


def update_executor_on_proof(db: Database, executor_id: str, hotkey_ss58: str = "",
                              epoch_id: int = 0, valid: bool = False):
    """Update executor identity + stats when a proof is received.

    Called after each verification to keep the executor DB fresh.
    This is the passive liveness signal: miners push proofs/heartbeats, validator records them.
    """
    now = datetime.utcnow()

    # Upsert executor identity
    upsert_executor(db, executor_id, hotkey_ss58=hotkey_ss58, last_seen=now, is_active=True)

    # Update stats
    with db.session() as s:
        stats = s.query(ExecutorStats).filter_by(executor_id=executor_id).first()
        if stats is None:
            stats = ExecutorStats(executor_id=executor_id, hotkey_ss58=hotkey_ss58,
                                  first_seen=now, updated_at=now)
            s.add(stats)

        stats.last_activity_at = now
        stats.last_epoch_id = epoch_id
        stats.hotkey_ss58 = hotkey_ss58 or stats.hotkey_ss58
        stats.updated_at = now

        if valid:
            stats.last_success_at = now
            stats.current_status = "ok"
            stats.current_reason = ""
        else:
            stats.last_fail_at = now
            stats.current_status = "fail"


def update_executor_on_heartbeat(db: Database, executor_id: str,
                                   hw_static: dict = None, hw_live: dict = None,
                                   hotkey_ss58: str = "", endpoint: str = ""):
    """Update executor records from a heartbeat push.

    hw_static: static hardware fingerprint (CPU, RAM, GPU, disk)
    hw_live: live metrics (CPU usage, RAM usage, GPU temp, etc.)
    """
    import json as _json
    now = datetime.utcnow()

    # Upsert executor identity
    upsert_executor(db, executor_id, hotkey_ss58=hotkey_ss58, endpoint=endpoint,
                     last_seen=now, is_active=True)

    # Static hardware
    if hw_static:
        gpus = hw_static.get("gpus", [])
        primary_gpu = gpus[0] if gpus else {}
        storage = (hw_static.get("storage") or {}).get("docker") or {}
        disk = hw_static.get("disk") or {}
        disk_total_bytes = int(storage.get("total_bytes") or disk.get("total_bytes") or 0)
        primary_disk_type = (
            storage.get("kind")
            or ("network" if storage.get("is_network_fs") else "")
            or disk.get("type")
            or ""
        )
        upsert_executor_hardware(db, executor_id,
            system_uuid=hw_static.get("system_uuid", ""),
            cpu_model=hw_static.get("cpu", {}).get("model", ""),
            cpu_arch=hw_static.get("cpu", {}).get("arch", ""),
            cpu_cores_physical=hw_static.get("cpu", {}).get("cores_physical", 0),
            cpu_cores_logical=hw_static.get("cpu", {}).get("cores_logical", 0),
            cpu_freq_max_mhz=hw_static.get("cpu", {}).get("freq_max_mhz", 0),
            ram_total_bytes=hw_static.get("ram", {}).get("total_bytes", 0),
            gpu_count=len(gpus),
            primary_gpu_name=primary_gpu.get("name", ""),
            primary_gpu_uuid=primary_gpu.get("uuid", ""),
            primary_gpu_vram_bytes=primary_gpu.get("vram_bytes", 0),
            gpu_vram_total_bytes=sum(g.get("vram_bytes", 0) for g in gpus),
            gpu_uuids_json=_json.dumps([g.get("uuid", "") for g in gpus]),
            driver_version=primary_gpu.get("driver_version", ""),
            disk_count=1 if disk_total_bytes else 0,
            disk_total_bytes=disk_total_bytes,
            primary_disk_type=primary_disk_type,
            sysbox_detected=hw_static.get("sysbox_detected", False),
            tee_platform=hw_static.get("tee_platform", ""),
            software_version=hw_static.get("software_version", ""),
            hw_static_json=_json.dumps(hw_static),
        )

    # Live metrics
    if hw_live:
        storage_live = (hw_live.get("storage") or {}).get("docker") or {}
        disk_used_bytes = int(storage_live.get("used_bytes") or hw_live.get("disk_used_bytes") or 0)
        disk_total_bytes = int(storage_live.get("total_bytes") or hw_live.get("disk_total_bytes") or 0)
        upsert_executor_stats(db, executor_id,
            hotkey_ss58=hotkey_ss58,
            endpoint=endpoint,
            cpu_usage_pct=hw_live.get("cpu_pct", 0),
            ram_used_bytes=hw_live.get("ram_used_bytes", 0),
            ram_total_bytes=hw_live.get("ram_total_bytes", 0),
            swap_used_bytes=hw_live.get("swap_used_bytes", 0),
            swap_total_bytes=hw_live.get("swap_total_bytes", 0),
            gpu_usage_pct=hw_live.get("gpu_pct", 0),
            gpu_vram_used_bytes=hw_live.get("gpu_vram_used_bytes", 0),
            gpu_vram_total_bytes=hw_live.get("gpu_vram_total_bytes", 0),
            gpu_temp_c=hw_live.get("gpu_temp_c", 0),
            gpu_power_w=hw_live.get("gpu_power_w", 0),
            disk_used_bytes=disk_used_bytes,
            disk_total_bytes=disk_total_bytes,
            net_rx_bytes=hw_live.get("net_rx_bytes", 0),
            net_tx_bytes=hw_live.get("net_tx_bytes", 0),
            hw_live_json=_json.dumps(hw_live),
            last_activity_at=now,
        )


def get_all_executor_stats(db: Database) -> list[dict]:
    """Get all executor stats as dicts (for API responses)."""
    with db.session() as s:
        stats = s.query(ExecutorStats).all()
        results = []
        for st in stats:
            d = {c.name: getattr(st, c.name) for c in st.__table__.columns}
            # Serialize datetimes
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat() if v else None
            results.append(d)
        return results


def get_all_executor_hardware(db: Database) -> dict:
    """Map of executor_id → hardware dict (cpu_cores, ram_bytes, disk_bytes …).

    Used by /marketplace/tiers and /rent to enrich responses with host
    specs without re-querying the chain or the executor's HTTP API.
    """
    import json as _json

    with db.session() as s:
        out: dict = {}
        stats_by_id = {st.executor_id: st for st in s.query(ExecutorStats).all()}
        for hw in s.query(ExecutorHardware).all():
            raw_static = {}
            if hw.hw_static_json:
                try:
                    raw_static = _json.loads(hw.hw_static_json)
                except Exception:
                    raw_static = {}
            raw_live = {}
            st = stats_by_id.get(hw.executor_id)
            if st and st.hw_live_json:
                try:
                    raw_live = _json.loads(st.hw_live_json)
                except Exception:
                    raw_live = {}
            static_storage = (raw_static.get("storage") or {}).get("docker") or {}
            live_storage = (raw_live.get("storage") or {}).get("docker") or {}
            disk_total_bytes = int(
                live_storage.get("total_bytes")
                or static_storage.get("total_bytes")
                or hw.disk_total_bytes
                or 0
            )
            disk_free_bytes = int(
                live_storage.get("available_bytes")
                or live_storage.get("free_bytes")
                or raw_live.get("disk_free_bytes")
                or static_storage.get("available_bytes")
                or static_storage.get("free_bytes")
                or 0
            )
            image_report = raw_static.get("container_images") or {}
            cached_images: dict = {}
            for item in image_report.get("cached") or []:
                ref = item.get("image") or item.get("reference")
                if not ref:
                    continue
                cached_images[ref] = {
                    "id": item.get("id") or "",
                    "repo_digests": item.get("repo_digests") or [],
                    "expected_digest": item.get("expected_digest") or "",
                    "digest_verified": bool(item.get("digest_verified")),
                    "size_bytes": int(item.get("size_bytes") or 0),
                }
            out[hw.executor_id] = {
                "cpu_model": hw.cpu_model,
                "cpu_cores": hw.cpu_cores_physical or hw.cpu_cores_logical,
                "ram_gb": (hw.ram_total_bytes or 0) // (1 << 30),
                # Renter-facing storage is usable free Docker storage, not
                # total filesystem size. Container images and existing layers
                # already consume part of the total and reduce what a rental
                # can write.
                "disk_gb": disk_free_bytes // (1 << 30),
                "disk_free_gb": disk_free_bytes // (1 << 30),
                "disk_total_gb": disk_total_bytes // (1 << 30),
                "gpu_name": hw.primary_gpu_name,
                "gpu_count": hw.gpu_count,
                "vram_gb": (hw.gpu_vram_total_bytes or 0) // (1 << 30),
                "driver_version": hw.driver_version,
                "cached_images": cached_images,
                "image_catalog": image_report.get("checked") or [],
                "image_checked_at": image_report.get("checked_at"),
            }
        return out


def get_executor_hardware(db: Database, executor_id: str) -> dict:
    return get_all_executor_hardware(db).get(executor_id, {})


def store_rental(db: Database, rental_id: str, executor_id: str, **kwargs):
    """Insert a Rental row. Raises if rental_id collides — caller is
    responsible for catching and regenerating. PK collision shows up as
    SQLAlchemy IntegrityError; we re-raise so the rent route can retry."""
    with db.session() as s:
        s.add(Rental(rental_id=rental_id, executor_id=executor_id, **kwargs))


def _rental_as_dict(r: Rental) -> dict:
    d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat() if v else None
    return d


def mark_rental_release_pending(db: Database, rental_id: str, reason: str,
                                error: str = "",
                                effective_ttl_seconds: int = 0) -> None:
    """Persist that a rental is waiting for on-chain release retry.

    This state is deliberately not terminal: terminated_at stays NULL and
    get_active_rentals() keeps returning the row so startup/orphan sweepers can
    retry markAvailable instead of losing the chain lock.
    """
    with db.session() as s:
        rental = s.query(Rental).filter_by(rental_id=rental_id).first()
        if not rental or rental.terminated_at is not None:
            return
        now = datetime.utcnow()
        rental.status = "release_pending"
        rental.release_reason = (reason or "user")[:32]
        if rental.release_requested_at is None:
            rental.release_requested_at = now
        rental.release_attempts = int(rental.release_attempts or 0) + 1
        rental.last_release_error = (error or "")[:1000]
        if effective_ttl_seconds > 0:
            current_ttl = int(rental.ttl_seconds or 0)
            if current_ttl <= 0 or effective_ttl_seconds < current_ttl:
                rental.ttl_seconds = int(effective_ttl_seconds)


def terminate_rental(db: Database, rental_id: str, reason: str = "user", end_block: int = 0):
    """Mark a rental as terminated. Computes actual_seconds and total_paid_rao
    from the price snapshot stored at creation time.
    """
    with db.session() as s:
        rental = s.query(Rental).filter_by(rental_id=rental_id).first()
        if not rental:
            return
        now = datetime.utcnow()
        rental.terminated_at = now
        rental.status = f"terminated_{reason}" if reason else "terminated"
        rental.release_reason = ""
        rental.last_release_error = ""
        if rental.created_at:
            rental.actual_seconds = int((now - rental.created_at).total_seconds())
        if end_block:
            rental.end_block = end_block
        # Billing: price_per_gpu_hour_rao × gpu_count × hours_actual.
        if rental.price_per_gpu_hour_rao and rental.gpu_count and rental.actual_seconds:
            rental.total_paid_rao = int(
                rental.price_per_gpu_hour_rao
                * rental.gpu_count
                * rental.actual_seconds
                / 3600
            )


def get_active_rentals(db: Database) -> list[dict]:
    """All rentals still owned by this validator and not terminal.

    Validator startup calls this to rebuild its in-memory _active_rentals
    dict — without it, every restart loses the rental index and on-chain
    is_rented flags get orphaned.
    """
    with db.session() as s:
        rows = s.query(Rental).filter(
            Rental.status.in_(("active", "release_pending")),
            Rental.terminated_at == None,  # noqa: E711
        ).all()
        return [_rental_as_dict(r) for r in rows]


def get_release_pending_rentals(db: Database) -> list[dict]:
    """Rentals waiting for a retryable markAvailable."""
    with db.session() as s:
        rows = s.query(Rental).filter(
            Rental.status == "release_pending",
            Rental.terminated_at == None,  # noqa: E711
        ).all()
        return [_rental_as_dict(r) for r in rows]


def get_requested_rentals_older_than(db: Database, age_seconds: float) -> list[dict]:
    """Rentals stuck in 'requested' state older than `age_seconds`.

    These are pre-flight rows the rent route wrote BEFORE markRented; if
    the validator crashed (or the chain tx hung) between insert and the
    promote-to-active, the row sits forever blocking that executor. The
    reconciler / orphan_sweeper read this to clean them up.
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(seconds=age_seconds)
    with db.session() as s:
        rows = s.query(Rental).filter(
            Rental.status == "requested",
            Rental.terminated_at == None,  # noqa: E711
            Rental.created_at < cutoff,
        ).all()
        return [_rental_as_dict(r) for r in rows]


def get_rental_history(db: Database, since_ts: float = 0, limit: int = 500,
                       renter_pubkey_fp: str = "") -> list[dict]:
    """Historical view of every rental, newest first.

    Optional filters: only since `since_ts` epoch seconds; only for one
    renter fingerprint. Results paginated by `limit`.
    """
    with db.session() as s:
        q = s.query(Rental)
        if since_ts > 0:
            q = q.filter(Rental.created_at >= datetime.utcfromtimestamp(since_ts))
        if renter_pubkey_fp:
            q = q.filter(Rental.renter_pubkey_fp == renter_pubkey_fp)
        rows = q.order_by(Rental.created_at.desc()).limit(limit).all()
        return [_rental_as_dict(r) for r in rows]


def get_rental_by_id(db: Database, rental_id: str) -> dict | None:
    """Direct rental lookup for settlement/recovery paths."""
    with db.session() as s:
        row = s.query(Rental).filter_by(rental_id=rental_id).first()
        return _rental_as_dict(row) if row else None


# ── Persistence helpers for in-memory state (audit C-7 fix) ──

def set_executor_hotkey_binding(db: Database, executor_id: str, hotkey_ss58: str) -> None:
    """Upsert the executor→hotkey binding. First-claim semantics handled
    by the caller; this just records the final state."""
    with db.session() as s:
        row = s.query(ExecutorHotkeyBinding).filter_by(executor_id=executor_id).first()
        if row is None:
            s.add(ExecutorHotkeyBinding(
                executor_id=executor_id, hotkey_ss58=hotkey_ss58,
            ))
        else:
            row.hotkey_ss58 = hotkey_ss58
            row.bound_at = datetime.utcnow()


def get_all_executor_hotkey_bindings(db: Database) -> dict[str, str]:
    with db.session() as s:
        return {r.executor_id: r.hotkey_ss58
                for r in s.query(ExecutorHotkeyBinding).all()}


def set_monitor_pubkey(db: Database, executor_id: str, monitor_ss58: str) -> None:
    with db.session() as s:
        row = s.query(MonitorPubkeyBinding).filter_by(executor_id=executor_id).first()
        if row is None:
            s.add(MonitorPubkeyBinding(
                executor_id=executor_id, monitor_ss58=monitor_ss58,
            ))
        else:
            row.monitor_ss58 = monitor_ss58
            row.bound_at = datetime.utcnow()


def get_all_monitor_pubkeys(db: Database) -> dict[str, str]:
    with db.session() as s:
        return {r.executor_id: r.monitor_ss58
                for r in s.query(MonitorPubkeyBinding).all()}


def add_rental_window(db: Database, rental_id: str, executor_id: str,
                       start_block: int) -> None:
    """Insert a rental window. end_block=0 means still active."""
    with db.session() as s:
        existing = s.query(RentalWindowRow).filter_by(rental_id=rental_id).first()
        if existing is None:
            s.add(RentalWindowRow(
                rental_id=rental_id, executor_id=executor_id,
                start_block=start_block,
            ))


def close_rental_window(db: Database, rental_id: str, end_block: int) -> None:
    with db.session() as s:
        row = s.query(RentalWindowRow).filter_by(rental_id=rental_id).first()
        if row is not None:
            if int(end_block or 0) < int(row.start_block or 0):
                return
            row.end_block = end_block


def get_all_rental_windows(db: Database) -> list[dict]:
    """Return all rental windows (open + closed). Used by rental_state
    at startup to rehydrate the block-anchored timeline.
    """
    with db.session() as s:
        out = []
        for r in s.query(RentalWindowRow).all():
            out.append({
                "rental_id": r.rental_id,
                "executor_id": r.executor_id,
                "start_block": int(r.start_block or 0),
                "end_block": int(r.end_block or 0),
            })
        return out


def get_owned_rental_ids(db: Database) -> set[str]:
    """Rental IDs created by this validator's API.

    `rental_windows` can also contain synthetic windows reconstructed from
    ComputeRegistry events. Those are useful for historical proof-mode lookup
    but must never be treated as locally-owned active rentals.
    """
    with db.session() as s:
        return {row[0] for row in s.query(Rental.rental_id).all()}


def get_chain_cursor(db: Database, key: str) -> int:
    with db.session() as s:
        row = s.query(ChainCursor).filter_by(key=key).first()
        return int(row.block_number or 0) if row else 0


def set_chain_cursor(db: Database, key: str, block_number: int) -> None:
    with db.session() as s:
        row = s.query(ChainCursor).filter_by(key=key).first()
        if row is None:
            s.add(ChainCursor(
                key=key,
                block_number=int(block_number),
                updated_at=datetime.utcnow(),
            ))
        else:
            row.block_number = int(block_number)
            row.updated_at = datetime.utcnow()


def get_chain_cursor_status(db: Database, key: str) -> dict:
    """Return cursor block + update time for freshness checks."""
    with db.session() as s:
        row = s.query(ChainCursor).filter_by(key=key).first()
        if row is None:
            return {"block_number": 0, "updated_at": None}
        return {
            "block_number": int(row.block_number or 0),
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }


def _registry_event_is_stale(row: RegistryExecutorState, block: int, log_index: int) -> bool:
    """True when an event is older than the row we already applied."""
    if block <= 0:
        return False
    current_block = int(row.last_event_block or 0)
    current_log = int(row.last_event_log_index or 0)
    return current_block > block or (current_block == block and current_log > log_index)


def upsert_registry_executor_state(
    db: Database,
    spec,
    *,
    block_number: int = 0,
    log_index: int = 0,
    event_kind: str = "reconcile",
) -> None:
    """Upsert current ComputeRegistry state from an ExecutorSpec-like object."""
    with db.session() as s:
        row = s.query(RegistryExecutorState).filter_by(
            executor_id=spec.executor_id,
        ).first()
        if row is None:
            row = RegistryExecutorState(executor_id=spec.executor_id)
            s.add(row)
        elif _registry_event_is_stale(row, int(block_number or 0), int(log_index or 0)):
            return

        row.miner_address = str(spec.miner_address or "")
        row.endpoint = str(spec.endpoint or "")
        row.gpu_model_hash = str(spec.gpu_model_hash or "").removeprefix("0x")
        row.gpu_count = int(spec.gpu_count or 0)
        row.vram_mb = int(spec.vram_mb or 0)
        row.price_per_gpu_hour = int(spec.price_per_gpu_hour or 0)
        row.expires_at = int(spec.expires_at or 0)
        row.is_active = bool(spec.is_active)
        row.is_rented = bool(spec.is_rented)
        row.is_removed = False
        if spec.miner_uid is not None:
            row.miner_uid = int(spec.miner_uid or 0)
        if spec.miner_registered is not None:
            row.miner_registered = bool(spec.miner_registered)
        if spec.uid_owner_match is not None:
            row.uid_owner_match = bool(spec.uid_owner_match)
        if int(block_number or 0) > 0:
            row.last_event_block = int(block_number)
            row.last_event_log_index = int(log_index or 0)
            row.last_event_kind = (event_kind or "")[:32]
        row.updated_at = datetime.utcnow()


def mark_registry_executor_rental_state(
    db: Database,
    executor_id: str,
    *,
    is_rented: bool,
    block_number: int = 0,
    log_index: int = 0,
    event_kind: str = "",
) -> None:
    """Apply RentalStarted/RentalEnded when no full-state event is available."""
    executor_id = str(executor_id).removeprefix("0x")
    with db.session() as s:
        row = s.query(RegistryExecutorState).filter_by(executor_id=executor_id).first()
        if row is None:
            row = RegistryExecutorState(executor_id=executor_id)
            s.add(row)
        elif _registry_event_is_stale(row, int(block_number or 0), int(log_index or 0)):
            return
        row.is_rented = bool(is_rented)
        if int(block_number or 0) > 0:
            row.last_event_block = int(block_number)
            row.last_event_log_index = int(log_index or 0)
            row.last_event_kind = (event_kind or ("rented" if is_rented else "available"))[:32]
        row.updated_at = datetime.utcnow()


def mark_registry_executor_inactive(
    db: Database,
    executor_id: str,
    *,
    removed: bool = False,
    block_number: int = 0,
    log_index: int = 0,
    event_kind: str = "inactive",
) -> None:
    """Mark an executor inactive or removed in the local registry index."""
    executor_id = str(executor_id).removeprefix("0x")
    with db.session() as s:
        row = s.query(RegistryExecutorState).filter_by(executor_id=executor_id).first()
        if row is None:
            row = RegistryExecutorState(executor_id=executor_id)
            s.add(row)
        elif _registry_event_is_stale(row, int(block_number or 0), int(log_index or 0)):
            return
        row.is_active = False
        row.is_rented = False
        row.is_removed = bool(removed)
        if int(block_number or 0) > 0:
            row.last_event_block = int(block_number)
            row.last_event_log_index = int(log_index or 0)
            row.last_event_kind = (event_kind or "inactive")[:32]
        row.updated_at = datetime.utcnow()


def mark_registry_reconcile_absent_inactive(
    db: Database,
    active_executor_ids: set[str],
    *,
    block_number: int = 0,
) -> int:
    """After a successful full active-list reconciliation, mark missing
    previously-active rows inactive. Returns rows changed.
    """
    normalized = {str(eid).removeprefix("0x") for eid in active_executor_ids}
    with db.session() as s:
        rows = s.query(RegistryExecutorState).filter(
            RegistryExecutorState.is_active == True,    # noqa: E712
            RegistryExecutorState.is_removed == False,  # noqa: E712
        ).all()
        changed = 0
        now = datetime.utcnow()
        for row in rows:
            if row.executor_id in normalized:
                continue
            row.is_active = False
            row.is_rented = False
            if int(block_number or 0) > 0:
                row.last_event_block = int(block_number)
                row.last_event_log_index = 0
                row.last_event_kind = "reconcile_absent"
            row.updated_at = now
            changed += 1
        return changed


def get_registry_executor_specs(
    db: Database,
    *,
    active_only: bool = True,
    unexpired_only: bool = True,
    include_removed: bool = False,
) -> list:
    """Return indexed registry rows as ExecutorSpec objects."""
    import time as _time
    from common.types import ExecutorSpec

    now = int(_time.time())
    with db.session() as s:
        q = s.query(RegistryExecutorState)
        if active_only:
            q = q.filter(RegistryExecutorState.is_active == True)  # noqa: E712
        if not include_removed:
            q = q.filter(RegistryExecutorState.is_removed == False)  # noqa: E712
        if unexpired_only:
            q = q.filter(RegistryExecutorState.expires_at > now)
        rows = q.order_by(RegistryExecutorState.executor_id.asc()).all()
        return [
            ExecutorSpec(
                executor_id=r.executor_id,
                miner_address=r.miner_address or "",
                endpoint=r.endpoint or "",
                gpu_model_hash=r.gpu_model_hash or "",
                gpu_count=int(r.gpu_count or 0),
                vram_mb=int(r.vram_mb or 0),
                price_per_gpu_hour=int(r.price_per_gpu_hour or 0),
                expires_at=int(r.expires_at or 0),
                is_active=bool(r.is_active),
                is_rented=bool(r.is_rented),
                miner_uid=int(r.miner_uid or 0),
                miner_registered=bool(r.miner_registered),
                uid_owner_match=bool(r.uid_owner_match),
            )
            for r in rows
        ]


def registry_executor_state_count(db: Database) -> int:
    with db.session() as s:
        return int(s.query(RegistryExecutorState).count())


def record_monitor_cycle_observation(db: Database, executor_id: str,
                                      cycle_id: int,
                                      expected_proof_seen: Optional[bool],
                                      extra_proofs_seen: Optional[int]) -> None:
    """Upsert one (executor_id, cycle_id) observation. If the monitor
    is in telemetry-only mode (both fields None) we record nothing
    — there's no signal to compare against later."""
    if expected_proof_seen is None and extra_proofs_seen is None:
        return
    with db.session() as s:
        row = s.query(MonitorCycleObservation).filter_by(
            executor_id=executor_id, cycle_id=cycle_id,
        ).first()
        if row is None:
            s.add(MonitorCycleObservation(
                executor_id=executor_id, cycle_id=cycle_id,
                expected_proof_seen=bool(expected_proof_seen),
                extra_proofs_seen=int(extra_proofs_seen or 0),
            ))
        else:
            # OR-merge: a True at any tick wins (proof was seen at
            # some point in the cycle). extra_proofs_seen takes max.
            if expected_proof_seen:
                row.expected_proof_seen = True
            row.extra_proofs_seen = max(
                row.extra_proofs_seen, int(extra_proofs_seen or 0),
            )
            row.observed_at = datetime.utcnow()


def get_monitor_suppression_stats(db: Database, executor_id: str,
                                   window_hours: float = 1.0) -> dict:
    """Compare validator-verified proofs vs monitor-witnessed proofs
    over the recent window.

    CRITICAL: we only count proofs whose cycle has a corresponding
    monitor observation. The monitor may have come online AFTER some
    proofs were verified (e.g., monitor restarted, first deploy),
    and proofs from cycles BEFORE the monitor's first report aren't
    measurable — they shouldn't count as "missed witness" because
    the monitor didn't exist to witness them. Without this gating,
    every honest miner gets falsely flagged immediately after a
    monitor restart.
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    with db.session() as s:
        # All monitor observations in the window define the eligibility set.
        observed = s.query(MonitorCycleObservation).filter(
            MonitorCycleObservation.executor_id == executor_id,
            MonitorCycleObservation.observed_at >= cutoff,
        ).all()
        observed_cycles = {r.cycle_id for r in observed}
        witnessed_cycles = {r.cycle_id for r in observed if r.expected_proof_seen}
        if not observed_cycles:
            return {"observed": 0, "verified": 0, "witnessed": 0, "matched": 0,
                    "verified_unwitnessed": 0,
                    "verified_unwitnessed_cycles": [],
                    "in_observation_window": False}

        # Validator's verified-valid proofs in the window — BUT only
        # for cycles the monitor was alive for.
        valid_cycles = {
            r.epoch_id for r in s.query(ProofResult).filter(
                ProofResult.executor_id == executor_id,
                ProofResult.valid == True,                      # noqa: E712
                ProofResult.tier != "skipped",
                ProofResult.verified_at >= cutoff,
            ).all()
        }
        valid_in_observed = valid_cycles & observed_cycles
        matched = valid_in_observed & witnessed_cycles
        verified_unwitnessed = valid_in_observed - witnessed_cycles
        return {
            "observed": len(observed_cycles),
            "verified": len(valid_in_observed),
            "witnessed": len(witnessed_cycles),
            "matched": len(matched),
            "verified_unwitnessed": len(verified_unwitnessed),
            "verified_unwitnessed_cycles": sorted(verified_unwitnessed),
            "in_observation_window": True,
        }


def get_monitor_recent_witness_stats(db: Database, executor_id: str,
                                      limit: int = 5,
                                      window_hours: float = 2.0) -> dict:
    """Latest validator-verified cycles that also have monitor observations.

    Used as a recovery guard for monitor_proof_suppression. The broad
    one-hour window catches sustained suppression, but after an operator fixes
    monitor configuration we should recover after a short clean streak instead
    of waiting for the whole bad window to age out.
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    limit = max(1, int(limit or 1))
    with db.session() as s:
        proofs = s.query(ProofResult).filter(
            ProofResult.executor_id == executor_id,
            ProofResult.valid == True,  # noqa: E712
            ProofResult.tier != "skipped",
            ProofResult.verified_at >= cutoff,
        ).order_by(ProofResult.verified_at.desc()).limit(limit * 4).all()
        cycles: list[dict] = []
        for proof in proofs:
            obs = s.query(MonitorCycleObservation).filter(
                MonitorCycleObservation.executor_id == executor_id,
                MonitorCycleObservation.cycle_id == proof.epoch_id,
                MonitorCycleObservation.observed_at >= cutoff,
            ).first()
            if obs is None:
                continue
            cycles.append({
                "cycle_id": int(proof.epoch_id),
                "matched": bool(obs.expected_proof_seen),
            })
            if len(cycles) >= limit:
                break
    matched = sum(1 for c in cycles if c["matched"])
    return {
        "verified": len(cycles),
        "matched": matched,
        "all_matched": bool(cycles) and matched == len(cycles),
        "cycles": cycles,
    }


HARD_SYBIL_REASONS = {
    "cross_monitor_pid_overlap",
    "endpoint_ip_dup",  # combined signal: same IP + same GPU UUID
    # An unauthorized docker process is using the rented GPU. The only
    # docker processes legitimately on a rented executor are those
    # inside the rental container(s) — anything else is either a sybil
    # rental container the miner didn't tell us about or other workloads
    # stealing compute. HARD ban.
    "gpu_process_outside_rental",
    # Validator-external statistical signal: recipe arrival latency
    # (recv_time - cycle_start_ts) is consistently elevated above the
    # executor's own rolling baseline. Externally measured, the miner
    # cannot fake this — they can only DELAY their broadcast, never
    # advance it. Catches GPU-contention-induced slowdown (sybil's
    # second executor running concurrent compute on the same chip).
    # Canary-driven: validator self-rents, runs known workload, KNOWS
    # the PID. If monitor's report omits that PID over N canaries →
    # rootkit / NVML lie. Statistical detection of K-1.
    "monitor_suppression",
    # Canary-driven: validator's canary PID appears in ANOTHER
    # executor's monitor — that monitor is on the SAME physical GPU.
    "canary_pid_overlap",
    # VRAM consistency at heartbeat: miner reports vram_bytes for a
    # given GPU model that's materially smaller than the model's spec
    # (e.g., claiming 40 GB on an A6000 to hide 8 GB of sybil). The
    # check is deterministic and unfakeable — the GPU name is on chain,
    # the spec table is public.
    "vram_underreport",
    "vram_overclaim",
    # Canary fill failed to allocate up to target. Either the miner
    # is lying about free VRAM (hiding sybil's allocation) OR a
    # concurrent sybil process is occupying VRAM. Validator measures
    # actual cudaMalloc behavior, miner can't fake.
    "canary_vram_fail",
    # HBM bandwidth measured during fill scan is materially below
    # the GPU model's spec. Indicates a kernel rootkit storing fake
    # allocations in host RAM (PCIe ≈ 32 GB/s vs HBM 700-2000 GB/s).
    "canary_bandwidth_low",
    # Returned checksum / sum_final doesn't match what the validator
    # re-derives from (nonce, device_index, seeded RNG). Indicates
    # the miner intercepted CUDA APIs without faithfully reproducing
    # the byte stream — strong rootkit signal.
    "canary_checksum_mismatch",
    # Canary container did not expose the minimum Docker-storage-backed
    # filesystem capacity or could not complete a small fsync write.
    # This catches misconfigured hosts that advertise rentable capacity
    # but cannot provide a usable rental filesystem.
    "canary_storage_fail",
    # N consecutive canary attempts errored (network unreachable,
    # pip install failed, ssh timeout, etc.) without ever completing
    # with a pass/fail. Indicates deliberate sabotage: a miner who
    # reliably breaks canary execution would keep last_canary_status
    # in {pass, error} forever, bypassing the canary_failed gate
    # while never actually being verified. After N errors in a row,
    # treat as effective failure.
    "canary_error_streak",
    # A validator-owned active rental still existed in DB/chain, but
    # the miner's signed `/containers` view no longer listed the host
    # container for multiple consecutive checks. That is a direct SLA
    # failure against a renter, so the executor is blocked from future
    # rentals/scoring until the validator's probation window expires.
    "rental_container_missing",
    # The expected rental container exists on the miner but remains in
    # an unusable Docker state (`exited`, `dead`, `removing`) across
    # multiple successful validator reads. Docker's restart policy should
    # recover a one-off in-container PID1 kill; sustained stopped state
    # is a miner-side rental SLA failure.
    "rental_container_stopped",
    # The miner endpoint hosting an active validator-owned rental stayed
    # unreachable across multiple signed `/containers` checks. Heartbeat
    # freshness may also drop this executor from public inventory, but
    # active-rental failure is stronger: block score/rentals until
    # probation expires.
    "rental_endpoint_unreachable",
    # The executor disappeared from the fresh active+unexpired chain
    # snapshot during an active validator-owned rental. This covers lease
    # expiry, deregistration/offline state, and other chain-liveness loss.
    "rental_chain_not_live",
    # The validator was actively renting this executor and the proof for
    # that rented window failed after policy grace. The rental is cut
    # immediately and the executor must sit out the probation window
    # before serving another renter.
    "rental_proof_fail",
}
RENTAL_PROBATION_REASONS = {
    "rental_container_missing",
    "rental_container_stopped",
    "rental_endpoint_unreachable",
    "rental_chain_not_live",
    "rental_proof_fail",
}
SOFT_SYBIL_REASONS = {
    "gpu_uuid_dup",      # self-reported, fakeable
    "system_uuid_dup",   # self-reported, fakeable
    "nvml_digest_outlier",
    # Passive monitor availability/identity/process observations are useful
    # for operator review, but the sidecar has repeatedly proven too brittle
    # to be a standalone hard gate across driver resets, container rebuilds,
    # and fast proof workers. Active canary/rental checks remain hard.
    "monitor_silent",
    "monitor_unhealthy",
    "monitor_pubkey_rotation",
    "behavioral_extra_proof",
    # Passive monitor proof-witness misses are an audit signal only.
    # Process sampling can miss short proof workers on fast GPUs and is
    # especially brittle during canary/rent mode transitions. Canary and
    # rental checks remain the hard enforcement path.
    "monitor_proof_suppression",
    # Statistical timing anomaly. Useful audit signal, but not strong
    # enough alone to auto-ban/rental-cancel: legitimate proof-mode
    # transitions and scheduler hiccups can move latency distributions.
    "proof_arrival_anomaly",
}


def open_sybil_flag(db: Database, executor_id: str, reason: str,
                     evidence: dict, severity: Optional[str] = None) -> bool:
    """Insert a new sybil flag. Severity defaults from the reason —
    hard signals ban automatically, soft signals are review-only.
    """
    import json as _json
    if severity is None:
        if reason in HARD_SYBIL_REASONS:
            severity = "hard"
        elif reason in SOFT_SYBIL_REASONS:
            severity = "soft"
        else:
            severity = "hard"  # unknown reason: be conservative
    with db.session() as s:
        existing = s.query(SybilFlag).filter(
            SybilFlag.executor_id == executor_id,
            SybilFlag.reason == reason,
            SybilFlag.cleared_at == None,  # noqa: E711
        ).first()
        if existing is not None:
            return False
        s.add(SybilFlag(
            executor_id=executor_id, reason=reason,
            severity=severity,
            evidence_json=_json.dumps(evidence),
        ))
        return True


def list_open_sybil_flags(db: Database) -> list[dict]:
    """All currently-open sybil flags (cleared_at IS NULL)."""
    with db.session() as s:
        rows = s.query(SybilFlag).filter(
            SybilFlag.cleared_at == None  # noqa: E711
        ).order_by(SybilFlag.flagged_at.desc()).all()
        out = []
        for r in rows:
            d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat() if v else None
            out.append(d)
        return out


def get_banned_executor_ids(db: Database) -> set[str]:
    """Executor IDs with at least one open HARD blocking flag.

    Soft flags (gpu_uuid_dup, system_uuid_dup, nvml_digest_outlier
    — all derived from miner-self-reported data that a smart sybil
    can fake to unique-but-plausible values) DO NOT auto-ban; they
    sit in the table for operator review. Hard flags come from
    independent witnesses (the monitor's signed reports, validator-
    side proof tracking, and active-rental SLA checks) and DO block
    score and future rentals.

    This is the audit C-4 fix: a smart attacker who picks unique
    fake UUIDs can dodge gpu_uuid_dup forever; the real signal is
    the monitor's PID-overlap attestation, which is hard.
    """
    with db.session() as s:
        rows = s.query(SybilFlag.executor_id).filter(
            SybilFlag.cleared_at == None,  # noqa: E711
            SybilFlag.severity == "hard",
        ).distinct().all()
        return {row[0] for row in rows}


def get_public_hidden_executor_ids(db: Database) -> set[str]:
    """Open hard-fraud executor IDs that should disappear from public views.

    Rental probation is different from fraud concealment: it blocks future
    rentals and score, but network/status views can still show the recovered
    executor as audit/probation. Sybil and monitor-fraud flags stay hidden.
    """
    with db.session() as s:
        rows = s.query(SybilFlag.executor_id).filter(
            SybilFlag.cleared_at == None,  # noqa: E711
            SybilFlag.severity == "hard",
            ~SybilFlag.reason.in_(RENTAL_PROBATION_REASONS),
        ).distinct().all()
        return {row[0] for row in rows}


def get_rental_probation_executor_ids(db: Database) -> set[str]:
    """Executor IDs blocked by active-rental SLA probation."""
    with db.session() as s:
        rows = s.query(SybilFlag.executor_id).filter(
            SybilFlag.cleared_at == None,  # noqa: E711
            SybilFlag.severity == "hard",
            SybilFlag.reason.in_(RENTAL_PROBATION_REASONS),
        ).distinct().all()
        return {row[0] for row in rows}


def clear_sybil_flag(db: Database, flag_id: int, cleared_by: str = "admin") -> bool:
    """Mark a flag as resolved. Idempotent; returns True if it
    actually transitioned from open → cleared.
    """
    with db.session() as s:
        row = s.query(SybilFlag).filter_by(id=flag_id).first()
        if row is None or row.cleared_at is not None:
            return False
        row.cleared_at = datetime.utcnow()
        row.cleared_by = cleared_by
        return True


def has_open_sybil_flag(db: Database, executor_id: str, reason: str) -> bool:
    """Whether the (executor_id, reason) pair has an open flag right now."""
    with db.session() as s:
        return s.query(SybilFlag).filter(
            SybilFlag.executor_id == executor_id,
            SybilFlag.reason == reason,
            SybilFlag.cleared_at == None,  # noqa: E711
        ).first() is not None


def clear_open_sybil_flag(db: Database, executor_id: str, reason: str,
                           cleared_by: str = "auto") -> bool:
    """Clear all open flags for (executor_id, reason). Returns True if
    at least one row transitioned. Used by self-healing signals
    (e.g. behavioral_extra_proof clean-streak recovery).
    """
    with db.session() as s:
        rows = s.query(SybilFlag).filter(
            SybilFlag.executor_id == executor_id,
            SybilFlag.reason == reason,
            SybilFlag.cleared_at == None,  # noqa: E711
        ).all()
        now = datetime.utcnow()
        cleared_any = False
        for row in rows:
            row.cleared_at = now
            row.cleared_by = cleared_by
            cleared_any = True
        return cleared_any


def backup_and_prune_table(db: Database, model, date_column, retain_days: int,
                           backup_dir: str = "", retain_backups: bool = False):
    """Export old rows to .jsonl.gz backup, then delete from live DB."""
    import gzip
    import json as _json
    from datetime import timedelta

    if not backup_dir:
        backup_dir = os.path.join(
            os.environ.get("NODEXO_DATA_DIR", os.path.expanduser("~/.nodexo")),
            "backups",
        )
    os.makedirs(backup_dir, exist_ok=True)

    cutoff = datetime.utcnow() - timedelta(days=retain_days)
    table_name = model.__tablename__

    with db.session() as s:
        old_rows = s.query(model).filter(date_column < cutoff).all()
        if not old_rows:
            return 0

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{table_name}_{ts}.jsonl.gz")

        with gzip.open(backup_path, "wt") as f:
            for row in old_rows:
                row_dict = {c.name: getattr(row, c.name) for c in row.__table__.columns}
                f.write(_json.dumps(row_dict, default=str) + "\n")

        deleted = s.query(model).filter(date_column < cutoff).delete()
        bt.logging.info(f"Backup: {deleted} rows from {table_name} → {backup_path}")

    return len(old_rows)


def _backup_and_prune_query(db: Database, model, query_builder, label: str,
                            backup_dir: str = "") -> int:
    """Export rows selected by query_builder(session), then delete them."""
    import gzip
    import json as _json

    if not backup_dir:
        backup_dir = os.path.join(
            os.environ.get("NODEXO_DATA_DIR", os.path.expanduser("~/.nodexo")),
            "backups",
        )
    os.makedirs(backup_dir, exist_ok=True)

    with db.session() as s:
        rows = query_builder(s).all()
        if not rows:
            return 0

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{label}_{ts}.jsonl.gz")

        with gzip.open(backup_path, "wt") as f:
            for row in rows:
                row_dict = {
                    c.name: getattr(row, c.name)
                    for c in row.__table__.columns
                }
                f.write(_json.dumps(row_dict, default=str) + "\n")

        for row in rows:
            s.delete(row)
        bt.logging.info(f"Pruned {len(rows)} rows from {label} → {backup_path}")
        return len(rows)


def prune_validator_db(
    db: Database,
    *,
    backup_dir: str = "",
    proof_retention_days: int | None = None,
    monitor_retention_days: int | None = None,
    rental_window_retention_days: int | None = None,
    sybil_flag_retention_days: int | None = None,
    idempotency_retention_days: int | None = None,
) -> dict[str, int]:
    """Back up and prune high-volume validator rows.

    This intentionally does not prune Rental history or active rental windows.
    Those are accounting/audit state, not high-volume telemetry.
    """
    from datetime import timedelta

    def _days(name: str, default: int, value: int | None) -> int:
        if value is not None:
            return int(value)
        try:
            return int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    proof_days = _days("VALIDATOR_PROOF_RETENTION_DAYS", 14, proof_retention_days)
    monitor_days = _days("VALIDATOR_MONITOR_RETENTION_DAYS", 7, monitor_retention_days)
    window_days = _days("VALIDATOR_RENTAL_WINDOW_RETENTION_DAYS", 90, rental_window_retention_days)
    flag_days = _days("VALIDATOR_SYBIL_FLAG_RETENTION_DAYS", 90, sybil_flag_retention_days)
    idem_days = _days("VALIDATOR_IDEMPOTENCY_RETENTION_DAYS", 30, idempotency_retention_days)

    summary: dict[str, int] = {}

    if proof_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=proof_days)
        summary["proof_results"] = _backup_and_prune_query(
            db,
            ProofResult,
            lambda s: s.query(ProofResult).filter(ProofResult.verified_at < cutoff),
            "proof_results",
            backup_dir,
        )
        summary["proof_receipts"] = _backup_and_prune_query(
            db,
            ProofReceipt,
            lambda s: s.query(ProofReceipt).filter(ProofReceipt.created_at < cutoff),
            "proof_receipts",
            backup_dir,
        )
    if monitor_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=monitor_days)
        summary["monitor_cycle_observations"] = _backup_and_prune_query(
            db,
            MonitorCycleObservation,
            lambda s: s.query(MonitorCycleObservation).filter(
                MonitorCycleObservation.observed_at < cutoff,
            ),
            "monitor_cycle_observations",
            backup_dir,
        )
    if window_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        summary["rental_windows"] = _backup_and_prune_query(
            db,
            RentalWindowRow,
            lambda s: s.query(RentalWindowRow).filter(
                RentalWindowRow.end_block != 0,
                RentalWindowRow.created_at < cutoff,
            ),
            "rental_windows",
            backup_dir,
        )
    if flag_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=flag_days)
        summary["sybil_flags"] = _backup_and_prune_query(
            db,
            SybilFlag,
            lambda s: s.query(SybilFlag).filter(
                SybilFlag.cleared_at != None,  # noqa: E711
                SybilFlag.flagged_at < cutoff,
            ),
            "sybil_flags",
            backup_dir,
        )
    if idem_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=idem_days)
        summary["idempotency_keys"] = _backup_and_prune_query(
            db,
            IdempotencyKey,
            lambda s: s.query(IdempotencyKey).filter(
                IdempotencyKey.created_at < cutoff,
            ),
            "idempotency_keys",
            backup_dir,
        )

    return summary
