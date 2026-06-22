"""
Executor scoring — market-price-proportional, per-endpoint, with hard
eligibility gates and a rolling proof-reliability multiplier.

For each registered executor (one endpoint):

    base    = GPU_HOURLY_PRICE[gpu_model] × gpu_count
    gates   = ALL of:
              (a) executor is active on chain (lease unexpired)
              (b) most recent valid proof within `proof_recency_s`
              (c) heartbeat within `heartbeat_recency_s`
              (d) zero open HARD sybil flags
              (e) last canary status is NOT "fail"
              (f) executor has passed at least one canary (no rentals
                  allowed for never-canaried executors — those score 0
                  too, except they are still EXPECTED to fail this gate
                  until the scheduler canaries them)
    util    = 1.0 if rented; IDLE_FRACTION otherwise
    rel     = rolling proof reliability multiplier, derived from recent
              pass rates once enough samples exist
    score   = 0                    if any hard gate fails
              base × util × rel    otherwise

Per-miner score (the weight set on chain against the miner's hotkey
UID) = sum of executor scores under that hotkey.

WHY THIS SHAPE
==============

- **Market-price proportional**: the validator pays emission in
  proportion to the GPU's market hourly value. An H100 (~$2.50/h)
  earns ~6× a 4090 (~$0.40/h).
- **Per-endpoint, sum-per-miner**: a miner with 4× A100 gets 4×
  the score of a miner with one.
- **Hard gates + reliability multiplier**: identity, recency, heartbeat,
  hard sybil flags, canary, and stake are binary eligibility gates. Proof
  pass rate is a continuous multiplier, so one timing outlier does not
  zero a miner, but repeated failures rapidly reduce weight.
- **Idle stipend (default 20%)**: pure-rented scoring would have
  miners earn zero whenever the marketplace is slow, encouraging
  them to go offline. A small idle stipend keeps inventory online
  while ranking rented miners 5× above idle ones.

ROADMAP — signals defined but not yet wired
============================================

These signals are defined as protocol inputs but are not part of the
current public testnet scoring path. Once the supporting telemetry is
persisted, each one becomes a multiplier on `base × util_factor` (or,
for the binary ones, an additional gate):

  - **uptime_ratio_24h**: heartbeat success rate over the last 24h.
    Multiplier ∈ [0, 1]. Already collectable from the heartbeat route's
    history but we don't store a time-series yet.
  - **port_ok**: validator-initiated TCP probe of the executor's
    advertised endpoint. Binary multiplier (0.5 if fail, 1.0 if pass).
    Catches operators whose firewall isn't actually open.
  - **registration_age_hours**: 48h linear ramp for newly-registered
    executors. Prevents flash-and-run sybils that register, briefly
    score high, and vanish.
  - **tee_attested**: TDX / SEV-SNP attestation verified. Bonus ~1.10
    (10% trust premium) since the GPU + host attest cryptographically.
  - **sysbox_detected**: rental container uses Sysbox runtime (the
    monitor already reports this in `hw_static`). Bonus ~1.05.
  - **collateral_above_min**: on-chain CollateralManager deposit above
    the per-GPU-tier minimum. Hard gate when minCollateral > 0.
  - **history_months**: months of clean operation. Small additive
    bonus (capped, e.g. +0.05 per month, max +0.20). Same-operator
    history transfers across re-registrations via miner_address.
  - **deep_check_consistent**: periodic hardware deep-check (driver
    + libcuda digest match across reboots; nvml_digest_outlier signal
    sustained 0). Small bonus.

When wiring, prefer multipliers over additive bumps so the formula
stays "base price × <multipliers product> × utilization". An executor
with TEE + sysbox + 6 months clean history would earn:
    base × 1.10 × 1.05 × 1.30 × util  ≈  1.50× the bare baseline.

That's still a moderate spread; market-price already drives the
dominant ranking signal.
"""
from __future__ import annotations

import calendar
import os
from dataclasses import dataclass
from typing import Optional

from common.config import canonical_gpu_model_name
from common.subnet_runtime_config import (
    DEFAULT_GPU_HOURLY_PRICE,
    DEFAULT_PRICE_PER_HOUR,
    get_subnet_runtime_config,
    policy_blacklist_reason,
)
from common.proof_timing import is_timing_model_calibrated

# Hourly prices in arbitrary normalised units (1.0 ≈ $1/hr market rate).
# Subnet operator tunes these as the cloud market shifts. Anything not
# listed falls back to DEFAULT_PRICE_PER_HOUR for display/quote helpers, but
# scoring still requires an explicit calibrated timing row.
GPU_HOURLY_PRICE: dict[str, float] = DEFAULT_GPU_HOURLY_PRICE


def lookup_usdc_price(gpu_model: str, gpu_count: int,
                       quote_multiplier: float | None = None) -> float:
    """Authoritative renter-facing USDC price for `gpu_count × gpu_model`
    for one hour. Called from /marketplace/tiers (display) and /rent
    (server-side x402 challenge amount). Never trust the client to
    pass this value; always recompute here.

    The runtime `renter_quote_multiplier` scales all checkout prices
    uniformly for testnet preview while preserving the GPU tier spread.
    Scoring passes quote_multiplier=1.0 so emission score value remains
    based on the unscaled price table.
    """
    runtime = get_subnet_runtime_config()
    canonical_model = canonical_gpu_model_name(gpu_model)
    per_gpu = runtime.pricing.gpu_hourly_price.get(
        canonical_model,
        runtime.pricing.default_price_per_hour,
    )
    multiplier = runtime.pricing.renter_quote_multiplier if quote_multiplier is None else quote_multiplier
    return round(per_gpu * max(gpu_count, 0) * multiplier, 6)


# Gate-window defaults. Production reads from env at config time.
DEFAULT_PROOF_RECENCY_S      = 600       # ~3.3 proof cycles
DEFAULT_HEARTBEAT_RECENCY_S  = 300       # heartbeat is ~60s; allow slop
DEFAULT_IDLE_FRACTION        = 0.20      # 20% of rented score when idle

# Minimum subnet stake (alpha, as reported by metagraph stake) the miner hotkey must hold to be
# eligible for scoring + rental. This is the "skin in the game" lever:
# the registration burn alone is small on testnet and varies on
# mainnet), so a sybil farm can easily afford many registrations. A
# stake floor on top forces the operator to either lock real capital
# per endpoint or accept that low-stake endpoints score zero and
# never see rental traffic. Set to 0 to disable.
DEFAULT_MIN_MINER_STAKE_TAO  = 0.0
DEFAULT_STAKE_PER_SCORE_TAO = 0.0
DEFAULT_STAKE_GPU_COUNT_EXPONENT = 1.0
DEFAULT_STAKE_REQUIREMENT_MULTIPLIER = 1.0


@dataclass
class ScoringContext:
    """Validator-side state for one executor at one moment. All
    DB-derived fields are caller-supplied so the scorer is pure."""
    executor_id: str
    hotkey_ss58: str
    gpu_model: str
    gpu_count: int
    is_rented: bool
    is_active: bool
    last_proof_valid_at: float            # unix ts; 0 if never
    last_heartbeat_at: float              # unix ts; 0 if never
    open_hard_flag_count: int             # from sybil_flags table
    last_canary_status: str               # "pass" | "fail" | "error" | "" (never)
    last_canary_at: float                 # unix ts; 0 if never
    now_ts: float                         # validator clock
    miner_address: str = ""               # EVM address when known
    miner_uid: int | None = None          # metagraph UID when known
    endpoint: str = ""                    # registered executor endpoint
    miner_stake_tao: float = 0.0          # mg.stake[uid] in subnet alpha; 0 if unresolved
    miner_required_stake_tao: float = 0.0 # aggregate hotkey requirement supplied by scoring loop
    pass_rate_1h: float = 1.0             # recent proof reliability
    pass_rate_24h: float = 1.0
    samples_1h: int = 0
    samples_24h: int = 0
    rental_runtime_ok: bool = True
    validator_outage_since_proof_s: float = 0.0
    validator_outage_since_heartbeat_s: float = 0.0


@dataclass
class ScoringConfig:
    proof_recency_s:     int   = DEFAULT_PROOF_RECENCY_S
    heartbeat_recency_s: int   = DEFAULT_HEARTBEAT_RECENCY_S
    idle_fraction:       float = DEFAULT_IDLE_FRACTION
    # When the canary scheduler isn't running on this validator
    # (CANARY_ENABLED unset), `last_canary_at == 0` forever — every
    # executor would score 0. Set canary_required=False to make the
    # canary gate permissive: never-canaried passes (i.e. canary is a
    # "if available, must not fail" check rather than a hard prereq).
    # The validator wires this from os.environ in _scoring_loop.
    canary_required:     bool  = True
    # Minimum miner-side subnet alpha stake for the executor to be eligible.
    # Zeroes score AND blocks /rent. 0.0 disables the gate. See
    # DEFAULT_MIN_MINER_STAKE_TAO for the rationale.
    min_miner_stake_tao: float = DEFAULT_MIN_MINER_STAKE_TAO
    # Optional proportional stake gate. Required stake is aggregated per
    # hotkey across all active executors:
    #   stake_per_score_tao × per-GPU score × gpu_count^exponent × multiplier
    # The validator loop writes the aggregate into ctx.miner_required_stake_tao.
    # If called standalone, score_one computes the per-executor requirement.
    stake_per_score_tao: float = DEFAULT_STAKE_PER_SCORE_TAO
    stake_gpu_count_exponent: float = DEFAULT_STAKE_GPU_COUNT_EXPONENT
    stake_requirement_multiplier: float = DEFAULT_STAKE_REQUIREMENT_MULTIPLIER
    stake_grace_until_ts: float = 0.0
    # Proof reliability is not an instant hard-zero for one outlier.
    # Once enough recent samples exist, quality = min(1h, 24h pass-rate
    # where available), reliability_factor = quality ** exponent. If
    # quality falls below min_reliability_gate, score goes to zero.
    min_reliability_samples: int = 5
    min_reliability_gate: float = 0.67
    reliability_exponent: float = 2.0
    reliability_miss_grace_slots: int = 20
    timing_grace_count: int = 3
    timing_grace_window_s: int = 3600
    timing_grace_consecutive_count: int = 2


def _env_float_override(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except Exception:
        return None


def scoring_config_from_runtime(*, canary_required: bool) -> ScoringConfig:
    """Build scorer config from the latest subnet runtime config.

    Local env can still override `MIN_MINER_STAKE_TAO` as an emergency
    validator-side alpha-stake floor, but normal subnet policy comes from the web config.
    """
    runtime = get_subnet_runtime_config().scoring
    min_stake = _env_float_override("MIN_MINER_STAKE_TAO")
    return ScoringConfig(
        proof_recency_s=runtime.proof_recency_s,
        heartbeat_recency_s=runtime.heartbeat_recency_s,
        idle_fraction=runtime.idle_fraction,
        canary_required=canary_required,
        min_miner_stake_tao=runtime.min_miner_stake_tao if min_stake is None else min_stake,
        stake_per_score_tao=runtime.stake_per_score_tao,
        stake_gpu_count_exponent=runtime.stake_gpu_count_exponent,
        stake_requirement_multiplier=runtime.stake_requirement_multiplier,
        stake_grace_until_ts=runtime.stake_grace_until_ts,
        min_reliability_samples=runtime.min_reliability_samples,
        min_reliability_gate=runtime.min_reliability_gate,
        reliability_exponent=runtime.reliability_exponent,
        reliability_miss_grace_slots=runtime.reliability_miss_grace_slots,
        timing_grace_count=runtime.timing_grace_count,
        timing_grace_window_s=runtime.timing_grace_window_s,
        timing_grace_consecutive_count=runtime.timing_grace_consecutive_count,
    )


@dataclass
class ScoringResult:
    executor_id: str
    hotkey_ss58: str
    base_price: float           # GPU price × count
    util_factor: float          # idle_fraction or 1.0
    score: float                # 0 or base × util
    zero_reason: str            # blank if score > 0; else gates that failed
    gates_passed: list[str]
    gates_failed: list[str]
    reliability_factor: float = 1.0
    miner_stake_alpha: float = 0.0
    executor_required_stake_alpha: float = 0.0
    miner_required_stake_alpha: float = 0.0
    stake_grace_until_ts: float = 0.0
    stake_ok: bool = True
    stake_in_grace: bool = False


def stake_requirement_for_context(ctx: ScoringContext, cfg: Optional[ScoringConfig] = None) -> float:
    """Return this executor's contribution to the miner-hotkey stake floor.

    The requirement intentionally uses un-discounted per-GPU score units, not
    the current renter quote multiplier, so testnet checkout discounts do not
    weaken sybil resistance.
    """
    cfg = cfg or ScoringConfig()
    if cfg.stake_per_score_tao <= 0 or cfg.stake_requirement_multiplier <= 0:
        return 0.0
    gpu_count = max(1, int(ctx.gpu_count or 1))
    per_gpu_score = lookup_usdc_price(ctx.gpu_model, 1, 1.0)
    count_factor = float(gpu_count) ** max(0.0, float(cfg.stake_gpu_count_exponent))
    return max(
        0.0,
        per_gpu_score
        * count_factor
        * float(cfg.stake_per_score_tao)
        * float(cfg.stake_requirement_multiplier),
    )


def effective_min_stake_tao(ctx: ScoringContext, cfg: Optional[ScoringConfig] = None) -> float:
    """Combined flat and proportional miner-hotkey stake requirement."""
    cfg = cfg or ScoringConfig()
    proportional = (
        float(ctx.miner_required_stake_tao or 0.0)
        if ctx.miner_required_stake_tao > 0
        else stake_requirement_for_context(ctx, cfg)
    )
    return max(float(cfg.min_miner_stake_tao or 0.0), proportional)


def score_one(ctx: ScoringContext, cfg: Optional[ScoringConfig] = None) -> ScoringResult:
    """Score one executor. Pure function over (ctx, cfg)."""
    cfg = cfg or ScoringConfig()
    base = lookup_usdc_price(ctx.gpu_model, ctx.gpu_count, 1.0)
    executor_required_stake = stake_requirement_for_context(ctx, cfg)

    passed: list[str] = []
    failed: list[str] = []

    # Gate A — active on chain (lease unexpired)
    if ctx.is_active:
        passed.append("active")
    else:
        failed.append("active")

    # Gate A1 — explicit subnet-operator blacklist policy.
    blacklist_reason = policy_blacklist_reason(
        ctx.executor_id,
        hotkey_ss58=ctx.hotkey_ss58,
        miner_address=ctx.miner_address,
        miner_uid=ctx.miner_uid,
        endpoint=ctx.endpoint,
    )
    if blacklist_reason:
        failed.append(blacklist_reason)
    else:
        passed.append("not_policy_blacklisted")

    # Gate B0 — reviewed proof timing exists for this GPU class/count. Unknown or
    # extrapolated models may be present in the config for calibration planning,
    # but they must not score or enter the renter marketplace.
    if is_timing_model_calibrated(ctx.gpu_model, ctx.gpu_count):
        passed.append("timing_calibrated")
    else:
        failed.append(f"timing_calibrated ({ctx.gpu_model or 'unknown'} × {ctx.gpu_count})")

    # Gate B — recent valid proof
    age_proof_raw = ctx.now_ts - (ctx.last_proof_valid_at or 0)
    age_proof = max(0.0, age_proof_raw - float(ctx.validator_outage_since_proof_s or 0.0))
    if ctx.last_proof_valid_at > 0 and age_proof <= cfg.proof_recency_s:
        passed.append("proof_recent")
    else:
        failed.append(f"proof_recent (last={age_proof:.0f}s ago)")

    # Gate B1 — the host can actually run isolated rental containers.
    # A proof/calibration-only host may still produce valid proofs, but it must
    # not score or enter the renter marketplace until the runtime is present.
    if ctx.rental_runtime_ok:
        passed.append("rental_runtime")
    else:
        failed.append("rental_runtime (sysbox missing)")

    # Gate C — recent heartbeat
    age_hb_raw = ctx.now_ts - (ctx.last_heartbeat_at or 0)
    age_hb = max(0.0, age_hb_raw - float(ctx.validator_outage_since_heartbeat_s or 0.0))
    if ctx.last_heartbeat_at > 0 and age_hb <= cfg.heartbeat_recency_s:
        passed.append("heartbeat_recent")
    else:
        failed.append(f"heartbeat_recent (last={age_hb:.0f}s ago)")

    # Gate D — no open HARD sybil flags
    if ctx.open_hard_flag_count == 0:
        passed.append("no_hard_flags")
    else:
        failed.append(f"open_hard_flags={ctx.open_hard_flag_count}")

    # Gate E — last canary must have passed when canaries are required.
    # If canary_required is True (the scheduler is running on this
    # validator), never-canaried and errored executors are gated out.
    # If False (scheduler disabled — e.g. peer validators that don't
    # run canaries), the gate degrades to "must not have FAILED a
    # canary" so scoring still works.
    if ctx.last_canary_status == "fail":
        failed.append("canary_failed")
    elif cfg.canary_required and ctx.last_canary_status != "pass":
        if ctx.last_canary_at == 0:
            failed.append("canary_pending (never canaried)")
        else:
            failed.append(f"canary_pending (last={ctx.last_canary_status or 'unknown'})")
    elif ctx.last_canary_at == 0:
        if cfg.canary_required:
            failed.append("canary_pending (never canaried)")
        else:
            passed.append("canary_skipped (scheduler disabled)")
    else:
        passed.append(f"canary_last={ctx.last_canary_status or 'pass'}")

    # Gate F — minimum miner stake on the subnet.
    # Skin-in-the-game lever: a sybil farm can afford many cheap
    # subnet registrations but cannot afford to lock N×MIN_STAKE alpha
    # per endpoint. Zero stake threshold disables the gate.
    required_stake = max(
        float(cfg.min_miner_stake_tao or 0.0),
        float(ctx.miner_required_stake_tao or 0.0)
        if ctx.miner_required_stake_tao > 0
        else executor_required_stake,
    )
    stake_in_grace = (
        required_stake > 0
        and ctx.miner_stake_tao < required_stake
        and float(cfg.stake_grace_until_ts or 0.0) > ctx.now_ts
    )
    stake_ok = required_stake <= 0 or ctx.miner_stake_tao >= required_stake or stake_in_grace
    if required_stake > 0:
        if ctx.miner_stake_tao >= required_stake:
            passed.append(
                f"min_stake (have={ctx.miner_stake_tao:.2f}α"
                f" ≥ {required_stake:.2f}α)"
            )
        elif stake_in_grace:
            passed.append(
                f"min_stake_grace (have={ctx.miner_stake_tao:.2f}α"
                f" < {required_stake:.2f}α)"
            )
        else:
            failed.append(
                f"min_stake (have={ctx.miner_stake_tao:.2f}α"
                f" < {required_stake:.2f}α required)"
            )

    # Gate G / multiplier — rolling proof reliability.
    # Do not require 100% forever: honest miners can miss a challenge during
    # upgrades, transient network failures, or validator-side hiccups. But a
    # miner that repeatedly fails should lose weight quickly even if it still
    # lands one valid proof inside proof_recency_s.
    reliability_rates: list[float] = []
    if ctx.samples_1h >= cfg.min_reliability_samples:
        reliability_rates.append(max(0.0, min(1.0, ctx.pass_rate_1h)))
    if ctx.samples_24h >= cfg.min_reliability_samples:
        reliability_rates.append(max(0.0, min(1.0, ctx.pass_rate_24h)))
    reliability_factor = 1.0
    if reliability_rates:
        quality = min(reliability_rates)
        if quality < cfg.min_reliability_gate:
            failed.append(f"proof_reliability ({quality:.2f} < {cfg.min_reliability_gate:.2f})")
        else:
            reliability_factor = quality ** cfg.reliability_exponent
            passed.append(f"proof_reliability={quality:.2f}")

    if failed:
        return ScoringResult(
            executor_id=ctx.executor_id,
            hotkey_ss58=ctx.hotkey_ss58,
            base_price=base,
            util_factor=0.0,
            score=0.0,
            zero_reason=" ; ".join(failed),
            gates_passed=passed,
            gates_failed=failed,
            reliability_factor=0.0,
            miner_stake_alpha=ctx.miner_stake_tao,
            executor_required_stake_alpha=executor_required_stake,
            miner_required_stake_alpha=required_stake,
            stake_grace_until_ts=float(cfg.stake_grace_until_ts or 0.0),
            stake_ok=stake_ok,
            stake_in_grace=stake_in_grace,
        )

    util_factor = 1.0 if ctx.is_rented else cfg.idle_fraction
    return ScoringResult(
        executor_id=ctx.executor_id,
        hotkey_ss58=ctx.hotkey_ss58,
        base_price=base,
        util_factor=util_factor,
        score=base * util_factor * reliability_factor,
        zero_reason="",
        gates_passed=passed,
        gates_failed=failed,
        reliability_factor=reliability_factor,
        miner_stake_alpha=ctx.miner_stake_tao,
        executor_required_stake_alpha=executor_required_stake,
        miner_required_stake_alpha=required_stake,
        stake_grace_until_ts=float(cfg.stake_grace_until_ts or 0.0),
        stake_ok=stake_ok,
        stake_in_grace=stake_in_grace,
    )


def _dt_to_unix(dt) -> float:
    """Convert a naive UTC datetime (as stored by SQLAlchemy via
    datetime.utcnow) to a Unix timestamp. .timestamp() on a naive
    datetime interprets as LOCAL time, off by the host's TZ offset;
    calendar.timegm(.utctimetuple()) is correct."""
    if dt is None:
        return 0.0
    return calendar.timegm(dt.utctimetuple())


def build_context_from_db(executor_id: str, hotkey_ss58: str, gpu_model: str,
                           gpu_count: int, is_rented: bool, is_active: bool,
                           db, now_ts: float,
                           miner_stake_tao: float = 0.0,
                           miner_required_stake_tao: float = 0.0,
                           miner_address: str = "",
                           miner_uid: int | None = None,
                           endpoint: str = "") -> ScoringContext:
    """Pull the DB-side state needed to score one executor:
      - last_proof_valid_at: latest verified_at from proof_results
      - last_heartbeat_at: Executor.last_seen
      - open_hard_flag_count: open HARD sybil_flags count
      - last_canary_status / last_canary_at: from canary_records

    DB errors return zeroed values which trip the relevant gate —
    fail-safe rather than fail-open."""
    from common.db import ProofResult, Executor, SybilFlag, CanaryRecord
    last_proof_at = 0.0
    last_hb_at = 0.0
    flag_count = 0
    last_canary_status = ""
    last_canary_at = 0.0
    pass_rate_1h = 1.0
    pass_rate_24h = 1.0
    samples_1h = 0
    samples_24h = 0
    rental_runtime_ok = True
    validator_outage_since_proof_s = 0.0
    validator_outage_since_heartbeat_s = 0.0
    try:
        from common.db import ExecutorHardware, ExecutorStats
        with db.session() as s:
            pr = s.query(ProofResult).filter(
                ProofResult.executor_id == executor_id,
                ProofResult.valid == True,  # noqa: E712
                ProofResult.tier != "skipped",
            ).order_by(ProofResult.verified_at.desc()).first()
            if pr is not None:
                last_proof_at = _dt_to_unix(pr.verified_at)
                from sqlalchemy import or_
                skipped = s.query(ProofResult).filter(
                    ProofResult.executor_id == executor_id,
                    ProofResult.valid == True,  # noqa: E712
                    ProofResult.tier == "skipped",
                    or_(
                        ProofResult.reason == None,  # noqa: E711
                        ~ProofResult.reason.like("timing_outlier_graced:%"),
                    ),
                ).order_by(ProofResult.verified_at.desc()).first()
                if skipped is not None:
                    last_proof_at = max(last_proof_at, _dt_to_unix(skipped.verified_at))

            ex = s.query(Executor).filter_by(executor_id=executor_id).first()
            if ex is not None:
                last_hb_at = _dt_to_unix(ex.last_seen)
                if not hotkey_ss58:
                    hotkey_ss58 = ex.hotkey_ss58 or ""
                if not miner_address:
                    miner_address = ex.miner_address or ""
                if not endpoint:
                    endpoint = ex.endpoint or ""

            hw = s.query(ExecutorHardware).filter_by(executor_id=executor_id).first()
            if hw is not None:
                rental_runtime_ok = bool(hw.sysbox_detected)

            flag_count = s.query(SybilFlag).filter(
                SybilFlag.executor_id == executor_id,
                SybilFlag.cleared_at == None,  # noqa: E711
                SybilFlag.severity == "hard",
            ).count()

            cr = s.query(CanaryRecord).filter_by(executor_id=executor_id).first()
            if cr is not None:
                last_canary_status = cr.last_status or ""
                last_canary_at = _dt_to_unix(cr.last_canaried_at)

            st = s.query(ExecutorStats).filter_by(executor_id=executor_id).first()
            if st is not None:
                pass_rate_1h = float(st.pass_rate_1h if st.pass_rate_1h is not None else 1.0)
                pass_rate_24h = float(st.pass_rate_24h if st.pass_rate_24h is not None else 1.0)
                samples_1h = int(st.samples_1h or 0)
                samples_24h = int(st.samples_24h or 0)
        try:
            from common.db import validator_outage_overlap_seconds
            if last_proof_at > 0:
                validator_outage_since_proof_s = validator_outage_overlap_seconds(
                    db, last_proof_at, now_ts,
                )
            if last_hb_at > 0:
                validator_outage_since_heartbeat_s = validator_outage_overlap_seconds(
                    db, last_hb_at, now_ts,
                )
        except Exception:
            pass
    except Exception:
        pass

    return ScoringContext(
        executor_id=executor_id, hotkey_ss58=hotkey_ss58,
        gpu_model=gpu_model, gpu_count=gpu_count,
        is_rented=is_rented, is_active=is_active,
        last_proof_valid_at=last_proof_at,
        last_heartbeat_at=last_hb_at,
        open_hard_flag_count=flag_count,
        last_canary_status=last_canary_status,
        last_canary_at=last_canary_at,
        miner_stake_tao=miner_stake_tao,
        miner_required_stake_tao=miner_required_stake_tao,
        miner_address=miner_address,
        miner_uid=miner_uid,
        endpoint=endpoint,
        pass_rate_1h=pass_rate_1h,
        pass_rate_24h=pass_rate_24h,
        samples_1h=samples_1h,
        samples_24h=samples_24h,
        rental_runtime_ok=rental_runtime_ok,
        validator_outage_since_proof_s=validator_outage_since_proof_s,
        validator_outage_since_heartbeat_s=validator_outage_since_heartbeat_s,
        now_ts=now_ts,
    )


def aggregate_by_hotkey(results: list[ScoringResult]) -> dict[str, float]:
    """Sum per-executor scores into per-miner (hotkey) scores. The
    validator hands this (or a normalised version) to the Bittensor
    weight-setter."""
    by_hk: dict[str, float] = {}
    for r in results:
        if not r.hotkey_ss58:
            continue
        by_hk[r.hotkey_ss58] = by_hk.get(r.hotkey_ss58, 0.0) + r.score
    return by_hk


# ── Executor state machine ────────────────────────────────────────
# Used by the rental gate, /instances UI, and the canary scheduler.
# State is DERIVED from DB+chain (not stored as a column); single source
# of truth is the underlying records.

EXECUTOR_STATE_CANARY_PENDING = "canary_pending"    # never canaried OR canary errored
EXECUTOR_STATE_BANNED         = "banned"            # last canary failed OR open HARD flag
EXECUTOR_STATE_RENTABLE       = "rentable"          # passed canary, no open flags, recent proofs
EXECUTOR_STATE_STALE          = "stale"             # no recent proofs / heartbeat
EXECUTOR_STATE_INACTIVE       = "inactive"          # lease expired or deregistered


def executor_state(ctx: ScoringContext, cfg: Optional[ScoringConfig] = None) -> str:
    """Map a ScoringContext to a single state string. Used by
    `rent.py` to gate `/rent` and by `browse.py` to render `/instances`.
    Order matters: more-severe states checked first."""
    cfg = cfg or ScoringConfig()
    if not ctx.is_active:
        return EXECUTOR_STATE_INACTIVE
    if policy_blacklist_reason(
        ctx.executor_id,
        hotkey_ss58=ctx.hotkey_ss58,
        miner_address=ctx.miner_address,
        miner_uid=ctx.miner_uid,
        endpoint=ctx.endpoint,
    ):
        return EXECUTOR_STATE_BANNED
    if ctx.open_hard_flag_count > 0 or ctx.last_canary_status == "fail":
        return EXECUTOR_STATE_BANNED
    if cfg.canary_required and ctx.last_canary_status != "pass":
        return EXECUTOR_STATE_CANARY_PENDING
    proof_fresh = (
        ctx.last_proof_valid_at > 0
        and (
            ctx.now_ts - ctx.last_proof_valid_at
            - float(ctx.validator_outage_since_proof_s or 0.0)
        ) <= cfg.proof_recency_s
    )
    hb_fresh = (
        ctx.last_heartbeat_at > 0
        and (
            ctx.now_ts - ctx.last_heartbeat_at
            - float(ctx.validator_outage_since_heartbeat_s or 0.0)
        ) <= cfg.heartbeat_recency_s
    )
    if not (proof_fresh and hb_fresh):
        return EXECUTOR_STATE_STALE
    # If we get here: active, canary-passed, no flags, fresh proof/HB
    return EXECUTOR_STATE_RENTABLE
