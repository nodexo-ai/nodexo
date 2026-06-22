"""Follower-validator score import.

Follower validators are disabled by default. When enabled, a low-resource
validator imports the primary validator's public `/api/instances` view and sets
weights from those scores instead of running local GPU proof verification.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from typing import Any, Callable

import bittensor as bt

from neurons.validator.scoring.scorer import ScoringResult


def follower_state_url() -> str:
    return (
        os.environ.get("VALIDATOR_FOLLOWER_STATE_URL", "")
        or os.environ.get("NODEXO_VALIDATOR_FOLLOWER_STATE_URL", "")
    ).strip()


def follower_poll_interval_s() -> float:
    raw = (
        os.environ.get("VALIDATOR_FOLLOWER_POLL_INTERVAL_S", "")
        or os.environ.get("NODEXO_VALIDATOR_FOLLOWER_POLL_INTERVAL_S", "")
        or "30"
    )
    try:
        return max(5.0, min(600.0, float(raw)))
    except Exception:
        return 30.0


def _timeout_s() -> float:
    raw = os.environ.get("VALIDATOR_FOLLOWER_TIMEOUT_S", "8")
    try:
        return max(1.0, min(30.0, float(raw)))
    except Exception:
        return 8.0


def _auth_headers() -> dict[str, str]:
    token = (
        os.environ.get("VALIDATOR_FOLLOWER_TOKEN", "")
        or os.environ.get("NODEXO_VALIDATOR_FOLLOWER_TOKEN", "")
    ).strip()
    if not token:
        return {}
    return {"authorization": f"Bearer {token}"}


def _fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "accept": "application/json",
            "user-agent": "nodexo-validator/follower",
            **_auth_headers(),
        },
    )
    with urllib.request.urlopen(req, timeout=_timeout_s()) as response:
        status = int(getattr(response, "status", 200))
        if status < 200 or status >= 300:
            raise RuntimeError(f"upstream HTTP {status}")
        data = response.read(5_000_000)
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("follower upstream response must be an object")
    return parsed


def scores_from_upstream_instances(instances: list[dict[str, Any]]) -> dict[str, ScoringResult]:
    """Convert public instance rows into local ScoringResult snapshots."""
    scores: dict[str, ScoringResult] = {}
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        executor_id = str(inst.get("instance_id") or "").strip()
        if not executor_id:
            continue
        hotkey = str(inst.get("hotkey") or "").strip()
        score_raw = inst.get("score") if isinstance(inst.get("score"), dict) else {}
        score_value = float(score_raw.get("value") or 0.0)
        zero_reason = str(score_raw.get("zero_reason") or "")
        gates_failed = [zero_reason] if score_value <= 0 and zero_reason else []
        scores[executor_id] = ScoringResult(
            executor_id=executor_id,
            hotkey_ss58=hotkey,
            base_price=float(score_raw.get("base_price") or 0.0),
            util_factor=float(score_raw.get("util_factor") or 0.0),
            score=max(0.0, score_value),
            zero_reason=zero_reason,
            gates_passed=[] if gates_failed else ["upstream_score"],
            gates_failed=gates_failed,
            reliability_factor=1.0,
            miner_stake_alpha=float(score_raw.get("miner_stake_alpha") or 0.0),
            executor_required_stake_alpha=float(score_raw.get("executor_required_stake_alpha") or 0.0),
            miner_required_stake_alpha=float(score_raw.get("miner_required_stake_alpha") or 0.0),
            stake_grace_until_ts=float(score_raw.get("stake_grace_until_ts") or 0.0),
            stake_ok=bool(score_raw.get("stake_ok", True)),
            stake_in_grace=bool(score_raw.get("stake_in_grace", False)),
        )
    return scores


async def run_follower_state_loop(
    apply_scores: Callable[[dict[str, ScoringResult]], None],
    *,
    url: str | None = None,
    interval_s: float | None = None,
) -> None:
    """Poll upstream state and hand converted scores to the validator."""
    target = (url or follower_state_url()).strip()
    if not target:
        raise RuntimeError("VALIDATOR_FOLLOWER_STATE_URL is required when follower mode is enabled")
    interval = follower_poll_interval_s() if interval_s is None else interval_s
    bt.logging.info(f"Follower validator state import enabled: {target} interval={interval:.0f}s")
    while True:
        try:
            payload = await asyncio.to_thread(_fetch_json, target)
            instances = payload.get("instances") if isinstance(payload, dict) else []
            if not isinstance(instances, list):
                raise ValueError("upstream payload missing instances list")
            scores = scores_from_upstream_instances(instances)
            apply_scores(scores)
            positive = sum(1 for score in scores.values() if score.score > 0)
            bt.logging.debug(
                f"Follower scores refreshed: {positive}/{len(scores)} positive"
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            bt.logging.warning(f"Follower state refresh failed: {e}")
        await asyncio.sleep(interval)
