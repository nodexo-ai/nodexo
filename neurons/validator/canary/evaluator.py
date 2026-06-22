"""
Canary result evaluator — reads fill.py's JSON output and decides
which sybil flags (if any) to open.

CPU-ONLY: the validator does not have a GPU. Integrity verification
is via numpy's PCG64 (bit-stable across hardware per the numpy
implementation contract). The evaluator re-derives the exact bytes
the miner's verify-chunk was supposed to be, hashes them, and
compares to the hash the miner returned.

Three independent signals, each fires its own flag:
  - allocated_mb < target * threshold → canary_vram_fail
  - hbm_bandwidth_gb_s below floor    → canary_bandwidth_low
  - verify_hash mismatch              → canary_checksum_mismatch
"""
from __future__ import annotations

import hashlib
from typing import Optional


def _expected_verify_hash(
    nonce_hex: str,
    device_index: int,
    verify_mb: int,
) -> Optional[str]:
    """CPU-only re-derivation of the expected verify-chunk SHA256.

    Must match fill.py's verify-chunk generation EXACTLY:
      seed   = sha256(nonce || "|fill|" || (10000 + device_index, LE u32))[:8]
                — wait, fill.py uses _seed_from_nonce(nonce, 10_000 + device_index)
                  which derives via sha256(nonce || "|fill|" || idx, LE u32).
      rng    = numpy.random.default_rng(seed)
      bytes  = rng.bytes(verify_mb * 1024 * 1024)
      hash   = sha256(bytes).hexdigest()

    Bit-stable across hardware: numpy's default_rng/PCG64 implementation
    is documented stable, and SHA256 is deterministic.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    # MUST match fill.py's _seed_from_nonce(nonce_hex, 10_000 + device_index).
    # fill.py code: int.from_bytes(sha256(nonce || b"|fill|" || idx_le32)[:8], 'little')
    idx = 10_000 + device_index
    seed_hash = hashlib.sha256(
        nonce_hex.encode() + b"|fill|" + idx.to_bytes(4, "little"),
    ).digest()
    seed_int = int.from_bytes(seed_hash[:8], "little", signed=False)

    rng = np.random.default_rng(seed_int)
    expected_bytes = rng.bytes(verify_mb * 1024 * 1024)
    return hashlib.sha256(expected_bytes).hexdigest()


def evaluate_fill_result(
    fill_result: dict,
    expected_nonce: str,
    target_mb_per_gpu: int,
    gpu_model_name: str,
    chunk_mb: int,
    storage_result: dict | None = None,
    min_storage_gb: int | None = None,
) -> dict:
    """Examine a fill.py JSON output, return decision dict:

      {
        "summary": "<short human description>",
        "flags": [(reason, evidence_dict), ...],
        "per_device": [<per-device decisions>],
      }
    """
    from neurons.validator.canary.runner import (
        ALLOC_RATIO_THRESHOLD, HBM_BANDWIDTH_FLOOR_GB_S,
        HBM_BANDWIDTH_FLOOR_DEFAULT,
    )

    out: dict = {"summary": "", "flags": [], "per_device": []}
    storage_failures: list[str] = []
    if min_storage_gb:
        storage = storage_result if isinstance(storage_result, dict) else {}
        total = int(storage.get("total_bytes") or 0)
        available = int(storage.get("available_bytes") or 0)
        threshold = int(min_storage_gb * (1 << 30) * 0.95)
        if total < threshold:
            storage_failures.append(
                f"storage total {total // (1 << 30)}GB < {min_storage_gb}GB"
            )
        if available < threshold:
            storage_failures.append(
                f"storage available {available // (1 << 30)}GB < {min_storage_gb}GB"
            )
        if not storage.get("write_ok"):
            storage_failures.append(
                str(storage.get("error") or "storage write probe failed")[:240]
            )
        if storage_failures:
            out["flags"].append((
                "canary_storage_fail",
                {
                    "min_storage_gb": min_storage_gb,
                    "total_bytes": total,
                    "available_bytes": available,
                    "write_mb": int(storage.get("write_mb") or 0),
                    "write_ok": bool(storage.get("write_ok")),
                    "error": str(storage.get("error") or "")[:240],
                },
            ))
    if not isinstance(fill_result, dict):
        out["summary"] = "; ".join(storage_failures + ["fill_result not a dict"])
        out["flags"].append(("canary_vram_fail", {"error": "no fill_result"}))
        return out

    # Replay defence — nonce on the wire must match what we sent.
    if fill_result.get("nonce") != expected_nonce:
        out["summary"] = "; ".join(
            storage_failures + ["nonce mismatch (replay or wrong run)"]
        )
        out["flags"].append((
            "canary_checksum_mismatch",
            {"expected_nonce": expected_nonce[:16],
             "got_nonce": str(fill_result.get("nonce", ""))[:16]},
        ))
        return out

    devices = fill_result.get("per_device", [])
    if not devices:
        out["summary"] = "; ".join(storage_failures + ["no per-device data"])
        out["flags"].append(("canary_vram_fail", {"error": "no per_device"}))
        return out

    floor = HBM_BANDWIDTH_FLOOR_GB_S.get(gpu_model_name, HBM_BANDWIDTH_FLOOR_DEFAULT)

    failed_summaries: list[str] = []
    for i, dev in enumerate(devices):
        idx = dev.get("index", i)
        per_dev = {"index": idx, "decisions": []}

        # Device-level error from the worker → straight vram_fail.
        if "error" in dev and "name" not in dev:
            out["flags"].append((
                "canary_vram_fail",
                {"device_index": idx, "device_error": dev["error"][:200]},
            ))
            failed_summaries.append(f"dev{idx}: error")
            per_dev["decisions"].append("device-level error")
            out["per_device"].append(per_dev)
            continue

        allocated = int(dev.get("allocated_mb", 0))
        target = int(dev.get("target_mb", target_mb_per_gpu))
        bw = float(dev.get("hbm_bandwidth_gb_s", 0.0))
        got_verify = str(dev.get("verify_hash", ""))
        verify_mb = int(dev.get("verify_mb", 0))

        # CHECK 1: allocation reached target. Sybil hiding VRAM trips
        # this — fill OOMs before reaching the cap.
        if target > 0:
            ratio = allocated / target
            if ratio < ALLOC_RATIO_THRESHOLD:
                out["flags"].append((
                    "canary_vram_fail",
                    {
                        "device_index": idx,
                        "allocated_mb": allocated,
                        "target_mb": target,
                        "ratio": round(ratio, 3),
                        "threshold": ALLOC_RATIO_THRESHOLD,
                        "oom": dev.get("oom", "")[:300],
                    },
                ))
                failed_summaries.append(
                    f"dev{idx}: alloc {allocated}/{target} MB ({ratio:.2%})"
                )
                per_dev["decisions"].append(
                    f"vram_fail: allocated {allocated}/{target} ({ratio:.2%})"
                )
                # Skip bandwidth/checksum on a vram_fail device — those
                # readings are meaningless when GPU was contended.
                out["per_device"].append(per_dev)
                continue

        # CHECK 2: HBM bandwidth clears the floor for this GPU. Rootkit
        # faking via host RAM gets PCIe-tier bandwidth, well below.
        # We tolerate bw=0 on scan_error (OOM during the reduction
        # kernel itself) since that doesn't indicate cheating.
        if bw > 0 and bw < floor:
            out["flags"].append((
                "canary_bandwidth_low",
                {
                    "device_index": idx,
                    "measured_gb_s": bw,
                    "floor_gb_s": floor,
                    "gpu_model": gpu_model_name,
                },
            ))
            failed_summaries.append(
                f"dev{idx}: bandwidth {bw:.1f} < floor {floor:.1f} GB/s"
            )
            per_dev["decisions"].append(
                f"bandwidth_low: {bw:.1f}<{floor:.1f}"
            )

        # CHECK 3: integrity via CPU-derived verify chunk.
        # A miner who faked the JSON output without engaging the GPU
        # cannot produce the right hash without actually running the
        # numpy PRNG and pushing bytes through GPU memory. If the
        # miner reports verify_mb=0 (or empty hash), that's also a
        # FAIL — they're trying to silently skip the check by
        # breaking numpy/torch in their container.
        if verify_mb == 0 or not got_verify:
            out["flags"].append((
                "canary_checksum_mismatch",
                {
                    "device_index": idx,
                    "reason": "verify_mb=0 or empty verify_hash (numpy install failed or sabotaged)",
                },
            ))
            failed_summaries.append(f"dev{idx}: verify skipped/empty")
            per_dev["decisions"].append("verify_missing")
        else:
            expected = _expected_verify_hash(expected_nonce, idx, verify_mb)
            if expected is None:
                # numpy missing on the VALIDATOR side. Without it we
                # can't re-derive; warn but don't fail.
                per_dev["decisions"].append("verify_skipped: validator has no numpy")
            elif got_verify != expected:
                out["flags"].append((
                    "canary_checksum_mismatch",
                    {
                        "device_index": idx,
                        "verify_mb": verify_mb,
                        "expected": expected,
                        "got": got_verify,
                    },
                ))
                failed_summaries.append(
                    f"dev{idx}: verify_hash {got_verify[:8]} != {expected[:8]}"
                )
                per_dev["decisions"].append("verify_mismatch")
            else:
                per_dev["decisions"].append("verify_ok")

        if not per_dev["decisions"]:
            per_dev["decisions"].append("pass")
        out["per_device"].append(per_dev)

    all_failures = storage_failures + failed_summaries
    out["summary"] = "pass" if not all_failures else "; ".join(all_failures)
    return out
