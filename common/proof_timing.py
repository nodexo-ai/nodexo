"""Shared proof timing model for miners and validators."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import RLock
from typing import Any

from common.config import GPU_VRAM_GB, canonical_gpu_model_name
from common.subnet_runtime_config import DEFAULT_GPU_HOURLY_PRICE


DEFAULT_WALL_CLOCK_GRACE_SEC = 30.0
DEFAULT_RECEIPT_SPREAD_GRACE_SEC = 12.0
DEFAULT_MIN_DELTA_FRACTION = 0.55
DEFAULT_MIN_DELTA_TOTAL_FRACTION_CAP = 0.30
DEFAULT_PASS0_RECEIPT_GRACE_SEC = 6.0


@dataclass
class GpuTimingModel:
    """Per-GPU-model timing thresholds for delta and wall-clock validation.

    Formulas:
      scaling = 1 + a1*(gpu_count-1) + a2*(gpu_count-1)^2
      delta_deadline = base1 * scaling * tol1
      wall_deadline  = wall_grace + base2 * scaling * tol2
      receipt_spread_deadline = receipt_spread_grace * scaling

    base1/base2 values MUST be calibrated against the optimized CUDA kernel.
    Multi-GPU miners run one worker per GPU in parallel, so scaling models
    contention/dispatch overhead, not serial runtime. The scaling fit is per
    GPU model from representative count samples such as 1x/3x/7x; every exact
    count does not need its own timing row. If N GPUs are serialized on one
    device, wall-clock and receipt-spread checks should trip.
    """
    base1: float
    base2: float
    a1: float = 0.142857
    a2: float = 0.0
    tol1: float = 1.1
    tol2: float = 1.1
    wall_grace: float = DEFAULT_WALL_CLOCK_GRACE_SEC
    receipt_spread_grace: float = DEFAULT_RECEIPT_SPREAD_GRACE_SEC
    calibrated: bool = False
    calibrated_gpu_counts: tuple[int, ...] = ()
    provenance: str = ""


def _assumed_timing_model(
    base1: float,
    base2: float,
    wall_grace: float,
    receipt_spread_grace: float,
    basis: str,
) -> GpuTimingModel:
    return GpuTimingModel(
        base1=base1,
        base2=base2,
        wall_grace=wall_grace,
        receipt_spread_grace=receipt_spread_grace,
        calibrated=False,
        provenance=(
            f"extrapolated ({basis}) - not measured; recalibrate before mainnet"
        ),
    )


def _extrapolated_timing_model(model: str) -> GpuTimingModel:
    vram = GPU_VRAM_GB.get(model, 24)
    table = {
        "NVIDIA GeForce RTX 3060": _assumed_timing_model(4.5, 10.0, 10, 7, "Ampere consumer 12GB"),
        "NVIDIA GeForce RTX 3060 Ti": _assumed_timing_model(4.0, 9.0, 10, 7, "Ampere consumer 8GB"),
        "NVIDIA GeForce RTX 3070": _assumed_timing_model(3.8, 8.5, 10, 7, "Ampere consumer 8GB"),
        "NVIDIA GeForce RTX 3070 Ti": _assumed_timing_model(3.5, 8.0, 10, 7, "Ampere consumer 8GB"),
        "NVIDIA GeForce RTX 3080": _assumed_timing_model(3.0, 7.0, 9, 6, "Ampere consumer 10GB"),
        "NVIDIA GeForce RTX 3080 Ti": _assumed_timing_model(2.8, 6.5, 9, 6, "Ampere consumer 12GB"),
        "NVIDIA GeForce RTX 3090": _assumed_timing_model(2.5, 6.0, 9, 6, "Ampere consumer 24GB"),
        "NVIDIA GeForce RTX 3090 Ti": _assumed_timing_model(2.4, 5.8, 9, 6, "Ampere consumer 24GB"),
        "NVIDIA GeForce RTX 4060 Ti": _assumed_timing_model(3.6, 8.0, 9, 6, "Ada consumer 16GB"),
        "NVIDIA GeForce RTX 4070": _assumed_timing_model(3.2, 7.0, 9, 6, "Ada consumer 12GB"),
        "NVIDIA GeForce RTX 4070 Ti": _assumed_timing_model(2.8, 6.0, 8, 5, "Ada consumer 12GB"),
        "NVIDIA GeForce RTX 4070 Ti SUPER": _assumed_timing_model(2.7, 5.8, 8, 5, "Ada consumer 16GB"),
        "NVIDIA GeForce RTX 4080": _assumed_timing_model(2.3, 5.2, 8, 5, "Ada consumer 16GB"),
        "NVIDIA GeForce RTX 4080 SUPER": _assumed_timing_model(2.2, 5.0, 8, 5, "Ada consumer 16GB"),
        "NVIDIA RTX 6000 Ada Generation": _assumed_timing_model(2.4, 5.5, 9, 6, "Ada workstation 48GB"),
        "NVIDIA A100-SXM4-40GB": _assumed_timing_model(5.0, 12.0, 12, 8, "A100 SXM 40GB"),
        "NVIDIA A100-PCIE-40GB": _assumed_timing_model(6.0, 14.0, 14, 9, "A100 PCIe 40GB"),
        "NVIDIA A100 80GB PCIe": _assumed_timing_model(6.0, 14.0, 14, 9, "A100 PCIe 80GB"),
        "NVIDIA L40": _assumed_timing_model(8.0, 18.0, 14, 9, "Ada L40 48GB"),
        "NVIDIA H100 NVL": _assumed_timing_model(7.0, 16.0, 12, 8, "H100 NVL 94GB"),
        "NVIDIA H200": _assumed_timing_model(8.0, 18.0, 12, 8, "H200 141GB"),
        "NVIDIA H200 NVL": _assumed_timing_model(8.0, 18.0, 12, 8, "H200 NVL 141GB"),
        "NVIDIA B200": _assumed_timing_model(6.0, 14.0, 12, 8, "Blackwell 192GB"),
        "NVIDIA B300 SXM6": _assumed_timing_model(5.0, 12.0, 12, 8, "Blackwell 288GB"),
    }
    return table.get(
        model,
        _assumed_timing_model(10.0, 25.0, 30, 12, f"{vram}GB fallback"),
    )


_TIMING_MODELS_LOCK = RLock()
_TIMING_CONFIG_SOURCE = "builtin"
_TIMING_CONFIG_VERSION = "builtin"


def _builtin_timing_models() -> dict[str, GpuTimingModel]:
    # Per-GPU timing model calibrated against the optimized CUDA kernel.
    # Entries tagged as extrapolated must be replaced with real measurements
    # before mainnet.
    measured_and_reviewed = {
        "NVIDIA GeForce RTX 4090": GpuTimingModel(
            base1=1.8, base2=4.0, wall_grace=8.0, receipt_spread_grace=5.0,
            calibrated=True,
            calibrated_gpu_counts=(1,),
            provenance="measured 2026-04-29 (cublasLt INT8, N=17664, pass=1.71s)",
        ),
        "NVIDIA RTX A6000": GpuTimingModel(
            base1=14.015, base2=18.737, tol1=1.2, tol2=1.35,
            wall_grace=24.0, receipt_spread_grace=18.0,
            calibrated=True,
            calibrated_gpu_counts=(1,),
            provenance="testnet fit 2026-06-10 v0.1.3; base from v0.1.2 p99/mean+2sd samples, delta tol1=1.20, wall tol2=1.35 operational margin; 1x only, multi-GPU scale unverified",
        ),
        "NVIDIA GeForce RTX 5090": GpuTimingModel(
            base1=2.072, base2=1.974, a1=0.734672,
            wall_grace=8.0, receipt_spread_grace=5.0,
            calibrated=True,
            calibrated_gpu_counts=(1, 4),
            provenance="testnet-468 calibrated fit 2026-07-23 from validator DB p99/mean+2sd free samples (n=56, counts=1,4)",
        ),
        "NVIDIA H100 80GB HBM3": GpuTimingModel(
            base1=10.846, base2=15.869, tol1=1.2, tol2=1.35,
            a1=0.168167,
            wall_grace=12.0, receipt_spread_grace=8.0,
            calibrated=True,
            calibrated_gpu_counts=(1, 2, 4, 8),
            provenance="testnet-468 validation 2026-07-22; production fit retained; 4x n=26 and 8x n=30 passed existing deadlines",
        ),
        "NVIDIA A100-SXM4-80GB": GpuTimingModel(
            base1=16.398, base2=26.68, a1=0.050997,
            wall_grace=12.0, receipt_spread_grace=8.0,
            calibrated=True,
            calibrated_gpu_counts=(1, 2, 4),
            provenance="production fit retained; testnet-468 2x validation 2026-07-23 (n=37); counts=1,2,4",
        ),
        "NVIDIA A100 80GB PCIe": GpuTimingModel(
            base1=21.362, base2=42.752, tol1=1.2, tol2=1.35,
            wall_grace=14.0, receipt_spread_grace=9.0,
            calibrated=True,
            calibrated_gpu_counts=(1,),
            provenance="testnet fit 2026-06-10 v0.1.3; base from v0.1.2 p99/mean+2sd samples, delta tol1=1.20, wall tol2=1.35 operational margin; 1x only, multi-GPU scale unverified",
        ),
        "NVIDIA L40S": GpuTimingModel(
            base1=7.0, base2=16.0, wall_grace=12.0, receipt_spread_grace=8.0,
            calibrated=False,
            provenance="extrapolated (~733 TOPS, N=23040, 48GB) - recalibrate before mainnet",
        ),
        "NVIDIA H200": GpuTimingModel(
            base1=11.006, base2=16.356, a1=0.009917,
            wall_grace=12.0, receipt_spread_grace=8.0,
            calibrated=True,
            calibrated_gpu_counts=(1, 2, 4, 8),
            provenance="testnet-468 calibrated fit 2026-07-23 from validator DB p99/mean+2sd free samples (n=149, counts=1,2,4,8)",
        ),
        "NVIDIA B200": GpuTimingModel(
            base1=9.47, base2=14.543, a1=0.142857,
            wall_grace=12.0, receipt_spread_grace=8.0,
            calibrated=True,
            calibrated_gpu_counts=(1,),
            provenance="testnet-468 calibrated fit 2026-07-23 from validator DB p99/mean+2sd free samples (n=25, counts=1)",
        ),
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": GpuTimingModel(
            base1=8.262, base2=0.001, a1=0.0, a2=0.000824,
            wall_grace=30.0, receipt_spread_grace=12.0,
            calibrated=True,
            calibrated_gpu_counts=(1, 2, 4),
            provenance="testnet-468 calibrated fit 2026-07-23 from validator DB p99/mean+2sd free samples (n=144, counts=1,2,4)",
        ),
    }
    out = {
        model: measured_and_reviewed.get(model) or _extrapolated_timing_model(model)
        for model in DEFAULT_GPU_HOURLY_PRICE
    }
    out.update({
        "default": GpuTimingModel(
            base1=10.0, base2=25.0,
            calibrated=False,
            provenance="default fallback - used only when GPU model is not in supported table",
        ),
    })
    return out


GPU_TIMING_MODELS: dict[str, GpuTimingModel] = _builtin_timing_models()


def timing_model_to_dict(model: GpuTimingModel) -> dict[str, Any]:
    return {
        "base1": model.base1,
        "base2": model.base2,
        "a1": model.a1,
        "a2": model.a2,
        "tol1": model.tol1,
        "tol2": model.tol2,
        "wall_grace": model.wall_grace,
        "receipt_spread_grace": model.receipt_spread_grace,
        "calibrated": bool(model.calibrated),
        "calibrated_gpu_counts": list(model.calibrated_gpu_counts),
        "provenance": model.provenance,
    }


def timing_models_snapshot() -> dict[str, dict[str, Any]]:
    with _TIMING_MODELS_LOCK:
        return {name: timing_model_to_dict(model) for name, model in GPU_TIMING_MODELS.items()}


def timing_config_metadata() -> dict[str, str]:
    with _TIMING_MODELS_LOCK:
        return {
            "source": _TIMING_CONFIG_SOURCE,
            "version": _TIMING_CONFIG_VERSION,
            "model_count": str(len(GPU_TIMING_MODELS)),
            "calibrated_count": str(sum(
                1 for name, model in GPU_TIMING_MODELS.items()
                if name != "default" and model.calibrated
            )),
        }


def _coerce_timing_model(name: str, raw: Any) -> GpuTimingModel:
    if not isinstance(raw, dict):
        raise ValueError(f"timing model {name!r} must be an object")
    required = ("base1", "base2")
    for key in required:
        if key not in raw:
            raise ValueError(f"timing model {name!r} missing {key}")
    provenance = str(raw.get("provenance", ""))[:240]
    model = GpuTimingModel(
        base1=float(raw["base1"]),
        base2=float(raw["base2"]),
        a1=float(raw.get("a1", 0.142857)),
        a2=float(raw.get("a2", 0.0)),
        tol1=float(raw.get("tol1", 1.1)),
        tol2=float(raw.get("tol2", 1.1)),
        wall_grace=float(raw.get("wall_grace", DEFAULT_WALL_CLOCK_GRACE_SEC)),
        receipt_spread_grace=float(raw.get("receipt_spread_grace", DEFAULT_RECEIPT_SPREAD_GRACE_SEC)),
        calibrated=(
            bool(raw["calibrated"])
            if "calibrated" in raw
            else _infer_calibrated_from_provenance(name, provenance)
        ),
        calibrated_gpu_counts=_coerce_calibrated_counts(raw.get("calibrated_gpu_counts")),
        provenance=provenance,
    )
    numeric = (
        model.base1, model.base2, model.a1, model.a2, model.tol1, model.tol2,
        model.wall_grace, model.receipt_spread_grace,
    )
    if any(not isfinite(value) for value in numeric):
        raise ValueError(f"timing model {name!r} contains non-finite values")
    if model.base1 <= 0 or model.base2 <= 0 or model.base1 > 10_000 or model.base2 > 10_000:
        raise ValueError(f"timing model {name!r} base values out of range")
    if model.a1 < 0 or model.a2 < 0 or model.a1 > 5.0 or model.a2 > 5.0:
        raise ValueError(f"timing model {name!r} scaling coefficients out of range")
    if not (1.0 <= model.tol1 <= 5.0 and 1.0 <= model.tol2 <= 5.0):
        raise ValueError(f"timing model {name!r} tolerances out of range")
    if (
        model.wall_grace < 0 or model.receipt_spread_grace < 0
        or model.wall_grace > 600 or model.receipt_spread_grace > 600
    ):
        raise ValueError(f"timing model {name!r} grace values must be non-negative")
    return model


def _coerce_calibrated_counts(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("calibrated_gpu_counts must be a list")
    counts: set[int] = set()
    for item in raw:
        try:
            count = int(item)
        except Exception as exc:
            raise ValueError("calibrated_gpu_counts must contain integers") from exc
        if count < 1 or count > 64:
            raise ValueError("calibrated_gpu_counts entries must be between 1 and 64")
        counts.add(count)
    return tuple(sorted(counts))


def _infer_calibrated_from_provenance(name: str, provenance: str) -> bool:
    """Compatibility for configs written before the explicit flag existed."""
    if canonical_gpu_model_name(name) == "default":
        return False
    text = str(provenance or "").lower()
    if any(token in text for token in ("not measured", "extrapolated", "fallback")):
        return False
    return any(token in text for token in ("measured", "recalibrated", "fitted"))


def apply_timing_models_from_config(config: dict[str, Any], *, source: str = "remote") -> int:
    """Replace timing models from a subnet config document.

    The document may either be `{timing_models: {...}}` or the model map
    directly. Updates are atomic: invalid input raises and leaves the current
    table untouched.
    """
    global _TIMING_CONFIG_SOURCE, _TIMING_CONFIG_VERSION
    raw_models = config.get("timing_models", config) if isinstance(config, dict) else None
    if not isinstance(raw_models, dict):
        raise ValueError("timing config missing timing_models object")
    parsed: dict[str, GpuTimingModel] = {}
    for name, raw in raw_models.items():
        canonical_name = canonical_gpu_model_name(str(name))
        if canonical_name in parsed:
            raise ValueError(
                f"duplicate timing model after canonicalization: {name!r} -> {canonical_name!r}"
            )
        parsed[canonical_name] = _coerce_timing_model(canonical_name, raw)
    if "default" not in parsed:
        raise ValueError("timing config must include a default model")
    version = str(config.get("version") or config.get("config_version") or "unknown") if isinstance(config, dict) else "unknown"
    with _TIMING_MODELS_LOCK:
        GPU_TIMING_MODELS.clear()
        GPU_TIMING_MODELS.update(parsed)
        _TIMING_CONFIG_SOURCE = str(source)
        _TIMING_CONFIG_VERSION = version[:80]
    return len(parsed)


def reset_timing_models_for_tests() -> None:
    global _TIMING_CONFIG_SOURCE, _TIMING_CONFIG_VERSION
    with _TIMING_MODELS_LOCK:
        GPU_TIMING_MODELS.clear()
        GPU_TIMING_MODELS.update(_builtin_timing_models())
        _TIMING_CONFIG_SOURCE = "builtin"
        _TIMING_CONFIG_VERSION = "builtin"


def get_timing_model(gpu_model: str | None) -> GpuTimingModel:
    with _TIMING_MODELS_LOCK:
        canonical_name = canonical_gpu_model_name(gpu_model)
        return GPU_TIMING_MODELS.get(
            canonical_name,
            GPU_TIMING_MODELS["default"],
        )


def is_timing_model_calibrated(gpu_model: str | None, gpu_count: int | None = None) -> bool:
    """Return whether a GPU model/count is inside calibrated timing coverage.

    `calibrated_gpu_counts` records representative measured count groups, not
    an exact allowlist. Once a row is fitted from samples such as 1x/3x/7x, the
    model is considered covered for every count from the smallest measured
    count through the largest measured count. Counts above the measured range
    remain blocked until the operator extends the calibration range.
    """
    with _TIMING_MODELS_LOCK:
        canonical_name = canonical_gpu_model_name(gpu_model)
        model = GPU_TIMING_MODELS.get(canonical_name)
        if not model or not model.calibrated:
            return False
        if gpu_count is None:
            return True
        try:
            count = max(1, int(gpu_count))
        except Exception:
            return False
        if not model.calibrated_gpu_counts:
            return False
        return min(model.calibrated_gpu_counts) <= count <= max(model.calibrated_gpu_counts)


def calibrated_timing_models_snapshot() -> dict[str, dict[str, Any]]:
    with _TIMING_MODELS_LOCK:
        return {
            name: timing_model_to_dict(model)
            for name, model in GPU_TIMING_MODELS.items()
            if name != "default" and model.calibrated
        }


def compute_gpu_count_scaling(model: GpuTimingModel, gpu_count: int) -> float:
    gpu_count = max(1, int(gpu_count or 1))
    return 1.0 + model.a1 * (gpu_count - 1) + model.a2 * (gpu_count - 1) ** 2


def compute_delta_deadline(model: GpuTimingModel, gpu_count: int) -> float:
    scaling = compute_gpu_count_scaling(model, gpu_count)
    return model.base1 * scaling * model.tol1


def compute_pass0_receipt_deadline(
    model: GpuTimingModel,
    gpu_count: int,
    jitter_s: float = 0.0,
    grace_s: float = DEFAULT_PASS0_RECEIPT_GRACE_SEC,
) -> float:
    """Latest credible validator receipt time for pass_0 after the beacon.

    pass_0 and pass_1 run the same commitment workload, so the calibrated
    one-pass deadline is also the GPU-agnostic bound for when pass_0 may first
    appear. The small grace is network/clock margin, not a GPU calibration knob.
    """
    try:
        jitter_s = max(0.0, float(jitter_s or 0.0))
    except Exception:
        jitter_s = 0.0
    try:
        grace_s = max(0.0, float(grace_s or 0.0))
    except Exception:
        grace_s = DEFAULT_PASS0_RECEIPT_GRACE_SEC
    return jitter_s + compute_delta_deadline(model, gpu_count) + grace_s


def compute_delta_floor(
    model: GpuTimingModel,
    gpu_count: int,
    fraction: float = DEFAULT_MIN_DELTA_FRACTION,
    total_fraction_cap: float = DEFAULT_MIN_DELTA_TOTAL_FRACTION_CAP,
) -> float:
    """Minimum credible validator-received pass delta.

    If pass_0 and pass_1 receipts arrive almost together, the receipt timing
    was batched after the work finished and cannot prove sequential GPU work.
    `base1` is deliberately conservative for upper deadlines on some calibrated
    rows, so the lower-bound floor is capped against total proof runtime.
    """
    try:
        fraction = float(fraction)
    except Exception:
        fraction = DEFAULT_MIN_DELTA_FRACTION
    fraction = min(1.0, max(0.0, fraction))
    try:
        total_fraction_cap = float(total_fraction_cap)
    except Exception:
        total_fraction_cap = DEFAULT_MIN_DELTA_TOTAL_FRACTION_CAP
    total_fraction_cap = min(1.0, max(0.0, total_fraction_cap))
    scaling = compute_gpu_count_scaling(model, gpu_count)
    base_floor = model.base1 * scaling * fraction
    total_cap = model.base2 * scaling * total_fraction_cap
    if total_cap <= 0:
        return base_floor
    return min(base_floor, total_cap)


def compute_wall_clock_deadline(
    model: GpuTimingModel,
    gpu_count: int,
    grace: float | None = None,
) -> float:
    scaling = compute_gpu_count_scaling(model, gpu_count)
    if grace is None:
        grace = model.wall_grace
    return grace + model.base2 * scaling * model.tol2


def compute_receipt_spread_deadline(model: GpuTimingModel, gpu_count: int) -> float:
    scaling = compute_gpu_count_scaling(model, gpu_count)
    return model.receipt_spread_grace * scaling
