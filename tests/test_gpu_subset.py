"""Tests for the optional GPU subset filter used by multi-executor hosts.

The ``bittensor`` stub that makes ``hardware_service`` importable lives in
``tests/conftest.py``. ``pynvml`` is imported inside ``detect_gpus()`` and is
stubbed per-test by the ``fake_pynvml`` fixture below.
"""
from __future__ import annotations

import sys
import types

import pytest

from neurons.miner.services import hardware_service as hw
from neurons.miner.services.hardware_service import (
    GPU_INDICES_ENV,
    GPU_UUIDS_ENV,
    GpuInfo,
    _assert_proof_slot_mapping_safe,
    _parse_gpu_filter_env,
    detect_gpus,
    get_or_create_executor_id,
    select_gpu_subset,
)


def make_gpus(count: int = 8) -> list[GpuInfo]:
    """Fake NVML enumeration: `count` GPUs at physical indices 0..count-1."""
    return [
        GpuInfo(
            index=i,
            name="NVIDIA H100 80GB HBM3",
            uuid=f"GPU-0000000{i}-1111-2222-3333-44444444444{i}",
            vram_mb=81559,
            driver_version="550.90.07",
            compute_capability="9.0",
        )
        for i in range(count)
    ]


# ── select_gpu_subset ──────────────────────────────────────────────────


def test_no_filter_returns_input_unchanged():
    gpus = make_gpus()
    for result in (
        select_gpu_subset(gpus),
        select_gpu_subset(gpus, uuids=[], indices=[]),
        select_gpu_subset(gpus, uuids=None, indices=None),
    ):
        # Same GpuInfo objects, in the same order — but a new list, so a
        # caller mutating the result cannot reach back into the input.
        assert result == gpus
        assert all(a is b for a, b in zip(result, gpus, strict=True))
        assert result is not gpus


def test_uuid_filter_keeps_requested_gpus():
    gpus = make_gpus()
    selected = select_gpu_subset(gpus, uuids=[gpus[1].uuid, gpus[4].uuid])
    assert [g.uuid for g in selected] == [gpus[1].uuid, gpus[4].uuid]


def test_uuid_filter_accepts_missing_prefix_and_any_case():
    gpus = make_gpus()
    bare_upper = gpus[3].uuid.removeprefix("GPU-").upper()
    lower_prefixed = gpus[6].uuid.lower()
    selected = select_gpu_subset(gpus, uuids=[bare_upper, lower_prefixed])
    assert [g.index for g in selected] == [3, 6]


def test_uuid_filter_result_follows_nvml_order_not_request_order():
    gpus = make_gpus()
    selected = select_gpu_subset(gpus, uuids=[gpus[5].uuid, gpus[2].uuid])
    assert [g.index for g in selected] == [2, 5]


def test_index_filter_keeps_requested_gpus():
    gpus = make_gpus()
    selected = select_gpu_subset(gpus, indices=[0, 7])
    assert [g.index for g in selected] == [0, 7]


def test_preserves_physical_index_and_does_not_renumber():
    """GpuInfo.index is the key into build_hw_static's per-physical-GPU MIG map.

    It is NOT what pins proof workers — those are pinned by ordinal slot; see
    test_proof_slot_mapping_guard_* below.
    """
    gpus = make_gpus()
    selected = select_gpu_subset(gpus, indices=[2, 5])
    assert [g.index for g in selected] == [2, 5]
    assert [g.index for g in selected] != [0, 1]
    # Same when selecting by UUID.
    by_uuid = select_gpu_subset(gpus, uuids=[gpus[2].uuid, gpus[5].uuid])
    assert [g.index for g in by_uuid] == [2, 5]
    # The GpuInfo objects are the originals, untouched.
    assert selected[0] is gpus[2]
    assert selected[1] is gpus[5]


def test_both_filters_set_raises():
    gpus = make_gpus()
    with pytest.raises(ValueError) as exc:
        select_gpu_subset(gpus, uuids=[gpus[0].uuid], indices=[0])
    assert GPU_UUIDS_ENV in str(exc.value)
    assert GPU_INDICES_ENV in str(exc.value)


def test_unknown_uuid_raises_and_lists_detected():
    gpus = make_gpus(2)
    with pytest.raises(ValueError) as exc:
        select_gpu_subset(gpus, uuids=[gpus[0].uuid, "GPU-deadbeef"])
    message = str(exc.value)
    assert "GPU-deadbeef" in message
    assert gpus[1].uuid in message  # detected set is reported


def test_out_of_range_index_raises_and_lists_detected():
    gpus = make_gpus(4)
    with pytest.raises(ValueError) as exc:
        select_gpu_subset(gpus, indices=[0, 9])
    message = str(exc.value)
    assert "9" in message
    assert "detected: 0, 1, 2, 3" in message


def test_filter_against_empty_detection_raises():
    with pytest.raises(ValueError):
        select_gpu_subset([], indices=[0])
    with pytest.raises(ValueError):
        select_gpu_subset([], uuids=["GPU-00000000"])


def test_no_filter_against_empty_detection_returns_empty():
    assert select_gpu_subset([]) == []


def test_non_strict_drops_missing_entries_instead_of_raising():
    """The heartbeat refresh must report survivors, not claim zero GPUs."""
    gpus = make_gpus(4)
    survivors = [g for g in gpus if g.index != 2]  # GPU 2 fell off the bus
    selected = select_gpu_subset(
        survivors, indices=[1, 2, 3], strict=False,
    )
    assert [g.index for g in selected] == [1, 3]

    by_uuid = select_gpu_subset(
        survivors,
        uuids=[gpus[1].uuid, gpus[2].uuid, gpus[3].uuid],
        strict=False,
    )
    assert [g.index for g in by_uuid] == [1, 3]


def test_non_strict_still_raises_when_nothing_survives():
    gpus = make_gpus(4)
    with pytest.raises(ValueError):
        select_gpu_subset(gpus, indices=[9], strict=False)


# ── proof slot mapping guard ───────────────────────────────────────────


def test_proof_slot_mapping_guard_accepts_a_prefix_subset():
    gpus = make_gpus()
    _assert_proof_slot_mapping_safe(select_gpu_subset(gpus, indices=[0, 1, 2]))
    _assert_proof_slot_mapping_safe(select_gpu_subset(gpus, indices=[0]))
    _assert_proof_slot_mapping_safe(gpus)


def test_proof_slot_mapping_guard_rejects_a_subset_not_starting_at_zero():
    """Proof workers get CUDA_VISIBLE_DEVICES=<slot>, slot in range(gpu_count).

    A subset of physical 4..7 would therefore prove on physical 0..3 — the
    neighbour's silicon — so it must be refused, not silently accepted.
    """
    gpus = make_gpus()
    with pytest.raises(ValueError) as exc:
        _assert_proof_slot_mapping_safe(
            select_gpu_subset(gpus, indices=[4, 5, 6, 7])
        )
    assert "CUDA_VISIBLE_DEVICES" in str(exc.value)


def test_proof_slot_mapping_guard_rejects_a_gap():
    gpus = make_gpus()
    with pytest.raises(ValueError):
        _assert_proof_slot_mapping_safe(select_gpu_subset(gpus, indices=[0, 2]))


# ── _parse_gpu_filter_env ──────────────────────────────────────────────


def test_parse_env_empty_by_default():
    assert _parse_gpu_filter_env({}) == ([], [])
    assert _parse_gpu_filter_env({GPU_UUIDS_ENV: "", GPU_INDICES_ENV: " "}) == ([], [])


def test_parse_env_is_lenient_about_whitespace_and_empty_entries():
    uuids, indices = _parse_gpu_filter_env(
        {GPU_UUIDS_ENV: " GPU-aaa , GPU-bbb ,", GPU_INDICES_ENV: ""}
    )
    assert uuids == ["GPU-aaa", "GPU-bbb"]
    assert indices == []

    uuids, indices = _parse_gpu_filter_env({GPU_INDICES_ENV: "0, 1 ,,2,"})
    assert uuids == []
    assert indices == [0, 1, 2]


def test_parse_env_rejects_non_integer_index():
    with pytest.raises(ValueError) as exc:
        _parse_gpu_filter_env({GPU_INDICES_ENV: "0,one"})
    assert GPU_INDICES_ENV in str(exc.value)


def test_parse_env_rejects_duplicate_entries():
    """"0,0" is a plausible typo for "0,1"; do not silently dedupe it."""
    with pytest.raises(ValueError) as exc:
        _parse_gpu_filter_env({GPU_INDICES_ENV: "0,0"})
    assert GPU_INDICES_ENV in str(exc.value)

    with pytest.raises(ValueError) as exc:
        _parse_gpu_filter_env({GPU_UUIDS_ENV: "GPU-aaa,gpu-AAA"})
    assert GPU_UUIDS_ENV in str(exc.value)


def test_parse_env_reads_os_environ_by_default(monkeypatch):
    monkeypatch.delenv(GPU_UUIDS_ENV, raising=False)
    monkeypatch.delenv(GPU_INDICES_ENV, raising=False)
    assert _parse_gpu_filter_env() == ([], [])
    monkeypatch.setenv(GPU_INDICES_ENV, "3,4")
    assert _parse_gpu_filter_env() == ([], [3, 4])


# ── detect_gpus: where the env vars are read and the policy applied ────


class _FakeHandle:
    def __init__(self, gpu: GpuInfo):
        self.gpu = gpu


class _FakeMem:
    def __init__(self, total_mb: int):
        self.total = total_mb * 1024 * 1024
        self.used = self.total // 4


def _fake_pynvml_module(gpus: list[GpuInfo]):
    mod = types.ModuleType("pynvml")
    mod.nvmlInit = lambda: None
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetCount = lambda: len(gpus)
    mod.nvmlDeviceGetHandleByIndex = lambda i: _FakeHandle(gpus[i])
    mod.nvmlDeviceGetName = lambda h: h.gpu.name
    mod.nvmlDeviceGetUUID = lambda h: h.gpu.uuid
    mod.nvmlDeviceGetMemoryInfo = lambda h: _FakeMem(h.gpu.vram_mb)
    mod.nvmlSystemGetDriverVersion = lambda: gpus[0].driver_version
    mod.nvmlDeviceGetCudaComputeCapability = lambda h: (
        int(h.gpu.compute_capability.split(".")[0]),
        int(h.gpu.compute_capability.split(".")[1]),
    )
    mod.NVML_TEMPERATURE_GPU = 0
    mod.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=42)
    mod.nvmlDeviceGetTemperature = lambda h, sensor: 60
    mod.nvmlDeviceGetPowerUsage = lambda h: 300_000
    return mod


@pytest.fixture
def fake_pynvml(monkeypatch):
    """Install a fake pynvml and clear the filter env for each test."""
    gpus = make_gpus()
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml_module(gpus))
    monkeypatch.delenv(GPU_UUIDS_ENV, raising=False)
    monkeypatch.delenv(GPU_INDICES_ENV, raising=False)
    return gpus


def test_detect_gpus_no_filter_returns_every_device(fake_pynvml, monkeypatch):
    detected = detect_gpus()
    assert [g.index for g in detected] == list(range(8))
    assert [g.uuid for g in detected] == [g.uuid for g in fake_pynvml]


def test_detect_gpus_index_filter_narrows_and_preserves_index(fake_pynvml, monkeypatch):
    monkeypatch.setenv(GPU_INDICES_ENV, "0,1,2")
    detected = detect_gpus()
    assert [g.index for g in detected] == [0, 1, 2]


def test_detect_gpus_uuid_filter_narrows(fake_pynvml, monkeypatch):
    monkeypatch.setenv(
        GPU_UUIDS_ENV, f"{fake_pynvml[0].uuid},{fake_pynvml[1].uuid}",
    )
    detected = detect_gpus()
    assert [g.uuid for g in detected] == [
        fake_pynvml[0].uuid, fake_pynvml[1].uuid,
    ]


def test_detect_gpus_fails_closed_on_both_filters_set(fake_pynvml, monkeypatch):
    monkeypatch.setenv(GPU_UUIDS_ENV, fake_pynvml[0].uuid)
    monkeypatch.setenv(GPU_INDICES_ENV, "0")
    assert detect_gpus() == []


def test_detect_gpus_fails_closed_on_unknown_uuid(fake_pynvml, monkeypatch):
    monkeypatch.setenv(GPU_UUIDS_ENV, "GPU-not-on-this-host")
    assert detect_gpus() == []


def test_detect_gpus_fails_closed_on_out_of_range_index(fake_pynvml, monkeypatch):
    monkeypatch.setenv(GPU_INDICES_ENV, "0,99")
    assert detect_gpus() == []


def test_detect_gpus_fails_closed_on_unsafe_proof_slot_mapping(fake_pynvml, monkeypatch):
    """The config that would prove on a neighbour's GPUs must not start."""
    monkeypatch.setenv(GPU_INDICES_ENV, "4,5,6,7")
    assert detect_gpus() == []
    monkeypatch.setenv(GPU_INDICES_ENV, "0,2")
    assert detect_gpus() == []


def test_detect_gpus_non_strict_keeps_survivors_of_a_vanished_device(monkeypatch):
    """A GPU dropping off the bus must not become a claim of zero GPUs."""
    all_gpus = make_gpus(4)
    survivors = [g for g in all_gpus if g.index != 2]
    # NVML now enumerates 3 devices; re-index them the way the driver would
    # not, keeping the fake simple: the filter is by UUID.
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml_module(survivors))
    monkeypatch.delenv(GPU_INDICES_ENV, raising=False)
    monkeypatch.setenv(
        GPU_UUIDS_ENV,
        ",".join(g.uuid for g in all_gpus[:3]),
    )
    assert detect_gpus(strict=True) == []
    relaxed = detect_gpus(strict=False)
    assert [g.uuid for g in relaxed] == [all_gpus[0].uuid, all_gpus[1].uuid]


# ── /hardware telemetry is scoped to the subset ────────────────────────


def test_gpu_utilization_reports_every_gpu_without_a_filter(fake_pynvml):
    assert [u["index"] for u in hw.get_gpu_utilization()] == list(range(8))


def test_gpu_utilization_is_scoped_to_the_index_filter(fake_pynvml, monkeypatch):
    monkeypatch.setenv(GPU_INDICES_ENV, "0,1")
    assert [u["index"] for u in hw.get_gpu_utilization()] == [0, 1]


def test_gpu_utilization_is_scoped_to_the_uuid_filter(fake_pynvml, monkeypatch):
    """A renter on a split host must not be shown a co-tenant's GPUs."""
    monkeypatch.setenv(GPU_UUIDS_ENV, fake_pynvml[3].uuid)
    assert [u["index"] for u in hw.get_gpu_utilization()] == [3]


def test_mig_summary_any_flags_ignore_gpus_outside_the_subset(monkeypatch):
    """mig_enabled_any drives marketplace tier; a co-tenant must not set it."""
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml_module(make_gpus(4)))
    monkeypatch.setattr(
        hw, "detect_mig_for_gpu",
        lambda handle: {
            "capable": handle.gpu.index == 3,
            "enabled": handle.gpu.index == 3,
            "devices": [],
        },
    )
    scoped = hw.detect_mig_summary([0, 1])
    assert scoped["capable_any"] is False
    assert scoped["enabled_any"] is False
    assert [e["index"] for e in scoped["per_gpu"]] == [0, 1]

    host_wide = hw.detect_mig_summary()
    assert host_wide["enabled_any"] is True


# ── executor identity derived from the subset ──────────────────────────


def _identity_dir(monkeypatch, tmp_path, name: str):
    """Point IDENTITY_PATH at a tmp dir so tests never touch ~/.nodexo.

    IDENTITY_PATH is resolved from NODEXO_DATA_DIR at module import time, so
    patching the attribute is the mechanism here; the env var is set only to
    keep anything reading it directly consistent.
    """
    data_dir = tmp_path / name
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NODEXO_DATA_DIR", str(data_dir))
    monkeypatch.setattr(hw, "IDENTITY_PATH", data_dir / "executor_identity.json")
    return data_dir


def test_disjoint_subsets_yield_different_stable_executor_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(hw, "_get_system_uuid", lambda: "11111111-2222-3333-4444-555555555555")
    gpus = make_gpus()
    first = select_gpu_subset(gpus, indices=[0])
    second = select_gpu_subset(gpus, indices=[1, 2, 3, 4, 5, 6, 7])
    assert not {g.uuid for g in first} & {g.uuid for g in second}

    _identity_dir(monkeypatch, tmp_path, "instance-a")
    id_a = get_or_create_executor_id(first)
    assert get_or_create_executor_id(first) == id_a  # stable across calls

    _identity_dir(monkeypatch, tmp_path, "instance-b")
    id_b = get_or_create_executor_id(second)
    assert get_or_create_executor_id(second) == id_b

    assert id_a != id_b
    # And neither equals the whole-host identity.
    _identity_dir(monkeypatch, tmp_path, "instance-all")
    id_all = get_or_create_executor_id(gpus)
    assert id_all not in (id_a, id_b)


def test_executor_id_refuses_empty_gpu_list_and_leaves_identity_intact(
    monkeypatch, tmp_path,
):
    """A rejected filter must not clobber a registered identity file."""
    monkeypatch.setattr(hw, "_get_system_uuid", lambda: "11111111-2222-3333-4444-555555555555")
    gpus = make_gpus(2)
    _identity_dir(monkeypatch, tmp_path, "instance-a")
    original = get_or_create_executor_id(gpus)
    before = hw.IDENTITY_PATH.read_text()

    with pytest.raises(ValueError) as exc:
        get_or_create_executor_id([])
    assert GPU_UUIDS_ENV in str(exc.value)

    assert hw.IDENTITY_PATH.read_text() == before
    assert get_or_create_executor_id(gpus) == original
