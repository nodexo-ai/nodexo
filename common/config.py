"""
Nodexo — chain configuration and subnet parameters.

Loads subnet contract addresses and runtime metadata from chain_config JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class ChainConfig:
    """On-chain contract addresses and subnet identity."""

    chain_id: int
    netuid: int
    compute_registry_address: str
    validator_registry_address: str
    collateral_manager_address: str
    subnet_config_address: str
    payment_gateway_address: str = ""
    payment_gateway_deploy_block: int = 0
    rpc_url: str = ""  # Resolved from subtensor network if empty
    compute_registry_deploy_block: int = 0
    mock: bool = False

    _NETWORK_CONFIGS: ClassVar[dict[str, str]] = {
        "test": "chain_config_testnet.json",
        "finney": "chain_config_mainnet.json",
    }
    _NETWORK_RPC_URLS: ClassVar[dict[str, str]] = {
        "test": "https://test.chain.opentensor.ai",
        "finney": "https://lite.chain.opentensor.ai",
    }

    @classmethod
    def resolve_config_path(
        cls,
        chain_config: str | Path | None,
        subtensor_network: str | None,
        repo_root: str | Path | None = None,
    ) -> str | None:
        """Resolve bundled chain config from network.

        Chain config is public deployment metadata. RPC endpoints are resolved
        separately from the selected network or an explicit operator endpoint.
        """
        if chain_config:
            return str(chain_config)
        network = str(subtensor_network or "").strip()
        filename = cls._NETWORK_CONFIGS.get(network)
        if not filename:
            return None
        bases = [Path.cwd()]
        if repo_root:
            bases.append(Path(repo_root))
        bases.append(Path(__file__).resolve().parent.parent)
        for base in bases:
            candidate = base / filename
            if candidate.exists():
                return str(candidate)
        return None

    @classmethod
    def resolve_rpc_url(
        cls,
        subtensor_network: str | None = None,
        chain_endpoint: str | None = None,
        evm_rpc_url: str | None = None,
    ) -> str | None:
        """Resolve the Bittensor EVM HTTP RPC endpoint.

        Priority:
        1. explicit EVM RPC override (`--evm-rpc-url` / env)
        2. explicit Subtensor chain endpoint, translated ws→http
        3. URL-like `--subtensor-network`, translated ws→http
        4. network default (`test`, `finney`)

        The chain config intentionally does not own this value.
        """
        explicit = (evm_rpc_url or "").strip()
        if explicit:
            return explicit
        env_override = (os.environ.get("NODEXO_EVM_RPC_URL") or os.environ.get("EVM_RPC_URL") or "").strip()
        if env_override:
            return env_override

        def _endpoint_to_http(value: str | None) -> str | None:
            raw = str(value or "").strip()
            if not raw:
                return None
            if raw.startswith(("http://", "https://")):
                return raw
            if raw.startswith("ws://"):
                return "http://" + raw[len("ws://"):]
            if raw.startswith("wss://"):
                return "https://" + raw[len("wss://"):]
            return None

        endpoint_rpc = _endpoint_to_http(chain_endpoint) or _endpoint_to_http(subtensor_network)
        if endpoint_rpc:
            return endpoint_rpc

        network = str(subtensor_network or "").strip()
        return cls._NETWORK_RPC_URLS.get(network)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        subtensor_network: str | None = None,
        chain_endpoint: str | None = None,
        rpc_url: str | None = None,
    ) -> ChainConfig:
        data = json.loads(Path(path).read_text())
        resolved_rpc_url = (
            cls.resolve_rpc_url(
                subtensor_network=subtensor_network,
                chain_endpoint=chain_endpoint,
                evm_rpc_url=rpc_url,
            )
            or data.get("rpc_url", "")
        )
        return cls(
            chain_id=data.get("chain_id", 0),
            netuid=data["netuid"],
            compute_registry_address=data["compute_registry_address"],
            validator_registry_address=data["validator_registry_address"],
            collateral_manager_address=data["collateral_manager_address"],
            subnet_config_address=data["subnet_config_address"],
            payment_gateway_address=data.get("payment_gateway_address", ""),
            payment_gateway_deploy_block=int(data.get("payment_gateway_deploy_block") or 0),
            rpc_url=resolved_rpc_url,
            compute_registry_deploy_block=int(
                data.get("compute_registry_deploy_block")
                or data.get("deployment_block")
                or 0
            ),
            mock=data.get("mock", False),
        )

    def to_json(self, path: str | Path) -> None:
        data = {
            "chain_id": self.chain_id,
            "netuid": self.netuid,
            "compute_registry_address": self.compute_registry_address,
            "validator_registry_address": self.validator_registry_address,
            "collateral_manager_address": self.collateral_manager_address,
            "subnet_config_address": self.subnet_config_address,
            "payment_gateway_address": self.payment_gateway_address,
            "payment_gateway_deploy_block": self.payment_gateway_deploy_block,
            "compute_registry_deploy_block": self.compute_registry_deploy_block,
            "mock": self.mock,
        }
        if self.rpc_url:
            data["rpc_url"] = self.rpc_url
        Path(path).write_text(json.dumps(data, indent=2))


# GPU VRAM lookup table (from ZkGEMM proof-of-compute config)
# Used for VRAM-scaled matrix dimension calculation.
GPU_VRAM_GB: dict[str, float] = {
    "NVIDIA GeForce RTX 3060": 12,
    "NVIDIA GeForce RTX 3060 Ti": 8,
    "NVIDIA GeForce RTX 3070": 8,
    "NVIDIA GeForce RTX 3070 Ti": 8,
    "NVIDIA GeForce RTX 3080": 10,
    "NVIDIA GeForce RTX 3080 Ti": 12,
    "NVIDIA GeForce RTX 3090": 24,
    "NVIDIA GeForce RTX 3090 Ti": 24,
    "NVIDIA GeForce RTX 4060 Ti": 16,
    "NVIDIA GeForce RTX 4070": 12,
    "NVIDIA GeForce RTX 4070 Ti": 12,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 16,
    "NVIDIA GeForce RTX 4080": 16,
    "NVIDIA GeForce RTX 4080 SUPER": 16,
    "NVIDIA GeForce RTX 4090": 24,
    "NVIDIA GeForce RTX 5090": 32,
    "NVIDIA RTX 6000 Ada Generation": 48,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 96,
    "NVIDIA RTX A6000": 48,
    "NVIDIA A100-SXM4-40GB": 40,
    "NVIDIA A100-PCIE-40GB": 40,
    "NVIDIA A100-SXM4-80GB": 80,
    "NVIDIA A100 80GB PCIe": 80,
    "NVIDIA L40": 48,
    "NVIDIA L40S": 48,
    "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA H100 NVL": 94,
    "NVIDIA H200": 141,
    "NVIDIA H200 NVL": 141,
    "NVIDIA B200": 192,
    "NVIDIA B300 SXM6": 288,
}


GPU_MODEL_ALIASES: dict[str, str] = {
    # Common CLI/UI shorthands. Keep this conservative: aliases must map to
    # the same physical class, because timing and pricing use the canonical key.
    "RTX 4090": "NVIDIA GeForce RTX 4090",
    "4090": "NVIDIA GeForce RTX 4090",
    "RTX 5090": "NVIDIA GeForce RTX 5090",
    "5090": "NVIDIA GeForce RTX 5090",
    "RTX A6000": "NVIDIA RTX A6000",
    "A6000": "NVIDIA RTX A6000",
    "RTX 6000 Ada": "NVIDIA RTX 6000 Ada Generation",
    "RTX 6000 Ada Generation": "NVIDIA RTX 6000 Ada Generation",
    "RTX PRO 6000": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "RTX PRO 6000 Blackwell": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "L40": "NVIDIA L40",
    "L40S": "NVIDIA L40S",
    "A100 40GB": "NVIDIA A100-SXM4-40GB",
    "A100 80GB": "NVIDIA A100-SXM4-80GB",
    "H100": "NVIDIA H100 80GB HBM3",
    "H100 80GB": "NVIDIA H100 80GB HBM3",
    "H100 80GB HBM3": "NVIDIA H100 80GB HBM3",
    "H100 NVL": "NVIDIA H100 NVL",
    "H200": "NVIDIA H200",
    "H200 NVL": "NVIDIA H200 NVL",
    "B200": "NVIDIA B200",
    "B300": "NVIDIA B300 SXM6",
    "B300 SXM6": "NVIDIA B300 SXM6",
}


def _gpu_model_key(value: str) -> str:
    key = " ".join(str(value or "").strip().split()).lower()
    if key.startswith("nvidia "):
        key = key[len("nvidia "):]
    return key


def canonical_gpu_model_name(value: str | None) -> str:
    """Return the canonical GPU model key used by pricing/timing tables.

    This intentionally avoids fuzzy typo matching. If the miner reports a model
    string we have never seen, keeping it unknown is safer than silently mapping
    it to the wrong timing profile.
    """
    raw = " ".join(str(value or "").strip().split())
    if not raw:
        return raw
    canonical_by_key = {_gpu_model_key(name): name for name in GPU_VRAM_GB}
    aliases_by_key = {
        _gpu_model_key(alias): canonical
        for alias, canonical in GPU_MODEL_ALIASES.items()
        if canonical in GPU_VRAM_GB
    }
    key = _gpu_model_key(raw)
    return aliases_by_key.get(key) or canonical_by_key.get(key) or raw


def gpu_model_hash_hex(value: str | None) -> str:
    """SHA-256 hash used in ComputeRegistry for the canonical model name."""
    return hashlib.sha256(canonical_gpu_model_name(value).encode()).hexdigest()


def gpu_model_hash_bytes(value: str | None) -> bytes:
    return bytes.fromhex(gpu_model_hash_hex(value))


def gpu_model_hash_lookup() -> dict[str, str]:
    """Map known canonical and alias hashes to canonical model names.

    Older miners may have registered the raw NVML string. New miners register
    the canonical name. Validators need to understand both hashes.
    """
    lookup = {
        hashlib.sha256(name.encode()).hexdigest(): name
        for name in GPU_VRAM_GB
    }
    for alias, canonical in GPU_MODEL_ALIASES.items():
        if canonical in GPU_VRAM_GB:
            lookup[hashlib.sha256(alias.strip().encode()).hexdigest()] = canonical
    return lookup


def gpu_model_name_for_hash(hash_hex: str | None) -> str:
    h = str(hash_hex or "").lower().removeprefix("0x")
    return gpu_model_hash_lookup().get(h, "Unknown GPU")


def gpu_model_hashes_for_name(value: str | None) -> set[str]:
    """Return all accepted hashes for a canonical model/user alias."""
    canonical = canonical_gpu_model_name(value)
    if canonical not in GPU_VRAM_GB:
        return set()
    hashes = {hashlib.sha256(canonical.encode()).hexdigest()}
    for alias, alias_canonical in GPU_MODEL_ALIASES.items():
        if alias_canonical == canonical:
            hashes.add(hashlib.sha256(alias.strip().encode()).hexdigest())
    return hashes


# Two-mode proof sizing. These constants MUST stay in sync between
# the miner's ProofService and the validator's verify_recipe — any drift
# rejects every proof, so they live here, not duplicated in either side.
BUFFER_HEAVY = 0.40   # unrented executor: ~40% of VRAM, full security signal
# Micro mode is an ABSOLUTE VRAM budget, not a fraction. If the renter
# pegs the GPU to 99% VRAM, a percentage-based micro (e.g. 0.02 = 2%
# of an A6000's 48 GB = 960 MB) won't fit. With a fixed budget, micro
# proofs survive even when the renter has reserved almost everything.
MICRO_VRAM_GB = 0.5   # ~500 MB target — fits even at 99% renter usage,
                      # still meaningful crypto signal (N≈3500 on any GPU)


def compute_matrix_dim(vram_gb: float, buffer: float = 0.50, block_size: int = 256) -> int:
    """Compute the VRAM-scaled matrix dimension n for ZkGEMM.

    Peak GPU memory during optimized field_gemm_8bit (per N² element):
      A(8) + B(8) + B_parts_cm(8) + A_part(1) + Q(4) + C_cm(8) = 37 bytes
    Plus ~64MB fixed (cublasLt workspaces).

    Use BUFFER_HEAVY (0.40) for unrented executors; for rented executors
    use compute_micro_matrix_dim() which derives a GPU-independent N
    targeting MICRO_VRAM_GB. GPU-agnostic: formula uses peak bytes, works
    for 8GB to 288GB GPUs.

    Reference: proof-of-compute/cbr/config.py lines 71-92
    """
    import math
    PEAK_BYTES_PER_ELEM = 37     # measured peak during field_gemm_8bit
    FIXED_OVERHEAD_BYTES = 64 * 1024 * 1024  # cublasLt workspaces
    OS_RESERVE_GB = 2.0          # reserve for OS + driver + PyTorch allocator

    usable_bytes = max(vram_gb - OS_RESERVE_GB, 0.5) * 1024**3
    available = buffer * usable_bytes - FIXED_OVERHEAD_BYTES
    if available <= 0:
        return block_size

    n_raw = math.sqrt(available / PEAK_BYTES_PER_ELEM)
    n = int(n_raw) // block_size * block_size
    return max(n, block_size)  # At least one block


def compute_micro_matrix_dim(block_size: int = 256) -> int:
    """Compute the matrix dim N for a micro proof (rented executor).

    GPU-independent — uses a fixed VRAM budget (MICRO_VRAM_GB) so that
    even a renter consuming 99% of VRAM leaves room for the proof. Same
    on a 24GB 4090 and a 192GB B200 — the security signal is "you can
    do this much GPU work in a fixed time", not "use most of your VRAM".
    """
    import math
    PEAK_BYTES_PER_ELEM = 37
    FIXED_OVERHEAD_BYTES = 64 * 1024 * 1024
    budget = MICRO_VRAM_GB * 1024**3 - FIXED_OVERHEAD_BYTES
    if budget <= 0:
        return block_size
    n_raw = math.sqrt(budget / PEAK_BYTES_PER_ELEM)
    n = int(n_raw) // block_size * block_size
    return max(n, block_size)
