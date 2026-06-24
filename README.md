# Nodexo

Verified decentralized compute on Bittensor.

This repository contains the public operator software for running Nodexo miners
and validators, plus the CLI and setup scripts used to manage them.

Start here:

- [Miner quickstart](docs/MINER_QUICKSTART.md)
- [Validator quickstart](docs/VALIDATOR_QUICKSTART.md)
- [Endpoint security](docs/ENDPOINT_SECURITY.md)
- [CLI reference](docs/CLI_REFERENCE.md)

## Install

Miner host:

```bash
curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash
```

Validator host:

```bash
curl -fsSL https://raw.githubusercontent.com/nodexo-ai/nodexo/main/install.sh | bash -s -- --validator
```

Manual install:

```bash
git clone https://github.com/nodexo-ai/nodexo.git
cd nodexo
bash scripts/setup_miner.sh      # GPU host
bash scripts/setup_validator.sh  # CPU-only validator host
```

## Minimum Host Requirements

| Role | Minimum | Recommended |
|---|---|---|
| Miner | Ubuntu/Debian GPU host, [supported NVIDIA GPU](docs/SUPPORTED_GPUS.md), working `nvidia-smi`, Docker, 4 CPU cores, 16 GB RAM, 40 GB free Docker storage | 8+ CPU cores, 32+ GB RAM, 100+ GB free Docker storage |
| Validator | Ubuntu/Debian host, 4 vCPU, 16 GB RAM, 100 GB disk, synchronized clock, Postgres, dedicated Subtensor RPC | 8+ vCPU, 32+ GB RAM, 200+ GB disk, local or private Subtensor RPC |

Both roles require a registered Bittensor hotkey for the target subnet.
Validators should not use public Subtensor RPC for production operation. Miners
also need the public API port and rental SSH port range reachable from
validators.
Validators do not require third-party API keys for normal proof verification
and weight setting.

## Networks

| Network | Runtime flag | Netuid | Chain config |
|---|---|---:|---|
| Testnet | `--subtensor-network test` | `468` | `chain_config_testnet.json` |
| Mainnet | `--subtensor-network finney` | `106` | `chain_config_mainnet.json` |

The setup scripts select the matching chain config from the network flag.
Operators can override the Subtensor transport with `--subtensor-endpoint` when
using a local node or private RPC endpoint:

```bash
--subtensor-network finney --subtensor-endpoint ws://127.0.0.1:9944
```

Validators should use a dedicated Subtensor endpoint. Public `test` and
`finney` endpoints are only suitable for setup checks or short local tests.

## CLI

For local fleet status after setup:

```bash
nodexo fleet
nodexo inventory
```
