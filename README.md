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

Validators should use a dedicated Subtensor endpoint in production. Public
`test` and `finney` endpoints are supported as a fallback, but they can be
rate-limited.

## CLI

For local fleet status after setup:

```bash
nodexo fleet
nodexo inventory
```
