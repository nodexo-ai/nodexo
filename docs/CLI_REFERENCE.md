# CLI Reference

`nodexo` is the single CLI for renter workflows, account API-key
automation, and miner/operator fleet operations.

Use the generated help for exact flags:

```bash
nodexo --help
nodexo rent --help
nodexo fleet --help
```

## Configuration

Public renter and account commands call the web app API:

```bash
export NODEXO_API_URL=https://nodexo.ai/api
```

For x402 rentals, provide an EVM key for the paying wallet:

```bash
export NODEXO_EVM_PRIVATE_KEY=0x...
```

For account-credit rentals and account reads, create an API key in Account
settings and provide it as a bearer secret:

```bash
export NODEXO_API_KEY=vc_...
```

During public testnet preview, inventory, marketplace, quote, account, and
operator commands are available. Non-admin rent, extend, and top-up commands
return the launch gate before payment signing, credit reserve, or provisioning.

Operator diagnostics may also need:

```bash
export NODEXO_SUBTENSOR_NETWORK=test
export NODEXO_SUBTENSOR_ENDPOINT=ws://127.0.0.1:9944  # optional private RPC
export NODEXO_VALIDATOR_URL=https://validator.nodexo.ai
```

## Inventory And Quotes

```bash
nodexo inventory
nodexo marketplace
nodexo quote --gpu A6000 --duration 1h --ssh-key ~/.ssh/id_ed25519.pub
nodexo quote --gpu H100 --gpu-count 4 --duration 4h --storage-gb 100 --memory-gb 64
```

`inventory` and `marketplace` use the public API by default. Use
`--validator-direct` only for explicit operator diagnostics.

## Create Rentals

Accountless x402 rental:

```bash
nodexo rent \
  --payment x402 \
  --gpu A6000 \
  --duration 1h \
  --ssh-key ~/.ssh/id_ed25519.pub
```

Credit-backed metered rental:

```bash
nodexo rent \
  --payment credits \
  --gpu A6000 \
  --storage-gb 30 \
  --memory-gb 16 \
  --ssh-key ~/.ssh/id_ed25519.pub
```

Useful agent flags:

```bash
nodexo rent \
  --gpu A6000 \
  --duration 1h \
  --idempotency-key agent-job-001 \
  --save ./rental.json \
  --json
```

## Rental Management

Recovery-token flow:

```bash
export NODEXO_RENTAL_RECOVERY_TOKEN=...

nodexo rental-info <rental_id>
nodexo ssh-config <rental_id>
nodexo connect <rental_id>
nodexo rental-extend <rental_id> --hours 4
nodexo rental-key add <rental_id> --ssh-key ~/.ssh/id_ed25519.pub
nodexo rental-key remove <rental_id> --key-text "ssh-ed25519 AAAA..."
nodexo rental-end <rental_id>
```

Account API-key flow:

```bash
export NODEXO_API_KEY=vc_...

nodexo account-rentals
nodexo rental-info <rental_id>
nodexo rental-key add <rental_id> --ssh-key ~/.ssh/id_ed25519.pub
nodexo rental-end <rental_id>
```

## Account Credits And Saved SSH Keys

```bash
nodexo credits
nodexo account-rentals
nodexo account-ssh-keys list
nodexo account-ssh-keys add --ssh-key ~/.ssh/id_ed25519.pub --label workstation
nodexo account-ssh-keys remove --key-id <id>
```

Saved account SSH keys require an account API key with the `keys` scope.
Rental creation with account credits requires the `rent` scope.

## Miner Fleet

Public API fleet view for a miner hotkey:

```bash
nodexo fleet
nodexo --wallet <coldkey> --hotkey <hotkey> fleet
nodexo fleet --watch 20
```

Direct chain diagnostics:

```bash
# Testnet: netuid 468
nodexo fleet --chain-direct --subtensor-network test
nodexo fleet --chain-direct --subtensor-network test --subtensor-endpoint ws://127.0.0.1:9944

# Mainnet: netuid 106
nodexo fleet --chain-direct --subtensor-network finney

nodexo fleet --chain-direct --show-stale
nodexo deregister --executor-id <executor_id_prefix> --wallet <coldkey> --hotkey <hotkey>
```

`--chain-direct` performs registry/metagraph reads and enables deregistration.
The default fleet view avoids direct RPC calls and reads the public API. On
public testnet, subtensor RPC endpoints can rate-limit metagraph reads; use
the default view for routine operator checks and `--chain-direct` for explicit
diagnostics.

When the subnet alpha-stake gate is active, the fleet dashboard shows current
hotkey stake, required stake, and any grace deadline. The requirement is
computed per miner hotkey from all active executors attached to that hotkey.

## Link Miner Hotkey To Account

The web Operator page can link miner hotkeys to a signed-in account. It creates
a one-time challenge and shows a complete `nodexo operator-claim sign ...`
command with that challenge already filled in. Run the generated command on a
machine that has the miner hotkey. It signs with the miner hotkey and does not
move funds or change chain state.

## Validator-Control Debug Commands

These commands are for operators and local debugging, not public renter flows:

```bash
nodexo rentals --ssh-key ~/.ssh/id_ed25519.pub
nodexo rentals --all
nodexo history --since 24h --limit 100
nodexo inventory --validator-direct
nodexo marketplace --validator-direct
```

Broad validator-control access requires `NODEXO_ADMIN_TOKEN` and should stay on
trusted operator networks.

## Funding

```bash
nodexo fund --wallet <coldkey> --hotkey <hotkey> --amount 1.0
```

Use only for operator EVM mirror funding on the selected subnet.
