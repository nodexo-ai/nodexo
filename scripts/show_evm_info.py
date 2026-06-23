#!/usr/bin/env python3
"""Show the EVM address and SS58 mirror for a Bittensor hotkey."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the EVM address and SS58 mirror for a Bittensor hotkey",
    )
    parser.add_argument("--wallet", required=True, help="Bittensor wallet name")
    parser.add_argument("--hotkey", required=True, help="Bittensor hotkey name")
    parser.add_argument(
        "--amount",
        default="0.1",
        help="TAO amount to show in the example funding command",
    )
    parser.add_argument(
        "--subtensor-network",
        default="finney",
        help="Network to show in the example funding command",
    )
    args = parser.parse_args()

    from common.chain.wallet import (
        derive_evm_account,
        get_ss58_mirror_address,
        load_hotkey_seed,
    )

    evm_account = derive_evm_account(load_hotkey_seed(args.wallet, args.hotkey))
    mirror_ss58 = get_ss58_mirror_address(evm_account.address)

    print(f"Wallet:      {args.wallet}/{args.hotkey}")
    print(f"EVM address: {evm_account.address}")
    print(f"SS58 mirror: {mirror_ss58}")
    print()
    print("To fund this EVM wallet for gas, transfer TAO to the SS58 mirror:")
    print(
        "  .venv/bin/nodexo "
        f"--wallet {args.wallet} --hotkey {args.hotkey} "
        f"--subtensor-network {args.subtensor_network} "
        f"fund --amount {args.amount}"
    )


if __name__ == "__main__":
    main()
