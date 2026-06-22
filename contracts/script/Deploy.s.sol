// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Script.sol";
import "../src/ComputeRegistry.sol";
import "../src/ValidatorRegistry.sol";
import "../src/CollateralManager.sol";
import "../src/SubnetConfig.sol";
import "../src/PaymentGateway.sol";
import "../src/ERC1967Proxy.sol";

/// @title Deploy — deploy all Nodexo contracts as UUPS proxies
/// @notice Usage:
///   forge script script/Deploy.s.sol --rpc-url $RPC_URL --broadcast --private-key $PK
///
/// Environment variables:
///   NETUID         — the Bittensor subnet ID (e.g., 405 for testnet)
///   META_ADDR      — IMetagraph precompile override (0 = default 0x802)
///   UID_LOOKUP_ADDR — IUidLookup precompile override (0 = default 0x806)
///   STAKING_ADDR   — IStakingV2 precompile override (0 = default 0x805)
///   COLLATERAL_DAYS — collateral period in days (default: 7)
///   OWNER_TREASURY — treasury address for TAO credit deposits
///   OWNER_CUT_BPS  — basis points cut from TAO deposits (default: 0, max: 2000)
contract DeployScript is Script {
    function run() external {
        uint16 netuid = uint16(vm.envOr("NETUID", uint256(405)));
        address metaAddr = vm.envOr("META_ADDR", address(0));
        address uidLookupAddr = vm.envOr("UID_LOOKUP_ADDR", address(0));
        address stakingAddr = vm.envOr("STAKING_ADDR", address(0));
        uint256 collateralDays = vm.envOr("COLLATERAL_DAYS", uint256(7));
        address ownerTreasury = vm.envOr("OWNER_TREASURY", msg.sender);
        uint16 ownerCutBps = uint16(vm.envOr("OWNER_CUT_BPS", uint256(0)));

        vm.startBroadcast();

        // ── 1. SubnetConfig ──────────────────────────────────────
        SubnetConfig configImpl = new SubnetConfig();
        ERC1967Proxy configProxy = new ERC1967Proxy(
            address(configImpl),
            abi.encodeCall(SubnetConfig.initialize, ())
        );
        console.log("SubnetConfig:", address(configProxy));

        // ── 2. ComputeRegistry ───────────────────────────────────
        ComputeRegistry computeImpl = new ComputeRegistry();
        ERC1967Proxy computeProxy = new ERC1967Proxy(
            address(computeImpl),
            abi.encodeCall(ComputeRegistry.initialize, (netuid, metaAddr, uidLookupAddr))
        );
        console.log("ComputeRegistry:", address(computeProxy));

        // ── 3. ValidatorRegistry ─────────────────────────────────
        ValidatorRegistry validatorImpl = new ValidatorRegistry();
        ERC1967Proxy validatorProxy = new ERC1967Proxy(
            address(validatorImpl),
            abi.encodeCall(ValidatorRegistry.initialize, (netuid, metaAddr, uidLookupAddr, stakingAddr))
        );
        console.log("ValidatorRegistry:", address(validatorProxy));

        // ── 4. CollateralManager ─────────────────────────────────
        CollateralManager collateralImpl = new CollateralManager();
        ERC1967Proxy collateralProxy = new ERC1967Proxy(
            address(collateralImpl),
            abi.encodeCall(CollateralManager.initialize, (netuid, metaAddr, collateralDays))
        );
        console.log("CollateralManager:", address(collateralProxy));

        // ── 5. PaymentGateway ─────────────────────────────────────
        PaymentGateway paymentImpl = new PaymentGateway();
        ERC1967Proxy paymentProxy = new ERC1967Proxy(
            address(paymentImpl),
            abi.encodeCall(PaymentGateway.initialize, (netuid, ownerTreasury, ownerCutBps, stakingAddr))
        );
        console.log("PaymentGateway:", address(paymentProxy));

        vm.stopBroadcast();

        // ── Print chain config JSON ──────────────────────────────
        console.log("\n--- chain_config.json ---");
        console.log("{");
        console.log('  "chain_id": 945,');
        console.log('  "netuid":', netuid, ",");
        console.log('  "compute_registry_address": "%s",', address(computeProxy));
        console.log('  "validator_registry_address": "%s",', address(validatorProxy));
        console.log('  "collateral_manager_address": "%s",', address(collateralProxy));
        console.log('  "subnet_config_address": "%s",', address(configProxy));
        console.log('  "payment_gateway_address": "%s"', address(paymentProxy));
        console.log("}");
    }
}
