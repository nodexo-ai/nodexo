// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title IStakingV2 — Bittensor StakingV2 precompile at 0x0805
/// @notice Stake/unstake TAO as alpha and read alpha stake amounts.
interface IStakingV2 {
    /// @notice Stake TAO as alpha on a subnet for a specific hotkey.
    ///         V2 deducts from the caller's mapped substrate balance.
    ///         Contract callers should receive native TAO first, convert wei
    ///         to RAO for the amount parameter, and not forward msg.value.
    /// @param hotkey The validator/miner hotkey Sr25519 pubkey.
    /// @param amount The amount to stake in RAO (1 TAO = 1e9 RAO).
    /// @param netuid The subnet to stake on.
    function addStake(bytes32 hotkey, uint256 amount, uint256 netuid) external payable;

    /// @notice Remove staked alpha from a subnet.
    /// @param hotkey The hotkey to unstake from.
    /// @param amount The amount to unstake, RAO-scale.
    /// @param netuid The subnet to unstake from.
    function removeStake(bytes32 hotkey, uint256 amount, uint256 netuid) external payable;

    /// @notice Get total alpha staked by a hotkey on a subnet.
    function getTotalAlphaStaked(bytes32 hotkey, uint256 netuid)
        external
        view
        returns (uint256);
}
