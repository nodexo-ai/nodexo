// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title IMetagraph — Bittensor precompile at 0x0802
/// @notice Read-only access to Substrate metagraph state from EVM.
interface IMetagraph {
    /// @notice Returns the number of UIDs registered on a subnet.
    function getUidCount(uint16 netuid) external view returns (uint16);

    /// @notice Returns the SR25519 public key (hotkey) for a given UID.
    function getHotkey(uint16 netuid, uint16 uid) external view returns (bytes32);

    /// @notice Returns whether a UID is a validator (has validator permit).
    function getValidatorStatus(uint16 netuid, uint16 uid) external view returns (bool);
}
