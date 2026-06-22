// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title IUidLookup — Bittensor precompile at 0x0806
/// @notice Lookup UIDs associated with an EVM address.
/// @dev Note: has a pagination bug in OTF subtensor (PR #1774) where
///      .take(limit) runs before .filter_map(). Workaround: use a generous limit.
interface IUidLookup {
    /// @notice Find UIDs associated with an EVM address on a subnet.
    /// @param netuid The subnet ID.
    /// @param evmAddress The EVM address to look up.
    /// @param limit Max results to return (set high due to pagination bug).
    /// @return uids Array of UIDs associated with the address.
    function uidLookup(uint16 netuid, address evmAddress, uint16 limit)
        external
        view
        returns (uint16[] memory uids);
}
