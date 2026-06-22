// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ISr25519Verify — Bittensor precompile at 0x0403
/// @notice Verifies SR25519 signatures on-chain. Used to prove hotkey ownership.
interface ISr25519Verify {
    /// @notice Verify an SR25519 signature.
    /// @param message The message that was signed (typically keccak256 of packed data).
    /// @param publicKey The SR25519 public key (Bittensor hotkey as bytes32).
    /// @param r The signature R component.
    /// @param s The signature S component.
    /// @return valid True if the signature is valid.
    function verify(bytes32 message, bytes32 publicKey, bytes32 r, bytes32 s)
        external
        view
        returns (bool valid);
}
