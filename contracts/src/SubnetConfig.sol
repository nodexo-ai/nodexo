// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {OwnableUpgradeable} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title SubnetConfig — on-chain governance parameters for Nodexo
/// @notice Stores tunable subnet parameters: GPU whitelist/weights, scoring weights,
///         burn configuration, proof parameters, TEE allowlist, and blacklists.
///         Read by validators at each tempo for scoring and verification.
contract SubnetConfig is Initializable, UUPSUpgradeable, OwnableUpgradeable {

    // ── GPU whitelist & weights ───────────────────────────────────
    /// @dev gpuModelHash → emission weight (basis points, 0-10000)
    mapping(bytes32 => uint256) public gpuWeights;
    /// @dev gpuModelHash → whether this GPU is allowed on the network
    mapping(bytes32 => bool) public gpuWhitelist;

    // ── Scoring weights (basis points, must sum to 10000) ─────────
    uint256 public proofWeight;         // Default: 4000 (40%)
    uint256 public availabilityWeight;  // Default: 2500 (25%)
    uint256 public hardwareWeight;      // Default: 2500 (25%)
    uint256 public trustWeight;         // Default: 1000 (10%)

    // ── Utilization bonus ─────────────────────────────────────────
    uint256 public utilizationBonusBps; // Default: 2000 (20% bonus when rented)

    // ── Burn / treasury ───────────────────────────────────────────
    uint256 public minerBurnBps;        // Basis points of miner emission to burn/redirect
    address public treasuryAddress;

    // ── Proof parameters ──────────────────────────────────────────
    uint256 public proofEpochBlocks;    // Default: 15 (~180s)
    uint256 public maxBlockAge;         // Default: 256 (Substrate block hash availability)
    uint8 public numChallenges;         // Default: 4 (blocks challenged per epoch)
    uint16 public numSpotChecks;        // Default: 500 (spot checks per block proof)
    uint256 public maxJitterBlocks;     // Default: 5 (per-executor jitter range)

    // ── TEE ───────────────────────────────────────────────────────
    /// @dev platformHash → whether this TEE platform is accepted
    mapping(bytes32 => bool) public teeAllowlist;

    // ── Blacklists ────────────────────────────────────────────────
    mapping(bytes32 => bool) public executorBlacklist;
    mapping(bytes32 => bool) public gpuBlacklist;

    // ── Feature flags ─────────────────────────────────────────────
    mapping(bytes32 => bool) public featureFlags;

    // ── Storage gap ───────────────────────────────────────────────
    uint256[40] private __gap;

    // ── Events ────────────────────────────────────────────────────
    event GpuWhitelistUpdated(bytes32 indexed gpuModelHash, bool allowed, uint256 weight);
    event ScoringWeightsUpdated(uint256 proof, uint256 availability, uint256 hardware, uint256 trust);
    event UtilizationBonusUpdated(uint256 bonusBps);
    event BurnConfigUpdated(uint256 burnBps, address treasury);
    event ProofParamsUpdated(uint256 epochBlocks, uint8 numChallenges, uint16 numSpotChecks);
    event TeeAllowlistUpdated(bytes32 indexed platformHash, bool allowed);
    event ExecutorBlacklistUpdated(bytes32 indexed executorId, bool blacklisted);
    event GpuBlacklistUpdated(bytes32 indexed gpuUuid, bool blacklisted);
    event FeatureFlagUpdated(bytes32 indexed flag, bool enabled);

    // ── Constructor ───────────────────────────────────────────────
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // ── Initializer ───────────────────────────────────────────────
    function initialize() external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();

        // Default scoring weights (must sum to 10000)
        proofWeight = 4000;
        availabilityWeight = 2500;
        hardwareWeight = 2500;
        trustWeight = 1000;

        // Default utilization bonus
        utilizationBonusBps = 2000; // 20%

        // Default proof parameters
        proofEpochBlocks = 15;     // ~180 seconds
        maxBlockAge = 256;
        numChallenges = 4;
        numSpotChecks = 500;
        maxJitterBlocks = 5;

        // No burn by default
        minerBurnBps = 0;
    }

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── GPU management ────────────────────────────────────────────

    /// @notice Add or update a GPU model on the whitelist.
    function setGpuConfig(bytes32 gpuModelHash, bool allowed, uint256 weight) external onlyOwner {
        gpuWhitelist[gpuModelHash] = allowed;
        gpuWeights[gpuModelHash] = weight;
        emit GpuWhitelistUpdated(gpuModelHash, allowed, weight);
    }

    /// @notice Batch-set GPU configs for multiple models.
    function setGpuConfigBatch(
        bytes32[] calldata gpuModelHashes,
        bool[] calldata allowed,
        uint256[] calldata weights
    ) external onlyOwner {
        require(
            gpuModelHashes.length == allowed.length && allowed.length == weights.length,
            "Array length mismatch"
        );
        for (uint256 i = 0; i < gpuModelHashes.length; i++) {
            gpuWhitelist[gpuModelHashes[i]] = allowed[i];
            gpuWeights[gpuModelHashes[i]] = weights[i];
            emit GpuWhitelistUpdated(gpuModelHashes[i], allowed[i], weights[i]);
        }
    }

    // ── Scoring configuration ─────────────────────────────────────

    function setScoringWeights(
        uint256 _proof,
        uint256 _availability,
        uint256 _hardware,
        uint256 _trust
    ) external onlyOwner {
        require(_proof + _availability + _hardware + _trust == 10000, "Must sum to 10000");
        proofWeight = _proof;
        availabilityWeight = _availability;
        hardwareWeight = _hardware;
        trustWeight = _trust;
        emit ScoringWeightsUpdated(_proof, _availability, _hardware, _trust);
    }

    function setUtilizationBonus(uint256 bonusBps) external onlyOwner {
        require(bonusBps <= 5000, "Max 50% bonus"); // Cap to prevent self-rent incentive
        utilizationBonusBps = bonusBps;
        emit UtilizationBonusUpdated(bonusBps);
    }

    // ── Burn configuration ────────────────────────────────────────

    function setBurnConfig(uint256 burnBps, address treasury) external onlyOwner {
        require(burnBps <= 10000, "Cannot burn more than 100%");
        minerBurnBps = burnBps;
        treasuryAddress = treasury;
        emit BurnConfigUpdated(burnBps, treasury);
    }

    // ── Proof parameters ──────────────────────────────────────────

    function setProofParams(
        uint256 _epochBlocks,
        uint8 _numChallenges,
        uint16 _numSpotChecks,
        uint256 _maxJitter
    ) external onlyOwner {
        require(_epochBlocks > 0 && _epochBlocks <= 256, "Epoch must be 1-256 blocks");
        require(_numChallenges > 0 && _numChallenges <= 16, "Challenges must be 1-16");
        proofEpochBlocks = _epochBlocks;
        numChallenges = _numChallenges;
        numSpotChecks = _numSpotChecks;
        maxJitterBlocks = _maxJitter;
        emit ProofParamsUpdated(_epochBlocks, _numChallenges, _numSpotChecks);
    }

    // ── TEE allowlist ─────────────────────────────────────────────

    function setTeeAllowlist(bytes32 platformHash, bool allowed) external onlyOwner {
        teeAllowlist[platformHash] = allowed;
        emit TeeAllowlistUpdated(platformHash, allowed);
    }

    // ── Blacklists ────────────────────────────────────────────────

    function setExecutorBlacklist(bytes32 executorId, bool blacklisted) external onlyOwner {
        executorBlacklist[executorId] = blacklisted;
        emit ExecutorBlacklistUpdated(executorId, blacklisted);
    }

    function setGpuBlacklist(bytes32 gpuUuid, bool blacklisted) external onlyOwner {
        gpuBlacklist[gpuUuid] = blacklisted;
        emit GpuBlacklistUpdated(gpuUuid, blacklisted);
    }

    // ── Feature flags ─────────────────────────────────────────────

    function setFeatureFlag(bytes32 flag, bool enabled) external onlyOwner {
        featureFlags[flag] = enabled;
        emit FeatureFlagUpdated(flag, enabled);
    }

    // ── Batch read (single RPC call for validators) ───────────────

    struct ScoringParams {
        uint256 proofWeight;
        uint256 availabilityWeight;
        uint256 hardwareWeight;
        uint256 trustWeight;
        uint256 utilizationBonusBps;
        uint256 minerBurnBps;
        uint256 proofEpochBlocks;
        uint8 numChallenges;
        uint16 numSpotChecks;
        uint256 maxJitterBlocks;
    }

    /// @notice Single-call read of all scoring parameters.
    function getScoringParams() external view returns (ScoringParams memory) {
        return ScoringParams({
            proofWeight: proofWeight,
            availabilityWeight: availabilityWeight,
            hardwareWeight: hardwareWeight,
            trustWeight: trustWeight,
            utilizationBonusBps: utilizationBonusBps,
            minerBurnBps: minerBurnBps,
            proofEpochBlocks: proofEpochBlocks,
            numChallenges: numChallenges,
            numSpotChecks: numSpotChecks,
            maxJitterBlocks: maxJitterBlocks
        });
    }
}
