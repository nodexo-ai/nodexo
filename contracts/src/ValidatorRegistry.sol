// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./interfaces/IMetagraph.sol";
import "./interfaces/IUidLookup.sol";
import "./interfaces/IStakingV2.sol";
import {ISr25519Verify} from "./interfaces/ISr25519Verify.sol";
import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {
    OwnableUpgradeable
} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title ValidatorRegistry — stake-gated validator directory for Nodexo
/// @notice Validators register their API endpoints so executors can discover
///         where to broadcast proofs and users can discover rental APIs.
contract ValidatorRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable {
    // ── Precompile addresses ──────────────────────────────────────
    IMetagraph public META;
    IUidLookup public UID_LOOKUP;
    IStakingV2 public STAKING;

    address constant DEFAULT_META = 0x0000000000000000000000000000000000000802;
    address constant DEFAULT_UID_LOOKUP = 0x0000000000000000000000000000000000000806;
    address constant DEFAULT_STAKING_V2 = 0x0000000000000000000000000000000000000805;
    address constant SR25519_VERIFY = 0x0000000000000000000000000000000000000403;

    // ── Subnet ────────────────────────────────────────────────────
    uint16 public netuid;

    // ── Stake thresholds ──────────────────────────────────────────
    uint256 public minValidatorStake;
    uint256 public minProxyStake;
    mapping(address => bool) public whitelisted;

    // ── EVM ↔ UID registration ────────────────────────────────────
    mapping(address => uint16) public evmToUid;
    mapping(address => bool) public evmRegistered;
    mapping(uint16 => address) public uidToEvm;

    // ── Validator data ────────────────────────────────────────────
    struct ValidatorInfo {
        string proxyEndpoint; // HTTPS URL for proof receipt + rental API
        uint16 uid;
        uint64 registeredAt;
        uint64 updatedAt;
        bool isActive;
    }

    mapping(address => ValidatorInfo) public validators;
    address[] public validatorList;
    mapping(address => uint256) private _validatorIndex;

    // ── Default validator ─────────────────────────────────────────
    address public defaultValidator;

    // ── Storage gap ───────────────────────────────────────────────
    uint256[44] private __gap;

    uint256 public constant MAX_ENDPOINT_BYTES = 512;

    // ── Events ────────────────────────────────────────────────────
    event EvmRegistered(address indexed evmAddr, uint16 uid);
    event EvmRegistrationReset(uint16 indexed uid, address oldAddr);
    event ValidatorRegistered(address indexed validator, uint16 uid, string proxyEndpoint);
    event ValidatorUpdated(address indexed validator, string newEndpoint);
    event ValidatorDeactivated(address indexed validator);
    event DefaultValidatorSet(address indexed validator);
    event StakeThresholdsUpdated(uint256 minValidator, uint256 minProxy);
    event WhitelistUpdated(address indexed addr, bool status);

    // ── Constructor ───────────────────────────────────────────────
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // ── Initializer ───────────────────────────────────────────────
    function initialize(
        uint16 _netuid,
        address metaAddr,
        address uidLookupAddr,
        address stakingAddr
    ) external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
        netuid = _netuid;
        META = IMetagraph(metaAddr == address(0) ? DEFAULT_META : metaAddr);
        UID_LOOKUP = IUidLookup(uidLookupAddr == address(0) ? DEFAULT_UID_LOOKUP : uidLookupAddr);
        STAKING = IStakingV2(stakingAddr == address(0) ? DEFAULT_STAKING_V2 : stakingAddr);
    }

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── EVM registration (same SR25519 pattern as ComputeRegistry) ─

    function registerEvm(uint16 uid, bytes32 sigR, bytes32 sigS) external {
        require(uid < META.getUidCount(netuid), "UID does not exist");
        bytes32 hotkey = META.getHotkey(netuid, uid);
        require(hotkey != bytes32(0), "UID has no hotkey");
        bytes32 message = keccak256(abi.encodePacked(msg.sender, uid, netuid, address(this)));
        require(
            ISr25519Verify(SR25519_VERIFY).verify(message, hotkey, sigR, sigS),
            "Invalid SR25519 signature"
        );

        uint16 oldUid = evmToUid[msg.sender];
        if (evmRegistered[msg.sender] && oldUid != uid && uidToEvm[oldUid] == msg.sender) {
            delete uidToEvm[oldUid];
            emit EvmRegistrationReset(oldUid, msg.sender);
        }

        address existing = uidToEvm[uid];
        if (existing != address(0) && existing != msg.sender) {
            delete evmRegistered[existing];
            delete evmToUid[existing];
            emit EvmRegistrationReset(uid, existing);
        }

        evmToUid[msg.sender] = uid;
        evmRegistered[msg.sender] = true;
        uidToEvm[uid] = msg.sender;
        emit EvmRegistered(msg.sender, uid);
    }

    /// @notice Reset a UID registration (key rotation or cleanup).
    function resetEvmRegistration(uint16 uid) external {
        address old = uidToEvm[uid];
        require(old != address(0), "UID not registered");
        require(msg.sender == old || msg.sender == owner(), "Not authorized");
        delete evmRegistered[old];
        delete evmToUid[old];
        delete uidToEvm[uid];
        emit EvmRegistrationReset(uid, old);
    }

    // ── Validator registration ────────────────────────────────────

    /// @notice Register as a validator with an optional proxy endpoint.
    function register(string calldata proxyEndpoint) external {
        require(evmRegistered[msg.sender], "Not EVM-registered");
        uint16 uid = evmToUid[msg.sender];
        require(META.getValidatorStatus(netuid, uid), "Not a validator on this subnet");

        _validateEndpoint(proxyEndpoint);
        bool hasProxy = bytes(proxyEndpoint).length > 0;
        if (!whitelisted[msg.sender]) {
            _requireStake(uid, hasProxy ? minProxyStake : minValidatorStake);
        }

        if (validators[msg.sender].registeredAt == 0) {
            // New registration
            validators[msg.sender] = ValidatorInfo({
                proxyEndpoint: proxyEndpoint,
                uid: uid,
                registeredAt: uint64(block.timestamp),
                updatedAt: uint64(block.timestamp),
                isActive: true
            });
            validatorList.push(msg.sender);
            _validatorIndex[msg.sender] = validatorList.length - 1;
        } else {
            // Update existing
            validators[msg.sender].proxyEndpoint = proxyEndpoint;
            validators[msg.sender].updatedAt = uint64(block.timestamp);
            validators[msg.sender].isActive = true;
        }

        emit ValidatorRegistered(msg.sender, uid, proxyEndpoint);
    }

    /// @notice Update proxy endpoint.
    function updateEndpoint(string calldata newEndpoint) external {
        require(validators[msg.sender].isActive, "Not an active validator");
        _validateEndpoint(newEndpoint);
        validators[msg.sender].proxyEndpoint = newEndpoint;
        validators[msg.sender].updatedAt = uint64(block.timestamp);
        emit ValidatorUpdated(msg.sender, newEndpoint);
    }

    /// @notice Deactivate self.
    function deactivate() external {
        _deactivateValidator(msg.sender);
    }

    /// @notice Deactivate a validator row. Callable by the validator or owner.
    function deactivateValidator(address validator) external {
        require(msg.sender == validator || msg.sender == owner(), "Not authorized");
        _deactivateValidator(validator);
    }

    // ── Discovery ─────────────────────────────────────────────────

    /// @notice Get all active validators with their endpoints.
    function getActiveValidators()
        external
        view
        returns (address[] memory addrs, string[] memory endpoints, uint16[] memory uids)
    {
        uint256 count = 0;
        for (uint256 i = 0; i < validatorList.length; i++) {
            if (validators[validatorList[i]].isActive) count++;
        }

        addrs = new address[](count);
        endpoints = new string[](count);
        uids = new uint16[](count);

        uint256 j = 0;
        for (uint256 i = 0; i < validatorList.length; i++) {
            address v = validatorList[i];
            if (validators[v].isActive) {
                addrs[j] = v;
                endpoints[j] = validators[v].proxyEndpoint;
                uids[j] = validators[v].uid;
                j++;
            }
        }
    }

    /// @notice Get a specific validator's info.
    function getValidator(address addr) external view returns (ValidatorInfo memory) {
        return validators[addr];
    }

    // ── Owner functions ───────────────────────────────────────────

    function setDefaultValidator(address v) external onlyOwner {
        defaultValidator = v;
        emit DefaultValidatorSet(v);
    }

    function setStakeThresholds(uint256 _minValidator, uint256 _minProxy) external onlyOwner {
        require(_minProxy >= _minValidator, "Proxy stake must be >= validator stake");
        minValidatorStake = _minValidator;
        minProxyStake = _minProxy;
        emit StakeThresholdsUpdated(_minValidator, _minProxy);
    }

    function setWhitelisted(address addr, bool status) external onlyOwner {
        whitelisted[addr] = status;
        emit WhitelistUpdated(addr, status);
    }

    // ── Internal ──────────────────────────────────────────────────

    function _requireStake(uint16 uid, uint256 minStake) internal view {
        if (minStake == 0) return;
        bytes32 hotkey = META.getHotkey(netuid, uid);
        uint256 stake = STAKING.getTotalAlphaStaked(hotkey, netuid);
        require(stake >= minStake, "Insufficient alpha stake");
    }

    function _validateEndpoint(string calldata endpoint) internal pure {
        require(bytes(endpoint).length <= MAX_ENDPOINT_BYTES, "Endpoint too long");
    }

    function _deactivateValidator(address validator) internal {
        require(validators[validator].isActive, "Not active");
        validators[validator].isActive = false;
        emit ValidatorDeactivated(validator);
    }
}
