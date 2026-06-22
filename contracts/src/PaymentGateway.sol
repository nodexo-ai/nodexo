// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./interfaces/IStakingV2.sol";
import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {OwnableUpgradeable} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import {ReentrancyGuardUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";

/// @title PaymentGateway — TAO credit deposits for Nodexo
/// @notice Users deposit TAO to buy non-withdrawable compute credits.
///         The web app verifies Deposit events and credits the renter's
///         off-chain account balance. On-chain funds are split between an
///         optional treasury cut and alpha staking for the selected validator.
contract PaymentGateway is Initializable, UUPSUpgradeable, OwnableUpgradeable, ReentrancyGuardUpgradeable {
    IStakingV2 public STAKING;
    address public constant DEFAULT_STAKING = 0x0000000000000000000000000000000000000805;

    uint16 public netuid;
    address public ownerTreasury;
    uint16 public ownerCutBps;

    uint16 public constant MAX_OWNER_CUT_BPS = 2000;

    mapping(address => uint256) public totalDeposited;

    uint256 public emergencyAnnouncedAt;
    uint256 public constant EMERGENCY_TIMELOCK = 7 days;

    event Deposit(address indexed user, bytes32 indexed validatorHotkey, uint256 taoAmount);
    event OwnerCutUpdated(uint16 oldBps, uint16 newBps);
    event TreasuryUpdated(address oldTreasury, address newTreasury);
    event EmergencyAnnounced(uint256 timestamp);
    event EmergencyWithdraw(bytes32 hotkey, uint256 amount);

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(
        uint16 _netuid,
        address _ownerTreasury,
        uint16 _ownerCutBps,
        address stakingAddr
    ) external initializer {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
        __ReentrancyGuard_init();

        require(_ownerTreasury != address(0), "Zero treasury");
        require(_ownerCutBps <= MAX_OWNER_CUT_BPS, "Cut too high");

        STAKING = IStakingV2(stakingAddr == address(0) ? DEFAULT_STAKING : stakingAddr);
        netuid = _netuid;
        ownerTreasury = _ownerTreasury;
        ownerCutBps = _ownerCutBps;
    }

    function _authorizeUpgrade(address) internal override onlyOwner {}

    /// @notice Deposit TAO and stake the non-treasury portion as alpha.
    /// @param validatorHotkey Validator hotkey Sr25519 pubkey receiving stake.
    function deposit(bytes32 validatorHotkey) external payable nonReentrant {
        require(msg.value > 0, "Zero deposit");
        uint256 ownerCut = (msg.value * ownerCutBps) / 10000;
        uint256 stakeCut = msg.value - ownerCut;

        totalDeposited[msg.sender] += msg.value;
        emit Deposit(msg.sender, validatorHotkey, msg.value);

        if (ownerCut > 0) {
            (bool sent,) = payable(ownerTreasury).call{value: ownerCut}("");
            require(sent, "Treasury transfer failed");
        }

        if (stakeCut > 0) {
            uint256 stakeRao = stakeCut / 1e9;
            require(stakeRao > 0, "Stake amount too small");
            // StakingV2 deducts from this contract's mapped balance. The
            // deposit already funded the contract, so forwarding msg.value
            // would double-charge. The amount parameter is RAO-scale.
            STAKING.addStake(validatorHotkey, stakeRao, uint256(netuid));
        }
    }

    function getDeposited(address user) external view returns (uint256) {
        return totalDeposited[user];
    }

    function setOwnerCut(uint16 newBps) external onlyOwner {
        require(newBps <= MAX_OWNER_CUT_BPS, "Cut too high");
        uint16 old = ownerCutBps;
        ownerCutBps = newBps;
        emit OwnerCutUpdated(old, newBps);
    }

    function setTreasury(address newTreasury) external onlyOwner {
        require(newTreasury != address(0), "Zero treasury");
        address old = ownerTreasury;
        ownerTreasury = newTreasury;
        emit TreasuryUpdated(old, newTreasury);
    }

    function announceEmergencyWithdraw() external onlyOwner {
        emergencyAnnouncedAt = block.timestamp;
        emit EmergencyAnnounced(block.timestamp);
    }

    function cancelEmergencyWithdraw() external onlyOwner {
        emergencyAnnouncedAt = 0;
    }

    function executeEmergencyWithdraw(bytes32 hotkey, uint256 amount) external onlyOwner {
        require(emergencyAnnouncedAt > 0, "Not announced");
        require(block.timestamp >= emergencyAnnouncedAt + EMERGENCY_TIMELOCK, "Timelock not expired");
        emergencyAnnouncedAt = 0;
        STAKING.removeStake(hotkey, amount, uint256(netuid));
        emit EmergencyWithdraw(hotkey, amount);
    }

    receive() external payable {}

    uint256[50] private __gap;
}
