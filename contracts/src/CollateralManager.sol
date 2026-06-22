// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./interfaces/IMetagraph.sol";
import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {OwnableUpgradeable} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title CollateralManager — TAO collateral deposits for Nodexo executors
/// @notice Miners deposit TAO proportional to their GPU value. Collateral serves as
///         anti-Sybil protection and enables slashing for misbehavior. Slash requires
///         multiple validator reports (threshold-based, not unilateral).
///
///         Inspired by Lium's celium_collateral_contracts but fully on-chain with
///         multi-validator slash governance.
contract CollateralManager is Initializable, UUPSUpgradeable, OwnableUpgradeable {

    // ── Precompile ────────────────────────────────────────────────
    IMetagraph public META;
    address constant DEFAULT_META = 0x0000000000000000000000000000000000000802;
    uint16 public netuid;

    // ── Deposit data ──────────────────────────────────────────────

    struct Deposit {
        uint256 amount;
        uint64 depositedAt;      // block.timestamp
        bytes32 executorId;
        bool locked;             // Locked during active rental
    }

    /// @dev miner address → executorId → deposit
    mapping(address => mapping(bytes32 => Deposit)) public deposits;

    // ── Collateral rates ──────────────────────────────────────────
    /// @dev gpuModelHash → required RAO per GPU per day
    ///      Set by owner via setCollateralRate(). Queried by validators.
    mapping(bytes32 => uint256) public collateralRates;

    /// @dev Collateral period in days (how many days of collateral required)
    uint256 public collateralDays;

    // ── Withdrawal cooldown ───────────────────────────────────────
    uint256 public constant WITHDRAWAL_COOLDOWN = 24 hours;
    mapping(address => mapping(bytes32 => uint64)) public withdrawalRequestedAt;

    // ── Slash governance ──────────────────────────────────────────
    uint8 public constant SLASH_THRESHOLD = 3;
    uint64 public constant SLASH_WINDOW = 7200; // ~24h in blocks

    struct SlashReport {
        address reporter;
        uint64 blockNumber;
        uint256 amount;
    }
    mapping(bytes32 => SlashReport[]) public slashReports;
    mapping(address => bool) public slashReporters;
    mapping(bytes32 => mapping(address => uint64)) private _slashReporterLastBlock;
    uint256 public slashedTreasuryBalance;

    // ── Storage gap ───────────────────────────────────────────────
    uint256[41] private __gap;

    // ── Events ────────────────────────────────────────────────────
    event Deposited(address indexed miner, bytes32 indexed executorId, uint256 amount);
    event WithdrawalRequested(address indexed miner, bytes32 indexed executorId);
    event Withdrawn(address indexed miner, bytes32 indexed executorId, uint256 amount);
    event SlashReported(bytes32 indexed executorId, address indexed reporter, uint256 amount);
    event Slashed(bytes32 indexed executorId, uint256 totalAmount);
    event SlashReporterSet(address indexed reporter, bool allowed);
    event CollateralRateSet(bytes32 indexed gpuModelHash, uint256 ratePerGpuPerDay);
    event CollateralDaysSet(uint256 days_);
    event DepositLocked(bytes32 indexed executorId);
    event DepositUnlocked(bytes32 indexed executorId);

    // ── Constructor ───────────────────────────────────────────────
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // ── Initializer ───────────────────────────────────────────────
    function initialize(uint16 _netuid, address metaAddr, uint256 _collateralDays)
        external
        initializer
    {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
        netuid = _netuid;
        META = IMetagraph(metaAddr == address(0) ? DEFAULT_META : metaAddr);
        collateralDays = _collateralDays;
    }

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── Deposit ───────────────────────────────────────────────────

    /// @notice Deposit TAO collateral for an executor.
    function deposit(bytes32 executorId) external payable {
        require(msg.value > 0, "Must send TAO");
        Deposit storage d = deposits[msg.sender][executorId];
        d.amount += msg.value;
        d.depositedAt = uint64(block.timestamp);
        d.executorId = executorId;
        emit Deposited(msg.sender, executorId, msg.value);
    }

    // ── Withdrawal (with cooldown) ────────────────────────────────

    /// @notice Request withdrawal. Must wait WITHDRAWAL_COOLDOWN before claiming.
    function requestWithdrawal(bytes32 executorId) external {
        Deposit storage d = deposits[msg.sender][executorId];
        require(d.amount > 0, "No deposit");
        require(!d.locked, "Deposit locked (active rental)");
        withdrawalRequestedAt[msg.sender][executorId] = uint64(block.timestamp);
        emit WithdrawalRequested(msg.sender, executorId);
    }

    /// @notice Claim withdrawal after cooldown period.
    function withdraw(bytes32 executorId) external {
        Deposit storage d = deposits[msg.sender][executorId];
        require(d.amount > 0, "No deposit");
        require(!d.locked, "Deposit locked");
        uint64 requestedAt = withdrawalRequestedAt[msg.sender][executorId];
        require(requestedAt > 0, "No withdrawal requested");
        require(block.timestamp >= requestedAt + WITHDRAWAL_COOLDOWN, "Cooldown not elapsed");

        uint256 amount = d.amount;
        d.amount = 0;
        delete withdrawalRequestedAt[msg.sender][executorId];

        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok, "Transfer failed");
        emit Withdrawn(msg.sender, executorId, amount);
    }

    // ── Lock/unlock (called by ComputeRegistry during rental) ─────

    function lockDeposit(address miner, bytes32 executorId) external onlyOwner {
        deposits[miner][executorId].locked = true;
        emit DepositLocked(executorId);
    }

    function unlockDeposit(address miner, bytes32 executorId) external onlyOwner {
        deposits[miner][executorId].locked = false;
        emit DepositUnlocked(executorId);
    }

    // ── Slash (multi-validator threshold) ──────────────────────────

    /// @notice Report a slash against an executor. Requires SLASH_THRESHOLD
    ///         owner-approved reporters within SLASH_WINDOW blocks. The
    ///         executed amount is the minimum amount agreed by the quorum.
    function reportSlash(bytes32 executorId, uint256 amount, address miner) external {
        require(slashReporters[msg.sender], "Not a slash reporter");
        require(amount > 0, "Amount must be positive");
        uint64 cutoff = uint64(block.number) > SLASH_WINDOW
            ? uint64(block.number) - SLASH_WINDOW
            : 0;
        uint64 last = _slashReporterLastBlock[executorId][msg.sender];
        require(last == 0 || last < cutoff, "Reporter already reported recently");

        slashReports[executorId].push(SlashReport({
            reporter: msg.sender,
            blockNumber: uint64(block.number),
            amount: amount
        }));
        _slashReporterLastBlock[executorId][msg.sender] = uint64(block.number);

        emit SlashReported(executorId, msg.sender, amount);

        (uint8 uniqueReporters, uint256 quorumAmount) = _recentSlashQuorum(executorId);
        if (uniqueReporters >= SLASH_THRESHOLD) {
            _executeSlash(executorId, miner, quorumAmount);
            delete slashReports[executorId];
        }
    }

    // ── Collateral sufficiency check ──────────────────────────────

    /// @notice Check if a miner has sufficient collateral for an executor.
    function isCollateralSufficient(
        address miner,
        bytes32 executorId,
        bytes32 gpuModelHash,
        uint8 gpuCount
    ) external view returns (bool) {
        uint256 rate = collateralRates[gpuModelHash];
        if (rate == 0) return true; // No rate set = no collateral required
        uint256 required = rate * gpuCount * collateralDays;
        return deposits[miner][executorId].amount >= required;
    }

    /// @notice Get the required collateral amount for an executor.
    function getRequiredCollateral(bytes32 gpuModelHash, uint8 gpuCount)
        external
        view
        returns (uint256)
    {
        return collateralRates[gpuModelHash] * gpuCount * collateralDays;
    }

    // ── Owner functions ───────────────────────────────────────────

    function setCollateralRate(bytes32 gpuModelHash, uint256 ratePerGpuPerDay) external onlyOwner {
        collateralRates[gpuModelHash] = ratePerGpuPerDay;
        emit CollateralRateSet(gpuModelHash, ratePerGpuPerDay);
    }

    function setCollateralDays(uint256 _days) external onlyOwner {
        collateralDays = _days;
        emit CollateralDaysSet(_days);
    }

    function setSlashReporter(address reporter, bool allowed) external onlyOwner {
        slashReporters[reporter] = allowed;
        emit SlashReporterSet(reporter, allowed);
    }

    // ── Internal ──────────────────────────────────────────────────

    function _recentSlashQuorum(bytes32 executorId) internal view returns (uint8, uint256) {
        SlashReport[] storage reports = slashReports[executorId];
        uint64 cutoff = uint64(block.number) > SLASH_WINDOW
            ? uint64(block.number) - SLASH_WINDOW
            : 0;

        // Track unique reporters (max 32 validators expected)
        address[32] memory seen;
        uint8 count = 0;
        uint256 minAmount = type(uint256).max;

        for (uint256 i = reports.length; i > 0; i--) {
            SlashReport storage r = reports[i - 1];
            if (r.blockNumber < cutoff) break;

            bool isDuplicate = false;
            for (uint8 j = 0; j < count; j++) {
                if (seen[j] == r.reporter) {
                    isDuplicate = true;
                    break;
                }
            }
            if (!isDuplicate && count < 32) {
                seen[count] = r.reporter;
                count++;
                if (r.amount < minAmount) {
                    minAmount = r.amount;
                }
            }
        }
        if (minAmount == type(uint256).max) {
            minAmount = 0;
        }
        return (count, minAmount);
    }

    function _executeSlash(bytes32 executorId, address miner, uint256 amount) internal {
        Deposit storage d = deposits[miner][executorId];
        uint256 slashAmount = amount > d.amount ? d.amount : amount;
        d.amount -= slashAmount;
        slashedTreasuryBalance += slashAmount;
        emit Slashed(executorId, slashAmount);
    }

    /// @notice Owner withdraws slashed TAO to treasury.
    function withdrawSlashed(address payable to, uint256 amount) external onlyOwner {
        require(amount <= slashedTreasuryBalance, "Insufficient slashed balance");
        slashedTreasuryBalance -= amount;
        (bool ok,) = to.call{value: amount}("");
        require(ok, "Transfer failed");
    }

    receive() external payable {}
}
