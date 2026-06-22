// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./interfaces/IMetagraph.sol";
import "./interfaces/IUidLookup.sol";
import {ISr25519Verify} from "./interfaces/ISr25519Verify.sol";
import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {
    OwnableUpgradeable
} from "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

/// @title ComputeRegistry — decentralized executor directory for Nodexo
/// @notice Miners register GPU executors (physical machines) with specs, endpoints,
///         and pricing. Validators discover executors, manage rental state, and
///         report offline machines. Access control uses Bittensor precompiles for
///         SR25519 hotkey ownership verification.
contract ComputeRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable {
    // ── Precompile addresses ──────────────────────────────────────
    address constant DEFAULT_META = 0x0000000000000000000000000000000000000802;
    address constant DEFAULT_UID_LOOKUP = 0x0000000000000000000000000000000000000806;
    address constant SR25519_VERIFY = 0x0000000000000000000000000000000000000403;

    // ── State variables ───────────────────────────────────────────
    IMetagraph public META;
    IUidLookup public UID_LOOKUP;
    uint16 public netuid;

    // ── EVM ↔ UID registration ────────────────────────────────────
    mapping(address => uint16) public evmToUid;
    mapping(address => bool) public evmRegistered;
    mapping(uint16 => address) public uidToEvm;

    // ── Executor registry ─────────────────────────────────────────

    struct ExecutorInfo {
        bytes32 executorId; // Persistent ID: SHA256(gpu_uuids || system_uuid)
        string endpoint; // HTTPS endpoint URL of the executor daemon
        bytes32 gpuModelHash; // keccak256(gpu model string) — cheaper than string keys
        uint8 gpuCount; // Number of GPUs on this machine
        uint32 vramMb; // VRAM per GPU in megabytes
        uint256 pricePerGpuHour; // Price in RAO (1 TAO = 10^9 RAO)
        uint64 registeredAt; // Block number when registered
        uint64 expiresAt; // Heartbeat lease expiry (unix timestamp)
        bool isActive; // Deactivated by offline reports or self-deregister
        bool isRented; // Currently under rental
    }

    struct ExecutorView {
        address miner;
        bytes32 executorId;
        string endpoint;
        bytes32 gpuModelHash;
        uint8 gpuCount;
        uint32 vramMb;
        uint256 pricePerGpuHour;
        uint64 expiresAt;
        bool isActive;
        bool isRented;
        uint16 minerUid;
        bool minerRegistered;
        bool uidOwnerMatches;
    }

    /// @dev Miner address → list of executors they manage
    mapping(address => ExecutorInfo[]) public minerExecutors;

    /// @dev executorId → miner address (prevent duplicate registration)
    mapping(bytes32 => address) public executorOwner;

    /// @dev keccak256(endpoint) → owner address, used to prevent endpoint hijacking
    mapping(bytes32 => address) private _endpointOwner;

    /// @dev All registered executor IDs for enumeration
    bytes32[] public allExecutorIds;

    /// @dev executorId → index in allExecutorIds (for O(1) lookup)
    mapping(bytes32 => uint256) private _executorIndex;

    // ── Offline reporting (with time decay) ───────────────────────

    struct OfflineReport {
        address reporter;
        uint64 blockNumber;
    }

    mapping(bytes32 => OfflineReport[]) public offlineReports;
    uint8 public constant OFFLINE_THRESHOLD = 3;
    uint64 public constant REPORT_DECAY_BLOCKS = 7200; // ~24h at 12s/block

    // ── Constants ─────────────────────────────────────────────────
    uint64 public constant LEASE_DURATION = 24 hours;
    uint256 public constant MAX_ENDPOINT_BYTES = 512;
    uint8 private constant ACTION_REGISTERED = 1;
    uint8 private constant ACTION_RENEWED = 2;
    uint8 private constant ACTION_SPECS_UPDATED = 3;
    uint8 private constant ACTION_ENDPOINT_UPDATED = 4;
    uint8 private constant ACTION_RENTED = 5;
    uint8 private constant ACTION_AVAILABLE = 6;
    uint8 private constant ACTION_DEACTIVATED = 8;

    // ── Production discovery indexes ──────────────────────────────
    /// @dev executorId → index in minerExecutors[owner].
    mapping(bytes32 => uint256) private _minerExecutorIndex;

    /// @dev Active executor IDs for production discovery. Inactive/offline
    ///      executors stay in allExecutorIds for auditability but are removed
    ///      from this list so validators do not scan dead inventory forever.
    bytes32[] public activeExecutorIds;

    /// @dev executorId → index+1 in activeExecutorIds. 0 means not listed.
    mapping(bytes32 => uint256) private _activeExecutorIndexPlusOne;

    /// @dev Incremented when an executor ID is deregistered/recycled so old
    ///      offline reporter de-dupe state does not carry to the new row.
    mapping(bytes32 => uint64) private _offlineReportGeneration;

    /// @dev executorId → reporter → packed(generation, blockNumber).
    mapping(bytes32 => mapping(address => uint256)) private _offlineReporterLast;

    /// @notice True when activeExecutorIds is safe for production discovery.
    /// @dev Fresh deployments set this in initialize(). UUPS upgrades from an
    ///      older implementation must backfill in chunks and then set it true.
    bool public activeExecutorIndexReady;

    // ── Storage gap for future upgrades ───────────────────────────
    uint256[38] private __gap;

    // ── Events ────────────────────────────────────────────────────
    event EvmRegistered(address indexed evmAddr, uint16 uid);
    event EvmRegistrationReset(uint16 indexed uid, address oldAddr);
    event ExecutorRegistered(
        address indexed miner, bytes32 indexed executorId, bytes32 gpuModelHash, uint8 gpuCount
    );
    event ExecutorDeregistered(address indexed miner, bytes32 indexed executorId);
    event ExecutorRenewed(address indexed miner, bytes32 indexed executorId);
    event ExecutorSpecsUpdated(address indexed miner, bytes32 indexed executorId);
    event EndpointClaimed(address indexed miner, bytes32 indexed endpointHash);
    event EndpointReleased(address indexed miner, bytes32 indexed endpointHash);
    event RentalStarted(bytes32 indexed executorId, address indexed validator);
    event RentalEnded(bytes32 indexed executorId, address indexed validator);
    event OfflineReported(bytes32 indexed executorId, address indexed reporter, uint8 recentCount);
    event ExecutorDeactivated(bytes32 indexed executorId, string reason);
    event ActiveExecutorIndexReady(bool ready);
    event ExecutorStateChanged(
        address indexed miner,
        bytes32 indexed executorId,
        uint8 indexed action,
        string endpoint,
        bytes32 gpuModelHash,
        uint8 gpuCount,
        uint32 vramMb,
        uint256 pricePerGpuHour,
        uint64 expiresAt,
        bool isActive,
        bool isRented
    );

    // ── Modifiers ─────────────────────────────────────────────────

    modifier onlyRegisteredNeuron() {
        require(evmRegistered[msg.sender], "Not EVM-registered");
        _;
    }

    modifier onlyValidator() {
        uint16 uid = evmToUid[msg.sender];
        require(evmRegistered[msg.sender], "Not EVM-registered");
        require(META.getValidatorStatus(netuid, uid), "Not a validator");
        _;
    }

    // ── Constructor (disable initializers for UUPS) ───────────────

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // ── Initializer ───────────────────────────────────────────────

    function initialize(uint16 _netuid, address metaAddr, address uidLookupAddr)
        external
        initializer
    {
        __Ownable_init(msg.sender);
        __UUPSUpgradeable_init();
        netuid = _netuid;
        META = IMetagraph(metaAddr == address(0) ? DEFAULT_META : metaAddr);
        UID_LOOKUP = IUidLookup(uidLookupAddr == address(0) ? DEFAULT_UID_LOOKUP : uidLookupAddr);
        activeExecutorIndexReady = true;
    }

    function _authorizeUpgrade(address) internal override onlyOwner {}

    // ── EVM ↔ UID registration (SR25519 identity proof) ───────────

    /// @notice Register EVM address → UID mapping with hotkey ownership proof.
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

    // ── Executor Registration ─────────────────────────────────────

    /// @notice Register a new executor (GPU machine) under this miner.
    function registerExecutor(
        bytes32 executorId,
        string calldata endpoint,
        bytes32 gpuModelHash,
        uint8 gpuCount,
        uint32 vramMb,
        uint256 pricePerGpuHour
    ) external onlyRegisteredNeuron {
        require(executorOwner[executorId] == address(0), "Executor already registered");
        require(gpuCount > 0 && gpuCount <= 16, "Invalid GPU count");
        require(vramMb > 0, "VRAM must be positive");
        _claimEndpoint(endpoint);

        minerExecutors[msg.sender].push(
            ExecutorInfo({
                executorId: executorId,
                endpoint: endpoint,
                gpuModelHash: gpuModelHash,
                gpuCount: gpuCount,
                vramMb: vramMb,
                pricePerGpuHour: pricePerGpuHour,
                registeredAt: uint64(block.number),
                expiresAt: uint64(block.timestamp + LEASE_DURATION),
                isActive: true,
                isRented: false
            })
        );

        executorOwner[executorId] = msg.sender;
        _minerExecutorIndex[executorId] = minerExecutors[msg.sender].length - 1;
        allExecutorIds.push(executorId);
        _executorIndex[executorId] = allExecutorIds.length - 1;
        _addActiveExecutor(executorId);

        emit ExecutorRegistered(msg.sender, executorId, gpuModelHash, gpuCount);
        _emitExecutorStateChanged(
            msg.sender,
            minerExecutors[msg.sender][minerExecutors[msg.sender].length - 1],
            ACTION_REGISTERED
        );
    }

    /// @notice Deregister an executor (miner removes it from the network).
    /// @dev Fully removes the entry: swap-and-pops from both the per-miner
    ///      array and the global `allExecutorIds` array, releases the
    ///      endpoint claim, and clears `executorOwner`. After this call the
    ///      same `executorId` can be freshly registered by anyone.
    ///      Soft offline-flagging is handled separately by `_deactivateExecutor`
    ///      (called from `reportOffline`); that path keeps the row visible
    ///      so operators can see they were flagged before they choose to
    ///      clean up.
    function deregisterExecutor(bytes32 executorId) external {
        require(executorOwner[executorId] == msg.sender, "Not your executor");
        (uint256 idx, bool found) = _findExecutorIndex(msg.sender, executorId);
        require(found, "Executor not found");

        ExecutorInfo[] storage execs = minerExecutors[msg.sender];
        require(!execs[idx].isRented, "Cannot deregister while rented");

        // Release endpoint claim BEFORE the swap — the storage string at
        // `execs[idx].endpoint` is the one we need to hash, and it'll be
        // overwritten by the swap below.
        _releaseEndpoint(execs[idx].endpoint);

        // Per-miner array: swap with last, pop.
        uint256 lastIdx = execs.length - 1;
        if (idx != lastIdx) {
            bytes32 movedExecutorId = execs[lastIdx].executorId;
            execs[idx] = execs[lastIdx];
            _minerExecutorIndex[movedExecutorId] = idx;
        }
        execs.pop();
        delete _minerExecutorIndex[executorId];

        // Global allExecutorIds: swap with last, pop, fix moved entry's index.
        uint256 globalIdx = _executorIndex[executorId];
        uint256 lastGlobal = allExecutorIds.length - 1;
        if (globalIdx != lastGlobal) {
            bytes32 movedId = allExecutorIds[lastGlobal];
            allExecutorIds[globalIdx] = movedId;
            _executorIndex[movedId] = globalIdx;
        }
        allExecutorIds.pop();
        delete _executorIndex[executorId];

        delete executorOwner[executorId];
        delete offlineReports[executorId];
        _offlineReportGeneration[executorId] += 1;
        _removeActiveExecutor(executorId);

        emit ExecutorDeregistered(msg.sender, executorId);
    }

    /// @notice Update executor specs (e.g., GPU upgrade).
    function updateExecutorSpecs(
        bytes32 executorId,
        bytes32 newGpuModelHash,
        uint8 newGpuCount,
        uint32 newVramMb,
        uint256 newPrice
    ) external {
        require(executorOwner[executorId] == msg.sender, "Not your executor");
        (uint256 idx, bool found) = _findExecutorIndex(msg.sender, executorId);
        require(found, "Executor not found");

        ExecutorInfo storage exec = minerExecutors[msg.sender][idx];
        require(exec.isActive, "Executor not active");
        require(!exec.isRented, "Cannot update while rented");
        require(newGpuCount > 0 && newGpuCount <= 16, "Invalid GPU count");
        require(newVramMb > 0, "VRAM must be positive");

        exec.gpuModelHash = newGpuModelHash;
        exec.gpuCount = newGpuCount;
        exec.vramMb = newVramMb;
        exec.pricePerGpuHour = newPrice;

        emit ExecutorSpecsUpdated(msg.sender, executorId);
        _emitExecutorStateChanged(msg.sender, exec, ACTION_SPECS_UPDATED);
    }

    /// @notice Update executor endpoint URL.
    function updateEndpoint(bytes32 executorId, string calldata newEndpoint) external {
        require(executorOwner[executorId] == msg.sender, "Not your executor");
        (uint256 idx, bool found) = _findExecutorIndex(msg.sender, executorId);
        require(found, "Executor not found");

        ExecutorInfo storage exec = minerExecutors[msg.sender][idx];
        require(exec.isActive, "Executor not active");

        _releaseEndpoint(exec.endpoint);
        _claimEndpoint(newEndpoint);
        exec.endpoint = newEndpoint;
        _emitExecutorStateChanged(msg.sender, exec, ACTION_ENDPOINT_UPDATED);
    }

    // ── Heartbeat (lease renewal) ─────────────────────────────────

    /// @notice Renew the lease on an executor. Must be called before expiry.
    function renewExecutor(bytes32 executorId) external {
        require(executorOwner[executorId] == msg.sender, "Not your executor");
        (uint256 idx, bool found) = _findExecutorIndex(msg.sender, executorId);
        require(found, "Executor not found");

        ExecutorInfo storage exec = minerExecutors[msg.sender][idx];
        require(exec.isActive, "Executor not active");
        exec.expiresAt = uint64(block.timestamp + LEASE_DURATION);

        emit ExecutorRenewed(msg.sender, executorId);
        _emitExecutorStateChanged(msg.sender, exec, ACTION_RENEWED);
    }

    // ── Rental state ──────────────────────────────────────────────

    /// @notice Mark an executor as rented. Called by validator at rental start.
    ///         On-chain first = atomic lock. If tx reverts, executor was already rented.
    function markRented(bytes32 executorId) external onlyValidator {
        address owner_ = executorOwner[executorId];
        require(owner_ != address(0), "Executor not registered");
        (uint256 idx, bool found) = _findExecutorIndex(owner_, executorId);
        require(found, "Executor not found");

        ExecutorInfo storage exec = minerExecutors[owner_][idx];
        require(exec.isActive, "Executor not active");
        require(!exec.isRented, "Already rented");
        require(exec.expiresAt > block.timestamp, "Lease expired");

        exec.isRented = true;
        emit RentalStarted(executorId, msg.sender);
        _emitExecutorStateChanged(owner_, exec, ACTION_RENTED);
    }

    /// @notice Mark an executor as available. Called by validator at rental end.
    function markAvailable(bytes32 executorId) external onlyValidator {
        address owner_ = executorOwner[executorId];
        require(owner_ != address(0), "Executor not registered");
        (uint256 idx, bool found) = _findExecutorIndex(owner_, executorId);
        require(found, "Executor not found");

        ExecutorInfo storage exec = minerExecutors[owner_][idx];
        exec.isRented = false;
        emit RentalEnded(executorId, msg.sender);
        _emitExecutorStateChanged(owner_, exec, ACTION_AVAILABLE);
    }

    // ── Offline reporting (with time-based decay) ─────────────────

    /// @notice Report an executor as offline. After OFFLINE_THRESHOLD recent
    ///         reports (within REPORT_DECAY_BLOCKS), the executor is deactivated.
    ///         Once the on-chain heartbeat lease has expired, one validator may
    ///         deactivate it; a live miner prevents this by renewing the lease.
    function reportOffline(bytes32 executorId) external onlyValidator {
        address owner_ = executorOwner[executorId];
        require(owner_ != address(0), "Executor not registered");
        (uint256 idx, bool found) = _findExecutorIndex(owner_, executorId);
        bool leaseExpired =
            found && minerExecutors[owner_][idx].expiresAt <= uint64(block.timestamp);
        bool cleanupExpiredActive = leaseExpired && minerExecutors[owner_][idx].isActive;
        uint64 cutoff = uint64(block.number) > REPORT_DECAY_BLOCKS
            ? uint64(block.number) - REPORT_DECAY_BLOCKS
            : 0;
        uint64 generation = _offlineReportGeneration[executorId];
        uint256 last = _offlineReporterLast[executorId][msg.sender];
        uint256 lastGeneration = last >> 64;
        uint256 lastBlock = last & type(uint64).max;
        require(
            cleanupExpiredActive || last == 0 || lastGeneration != uint256(generation)
                || lastBlock < cutoff,
            "Validator already reported recently"
        );

        offlineReports[executorId].push(
            OfflineReport({reporter: msg.sender, blockNumber: uint64(block.number)})
        );
        _offlineReporterLast[executorId][msg.sender] =
            (uint256(generation) << 64) | uint64(block.number);

        uint8 recentCount = _countRecentReports(executorId);
        if (leaseExpired) {
            _deactivateExecutor(executorId, "lease_expired");
        } else if (recentCount >= OFFLINE_THRESHOLD) {
            _deactivateExecutor(executorId, "offline_threshold");
        }

        emit OfflineReported(executorId, msg.sender, recentCount);
    }

    // ── Discovery queries (view functions) ────────────────────────

    /// @notice Get all executor IDs (for enumeration — use with getExecutorInfo).
    function getExecutorCount() external view returns (uint256) {
        return allExecutorIds.length;
    }

    /// @notice Get count of active-listed executor IDs.
    /// @dev Expired leases are still present until validators report them
    ///      offline or miners deregister. Callers must still check expiresAt.
    function getActiveExecutorCount() external view returns (uint256) {
        return activeExecutorIds.length;
    }

    /// @notice Get executor info by ID.
    function getExecutorInfo(bytes32 executorId)
        external
        view
        returns (
            address miner,
            string memory endpoint,
            bytes32 gpuModelHash,
            uint8 gpuCount,
            uint32 vramMb,
            uint256 pricePerGpuHour,
            uint64 expiresAt,
            bool isActive,
            bool isRented
        )
    {
        address owner_ = executorOwner[executorId];
        require(owner_ != address(0), "Executor not registered");
        (uint256 idx, bool found) = _findExecutorIndex(owner_, executorId);
        require(found, "Executor not found");

        ExecutorInfo storage exec = minerExecutors[owner_][idx];
        return (
            owner_,
            exec.endpoint,
            exec.gpuModelHash,
            exec.gpuCount,
            exec.vramMb,
            exec.pricePerGpuHour,
            exec.expiresAt,
            exec.isActive,
            exec.isRented
        );
    }

    /// @notice Get all executors for a specific miner.
    function getMinerExecutors(address miner) external view returns (ExecutorInfo[] memory) {
        return minerExecutors[miner];
    }

    /// @notice Get paginated executor IDs.
    function getExecutorIdsPaginated(uint256 offset, uint256 limit)
        external
        view
        returns (bytes32[] memory)
    {
        uint256 total = allExecutorIds.length;
        if (offset >= total) return new bytes32[](0);
        uint256 count = limit;
        if (offset + count > total) count = total - offset;
        bytes32[] memory result = new bytes32[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = allExecutorIds[offset + i];
        }
        return result;
    }

    /// @notice Get paginated active-listed executor IDs.
    function getActiveExecutorIdsPaginated(uint256 offset, uint256 limit)
        external
        view
        returns (bytes32[] memory)
    {
        uint256 total = activeExecutorIds.length;
        if (offset >= total) return new bytes32[](0);
        uint256 count = limit;
        if (offset + count > total) count = total - offset;
        bytes32[] memory result = new bytes32[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = activeExecutorIds[offset + i];
        }
        return result;
    }

    /// @notice Get paginated executor details in one eth_call.
    /// @dev Validators should prefer getActiveExecutorsPaginated when
    ///      activeExecutorIndexReady=true. This all-row view is retained for
    ///      audit/debug and older deployments.
    function getExecutorsPaginated(uint256 offset, uint256 limit)
        external
        view
        returns (ExecutorView[] memory)
    {
        uint256 total = allExecutorIds.length;
        if (offset >= total) return new ExecutorView[](0);
        uint256 count = limit;
        if (offset + count > total) count = total - offset;

        ExecutorView[] memory result = new ExecutorView[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = _executorView(allExecutorIds[offset + i]);
        }
        return result;
    }

    /// @notice Get paginated active-listed executor details in one eth_call.
    /// @dev Production validator discovery should prefer this over
    ///      getExecutorsPaginated because inactive/offline rows are excluded
    ///      from the iteration set.
    function getActiveExecutorsPaginated(uint256 offset, uint256 limit)
        external
        view
        returns (ExecutorView[] memory)
    {
        uint256 total = activeExecutorIds.length;
        if (offset >= total) return new ExecutorView[](0);
        uint256 count = limit;
        if (offset + count > total) count = total - offset;

        ExecutorView[] memory result = new ExecutorView[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = _executorView(activeExecutorIds[offset + i]);
        }
        return result;
    }

    /// @notice Backfill activeExecutorIds after a UUPS upgrade from an older
    ///         implementation. Safe to call in chunks.
    function rebuildActiveExecutorIndex(uint256 offset, uint256 limit) external onlyOwner {
        uint256 total = allExecutorIds.length;
        if (offset >= total) return;
        uint256 count = limit;
        if (offset + count > total) count = total - offset;
        for (uint256 i = 0; i < count; i++) {
            bytes32 executorId = allExecutorIds[offset + i];
            address owner_ = executorOwner[executorId];
            (uint256 idx, bool found) = _findExecutorIndex(owner_, executorId);
            if (owner_ != address(0) && found && minerExecutors[owner_][idx].isActive) {
                _addActiveExecutor(executorId);
            } else {
                _removeActiveExecutor(executorId);
            }
        }
    }

    /// @notice Enable/disable the active-executor discovery index.
    /// @dev After UUPS upgrade, leave false until rebuildActiveExecutorIndex()
    ///      has covered the whole allExecutorIds range. Fresh deployments are
    ///      marked ready by initialize().
    function setActiveExecutorIndexReady(bool ready) external onlyOwner {
        activeExecutorIndexReady = ready;
        emit ActiveExecutorIndexReady(ready);
    }

    // ── Internal helpers ──────────────────────────────────────────

    function _executorView(bytes32 executorId) internal view returns (ExecutorView memory view_) {
        address owner_ = executorOwner[executorId];
        (uint256 idx, bool found) = _findExecutorIndex(owner_, executorId);
        if (owner_ == address(0) || !found) {
            view_.executorId = executorId;
            return view_;
        }
        ExecutorInfo storage exec = minerExecutors[owner_][idx];
        uint16 uid = evmToUid[owner_];
        bool registered = evmRegistered[owner_];
        return ExecutorView({
            miner: owner_,
            executorId: exec.executorId,
            endpoint: exec.endpoint,
            gpuModelHash: exec.gpuModelHash,
            gpuCount: exec.gpuCount,
            vramMb: exec.vramMb,
            pricePerGpuHour: exec.pricePerGpuHour,
            expiresAt: exec.expiresAt,
            isActive: exec.isActive,
            isRented: exec.isRented,
            minerUid: uid,
            minerRegistered: registered,
            uidOwnerMatches: registered && uidToEvm[uid] == owner_
        });
    }

    function _claimEndpoint(string calldata endpoint) internal {
        uint256 len = bytes(endpoint).length;
        require(len > 0, "Endpoint required");
        require(len <= MAX_ENDPOINT_BYTES, "Endpoint too long");
        bytes32 h = keccak256(bytes(endpoint));
        address current = _endpointOwner[h];
        require(current == address(0) || current == msg.sender, "Endpoint owned by another miner");
        if (current == address(0)) {
            _endpointOwner[h] = msg.sender;
            emit EndpointClaimed(msg.sender, h);
        }
    }

    function _releaseEndpoint(string storage endpoint) internal {
        bytes32 h = keccak256(bytes(endpoint));
        if (_endpointOwner[h] == msg.sender) {
            delete _endpointOwner[h];
            emit EndpointReleased(msg.sender, h);
        }
    }

    function _addActiveExecutor(bytes32 executorId) internal {
        if (_activeExecutorIndexPlusOne[executorId] != 0) return;
        activeExecutorIds.push(executorId);
        _activeExecutorIndexPlusOne[executorId] = activeExecutorIds.length;
    }

    function _removeActiveExecutor(bytes32 executorId) internal {
        uint256 indexPlusOne = _activeExecutorIndexPlusOne[executorId];
        if (indexPlusOne == 0) return;
        uint256 idx = indexPlusOne - 1;
        uint256 lastIdx = activeExecutorIds.length - 1;
        if (idx != lastIdx) {
            bytes32 movedId = activeExecutorIds[lastIdx];
            activeExecutorIds[idx] = movedId;
            _activeExecutorIndexPlusOne[movedId] = indexPlusOne;
        }
        activeExecutorIds.pop();
        delete _activeExecutorIndexPlusOne[executorId];
    }

    function _emitExecutorStateChanged(
        address miner,
        ExecutorInfo storage exec,
        uint8 action
    ) internal {
        emit ExecutorStateChanged(
            miner,
            exec.executorId,
            action,
            exec.endpoint,
            exec.gpuModelHash,
            exec.gpuCount,
            exec.vramMb,
            exec.pricePerGpuHour,
            exec.expiresAt,
            exec.isActive,
            exec.isRented
        );
    }

    function _findExecutorIndex(address miner, bytes32 executorId)
        internal
        view
        returns (uint256, bool)
    {
        ExecutorInfo[] storage execs = minerExecutors[miner];
        uint256 idx = _minerExecutorIndex[executorId];
        if (idx < execs.length && execs[idx].executorId == executorId) {
            return (idx, true);
        }
        for (uint256 i = 0; i < execs.length; i++) {
            if (execs[i].executorId == executorId) return (i, true);
        }
        return (0, false);
    }

    function _countRecentReports(bytes32 executorId) internal view returns (uint8) {
        OfflineReport[] storage reports = offlineReports[executorId];
        uint8 count = 0;
        uint64 cutoff = uint64(block.number) > REPORT_DECAY_BLOCKS
            ? uint64(block.number) - REPORT_DECAY_BLOCKS
            : 0;
        // Count unique validators from the end (most recent first). This keeps
        // the quorum semantics correct even if an upgraded deployment already
        // contains duplicate historical reports from the same validator.
        for (uint256 i = reports.length; i > 0; i--) {
            OfflineReport storage report = reports[i - 1];
            if (report.blockNumber < cutoff) {
                break; // Reports are chronological, stop at first old one
            }
            bool duplicateReporter = false;
            for (uint256 j = i; j < reports.length; j++) {
                if (reports[j].reporter == report.reporter) {
                    duplicateReporter = true;
                    break;
                }
            }
            if (!duplicateReporter) {
                count++;
                if (count >= OFFLINE_THRESHOLD) return count;
            }
        }
        return count;
    }

    function _deactivateExecutor(bytes32 executorId, string memory reason) internal {
        address owner_ = executorOwner[executorId];
        (uint256 idx, bool found) = _findExecutorIndex(owner_, executorId);
        if (found) {
            minerExecutors[owner_][idx].isActive = false;
            _removeActiveExecutor(executorId);
            _emitExecutorStateChanged(owner_, minerExecutors[owner_][idx], ACTION_DEACTIVATED);
        }
        emit ExecutorDeactivated(executorId, reason);
    }
}
