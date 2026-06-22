// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC1967Proxy as OZProxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

/// @dev Re-export for Foundry artifact resolution.
contract ERC1967Proxy is OZProxy {
    constructor(address implementation, bytes memory data) OZProxy(implementation, data) {}
}
