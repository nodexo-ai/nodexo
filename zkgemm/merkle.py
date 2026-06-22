"""
Merkle tree utilities for the block-committed ZkGEMM protocol.

Handles:
- Path extraction from the flat tree produced by block_merkle_commit
- Path verification (CPU-side, used by verifier and EVM contract)
- Block hashing to reproduce leaf hashes from C_block data

The flat tree layout matches the CUDA implementation:
  [leaves | level1 | level2 | ... | root]
  Each node is 32 bytes (SHA-256).
  Odd levels use "odd-dup": the last node pairs with itself.
"""

from __future__ import annotations

import hashlib
import struct


# Domain tag matching the CUDA constant
_DOM_TAG = b"ZKGEMM_BLK_V1"


def _level_sizes(num_leaves: int) -> list[int]:
    """Compute the size of each level of the Merkle tree."""
    sizes = []
    n = num_leaves
    while True:
        sizes.append(n)
        if n <= 1:
            break
        n = (n + 1) >> 1
    return sizes


def _level_offsets(num_leaves: int) -> list[int]:
    """Compute byte offset of each level in the flat tree."""
    sizes = _level_sizes(num_leaves)
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return offsets


def extract_merkle_path(
    flat_tree: bytes, num_leaves: int, leaf_index: int
) -> list[tuple[bytes, bool]]:
    """Extract a Merkle authentication path for the given leaf.

    Args:
        flat_tree: The flat tree as bytes (from block_merkle_commit).
        num_leaves: Total number of leaves.
        leaf_index: Index of the leaf to extract the path for.

    Returns:
        List of (sibling_hash, is_left) tuples from leaf to root.
        is_left=True means the sibling is to the left (we are the right child).
    """
    offsets = _level_offsets(num_leaves)
    sizes = _level_sizes(num_leaves)

    path = []
    idx = leaf_index
    for level in range(len(sizes) - 1):
        level_start = offsets[level] * 32
        level_count = sizes[level]

        # Determine sibling index
        if idx % 2 == 0:
            # We are left child, sibling is right
            sib_idx = idx + 1
            if sib_idx >= level_count:
                sib_idx = idx  # odd-dup: pair with self
            is_left = False
        else:
            # We are right child, sibling is left
            sib_idx = idx - 1
            is_left = True

        sib_hash = flat_tree[level_start + sib_idx * 32:
                             level_start + sib_idx * 32 + 32]
        path.append((sib_hash, is_left))

        # Move to parent index
        idx = idx >> 1

    return path


def hash_block(C_block_flat: list[int], bs: int) -> bytes:
    """Hash a bs×bs block of C values the same way the CUDA kernel does.

    The CUDA kernel hashes each row as raw int64 bytes (little-endian),
    then the leaf is SHA256(domain_tag || block_hash).
    """
    # Stage 1: hash block contents
    h = hashlib.sha256()
    for a in range(bs):
        for b in range(bs):
            h.update(struct.pack("<q", C_block_flat[a * bs + b]))
    block_hash = h.digest()

    # Stage 2: domain-separated leaf
    h2 = hashlib.sha256()
    # DOM_TAG padded to 16 bytes on GPU (constant memory), but only DOM_LEN=13 bytes used
    h2.update(_DOM_TAG)
    h2.update(block_hash)
    leaf_hash = h2.digest()

    return leaf_hash


def verify_merkle_path(
    root: bytes, leaf_hash: bytes, path: list[tuple[bytes, bool]]
) -> bool:
    """Verify a Merkle authentication path.

    Args:
        root: Expected Merkle root (32 bytes).
        leaf_hash: Hash of the leaf node (32 bytes).
        path: Authentication path from extract_merkle_path.

    Returns:
        True if the path is valid.
    """
    current = leaf_hash
    for sibling, is_left in path:
        h = hashlib.sha256()
        if is_left:
            h.update(sibling)
            h.update(current)
        else:
            h.update(current)
            h.update(sibling)
        current = h.digest()

    return current == root


def block_index(bi: int, bj: int, blocks_per_row: int) -> int:
    """Convert (bi, bj) block coordinates to flat leaf index.

    Matches CUDA: block_idx = bi * num_blocks_per_row + bj.
    """
    return bi * blocks_per_row + bj
