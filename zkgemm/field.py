"""
Mersenne prime field arithmetic.

Field: F_p where p = 2^61 - 1 (a Mersenne prime).

All field elements are plain Python ints in [0, P).
The Mersenne structure allows fast modular reduction:
  x mod p = (x >> 61) + (x & p), with a conditional subtract.
"""

from __future__ import annotations

import secrets

# Mersenne prime: 2^61 - 1
P = (1 << 61) - 1  # 2305843009213693951


def mod_mersenne(x: int) -> int:
    """Fast reduction modulo p = 2^61 - 1.

    For non-negative x up to ~122 bits (product of two 61-bit numbers),
    we use: x mod p = (x >> 61) + (x & p), then conditionally subtract p.

    For negative or very large values, falls back to Python's % operator.
    """
    if x < 0:
        return x % P
    # Split into high and low parts
    lo = x & P
    hi = x >> 61
    r = lo + hi
    # One more reduction may be needed since lo + hi can be up to 2*P
    if r >= P:
        r -= P
    return r


def add(a: int, b: int) -> int:
    """(a + b) mod p."""
    r = a + b
    if r >= P:
        r -= P
    return r


def sub(a: int, b: int) -> int:
    """(a - b) mod p."""
    r = a - b
    if r < 0:
        r += P
    return r


def mul(a: int, b: int) -> int:
    """(a * b) mod p."""
    return mod_mersenne(a * b)


def inv(a: int) -> int:
    """Multiplicative inverse: a^{-1} mod p via Fermat's little theorem."""
    assert a % P != 0, "Cannot invert zero"
    return pow(a, P - 2, P)


def div(a: int, b: int) -> int:
    """(a / b) mod p = a * b^{-1} mod p."""
    return mul(a, inv(b))


def neg(a: int) -> int:
    """(-a) mod p."""
    if a == 0:
        return 0
    return P - a


def rand_element() -> int:
    """Uniformly random element in [0, P)."""
    return secrets.randbelow(P)


def from_int(x: int) -> int:
    """Convert arbitrary integer to field element in [0, P)."""
    return x % P
